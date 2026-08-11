"""The game loop (PLAN.md section 2).

Observe -> ask the policy for a batched trajectory -> execute it one cell at a
time, halting at the first collision -> observe again. Success and collision are
both O(1) checks against the `Scene`; nothing here inspects pixels.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .memory import FogMemory
from .moves import Direction, Move
from .pipeline_log import PipelineLogger
from .policies import Policy, PolicyQuit
from .scene import VIEW_SIZE, Scene
from .state import State
from .vector import Vector2D


@dataclass
class EpisodeResult:
    solved: bool
    reason: str
    steps: int
    calls: int
    collisions: int
    explored: float
    wall_time_s: float
    start: tuple[int, int]
    key: tuple[int, int]
    trajectory: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "SOLVED" if self.solved else "not solved"
        return (f"{verdict} — {self.reason} | steps={self.steps} calls={self.calls} "
                f"collisions={self.collisions} explored={self.explored:.0%} "
                f"time={self.wall_time_s:.1f}s")


class Episode:
    """One run of one policy against one scene."""

    def __init__(
        self,
        scene: Scene,
        policy: Policy,
        *,
        max_steps: float = 1200,
        max_calls: float = 60,
        renderer=None,
        log: PipelineLogger | None = None,
        step_delay: float = 0.0,
        view_size: int = VIEW_SIZE,
    ):
        self.scene = scene
        self.policy = policy
        self.max_steps = max_steps
        self.max_calls = max_calls
        self.renderer = renderer
        self.log = log if log is not None else PipelineLogger("episode")
        self.step_delay = step_delay
        self.view_size = view_size

        self.steps = 0
        self.calls = 0
        self.collisions = 0
        self.start = scene.player.position
        self._aborted = False

    # -------------------------------------------------------------------- run

    def run(self) -> EpisodeResult:
        started = time.monotonic()
        outcome = "start"
        reason = "step budget exhausted"
        trajectory: list[dict] = []
        self.log.start(self.steps, f"episode start: spawn={self.start} key={self.scene.key.position} "
                                    f"max_steps={self.max_steps} max_calls={self.max_calls}")

        while True:
            if self.scene.is_solved():
                reason = "reached the key"
                break
            if self.steps >= self.max_steps:
                reason = "step budget exhausted"
                break
            if self.calls >= self.max_calls:
                reason = "agent-call budget exhausted"
                break

            state = State.observe(self.scene, step=self.steps, last_outcome=outcome, size=self.view_size)
            self.policy.observe(state)
            if not self._draw(state, "thinking"):
                reason = "closed by user"
                break

            self.log.start(self.steps, f"call {self.calls + 1}: requesting move @{self.scene.player.position}")
            try:
                move = self._act(state)
            except PolicyQuit:
                reason = "closed by user"
                break
            if move is None:
                reason = "closed by user"
                break
            self.calls += 1

            executed, blocked = self._execute(move)
            if self._aborted:
                reason = "closed by user"
                break

            outcome = self._describe(executed, len(move.actions), blocked)
            self.log.end(self.steps, f"call {self.calls}: {outcome} :: {move.reasoning[:110]}")
            trajectory.append({
                "call": self.calls,
                "step": self.steps,
                "position": self.scene.player.position.as_tuple(),
                "planned": [d.name.lower() for d in move.actions],
                "executed": executed,
                "blocked": blocked,
                "reasoning": move.reasoning,
            })

            if self.scene.is_solved():
                reason = "reached the key"
                break

        solved = self.scene.is_solved()
        memory = getattr(self.policy, "memory", None)
        result = EpisodeResult(
            solved=solved,
            reason=reason,
            steps=self.steps,
            calls=self.calls,
            collisions=self.collisions,
            explored=memory.explored_fraction() if memory else 0.0,
            wall_time_s=time.monotonic() - started,
            start=self.start.as_tuple(),
            key=self.scene.key.position.as_tuple(),
            trajectory=trajectory,
        )
        self.log.end(self.steps, f"episode end: {result.summary()}")
        self._draw(State.observe(self.scene, step=self.steps, last_outcome=reason, size=self.view_size),
                   "solved" if solved else "stopped")
        return result

    # ---------------------------------------------------------------- internals

    def _act(self, state: State) -> Move | None:
        """Ask the policy for a move. None means the user closed the window.

        `renderer.draw()` is the only thing that drains pygame's event queue, so
        against a window the wait — for an agent subprocess or for a keypress —
        has to happen in a loop that keeps drawing. Otherwise the window stops
        answering the OS and gets reported as hung.
        """
        if self.renderer is None:
            return self.policy.act(state)
        if self.policy.main_thread_only:
            return self._act_polled(state)
        return self._act_threaded(state)

    def _act_polled(self, state: State) -> Move | None:
        """An interactive policy has nothing to report until the user acts, so
        poll it between frames — the key it reads comes from `draw()` itself."""
        while True:
            move = self.policy.act(state)
            if move is not None:
                return move
            if not self._draw(state, "thinking"):
                return None

    def _act_threaded(self, state: State) -> Move | None:
        """Run a policy that blocks on I/O off the main thread, drawing meanwhile.

        Safe because `observe()` already folded this step into the fog map on
        this thread: `act` only reads the map the renderer is drawing.
        """
        box: dict = {}

        def work() -> None:
            try:
                box["move"] = self.policy.act(state)
            except BaseException as exc:  # re-raised on the main thread below
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            while thread.is_alive():
                # `_draw` ends in `clock.tick(fps)`, so this paces itself.
                if not self._draw(state, "thinking", time.monotonic() - started):
                    return None
        finally:
            # Both ways out with the call still running — the user closed the
            # window, or Ctrl+C interrupted this thread while the worker held the
            # subprocess — leave a `claude` that would outlive the harness in its
            # own process group. The join is bounded so quitting stays snappy.
            if thread.is_alive():
                self.policy.cancel()
                thread.join(timeout=2)
        thread.join()
        if "error" in box:
            raise box["error"]
        return box["move"]

    def _execute(self, move) -> tuple[int, bool]:
        """Walk the trajectory; stop at the first wall or when a budget runs out."""
        executed = 0
        for direction in move.actions:
            if not self.scene.step(direction):
                self.collisions += 1
                return executed, True
            self.steps += 1
            executed += 1
            state = State.observe(self.scene, step=self.steps, last_outcome="in flight", size=self.view_size)
            # The player observes from every cell it walks through, not just from
            # where the trajectory happens to end. Folding the in-flight views in
            # here is what makes the fog map cover the whole path — otherwise a
            # long trajectory reveals only the 10x10 window around its last cell.
            self.policy.observe(state)
            if not self._draw(state, "moving"):
                self._aborted = True
                return executed, False
            if self.step_delay:
                time.sleep(self.step_delay)
            if self.scene.is_solved() or self.steps >= self.max_steps:
                break
        return executed, False

    def _describe(self, executed: int, planned: int, blocked: bool) -> str:
        if blocked:
            return (f"executed {executed}/{planned} moves, then the next move hit a wall — "
                    f"the remaining {planned - executed} were discarded")
        if executed < planned:
            reason = "you reached the key" if self.scene.is_solved() else "the step budget ran out"
            return f"executed {executed}/{planned} moves, then {reason}"
        return f"executed all {executed}/{planned} moves without collision"

    def _draw(self, state: State, phase: str, waiting_s: float = 0.0) -> bool:
        if self.renderer is None:
            return True
        memory = getattr(self.policy, "memory", None)
        agent = getattr(self.policy, "agent", None)
        return self.renderer.draw(
            self.scene, state, memory,
            {"phase": phase, "steps": self.steps, "calls": self.calls,
             "collisions": self.collisions, "max_steps": self.max_steps,
             "policy": self.policy.name, "agent_model": getattr(agent, "model", None),
             "agent_effort": getattr(agent, "effort", None), "waiting_s": waiting_s},
        )


class Replay:
    """Keeps a `PygameRenderer` window open after an episode ends and lets the
    user re-watch the recorded trajectory: SPACE toggles play/pause (and
    restarts from the beginning once playback has run to the end), and
    closing the window quits.
    """

    def __init__(
        self,
        scene: Scene,
        renderer,
        result: EpisodeResult,
        policy_name: str,
        *,
        agent_model: str | None = None,
        agent_effort: str | None = None,
        memory=None,
        step_delay: float = 0.0,
        view_size: int = VIEW_SIZE,
    ):
        self.scene = scene
        self.renderer = renderer
        self.result = result
        self.policy_name = policy_name
        self.agent_model = agent_model
        self.agent_effort = agent_effort
        # `step_delay` may be 0 for a fast original run; replay still needs to
        # be watchable, so floor it.
        self.step_delay = step_delay if step_delay > 0 else 0.06
        self.view_size = view_size
        self.start = Vector2D(*result.start)
        self.actions = self._flatten(result.trajectory)
        # Shown at rest — before the first R/SPACE press, and again once
        # playback runs to the end: the fully accumulated fog, exactly as the
        # episode left it. A live episode's replay reuses the policy's own fog
        # memory; a standalone replay loaded from disk has none persisted (only
        # the final `explored` fraction is saved, not the grid), so it's
        # rebuilt once by walking the recorded path.
        self.final_memory = memory if memory is not None else self._walked_memory()
        #: The memory actually handed to the renderer. While playing, this is
        #: a fresh memory rebuilt step by step, so fog mode reveals the map
        #: progressively — the way it looked live — instead of showing
        #: everything explored from the very first frame.
        self.memory = self.final_memory

    @staticmethod
    def _flatten(trajectory: list[dict]) -> list[Direction]:
        """Re-derive the single-step action sequence the player actually walked."""
        return [Direction.parse(name)
                for call in trajectory
                for name in call["planned"][: call["executed"]]]

    def _walked_memory(self) -> FogMemory:
        memory = FogMemory(height=self.scene.height, width=self.scene.width)
        self.scene.player.position = self.start
        self._integrate(memory, 0)
        for i, direction in enumerate(self.actions, start=1):
            self.scene.step(direction)
            self._integrate(memory, i)
        return memory

    def _integrate(self, memory: FogMemory, index: int) -> None:
        state = State.observe(self.scene, step=index, last_outcome=self.result.reason,
                               size=self.view_size)
        memory.integrate(state)

    def _restart(self) -> None:
        """Rewind to the start with a blank fog memory, so a fresh playthrough
        reveals the map progressively instead of starting fully explored."""
        self.scene.player.position = self.start
        self.memory = FogMemory(height=self.scene.height, width=self.scene.width)
        self._integrate(self.memory, 0)

    def run(self) -> None:
        pygame = self.renderer.pygame
        playing = False
        index = len(self.actions)  # matches the final frame already on screen
        phase = "done" if self.actions else "idle"

        while True:
            # `renderer.draw()` is the *only* place that drains pygame's event
            # queue (QUIT, and the god-view button click). Calling it every
            # tick — instead of only on state changes — is what lets those
            # keep working while replay is idle/paused, not just while playing.
            if not self._draw(index, phase):
                return
            keys = self.renderer.pending_keys

            if self.actions and pygame.K_SPACE in keys:
                if index >= len(self.actions):  # at the end: space restarts, like most players
                    self._restart()
                    index = 0
                playing = not playing

            if playing and index < len(self.actions):
                self.scene.step(self.actions[index])
                index += 1
                self._integrate(self.memory, index)
                done = index >= len(self.actions)
                phase = "done" if done else "replaying"
                if done:
                    playing = False
                    self.memory = self.final_memory  # snap back to the authoritative accumulated view
                elif self.step_delay:
                    time.sleep(self.step_delay)
            else:
                playing = False
                if self.actions and 0 < index < len(self.actions):
                    phase = "paused"

    def _draw(self, index: int, phase: str) -> bool:
        state = State.observe(self.scene, step=index, last_outcome=self.result.reason, size=self.view_size)
        return self.renderer.draw(
            self.scene, state, self.memory,
            {"phase": phase, "steps": index, "calls": self.result.calls,
             "collisions": self.result.collisions, "max_steps": len(self.actions),
             "policy": self.policy_name, "agent_model": self.agent_model,
             "agent_effort": self.agent_effort},
        )


def save_episode(path: str | Path, maze: str | Path, result: EpisodeResult,
                 extra: dict | None = None) -> Path:
    """Persist the full trajectory so an episode can be replayed/audited.

    Nothing about the maze is copied — it already lives under `mazes/`, terrain
    and metadata alike, so the episode only records the path it was played on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"maze": str(maze), **asdict(result)}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_episode(path: str | Path) -> tuple[Scene, EpisodeResult, str, str | None, str | None]:
    """Inverse of `save_episode`: reload an episode plus the maze it ran on."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = Scene.load(payload["maze"])
    result = EpisodeResult(**{k: v for k, v in payload.items() if k in EpisodeResult.__dataclass_fields__})
    return (scene, result, payload.get("policy", "unknown"),
            payload.get("agent_model"), payload.get("agent_effort"))
