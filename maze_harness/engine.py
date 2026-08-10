"""The game loop (PLAN.md section 2).

Observe -> ask the policy for a batched trajectory -> execute it one cell at a
time, halting at the first collision -> observe again. Success and collision are
both O(1) checks against the `Scene`; nothing here inspects pixels.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .policies import Policy
from .scene import Scene
from .state import State


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
        max_steps: int = 1200,
        max_calls: int = 60,
        renderer=None,
        log: Callable[[str], None] = print,
        step_delay: float = 0.0,
    ):
        self.scene = scene
        self.policy = policy
        self.max_steps = max_steps
        self.max_calls = max_calls
        self.renderer = renderer
        self.log = log
        self.step_delay = step_delay

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

            state = State.observe(self.scene, step=self.steps, last_outcome=outcome)
            if not self._draw(state, "thinking"):
                reason = "closed by user"
                break

            move = self.policy.act(state)
            self.calls += 1

            executed, blocked = self._execute(move)
            if self._aborted:
                reason = "closed by user"
                break

            outcome = self._describe(executed, len(move.actions), blocked)
            self.log(f"[step {self.steps:>4}] call {self.calls:>2} @{self.scene.player.position} "
                     f"{outcome} :: {move.reasoning[:110]}")
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
        self._draw(State.observe(self.scene, step=self.steps, last_outcome=reason),
                   "solved" if solved else "stopped")
        return result

    # ---------------------------------------------------------------- internals

    def _execute(self, move) -> tuple[int, bool]:
        """Walk the trajectory; stop at the first wall or when a budget runs out."""
        executed = 0
        for direction in move.actions:
            if not self.scene.step(direction):
                self.collisions += 1
                return executed, True
            self.steps += 1
            executed += 1
            state = State.observe(self.scene, step=self.steps, last_outcome="in flight")
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

    def _draw(self, state: State, phase: str) -> bool:
        if self.renderer is None:
            return True
        memory = getattr(self.policy, "memory", None)
        return self.renderer.draw(
            self.scene, state, memory,
            {"phase": phase, "steps": self.steps, "calls": self.calls,
             "collisions": self.collisions, "max_steps": self.max_steps,
             "policy": self.policy.name},
        )


def save_run(directory: str | Path, scene: Scene, result: EpisodeResult,
             extra: dict | None = None) -> Path:
    """Persist the scene and the full trajectory so a run can be replayed/audited."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene.save(directory / "scene.json")
    payload = asdict(result)
    payload["scene_meta"] = scene.meta
    if extra:
        payload.update(extra)
    (directory / "episode.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory
