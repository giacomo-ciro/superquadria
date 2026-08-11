"""The game loop (3D_PLAN.md section 7).

    observe -> integrate memory -> request a target-position batch
            -> validate/limit the batch
            -> orient and fly each segment incrementally
            -> stop on collision, key pickup, budget, or window close
            -> observe again

Success and collision are code-level checks against the `Scene`; nothing here
inspects pixels. Budgets are separate counters rather than one `steps` number,
which stops meaning anything once movement is continuous.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harness_2d.pipeline_log import PipelineLogger

from .geometry import Vec3
from .memory import SpatialMemory
from .moves import MAX_WAYPOINTS, Trajectory
from .policies import Policy, PolicyQuit
from .scene import Scene
from .state import State
from .superquadrics import Sensor


@dataclass
class EpisodeResult:
    solved: bool
    reason: str
    calls: int
    collisions: int
    distance: float
    observed: int
    wall_time_s: float
    start: tuple[float, float, float]
    start_forward: tuple[float, float, float]
    key: tuple[float, float, float]
    trajectory: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "SOLVED" if self.solved else "not solved"
        return (f"{verdict} — {self.reason} | calls={self.calls} collisions={self.collisions} "
                f"distance={self.distance:.0f} observed={self.observed} "
                f"time={self.wall_time_s:.1f}s")


@dataclass
class Budgets:
    """Every limit an episode enforces, each one its own counter."""

    max_calls: float = 40
    max_waypoints: int = MAX_WAYPOINTS
    max_segment: float = 15.0
    max_distance: float = 600.0
    max_collisions: float = 20
    move_increment: float = 0.5


class Episode:
    """One run of one policy against one scene."""

    def __init__(self, scene: Scene, policy: Policy, *, sensor: Sensor, budgets: Budgets,
                 renderer=None, log: PipelineLogger | None = None, step_delay: float = 0.0):
        self.scene = scene
        self.policy = policy
        self.sensor = sensor
        self.budgets = budgets
        self.renderer = renderer
        self.log = log if log is not None else PipelineLogger("episode3d")
        self.step_delay = step_delay

        self.calls = 0
        self.collisions = 0
        self.distance = 0.0
        self.start = scene.player.position
        self.start_forward = scene.player.forward
        self._aborted = False

    # -------------------------------------------------------------------- run

    def run(self) -> EpisodeResult:
        started = time.monotonic()
        budgets = self.budgets
        outcome = "start"
        reason = "agent-call budget exhausted"
        trajectory_log: list[dict] = []
        self.log.start(0, f"episode start: spawn={self.start} key={self.scene.key.position} "
                          f"max_calls={budgets.max_calls} max_distance={budgets.max_distance}")

        while True:
            if self.scene.key_reached():
                reason = "reached the key"
                break
            if self.calls >= budgets.max_calls:
                reason = "agent-call budget exhausted"
                break
            if self.distance >= budgets.max_distance:
                reason = "distance budget exhausted"
                break
            if self.collisions >= budgets.max_collisions:
                reason = "collision budget exhausted"
                break

            state = self._observe(outcome)
            self.policy.observe(state)
            if not self._draw(state, "thinking"):
                reason = "closed by user"
                break

            self.log.start(self.calls, f"call {self.calls + 1}: requesting a batch "
                                       f"@{self.scene.player.position}")
            try:
                trajectory = self._act(state)
            except PolicyQuit:
                reason = "closed by user"
                break
            if trajectory is None:
                reason = "closed by user"
                break
            self.calls += 1

            record = self._execute(trajectory)
            if self._aborted:
                reason = "closed by user"
                break

            outcome = self._describe(record, len(trajectory))
            record["call"] = self.calls
            record["reasoning"] = trajectory.reasoning
            trajectory_log.append(record)
            self.log.end(self.calls, f"call {self.calls}: {outcome}")

            if self.scene.key_reached():
                reason = "reached the key"
                break

        solved = self.scene.key_reached()
        memory = self.policy.memory
        result = EpisodeResult(
            solved=solved,
            reason=reason,
            calls=self.calls,
            collisions=self.collisions,
            distance=round(self.distance, 2),
            observed=len(memory.primitives) if memory else 0,
            wall_time_s=time.monotonic() - started,
            start=self.start.rounded(3),
            start_forward=self.start_forward.rounded(4),
            key=self.scene.key.position.rounded(3),
            trajectory=trajectory_log,
        )
        self.log.end(self.calls, f"episode end: {result.summary()}")
        self._draw(self._observe(reason), "solved" if solved else "stopped")
        return result

    # --------------------------------------------------------------- internals

    def _observe(self, outcome: str) -> State:
        return State.observe(self.scene, self.sensor, calls=self.calls, distance=self.distance,
                             collisions=self.collisions, last_outcome=outcome)

    def _act(self, state: State) -> Trajectory | None:
        """Ask the policy for a batch. None means the user closed the window.

        `renderer.draw()` is the only thing that steps Ursina, so against a
        window the wait — for an agent subprocess or for a keypress — has to
        happen in a loop that keeps drawing, or the window stops answering the
        OS and gets reported as hung.
        """
        if self.renderer is None:
            return self.policy.act(state)
        if self.policy.main_thread_only:
            return self._act_polled(state)
        return self._act_threaded(state)

    def _act_polled(self, state: State) -> Trajectory | None:
        """An interactive policy has nothing to report until the user acts, so
        poll it between frames — the input it reads comes from `draw()` itself."""
        while True:
            trajectory = self.policy.act(state)
            if trajectory is not None:
                return trajectory
            if not self._draw(state, "flying"):
                return None

    def _act_threaded(self, state: State) -> Trajectory | None:
        """Run a policy that blocks on I/O off the main thread, drawing meanwhile.

        Safe because `observe()` already folded this pose into memory on this
        thread: `act` only reads what the renderer is drawing.
        """
        box: dict = {}

        def work() -> None:
            try:
                box["trajectory"] = self.policy.act(state)
            except BaseException as exc:  # re-raised on the main thread below
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            while thread.is_alive():
                if not self._draw(state, "thinking", time.monotonic() - started):
                    return None
        finally:
            # Both ways out with the call still running — the user closed the
            # window, or Ctrl+C interrupted this thread — would leave a CLI that
            # outlives the harness burning tokens on an answer nobody reads.
            if thread.is_alive():
                self.policy.cancel()
                thread.join(timeout=2)
        thread.join()
        if "error" in box:
            raise box["error"]
        return box["trajectory"]

    def _execute(self, trajectory: Trajectory) -> dict:
        """Fly the batch, halting at the first blocked increment.

        Each segment is clamped to the distance cap, the camera is pointed along
        it before departure (agent batches only), and the key is checked after
        every increment rather than only at the endpoints.
        """
        budgets = self.budgets
        player = self.scene.player
        requested = [p.rounded(2) for p in trajectory.positions[: budgets.max_waypoints]]
        accepted: list[tuple[float, float, float]] = []
        poses: list[list[float]] = []
        blocked_by: int | None = None
        before = len(self.policy.memory.primitives) if self.policy.memory else 0

        for target in trajectory.positions[: budgets.max_waypoints]:
            origin = player.position
            delta = target - origin
            length = delta.length()
            if length < 1e-6:
                continue  # a repeated position is a no-op and cannot turn the camera
            if length > budgets.max_segment:
                length = budgets.max_segment
                target = origin + delta.normalized() * length
            accepted.append(target.rounded(2))

            if trajectory.steer:
                player.forward = delta.normalized()
                state = self._observe("turning")
                self.policy.observe(state)
                if not self._draw(state, "turning"):
                    self._aborted = True
                    return self._record(requested, accepted, poses, blocked_by, before)

            steps = max(1, math.ceil(length / budgets.move_increment))
            stop = False
            for i in range(1, steps + 1):
                point = origin + (target - origin) * (i / steps)
                blocker = self.scene.blocker(point)
                if blocker is not None:
                    self.collisions += 1
                    blocked_by = blocker
                    stop = True
                    break
                self.distance += point.distance_to(player.position)
                player.position = point
                poses.append([*point.rounded(3), *player.forward.rounded(4)])

                state = self._observe("in flight")
                self.policy.observe(state)
                if not self._draw(state, "flying"):
                    self._aborted = True
                    return self._record(requested, accepted, poses, blocked_by, before)
                if self.step_delay:
                    time.sleep(self.step_delay)
                if self.scene.key_reached() or self.distance >= budgets.max_distance:
                    stop = True
                    break
            if stop:
                break

        return self._record(requested, accepted, poses, blocked_by, before)

    def _record(self, requested, accepted, poses, blocked_by, observed_before) -> dict:
        after = len(self.policy.memory.primitives) if self.policy.memory else 0
        return {"requested": [list(p) for p in requested], "accepted": [list(p) for p in accepted],
                "poses": poses, "blocked_by": blocked_by, "observed": after - observed_before}

    def _describe(self, record: dict, planned: int) -> str:
        done, blocked_by = len(record["accepted"]), record["blocked_by"]
        gained = f" You detected {record['observed']} new shape(s)." if record["observed"] else ""
        if blocked_by is not None:
            return (f"you flew {done - 1} of {planned} targets, then hit "
                    f"{self.scene.describe_blocker(blocked_by)} on the way to the next one — "
                    f"the rest of the batch was discarded.{gained}")
        if self.scene.key_reached():
            return f"you reached the key.{gained}"
        if done < planned:
            return f"you flew {done} of {planned} targets, then a budget ran out.{gained}"
        return f"you flew all {done} targets without touching anything.{gained}"

    def _draw(self, state: State, phase: str, waiting_s: float = 0.0) -> bool:
        if self.renderer is None:
            return True
        return self.renderer.draw(self.scene, state, self.policy.memory, {
            "phase": phase, "calls": self.calls, "max_calls": self.budgets.max_calls,
            "collisions": self.collisions, "distance": self.distance,
            "max_distance": self.budgets.max_distance, "policy": self.policy.name,
            "agent_model": getattr(getattr(self.policy, "agent", None), "model", None),
            "agent_effort": getattr(getattr(self.policy, "agent", None), "effort", None),
            "waiting_s": waiting_s,
        })


class Replay:
    """Re-watch a recorded episode: SPACE plays/pauses (and restarts from the
    beginning once playback has run to the end), closing the window quits.

    Consumes only the saved world and episode JSON — it never calls an agent.
    """

    def __init__(self, scene: Scene, renderer, result: EpisodeResult, policy_name: str, *,
                 sensor: Sensor, agent_model: str | None = None, agent_effort: str | None = None,
                 step_delay: float = 0.0):
        self.scene = scene
        self.renderer = renderer
        self.result = result
        self.policy_name = policy_name
        self.sensor = sensor
        self.agent_model = agent_model
        self.agent_effort = agent_effort
        #: A fast original run may have had no delay; replay still has to be
        #: watchable, so floor it.
        self.step_delay = step_delay if step_delay > 0 else 0.02
        self.poses = [(Vec3(*pose[:3]), Vec3(*pose[3:])) for call in result.trajectory
                      for pose in call["poses"]]
        self.start = (Vec3(*result.start), Vec3(*result.start_forward))
        #: Rebuilt as playback advances rather than up front: one sensor pass per
        #: recorded increment is real work, and doing it lazily is also what makes
        #: the remembered count climb the way it did live.
        self.memory = SpatialMemory()
        self._seek(0)

    def run(self) -> None:
        playing = False
        index = 0
        phase = "idle"

        while True:
            if not self._draw(index, phase):
                return
            if self.poses and "space" in self.renderer.pending_keys:
                if index >= len(self.poses):  # at the end: space restarts
                    self._seek(0)
                    index = 0
                playing = not playing

            if playing and index < len(self.poses):
                self._seek(index + 1)
                index += 1
                done = index >= len(self.poses)
                phase = "done" if done else "replaying"
                if done:
                    playing = False
                elif self.step_delay:
                    time.sleep(self.step_delay)
            else:
                playing = False
                if self.poses and 0 < index < len(self.poses):
                    phase = "paused"

    def _seek(self, index: int) -> None:
        position, forward = self.poses[index - 1] if index else self.start
        self.scene.player.position = position
        self.scene.player.forward = forward
        if index == 0:
            self.memory = SpatialMemory()
        self.memory.integrate(self._state())

    def _state(self) -> State:
        return State.observe(self.scene, self.sensor, calls=self.result.calls,
                             distance=self.result.distance, collisions=self.result.collisions,
                             last_outcome=self.result.reason)

    def _draw(self, index: int, phase: str) -> bool:
        return self.renderer.draw(self.scene, self._state(), self.memory, {
            "phase": phase, "calls": self.result.calls, "max_calls": self.result.calls,
            "collisions": self.result.collisions,
            "distance": self.result.distance * (index / max(1, len(self.poses))),
            "max_distance": self.result.distance, "policy": self.policy_name,
            "agent_model": self.agent_model, "agent_effort": self.agent_effort,
            "replay": f"{index}/{len(self.poses)}",
        })


def save_episode(path: str | Path, world: str | Path, result: EpisodeResult,
                 extra: dict | None = None) -> Path:
    """Persist the full trajectory so an episode can be replayed and audited.

    Nothing about the world is copied — it already lives under `worlds/`,
    primitives and metadata alike, so the episode records only the path it ran on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"world": str(world), **asdict(result)}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_episode(path: str | Path, *, collision_resolution: int = 8):
    """Inverse of `save_episode`: reload an episode plus the world it ran on."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = Scene.load(payload["world"], collision_resolution=collision_resolution)
    result = EpisodeResult(**{k: v for k, v in payload.items()
                              if k in EpisodeResult.__dataclass_fields__})
    return (scene, result, payload.get("policy", "unknown"),
            payload.get("agent_model"), payload.get("agent_effort"))
