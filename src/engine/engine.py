"""The game loop.

    observe -> integrate memory -> request a target-position batch
            -> validate/limit the batch
            -> orient and fly each segment incrementally, picking up objects in passing
            -> stop on collision, a door attempt, budget, or window close
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
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from itertools import accumulate
from pathlib import Path

from geometry import Vec3
from geometry.superquadrics import DOOR, LOCK, OBJECT, PORTAL, Sensor
from navigation import SpatialMemory, State, Trajectory
from navigation.moves import MAX_WAYPOINTS, MOVE, PICK, PLACE, Action
from navigation.policies import Policy, PolicyQuit
from world.scene import BOUNDS, Scene
from world.task import attempt_unlock, pick_at, place_at, task_for_door_or_lock, try_pickup

from .pipeline_log import PipelineLogger


def _blocker_tag(blocked_by: int) -> str:
    """A collision in the few characters a history row has room for."""
    return f"hit {'bounds' if blocked_by == BOUNDS else f'#{blocked_by}'}"


def move_row(record: dict) -> dict:
    """One call record as the renderer's move-history row."""
    blocked_by = record.get("blocked_by")
    blocked = blocked_by is not None
    return {
        "call": record["call"],
        "gen_s": record.get("gen_s", 0.0),
        "flown": len(record["accepted"]) - (1 if blocked else 0),
        "planned": len(record["requested"]),
        "result": record.get("result") or (_blocker_tag(blocked_by) if blocked else "acted"),
    }


@dataclass
class EpisodeResult:
    solved: bool
    reason: str
    calls: int
    collisions: int
    distance: float
    observed: int
    rooms_cleared: int
    failed_attempts: int
    wall_time_s: float
    start: tuple[float, float, float]
    start_forward: tuple[float, float, float]
    trajectory: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "SOLVED" if self.solved else "not solved"
        return (f"{verdict} — {self.reason} | calls={self.calls} collisions={self.collisions} "
                f"distance={self.distance:.0f} rooms_cleared={self.rooms_cleared} "
                f"failed_attempts={self.failed_attempts} observed={self.observed} "
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
        self.rooms_cleared = 0
        #: One task per room on the path — the episode is solved once every
        #: room's door has been opened, not just the first.
        self.rooms_to_clear = len(scene.meta.get("task", {})) or 1
        self.failed_attempts = 0
        self.start = scene.player.position
        self.start_forward = scene.player.forward
        self._aborted = False
        #: One row per completed agent call, drawn as the window's move history.
        self.history: list[dict] = []
        self.started = time.monotonic()
        #: Sensor stride bookkeeping, carried across calls so short manual steps
        #: accumulate towards a pass rather than each forcing one.
        self._since_sensed = 0.0
        self._last_state: State | None = None
        self._active_turn: dict | None = None

    # -------------------------------------------------------------------- run

    def run(self) -> EpisodeResult:
        self.started = time.monotonic()
        budgets = self.budgets
        outcome = "start"
        reason = "agent-call budget exhausted"
        trajectory_log: list[dict] = []
        self.log.start(0, f"episode start: spawn={self.start} "
                          f"room={self.scene.current_room_name(self.start)} "
                          f"max_calls={budgets.max_calls} max_distance={budgets.max_distance}")

        while True:
            if self.rooms_cleared >= self.rooms_to_clear:
                reason = "reached the exit"
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
            asked = time.monotonic()
            try:
                trajectory = self._act(state)
            except PolicyQuit:
                reason = "closed by user"
                break
            if trajectory is None:
                reason = "closed by user"
                break
            gen_s = time.monotonic() - asked
            self.calls += 1

            if trajectory.reasoning:
                self.log.info(self.calls, f"reasoning: {trajectory.reasoning}")
            for idx, act in enumerate(trajectory.actions, 1):
                self.log.info(self.calls, f"action {idx}: {act.type.upper()} @ {act.target.rounded(2)}")

            departed = time.monotonic()
            record = self._execute(trajectory, gen_s=gen_s)
            if self._aborted:
                reason = "closed by user"
                break

            outcome = self._describe(record, len(trajectory))
            record["call"] = self.calls
            record["reasoning"] = trajectory.reasoning
            record["gen_s"] = round(gen_s, 1)
            # Thinking and flying, split: together they account for the episode's
            # whole wall clock, which is what a replay needs to re-time it.
            record["fly_s"] = round(time.monotonic() - departed, 2)
            record["result"] = self._tag(record)
            trajectory_log.append(record)
            self.history.append(move_row(record))
            self.log.end(self.calls, f"call {self.calls}: {outcome}")

            if self.rooms_cleared >= self.rooms_to_clear:
                reason = "reached the exit"
                break

        solved = self.rooms_cleared >= self.rooms_to_clear
        memory = self.policy.memory
        result = EpisodeResult(
            solved=solved,
            reason=reason,
            calls=self.calls,
            collisions=self.collisions,
            distance=round(self.distance, 2),
            observed=len(memory.primitives) if memory else 0,
            rooms_cleared=self.rooms_cleared,
            failed_attempts=self.failed_attempts,
            wall_time_s=time.monotonic() - self.started,
            start=self.start.rounded(3),
            start_forward=self.start_forward.rounded(4),
            trajectory=trajectory_log,
        )
        self.log.end(self.calls, f"episode end: {result.summary()}")
        self._draw(self._observe(reason), "solved" if solved else "stopped")
        return result

    # --------------------------------------------------------------- internals

    def _observe(self, outcome: str) -> State:
        return State.observe(self.scene, self.sensor, calls=self.calls, distance=self.distance,
                             collisions=self.collisions, rooms_cleared=self.rooms_cleared,
                             failed_attempts=self.failed_attempts, last_outcome=outcome)

    def _sense(self, outcome: str, flown: float) -> State:
        """Fold this pose into memory, but only every `sensor.stride` units.

        A sensor pass dominates the cost of a movement increment while the sensor
        itself reaches tens of units, so running one every increment re-detects
        the same primitives from half a unit further along. Between passes the
        previous observation stands — the pose the agent actually plans from is
        always sensed in full, at the top of `run`.

        This only ever *removes* observations, so memory still holds nothing but
        what a sensor query returned.
        """
        self._since_sensed += flown
        if self._last_state is None or self._since_sensed >= self.sensor.stride:
            self._since_sensed = 0.0
            self._last_state = self._observe(outcome)
            self.policy.observe(self._last_state)
        return self._last_state

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

    def _execute(self, trajectory: Trajectory, gen_s: float = 0.0) -> dict:
        """Execute a batch of actions (move, pick, place).

        Each `move` segment is clamped to max_segment and stepped incrementally.
        Hitting a solid wall/obstacle halts movement and discards subsequent actions.
        `pick` and `place` require being within reach of the target.
        """
        budgets = self.budgets
        player = self.scene.player
        requested = [a.to_dict() for a in trajectory.actions[: budgets.max_waypoints]]
        accepted: list[dict] = []
        poses: list[list[float]] = []
        events: list[dict] = []
        blocked_by: int | None = None
        door_result: str | None = None
        picked_up: list[str] = []
        placed: list[str] = []
        action_failed: str | None = None
        before = len(self.policy.memory.primitives) if self.policy.memory else 0

        active_actions: list[dict] = []
        for idx, act in enumerate(trajectory.actions[: budgets.max_waypoints], 1):
            dist = round((act.target - player.position).length(), 1) if act.type == MOVE and idx == 1 else None
            active_actions.append({
                "idx": idx,
                "type": act.type.upper(),
                "target": [round(c, 1) for c in act.target.as_tuple()],
                "status": "pending",
                "info": "",
                "dist": dist,
            })
        self._active_turn = {
            "call": self.calls,
            "reasoning": trajectory.reasoning,
            "actions": active_actions,
            "gen_s": round(gen_s, 1),
        }

        for action_idx, action in enumerate(trajectory.actions[: budgets.max_waypoints]):
            active_actions[action_idx]["status"] = "active"
            if action.type == MOVE:
                target = action.target
                origin = player.position
                delta = target - origin
                length = delta.length()
                if length < 1e-6:
                    accepted.append(action.to_dict())
                    active_actions[action_idx]["status"] = "done"
                    continue
                if length > budgets.max_segment:
                    length = budgets.max_segment
                    target = origin + delta.normalized() * length
                accepted.append(Action(type=MOVE, target=target.rounded(2)).to_dict())

                if trajectory.steer:
                    player.forward = delta.normalized()
                    state = self._observe("turning")
                    self.policy.observe(state)
                    if not self._draw(state, "turning"):
                        self._aborted = True
                        return self._record(requested, accepted, poses, blocked_by, before,
                                            door_result=door_result, picked_up=picked_up,
                                            placed=placed, action_failed=action_failed,
                                            events=events)

                steps = max(1, math.ceil(length / budgets.move_increment))
                stop = False
                for i in range(1, steps + 1):
                    point = origin + (target - origin) * (i / steps)
                    blocker = self.scene.blocker(point)
                    if blocker is not None:
                        self.collisions += 1
                        blocked_by = blocker
                        stop = True
                        active_actions[action_idx]["status"] = "failed"
                        active_actions[action_idx]["info"] = _blocker_tag(blocker)
                        break
                    flown = point.distance_to(player.position)
                    self.distance += flown
                    player.position = point
                    poses.append([*point.rounded(3), *player.forward.rounded(4)])

                    state = self._sense("in flight", flown)
                    if not self._draw(state, "flying"):
                        self._aborted = True
                        return self._record(requested, accepted, poses, blocked_by, before,
                                            door_result=door_result, picked_up=picked_up,
                                            placed=placed, action_failed=action_failed,
                                            events=events)
                    if self.step_delay:
                        time.sleep(self.step_delay)
                    if self.distance >= budgets.max_distance:
                        stop = True
                        break
                if not stop:
                    active_actions[action_idx]["status"] = "done"
                else:
                    break

            elif action.type == PICK:
                accepted.append(action.to_dict())
                success, desc = pick_at(self.scene, action.target)
                if success:
                    name = desc.replace("picked up ", "")
                    picked_up.append(name)
                    events.append({"pose_offset": len(poses), "type": "pick", "name": name,
                                   "target": action.target.as_tuple()})
                    active_actions[action_idx]["status"] = "done"
                    active_actions[action_idx]["info"] = name[:10]
                    state = self._observe("picked up object")
                    self.policy.observe(state)
                    if not self._draw(state, "picking"):
                        self._aborted = True
                        return self._record(requested, accepted, poses, blocked_by, before,
                                            door_result=door_result, picked_up=picked_up,
                                            placed=placed, action_failed=action_failed,
                                            events=events)
                else:
                    action_failed = f"pick failed: {desc}"
                    active_actions[action_idx]["status"] = "failed"
                    active_actions[action_idx]["info"] = "fail reach"
                    break

            elif action.type == PLACE:
                accepted.append(action.to_dict())
                tol = self.scene.meta.get("match_tolerance", {})
                success, desc, d_res = place_at(self.scene, action.target,
                                                tol_scale=tol.get("scale", 0.05),
                                                tol_exp=tol.get("exponents", 0.05))
                if success:
                    if d_res == "opened":
                        self.rooms_cleared += 1
                        door_result = "opened"
                        active_actions[action_idx]["info"] = "door opened"
                    elif d_res == "locked":
                        self.failed_attempts += 1
                        door_result = "locked"
                        active_actions[action_idx]["info"] = "mismatch"
                    else:
                        placed.append(desc)
                        active_actions[action_idx]["info"] = "placed"
                    active_actions[action_idx]["status"] = "done"
                    events.append({"pose_offset": len(poses), "type": "place", "door_result": d_res,
                                   "desc": desc, "target": action.target.as_tuple()})
                    state = self._observe("placed object")
                    self.policy.observe(state)
                    if not self._draw(state, "placing"):
                        self._aborted = True
                        return self._record(requested, accepted, poses, blocked_by, before,
                                            door_result=door_result, picked_up=picked_up,
                                            placed=placed, action_failed=action_failed,
                                            events=events)
                    if door_result == "opened":
                        break
                else:
                    action_failed = f"place failed: {desc}"
                    active_actions[action_idx]["status"] = "failed"
                    active_actions[action_idx]["info"] = "fail reach"
                    break

        return self._record(requested, accepted, poses, blocked_by, before,
                            door_result=door_result, picked_up=picked_up,
                            placed=placed, action_failed=action_failed,
                            events=events)

    def _record(self, requested, accepted, poses, blocked_by, observed_before, *,
               door_result: str | None = None, picked_up=(), placed=(),
               action_failed: str | None = None, events=()) -> dict:
        after = len(self.policy.memory.primitives) if self.policy.memory else 0
        return {"requested": requested, "accepted": accepted,
                "poses": poses, "blocked_by": blocked_by, "observed": after - observed_before,
                "door_result": door_result, "picked_up": list(picked_up),
                "placed": list(placed), "action_failed": action_failed,
                "events": list(events)}

    def _tag(self, record: dict) -> str:
        """The same call, in the few characters a history row has room for."""
        gained = f" +{record['observed']}" if record["observed"] else ""
        if record.get("door_result") == "opened":
            return f"door open!{gained}"
        if record.get("door_result") == "locked":
            return f"wrong obj{gained}"
        if record.get("action_failed"):
            return f"action fail{gained}"
        if record.get("picked_up"):
            return f"picked {record['picked_up'][0][:8]}{gained}"
        if record.get("placed"):
            return f"placed obj{gained}"
        blocked_by = record.get("blocked_by")
        if blocked_by is not None:
            return f"{_blocker_tag(blocked_by)}{gained}"
        return f"acted{gained}"

    def _describe(self, record: dict, planned: int) -> str:
        done = len(record["accepted"])
        blocked_by = record.get("blocked_by")
        gained = f" You detected {record['observed']} new shape(s)." if record.get("observed") else ""
        picked = f" You picked up {', '.join(record['picked_up'])}." if record.get("picked_up") else ""
        placed = f" {'; '.join(record['placed'])}." if record.get("placed") else ""
        failed = f" (Stopped because {record['action_failed']})" if record.get("action_failed") else ""

        if record.get("door_result") == "opened":
            return f"you executed {done} of {planned} actions, and the door opened!{gained}"
        if record.get("door_result") == "locked":
            return (f"you executed {done} of {planned} actions, and tried the locked door — it did "
                    f"not match and the object was destroyed.{gained}")
        if blocked_by is not None:
            return (f"you executed {done - 1} of {planned} actions, then hit "
                    f"{self.scene.describe_blocker(blocked_by)} — the rest of the batch was discarded.{gained}{picked}{placed}")
        if record.get("action_failed"):
            return f"you executed {done} of {planned} actions.{failed}{gained}{picked}{placed}"
        if done < planned:
            return f"you executed {done} of {planned} actions, then a budget ran out.{gained}{picked}{placed}"
        return f"you executed all {done} actions successfully.{gained}{picked}{placed}"

    def _draw(self, state: State, phase: str, waiting_s: float = 0.0) -> bool:
        if self.renderer is None:
            return True
        return self.renderer.draw(self.scene, state, self.policy.memory, {
            "phase": phase, "calls": self.calls, "max_calls": self.budgets.max_calls,
            "collisions": self.collisions, "distance": self.distance,
            "max_collisions": self.budgets.max_collisions,
            "rooms_cleared": self.rooms_cleared, "rooms_total": self.rooms_to_clear,
            "failed_attempts": self.failed_attempts,
            "max_distance": self.budgets.max_distance, "policy": self.policy.name,
            "agent_model": getattr(getattr(self.policy, "agent", None), "model", None),
            "agent_effort": getattr(getattr(self.policy, "agent", None), "effort", None),
            "waiting_s": waiting_s,
            "history": self.history,
            "active_turn": self._active_turn,
            "match_tolerance": self.scene.meta.get("match_tolerance", {}),
            "elapsed_s": time.monotonic() - self.started,
        })


def _extract_events(trajectory: list[dict]) -> list[tuple[int, dict]]:
    """Flatten all events (pick, place, unlock) with their global pose index."""
    all_events: list[tuple[int, dict]] = []
    global_offset = 0
    for c in trajectory:
        if "events" in c and c["events"]:
            for e in c["events"]:
                offset = e.get("pose_offset", e.get("pose_index", 0))
                all_events.append((global_offset + offset, e))
        else:
            actions = c.get("accepted", [])
            poses = c.get("poses", [])
            picked_up = list(c.get("picked_up", []))
            placed = list(c.get("placed", []))
            door_result = c.get("door_result")
            moves_left = sum(1 for a in actions if a.get("type") == MOVE)
            local_p = 0
            pick_idx = 0
            placed_idx = 0
            for a in actions:
                atype = a.get("type")
                target = a.get("target")
                if atype == MOVE:
                    moves_left -= 1
                    if moves_left == 0:
                        local_p = len(poses)
                    else:
                        best_idx = local_p
                        best_d = float("inf")
                        for i in range(local_p, len(poses)):
                            d = math.sqrt(sum((x - y) ** 2 for x, y in zip(poses[i][:3], target)))
                            if d < best_d:
                                best_d = d
                                best_idx = i
                            if d < 1e-2:
                                break
                        local_p = best_idx + 1
                elif atype == PICK:
                    if pick_idx < len(picked_up):
                        name = picked_up[pick_idx]
                        pick_idx += 1
                        all_events.append((global_offset + local_p, {"type": "pick", "name": name, "target": target}))
                elif atype == PLACE:
                    if door_result is not None:
                        all_events.append((global_offset + local_p, {"type": "place", "door_result": door_result, "target": target}))
                    elif placed_idx < len(placed):
                        desc = placed[placed_idx]
                        placed_idx += 1
                        all_events.append((global_offset + local_p, {"type": "place", "door_result": None, "desc": desc, "target": target}))
        global_offset += len(c.get("poses", []))
    return all_events


class Replay:
    """Re-watch a recorded episode: SPACE plays/pauses (and restarts from the
    beginning once playback has run to the end), R restarts from the beginning,
    closing the window quits.

    Consumes only the saved world and episode JSON — it never calls an agent.
    """

    def __init__(self, scene: Scene, renderer, result: EpisodeResult, policy_name: str, *,
                 sensor: Sensor, agent_model: str | None = None, agent_effort: str | None = None,
                 step_delay: float = 0.0, max_collisions: float | None = None):
        self._initial_scene_data = scene.to_dict()
        self.collision_resolution = scene.primitives.collision_resolution
        self.scene = scene
        self.renderer = renderer
        self.result = result
        self.policy_name = policy_name
        self.sensor = sensor
        self.agent_model = agent_model
        self.agent_effort = agent_effort
        self.step_delay = step_delay if step_delay > 0 else 0.02
        if max_collisions is None:
            self.max_collisions = math.inf if policy_name == "manual" else Budgets.max_collisions
        else:
            self.max_collisions = max_collisions
        self.poses = [(Vec3(*pose[:3]), Vec3(*pose[3:])) for call in result.trajectory
                      for pose in call["poses"]]
        self.history = [move_row(call) for call in result.trajectory]
        self.call_ends = list(accumulate(len(call["poses"]) for call in result.trajectory))
        self.call_collisions = [0]
        c_count = 0
        for call in result.trajectory:
            if call.get("blocked_by") is not None:
                c_count += 1
            self.call_collisions.append(c_count)
        self.events = _extract_events(result.trajectory)
        self.start = (Vec3(*result.start), Vec3(*result.start_forward))
        self.memory = SpatialMemory()
        self._current_index = 0
        self.rooms_cleared = 0
        self.rooms_total = len(self._initial_scene_data.get("meta", {}).get("task", {})) or 1
        self.failed_attempts = 0
        self._seek(0)

    def _reset_scene(self) -> None:
        fresh = Scene.from_dict(self._initial_scene_data, collision_resolution=self.collision_resolution)
        self.scene.primitives = fresh.primitives
        self.scene.player = fresh.player
        self.rooms_cleared = 0
        self.failed_attempts = 0

    def run(self) -> None:
        playing = False
        index = 0
        phase = "idle"

        while True:
            if not self._draw(index, phase):
                return
            if self.poses and "r" in self.renderer.pending_keys:
                self._seek(0)
                index = 0
                playing = False
                phase = "idle"
            if self.poses and "space" in self.renderer.pending_keys:
                if index >= len(self.poses):  # at the end: space restarts
                    self._seek(0)
                    index = 0
                playing = not playing

            if playing and index < len(self.poses):
                self._step_forward()
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
                elif index == 0:
                    phase = "idle"

    def _seek(self, target_index: int) -> None:
        self._reset_scene()
        if target_index <= 0 or not self.poses:
            self._current_index = 0
            self.memory = SpatialMemory()
            self.scene.player.position = self.start[0]
            self.scene.player.forward = self.start[1]
            self.memory.integrate(self._state())
            return

        self._current_index = min(target_index, len(self.poses))
        point, forward = self.poses[self._current_index - 1]
        self.scene.player.position = point
        self.scene.player.forward = forward

        for g_pose, ev in self.events:
            if g_pose <= self._current_index:
                self._apply_event(ev)

        self.memory.integrate(self._state())

    def _step_forward(self) -> None:
        """Advance exactly one recorded pose during playback.

        `_seek` rebuilds `Scene`/`SuperquadricHandler` from the saved JSON so it
        is correct from any starting point (restart, scrubbing) — but that also
        discards the handler's per-primitive mesh/collision caches, so paying
        that cost every frame during ordinary forward playback forces a full
        re-tessellation and re-raycast of the scene each frame, instead of the
        once-per-episode cost a live run pays. Stepping the existing scene
        forward in place keeps those caches warm.
        """
        prev_index = self._current_index
        self._current_index = min(self._current_index + 1, len(self.poses))
        point, forward = self.poses[self._current_index - 1]
        self.scene.player.position = point
        self.scene.player.forward = forward

        for g_pose, ev in self.events:
            if prev_index < g_pose <= self._current_index:
                self._apply_event(ev)

        self.memory.integrate(self._state())

    def _apply_event(self, ev: dict) -> None:
        """Fold one recorded pick/place event into the scene."""
        if ev["type"] == "pick":
            name = ev["name"]
            matching = [p for p in list(self.scene.primitives) if p.assembly == name]
            if matching:
                self.scene.player.carrying = matching
                for p in matching:
                    self.scene.primitives.remove(p.id)
        elif ev["type"] == "place":
            if ev.get("door_result") == "opened":
                self.rooms_cleared += 1
                self.scene.player.carrying = None
                # Open the door and lock this placement targeted
                target = Vec3(*ev["target"])
                candidates = [p for p in self.scene.primitives if p.kind in (DOOR, LOCK)]
                if candidates:
                    nearest = min(candidates, key=lambda p: p.position.distance_to(target))
                    task = task_for_door_or_lock(self.scene, nearest.id)
                    if task is not None:
                        door_id = task.get("door_id")
                        if door_id is not None and door_id in self.scene.primitives:
                            self.scene.primitives.get(door_id).kind = PORTAL
                        lock_ids = task.get("lock_ids", [task.get("lock_id")])
                        for lid in lock_ids:
                            if lid is not None and lid in self.scene.primitives:
                                self.scene.primitives.get(lid).kind = PORTAL
            elif ev.get("door_result") == "locked":
                self.failed_attempts += 1
                self.scene.player.carrying = None
            else:
                carried = self.scene.player.carrying
                if carried:
                    centroid = sum((p.position for p in carried), Vec3(0.0, 0.0, 0.0)) * (1.0 / len(carried))
                    target = Vec3(*ev["target"])
                    delta = target - centroid
                    for p in carried:
                        p.position = p.position + delta
                        self.scene.primitives.add(p)
                    self.scene.player.carrying = None

    def _state(self) -> State:
        call_idx = bisect_right(self.call_ends, self._current_index)
        cur_collisions = self.call_collisions[min(call_idx, len(self.call_collisions) - 1)]
        return State.observe(self.scene, self.sensor, calls=self.result.calls,
                             distance=self.result.distance, collisions=cur_collisions,
                             rooms_cleared=self.rooms_cleared,
                             failed_attempts=self.failed_attempts,
                             last_outcome=self.result.reason)

    def _draw(self, index: int, phase: str) -> bool:
        call_idx = bisect_right(self.call_ends, index)
        cur_collisions = self.call_collisions[min(call_idx, len(self.call_collisions) - 1)]
        active_turn = None
        if self.result.trajectory:
            target_call_idx = min(call_idx, len(self.result.trajectory) - 1)
            call = self.result.trajectory[target_call_idx]
            requested = call.get("requested", [])
            accepted = call.get("accepted", [])
            blocked_by = call.get("blocked_by")
            actions = []
            for j, a in enumerate(requested, 1):
                status = "done" if j <= len(accepted) else ("failed" if j == len(accepted) + 1 and blocked_by is not None else "pending")
                info_str = _blocker_tag(blocked_by) if status == "failed" and blocked_by is not None else ""
                actions.append({
                    "idx": j,
                    "type": a.get("type", "MOVE").upper(),
                    "target": [round(c, 1) for c in a.get("target", [0, 0, 0])],
                    "status": status,
                    "info": info_str,
                })
            active_turn = {
                "call": call.get("call", target_call_idx + 1),
                "reasoning": call.get("reasoning", ""),
                "actions": actions,
                "gen_s": call.get("gen_s", 0.0),
            }

        return self.renderer.draw(self.scene, self._state(), self.memory, {
            "phase": phase, "calls": self.result.calls, "max_calls": self.result.calls,
            "collisions": cur_collisions,
            "max_collisions": self.max_collisions,
            "rooms_cleared": self.rooms_cleared, "rooms_total": self.rooms_total,
            "failed_attempts": self.failed_attempts,
            "distance": self.result.distance * (index / max(1, len(self.poses))),
            "max_distance": self.result.distance, "policy": self.policy_name,
            "agent_model": self.agent_model, "agent_effort": self.agent_effort,
            "replay": f"{index}/{len(self.poses)}",
            "history": self.history[:bisect_right(self.call_ends, index)],
            "active_turn": active_turn,
            "match_tolerance": self.scene.meta.get("match_tolerance", {}),
            # The original run's clock, not this playback's: how long the episode
            # took is a property of the episode.
            "elapsed_s": self.result.wall_time_s,
        })


def save_episode(path: str | Path, world: str | Path, result: EpisodeResult,
                 extra: dict | None = None) -> Path:
    """Persist the full trajectory so an episode can be replayed and audited.

    Nothing about the world is copied — it already lives under `worlds_3d/`,
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
            payload.get("agent_model"), payload.get("agent_effort"),
            payload.get("max_collisions"))
