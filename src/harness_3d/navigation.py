"""The navigation agent: parameter observation in, target positions out.

Collates the prompt, calls the agent through `Agent.run` with
`TRAJECTORY_SCHEMA` pinned, and parses the structured output into a
`Trajectory` (3D_PLAN.md section 6/7).

There is no fallback policy behind it: a failed call costs the turn and is
counted, because every alternative would have to route using knowledge the agent
is not allowed to have.
"""

from __future__ import annotations

from harness_common.agents.base import Agent, AgentError
from harness_common.pipeline_log import PipelineLogger

from .memory import SpatialMemory
from .moves import MAX_WAYPOINTS, TRAJECTORY_SCHEMA, Trajectory
from .policies import Policy
from .state import State

SYSTEM_PROMPT = (
    "You are the navigation policy for an agent flying through a 3D world built "
    "entirely from superquadrics. You never see images — you see the parameters of "
    "the shapes your sensor has detected. You answer only by emitting target "
    "positions through the StructuredOutput tool. Be decisive: commit to a heading "
    "and cover ground."
)


class WaypointNavigator(Policy):
    """Navigation policy backed by an `Agent` (whichever CLI the config selected)."""

    name = "agent"

    def __init__(self, agent: Agent, *, sensor_range: float, max_waypoints: int = MAX_WAYPOINTS,
                 max_segment: float = 15.0, call_budget: int | None = None,
                 distance_budget: float | None = None, collision_budget: int | None = None,
                 log: PipelineLogger | None = None):
        self.agent = agent
        # Saved into every episode as `policy`, so it has to say which CLI
        # actually played — a codex run recorded as "claude" is a corrupt
        # comparison.
        self.name = getattr(agent, "binary", type(self).name)
        self.sensor_range = sensor_range
        self.max_waypoints = max_waypoints
        self.max_segment = max_segment
        self.call_budget = call_budget
        self.distance_budget = distance_budget
        self.collision_budget = collision_budget
        self.log = log if log is not None else PipelineLogger("nav3d")
        self.failures = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.agent.cancel()

    def act(self, state: State) -> Trajectory:
        prompt = self.build_prompt(state, self.memory)
        try:
            payload = self.agent.run(prompt, TRAJECTORY_SCHEMA, system=SYSTEM_PROMPT)
            trajectory = Trajectory.from_structured_output(payload, limit=self.max_waypoints)
            if not trajectory.positions:
                raise ValueError("agent returned no usable positions")
            return trajectory
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            if self.cancelled:
                raise  # the user quit mid-call; the engine has stopped caring
            self.failures += 1
            self.log.info(state.calls, f"agent call failed ({exc}); this turn does not move")
            return Trajectory([], reasoning=f"call failed: {exc}")

    # ------------------------------------------------------------------ prompt

    def build_prompt(self, state: State, memory: SpatialMemory) -> str:
        half = state.bounds / 2
        visible_ids = {p.id for p in state.visible}
        remembered = memory.remembered(exclude=visible_ids)

        key = memory.key
        if key is not None:
            offset = key.position - state.player_position
            key_line = (f"THE KEY is at {key.position} — {offset.length():.1f} units away, "
                        f"offset {offset}. Fly to it and touch it.")
        else:
            key_line = ("THE KEY has not been detected yet. It is a small gold shape near one of "
                        "the assemblies. Fly somewhere you have not looked from yet.")

        budget = ""
        if self.call_budget is not None:
            budget = f" You have about {max(0, self.call_budget - state.calls)} calls left."
        travelled = f"Travelled {state.distance:.0f}"
        if self.distance_budget is not None:
            travelled += f" of {self.distance_budget:.0f}"
        collided = f"{state.collisions} collision(s)"
        if self.collision_budget is not None:
            collided += f" of {self.collision_budget} allowed"

        return f"""Find the key in a 3D world made of superquadrics.

The world is a cube: every axis runs from {-half:.0f} to {half:.0f} (+X right, +Y up, +Z forward).
Leaving it is a collision. There is no gravity and no ground — you fly.

You are at {state.player_position}, looking along {state.forward}.
This is call {state.calls + 1}.{budget}
{travelled} units so far, with {collided}.
Result of your previous batch: {state.last_outcome}

YOUR SENSOR reaches {self.sensor_range:.0f} units, only within your field of view, and only to
shapes nothing else is hiding. Once any part of a shape is detected you receive its
complete parameters and keep them forever.

VISIBLE NOW ({len(state.visible)} shapes):
{memory.table(state.player_position, state.visible)}

REMEMBERED FROM EARLIER ({len(remembered)} shapes, not currently in view):
{memory.table(state.player_position, remembered)}

You have observed from {len(memory.poses)} distinct vantage points, and have seen these
assemblies: {", ".join(sorted(memory.assemblies())) or "none yet"}.

{key_line}

READING A SHAPE: `centre` is its middle in world coordinates, `relative` is that minus
your position, `size` is its half-extent along its own three axes before rotation, and
`exps` set its form — (1, 1) is an ellipsoid, (0.2, 0.2) a box, (0.2, 1) a cylinder,
above 2 a pinched, concave shape. A shape is solid: touching one halts your batch.

HOW MOVEMENT WORKS:
- You return up to {self.max_waypoints} absolute target positions. Each is flown in order.
- Before each one, your camera turns to face it, so what you can see next is decided by
  where you fly. Repeating your current position does nothing and cannot turn the camera.
- A segment longer than {self.max_segment:.0f} units is shortened to {self.max_segment:.0f} along the same direction.
- The first segment that touches a shape or the world boundary halts the whole batch;
  every remaining target is discarded. So put the risky approach LAST.
- Keep a couple of units of clearance from the surface of any shape you pass.

Return target positions that make real progress: pick a direction with unexplored volume
in it, and check your straight line to each target does not pass through a remembered shape.
"""
