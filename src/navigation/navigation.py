"""The navigation agent: parameter observation in, target positions out.

Collates the prompt, calls the agent through `Agent.run` with
`TRAJECTORY_SCHEMA` pinned, and parses the structured output into a
`Trajectory`.

There is no fallback policy behind it: a failed call costs the turn and is
counted, because every alternative would have to route using knowledge the agent
is not allowed to have.
"""

from __future__ import annotations

import math

from agents.base import Agent, AgentError
from engine.logger import Logger, logger

from .memory import SpatialMemory
from .moves import MAX_WAYPOINTS, TRAJECTORY_SCHEMA, Trajectory
from .policies import Policy
from prompts.navigation import NAV_SYSTEM_PROMPT, nav_user_prompt
from .state import State


class WaypointNavigator(Policy):
    """Navigation policy backed by an `Agent` (whichever CLI the config selected)."""

    name = "agent"

    def __init__(self, agent: Agent, *, sensor_range: float = math.inf, max_waypoints: int = MAX_WAYPOINTS,
                 max_segment: float = 15.0, call_budget: int | None = None,
                 distance_budget: float | None = None, collision_budget: int | None = None,
                 log: Logger | None = None):
        self.agent = agent
        self.name = getattr(agent, "binary", type(self).name)
        self.sensor_range = sensor_range
        self.max_waypoints = max_waypoints
        self.max_segment = max_segment
        self.call_budget = call_budget
        self.distance_budget = distance_budget
        self.collision_budget = collision_budget
        self.log = log if log is not None else Logger("navigation")
        self.failures = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.agent.cancel()

    def act(self, state: State) -> Trajectory:
        prompt = self.build_prompt(state, self.memory)
        try:
            payload = self.agent.run(prompt, TRAJECTORY_SCHEMA, system=NAV_SYSTEM_PROMPT)
            trajectory = Trajectory.from_structured_output(payload, limit=self.max_waypoints)
            if not trajectory.actions:
                raise ValueError("agent returned no usable actions")
            return trajectory
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            if self.cancelled:
                raise  # the user quit mid-call; the engine has stopped caring
            self.failures += 1
            self.log.log(f"call {state.calls + 1}: agent call failed ({exc}); this turn does not move", stage="navigation:agent")
            return Trajectory([], reasoning=f"call failed: {exc}")

    # ------------------------------------------------------------------ prompt

    def build_prompt(self, state: State, memory: SpatialMemory) -> str:
        return nav_user_prompt(
            state, memory,
            sensor_range=self.sensor_range,
            max_waypoints=self.max_waypoints,
            max_segment=self.max_segment,
            call_budget=self.call_budget,
            distance_budget=self.distance_budget,
            collision_budget=self.collision_budget,
        )
