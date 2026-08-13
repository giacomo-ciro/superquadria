"""The navigation agent: observation in, batched trajectory out.

Collates the prompt, calls the agent through `Agent.run` with `MOVE_SCHEMA`
pinned, and parses the structured output into a `Move` (PLAN.md section 4).
"""

from __future__ import annotations

from collections import deque

from harness_common.agents.base import Agent, AgentError
from harness_common.pipeline_log import PipelineLogger

from .memory import FogMemory
from .moves import MAX_TRAJECTORY, MOVE_SCHEMA, Move
from .policies import FrontierPolicy, Policy
from .state import State

SYSTEM_PROMPT = (
    "You are the navigation policy for a grid-maze agent. You see a small window "
    "of the world and a map of what you have already seen. You answer only by "
    "emitting a trajectory through the StructuredOutput tool. Be decisive: commit "
    "to a direction and cover ground."
)


class ClaudeNavigator(Policy):
    """Navigation policy backed by an `Agent` (whichever CLI the config selected)."""

    name = "agent"

    def __init__(self, agent: Agent, *, max_steps: int = MAX_TRAJECTORY,
                 step_budget: int | None = None, fallback: Policy | None = None,
                 log: PipelineLogger | None = None):
        self.agent = agent
        # Saved into every episode as `policy`, so it has to say which CLI actually
        # played — a codex run recorded as "claude" is a corrupt comparison.
        self.name = getattr(agent, "binary", type(self).name)
        self.max_steps = max_steps
        self.step_budget = step_budget
        # If a call fails outright, keep the episode alive with the offline
        # policy rather than throwing away the whole run.
        self.fallback = fallback if fallback is not None else FrontierPolicy()
        self.log = log if log is not None else PipelineLogger("nav")
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=6)
        self.failures = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.agent.cancel()

    def act(self, state: State) -> Move:
        memory = self.memory
        self.recent_positions.append(state.player_position.as_tuple())
        prompt = self.build_prompt(state, memory)
        try:
            payload = self.agent.run(prompt, MOVE_SCHEMA, system=SYSTEM_PROMPT)
            move = Move.from_structured_output(payload)
            if not move.actions:
                raise ValueError("agent returned an empty trajectory")
            return Move(move.actions[: self.max_steps], raw=move.raw)
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            if self.cancelled:
                raise  # the user quit mid-call; not an agent failure, and the
                       # engine has already stopped caring about the answer
            self.failures += 1
            self.log.info(state.step, f"agent call failed ({exc}); falling back for this turn")
            self.fallback.memory = memory  # share the map, don't re-explore
            return self.fallback.act(state)

    # ------------------------------------------------------------------ prompt

    def build_prompt(self, state: State, memory: FogMemory) -> str:
        height, width = state.world_size
        unknown = memory.unknown_by_direction(state.player_position)

        if memory.key_position is not None:
            key_line = (f"The key is at {memory.key_position} — you have seen it. "
                        "Route to it and stand on it.")
        else:
            key_line = ("The key has not been seen yet. It is somewhere in the '?' region; "
                        "explore to uncover it.")

        stalled = ""
        if len(self.recent_positions) >= 3 and len(set(self.recent_positions)) == 1:
            stalled = ("\nWARNING: your last few plans moved you nowhere. You are walking into "
                       "a wall on the first step. Pick a different first direction.\n")

        budget = ""
        if self.step_budget is not None:
            budget = f" You have about {max(0, self.step_budget - state.step)} steps left."

        return f"""Find the key in a {height}x{width} grid maze.

You are at {state.player_position}. This is step {state.step}.{budget}
Result of your previous trajectory: {state.last_outcome}

CURRENT VIEW — the {len(state.grid)}x{len(state.grid[0])} window around you
('.' floor, '#' wall, 'K' key, 'P' you). Its top-left corner is world
{state.view_origin}:
{state.to_text()}

REMEMBERED MAP — every cell you have observed so far; '?' is never-seen:
{memory.render(state.player_position)}

Unseen cells lying in each direction: up={unknown['up']}, down={unknown['down']}, \
left={unknown['left']}, right={unknown['right']}.
Explored so far: {memory.explored_fraction():.0%} of the maze.

{key_line}
{stalled}
HOW MOVEMENT WORKS:
- Your actions run in order, one cell per action: up = row-1, down = row+1, left = col-1, right = col+1.
- The FIRST action that would step into a wall halts execution; every remaining action in that
  plan is discarded. So a plan that starts by walking into a wall wastes the whole turn.
- '?' cells may be floor or wall. Walking into them is how you reveal them, but keep the risky
  step near the END of a plan so a wrong guess costs you little.

Return up to {self.max_steps} actions that make real progress: trace a concrete route over the
remembered map, cell by cell, and check each step is not '#' before committing to it.
"""
