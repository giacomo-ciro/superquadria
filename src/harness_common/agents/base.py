"""The Agent abstraction (PLAN.md section 4/5).

An agent is one thing: `run(prompt, schema) -> structured_output`. Everything a
harness asks of a model — invent a theme, draw a band of rows, choose the next
trajectory — goes through this single call, so swapping the backend (which CLI
drives the tmux session, `claude` or `codex`) touches nothing else in the
codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class AgentError(RuntimeError):
    """A failed agent invocation, whichever backend raised it."""


class Agent(ABC):
    """Produces JSON conforming to `schema` in response to `prompt`."""

    @abstractmethod
    def run(self, prompt: str, schema: dict, *, system: str | None = None) -> dict:
        """Return the structured output, or raise on failure."""

    def cancel(self) -> None:
        """Abandon an in-flight `run`, called from another thread when the user
        quits. Agents that spawn processes kill them here and stop retrying; an
        agent with nothing to clean up needs no override."""


class ScriptedAgent(Agent):
    """Deterministic stand-in used by the tests and by `--dry-run`.

    Takes either a list of payloads (returned in order, last one repeating) or a
    callable that maps (prompt, schema) to a payload.
    """

    def __init__(self, responses: list[dict] | Callable[[str, dict], dict]):
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    def run(self, prompt: str, schema: dict, *, system: str | None = None) -> dict:
        self.calls.append((prompt, schema))
        if callable(self._responses):
            return self._responses(prompt, schema)
        if not self._responses:
            raise RuntimeError("ScriptedAgent ran out of responses")
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]
