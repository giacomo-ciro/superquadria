"""Policy interface and the offline reference policy.

`FrontierPolicy` is a deterministic, zero-API-call baseline: it explores toward
the nearest unseen cell and beelines once the key is in memory. It exists so the
environment can be exercised end to end without spending tokens, and so the
agent's performance has something to be measured against.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .memory import FogMemory
from .moves import MAX_TRAJECTORY, Direction, Move
from .state import State


class Policy(ABC):
    """Turns an observation into a batched trajectory."""

    name = "policy"

    @abstractmethod
    def act(self, state: State) -> Move:
        ...

    #: Populated by policies that keep a fog-of-war map; the renderer draws it.
    memory: FogMemory | None = None

    def _ensure_memory(self, state: State) -> FogMemory:
        if self.memory is None:
            height, width = state.world_size
            self.memory = FogMemory(height=height, width=width)
        self.memory.integrate(state)
        return self.memory


class FrontierPolicy(Policy):
    """Greedy frontier exploration over the remembered map."""

    name = "frontier"

    def __init__(self, seed: int | None = None, max_steps: int = MAX_TRAJECTORY):
        self._rng = random.Random(seed)
        self._max_steps = max_steps

    def act(self, state: State) -> Move:
        memory = self._ensure_memory(state)
        here = state.player_position

        if memory.key_position is not None:
            target = memory.key_position
            path = memory.plan(here, lambda p: p == target)
            if path:
                return Move(path[: self._max_steps], reasoning=f"key seen at {target}; closing in")

        path = memory.plan(here, memory.is_unknown)
        if path:
            return Move(path[: self._max_steps], reasoning="heading for the nearest unseen cell")

        # Fully explored (or boxed in): jiggle so the episode can still end.
        return Move([self._rng.choice(list(Direction))], reasoning="no frontier left; probing")
