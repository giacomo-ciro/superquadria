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


class PolicyQuit(Exception):
    """Raised by an interactive policy when the user quits mid-turn (e.g. closes
    the window while `ManualPolicy` is blocked waiting for a keypress)."""


class Policy(ABC):
    """Turns an observation into a batched trajectory."""

    name = "policy"

    #: Interactive policies read the render window, so the engine polls their
    #: `act` between frames instead of calling it once and waiting on it.
    main_thread_only = False

    @abstractmethod
    def act(self, state: State) -> Move | None:
        """The next trajectory. Only a `main_thread_only` policy may return
        None, meaning "nothing to do yet — ask again next frame"."""

    def cancel(self) -> None:
        """Abandon an in-flight `act`. The engine calls this from the render
        thread when the user closes the window; policies that block on a
        subprocess override it to kill one. Default is a no-op."""

    #: Populated by `observe`; the renderer draws it.
    memory: FogMemory | None = None

    def observe(self, state: State) -> FogMemory:
        """Fold one observation into the fog-of-war map.

        The engine calls this before `act` and again for every cell walked while
        executing the returned trajectory. Both happen on the main thread, which
        keeps every write to the map on the thread that renders it: `act` — the
        only part that may run off-thread — never writes.
        """
        if self.memory is None:
            height, width = state.world_size
            self.memory = FogMemory(height=height, width=width)
        self.memory.integrate(state)
        return self.memory


class FrontierPolicy(Policy):
    """Greedy frontier exploration over the remembered map."""

    name = "frontier"

    def __init__(self, max_steps: int = MAX_TRAJECTORY):
        self._rng = random.Random()
        self._max_steps = max_steps

    def act(self, state: State) -> Move:
        memory = self.memory
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


class ManualPolicy(Policy):
    """Human-controlled policy: one arrow-key press is one single-step `Move`.

    Returns None until a key arrives, rather than waiting for one itself. That
    matters because `PygameRenderer.draw()` is the only place the event queue is
    drained: a policy that drains it on its own swallows the window-resize and
    god-view-button events draw() never gets to see. So this reads the keys the
    last frame collected and lets the engine keep drawing between polls.
    """

    name = "manual"
    main_thread_only = True

    def __init__(self, renderer):
        self.renderer = renderer

    def act(self, state: State) -> Move | None:
        pygame = self.renderer.pygame
        key_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }
        for key in self.renderer.pending_keys:
            if key in (pygame.K_ESCAPE, pygame.K_q):
                raise PolicyQuit()
            if key in key_map:
                return Move([key_map[key]], reasoning="manual move")
        return None
