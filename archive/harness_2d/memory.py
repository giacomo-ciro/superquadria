"""Fog-of-war memory: what the player has seen so far.

The *observation* stays exactly what PLAN.md specifies — the 10x10 local view and
nothing else. But an agent re-invoked per turn is stateless, and remembering the
cells it has already looked at is bookkeeping the harness can do for it. This
class accumulates past observations into a map; it never contains a cell the
player has not actually observed, so it leaks nothing about the unexplored maze.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .entities import GLYPHS, Tile
from .moves import Direction
from .state import State
from .vector import Vector2D


@dataclass
class FogMemory:
    height: int
    width: int
    grid: list[bytearray] = field(init=False)
    visited: set[tuple[int, int]] = field(default_factory=set)
    key_position: Vector2D | None = field(default=None, init=False)
    #: Bumped whenever new cells are revealed, so the renderer can cache.
    revision: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.grid = [
            bytearray(int(Tile.UNKNOWN) for _ in range(self.width))
            for _ in range(self.height)
        ]

    # ------------------------------------------------------------------ update

    def integrate(self, state: State) -> int:
        """Fold one observation into the map. Returns how many cells were new."""
        origin = state.view_origin
        revealed = 0
        for dr, row in enumerate(state.grid):
            for dc, code in enumerate(row):
                r, c = origin.row + dr, origin.col + dc
                if not (0 <= r < self.height and 0 <= c < self.width):
                    continue
                # The player glyph describes the player, not the terrain it stands on.
                tile = Tile.EMPTY if code == Tile.PLAYER else Tile(code)
                if tile == Tile.KEY:
                    self.key_position = Vector2D(r, c)
                if self.grid[r][c] == Tile.UNKNOWN:
                    revealed += 1
                self.grid[r][c] = int(tile)
        self.visited.add(state.player_position.as_tuple())
        if revealed:
            self.revision += 1
        return revealed

    # ------------------------------------------------------------------ access

    def at(self, pos: Vector2D) -> int:
        if not (0 <= pos.row < self.height and 0 <= pos.col < self.width):
            return int(Tile.WALL)
        return self.grid[pos.row][pos.col]

    def is_known_open(self, pos: Vector2D) -> bool:
        return self.at(pos) in (Tile.EMPTY, Tile.KEY)

    def is_unknown(self, pos: Vector2D) -> bool:
        return self.at(pos) == Tile.UNKNOWN

    def explored_fraction(self) -> float:
        known = sum(1 for row in self.grid for cell in row if cell != Tile.UNKNOWN)
        return known / (self.height * self.width)

    def unknown_by_direction(self, pos: Vector2D) -> dict[str, int]:
        """How much unseen map lies in each cardinal half-plane from `pos`."""
        counts = {"up": 0, "down": 0, "left": 0, "right": 0}
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell != Tile.UNKNOWN:
                    continue
                if r < pos.row:
                    counts["up"] += 1
                elif r > pos.row:
                    counts["down"] += 1
                if c < pos.col:
                    counts["left"] += 1
                elif c > pos.col:
                    counts["right"] += 1
        return counts

    # ----------------------------------------------------------------- routing

    def plan(
        self,
        start: Vector2D,
        is_goal: Callable[[Vector2D], bool],
        *,
        allow_unknown: bool = True,
        limit: int = 4096,
    ) -> list[Direction] | None:
        """Shortest route over remembered cells to the first cell matching `is_goal`.

        Unknown cells are optimistically treated as walkable: that is what makes
        the route an exploration plan rather than a tour of known ground. If an
        unknown cell turns out to be a wall the trajectory simply halts there and
        the next observation corrects the map.
        """
        if is_goal(start):
            return []
        seen: dict[tuple[int, int], tuple[Vector2D, Direction] | None] = {
            start.as_tuple(): None
        }
        queue: deque[Vector2D] = deque([start])
        expanded = 0
        while queue and expanded < limit:
            cur = queue.popleft()
            expanded += 1
            for direction in Direction:
                nxt = cur + direction.delta
                if nxt.as_tuple() in seen:
                    continue
                if not (0 <= nxt.row < self.height and 0 <= nxt.col < self.width):
                    continue
                walkable = self.is_known_open(nxt) or (
                    allow_unknown and self.is_unknown(nxt)
                )
                if not walkable:
                    continue
                seen[nxt.as_tuple()] = (cur, direction)
                if is_goal(nxt):
                    return self._reconstruct(seen, nxt)
                queue.append(nxt)
        return None

    @staticmethod
    def _reconstruct(seen: dict, node: Vector2D) -> list[Direction]:
        path: list[Direction] = []
        while seen[node.as_tuple()] is not None:
            prev, direction = seen[node.as_tuple()]
            path.append(direction)
            node = prev
        return list(reversed(path))

    # --------------------------------------------------------------- rendering

    def render(
        self, player: Vector2D, *, marks: Iterable[tuple[Vector2D, str]] = ()
    ) -> str:
        """The remembered map as text, with column rulers and the player marked."""
        overlay = {pos.as_tuple(): glyph for pos, glyph in marks}
        overlay[player.as_tuple()] = "P"
        tens = "".join(str((c // 10) % 10) for c in range(self.width))
        units = "".join(str(c % 10) for c in range(self.width))
        lines = [f"     {tens}", f"     {units}"]
        for r, row in enumerate(self.grid):
            cells = [
                overlay.get((r, c), GLYPHS.get(cell, "?")) for c, cell in enumerate(row)
            ]
            lines.append(f"{r:>4} " + "".join(cells))
        return "\n".join(lines)
