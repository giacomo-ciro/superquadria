"""The authoritative world model.

`Scene` owns the static terrain grid and the dynamic entities, and is the single
source of truth for the two verifiable, code-level predicates the harness cares
about (PLAN.md section 2):

  * collision  -> `is_blocked(pos)`, an O(1) terrain lookup
  * success    -> `is_solved()`, an O(1) equality check on two grid positions

Neither is inferred from pixels or from a VLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .entities import GLYPHS, Entity, Tile
from .moves import Direction
from .vector import Vector2D

#: Side length of the square local observation handed to the agent.
VIEW_SIZE = 10


@dataclass
class Scene:
    """A rectangular grid of terrain plus the entities standing on it."""

    width: int
    height: int
    terrain: list[bytearray]  # terrain[row][col] == Tile.WALL or Tile.EMPTY
    entities: list[Entity] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups

    def in_bounds(self, pos: Vector2D) -> bool:
        return 0 <= pos.row < self.height and 0 <= pos.col < self.width

    def is_wall(self, pos: Vector2D) -> bool:
        return self.in_bounds(pos) and self.terrain[pos.row][pos.col] == Tile.WALL

    def is_blocked(self, pos: Vector2D) -> bool:
        """Collision predicate: a wall cell, or off the edge of the world."""
        return not self.in_bounds(pos) or self.is_wall(pos)

    def entity(self, kind: Tile) -> Entity:
        for ent in self.entities:
            if ent.kind == kind:
                return ent
        raise KeyError(f"scene has no {Tile(kind).name.lower()} entity")

    @property
    def player(self) -> Entity:
        return self.entity(Tile.PLAYER)

    @property
    def key(self) -> Entity:
        return self.entity(Tile.KEY)

    def is_solved(self) -> bool:
        """Success predicate: player position == key position."""
        return self.player.position == self.key.position

    # ------------------------------------------------------------- simulation

    def step(self, direction: Direction) -> bool:
        """Attempt one step. Returns True if the player moved, False on collision.

        A False return is what halts the remainder of a batched trajectory.
        """
        target = self.player.position + direction.delta
        if self.is_blocked(target):
            return False
        self.player.position = target
        return True

    # ------------------------------------------------------------ observation

    def local_view(
        self, center: Vector2D | None = None, size: int = VIEW_SIZE
    ) -> list[list[int]]:
        """The size x size window of tile codes the agent observes.

        The player sits at offset `(size - 1) // 2` in both axes; cells outside
        the world read as walls so the agent perceives the boundary the same way
        it perceives any other obstacle.
        """
        center = self.player.position if center is None else center
        offset = (size - 1) // 2
        top_left = Vector2D(center.row - offset, center.col - offset)
        occupants = {
            ent.position: ent.kind for ent in self.entities if ent.kind != Tile.PLAYER
        }

        view: list[list[int]] = []
        for dr in range(size):
            row: list[int] = []
            for dc in range(size):
                pos = Vector2D(top_left.row + dr, top_left.col + dc)
                if not self.in_bounds(pos):
                    row.append(int(Tile.WALL))
                elif pos == self.player.position:
                    row.append(int(Tile.PLAYER))
                elif pos in occupants:
                    row.append(int(occupants[pos]))
                else:
                    row.append(int(self.terrain[pos.row][pos.col]))
            view.append(row)
        return view

    def view_origin(self, size: int = VIEW_SIZE) -> Vector2D:
        """Absolute coordinate of the local view's top-left corner."""
        offset = (size - 1) // 2
        return Vector2D(
            self.player.position.row - offset, self.player.position.col - offset
        )

    # -------------------------------------------------------- (de)serialisation

    def to_rows(self) -> list[str]:
        """Terrain as '#'/'.' strings (entities excluded)."""
        return ["".join(GLYPHS[cell] for cell in row) for row in self.terrain]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "terrain": self.to_rows(),
            "entities": [ent.to_dict() for ent in self.entities],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        from .maze_utils import (
            parse_rows,  # deferred: maze_utils pulls in Scene helpers
        )

        width, height = int(data["width"]), int(data["height"])
        terrain = parse_rows(data["terrain"], width=width, height=height)
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        return cls(
            width=width,
            height=height,
            terrain=terrain,
            entities=entities,
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
