"""Scene entities and the integer codes the agent sees.

The `State` handed to the agent is a 2D array of these integers (PLAN.md
section 3), so the enum doubles as the observation vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .vector import Vector2D


class Tile(IntEnum):
    """Integer codes used in the observation grid."""

    EMPTY = 0
    WALL = 1
    KEY = 2
    PLAYER = 3
    UNKNOWN = 9  # only ever appears in the harness-side fog-of-war memory


#: Single-character glyphs for text renderings (prompts, terminal, tests).
GLYPHS: dict[int, str] = {
    Tile.EMPTY: ".",
    Tile.WALL: "#",
    Tile.KEY: "K",
    Tile.PLAYER: "P",
    Tile.UNKNOWN: "?",
}

#: Inverse of GLYPHS plus the aliases a language model tends to emit anyway.
GLYPH_TO_TILE: dict[str, int] = {
    ".": Tile.EMPTY,
    " ": Tile.EMPTY,
    "0": Tile.EMPTY,
    "-": Tile.EMPTY,
    "_": Tile.EMPTY,
    "#": Tile.WALL,
    "1": Tile.WALL,
    "X": Tile.WALL,
    "x": Tile.WALL,
    "*": Tile.WALL,
    "@": Tile.WALL,
}


@dataclass
class Entity:
    """A dynamic occupant of the scene (the player, the key).

    Static walls are *not* modelled as entities: a 100x100 maze would need ~5000
    of them and they are only ever queried positionally, so `Scene` keeps them in
    a terrain grid for O(1) lookups instead.
    """

    kind: Tile
    position: Vector2D

    def to_dict(self) -> dict:
        return {
            "kind": Tile(self.kind).name.lower(),
            "row": self.position.row,
            "col": self.position.col,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            kind=Tile[data["kind"].upper()],
            position=Vector2D(int(data["row"]), int(data["col"])),
        )
