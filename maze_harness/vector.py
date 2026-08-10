"""Integer 2D grid coordinates.

The MVP is fully discretised: the player occupies exactly one cell and moves one
cell at a time in a cardinal direction, so integer coordinates are sufficient and
there is no float precision to reason about (PLAN.md section 3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vector2D:
    """A (row, col) grid coordinate. Row grows downward, col grows rightward."""

    row: int
    col: int

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.row + other.row, self.col + other.col)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.row - other.row, self.col - other.col)

    def manhattan(self, other: "Vector2D") -> int:
        return abs(self.row - other.row) + abs(self.col - other.col)

    def as_tuple(self) -> tuple[int, int]:
        return (self.row, self.col)

    def __repr__(self) -> str:  # compact, shows up in prompts and logs
        return f"({self.row}, {self.col})"
