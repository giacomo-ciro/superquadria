"""The `State` object handed to the agent.

Per PLAN.md section 3 the observation proper is the 2D array of integers (the
10x10 local view). The remaining fields are harness bookkeeping — step counter,
absolute position, outcome of the previous trajectory — which the agent needs to
act coherently across calls but which reveal nothing about unobserved cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entities import GLYPHS, Tile
from .scene import VIEW_SIZE, Scene
from .vector import Vector2D


def grid_to_text(grid: list[list[int]], *, indent: str = "  ") -> str:
    """Render a tile-code grid as glyph rows with a column ruler."""
    width = len(grid[0]) if grid else 0
    ruler = "".join(str(c % 10) for c in range(width))
    lines = [f"{indent}   {ruler}"]
    for r, row in enumerate(grid):
        lines.append(
            f"{indent}{r:>2} " + "".join(GLYPHS.get(cell, "?") for cell in row)
        )
    return "\n".join(lines)


@dataclass
class State:
    """One observation tick."""

    grid: list[list[int]]  # the local view: the agent's only real percept
    player_local: Vector2D  # where the player sits inside `grid`
    player_position: Vector2D  # absolute position (odometry, not vision)
    view_origin: Vector2D  # absolute coordinate of grid[0][0]
    world_size: tuple[int, int]  # (height, width)
    step: int = 0
    last_outcome: str = "start"  # human-readable result of the previous trajectory
    key_visible: bool = False
    extras: dict = field(default_factory=dict)

    @classmethod
    def observe(
        cls,
        scene: Scene,
        *,
        step: int = 0,
        last_outcome: str = "start",
        size: int = VIEW_SIZE,
    ) -> "State":
        grid = scene.local_view(size=size)
        offset = (size - 1) // 2
        return cls(
            grid=grid,
            player_local=Vector2D(offset, offset),
            player_position=scene.player.position,
            view_origin=scene.view_origin(size=size),
            world_size=(scene.height, scene.width),
            step=step,
            last_outcome=last_outcome,
            key_visible=any(Tile.KEY in row for row in grid),
        )

    def to_text(self) -> str:
        return grid_to_text(self.grid)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "grid": self.grid,
            "player_local": self.player_local.as_tuple(),
            "player_position": self.player_position.as_tuple(),
            "view_origin": self.view_origin.as_tuple(),
            "world_size": list(self.world_size),
            "last_outcome": self.last_outcome,
            "key_visible": self.key_visible,
        }
