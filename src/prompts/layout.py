"""Floor plan and building layout prompt for 3D generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.layout import WorldConfig


def layout_prompt(cfg: WorldConfig, brief: str | None) -> str:
    """Prompt asking the layout agent to design the building floor plan."""
    if brief:
        intent = (
            f'The operator asked for this building:\n"{brief}"\n'
            "Turn that request into a floor plan."
        )
    else:
        intent = (
            "Nobody has specified what this building should be. Decide for "
            "yourself, and give it a distinctive spatial character rather than "
            "a generic office grid."
        )
    cells = cfg.cells
    return f"""You are designing the floor plan of a building made entirely of \
superquadrics.

{intent}

The floor plate is a {cells}x{cells} integer grid, cell [0, 0] to [{cells}, {cells}]. \
A room is a
rectangle on that grid: `origin` is its low corner [x, z], `size` is its extent \
[width, depth].
Rooms may be stacked across up to {cfg.max_levels} level(s) (0-indexed).

HARD CONSTRAINTS
- Every room's `size` must be at least {cfg.min_cells}x{cfg.min_cells} cells.
- A room's footprint (`origin` to `origin + size`) must stay inside [0, {cells}] \
on both axes.
- Room footprints on the same level must not overlap.
- Every `id` is a unique slug; every room needs a distinctive `name`, a `style` \
(the whole
  brief the room will be furnished from, describing its function and materials), \
a `key_concept`
  (a recognizable object archetype like that serves as the key to unlock this \
room's door), and
  a `palette` object with RGB floats in range 0.0-1.0 for `wall`, `floor`, and \
`ceiling` colors.
- List every doorway you want as a `connections` entry between two room ids. \
Only rooms sharing
  a long-enough wall (same level) or footprint (adjacent levels) can actually be \
connected — the
  harness drops anything that does not geometrically fit and repairs the rest \
into one connected
  building, so propose freely.
- The first room you list is where the player starts.
- A single room is fine if the brief calls for one — the exit door opens onto \
nothing and needs
  no second room. If the brief implies more than one space, connect them with doorways.

Design a building with real spatial character: rooms of varied size and proportion, \
corridors
where they help, and a floor plan that reads as one coherent place rather than a \
grid of equal
boxes.
"""
