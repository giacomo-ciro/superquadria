"""Room furnishing prompt with multi-primitive assembly guidance for 3D generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

IN_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

if TYPE_CHECKING:
    from geometry import Vec3
    from world.layout import Door, Room, WorldConfig


def _local_door_box(door: Door, origin: Vec3) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    u_axis, v_axis = IN_PLANE[door.axis]
    world_lo, world_hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    world_lo[door.axis] = world_hi[door.axis] = door.coord
    world_lo[u_axis], world_hi[u_axis] = door.hole[0], door.hole[2]
    world_lo[v_axis], world_hi[v_axis] = door.hole[1], door.hole[3]
    o = origin.as_tuple()
    lo = tuple(world_lo[i] - o[i] for i in range(3))
    hi = tuple(world_hi[i] - o[i] for i in range(3))
    lo_min = (min(lo[0], hi[0]), min(lo[1], hi[1]), min(lo[2], hi[2]))
    hi_max = (max(lo[0], hi[0]), max(lo[1], hi[1]), max(lo[2], hi[2]))
    return (lo_min, hi_max)


def room_prompt(room: Room, doors: list[Door], cfg: WorldConfig, interior, origin: Vec3, *,
                max_primitives: int, furniture_target: int, brief: str | None) -> str:
    """Prompt asking the room agent to furnish one room with superquadric assemblies."""
    (x0, y0, z0), (x1, y1, z1) = interior
    half_x, half_z, height = (x1 - x0) / 2, (z1 - z0) / 2, (y1 - y0)
    door_lines = []
    for door in doors:
        door_lo, door_hi = _local_door_box(door, origin)
        door_lines.append(
            f"- Doorway spans local X [{door_lo[0]:.1f}, {door_hi[0]:.1f}], "
            f"Y [{door_lo[1]:.1f}, {door_hi[1]:.1f}], "
            f"Z [{door_lo[2]:.1f}, {door_hi[2]:.1f}]."
        )
    door_text = "\n".join(door_lines)
    if door_text:
        door_text = f"\n{door_text}\n  Never place furniture inside doorways or blocking paths through them."
    building = f'\nThe building overall: "{brief}"\n' if brief else ""
    palette_info = (
        f"ROOM COLOR PALETTE (RGB 0.0 - 1.0):\n"
        f"- Wall:    [{room.wall_color[0]:.2f}, {room.wall_color[1]:.2f}, {room.wall_color[2]:.2f}]\n"
        f"- Floor:   [{room.floor_color[0]:.2f}, {room.floor_color[1]:.2f}, {room.floor_color[2]:.2f}]\n"
        f"- Ceiling: [{room.ceiling_color[0]:.2f}, {room.ceiling_color[1]:.2f}, {room.ceiling_color[2]:.2f}]"
    )
    return f"""You are furnishing one room of a building made entirely of 3D superquadrics.

ROOM: "{room.name}" — {room.style}{building}
{palette_info}
KEY OBJECT REQUIRED: "{room.key_concept}"
You MUST design one distinct assembly whose `name` contains or matches "{room.key_concept}" (e.g. "carved_{room.key_concept}"). This key object will be used by the player to unlock the room's door. The remaining assemblies should be functional furniture, architectural fixtures, or themed decor for this room.

ROOM BOUNDS (Room Local Frame)
- Floor is at Y = 0.0, Ceiling is at Y = {height:.1f}.
- Floor spans X: [{-half_x:.1f}, {half_x:.1f}], Z: [{-half_z:.1f}, {half_z:.1f}].{door_text}

ASSEMBLY ARCHITECTURE (Design in Local Coordinates, Place in Room)
Each piece of furniture is an `assembly` with:
- `name`: Descriptive name (e.g. "dining_table", "reading_armchair", "oak_bookshelf", "{room.key_concept}").
- `position`: [px, py, pz] placement in the room. For floor items, set py = 0.0. Ensure [px, pz] stays well within floor bounds.
- `yaw`: Horizontal rotation angle in degrees around the Y-axis (0.0 to 360.0).
- `primitives`: Array of 2 to 6 coordinated parts built in OBJECT-LOCAL coordinates.

OBJECT-LOCAL COORDINATES (Base-Centered Metric)
- Local Origin (0, 0, 0) is the HORIZONTAL CENTER of the base resting on the floor (Y = 0).
- `offset`: [dx, dy, dz] is the centre of the primitive relative to the assembly's base origin (0, 0, 0).
- `scale`: [sx, sy, sz] is half-extents in meters along local axes (full size is 2*sx × 2*sy × 2*sz).
- `rotation`: [rx, ry, rz] Euler degrees along local axes.

COLOR & MATERIAL RULES
- Every primitive MUST have a `color` array `[r, g, b]` (floats 0.0 to 1.0).
- Coordinate colors across parts of the same assembly (e.g. warm wood tabletop with dark iron legs).

OUTPUT REQUIREMENTS
- Aim for {furniture_target} total primitives partitioned across 3 to 6 rich assemblies (2 to 6 primitives per assembly).
- Total primitives must not exceed {max_primitives}.
- Ensure every assembly stays entirely within room bounds and does not block doorways.
"""
