"""Room furnishing prompt: two-stage (model objects, then place them) generation."""

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
                brief: str | None) -> str:
    """Prompt asking the room agent to model objects, then place them in one room."""
    (x0, y0, z0), (x1, y1, z1) = interior
    half_x, half_z, height = (x1 - x0) / 2, (z1 - z0) / 2, (y1 - y0)
    lane = round(2 * cfg.player_radius * 1.3, 1)

    door_lines = []
    for door in doors:
        door_lo, door_hi = _local_door_box(door, origin)
        door_lines.append(
            f"- A doorway occupies X [{door_lo[0]:.1f}, {door_hi[0]:.1f}], "
            f"Y [{door_lo[1]:.1f}, {door_hi[1]:.1f}], "
            f"Z [{door_lo[2]:.1f}, {door_hi[2]:.1f}]."
        )
    door_text = "\n".join(door_lines) if door_lines else "- This room has no doorway."
    building = f'\nThe building this room belongs to: "{brief}"\n' if brief else ""

    return f"""You are dressing one room of a building made entirely of superquadrics, for a
video game. Nothing is imported: every chair leg, every book, every candle flame is
a superquadric you specify by hand. Your job is to make the room look like a place,
not like a pile of blocks.

You work in two stages, both in one JSON answer.

STAGE 1 — MODEL the objects, each in its own local frame, as if on a workbench.
STAGE 2 — PLACE those objects in the room, by id, as many times as you like.

Modelling is where the craft goes. Placing is cheap: one line per instance. Build a
chair once and stand six of them around a table; build one book and shelve twenty.


================================================================================
THE PRIMITIVE
================================================================================
Every part of every object is one superquadric, the surface

    x = sx * cos^e1(u) * cos^e2(v)
    y = sy * cos^e1(u) * sin^e2(v)        u in [-pi/2, pi/2],  v in [-pi, pi]
    z = sz * sin^e1(u)

`scale` = [sx, sy, sz] are HALF-EXTENTS in metres. The surface never leaves the box
2*sx x 2*sy x 2*sz, whatever the exponents do, so `scale` alone fixes the footprint.

`exponents` = [e1, e2] bend that box into a shape. e2 shapes the XY cross-section;
e1 shapes the Z profile and therefore the end caps. Some examples, just for reference:

    [0.1, 0.1]  box, slab, plank, panel, tabletop, book
    [0.2, 0.6]  softened box — most furniture parts read better here than at 0.1
    [0.1, 1.0]  cylinder, its axis along the part's own local +Z
    [0.3, 1.0]  capsule, rounded rod, bolster, limb
    [0.5, 0.5]  rounded box — cushions, upholstery, loaves, bundles
    [1.0, 1.0]  sphere or ellipsoid
    [1.0, 0.1]  square-section pillow, bevelled slab
    [1.5, 1.0]  waisted, pinched — vases, urns, hourglass forms
    [2.0, 1.0]  spindle, double cone, finial, spike, pointed leaf
    [2.0, 2.0]  octahedron, faceted gem, crystal
    [0.1, 2.0]  diamond-section prism

Two consequences worth internalising. A cylinder's axis is its own local +Z, so a
vertical post is a cylinder rotated `[90, 0, 0]` and a horizontal rail running along
X is one rotated `[0, 90, 0]`. And since exponents cannot push the surface outside
the half-extent box, a cylinder of `scale [0.03, 0.03, 0.4]` is a rod 6 cm across
and 80 cm long — not 40 cm long.


================================================================================
STAGE 1 — OBJECTS (modelled in the object's own local frame)
================================================================================
Each entry in `objects` is a recipe, built once, knowing nothing about where it will
stand:

  `id`         unique slug, e.g. "ladderback_chair", "brass_candlestick".
  `primitives` the parts, 1 to 12 of them.

The object frame:
- Origin [0, 0, 0] is the centre of the object's footprint, at the level it rests on.
  +Y is up. Build the object so it sits on Y = 0 and grows upward.
- The object FACES +Z. A chair's backrest is at -Z and you sit looking toward +Z; a
  cabinet's doors open toward +Z. Be consistent — placement rotation depends on it.
- `offset` [dx, dy, dz]: the CENTRE of that part, in metres from the origin. A part
  resting on the ground has dy equal to its own half-height.
- `rotation` [rx, ry, rz]: degrees about the part's own centre, applied X, then Y,
  then Z.
- `scale`, `exponents`: as above. `color`: [r, g, b], floats 0.0-1.0.

Model the real thing. A table is not a slab: it is a top, an apron, four legs, and
the legs are inset from the corners. A bookshelf has a back panel, side panels, and
shelves at believable spacings. Proportions must be right in metres — a seat is
0.45 m off the floor, a worktop 0.9 m, a door handle 1.0 m, a shelf 0.3 m deep.


================================================================================
STAGE 2 — PLACEMENTS (the room's frame)
================================================================================
Each entry in `placements` stands one instance of one object in the room:

  `object`    the `id` of an object from stage 1.
  `position`  [px, py, pz] where the object's local origin lands. py = 0.0 puts it on
              the floor; raise py to stand a candlestick on a table 0.75 m tall.
  `rotation`  [rx, ry, rz] degrees, applied to the whole object about its origin. For
              anything resting on the floor use [0, yaw, 0] and turn it with yaw
              alone. Non-zero rx/rz is for leaning, tipping, wall-mounting and
              hanging — and then you must raise py yourself so it does not sink into
              the floor.
  `scale`     one uniform multiplier (1.0 = as modelled). Use it for variety — three
              books at 0.9, 1.0, 1.15 — not to fake a different object.

Repetition is your budget multiplier and your main source of richness. Use it.


================================================================================
THE ROOM
================================================================================
"{room.name}" — {room.style}{building}
Interior, in the room's local frame:
- Floor at Y = 0.0, ceiling at Y = {height:.1f}.
- Floor spans X [{-half_x:.1f}, {half_x:.1f}], Z [{-half_z:.1f}, {half_z:.1f}].
{door_text}

Palette already applied to the shell (harmonise with it; do not repaint it):
- Wall    [{room.wall_color[0]:.2f}, {room.wall_color[1]:.2f}, {room.wall_color[2]:.2f}]
- Floor   [{room.floor_color[0]:.2f}, {room.floor_color[1]:.2f}, {room.floor_color[2]:.2f}]
- Ceiling [{room.ceiling_color[0]:.2f}, {room.ceiling_color[1]:.2f}, {room.ceiling_color[2]:.2f}]

REQUIRED KEY OBJECT: "{room.key_concept}"
Exactly one object must have an `id` containing "{room.key_concept}", and it must have
EXACTLY ONE placement — the player identifies it by shape to unlock the door, so a
second copy makes the puzzle ambiguous. Model it as the most recognisable, most
carefully built thing in the room. Everything else is furniture, fixtures and decor.


================================================================================
HARD CONSTRAINTS — violations are dropped, not repaired
================================================================================
1. INSIDE THE ROOM: every primitive of every instance must lie entirely within the
   interior box above. Nothing may poke through a wall, the floor or the ceiling.
   An object flush against a wall sits at that wall's coordinate minus its own half
   depth. Any part that pokes out is dropped, leaving the object mutilated.
2. CLEAR OF DOORWAYS: nothing in or in front of a doorway.
3. NAVIGABLE: the player is a sphere of radius {cfg.player_radius:.1f} m and must be able to
   fly from anywhere in the room to every doorway. Leave open lanes at least {lane:.1f} m
   wide between obstacles, and keep the middle of the room crossable. If that check
   fails, THE ENTIRE ROOM IS DISCARDED and ships empty — the most expensive mistake
   you can make here, so keep circulation obvious.
4. Every `scale` component > 0. Every exponent > 0.


================================================================================
MAKING IT GOOD
================================================================================
- There is no budget. Use as many objects, parts and placements as the room
  genuinely needs — a well-built chair is worth seven primitives.
- Fill the room. Bare floor and bare walls read as unfinished. Put things against
  walls, in corners, on top of other things.
- Layer scales: large anchors (bed, table, shelving), mid pieces (chairs, chests,
  stools), then the small stuff that sells it — books, cups, tools, bottles, a
  folded cloth. The small stuff is where the room stops looking procedural.
- Vary the repeats. Six identical chairs at identical yaw look stamped; turn them,
  pull one out from the table, tilt one at 8 degrees.
- Colour with intent inside an object — warm oak top on dark iron legs, brass
  fittings on painted wood — and let the room read as one palette overall.
- Silhouette first. If the object is unrecognisable from across the room, add the
  parts that make it readable rather than parts that add detail.

Think about the layout before you write anything: what the room is for, where a
person walks, what stands against which wall. Then model, then place.

Put your layout reasoning in `notes`, briefly.
"""
