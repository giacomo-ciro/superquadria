"""Shell construction: every wall, floor, ceiling and door slab, from a `Layout`.

One mechanism does all of it. Both a wall face with a doorway and a
level-boundary slab with a floor hole are *a plane, a covered region, and some
holes*, so:

1. Group every requested surface by its plane (axis + coordinate). Two
   edge-adjacent rooms requesting the same wall plane land in the same group,
   which removes coincident duplicate geometry without an ownership rule.
2. Per plane: collect covered rectangles (room faces) and hole rectangles
   (doorways on the task path only).
3. Decompose `union(covered) - union(holes)` into non-overlapping axis-aligned
   rectangles by coordinate compression.
4. Emit one box superquadric per rectangle, plus a solid, locked door slab per
   doorway on the task path — every room the path passes through gates its own
   way out, not just the last one.

Measured (`archive/calibrate.py`, 13 configurations): a 12-room, 2-level
building comes out at 22-27 shell boxes, ~37-42 primitives total.
"""

from __future__ import annotations

from geometry import Vec3
from geometry.superquadrics import DOOR, OBSTACLE, Superquadric
from .layout import IN_PLANE, Door, Layout, Triple, WorldConfig

#: Boxes are round at the corners at any finite exponent; this is as sharp as
#: the calibration gate exercised.
SHELL_EXPONENTS = (0.1, 0.1)

#: Structure brightness. Doors take their own value so a leaf reads as a
#: distinctly lighter panel in the wall rather than vanishing into it — settled
#: at the calibration gate. Not a config key.
WALL_GREY = 0.35
DOOR_GREY = 0.71


def decompose(covered: list[tuple], holes: list[tuple]) -> list[tuple]:
    """`union(covered) - union(holes)` as non-overlapping axis-aligned rectangles.

    Coordinate compression: every rectangle edge becomes a grid line, each cell
    is tested once at its centre, then runs are merged along v and identical
    runs merged along u.
    """
    if not covered:
        return []
    us = sorted({round(v, 6) for r in covered + holes for v in (r[0], r[2])})
    vs = sorted({round(v, 6) for r in covered + holes for v in (r[1], r[3])})
    spans: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(us) - 1):
        cu = (us[i] + us[i + 1]) / 2
        run, j = [], 0
        while j < len(vs) - 1:
            cv = (vs[j] + vs[j + 1]) / 2
            on = (any(r[0] < cu < r[2] and r[1] < cv < r[3] for r in covered)
                  and not any(h[0] < cu < h[2] and h[1] < cv < h[3] for h in holes))
            if on:
                j0 = j
                while j < len(vs) - 1:
                    cv = (vs[j] + vs[j + 1]) / 2
                    if not (any(r[0] < cu < r[2] and r[1] < cv < r[3] for r in covered)
                            and not any(h[0] < cu < h[2] and h[1] < cv < h[3] for h in holes)):
                        break
                    j += 1
                run.append((j0, j))
            else:
                j += 1
        spans[i] = run
    rects, used = [], set()
    for i in range(len(us) - 1):
        for span in spans[i]:
            if (i, span) in used:
                continue
            i1 = i + 1
            while i1 < len(us) - 1 and span in spans[i1] and (i1, span) not in used:
                used.add((i1, span))
                i1 += 1
            rects.append((us[i], vs[span[0]], us[i1], vs[span[1]]))
    return rects


def build_shell(layout: Layout, cfg: WorldConfig) -> tuple[list[Superquadric], int]:
    """Every wall, floor and ceiling, plus one solid, locked door slab per
    doorway on the task path.

    Every room the path passes through gates its own way forward: `task.py`
    places that room's peg and decoys and locks this slab, so the agent must
    solve room *n* before it can ever reach room *n+1*.

    Mutates each `Door` in `layout.doors` with the primitive id of its slab.
    """
    t = cfg.wall_thickness
    planes: dict[tuple[int, float], dict] = {}

    for room in layout.rooms:
        (x0, y0, z0), (x1, y1, z1) = room.box(cfg)
        for axis, coord, rect in ((0, x0, (y0, z0, y1, z1)), (0, x1, (y0, z0, y1, z1)),
                                  (2, z0, (x0, y0, x1, y1)), (2, z1, (x0, y0, x1, y1)),
                                  (1, y0, (x0, z0, x1, z1)), (1, y1, (x0, z0, x1, z1))):
            group = planes.setdefault((axis, round(coord, 6)),
                                      {"covered": [], "holes": [], "rooms": set()})
            group["covered"].append(rect)
            group["rooms"].add(room.index)

    for door in layout.doors:
        key = (door.axis, round(door.coord, 6))
        if key in planes:
            planes[key]["holes"].append(door.hole)

    prims: list[Superquadric] = []
    next_id = 0
    for (axis, coord), group in planes.items():
        u_axis, v_axis = IN_PLANE[axis]
        for u0, v0, u1, v1 in decompose(group["covered"], group["holes"]):
            pos, scale = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            pos[axis], scale[axis] = coord, t / 2
            pos[u_axis], scale[u_axis] = (u0 + u1) / 2, (u1 - u0) / 2
            pos[v_axis], scale[v_axis] = (v0 + v1) / 2, (v1 - v0) / 2
            owner_room = layout.rooms[min(group["rooms"])]
            if axis == 1:
                y0_room = -cfg.half + owner_room.level * cfg.level_height
                if abs(coord - y0_room) < 1e-4:
                    face = "floor"
                    prim_color = owner_room.floor_color
                else:
                    face = "ceiling"
                    prim_color = owner_room.ceiling_color
            else:
                face = "wall"
                prim_color = owner_room.wall_color
            prims.append(Superquadric(id=next_id, kind=OBSTACLE, assembly=face,
                                      position=Vec3(*pos), rotation=(0.0, 0.0, 0.0),
                                      scale=tuple(scale), exponents=SHELL_EXPONENTS,
                                      color=prim_color))
            next_id += 1

    for door in layout.doors:
        axis = door.axis
        u_axis, v_axis = IN_PLANE[axis]
        u0, v0, u1, v1 = door.hole
        pos, scale = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        pos[axis], scale[axis] = door.coord, t / 2
        pos[u_axis], scale[u_axis] = (u0 + u1) / 2, (u1 - u0) / 2
        pos[v_axis], scale[v_axis] = (v0 + v1) / 2, (v1 - v0) / 2
        prims.append(Superquadric(id=next_id, kind=DOOR, assembly="door",
                                  position=Vec3(*pos), rotation=(0.0, 0.0, 0.0),
                                  scale=tuple(scale), exponents=SHELL_EXPONENTS,
                                  color=(DOOR_GREY,) * 3))
        door.id = next_id
        next_id += 1

    return prims, next_id


def door_clearance(door: Door, cfg: WorldConfig) -> tuple[Triple, Triple]:
    """The door rectangle extruded by `wall_thickness/2 + 2*player_radius` on
    both sides. Nothing generated — furniture, peg or decoy — may intrude on it.

    Tested against a primitive's AABB, not its centre: testing the centre alone
    lets a wide piece of furniture straddle the doorway with its middle outside
    the volume.
    """
    axis, coord = door.axis, door.coord
    u0, v0, u1, v1 = door.hole
    u_axis, v_axis = IN_PLANE[axis]
    depth = cfg.wall_thickness / 2 + 2 * cfg.player_radius
    lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lo[axis], hi[axis] = coord - depth, coord + depth
    lo[u_axis], hi[u_axis] = u0, u1
    lo[v_axis], hi[v_axis] = v0, v1
    return (tuple(lo), tuple(hi))
