"""Calibration rig for the room/peg task — step 1 of docs/improvement.md.

**Archived: the gate is closed and the numbers are locked into `configs/3d.yaml`.**
Kept because it is the only executable record of how they were chosen, and
because the fallback in the plan's risk list — the harness partitions each floor
and the agent only names the rooms — is this file's `partition()`.

It exists to settle the numbers the plan leaves open, by building the procedural
(zero-agent-call) path far enough to render and letting you fly it with every
knob live.

Nothing here is the implementation. It reuses the harness's own `Superquadric`,
`build_mesh`, collision and sensor code so that what you judge is what the
harness will produce, but the layout, shell and task code below is a sketch —
the real versions land in layout.py / shell.py / task.py afterwards.

    python archive/calibrate.py  # fly it
    python archive/check.py      # headless geometry assertions

Controls
    mouse / WASD / space / shift   fly
    up / down                      select a knob
    left / right                   adjust it (world rebuilds)
    e                              probe: step max_segment along your look at
                                   move_increment, report what stops you
    c                              collision on/off (fly outside to inspect)
    m                              release / recapture the mouse
    p                              back to spawn
    enter                          print every knob, grouped for configs/3d.yaml
    escape                         print them once more, then quit
"""

from __future__ import annotations

import math
import random
import time

from geometry import Vec3, look_angles, look_vector
from engine.render import BACKGROUND, _matte_shader
from geometry.superquadrics import (KEY, OBSTACLE, Sensor, Superquadric,
                                   SuperquadricHandler, build_mesh)

MESH_RESOLUTION = 16
COLLISION_RESOLUTION = 8
SENSOR_ASPECT = 1.6
#: Unbounded, and no longer a knob. Inside a sealed room occlusion is the real
#: limit and a distance cap never bound anything — measured identical to range 45
#: on every room configuration. What you can see is what a wall is not hiding.
SENSOR_RANGE = math.inf
FLY_SPEED = 12.0
MOUSE_SENSITIVITY = 40.0

#: Task hues, per new_task.md. Structure grey is a knob, because it is one of
#: the two things the gate is meant to settle.
PEG_COLOR = (0.15, 0.90, 0.30)
DECOY_COLOR = (0.95, 0.15, 0.15)
LOCK_COLOR = (1.00, 0.85, 0.10)
#: Stand-in for a room's generated palette — deliberately nothing near the task
#: hues, which is the constraint the room call will have to carry.
FURNITURE_PALETTE = [(0.55, 0.42, 0.30), (0.30, 0.45, 0.50),
                     (0.45, 0.40, 0.55), (0.35, 0.50, 0.42)]

ROOM_NAMES = ["Copper Galley", "Ash Stair", "Slate Parlour", "Amber Vault",
              "Cinder Hall", "Quill Study", "Lime Pantry", "Iron Landing",
              "Moss Atrium", "Chalk Gallery", "Rust Workshop", "Pearl Solar"]


# ------------------------------------------------------------------- the knobs

class Knob:
    def __init__(self, name, value, step, lo, hi, fmt="{:.2f}", geometric=True, help=""):
        self.name, self.value, self.step = name, value, step
        self.lo, self.hi, self.fmt, self.geometric = lo, hi, fmt, geometric
        self.help = help

    def nudge(self, direction):
        before = self.value
        self.value = min(self.hi, max(self.lo, self.value + direction * self.step))
        # Float steps accumulate binary dust that shows up in the printed YAML.
        self.value = round(self.value, 4)
        return self.value != before

    def text(self):
        return self.fmt.format(self.value)


def print_knob_guide(knobs):
    """What every knob is, printed once at startup.

    The window has room for a name and a number and nothing else, which is not
    enough to dial a knob you have never met. This is the missing half.
    """
    print("\n  knob               start     range          what it does")
    print("  " + "-" * 100)
    for knob in knobs:
        span = f"{knob.fmt.format(knob.lo)} .. {knob.fmt.format(knob.hi)}"
        print(f"  {knob.name:<17}{knob.text():>7}   {span:<14} {knob.help}")
    print("\n  up/down picks one, left/right moves it. Watch the right-hand panel: it "
          "turns anything\n  broken into a '!' warning line, which is faster than "
          "reasoning about the numbers.\n")


#: Which `configs/3d.yaml` section each knob lands in, and under what key where
#: the knob name and the config key differ. Anything not claimed here is rig-only
#: and gets printed commented out — derived that way round on purpose, so adding
#: a knob can never silently drop it from the dump.
DESTINATIONS = [
    ("world", {"bounds": "bounds", "grid_unit": "grid_unit", "level_height": "level_height",
               "wall_thickness": "wall_thickness", "door_width": "door_width",
               "door_height": "door_height", "player_radius": "player_radius",
               "levels": "levels"}),
    ("generation", {"object_scale_min": "object_scale_min",
                    "object_scale_max": "object_scale_max", "decoys": "decoys"}),
    ("sensor", {"sensor_fov": "fov"}),
    ("episode", {"move_increment": "move_increment", "max_segment": "max_segment"}),
]

#: Why a knob has no config key, so the commented block explains itself.
RIG_ONLY = {
    "seed": "picks which world you are looking at",
    "rooms_per_level": "the layout agent's job; the rig only stands in for it",
    "furniture": "the room agent's job; capped by generation.max_primitives",
    "wall_grey": "a render.py constant, not a config key",
    "door_grey": "a render.py constant, not a config key",
    "furniture_grey": "a rig toggle for judging structure without distraction",
}


def default_knobs():
    """Every geometric figure starts at docs/improvement.md's placeholder, so the
    gate begins from what the plan currently claims rather than from a better
    guess. The two colour values are the exception: they are settled, by eye.
    """
    i = "{:.0f}"
    return [
        Knob("bounds", 60.0, 2.0, 12.0, 120.0, i,
             help="side of the world cube — the whole building's floor plate and height"),
        Knob("grid_unit", 4.0, 0.5, 1.0, 10.0,
             help="layout grid cell; rooms are whole multiples, so it sets how "
                  "coarsely room sizes can vary"),
        Knob("level_height", 5.0, 0.25, 2.0, 12.0,
             help="floor-to-floor height of one storey; the ceiling you fly under is "
                  "this minus wall_thickness"),
        Knob("wall_thickness", 0.4, 0.05, 0.05, 2.0,
             help="thickness of every wall, floor and ceiling slab"),
        Knob("door_width", 3.0, 0.25, 0.5, 8.0,
             help="doorway width; under ~2x the player diameter it gets hard to thread"),
        Knob("door_height", 4.0, 0.25, 1.0, 10.0,
             help="doorway height; silently clamped to level_height - wall_thickness"),
        Knob("player_radius", 0.6, 0.05, 0.1, 2.0,
             help="the player is a sphere this big — every clearance is relative to it, "
                  "so this is the scale of the world"),
        Knob("object_scale_min", 0.30, 0.05, 0.05, 3.0,
             help="smallest semiaxis a peg or decoy can have"),
        Knob("object_scale_max", 0.90, 0.05, 0.05, 4.0,
             help="largest semiaxis a peg or decoy can have; big objects crowd a small "
                  "room until nothing fits"),
        Knob("rooms_per_level", 6, 1, 1, 24, i,
             help="how many rooms each floor is cut into — more rooms, smaller rooms"),
        Knob("levels", 2, 1, 1, 5, i,
             help="storeys stacked inside the cube"),
        Knob("decoys", 3, 1, 0, 12, i,
             help="wrong-shaped objects sharing the room with the one true peg"),
        Knob("furniture", 10, 1, 0, 40, i,
             help="obstacles in the spawn room; this is what makes finding the peg "
                  "take actual searching"),
        Knob("seed", 0, 1, 0, 999, i,
             help="which world you get — step it to test a decision across worlds "
                  "rather than one lucky layout"),
        Knob("sensor_fov", 60.0, 5.0, 20.0, 140.0, i,
             help="horizontal field of view in degrees, for the sensor and this camera"),
        Knob("move_increment", 0.5, 0.05, 0.05, 3.0,
             help="collision sampling step; above 2*player_radius + wall_thickness the "
                  "player tunnels straight through walls"),
        Knob("max_segment", 15.0, 1.0, 1.0, 60.0,
             help="furthest a single waypoint may be; longer than a room span and "
                  "waypoints overshoot into walls"),
        Knob("wall_grey", 0.35, 0.02, 0.02, 1.0, geometric=False,
             help="brightness of walls, floors and ceilings"),
        Knob("door_grey", 0.71, 0.02, 0.02, 1.0, geometric=False,
             help="brightness of the door leaf; at wall_grey the door vanishes into "
                  "the wall and only the lock marks it"),
        Knob("furniture_grey", 1.0, 1.0, 0.0, 1.0, i, geometric=False,
             help="1 renders furniture grey like structure, 0 in the room's palette"),
    ]


# ----------------------------------------------------------------- the layout

class Room:
    """A levelled 2D footprint on the integer grid, exactly as LAYOUT_SCHEMA."""

    def __init__(self, index, name, level, ix0, iz0, ix1, iz1):
        self.index, self.name, self.level = index, name, level
        self.ix0, self.iz0, self.ix1, self.iz1 = ix0, iz0, ix1, iz1

    def box(self, k):
        """World-space (lo, hi) of the shell planes bounding this room."""
        g, half = k["grid_unit"], k["bounds"] / 2
        y0 = -half + self.level * k["level_height"]
        return ((-half + self.ix0 * g, y0, -half + self.iz0 * g),
                (-half + self.ix1 * g, y0 + k["level_height"], -half + self.iz1 * g))

    def interior(self, k):
        """The free volume: the shell box inset by half a wall on every face."""
        (x0, y0, z0), (x1, y1, z1) = self.box(k)
        t = k["wall_thickness"] / 2
        return ((x0 + t, y0 + t, z0 + t), (x1 - t, y1 - t, z1 - t))

    def centre(self, k):
        (x0, y0, z0), (x1, y1, z1) = self.interior(k)
        return Vec3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)


def partition(rng, cells, count, min_cells):
    """Binary space partition of the floor plate into `count` footprints.

    Stops early when nothing left is big enough to split, which is the honest
    answer for a plate that cannot hold that many rooms at this grid.
    """
    rects = [(0, 0, cells, cells)]
    while len(rects) < count:
        splittable = [r for r in rects
                      if r[2] - r[0] >= 2 * min_cells or r[3] - r[1] >= 2 * min_cells]
        if not splittable:
            break
        u0, v0, u1, v1 = max(splittable, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        rects.remove((u0, v0, u1, v1))
        wide = (u1 - u0) >= (v1 - v0)
        if wide and u1 - u0 >= 2 * min_cells:
            cut = rng.randint(u0 + min_cells, u1 - min_cells)
            rects += [(u0, v0, cut, v1), (cut, v0, u1, v1)]
        elif v1 - v0 >= 2 * min_cells:
            cut = rng.randint(v0 + min_cells, v1 - min_cells)
            rects += [(u0, v0, u1, cut), (u0, cut, u1, v1)]
        else:
            cut = rng.randint(u0 + min_cells, u1 - min_cells)
            rects += [(u0, v0, cut, v1), (cut, v0, u1, v1)]
    return rects


def build_rooms(k, rng):
    g = k["grid_unit"]
    cells = max(2, int(k["bounds"] / g))
    # A room has to be able to hold a doorway in a wall and still be a room.
    min_cells = max(2, math.ceil((k["door_width"] + 2 * k["wall_thickness"]) / g) + 1)
    rooms, index = [], 0
    for level in range(int(k["levels"])):
        for ix0, iz0, ix1, iz1 in partition(rng, cells, int(k["rooms_per_level"]), min_cells):
            rooms.append(Room(index, ROOM_NAMES[index % len(ROOM_NAMES)],
                              level, ix0, iz0, ix1, iz1))
            index += 1
    return rooms


def adjacencies(k, rooms):
    """Every pair that *could* hold a doorway: edge-adjacent on one level, or
    footprint-overlapping across two. The clearance test is what makes a shared
    edge a candidate rather than merely a touch."""
    need = k["door_width"] + 2 * k["wall_thickness"]
    g = k["grid_unit"]
    found = {}
    for a in rooms:
        for b in rooms:
            if b.index <= a.index:
                continue
            if a.level == b.level:
                for axis, (a0, a1, b0, b1, oa0, oa1, ob0, ob1) in (
                        (0, (a.ix0, a.ix1, b.ix0, b.ix1, a.iz0, a.iz1, b.iz0, b.iz1)),
                        (2, (a.iz0, a.iz1, b.iz0, b.iz1, a.ix0, a.ix1, b.ix0, b.ix1))):
                    if a1 != b0 and b1 != a0:
                        continue
                    lo, hi = max(oa0, ob0), min(oa1, ob1)
                    if (hi - lo) * g >= need:
                        found[(a.index, b.index)] = (axis, a1 if a1 == b0 else a0, lo, hi)
            elif abs(a.level - b.level) == 1:
                lo_x, hi_x = max(a.ix0, b.ix0), min(a.ix1, b.ix1)
                lo_z, hi_z = max(a.iz0, b.iz0), min(a.iz1, b.iz1)
                if (hi_x - lo_x) * g >= need and (hi_z - lo_z) * g >= need:
                    upper = a if a.level > b.level else b
                    found[(a.index, b.index)] = (1, upper.level, (lo_x, hi_x), (lo_z, hi_z))
    return found


def task_path(rooms, adj, start=0):
    """BFS from the spawn room, then the longest shortest-path chain out of it.
    Only these edges become doorways."""
    graph = {r.index: [] for r in rooms}
    for a, b in adj:
        graph[a].append(b)
        graph[b].append(a)
    parent, order = {start: None}, [start]
    for node in order:
        for nxt in graph[node]:
            if nxt not in parent:
                parent[nxt] = node
                order.append(nxt)
    path, node = [], order[-1]
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


# ------------------------------------------------------------------ the shell

def decompose(covered, holes):
    """union(covered) - union(holes) as non-overlapping axis-aligned rectangles.

    Coordinate compression: every rectangle edge becomes a grid line, each cell
    is tested once at its centre, then runs are merged along v and identical
    runs merged along u.
    """
    if not covered:
        return []
    us = sorted({round(v, 6) for r in covered + holes for v in (r[0], r[2])})
    vs = sorted({round(v, 6) for r in covered + holes for v in (r[1], r[3])})
    spans = {}
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


#: In-plane axes for a plane normal to world axis N, in increasing world order.
IN_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def door_extent(k):
    """Door height, clamped to what the room can actually hold.

    The sill sits half a wall above the floor plane and the ceiling underside
    half a wall below the next one, so a leaf taller than
    `level_height - wall_thickness` cuts a floor-to-ceiling hole and leaves the
    slab poking through into the level above. Clamped rather than rejected: the
    rig's job is to let you dial a knob past its limit and *see* the limit.
    """
    return min(k["door_height"], k["level_height"] - k["wall_thickness"])


def build_shell(k, rooms, adj, path):
    """Every wall, floor and ceiling, plus one solid door per path edge.

    One mechanism: group requested surfaces by plane, subtract the doorways,
    decompose the remainder into rectangles, emit a box per rectangle. Two rooms
    asking for the same wall plane land in the same group, so the shared wall is
    built once with no ownership rule.
    """
    t = k["wall_thickness"]
    planes, doors = {}, []

    for room in rooms:
        (x0, y0, z0), (x1, y1, z1) = room.box(k)
        for axis, coord, rect in ((0, x0, (y0, z0, y1, z1)), (0, x1, (y0, z0, y1, z1)),
                                  (2, z0, (x0, y0, x1, y1)), (2, z1, (x0, y0, x1, y1)),
                                  (1, y0, (x0, z0, x1, z1)), (1, y1, (x0, z0, x1, z1))):
            group = planes.setdefault((axis, round(coord, 6)), {"covered": [], "holes": [],
                                                                "rooms": set()})
            group["covered"].append(rect)
            group["rooms"].add(room.index)

    half, g = k["bounds"] / 2, k["grid_unit"]
    for a, b in zip(path, path[1:]):
        entry = adj.get((a, b)) or adj.get((b, a))
        axis = entry[0]
        if axis in (0, 2):
            _, cell, lo, hi = entry
            coord = -half + cell * g
            across = -half + (lo + hi) / 2 * g
            sill = -half + rooms[a].level * k["level_height"] + t / 2
            vertical = (sill, sill + door_extent(k))
            along = (across - k["door_width"] / 2, across + k["door_width"] / 2)
            # The plane's in-plane axes are ordered by world axis index, so a
            # wall normal to X reads (Y, Z) and one normal to Z reads (X, Y) —
            # vertical is the *first* pair on one and the second on the other.
            hole = ((vertical[0], along[0], vertical[1], along[1]) if axis == 0
                    else (along[0], vertical[0], along[1], vertical[1]))
        else:
            _, level, (lo_x, hi_x), (lo_z, hi_z) = entry
            coord = -half + level * k["level_height"]
            cx = -half + (lo_x + hi_x) / 2 * g
            cz = -half + (lo_z + hi_z) / 2 * g
            d = k["door_width"] / 2
            vertical = None
            hole = (cx - d, cz - d, cx + d, cz + d)
        planes[(axis, round(coord, 6))]["holes"].append(hole)
        doors.append({"a": a, "b": b, "axis": axis, "coord": coord, "hole": hole,
                      "vertical": vertical})

    prims, roles, next_id = [], {}, 0
    for (axis, coord), group in planes.items():
        u_axis, v_axis = IN_PLANE[axis]
        for u0, v0, u1, v1 in decompose(group["covered"], group["holes"]):
            pos, scale = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            pos[axis], scale[axis] = coord, t / 2
            pos[u_axis], scale[u_axis] = (u0 + u1) / 2, (u1 - u0) / 2
            pos[v_axis], scale[v_axis] = (v0 + v1) / 2, (v1 - v0) / 2
            owner = rooms[min(group["rooms"])].name
            face = {0: "wall", 1: "floor", 2: "wall"}[axis]
            prims.append(Superquadric(id=next_id, kind=OBSTACLE, assembly=f"{owner} {face}",
                                      position=Vec3(*pos), rotation=(0.0, 0.0, 0.0),
                                      scale=tuple(scale), exponents=(0.1, 0.1),
                                      color=(0.0, 0.0, 0.0)))
            roles[next_id] = "wall"
            next_id += 1

    for door in doors:
        axis = door["axis"]
        u_axis, v_axis = IN_PLANE[axis]
        u0, v0, u1, v1 = door["hole"]
        pos, scale = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        pos[axis], scale[axis] = door["coord"], t / 2
        pos[u_axis], scale[u_axis] = (u0 + u1) / 2, (u1 - u0) / 2
        pos[v_axis], scale[v_axis] = (v0 + v1) / 2, (v1 - v0) / 2
        names = (rooms[door["a"]].name, rooms[door["b"]].name)
        prims.append(Superquadric(id=next_id, kind=OBSTACLE,
                                  assembly=f"locked door: {names[0]} -> {names[1]}",
                                  position=Vec3(*pos), rotation=(0.0, 0.0, 0.0),
                                  scale=tuple(scale), exponents=(0.1, 0.1),
                                  color=(0.0, 0.0, 0.0)))
        roles[next_id] = "door"
        door["id"] = next_id
        next_id += 1

    return prims, roles, doors, next_id


# ---------------------------------------------------------------- the contents

def free_position(rng, interior, radius, taken, avoid, tries=200, floor_y=None):
    """A point in the interior box, clear of everything already placed and of
    the door clearance volume. `floor_y` pins the centre height instead of
    sampling it, which is how furniture ends up resting on the floor.
    """
    (x0, y0, z0), (x1, y1, z1) = interior
    if x1 - x0 < 2 * radius or z1 - z0 < 2 * radius:
        return None
    for _ in range(tries):
        y = floor_y if floor_y is not None else rng.uniform(min(y0 + radius, y1),
                                                            max(y1 - radius, y0))
        p = Vec3(rng.uniform(x0 + radius, x1 - radius), y,
                 rng.uniform(z0 + radius, z1 - radius))
        if any(p.distance_to(q) < r + radius for q, r in taken):
            continue
        # Against the *expanded* volume: testing the centre alone lets a wide
        # piece of furniture straddle the doorway with its middle outside it.
        if any(all(lo[a] - radius <= c <= hi[a] + radius
                   for a, c in enumerate(p.as_tuple()))
               for lo, hi in avoid):
            continue
        return p
    return None


def door_clearance(k, door):
    """The door rectangle extruded by wall_thickness/2 + 2*player_radius on both
    sides — nothing generated may intrude on it."""
    axis, coord = door["axis"], door["coord"]
    u0, v0, u1, v1 = door["hole"]
    u_axis, v_axis = IN_PLANE[axis]
    depth = k["wall_thickness"] / 2 + 2 * k["player_radius"]
    lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lo[axis], hi[axis] = coord - depth, coord + depth
    lo[u_axis], hi[u_axis] = u0, u1
    lo[v_axis], hi[v_axis] = v0, v1
    return (tuple(lo), tuple(hi))


def build_contents(k, rng, room, doors, next_id, spawn):
    """Furniture, then 1 peg + N decoys, in the spawn room."""
    prims, roles = [], {}
    avoid = [door_clearance(k, d) for d in doors if room.index in (d["a"], d["b"])]
    (x0, y0, z0), (x1, y1, z1) = room.interior(k)
    taken = [(spawn, k["player_radius"] * 3)]

    for n in range(int(k["furniture"])):
        s = (rng.uniform(0.3, 1.6), rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.6))
        radius = math.sqrt(sum(a * a for a in s))
        # Furniture rests on the floor rather than floating, which is what makes
        # it occlude the way real furniture does.
        p = free_position(rng, room.interior(k), radius, taken, avoid, floor_y=y0 + s[1])
        if p is None:
            continue
        taken.append((p, radius))
        prims.append(Superquadric(id=next_id, kind=OBSTACLE,
                                  assembly=f"{room.name} furniture-{n}", position=p,
                                  rotation=(0.0, rng.uniform(0, 360), 0.0), scale=s,
                                  exponents=(rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)),
                                  color=(0.0, 0.0, 0.0)))
        roles[next_id] = "furniture"
        next_id += 1

    lo, hi = k["object_scale_min"], k["object_scale_max"]
    peg_scale = tuple(rng.uniform(lo, hi) for _ in range(3))
    peg_exp = (rng.uniform(0.2, 2.0), rng.uniform(0.2, 2.0))
    shapes = [(peg_scale, peg_exp, "peg")]
    for _ in range(int(k["decoys"])):
        # Perturbed off the peg on at least one component, at a spread of
        # difficulties: some obviously wrong, some a single near miss.
        s, e = list(peg_scale), list(peg_exp)
        for _ in range(rng.choice([1, 1, 2, 3])):
            if rng.random() < 0.6:
                i = rng.randrange(3)
                s[i] = min(hi, max(lo, s[i] + rng.choice([-1, 1]) * rng.uniform(0.15, 0.5)))
            else:
                i = rng.randrange(2)
                e[i] = min(2.5, max(0.15, e[i] + rng.choice([-1, 1]) * rng.uniform(0.3, 1.2)))
        shapes.append((tuple(s), tuple(e), "decoy"))

    # The label comes from the shuffled order, so `object-0` is never the answer.
    # Placement, though, tries the peg *first*: in a room too tight to hold every
    # object it must not lose its slot to a decoy, which would leave the room
    # unsolvable with no signal that anything went wrong.
    rng.shuffle(shapes)
    dropped = []
    for n in sorted(range(len(shapes)), key=lambda i: shapes[i][2] != "peg"):
        s, e, role = shapes[n]
        radius = math.sqrt(sum(a * a for a in s))
        p = free_position(rng, room.interior(k), radius + k["player_radius"], taken, avoid)
        if p is None:
            dropped.append(role)
            continue
        # Separation exists so two objects do not read as one pile. That is a
        # judgement made by the observer, so it scales with the player rather
        # than with a constant that swamps the room once objects get small.
        taken.append((p, radius + 2 * k["player_radius"]))
        prims.append(Superquadric(id=next_id, kind=KEY, assembly=f"object-{n}", position=p,
                                  rotation=(0.0, rng.uniform(0, 360), 0.0), scale=s,
                                  exponents=e, color=(0.0, 0.0, 0.0)))
        roles[next_id] = role
        next_id += 1

    return prims, roles, peg_scale, peg_exp, next_id, dropped


def build_lock(k, door, room, peg_scale, peg_exp, next_id):
    """The answer, displayed on the puzzle: a lock primitive carrying the target
    parameters, mounted proud of the spawn-room face of the door."""
    axis, coord = door["axis"], door["coord"]
    u0, v0, u1, v1 = door["hole"]
    u_axis, v_axis = IN_PLANE[axis]
    pos = [0.0, 0.0, 0.0]
    pos[u_axis], pos[v_axis] = (u0 + u1) / 2, (v0 + v1) / 2
    if door["vertical"] is not None:
        # Mounted at eye height on the leaf rather than at its centre, so it is
        # the first thing read on approach.
        pos[1] = door["vertical"][0] + (door["vertical"][1] - door["vertical"][0]) * 0.62
    inward = 1.0 if room.centre(k).as_tuple()[axis] > coord else -1.0
    pos[axis] = coord + inward * (k["wall_thickness"] / 2 + max(peg_scale) + 0.05)
    return Superquadric(id=next_id, kind=KEY, assembly="lock", position=Vec3(*pos),
                        rotation=(0.0, 0.0, 0.0), scale=peg_scale, exponents=peg_exp,
                        color=LOCK_COLOR)


# ------------------------------------------------------------------- the world

class World:
    def __init__(self, k, rng):
        self.rooms = build_rooms(k, rng)
        self.adj = adjacencies(k, self.rooms)
        self.path = task_path(self.rooms, self.adj)
        prims, roles, doors, next_id = build_shell(k, self.rooms, self.adj, self.path)
        self.doors = doors
        spawn_room = self.rooms[self.path[0]]
        self.spawn_room = spawn_room
        self.spawn = spawn_room.centre(k)

        first = next((d for d in doors if spawn_room.index in (d["a"], d["b"])), None)
        contents, croles, peg_scale, peg_exp, next_id, dropped = build_contents(
            k, rng, spawn_room, doors, next_id, self.spawn)
        prims += contents
        roles.update(croles)
        self.peg_scale, self.peg_exp = peg_scale, peg_exp
        self.dropped = dropped

        if first is not None:
            lock = build_lock(k, first, spawn_room, peg_scale, peg_exp, next_id)
            prims.append(lock)
            roles[lock.id] = "lock"
            # Spawn facing away from the door: the search should not start solved.
            away = (self.spawn - lock.position)
            self.forward = Vec3(away.x, 0.0, away.z).normalized()
        else:
            self.forward = Vec3(0.0, 0.0, 1.0)

        self.roles = roles
        self.handler = SuperquadricHandler(prims, collision_resolution=COLLISION_RESOLUTION)
        self.walls = sum(1 for r in roles.values() if r == "wall")

    def recolour(self, k):
        wall = (k["wall_grey"],) * 3
        door = (k["door_grey"],) * 3
        for prim in self.handler:
            role = self.roles[prim.id]
            if role == "wall":
                prim.color = wall
            elif role == "door":
                prim.color = door
            elif role == "furniture":
                prim.color = ((k["wall_grey"],) * 3 if k["furniture_grey"] >= 0.5 else
                              FURNITURE_PALETTE[prim.id % len(FURNITURE_PALETTE)])
            elif role == "peg":
                prim.color = PEG_COLOR
            elif role == "decoy":
                prim.color = DECOY_COLOR
            else:
                prim.color = LOCK_COLOR

    def peg_visible_from_spawn(self, k):
        sensor = Sensor(range=SENSOR_RANGE, fov=k["sensor_fov"], aspect=SENSOR_ASPECT)
        seen = self.handler.visible_from(self.spawn, self.forward, sensor)
        return any(self.roles[p.id] == "peg" for p in seen)


# --------------------------------------------------------------------- the app

class Calibrator:
    def __init__(self):
        from ursina import Entity, Text, Ursina, camera, color, mouse, window

        self.knobs = default_knobs()
        self.selected = 0
        self.k = {knob.name: knob.value for knob in self.knobs}
        print_knob_guide(self.knobs)

        self.app = Ursina(title="calibration gate", size=(1400, 820), vsync=True,
                          development_mode=False, editor_ui_enabled=False,
                          show_ursina_splash=False)
        window.color = color.rgb(*BACKGROUND)
        self.camera, self.color, self.Entity, self.mouse = camera, color, Entity, mouse
        camera.clip_plane_near = 0.05
        self.shader = _matte_shader()

        self.entities = {}
        self.collide = True
        self.touching = "-"
        self.probe_text = "-"
        self.probe_line = None
        self.sensor_ms = 0.0
        self._last_sensor = 0.0
        self.rebuild_ms = 0.0

        self.hud = Text(parent=camera.ui, text="", position=(-0.86, 0.47), scale=0.7,
                        color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.facts = Text(parent=camera.ui, text="", position=(0.30, 0.47), scale=0.7,
                          color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.footer = Text(parent=camera.ui, text="", position=(-0.86, -0.43), scale=0.65,
                           color=color.rgba(0.55, 0.60, 0.68, 1.0), font="VeraMono.ttf")
        Text(parent=camera.ui, text="+", position=(0, 0), origin=(0, 0), scale=1.2,
             color=color.rgba(0.9, 0.9, 0.9, 0.5))

        Entity(input=self._on_input, update=self._update)
        mouse.locked = True
        self.rebuild()

    # ------------------------------------------------------------------ build

    def rebuild(self, geometry=True):
        from ursina import Mesh, destroy
        from ursina.shaders import unlit_shader

        self.k = {knob.name: knob.value for knob in self.knobs}
        started = time.perf_counter()
        if geometry:
            self.world = World(self.k, random.Random(int(self.k["seed"])))
            self.position = self.world.spawn
            self.pitch, self.yaw = look_angles(self.world.forward)
        self.world.recolour(self.k)

        for entity in self.entities.values():
            destroy(entity)
        self.entities = {}
        for prim in self.world.handler:
            mesh = self.world.handler.mesh_data(prim.id, MESH_RESOLUTION)
            solid = self.world.roles[prim.id] in ("wall", "door", "furniture")
            self.entities[prim.id] = self.Entity(
                model=Mesh(vertices=mesh.vertices, triangles=mesh.triangles,
                           normals=mesh.normals,
                           colors=[self.color.rgba(*prim.color, 1.0)] * len(mesh.vertices)),
                # Unlit for the task hues, exactly as the key renders today: flat
                # full saturation is what keeps red/green/yellow unmistakable.
                shader=self.shader if solid else unlit_shader)
        self.camera.fov = self.k["sensor_fov"]
        self.camera.clip_plane_far = self.k["bounds"] * 4
        self.rebuild_ms = (time.perf_counter() - started) * 1000

    def _recolour_only(self):
        self.world.recolour(self.k)
        for prim in self.world.handler:
            self.entities[prim.id].model.colors = [
                self.color.rgba(*prim.color, 1.0)] * len(self.entities[prim.id].model.vertices)
            self.entities[prim.id].model.generate()

    # ------------------------------------------------------------------ input

    def _on_input(self, key):
        from ursina import application

        if key == "escape":
            self._print_yaml()
            application.quit()
        elif key in ("up arrow", "down arrow"):
            self.selected = (self.selected + (1 if key == "down arrow" else -1)) % len(self.knobs)
        elif key in ("left arrow", "right arrow"):
            knob = self.knobs[self.selected]
            if knob.nudge(1 if key == "right arrow" else -1):
                self.k = {kn.name: kn.value for kn in self.knobs}
                if knob.geometric:
                    self.rebuild()
                else:
                    self._recolour_only()
        elif key == "c":
            self.collide = not self.collide
        elif key == "m":
            self.mouse.locked = not self.mouse.locked
        elif key == "p":
            self.position = self.world.spawn
            self.pitch, self.yaw = look_angles(self.world.forward)
        elif key == "e":
            self._probe()
        elif key == "enter":
            self._print_yaml()

    def _probe(self):
        """Walk one agent segment the way engine.py does: max_segment long,
        stepping move_increment, stopping at the first blocker."""
        from ursina import Mesh, destroy

        forward = look_vector(self.pitch, self.yaw)
        step, reach = self.k["move_increment"], self.k["max_segment"]
        radius, half = self.k["player_radius"], self.k["bounds"] / 2
        stop, blocker, travelled = self.position, None, 0.0
        while travelled < reach:
            travelled = min(reach, travelled + step)
            probe = self.position + forward * travelled
            if max(abs(probe.x), abs(probe.y), abs(probe.z)) > half - radius:
                blocker, travelled = "the world boundary", travelled
                break
            hit = self.world.handler.blocking_primitive(probe, radius)
            if hit is not None:
                blocker = hit.assembly
                break
            stop = probe
        self.probe_text = (f"{travelled:.1f}u -> {blocker}" if blocker
                           else f"{travelled:.1f}u -> clear")
        if self.probe_line is not None:
            destroy(self.probe_line)
        self.probe_line = self.Entity(
            model=Mesh(vertices=[self.position.as_tuple(), stop.as_tuple()],
                       mode="line", thickness=3),
            color=self.color.rgba(1.0, 0.6, 0.1, 0.9))

    def _print_yaml(self):
        """Every knob you can move, grouped by the `configs/3d.yaml` section it
        belongs to so the block actually pastes.

        Printed on `enter` and again on the way out, because an hour of dialling
        should not depend on remembering to press a key before quitting.
        """
        by_name = {knob.name: knob for knob in self.knobs}
        claimed = set()
        print("\n# --- calibration gate: paste into configs/3d.yaml ---")
        for section, keys in DESTINATIONS:
            print(f"{section}:")
            for name, key in keys.items():
                # The block is meant to be pasted, so a knob the geometry did not
                # actually honour has to say so on its own line — the derived
                # summary is too far away to stop you pasting it.
                note = ""
                if name == "door_height" and door_extent(self.k) < by_name[name].value:
                    note = (f"   # NOT HONOURED: clamped to {door_extent(self.k):.2f} by "
                            f"level_height - wall_thickness")
                print(f"  {key}: {by_name[name].text()}{note}")
                claimed.add(name)

        print("\n# no config key of their own:")
        for name, knob in by_name.items():
            if name not in claimed:
                why = RIG_ONLY.get(name, "not yet mapped to a config key")
                print(f"#   {name + ':':<17}{knob.text():>7}   # {why}")

        world, k = self.world, self.k
        print(f"\n# derived: {len(world.handler)} primitives, {world.walls} shell boxes, "
              f"{len(world.rooms)} rooms, path {len(world.path)}")
        print(f"# derived: door {k['door_width']:.2f} x {door_extent(k):.2f}u"
              f"{' (CLAMPED by the ceiling)' if door_extent(k) < k['door_height'] else ''}, "
              f"{k['door_width'] / (2 * k['player_radius']):.1f}x the player")
        if world.dropped:
            print(f"# derived: {len(world.dropped)} object(s) had nowhere to go"
                  f"{' INCLUDING THE PEG' if 'peg' in world.dropped else ''}")
        print()

    # ----------------------------------------------------------------- update

    def _update(self):
        from ursina import held_keys, time as utime

        if self.mouse.locked:
            self.yaw += self.mouse.velocity[0] * MOUSE_SENSITIVITY
            self.pitch = max(-89.0, min(89.0, self.pitch - self.mouse.velocity[1]
                                        * MOUSE_SENSITIVITY))
        ground = look_vector(0.0, self.yaw)
        side = Vec3(ground.z, 0.0, -ground.x)
        move = Vec3(0.0, 0.0, 0.0)
        for key, vector in (("w", ground), ("s", -ground), ("d", side), ("a", -side),
                            ("space", Vec3(0, 1, 0)), ("shift", Vec3(0, -1, 0)),
                            ("left shift", Vec3(0, -1, 0))):
            if held_keys[key]:
                move = move + vector
        if move.length_squared() > 0:
            target = self.position + move.normalized() * (FLY_SPEED * utime.dt)
            if not self.collide or self.world.handler.blocking_primitive(
                    target, self.k["player_radius"]) is None:
                self.position = target

        forward = look_vector(self.pitch, self.yaw)
        self.camera.position = self.position.as_tuple()
        self.camera.rotation = (self.pitch, self.yaw, 0)

        touched = [p for p in self.world.handler
                   if self.world.roles[p.id] in ("peg", "decoy", "lock")
                   and self.world.handler.touches(p, self.position, self.k["player_radius"])]
        self.touching = ", ".join(f"{p.assembly}[{self.world.roles[p.id]}]"
                                  for p in touched) or "-"

        now = time.perf_counter()
        if now - self._last_sensor > 1.0:
            sensor = Sensor(range=SENSOR_RANGE, fov=self.k["sensor_fov"],
                            aspect=SENSOR_ASPECT)
            started = time.perf_counter()
            self.visible = len(self.world.handler.visible_from(self.position, forward, sensor))
            self.sensor_ms = (time.perf_counter() - started) * 1000
            self._last_sensor = now
        self._draw_hud()

    def _draw_hud(self):
        rows = []
        for i, knob in enumerate(self.knobs):
            mark = ">" if i == self.selected else " "
            rows.append(f"{mark} {knob.name:<18}{knob.text():>7}")
        self.hud.text = "\n".join(rows)

        k, world = self.k, self.world
        spans = sorted((r.ix1 - r.ix0) * k["grid_unit"] for r in world.rooms)
        spans_z = sorted((r.iz1 - r.iz0) * k["grid_unit"] for r in world.rooms)
        median = spans[len(spans) // 2]
        median_z = spans_z[len(spans_z) // 2]
        interior_h = k["level_height"] - k["wall_thickness"]
        diameter = 2 * k["player_radius"]
        clearance = k["door_width"] / diameter
        tunnel = k["move_increment"] / (diameter + k["wall_thickness"])
        (x0, y0, z0), (x1, y1, z1) = world.spawn_room.interior(k)

        warn = []
        if clearance < 2.0:
            warn.append(f"! door is {clearance:.1f}x the player - hard to thread")
        if interior_h < 3 * diameter:
            warn.append("! ceiling under 3 player-diameters")
        if tunnel >= 1.0:
            warn.append("! move_increment can tunnel a wall")
        if k["max_segment"] > median * 1.5:
            warn.append(f"! max_segment {k['max_segment']:.0f}u vs {median:.0f}u room span")
        if len(world.path) < 2:
            warn.append("! no task path - spawn room has no doorway")
        if door_extent(k) < k["door_height"] - 1e-9:
            warn.append(f"! door clamped to {door_extent(k):.2f}u by the ceiling")
        if "peg" in world.dropped:
            warn.append("! NO PEG PLACED - the room is unsolvable")
        elif world.dropped:
            warn.append(f"! {len(world.dropped)} decoy(s) had nowhere to go")
        if world.peg_visible_from_spawn(k):
            warn.append("! peg visible from spawn - no search required")

        self.facts.text = "\n".join([
            f"rooms        {len(world.rooms)} over {int(k['levels'])} level(s)",
            f"room span    {median:.0f} x {median_z:.0f} x {interior_h:.1f} u (median)",
            f"spawn room   {world.spawn_room.name}",
            f"             {x1 - x0:.1f} x {z1 - z0:.1f} x {y1 - y0:.1f} u",
            f"task path    {len(world.path)} rooms, {len(world.doors)} doors",
            f"primitives   {len(world.handler)}  ({world.walls} shell boxes)",
            f"door clear   {k['door_width']:.2f}u / {diameter:.2f}u player = {clearance:.1f}x",
            f"lock/peg     scale {tuple(round(a, 2) for a in world.peg_scale)}",
            f"             exps  {tuple(round(a, 2) for a in world.peg_exp)}",
            f"visible      {getattr(self, 'visible', 0)} prims in {self.sensor_ms:.1f} ms",
            f"rebuild      {self.rebuild_ms:.0f} ms",
            f"touching     {self.touching}",
            f"probe (e)    {self.probe_text}",
            "",
            *warn,
        ])
        self.footer.text = ("up/down select   left/right adjust   e probe   c collision "
                            f"{'on' if self.collide else 'OFF'}   m mouse   p spawn   "
                            "enter print   esc print+quit")


def _shot_poses(world, k):
    """Fixed viewpoints worth capturing: what the agent sees at t=0, the lock on
    its door, an object close up, and the whole shell from outside."""
    lock = next((p for p in world.handler if world.roles[p.id] == "lock"), None)
    peg = next((p for p in world.handler if world.roles[p.id] == "peg"), None)
    poses = [("spawn", world.spawn, *look_angles(world.forward))]
    for name, target, back in (("door", lock, 5.0), ("peg", peg, 2.5)):
        if target is None:
            continue
        toward = (world.spawn - target.position).normalized()
        eye = target.position + toward * back
        poses.append((name, eye, *look_angles(target.position - eye)))
    half = k["bounds"] / 2
    eye = Vec3(half * 1.6, half * 1.1, half * 1.6)
    poses.append(("outside", eye, *look_angles(Vec3(0, 0, 0) - eye)))
    return poses


def take_shots(directory):
    import os

    from panda3d.core import Filename
    from ursina import application

    os.makedirs(directory, exist_ok=True)
    rig = Calibrator()
    poses = _shot_poses(rig.world, rig.k)
    state = {"frame": 0, "index": 0}

    def shoot():
        # A couple of frames per pose, so the camera move is on screen before
        # the buffer is read back.
        state["frame"] += 1
        if state["frame"] < 3:
            return
        state["frame"] = 0
        if state["index"] >= len(poses):
            # The write below always lags the camera move by one pose, so the
            # final pose still has to be flushed before quitting.
            path = f"{directory}/{poses[-1][0]}.png"
            application.base.win.saveScreenshot(Filename.fromOsSpecific(path))
            print("wrote", path)
            application.quit()
            return
        name, eye, pitch, yaw = poses[state["index"]]
        rig.position, rig.pitch, rig.yaw = eye, pitch, yaw
        if state["index"] > 0:
            path = f"{directory}/{poses[state['index'] - 1][0]}.png"
            application.base.win.saveScreenshot(Filename.fromOsSpecific(path))
            print("wrote", path)
        state["index"] += 1

    rig.Entity(update=shoot)
    return rig


if __name__ == "__main__":
    import sys

    if "--shot" in sys.argv:
        take_shots(sys.argv[sys.argv.index("--shot") + 1]).app.run()
    else:
        Calibrator().app.run()
