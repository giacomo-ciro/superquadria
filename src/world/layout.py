"""The building's floor plan: rooms, doorways, validation and repair.

`Layout` is an abstract, levelled 2D model — a room is `{level, origin, size}`
on an integer grid, never a 3D box the agent has to get exactly right. The
harness turns that abstraction into walls in `shell.py`; nothing here emits a
primitive.

Unlike the old freeform world, a layout does not get thrown away wholesale when
part of it is wrong. `validate_and_repair` drops what is broken and stitches the
rest into one connected building, so the invariant ("a connected room graph with
a well-defined feed-forward path") belongs to the harness, not to the model's
prompt compliance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from geometry.geometry import Vec3

Triple = tuple[float, float, float]

#: In-plane axes for a plane normal to world axis N, in increasing world order.
#: A wall normal to X reads (Y, Z); one normal to Z reads (X, Y). Getting this
#: backwards silently transposes door width and height.
IN_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


@dataclass(frozen=True)
class WorldConfig:
    """Every geometric knob `layout.py`, `shell.py` and `task.py` share."""

    bounds: float = 60.0
    grid_unit: float = 4.0
    level_height: float = 5.0
    wall_thickness: float = 0.4
    door_width: float = 3.0
    door_height: float = 4.0
    player_radius: float = 0.6
    max_levels: int = 1
    max_rooms: int = 1

    @property
    def half(self) -> float:
        return self.bounds / 2

    @property
    def cells(self) -> int:
        return max(2, int(self.bounds / self.grid_unit))

    @property
    def min_cells(self) -> int:
        """A room must be able to hold a doorway in a wall and still be a room."""
        need = self.door_width + 2 * self.wall_thickness
        return max(2, math.ceil(need / self.grid_unit) + 1)

    @property
    def door_extent(self) -> float:
        """Door height, clamped to what the room can actually hold.

        The sill sits half a wall above the floor plane and the ceiling
        underside half a wall below the next one, so a leaf taller than
        `level_height - wall_thickness` would cut a floor-to-ceiling hole and
        poke through into the level above.
        """
        return min(self.door_height, self.level_height - self.wall_thickness)


@dataclass
class Room:
    """A levelled 2D footprint, exactly as `LAYOUT_SCHEMA` describes it."""

    id: str
    name: str
    style: str
    level: int
    origin: tuple[int, int]
    size: tuple[int, int]
    key_concept: str = "key"
    wall_color: Triple = (0.80, 0.80, 0.78)
    floor_color: Triple = (0.40, 0.36, 0.32)
    ceiling_color: Triple = (0.92, 0.92, 0.90)
    #: Position in the final, repaired room list. -1 until validation assigns it.
    index: int = -1

    @property
    def ix0(self) -> int:
        return self.origin[0]

    @property
    def iz0(self) -> int:
        return self.origin[1]

    @property
    def ix1(self) -> int:
        return self.origin[0] + self.size[0]

    @property
    def iz1(self) -> int:
        return self.origin[1] + self.size[1]

    def box(self, cfg: WorldConfig) -> tuple[Triple, Triple]:
        """World-space (lo, hi) of the shell planes bounding this room."""
        g, half = cfg.grid_unit, cfg.half
        y0 = -half + self.level * cfg.level_height
        return (
            (-half + self.ix0 * g, y0, -half + self.iz0 * g),
            (-half + self.ix1 * g, y0 + cfg.level_height, -half + self.iz1 * g),
        )

    def interior(self, cfg: WorldConfig) -> tuple[Triple, Triple]:
        """The free volume: the shell box inset by half a wall on every face."""
        (x0, y0, z0), (x1, y1, z1) = self.box(cfg)
        t = cfg.wall_thickness / 2
        return ((x0 + t, y0 + t, z0 + t), (x1 - t, y1 - t, z1 - t))

    def centre(self, cfg: WorldConfig) -> Vec3:
        (x0, y0, z0), (x1, y1, z1) = self.interior(cfg)
        return Vec3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    def contains_point(self, cfg: WorldConfig, position: Vec3) -> bool:
        (x0, y0, z0), (x1, y1, z1) = self.box(cfg)
        p = position.as_tuple()
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1


@dataclass
class Door:
    """A doorway cut along the task path — geometry only, no primitive yet.

    `id` is filled in by `shell.build_shell` once the door slab primitive
    actually exists, so a `Door` is complete only after the shell is built.
    """

    a: int  # "from" room index
    b: (
        int | None
    )  # "to" room index — None for the exit door: it leads nowhere, it ends the game
    axis: int  # 0 or 2: a wall door; 1: a floor hole between levels
    coord: float  # world-space coordinate of the plane the door sits in
    hole: tuple[
        float, float, float, float
    ]  # u0, v0, u1, v1 in the plane's in-plane axes
    vertical: tuple[float, float] | None  # (sill, lintel) world Y, wall doors only
    id: int | None = None


@dataclass
class Layout:
    """A validated, always-connected building."""

    theme: str
    description: str
    rooms: list[Room]
    #: Every accepted edge (agent-proposed or stitched for connectivity) — the
    #: rest of the connection list is advisory; it shaped the plan but does not
    #: become geometry.
    connections: list[tuple[int, int]]
    adjacency: dict[tuple[int, int], tuple]
    #: Room indices on the feed-forward task path, spawn room first.
    path: list[int]
    #: Doorways cut along that path — the only connections that become geometry.
    doors: list[Door]

    @property
    def spawn_room(self) -> Room:
        return self.rooms[self.path[0]]


# ------------------------------------------------------------- validate/repair


def _parse_color(val: object, default: Triple) -> Triple:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return (
                max(0.0, min(1.0, float(val[0]))),
                max(0.0, min(1.0, float(val[1]))),
                max(0.0, min(1.0, float(val[2]))),
            )
        except (ValueError, TypeError):
            pass
    return default


def _parse_room(item: dict, cfg: WorldConfig) -> Room:
    room_id = " ".join(str(item.get("id", "")).split())[:64]
    if not room_id:
        raise ValueError("empty room id")
    name = " ".join(str(item.get("name", "")).split())[:60] or room_id
    style = " ".join(str(item.get("style", "")).split())[:400]
    key_concept = " ".join(str(item.get("key_concept", "")).split())[:40] or "key"
    level = int(item["level"])
    if not (0 <= level < cfg.max_levels):
        raise ValueError("level out of range")
    ox, oz = (int(v) for v in item["origin"])
    sx, sz = (int(v) for v in item["size"])
    if sx < cfg.min_cells or sz < cfg.min_cells:
        raise ValueError("room below the minimum size")
    if ox < 0 or oz < 0 or ox + sx > cfg.cells or oz + sz > cfg.cells:
        raise ValueError("footprint outside the floor plate")
    palette_dict = item.get("palette") if isinstance(item.get("palette"), dict) else {}
    wall_color = _parse_color(
        palette_dict.get("wall") or item.get("wall_color"), (0.80, 0.80, 0.78)
    )
    floor_color = _parse_color(
        palette_dict.get("floor") or item.get("floor_color"), (0.40, 0.36, 0.32)
    )
    ceiling_color = _parse_color(
        palette_dict.get("ceiling") or item.get("ceiling_color"), (0.92, 0.92, 0.90)
    )
    return Room(
        id=room_id,
        name=name,
        style=style,
        level=level,
        origin=(ox, oz),
        size=(sx, sz),
        key_concept=key_concept,
        wall_color=wall_color,
        floor_color=floor_color,
        ceiling_color=ceiling_color,
    )


def _footprints_overlap(a: Room, b: Room) -> bool:
    return a.ix0 < b.ix1 and b.ix0 < a.ix1 and a.iz0 < b.iz1 and b.iz0 < a.iz1


def _geometric_adjacency(
    rooms: list[Room], cfg: WorldConfig
) -> dict[tuple[str, str], tuple]:
    """Every room pair that *could* hold a doorway: edge-adjacent on one level
    with enough shared edge for a door, or footprint-overlapping across two
    adjacent levels with enough overlap on both axes."""
    need = cfg.door_width + 2 * cfg.wall_thickness
    g = cfg.grid_unit
    found: dict[tuple[str, str], tuple] = {}
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            if a.level == b.level:
                for axis, (a0, a1, b0, b1, oa0, oa1, ob0, ob1) in (
                    (0, (a.ix0, a.ix1, b.ix0, b.ix1, a.iz0, a.iz1, b.iz0, b.iz1)),
                    (2, (a.iz0, a.iz1, b.iz0, b.iz1, a.ix0, a.ix1, b.ix0, b.ix1)),
                ):
                    if a1 != b0 and b1 != a0:
                        continue
                    lo, hi = max(oa0, ob0), min(oa1, ob1)
                    if (hi - lo) * g >= need:
                        found[(a.id, b.id)] = (axis, a1 if a1 == b0 else a0, lo, hi)
            elif abs(a.level - b.level) == 1:
                lo_x, hi_x = max(a.ix0, b.ix0), min(a.ix1, b.ix1)
                lo_z, hi_z = max(a.iz0, b.iz0), min(a.iz1, b.iz1)
                if (hi_x - lo_x) * g >= need and (hi_z - lo_z) * g >= need:
                    upper = a if a.level > b.level else b
                    found[(a.id, b.id)] = (1, upper.level, (lo_x, hi_x), (lo_z, hi_z))
    return found


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _task_path(graph: dict[int, set[int]], start: int = 0) -> list[int]:
    """BFS from `start`, then the longest shortest-path chain out of it."""
    parent: dict[int, int | None] = {start: None}
    order = [start]
    for node in order:
        for nxt in graph.get(node, ()):
            if nxt not in parent:
                parent[nxt] = node
                order.append(nxt)
    node = order[-1]
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def _wall_door_geometry(
    cell: float, lo: float, hi: float, level: int, axis: int, cfg: WorldConfig
) -> tuple[float, tuple, tuple]:
    """coord, hole, vertical for a doorway in a wall normal to `axis` (0 or 2),
    centred on the [lo, hi] cell span shared by the two rooms it opens between
    — or, for an exit door, the free span of a single room's exterior wall."""
    half, g = cfg.half, cfg.grid_unit
    coord = -half + cell * g
    across = -half + (lo + hi) / 2 * g
    sill = -half + level * cfg.level_height + cfg.wall_thickness / 2
    vertical = (sill, sill + cfg.door_extent)
    along = (across - cfg.door_width / 2, across + cfg.door_width / 2)
    hole = (
        (vertical[0], along[0], vertical[1], along[1])
        if axis == 0
        else (along[0], vertical[0], along[1], vertical[1])
    )
    return coord, hole, vertical


def _doors_for_path(
    rooms: list[Room],
    adjacency: dict[tuple[int, int], tuple],
    path: list[int],
    cfg: WorldConfig,
) -> list[Door]:
    doors = []
    for a, b in zip(path, path[1:]):
        entry = adjacency.get((a, b)) or adjacency.get((b, a))
        if entry is None:
            continue  # cannot happen: path only ever walks graph edges
        axis = entry[0]
        if axis in (0, 2):
            _, cell, lo, hi = entry
            coord, hole, vertical = _wall_door_geometry(
                cell, lo, hi, rooms[a].level, axis, cfg
            )
        else:
            _, level, (lo_x, hi_x), (lo_z, hi_z) = entry
            half, g = cfg.half, cfg.grid_unit
            coord = -half + level * cfg.level_height
            cx = -half + (lo_x + hi_x) / 2 * g
            cz = -half + (lo_z + hi_z) / 2 * g
            d = cfg.door_width / 2
            vertical = None
            hole = (cx - d, cz - d, cx + d, cz + d)
        doors.append(
            Door(a=a, b=b, axis=axis, coord=coord, hole=hole, vertical=vertical)
        )
    return doors


def _shares_wall(
    other: Room, level: int, axis: int, cell: int, lo: int, hi: int
) -> bool:
    """Does `other` cover any part of the [lo, hi] span of the wall at
    `axis`/`cell`? Used to find a wall with nothing behind it at all — not
    just nothing on the task path — so the exit door never opens into a real,
    if unconnected, room."""
    if other.level != level:
        return False
    if axis == 0:
        return (
            (other.ix0 == cell or other.ix1 == cell)
            and other.iz0 < hi
            and lo < other.iz1
        )
    return (
        (other.iz0 == cell or other.iz1 == cell) and other.ix0 < hi and lo < other.ix1
    )


def _exterior_wall(
    room: Room, rooms: list[Room], cfg: WorldConfig
) -> tuple[int, int, int, int] | None:
    """A wall of `room` touching no other room and wide enough for a doorway,
    or None if all four are shared. The exit door goes here: cut into a wall
    with nothing behind it, "leads nowhere" is true by construction rather
    than by convention."""
    g = cfg.grid_unit
    candidates = [
        (0, room.ix0, room.iz0, room.iz1),
        (0, room.ix1, room.iz0, room.iz1),
        (2, room.iz0, room.ix0, room.ix1),
        (2, room.iz1, room.ix0, room.ix1),
    ]
    for axis, cell, lo, hi in candidates:
        if (hi - lo) * g < cfg.door_width + 2 * cfg.wall_thickness:
            continue
        if not any(
            _shares_wall(other, room.level, axis, cell, lo, hi)
            for other in rooms
            if other.id != room.id
        ):
            return axis, cell, lo, hi
    return None


def _exit_door(room: Room, rooms: list[Room], cfg: WorldConfig) -> Door | None:
    """The task's one locked door: on `room`, opening onto nothing. None if
    `room` has no free exterior wall to cut it into."""
    wall = _exterior_wall(room, rooms, cfg)
    if wall is None:
        return None
    axis, cell, lo, hi = wall
    coord, hole, vertical = _wall_door_geometry(cell, lo, hi, room.level, axis, cfg)
    return Door(
        a=room.index, b=None, axis=axis, coord=coord, hole=hole, vertical=vertical
    )


def validate_and_repair(raw: dict, cfg: WorldConfig) -> Layout:
    """Turn a possibly-broken agent layout into an always-connected `Layout`.

    Raises `ValueError` only when nothing usable survives at all (no rooms, or
    the last room on the task path has no free exterior wall for the exit
    door) — the caller's cue to fall back rather than repair, exactly as an
    unnavigable generated world used to. A single room is fine on its own: the
    exit door needs no second room to open onto, only a wall of its own.
    """
    theme = str(raw.get("theme", "Untitled"))[:80].strip() or "Untitled"
    description = str(raw.get("description", ""))[:600]

    raw_rooms = raw.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError("rooms is not a non-empty list")

    accepted: dict[str, Room] = {}
    for item in raw_rooms[: cfg.max_rooms]:
        try:
            room = _parse_room(item, cfg)
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        if room.id in accepted:
            continue
        if any(
            room.level == other.level and _footprints_overlap(room, other)
            for other in accepted.values()
        ):
            continue
        accepted[room.id] = room
    if not accepted:
        raise ValueError("no valid rooms survived validation")

    adjacency = _geometric_adjacency(list(accepted.values()), cfg)

    connections: set[tuple[str, str]] = set()
    for item in raw.get("connections") or []:
        try:
            a, b = str(item["from"]), str(item["to"])
        except (KeyError, TypeError):
            continue
        if a not in accepted or b not in accepted or a == b:
            continue
        if (a, b) in adjacency:
            connections.add((a, b))
        elif (b, a) in adjacency:
            connections.add((b, a))

    # The first room the agent listed that survived validation is the spawn
    # room — the same convention the calibration rig used.
    spawn_id = next(iter(accepted))
    graph: dict[str, set[str]] = {rid: set() for rid in accepted}
    for a, b in connections:
        graph[a].add(b)
        graph[b].add(a)

    reachable = _reachable(graph, spawn_id)
    changed = True
    while len(reachable) < len(accepted) and changed:
        changed = False
        for a, b in adjacency:
            if (a in reachable) != (b in reachable):
                connections.add((a, b))
                graph[a].add(b)
                graph[b].add(a)
                reachable = _reachable(graph, spawn_id)
                changed = True
                break

    for rid in list(accepted):
        if rid not in reachable:
            del accepted[rid]
    connections = {(a, b) for a, b in connections if a in accepted and b in accepted}
    adjacency = {
        k: v for k, v in adjacency.items() if k[0] in accepted and k[1] in accepted
    }

    ordered_ids = [spawn_id, *(rid for rid in accepted if rid != spawn_id)]
    rooms: list[Room] = []
    for index, rid in enumerate(ordered_ids):
        room = accepted[rid]
        room.index = index
        rooms.append(room)
    index_of = {room.id: room.index for room in rooms}

    index_graph: dict[int, set[int]] = {room.index: set() for room in rooms}
    for a, b in connections:
        index_graph[index_of[a]].add(index_of[b])
        index_graph[index_of[b]].add(index_of[a])
    index_adjacency = {
        (index_of[a], index_of[b]): value for (a, b), value in adjacency.items()
    }

    path = _task_path(index_graph, start=0)
    doors = _doors_for_path(rooms, index_adjacency, path, cfg)
    exit_door = _exit_door(rooms[path[-1]], rooms, cfg)
    if exit_door is None:
        raise ValueError(
            "last room on the task path has no exterior wall for an exit door"
        )
    doors.append(exit_door)

    return Layout(
        theme=theme,
        description=description,
        rooms=rooms,
        connections=sorted((index_of[a], index_of[b]) for a, b in connections),
        adjacency=index_adjacency,
        path=path,
        doors=doors,
    )
