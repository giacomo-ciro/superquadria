"""Headless checks on the calibration rig's geometry, so the window only has to
answer questions an eye can answer."""

import random
import sys
from pathlib import Path

# Its own directory, not the working one: the pair lives in archive/ and has to
# import each other from wherever it is run.
sys.path.insert(0, str(Path(__file__).parent))
from calibrate import IN_PLANE, World, default_knobs, door_extent  # noqa: E402


def check(seed, overrides=None):
    k = {kn.name: kn.value for kn in default_knobs()}
    k.update(overrides or {})
    w = World(k, random.Random(seed))
    problems = []

    # 1. Every doorway hole is filled by exactly one door slab, and that slab
    #    sits on the plane it was cut from.
    for door in w.doors:
        prim = w.handler.get(door["id"])
        if abs(prim.position.as_tuple()[door["axis"]] - door["coord"]) > 1e-6:
            problems.append(f"door {door['id']} off its plane")
        u_axis, v_axis = IN_PLANE[door["axis"]]
        u0, v0, u1, v1 = door["hole"]
        got = (prim.scale[u_axis] * 2, prim.scale[v_axis] * 2)
        want = (u1 - u0, v1 - v0)
        if any(abs(a - b) > 1e-6 for a, b in zip(got, want)):
            problems.append(f"door {door['id']} {got} does not fill hole {want}")

    # 2. A door leaf must be as tall as door_height and as wide as door_width —
    #    catches the in-plane axis order silently transposing them. The height is
    #    the *clamped* one: a leaf taller than the room is not a valid door.
    tall = door_extent(k)
    for door in w.doors:
        if door["vertical"] is None:
            continue
        prim = w.handler.get(door["id"])
        if abs(prim.scale[1] * 2 - tall) > 1e-6:
            problems.append(f"door {door['id']} height {prim.scale[1] * 2:.2f} != {tall}")
        wide = prim.scale[0] * 2 if door["axis"] == 2 else prim.scale[2] * 2
        if abs(wide - k["door_width"]) > 1e-6:
            problems.append(f"door {door['id']} width {wide:.2f} != {k['door_width']}")

    # 2b. And it must stay inside the shell of the room it serves. A leaf taller
    #     than `level_height - wall_thickness` cuts a floor-to-ceiling hole and
    #     pokes through into the level above — invisible to check 2, which only
    #     ever compared the leaf against the knob that produced it.
    for door in w.doors:
        prim = w.handler.get(door["id"])
        lo, hi = prim.aabb()
        blo, bhi = w.rooms[door["a"]].box(k)
        # Only the in-plane axes: on its own normal the slab straddles the shared
        # plane by half a wall, which is exactly where it belongs.
        for axis in IN_PLANE[door["axis"]]:
            if lo[axis] < blo[axis] - 1e-6 or hi[axis] > bhi[axis] + 1e-6:
                problems.append(f"door {door['id']} escapes its room's shell on axis {axis}")
                break

    # 3. Sealed against the player: fly a straight segment out through every part
    #    of every shell plane bounding the spawn room, using the engine's own
    #    predicate at the engine's own increment. Not one may get through.
    #    Doorways are filled by solid door slabs, so the enclosure admits no
    #    exception — a single gap makes "the world has boundaries" false, which
    #    is the whole argument for decision 2.
    from geometry.geometry import Vec3
    room = w.spawn_room
    (bx0, by0, bz0), (bx1, by1, bz1) = room.box(k)
    r, inc = k["player_radius"], k["move_increment"]
    reach = k["wall_thickness"] + 4 * r
    leaks, samples, n = 0, 0, 14
    faces = [(0, bx0, 1), (0, bx1, -1), (2, bz0, 1), (2, bz1, -1),
             (1, by0, 1), (1, by1, -1)]
    for axis, coord, inward in faces:
        u_axis, v_axis = IN_PLANE[axis]
        span = {0: (bx0, bx1), 1: (by0, by1), 2: (bz0, bz1)}
        u0, u1 = span[u_axis]
        v0, v1 = span[v_axis]
        for i in range(n):
            for j in range(n):
                start = [0.0, 0.0, 0.0]
                start[axis] = coord + inward * (k["wall_thickness"] / 2 + r + 0.01)
                start[u_axis] = u0 + (u1 - u0) * (i + 0.5) / n
                start[v_axis] = v0 + (v1 - v0) * (j + 0.5) / n
                end = list(start)
                end[axis] = coord - inward * reach
                samples += 1
                if w.handler.segment_is_clear(Vec3(*start), Vec3(*end), r, inc):
                    leaks += 1
    if leaks:
        problems.append(f"shell leaks: the player flies out through "
                        f"{leaks}/{samples} of the enclosure")

    # 4. Objects and furniture stay out of the door clearance volume.
    from calibrate import door_clearance
    volumes = [door_clearance(k, d) for d in w.doors if room.index in (d["a"], d["b"])]
    for prim in w.handler:
        if w.roles[prim.id] not in ("furniture", "peg", "decoy"):
            continue
        lo, hi = prim.aabb()
        for vlo, vhi in volumes:
            if all(lo[a] < vhi[a] and hi[a] > vlo[a] for a in range(3)):
                problems.append(f"{prim.assembly} intrudes on a door clearance volume")

    # 5. The spawn point is not inside anything.
    if w.handler.blocking_primitive(w.spawn, k["player_radius"]) is not None:
        problems.append("spawn is blocked")

    # 6. The room actually holds the answer. Placement gives up after a bounded
    #    number of tries, and a room that came out without its peg is unsolvable
    #    — a far worse outcome than a room with one decoy fewer, and one that
    #    otherwise leaves no trace at all.
    #    A room one decoy short is still a valid room, so only the peg is fatal;
    #    the drop count rides along on the summary line as a crowding signal.
    if "peg" in w.dropped:
        problems.append("no peg placed - the room is unsolvable")

    return w, problems


if __name__ == "__main__":
    cases = [(s, None) for s in range(6)]
    cases += [(0, {"levels": 1, "rooms_per_level": 9}),
              (1, {"levels": 3, "rooms_per_level": 4}),
              (2, {"bounds": 30.0, "grid_unit": 2.0, "rooms_per_level": 6}),
              (3, {"wall_thickness": 0.8, "door_width": 4.0, "door_height": 5.0}),
              (4, {"grid_unit": 3.0, "level_height": 3.5, "rooms_per_level": 8}),
              # A tight plate at the placeholder player size: the combination
              # that used to give a door taller than the room and a room with no
              # peg in it, both silently.
              (5, {"bounds": 30.0, "grid_unit": 2.0, "level_height": 3.25}),
              # Human-scale: the player small enough that a door can be a door.
              (0, {"bounds": 30.0, "grid_unit": 2.0, "level_height": 3.25,
                   "wall_thickness": 0.3, "door_width": 1.5, "door_height": 2.2,
                   "player_radius": 0.3, "object_scale_min": 0.15,
                   "object_scale_max": 0.45})]
    failed = 0
    for seed, over in cases:
        w, problems = check(seed, over)
        tag = f"seed={seed} {over or ''}"
        print(f"{'FAIL' if problems else 'ok  '} {tag:<58} "
              f"rooms={len(w.rooms):>2} path={len(w.path)} doors={len(w.doors)} "
              f"prims={len(w.handler):>3} shell={w.walls:>3} dropped={len(w.dropped)}")
        for p in problems:
            print(f"       - {p}")
        failed += bool(problems)
    print(f"\n{len(cases) - failed}/{len(cases)} configurations clean")
