"""The shape-matching task: embossed locks, multi-primitive keys, and runtime
pickup/placement.

Every room on the task path specifies a `key_concept` (e.g. 'guitar', 'tree', 'chair').
One of the multi-primitive assemblies in the room is the key matching that concept.
The door displays an embossed relief replica of the key assembly in gold.
The agent explores the room, inspects the anonymized assemblies, picks up the matching
assembly, carries it to the door, and unlocks it.
"""

from geometry import Vec3
from geometry.superquadrics import DOOR, LOCK, PORTAL, Superquadric, SuperquadricHandler

from .layout import IN_PLANE, Door, Room, WorldConfig
from .scene import Scene

LOCK_COLOR = (1.00, 0.85, 0.10)
DEFAULT_REACH = 2.5


# ------------------------------------------------------------------ lock building


def build_lock(
    door: Door,
    room: Room,
    key_prims: list[Superquadric],
    handler: SuperquadricHandler,
    cfg: WorldConfig,
) -> list[Superquadric]:
    """Build an embossed relief lock on the door representing the key assembly.

    Scales down the key primitives to fit as an emblem on the door slab at ~0.62 height,
    facing into the room.
    """
    if not key_prims:
        return []

    # Compute bounding box and centroid of key primitives
    min_x = min(p.position.x - max(p.scale) for p in key_prims)
    max_x = max(p.position.x + max(p.scale) for p in key_prims)
    min_y = min(p.position.y - max(p.scale) for p in key_prims)
    max_y = max(p.position.y + max(p.scale) for p in key_prims)
    min_z = min(p.position.z - max(p.scale) for p in key_prims)
    max_z = max(p.position.z + max(p.scale) for p in key_prims)

    center = Vec3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
    extent = max(max_x - min_x, max_y - min_y, max_z - min_z, 0.1)

    # Target maximum dimension on the door
    target_extent = min(cfg.door_width * 0.45, 0.8)
    scale_factor = target_extent / extent

    axis, coord = door.axis, door.coord
    u0, v0, u1, v1 = door.hole
    u_axis, v_axis = IN_PLANE[axis]

    pos = [0.0, 0.0, 0.0]
    pos[u_axis] = (u0 + u1) / 2
    pos[v_axis] = (v0 + v1) / 2
    if door.vertical is not None:
        pos[1] = door.vertical[0] + (door.vertical[1] - door.vertical[0]) * 0.62

    inward = 1.0 if room.centre(cfg).as_tuple()[axis] > coord else -1.0
    pos[axis] = coord + inward * (cfg.wall_thickness / 2 + 0.06)
    door_target = Vec3(*pos)

    lock_primitives = []
    for prim in key_prims:
        rel_pos = prim.position - center
        scaled_rel = Vec3(
            rel_pos.x * scale_factor, rel_pos.y * scale_factor, rel_pos.z * scale_factor
        )
        new_pos = door_target + scaled_rel
        scaled_size = tuple(round(max(0.01, a * scale_factor), 4) for a in prim.scale)

        lock_prim = Superquadric(
            id=handler.next_id(),
            kind=LOCK,
            assembly="door",
            position=new_pos,
            rotation=prim.rotation,
            scale=scaled_size,
            exponents=prim.exponents,
            color=LOCK_COLOR,
        )
        handler.add(lock_prim)
        lock_primitives.append(lock_prim)

    return lock_primitives


# -------------------------------------------------------------- runtime rules


STRUCTURAL_KEYWORDS = (
    "wall",
    "floor",
    "ceiling",
    "door",
    "frame",
    "portal",
    "lock",
    "shell",
)


def is_pickable(prim: Superquadric) -> bool:
    """Is this primitive an interactable object/furniture rather than structural
    architecture?"""
    if prim.kind in (DOOR, PORTAL, LOCK):
        return False
    assembly_lower = prim.assembly.lower()
    if any(k in assembly_lower for k in STRUCTURAL_KEYWORDS):
        return False
    return True


def pick_at(
    scene: Scene, target: Vec3, *, reach: float = DEFAULT_REACH
) -> tuple[bool, str]:
    """Pick up the whole assembly at or near `target`.

    Must be within `reach` of the player.
    Returns (success, description).
    """
    if scene.player.carrying is not None and len(scene.player.carrying) > 0:
        return False, "hands are already full"

    player_pos = scene.player.position
    candidates = []
    for prim in scene.primitives:
        if is_pickable(prim):
            d_player = (prim.position - player_pos).length()
            d_target = (prim.position - target).length()
            if d_player <= reach:
                candidates.append((d_target, d_player, prim))

    if not candidates:
        for prim in scene.primitives:
            if is_pickable(prim) and (prim.position - target).length() <= reach:
                d_player = (prim.position - player_pos).length()
                return False, f"object is too far ({d_player:.1f} > reach {reach:.1f})"
        return False, "no object found within reach"

    candidates.sort(key=lambda item: item[0])
    chosen = candidates[0][2]
    target_assembly = chosen.assembly
    assembly_prims = [
        p for p in list(scene.primitives) if p.assembly == target_assembly
    ]

    for p in assembly_prims:
        scene.primitives.remove(p.id)

    scene.player.carrying = assembly_prims
    return True, f"picked up {target_assembly}"


def place_at(
    scene: Scene,
    target: Vec3,
    *,
    reach: float = DEFAULT_REACH,
) -> tuple[bool, str, str | None]:
    """Place the carried assembly at `target`, or attempt unlock if target or
    player is near a door/lock.

    Returns (success, description, door_result: 'opened' | 'locked' | None).
    """
    carrying = scene.player.carrying
    if not carrying:
        return False, "not carrying anything to place", None

    player_pos = scene.player.position
    if (target - player_pos).length() > reach:
        return (
            False,
            f"target is too far ({player_pos.distance_to(target):.1f} > "
            f"reach {reach:.1f})",
            None,
        )

    # Check if target or player is near a DOOR or LOCK
    door_or_lock = None
    for prim in scene.primitives:
        if prim.kind in (DOOR, LOCK):
            if (prim.position - target).length() <= reach or (
                prim.position - player_pos
            ).length() <= reach:
                door_or_lock = prim
                break

    if door_or_lock is not None:
        opened = attempt_unlock(scene, door_or_lock.id)
        if opened:
            return True, "unlocked the door with the carried object", "opened"
        else:
            return (
                True,
                "tried the locked door — mismatch, carried object was destroyed",
                "locked",
            )

    # Otherwise place down in the room preserving relative geometry
    centroid = sum((p.position for p in carrying), Vec3(0, 0, 0)) * (
        1.0 / len(carrying)
    )
    delta = target - centroid
    for prim in carrying:
        prim.position = prim.position + delta
        scene.primitives.add(prim)

    assembly_name = carrying[0].assembly
    scene.player.carrying = None
    return True, f"placed {assembly_name} at {target.rounded(1)}", None


def task_for_door_or_lock(scene: Scene, prim_id: int) -> dict | None:
    """Which room's task record owns a given door or lock primitive id."""
    for entry in scene.meta.get("task", {}).values():
        if entry.get("door_id") == prim_id:
            return entry
        lock_ids = entry.get("lock_ids", [])
        if prim_id in lock_ids or entry.get("lock_id") == prim_id:
            return entry
    return None


def attempt_unlock(scene: Scene, prim_id: int) -> bool:
    """Try the carried assembly against a door or its lock.

    Always clears inventory. If the carried assembly matches the task's key,
    the door becomes a portal. Otherwise the carried assembly is destroyed.
    """
    carrying = scene.player.carrying
    if not carrying:
        return False

    task = task_for_door_or_lock(scene, prim_id)
    if task is None:
        return False

    carried_assembly = carrying[0].assembly
    target_assembly = task.get("key_assembly")
    key_concept = task.get("key_concept", "")

    # Match by assembly ID, name, or concept
    is_match = carried_assembly == target_assembly or (
        key_concept and key_concept.lower() in carried_assembly.lower()
    )

    if is_match:
        door_id = task.get("door_id")
        if door_id is not None and door_id in scene.primitives:
            scene.primitives.get(door_id).kind = PORTAL
        lock_ids = task.get("lock_ids", [task.get("lock_id")])
        for lid in lock_ids:
            if lid is not None and lid in scene.primitives:
                scene.primitives.get(lid).kind = PORTAL

    scene.player.carrying = None
    return is_match
