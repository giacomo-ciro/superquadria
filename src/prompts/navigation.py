"""System and user navigation prompts for 3D flight/waypoint policy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navigation.memory import SpatialMemory
    from navigation.state import State

NAV_SYSTEM_PROMPT: str = (
    "You are the navigation policy for an agent flying through a 3D world built "
    "entirely from superquadrics. You never see images — you see the parameters of "
    "the shapes your sensor has detected. You answer only by emitting a batch of "
    "actions (move, pick, place). Be decisive: plan efficient routes, pick the "
    "right shape, and open each room's door in turn."
)


def _sensor_reach_text(sensor_range: float) -> str:
    if math.isinf(sensor_range):
        return "has unlimited range: distance never hides anything, only walls do"
    return f"reaches {sensor_range:.0f} units"


def nav_user_prompt(
    state: State,
    memory: SpatialMemory,
    *,
    sensor_range: float,
    max_waypoints: int,
    max_segment: float,
    call_budget: int | None = None,
    distance_budget: float | None = None,
    collision_budget: int | None = None,
) -> str:
    """Prompt asking the navigation agent to plan a batch of actions from its
    current observation."""
    visible_ids = {p.id for p in state.visible}
    remembered = memory.remembered(exclude=visible_ids)
    if state.carrying and "assembly" in state.carrying:
        remembered = [p for p in remembered if p.assembly != state.carrying["assembly"]]

    if state.lock is not None and "concept" in state.lock:
        lock_line = f"THE DOOR LOCK REQUIRES: {state.lock['concept'].upper()}"
    else:
        lock_line = "THE DOOR LOCK REQUIRES: NONE"

    if state.carrying is not None:
        carrying_line = (
            f"You are CARRYING: {state.carrying['assembly']} "
            f"({state.carrying['parts']} part(s)). "
            f"Place it at this room's locked door to attempt "
            f"opening it, or drop it on the floor if you want to free your hands."
        )
    else:
        carrying_line = (
            "You are carrying nothing. Fly close to any assembly (<= 2.5 units) "
            "and issue a `pick` action to pick up the entire assembly."
        )

    budget = ""
    if call_budget is not None:
        budget = f" You have about {max(0, call_budget - state.calls)} calls left."
    travelled = f"Travelled {state.distance:.0f}"
    if distance_budget is not None:
        travelled += f" of {distance_budget:.0f}"
    collided = f"{state.collisions} collision(s)"
    if collision_budget is not None:
        collided += f" of {collision_budget} allowed"

    reach = _sensor_reach_text(sensor_range)

    return f"""Find the object matching the required shape name, carry it to the door, and open it. Each room on your route has its own locked door requiring its own key shape — solving this one only lets you into the next room, which has a new lock requirement.

The building is made entirely of superquadrics. Walls, floors, ceilings, furniture and the locked door are solid and block you.

You are in {state.room or "an unnamed room"}, at {state.player_position}, looking along {state.forward}. This is call {state.calls + 1}.{budget}
{travelled} units so far, with {collided}.
Failed attempts: {state.failed_attempts}.
Result of your previous batch: {state.last_outcome}

{lock_line}
{carrying_line}

HOW THE TASK WORKS:
- Each room on your path has a locked door requiring a specific shape name (e.g. GUITAR, CHAIR, LAMP, SWORD, TELESCOPE) shown above at 'THE DOOR LOCK REQUIRES'.
- Explore the room and inspect the assemblies detected by your sensor. Each assembly (assembly-0, assembly-1, ...) is a compound object made of one or more superquadric primitives.
- Determine which assembly matches the required shape name based on its geometric composition (the sizes, exponents, and relative positions of its parts).
- Move close to the candidate assembly (within 2.5 units), pick it up, carry it to the locked door, and place it there:
  * If it is the correct key object: the door unlocks and opens! Fly through into the next room.
  * If it is the wrong object: the carried assembly is destroyed and your hands become empty. Return into the room to find the real key.
- You can pick up any furniture/prop assembly in the room. If you realize you picked the wrong object before reaching the door, you can drop it on the floor anywhere.

YOUR SENSOR {reach}, only within your field of view, and only to shapes nothing else is hiding in your current room. Once any part of an assembly is detected you receive all its shapes and remember them while in this room.

VISIBLE NOW ({len(state.visible)} shapes):
{memory.table(state.player_position, state.visible)}

REMEMBERED FROM EARLIER ({len(remembered)} shapes, not currently in view):
{memory.table(state.player_position, remembered)}

You have observed from {len(memory.room_poses)} distinct vantage points in this room.

READING A SHAPE: `assembly` groups primitives that belong together as parts of one object (assembly-0, assembly-1, ...). Walls, floor, ceiling, and the door (assembly 'door') form the room structure. `centre` is its middle in world coordinates, `dist` is Euclidean distance from you, `size` is its half-extent along its own three axes before rotation, and `exps` set its form — (1, 1) is an ellipsoid, (0.1, 0.1) a box, (0.1, 1) a cylinder, above 2 a pinched, concave shape.

ACTIONS YOU CAN RETURN (up to {max_waypoints} per batch):
1. `{{"type": "move", "target": [x, y, z]}}`:
   - Flies towards destination [x, y, z]. Each segment is capped to {max_segment:.0f} units.
   - Flying past an object will NEVER accidentally pick it up.
   - Hitting a solid wall halts the batch immediately.
2. `{{"type": "pick", "target": [x, y, z]}}`:
   - Picks up the entire assembly at/near [x, y, z].
   - You MUST be within 2.5 units reach of the object for `pick` to succeed.
3. `{{"type": "place", "target": [x, y, z]}}`:
   - Places the carried assembly at [x, y, z] (must be within 2.5 units reach).
   - If placed at the door, it attempts to unlock the door (consuming the object).
   - If placed away from the door, it drops the assembly on the floor at that spot.

Plan a sequence of actions: move to position, pick the object, move to the door, place at the door.
"""
