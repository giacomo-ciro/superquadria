"""The observation handed to the navigation agent.

This is the harness's explicit departure from TASK.md's vision policy: the agent
never sees a screenshot. It sees the *parameters* of the superquadrics its sensor
has detected — the same validated numbers the renderer draws from — plus its own
pose, the shape-matching task state, and episode bookkeeping.

What it never sees: meshes, collision triangles, primitives it has not observed,
and the generation-time reachability/flood graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from geometry import Vec3
from geometry.superquadrics import Sensor, Superquadric

if TYPE_CHECKING:
    from world.scene import Scene


def describe(prim: Superquadric, viewer: Vec3) -> dict:
    """One primitive as the agent sees it."""
    offset = prim.position - viewer
    return {
        "id": prim.id,
        "assembly": " ".join(prim.assembly.split()),
        "position": list(prim.position.rounded(1)),
        "distance": round(offset.length(), 1),
        "scale": [round(a, 1) for a in prim.scale],
        "rotation": [round(a, 0) for a in prim.rotation],
        "exponents": [round(e, 2) for e in prim.exponents],
    }


def _carrying_dict(carrying: list[Superquadric] | None) -> dict | None:
    """The held assembly's details."""
    if not carrying:
        return None
    return {
        "assembly": carrying[0].assembly,
        "parts": len(carrying),
        "shapes": [{"scale": [round(a, 2) for a in p.scale],
                    "exponents": [round(e, 2) for e in p.exponents]}
                   for p in carrying],
    }


def _current_lock(scene: Scene, position: Vec3) -> dict | None:
    """The target object for the lock in the room the player is currently in."""
    room_index = scene.current_room_index(position)
    if room_index is None:
        return None
    entry = scene.meta.get("task", {}).get(str(room_index))
    if not entry:
        return None
    return {
        "concept": entry.get("key_concept", "key"),
        "door_id": entry.get("door_id"),
    }


@dataclass
class State:
    """One observation tick."""

    player_position: Vec3
    forward: Vec3
    bounds: float
    visible: list[Superquadric]
    room: str | None
    lock: dict | None
    carrying: dict | None
    calls: int = 0
    distance: float = 0.0
    collisions: int = 0
    rooms_cleared: int = 0
    failed_attempts: int = 0
    last_outcome: str = "start"
    room_description: str | None = None

    @classmethod
    def observe(cls, scene: Scene, sensor: Sensor, *, calls: int = 0, distance: float = 0.0,
                collisions: int = 0, rooms_cleared: int = 0, failed_attempts: int = 0,
                last_outcome: str = "start") -> "State":
        visible = scene.visible(sensor)
        return cls(
            player_position=scene.player.position,
            forward=scene.player.forward,
            bounds=scene.bounds,
            visible=visible,
            room=scene.current_room_name(scene.player.position),
            lock=_current_lock(scene, scene.player.position),
            carrying=_carrying_dict(scene.player.carrying),
            calls=calls,
            distance=distance,
            collisions=collisions,
            rooms_cleared=rooms_cleared,
            failed_attempts=failed_attempts,
            last_outcome=last_outcome,
            room_description=scene.current_room_description(scene.player.position),
        )

    def to_dict(self) -> dict:
        return {
            "position": list(self.player_position.rounded(2)),
            "forward": list(self.forward.rounded(3)),
            "bounds": self.bounds,
            "visible": [describe(p, self.player_position) for p in self.visible],
            "room": self.room,
            "room_description": self.room_description,
            "lock": self.lock,
            "carrying": self.carrying,
            "calls": self.calls,
            "distance": round(self.distance, 1),
            "collisions": self.collisions,
            "rooms_cleared": self.rooms_cleared,
            "failed_attempts": self.failed_attempts,
            "last_outcome": self.last_outcome,
        }
