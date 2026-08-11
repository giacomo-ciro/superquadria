"""The authoritative 3D world model.

`Scene` is the single source of truth for the two verifiable, code-level
predicates the harness cares about:

  * collision -> `blocker(position)`, the cube boundary plus sphere-versus-mesh
  * success   -> `key_reached()`, the same intersection test applied to the
                 non-solid key primitive

Neither is inferred from pixels, and neither is delegated to Ursina — the
renderer is optional and never authoritative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import FORWARD, Vec3
from .superquadrics import KEY, KEY_COLOR, Sensor, Superquadric, SuperquadricHandler

#: Stands in for a primitive ID when the thing you hit is the world boundary.
#: Real primitive IDs are non-negative, so the two can never be confused.
BOUNDS = -1


@dataclass
class Player:
    """A sphere with a look direction. No gravity, velocity or inertia: this is
    controlled flight, not a physics simulator."""

    position: Vec3
    forward: Vec3 = FORWARD
    radius: float = 0.6

    def to_dict(self) -> dict:
        return {"position": list(self.position.rounded(3)),
                "forward": list(self.forward.rounded(4)),
                "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        return cls(position=Vec3.parse(data["position"]),
                   forward=Vec3.parse(data.get("forward", (0.0, 0.0, 1.0))).normalized(),
                   radius=float(data.get("radius", 0.6)))


@dataclass
class Scene:
    """A cube of free space with superquadrics in it, plus the player."""

    bounds: float
    primitives: SuperquadricHandler
    player: Player
    key_id: int
    meta: dict = field(default_factory=dict)

    @property
    def half(self) -> float:
        return self.bounds / 2

    @property
    def key(self) -> Superquadric:
        return self.primitives.get(self.key_id)

    # --------------------------------------------------------------- predicates

    def in_bounds(self, position: Vec3) -> bool:
        """Is the whole player sphere inside the playable cube?"""
        limit = self.half - self.player.radius
        return (abs(position.x) <= limit and abs(position.y) <= limit and abs(position.z) <= limit)

    def blocker(self, position: Vec3) -> int | None:
        """What stops the player standing here — `BOUNDS`, a primitive ID, or None."""
        if not self.in_bounds(position):
            return BOUNDS
        prim = self.primitives.blocking_primitive(position, self.player.radius)
        return prim.id if prim is not None else None

    def is_blocked(self, position: Vec3) -> bool:
        return self.blocker(position) is not None

    def key_reached(self) -> bool:
        """Success: the player sphere intersects the key's own surface.

        Deliberately the same intersection collision uses, so the key's scale
        stays visually meaningful instead of hiding an arbitrary centre-distance
        threshold.
        """
        return self.primitives.touches(self.key, self.player.position, self.player.radius)

    def visible(self, sensor: Sensor) -> list[Superquadric]:
        return self.primitives.visible_from(self.player.position, self.player.forward, sensor)

    def describe_blocker(self, blocker: int | None) -> str:
        if blocker is None:
            return "nothing"
        if blocker == BOUNDS:
            return "the world boundary"
        prim = self.primitives.get(blocker)
        return f"primitive {prim.id} ({prim.assembly})"

    # ---------------------------------------------------------- (de)serialisation

    def to_dict(self) -> dict:
        return {
            "bounds": self.bounds,
            "player": self.player.to_dict(),
            "key_id": self.key_id,
            "primitives": self.primitives.to_list(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict, *, collision_resolution: int = 8) -> "Scene":
        return cls(
            bounds=float(data["bounds"]),
            primitives=SuperquadricHandler.from_list(data["primitives"],
                                                     collision_resolution=collision_resolution),
            player=Player.from_dict(data["player"]),
            key_id=int(data["key_id"]),
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, *, collision_resolution: int = 8) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")),
                             collision_resolution=collision_resolution)


def key_primitive(prim_id: int, position: Vec3, scale: float) -> Superquadric:
    """The collectible: a small gold gem, non-blocking, harness-created only."""
    return Superquadric(id=prim_id, kind=KEY, assembly="key", position=position,
                        rotation=(0.0, 0.0, 0.0), scale=(scale, scale, scale),
                        exponents=(0.6, 0.6), color=KEY_COLOR)
