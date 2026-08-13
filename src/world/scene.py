"""The authoritative 3D world model.

`Scene` is the single source of truth for the verifiable, code-level predicates
the harness cares about: collision -> `blocker(position)`, the cube boundary
plus sphere-versus-mesh; contact -> `contacts(position)`, what the task module
resolves pickup and unlock attempts from. Success (`rooms_cleared`) is episode
bookkeeping, not a scene predicate — it happens once, on the frame a door opens.

Neither is inferred from pixels, and neither is delegated to Ursina — the
renderer is optional and never authoritative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from geometry import FORWARD, Vec3
from geometry.superquadrics import LOCK, Sensor, Superquadric, SuperquadricHandler

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
    #: The assembly the player is holding — `None`, or list of primitives.
    carrying: list[Superquadric] | None = None

    def to_dict(self) -> dict:
        carrying_dict = None
        if self.carrying is not None:
            carrying_dict = [p.to_dict() for p in self.carrying]
        return {"position": list(self.position.rounded(3)),
                "forward": list(self.forward.rounded(4)),
                "radius": self.radius,
                "carrying": carrying_dict}

    @classmethod
    def from_dict(cls, data: dict) -> "Player":
        carrying_raw = data.get("carrying")
        carrying = None
        if isinstance(carrying_raw, list):
            carrying = [Superquadric.from_dict(item) for item in carrying_raw]
        elif isinstance(carrying_raw, dict):
            carrying = [Superquadric.from_dict(carrying_raw)]
        return cls(position=Vec3.parse(data["position"]),
                   forward=Vec3.parse(data.get("forward", (0.0, 0.0, 1.0))).normalized(),
                   radius=float(data.get("radius", 0.6)),
                   carrying=carrying)


@dataclass
class Scene:
    """A cube of free space with superquadrics in it, plus the player."""

    bounds: float
    primitives: SuperquadricHandler
    player: Player
    meta: dict = field(default_factory=dict)

    @property
    def half(self) -> float:
        return self.bounds / 2

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

    def contacts(self, position: Vec3) -> list[Superquadric]:
        """Every primitive the player sphere is touching here, solid or not."""
        return self.primitives.touching(position, self.player.radius)

    def visible(self, sensor: Sensor) -> list[Superquadric]:
        """Primitives the live sensor detects: in range and field of view, and
        in the player's current room.

        `SuperquadricHandler.visible_from` genuinely raycasts for occlusion,
        which is too expensive to call every frame — this uses the cheap
        cone-only `visible_cone_from` instead and substitutes room membership
        for real occlusion. A room's walls always separate it from the next,
        so the agent still never sees through one; what it loses is occlusion
        *within* a room, e.g. an object tucked behind a piece of furniture is
        now visible as soon as it is in range and in view, where genuine
        raycasting would have kept it hidden a little longer.
        """
        visible = self.primitives.visible_cone_from(self.player.position, self.player.forward, sensor)
        room = self._room_at(self.player.position)
        if room is None:
            return [p for p in visible if p.kind != LOCK and not p.assembly.startswith("lock-")]
        (x0, y0, z0), (x1, y1, z1) = room["box"]
        # Matches the tolerance `_find_walls_without_doors` uses for the same
        # box-membership test: a wall or doorway shared between two rooms sits
        # exactly on their common boundary, and rounding at save time can put
        # it a hair outside a bare inclusive test.
        margin = 0.2
        return [p for p in visible if
                p.kind != LOCK and
                not p.assembly.startswith("lock-") and
                x0 - margin <= p.position.x <= x1 + margin and
                y0 - margin <= p.position.y <= y1 + margin and
                z0 - margin <= p.position.z <= z1 + margin]

    def describe_blocker(self, blocker: int | None) -> str:
        if blocker is None:
            return "nothing"
        if blocker == BOUNDS:
            return "the world boundary"
        prim = self.primitives.get(blocker)
        return f"primitive {prim.id} ({prim.assembly})"

    def _room_at(self, position: Vec3) -> dict | None:
        """The `meta.layout` room a point falls in, or None — a point-in-box
        test against a handful of rooms costs nothing, so nothing is stored.
        """
        p = position.as_tuple()
        for room in self.meta.get("layout", {}).get("rooms", []):
            (x0, y0, z0), (x1, y1, z1) = room["box"]
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1:
                return room
        return None

    def current_room_name(self, position: Vec3) -> str | None:
        room = self._room_at(position)
        return room["name"] if room else None

    def current_room_index(self, position: Vec3) -> int | None:
        """Which room's task (`meta.task[str(index)]`) applies at this point —
        each room on the path locks its own door."""
        room = self._room_at(position)
        return room["index"] if room else None

    # ---------------------------------------------------------- (de)serialisation

    def to_dict(self) -> dict:
        return {
            "bounds": self.bounds,
            "player": self.player.to_dict(),
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
