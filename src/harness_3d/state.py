"""The observation handed to the navigation agent.

This is the harness's explicit departure from TASK.md's vision policy: the agent
never sees a screenshot. It sees the *parameters* of the superquadrics its sensor
has detected — the same validated numbers the renderer draws from — plus its own
pose and episode bookkeeping (3D_PLAN.md section 6).

What it never sees: meshes, collision triangles, primitives it has not observed,
and the generation-time reachability graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Vec3
from .scene import Scene
from .superquadrics import KEY, Sensor, Superquadric


def describe(prim: Superquadric, viewer: Vec3) -> dict:
    """One primitive as the agent sees it.

    The player-relative centre is computed here on purpose: without it every
    navigation turn burns reasoning on coordinate subtraction the harness can do
    exactly.
    """
    offset = prim.position - viewer
    return {
        "id": prim.id,
        "kind": prim.kind,
        "assembly": prim.assembly,
        "position": list(prim.position.rounded(1)),
        "relative": list(offset.rounded(1)),
        "distance": round(offset.length(), 1),
        "scale": [round(a, 1) for a in prim.scale],
        "rotation": [round(a, 0) for a in prim.rotation],
        "exponents": [round(e, 2) for e in prim.exponents],
    }


@dataclass
class State:
    """One observation tick."""

    player_position: Vec3
    forward: Vec3
    bounds: float
    visible: list[Superquadric]
    key_seen: bool
    calls: int = 0
    distance: float = 0.0
    collisions: int = 0
    last_outcome: str = "start"

    @classmethod
    def observe(cls, scene: Scene, sensor: Sensor, *, calls: int = 0, distance: float = 0.0,
                collisions: int = 0, last_outcome: str = "start") -> "State":
        visible = scene.visible(sensor)
        return cls(
            player_position=scene.player.position,
            forward=scene.player.forward,
            bounds=scene.bounds,
            visible=visible,
            key_seen=any(prim.kind == KEY for prim in visible),
            calls=calls,
            distance=distance,
            collisions=collisions,
            last_outcome=last_outcome,
        )

    def to_dict(self) -> dict:
        return {
            "position": list(self.player_position.rounded(2)),
            "forward": list(self.forward.rounded(3)),
            "bounds": self.bounds,
            "visible": [describe(p, self.player_position) for p in self.visible],
            "key_seen": self.key_seen,
            "calls": self.calls,
            "distance": round(self.distance, 1),
            "collisions": self.collisions,
            "last_outcome": self.last_outcome,
        }
