"""What the player has actually observed, remembered across turns.

The agent is stateless between calls, so remembering the primitives it has
already detected is bookkeeping the harness can do for it. The rule from
3D_PLAN.md section 6 is absolute: this only ever holds primitives a real sensor
query returned. It never copies the handler, and it never touches the
generation-time reachability graph.

It also keeps a coarse record of the poses already sampled, so the agent can tell
exploring apart from looking at the same volume again from a slightly different
spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Vec3
from .state import State, describe
from .superquadrics import KEY, Superquadric


@dataclass
class SpatialMemory:
    #: Size of the bucket a sensor pose is recorded in — position cell plus the
    #: coarse look octant. Fine enough to distinguish real vantage points,
    #: coarse enough that drifting one increment is not "somewhere new".
    pose_cell: float = 4.0

    primitives: dict[int, Superquadric] = field(default_factory=dict)
    key_id: int | None = field(default=None, init=False)
    poses: set[tuple] = field(default_factory=set)
    #: Bumped whenever something new is learned, so the renderer can cache.
    revision: int = field(default=0, init=False)

    # ------------------------------------------------------------------ update

    def integrate(self, state: State) -> int:
        """Fold one observation in. Returns how many primitives were new."""
        revealed = 0
        for prim in state.visible:
            if prim.id not in self.primitives:
                revealed += 1
            self.primitives[prim.id] = prim
            if prim.kind == KEY:
                self.key_id = prim.id
        self.poses.add(self._pose_key(state.player_position, state.forward))
        if revealed:
            self.revision += 1
        return revealed

    def _pose_key(self, position: Vec3, forward: Vec3) -> tuple:
        f = forward.normalized()
        return (int(position.x // self.pose_cell), int(position.y // self.pose_cell),
                int(position.z // self.pose_cell),
                round(f.x), round(f.y), round(f.z))

    # ------------------------------------------------------------------ access

    @property
    def key(self) -> Superquadric | None:
        return self.primitives.get(self.key_id) if self.key_id is not None else None

    def remembered(self, exclude: set[int] = frozenset()) -> list[Superquadric]:
        return [p for pid, p in self.primitives.items() if pid not in exclude]

    def assemblies(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for prim in self.primitives.values():
            counts[prim.assembly] = counts.get(prim.assembly, 0) + 1
        return counts

    # --------------------------------------------------------------- rendering

    def table(self, viewer: Vec3, primitives: list[Superquadric], *, limit: int = 40) -> str:
        """The primitive parameters as a fixed-width table for the prompt.

        Nearest first and capped: a distant primitive the agent already flew past
        is worth less prompt than the one it is about to hit.
        """
        if not primitives:
            return "  (none)"
        rows = [("  id  assembly          centre            relative          dist  "
                 "size            exps")]
        ordered = sorted(primitives, key=lambda p: (p.position - viewer).length())
        for prim in ordered[:limit]:
            item = describe(prim, viewer)
            centre = "(%s)" % ", ".join(f"{v:g}" for v in item["position"])
            relative = "(%s)" % ", ".join(f"{v:+g}" for v in item["relative"])
            size = "(%s)" % ", ".join(f"{v:g}" for v in item["scale"])
            label = prim.assembly if prim.kind != KEY else "THE KEY"
            rows.append(f"  {prim.id:>2}  {label[:16]:<16}  {centre:<17} {relative:<17} "
                        f"{item['distance']:>5}  {size:<15} ({item['exponents'][0]:g}, "
                        f"{item['exponents'][1]:g})")
        if len(ordered) > limit:
            rows.append(f"  ... and {len(ordered) - limit} more further away")
        return "\n".join(rows)
