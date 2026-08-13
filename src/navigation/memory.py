"""What the player has actually observed, remembered across turns.

The agent is stateless between calls, so remembering the primitives it has
already detected is bookkeeping the harness can do for it. The rule is
absolute: this only ever holds primitives a real sensor query returned. It
never copies the handler, and it never touches the generation-time
reachability/flood graphs.

It also keeps a coarse record of the poses already sampled, so the agent can tell
exploring apart from looking at the same volume again from a slightly different
spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geometry import Vec3
from geometry.superquadrics import LOCK, Superquadric
from .state import State, describe


@dataclass
class SpatialMemory:
    #: Size of the bucket a sensor pose is recorded in — position cell plus the
    #: coarse look octant. Fine enough to distinguish real vantage points,
    #: coarse enough that drifting one increment is not "somewhere new".
    pose_cell: float = 4.0

    current_room: str | None = None
    #: All primitives observed up to now across the entire episode (used by the renderer).
    primitives: dict[int, Superquadric] = field(default_factory=dict)
    #: All poses sampled across the whole episode (used by the renderer/stats).
    poses: set[tuple] = field(default_factory=set)
    #: Primitives observed in the current room only (used by the navigation prompt).
    room_primitives: dict[int, Superquadric] = field(default_factory=dict)
    #: Poses sampled in the current room only.
    room_poses: set[tuple] = field(default_factory=set)
    #: Bumped whenever something new is learned, so the renderer can cache.
    revision: int = field(default=0, init=False)

    # ------------------------------------------------------------------ update

    def clear(self) -> None:
        """Forget all observed primitives and poses (both global and per-room)."""
        self.primitives.clear()
        self.poses.clear()
        self.room_primitives.clear()
        self.room_poses.clear()
        self.revision += 1

    def forget_assembly(self, assembly: str) -> None:
        """Remove all primitives belonging to an assembly from memory (e.g. when picked up or consumed)."""
        to_del_room = [pid for pid, p in self.room_primitives.items() if p.assembly == assembly]
        for pid in to_del_room:
            del self.room_primitives[pid]
        to_del_global = [pid for pid, p in self.primitives.items() if p.assembly == assembly]
        for pid in to_del_global:
            del self.primitives[pid]
        if to_del_room or to_del_global:
            self.revision += 1

    def forget_ids(self, ids: set[int] | list[int]) -> None:
        """Remove specific primitive IDs from memory."""
        changed = False
        for pid in ids:
            if pid in self.room_primitives:
                del self.room_primitives[pid]
                changed = True
            if pid in self.primitives:
                del self.primitives[pid]
                changed = True
        if changed:
            self.revision += 1

    def integrate(self, state: State) -> int:
        """Fold one observation in. Returns how many primitives were newly discovered.

        The global `primitives` and `poses` accumulate everything observed up to now
        for the renderer. The `room_primitives` and `room_poses` hold only what has
        been observed in the current room, resetting whenever the player enters a new room.
        """
        if state.room is not None:
            if self.current_room is None:
                self.current_room = state.room
            elif state.room != self.current_room:
                self.room_primitives.clear()
                self.room_poses.clear()
                self.current_room = state.room

        revealed = 0
        for prim in state.visible:
            if prim.kind == LOCK or prim.assembly.startswith("lock-"):
                continue
            if prim.id not in self.primitives:
                revealed += 1
            self.primitives[prim.id] = prim
            self.room_primitives[prim.id] = prim

        if state.carrying and "assembly" in state.carrying:
            self.forget_assembly(state.carrying["assembly"])

        pose_key = self._pose_key(state.player_position, state.forward)
        self.poses.add(pose_key)
        self.room_poses.add(pose_key)

        if revealed:
            self.revision += 1
        return revealed

    def _pose_key(self, position: Vec3, forward: Vec3) -> tuple:
        f = forward.normalized()
        return (int(position.x // self.pose_cell), int(position.y // self.pose_cell),
                int(position.z // self.pose_cell),
                round(f.x), round(f.y), round(f.z))

    # ------------------------------------------------------------------ access

    def remembered(self, exclude: set[int] = frozenset()) -> list[Superquadric]:
        """Primitives observed in the current room, excluding those currently in view."""
        return [p for pid, p in self.room_primitives.items()
                if pid not in exclude and p.kind != LOCK and not p.assembly.startswith("lock-")]

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
        valid = [p for p in primitives if p.kind != LOCK and not p.assembly.startswith("lock-")]
        if not valid:
            return "  (none)"
        rows = [("  id  assembly    centre               dist  size             exps")]
        ordered = sorted(valid, key=lambda p: (p.position - viewer).length())
        for prim in ordered[:limit]:
            item = describe(prim, viewer)
            centre = "(%s)" % ", ".join(f"{v:g}" for v in item["position"])
            size = "(%s)" % ", ".join(f"{v:g}" for v in item["scale"])
            assembly_str = " ".join(prim.assembly.split())[:10]
            rows.append(f"  {prim.id:>2}  {assembly_str:<10}  "
                        f"{centre:<19} {item['distance']:>5}  {size:<15} "
                        f"({item['exponents'][0]:g}, {item['exponents'][1]:g})")
        if len(ordered) > limit:
            rows.append(f"  ... and {len(ordered) - limit} more further away")
        return "\n".join(rows)
