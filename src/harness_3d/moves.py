"""The action vocabulary: a batch of absolute world-space target positions.

Target positions rather than controller tokens (3D_PLAN.md section 2). The agent
is handed explicit geometry, so it should be allowed to answer geometrically;
restricting it to six axis steps would be a 3D grid with a fancy renderer.

The requested movement vector *is* `target - current_position`, so separate
direction and distance fields would only be a second chance to contradict it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Vec3

#: Hard cap on how many target positions one agent call may plan. Together with
#: the per-segment distance cap, this is what keeps one bad response from
#: consuming the whole episode.
MAX_WAYPOINTS = 8


@dataclass
class Trajectory:
    """A batch of absolute destinations, flown in order."""

    positions: list[Vec3]
    #: Agent trajectories point the camera at each segment's target before
    #: flying it. Manual flight does not: the mouse owns the look direction and
    #: may change it mid-segment.
    steer: bool = True
    reasoning: str = ""
    raw: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_structured_output(cls, payload: dict, *, limit: int = MAX_WAYPOINTS) -> "Trajectory":
        """Parse the agent's JSON, tolerating minor shape drift.

        Truncates at the first unusable entry rather than skipping it: the rest
        of a plan is written on the assumption that the dropped waypoint was
        reached, so silently splicing it out would fly a route nobody chose.
        """
        raw = payload.get("positions") or []
        if isinstance(raw, dict):  # e.g. {"0": [...], "1": [...]}
            raw = [raw[k] for k in sorted(raw)]
        positions: list[Vec3] = []
        for item in raw[:limit]:
            if isinstance(item, dict):
                item = [item.get("x"), item.get("y"), item.get("z")]
            try:
                positions.append(Vec3.parse(item))
            except (ValueError, TypeError):
                break
        return cls(positions=positions, reasoning=str(payload.get("reasoning", ""))[:400], raw=payload)


#: JSON Schema pinned for the navigation agent.
TRAJECTORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "positions": {
            "type": "array",
            "description": "Absolute world-space destinations, flown in order.",
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "minItems": 1,
            "maxItems": MAX_WAYPOINTS,
        },
        "reasoning": {"type": "string", "description": "One sentence on what this batch is for."},
    },
    "required": ["positions", "reasoning"],
    "additionalProperties": False,
}
