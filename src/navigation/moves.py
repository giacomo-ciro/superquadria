"""The action vocabulary: a batch of actions (move, pick, place).

The agent answers with explicit geometry:
- `move`: fly to a world-space destination.
- `pick`: pick up a movable object at a world-space target (within reach).
- `place`: place carried object at a target, or unlock if placed at a door/lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geometry import Vec3

MOVE = "move"
PICK = "pick"
PLACE = "place"
ACTIONS = (MOVE, PICK, PLACE)

#: Hard cap on how many actions one agent call may plan.
MAX_WAYPOINTS = 8


@dataclass
class Action:
    """One discrete command in an action batch."""

    type: str  # "move" | "pick" | "place"
    target: Vec3

    def __post_init__(self) -> None:
        if not isinstance(self.target, Vec3):
            self.target = Vec3.parse(self.target)

    def to_dict(self) -> dict:
        return {"type": self.type, "target": list(self.target.rounded(3))}

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        atype = str(data.get("type", MOVE)).lower().strip()
        if atype not in ACTIONS:
            atype = MOVE
        target = Vec3.parse(data.get("target", [0.0, 0.0, 0.0]))
        return cls(type=atype, target=target)


@dataclass
class Trajectory:
    """A batch of actions (move, pick, place), executed in order."""

    actions: list[Action]
    steer: bool = True
    reasoning: str = ""
    raw: dict = field(default_factory=dict)

    def __init__(
        self,
        actions_or_positions=(),
        *,
        steer: bool = True,
        reasoning: str = "",
        raw: dict | None = None,
    ):
        self.steer = steer
        self.reasoning = reasoning
        self.raw = raw if raw is not None else {}
        self.actions = []
        for item in actions_or_positions:
            if isinstance(item, Action):
                self.actions.append(item)
            elif isinstance(item, Vec3):
                self.actions.append(Action(type=MOVE, target=item))
            elif isinstance(item, (list, tuple)):
                self.actions.append(Action(type=MOVE, target=Vec3.parse(item)))

    @property
    def positions(self) -> list[Vec3]:
        """Convenience property for movement destinations."""
        return [a.target for a in self.actions if a.type == MOVE]

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def from_structured_output(
        cls, payload: dict, *, limit: int = MAX_WAYPOINTS
    ) -> "Trajectory":
        """Parse the agent's JSON, tolerating minor shape drift."""
        actions: list[Action] = []
        raw_actions = payload.get("actions")
        if isinstance(raw_actions, list):
            for item in raw_actions[:limit]:
                if isinstance(item, dict) and "target" in item:
                    try:
                        actions.append(Action.from_dict(item))
                    except (ValueError, TypeError):
                        break
            return cls(
                actions, reasoning=str(payload.get("reasoning", ""))[:400], raw=payload
            )

        # Fallback to legacy "positions" if returned
        raw_positions = payload.get("positions") or []
        if isinstance(raw_positions, dict):
            raw_positions = [raw_positions[k] for k in sorted(raw_positions)]
        for item in raw_positions[:limit]:
            if isinstance(item, dict):
                item = [item.get("x"), item.get("y"), item.get("z")]
            try:
                actions.append(Action(type=MOVE, target=Vec3.parse(item)))
            except (ValueError, TypeError):
                break
        return cls(
            actions, reasoning=str(payload.get("reasoning", ""))[:400], raw=payload
        )


#: JSON Schema pinned for the navigation agent.
TRAJECTORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": "Batch of actions executed in order.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["move", "pick", "place"],
                        "description": (
                            "'move' flies to target; 'pick' picks up object at "
                            "target (within reach); 'place' places carried object "
                            "at target or unlocks if at door/lock."
                        ),
                    },
                    "target": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            "Absolute world-space destination or object "
                            "coordinate [x, y, z]."
                        ),
                    },
                },
                "required": ["type", "target"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": MAX_WAYPOINTS,
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence on what this batch is for.",
        },
    },
    "required": ["actions", "reasoning"],
    "additionalProperties": False,
}
