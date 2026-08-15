"""Directions and the batched `Move` the navigation agent returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .vector import Vector2D

#: Hard cap on how many steps one agent call may plan. Keeps a single bad plan
#: from burning the whole step budget and keeps the JSON small.
MAX_TRAJECTORY = 24


class Direction(Enum):
    UP = Vector2D(-1, 0)
    DOWN = Vector2D(1, 0)
    LEFT = Vector2D(0, -1)
    RIGHT = Vector2D(0, 1)

    @property
    def delta(self) -> Vector2D:
        return self.value

    @classmethod
    def parse(cls, token: str) -> "Direction":
        key = str(token).strip().upper()
        aliases = {
            "U": "UP",
            "D": "DOWN",
            "L": "LEFT",
            "R": "RIGHT",
            "NORTH": "UP",
            "SOUTH": "DOWN",
            "WEST": "LEFT",
            "EAST": "RIGHT",
        }
        key = aliases.get(key, key)
        try:
            return cls[key]
        except KeyError as exc:
            raise ValueError(f"unknown direction {token!r}") from exc


@dataclass
class Move:
    """A batched trajectory: a sequence of directional actions plus rationale."""

    actions: list[Direction]
    reasoning: str = ""
    raw: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def from_structured_output(cls, payload: dict) -> "Move":
        """Build a Move from the agent's JSON, tolerating minor shape drift."""
        actions_raw = payload.get("actions") or []
        if isinstance(actions_raw, str):  # e.g. "up,up,left"
            actions_raw = [t for t in actions_raw.replace(",", " ").split() if t]
        actions = [Direction.parse(token) for token in actions_raw][:MAX_TRAJECTORY]
        return cls(actions=actions, raw=payload)


#: JSON Schema pinned via `claude --json-schema` for the navigation agent.
MOVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "minItems": 1,
            "maxItems": MAX_TRAJECTORY,
        },
    },
    "required": ["actions"],
    "additionalProperties": False,
}
