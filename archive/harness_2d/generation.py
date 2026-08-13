"""Agent-driven maze generation (PLAN.md section 2).

One call, through the `Agent.run(prompt, schema)` interface: the agent invents
the maze concept (theme, description, structural rules) and draws the entire
square grid in the same response. An optional brief can instruct that concept;
without one, the agent decides for itself.

Whatever the model returns is used as-is: no connectivity repair, no stitching.
Undersized or malformed output falls back to a fully procedural maze.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from harness_common.agents.base import Agent, AgentError
from harness_common.pipeline_log import PipelineLogger

from .entities import Entity, Tile
from .maze_utils import (
    ensure_border_walls,
    open_ratio,
    parse_rows,
    procedural_maze,
    random_open_cell,
)
from .scene import VIEW_SIZE, Scene

MAZE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "description": "Short name for this maze, e.g. 'Collapsed Archive'."},
        "description": {"type": "string", "description": "2-3 sentences describing the spatial character of the maze."},
        "structural_rules": {
            "type": "array",
            "description": "3-6 concrete, drawable rules, e.g. 'rooms of 6x6 connected by single-cell doorways'.",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "target_open_ratio": {
            "type": "number",
            "description": "Fraction of cells that should be walkable, between 0.25 and 0.6.",
        },
        "rows": {
            "type": "array",
            "description": "The maze's rows, top to bottom. Each string uses only '#' (wall) and '.' (floor).",
            "items": {"type": "string"},
        },
    },
    "required": ["theme", "description", "structural_rules", "target_open_ratio", "rows"],
    "additionalProperties": False,
}


@dataclass
class MazeGenerator:
    #: Only `generate_offline` may be called with agent=None.
    agent: Agent | None
    side_length: int = 100
    view_size: int = VIEW_SIZE
    log: PipelineLogger = field(default_factory=lambda: PipelineLogger("gen"))
    #: Minimum Manhattan distance between spawn and key, so the episode is a real
    #: search rather than a stroll. Defaults to a third of the square's perimeter;
    #: `random_open_cell` relaxes it if no cell is that far away.
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random()
        self.min_key_distance = 2 * self.side_length // 3

    # --------------------------------------------------------------- top level

    def generate(self, brief_hint: str | None = None) -> Scene:
        self.log.start("generate", f"requesting {self.side_length}x{self.side_length} design+maze")
        brief, terrain, source = self._generate_maze(brief_hint)
        self.log.info("generate", f"theme: {brief['theme']} — {brief['description']}")

        ensure_border_walls(terrain)
        self.log.end("generate", f"maze drawn ({source}), open ratio {open_ratio(terrain):.2f}")

        scene = self._populate(terrain)
        scene.meta = {
            "generator": "agent-single-shot",
            "theme": brief["theme"],
            "description": brief["description"],
            "structural_rules": brief["structural_rules"],
            "source": source,
            "open_ratio": round(open_ratio(terrain), 4),
        }
        return scene

    def generate_offline(self) -> Scene:
        """Procedural maze with no agent calls — for smoke tests and demos."""
        terrain = procedural_maze(self.side_length, self._rng)
        scene = self._populate(terrain)
        scene.meta = {"generator": "procedural",
                      "open_ratio": round(open_ratio(terrain), 4)}
        return scene

    # ------------------------------------------------------------------ stages

    def _generate_maze(self, brief_hint: str | None) -> tuple[dict, list[bytearray], str]:
        if brief_hint:
            intent = (f"The operator asked for this environment:\n\"{brief_hint}\"\n"
                      "Turn that request into a concrete, drawable specification.")
        else:
            intent = ("Nobody has specified what this maze should be. Decide for yourself, "
                      "and pick something with distinctive spatial structure rather than "
                      "uniform noise.")
        prompt = f"""You are designing and drawing a {self.side_length}x{self.side_length} 2D grid maze for a game.

{intent}

A player with a {self.view_size}x{self.view_size} field of view must search this maze for a hidden key, so it
needs long sightlines and landmarks, connected floor, and no vast empty halls.

Along with the design (theme, description, structural rules, target open ratio),
draw the maze itself in the same response:

OUTPUT FORMAT for `rows`:
- Exactly {self.side_length} strings, in top-to-bottom order.
- Each string is exactly {self.side_length} characters, one per column (column 0 first).
- '#' is wall, '.' is walkable floor. No other characters, no spaces, no row numbers.

CONSTRAINTS:
- Row 0, row {self.side_length - 1}, column 0, and column {self.side_length - 1} must be '#' (outer wall).
- Roughly `target_open_ratio` of cells should be '.'.
- All floor must form one connected network reachable from any other floor cell —
  no sealed pockets.
- The drawn maze must actually follow the `structural_rules` you specify.
"""
        try:
            payload = self.agent.run(prompt, MAZE_SCHEMA)
            rows = parse_rows(payload["rows"], width=self.side_length, height=self.side_length)
            ratio = open_ratio(rows)
            if ratio < 0.05:
                raise ValueError(f"maze came back {1 - ratio:.0%} wall")
            return payload, rows, "agent"
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            # A bad single call must not sink the whole generation run.
            self.log.info("generate", f"maze generation fell back to procedural: {exc}")
            brief = {
                "theme": "Procedural fallback",
                "description": brief_hint or "Randomised depth-first maze (agent call failed).",
                "structural_rules": [],
                "target_open_ratio": 0.4,
            }
            terrain = procedural_maze(self.side_length, self._rng)
            return brief, terrain, f"fallback:{type(exc).__name__}"

    # --------------------------------------------------------------- placement

    def _populate(self, terrain: list[bytearray]) -> Scene:
        spawn = random_open_cell(terrain, self._rng)
        key = random_open_cell(terrain, self._rng, exclude=[spawn],
                               anchor=spawn, min_distance=self.min_key_distance)
        return Scene(
            width=self.side_length,
            height=self.side_length,
            terrain=terrain,
            entities=[Entity(Tile.PLAYER, spawn), Entity(Tile.KEY, key)],
        )
