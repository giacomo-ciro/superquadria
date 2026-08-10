"""Agent-driven maze generation, in batches (PLAN.md section 2).

Two stages, both going through the same `Agent.run(prompt, schema)` interface:

1. **Design brief** — one call in which the agent invents the maze concept. For
   the MVP nobody hands it a text command; it decides for itself. (`--brief` lets
   you seed that decision, which is the hook the "user-authored text commands"
   future-work item plugs into.)
2. **Bands** — the 100x100 grid is cut into horizontal bands of `band_height`
   rows, one call per band, all sharing the brief so they read as one maze. Bands
   are independent, so they run concurrently.

Because bands are drawn independently, the harness owns stitching and does not
trust the model for it: it pins a set of *connector columns* on every band
boundary (told to the model, then enforced in code), and finishes with
`connect_regions`, which carves the minimum corridors needed to make every open
cell reachable. The result is a maze that is always solvable, whatever the model
returned.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .agents.base import Agent
from .agents.claude_cli import ClaudeCallError
from .entities import Entity, Tile
from .maze_utils import (
    connect_regions,
    ensure_border_walls,
    open_ratio,
    parse_rows,
    procedural_maze,
    random_open_cell,
)
from .scene import Scene

BRIEF_SCHEMA: dict = {
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
    },
    "required": ["theme", "description", "structural_rules", "target_open_ratio"],
    "additionalProperties": False,
}

BAND_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "description": "The band's rows, top to bottom. Each string uses only '#' (wall) and '.' (floor).",
            "items": {"type": "string"},
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}

#: How many openings to pin on each band boundary.
CONNECTORS_PER_SEAM = 6


@dataclass
class Band:
    """One generated horizontal slice, plus how it was produced."""

    index: int
    top_row: int
    rows: list[bytearray]
    source: str  # "agent" or "fallback:<reason>"


@dataclass
class MazeGenerator:
    #: Only `generate_offline` may be called with agent=None.
    agent: Agent | None
    width: int = 100
    height: int = 100
    band_height: int = 10
    workers: int = 4
    seed: int | None = None
    log: Callable[[str], None] = print
    #: Minimum Manhattan distance between spawn and key, so the episode is a real
    #: search rather than a stroll. Defaults to a third of the map perimeter;
    #: `random_open_cell` relaxes it if no cell is that far away.
    min_key_distance: int | None = None
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.height % self.band_height:
            raise ValueError("height must be a multiple of band_height")
        if self.min_key_distance is None:
            self.min_key_distance = (self.width + self.height) // 3

    # --------------------------------------------------------------- top level

    def generate(self, brief_hint: str | None = None) -> Scene:
        brief = self._design_brief(brief_hint)
        self.log(f"[gen] theme: {brief['theme']} — {brief['description']}")

        seams = self._connector_columns()
        n_bands = self.height // self.band_height
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            bands = list(pool.map(
                lambda i: self._generate_band(i, n_bands, brief, seams),
                range(n_bands),
            ))

        terrain: list[bytearray] = []
        for band in bands:
            terrain.extend(band.rows)

        ensure_border_walls(terrain)
        self._apply_seams(terrain, seams)
        carved = connect_regions(terrain)
        self.log(f"[gen] stitched {len(seams)} seams, carved {carved} repair corridors, "
                 f"open ratio {open_ratio(terrain):.2f}")

        scene = self._populate(terrain)
        scene.meta = {
            "generator": "agent-bands",
            "theme": brief["theme"],
            "description": brief["description"],
            "structural_rules": brief["structural_rules"],
            "seed": self.seed,
            "bands": [{"index": b.index, "source": b.source} for b in bands],
            "repair_corridors": carved,
            "open_ratio": round(open_ratio(terrain), 4),
        }
        return scene

    def generate_offline(self) -> Scene:
        """Procedural maze with no agent calls — for smoke tests and demos."""
        terrain = procedural_maze(self.width, self.height, self._rng)
        scene = self._populate(terrain)
        scene.meta = {"generator": "procedural", "seed": self.seed,
                      "open_ratio": round(open_ratio(terrain), 4)}
        return scene

    # ------------------------------------------------------------------ stages

    def _design_brief(self, brief_hint: str | None) -> dict:
        if brief_hint:
            intent = (f"The operator asked for this environment:\n\"{brief_hint}\"\n"
                      "Turn that request into a concrete, drawable specification.")
        else:
            intent = ("Nobody has specified what this maze should be. Decide for yourself, "
                      "and pick something with distinctive spatial structure rather than "
                      "uniform noise.")
        prompt = f"""You are designing a {self.height}x{self.width} 2D grid maze for a game.

{intent}

The maze will be drawn one horizontal band at a time by separate workers who see
only your brief, so the rules must be stated in absolute grid terms (row and
column numbers, repeating periods, sizes) rather than "continue from above".

A player with a 10x10 field of view must search this maze for a hidden key, so it
needs long sightlines and landmarks, connected floor, and no vast empty halls.
"""
        brief = self.agent.run(prompt, BRIEF_SCHEMA)
        brief.setdefault("target_open_ratio", 0.4)
        return brief

    def _generate_band(self, index: int, n_bands: int, brief: dict,
                       seams: dict[int, list[int]]) -> Band:
        top = index * self.band_height
        bottom = top + self.band_height - 1
        rules = "\n".join(f"- {r}" for r in brief.get("structural_rules", []))

        edges = []
        if index > 0:
            cols = ", ".join(str(c) for c in seams[index - 1])
            edges.append(f"- Your FIRST row (global row {top}) must be '.' at columns: {cols}.")
        if index < n_bands - 1:
            cols = ", ".join(str(c) for c in seams[index])
            edges.append(f"- Your LAST row (global row {bottom}) must be '.' at columns: {cols}.")

        prompt = f"""Draw one horizontal band of a {self.height}x{self.width} grid maze.

THEME: {brief['theme']}
{brief['description']}

STRUCTURAL RULES (apply to the whole maze; your band is one slice of it):
{rules}

YOUR BAND: global rows {top} to {bottom} inclusive — band {index + 1} of {n_bands}.

OUTPUT FORMAT:
- Exactly {self.band_height} strings, in top-to-bottom order.
- Each string is exactly {self.width} characters, one per column (column 0 first).
- '#' is wall, '.' is walkable floor. No other characters, no spaces, no row numbers.

CONSTRAINTS:
- Column 0 and column {self.width - 1} must be '#' in every row (outer wall).
- Roughly {brief['target_open_ratio']:.0%} of your cells should be '.'.
- Floor must form connected corridors and rooms within your band — no sealed pockets.
{chr(10).join(edges)}

Work out the band's structure from the rules using the absolute row and column
numbers above, so your slice lines up with the bands drawn by the other workers.
"""
        try:
            payload = self.agent.run(prompt, BAND_SCHEMA)
            rows = parse_rows(payload.get("rows", []), width=self.width, height=self.band_height)
            ratio = open_ratio(rows)
            if ratio < 0.05:
                raise ValueError(f"band {index} came back {1 - ratio:.0%} wall")
            self.log(f"[gen] band {index + 1}/{n_bands} drawn (open {ratio:.2f})")
            return Band(index, top, rows, "agent")
        except (ClaudeCallError, ValueError, KeyError, TypeError) as exc:
            # One flaky band must not sink a 100x100 generation run.
            self.log(f"[gen] band {index + 1}/{n_bands} fell back to procedural: {exc}")
            rows = self._fallback_band(index)
            return Band(index, top, rows, f"fallback:{type(exc).__name__}")

    def _fallback_band(self, index: int) -> list[bytearray]:
        # Own RNG per band: `_generate_band` runs on the thread pool, and sharing
        # `self._rng` across threads would make fallbacks non-reproducible.
        rng = random.Random((self.seed or 0) * 1000 + index)
        height = self.band_height if self.band_height % 2 else self.band_height + 1
        rows = procedural_maze(self.width, height, rng)
        return rows[: self.band_height]

    # ---------------------------------------------------------------- stitching

    def _connector_columns(self) -> dict[int, list[int]]:
        """Pin the openings on each band seam: {band_index: [columns]}.

        Chosen by the harness (not the model) so parallel bands agree, and so the
        contract can be enforced in code afterwards.
        """
        interior = range(2, self.width - 2)
        seams: dict[int, list[int]] = {}
        for seam in range(self.height // self.band_height - 1):
            seams[seam] = sorted(self._rng.sample(list(interior), CONNECTORS_PER_SEAM))
        return seams

    def _apply_seams(self, terrain: list[bytearray], seams: dict[int, list[int]]) -> None:
        """Force the pinned seam openings, whatever the model actually drew."""
        for seam, columns in seams.items():
            lower = (seam + 1) * self.band_height  # first row of the next band
            for col in columns:
                terrain[lower - 1][col] = Tile.EMPTY
                terrain[lower][col] = Tile.EMPTY

    # --------------------------------------------------------------- placement

    def _populate(self, terrain: list[bytearray]) -> Scene:
        spawn = random_open_cell(terrain, self._rng)
        key = random_open_cell(terrain, self._rng, exclude=[spawn],
                               anchor=spawn, min_distance=self.min_key_distance)
        return Scene(
            width=self.width,
            height=self.height,
            terrain=terrain,
            entities=[Entity(Tile.PLAYER, spawn), Entity(Tile.KEY, key)],
        )
