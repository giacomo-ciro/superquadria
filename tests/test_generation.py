"""Generation with a scripted agent: the harness must survive whatever comes back."""

import random

from maze_harness.agents import ClaudeCallError, ScriptedAgent
from maze_harness.generation import BAND_SCHEMA, BRIEF_SCHEMA, MazeGenerator
from maze_harness.maze_utils import bfs_path, label_regions
from maze_harness.entities import Tile

BRIEF = {
    "theme": "Test Warrens",
    "description": "corridors",
    "structural_rules": ["a", "b", "c"],
    "target_open_ratio": 0.4,
}


def scripted(band_rows):
    """An agent that answers the brief call, then every band call."""
    def respond(prompt, schema):
        return dict(BRIEF) if schema is BRIEF_SCHEMA else {"rows": band_rows()}
    return ScriptedAgent(respond)


def generator(agent, **kwargs):
    return MazeGenerator(agent=agent, width=20, height=20, band_height=5,
                         workers=2, seed=1, log=lambda *_: None,
                         min_key_distance=10, **kwargs)


def checkerboard_rows(width=20, height=5):
    rng = random.Random(0)
    return ["".join("#" if rng.random() < 0.4 else "." for _ in range(width))
            for _ in range(height)]


def test_generated_maze_is_always_solvable():
    scene = generator(scripted(checkerboard_rows)).generate()
    assert label_regions(scene.terrain)[1] == 1
    assert bfs_path(scene.terrain, scene.player.position, scene.key.position) is not None
    assert scene.meta["theme"] == "Test Warrens"
    assert all(b["source"] == "agent" for b in scene.meta["bands"])


def test_seams_are_enforced_even_if_the_model_ignores_them():
    # A model that returns nothing but wall would seal every band from its neighbours.
    scene = generator(scripted(lambda: ["#" * 20] * 5)).generate()
    # Repair still leaves a connected, walkable world with both entities on floor.
    assert label_regions(scene.terrain)[1] == 1
    assert scene.terrain[scene.player.position.row][scene.player.position.col] != Tile.WALL
    assert bfs_path(scene.terrain, scene.player.position, scene.key.position) is not None


def test_a_failing_band_falls_back_to_procedural():
    calls = {"n": 0}

    def respond(prompt, schema):
        if schema is BRIEF_SCHEMA:
            return dict(BRIEF)
        calls["n"] += 1
        if calls["n"] == 2:
            raise ClaudeCallError("simulated API failure")
        return {"rows": checkerboard_rows()}

    scene = generator(ScriptedAgent(respond)).generate()
    sources = [b["source"] for b in scene.meta["bands"]]
    assert sum(s.startswith("fallback") for s in sources) == 1
    assert label_regions(scene.terrain)[1] == 1


def test_malformed_rows_are_repaired_not_fatal():
    # Wrong row count, wrong widths, stray characters.
    bad = ["....", "#" * 60, "??abc", "..."]
    scene = generator(scripted(lambda: list(bad))).generate()
    assert len(scene.terrain) == 20
    assert all(len(row) == 20 for row in scene.terrain)
    assert label_regions(scene.terrain)[1] == 1


def test_band_prompt_states_the_seam_contract():
    agent = scripted(checkerboard_rows)
    generator(agent).generate()
    band_prompts = [p for p, schema in agent.calls if schema is BAND_SCHEMA]
    assert len(band_prompts) == 4
    assert "global rows 0 to 4" in band_prompts[0]
    assert "must be '.' at columns" in band_prompts[0]
    assert "Test Warrens" in band_prompts[0]


def test_offline_generation_needs_no_agent():
    scene = MazeGenerator(agent=None, width=41, height=41, band_height=41,
                          seed=2, min_key_distance=10).generate_offline()
    assert scene.meta["generator"] == "procedural"
    assert bfs_path(scene.terrain, scene.player.position, scene.key.position) is not None
