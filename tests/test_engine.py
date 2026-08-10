from maze_harness.agents import ScriptedAgent
from maze_harness.engine import Episode
from maze_harness.entities import Entity, Tile
from maze_harness.maze_utils import parse_rows, procedural_maze, random_open_cell
from maze_harness.memory import FogMemory
from maze_harness.moves import Direction, Move
from maze_harness.navigation import ClaudeNavigator
from maze_harness.policies import FrontierPolicy, Policy
from maze_harness.scene import Scene
from maze_harness.state import State
from maze_harness.vector import Vector2D
import random

CORRIDOR = [
    "#######",
    "#.....#",
    "#####.#",
    "#.....#",
    "#######",
]


def corridor_scene(player=(1, 1), key=(3, 1)):
    terrain = parse_rows(CORRIDOR, width=7, height=5)
    return Scene(width=7, height=5, terrain=terrain,
                 entities=[Entity(Tile.PLAYER, Vector2D(*player)),
                           Entity(Tile.KEY, Vector2D(*key))])


class FixedPolicy(Policy):
    """Replays a canned list of trajectories."""

    name = "fixed"

    def __init__(self, plans):
        self.plans = list(plans)

    def act(self, state):
        self._ensure_memory(state)
        return Move(self.plans.pop(0) if self.plans else [Direction.UP], reasoning="canned")


def test_trajectory_halts_at_the_first_wall_and_discards_the_rest():
    scene = corridor_scene()
    plan = [Direction.RIGHT, Direction.RIGHT, Direction.UP, Direction.RIGHT, Direction.RIGHT]
    episode = Episode(scene, FixedPolicy([plan]), max_steps=50, max_calls=1)
    result = episode.run()
    # Two rights land at (1,3); UP is a wall, so the last two moves never happen.
    assert scene.player.position == Vector2D(1, 3)
    assert result.steps == 2
    assert result.collisions == 1
    assert "hit a wall" in result.trajectory[0]["reasoning"] or result.trajectory[0]["blocked"]


def test_episode_stops_the_moment_the_key_is_reached():
    scene = corridor_scene(player=(3, 3), key=(3, 1))
    plan = [Direction.LEFT] * 6           # would overshoot into the wall at (3,0)
    result = Episode(scene, FixedPolicy([plan]), max_steps=50, max_calls=1).run()
    assert result.solved and result.reason == "reached the key"
    assert result.steps == 2
    assert result.collisions == 0


def test_budgets_terminate_the_loop():
    scene = corridor_scene(key=(1, 5))
    spinner = FixedPolicy([])             # always plans UP into a wall
    result = Episode(scene, spinner, max_steps=50, max_calls=4).run()
    assert not result.solved
    assert result.reason == "agent-call budget exhausted"
    assert result.calls == 4


def test_frontier_policy_solves_a_procedural_maze():
    rng = random.Random(11)
    terrain = procedural_maze(41, 41, rng)
    spawn = random_open_cell(terrain, rng)
    key = random_open_cell(terrain, rng, exclude=[spawn], anchor=spawn, min_distance=20)
    scene = Scene(width=41, height=41, terrain=terrain,
                  entities=[Entity(Tile.PLAYER, spawn), Entity(Tile.KEY, key)])
    result = Episode(scene, FrontierPolicy(seed=0), max_steps=6000, max_calls=800).run()
    assert result.solved, result.summary()


def test_navigator_falls_back_when_the_agent_misbehaves():
    scene = corridor_scene()
    broken = ScriptedAgent(lambda prompt, schema: {"reasoning": "", "actions": []})
    navigator = ClaudeNavigator(broken, log=lambda *_: None)
    result = Episode(scene, navigator, max_steps=200, max_calls=40).run()
    assert navigator.failures > 0          # every call was rejected...
    assert result.solved                   # ...and the offline fallback still finished


def test_navigator_prompt_contains_the_observation_and_the_map():
    scene = corridor_scene()
    seen = {}

    def respond(prompt, schema):
        seen["prompt"] = prompt
        return {"reasoning": "go", "actions": ["right"]}

    navigator = ClaudeNavigator(ScriptedAgent(respond))
    move = navigator.act(State.observe(scene))
    assert move.actions == [Direction.RIGHT]
    prompt = seen["prompt"]
    assert "CURRENT VIEW" in prompt and "REMEMBERED MAP" in prompt
    assert "halts execution" in prompt
    # The 10x10 window covers this whole tiny world, so the key is already in memory.
    assert "you have seen it" in prompt

    # On a world larger than the window, the key stays hidden and the prompt says so.
    rng = random.Random(5)
    big = Scene(width=41, height=41, terrain=procedural_maze(41, 41, rng),
                entities=[Entity(Tile.PLAYER, Vector2D(1, 1)), Entity(Tile.KEY, Vector2D(39, 39))])
    ClaudeNavigator(ScriptedAgent(respond)).act(State.observe(big))
    assert "has not been seen yet" in seen["prompt"]


def test_memory_only_ever_holds_observed_cells():
    scene = corridor_scene(key=(3, 5))
    memory = FogMemory(height=5, width=7)
    memory.integrate(State.observe(scene))
    # The 10x10 window covers this whole 5x7 world, so everything is known...
    assert memory.explored_fraction() == 1.0
    assert memory.key_position == Vector2D(3, 5)

    # ...whereas a single cell of a big world leaves the rest unknown.
    sparse = FogMemory(height=100, width=100)
    assert sparse.explored_fraction() == 0.0
    assert sparse.is_unknown(Vector2D(50, 50))
    assert sparse.key_position is None
