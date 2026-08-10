from maze_harness.entities import Entity, Tile
from maze_harness.maze_utils import parse_rows
from maze_harness.moves import Direction, Move
from maze_harness.scene import Scene
from maze_harness.state import State
from maze_harness.vector import Vector2D


def make_scene(rows, player=(1, 1), key=(1, 3)):
    terrain = parse_rows(rows, width=len(rows[0]), height=len(rows))
    return Scene(
        width=len(rows[0]),
        height=len(rows),
        terrain=terrain,
        entities=[Entity(Tile.PLAYER, Vector2D(*player)), Entity(Tile.KEY, Vector2D(*key))],
    )


ROOM = [
    "#####",
    "#...#",
    "#.#.#",
    "#...#",
    "#####",
]


def test_collision_is_a_terrain_lookup():
    scene = make_scene(ROOM)
    assert scene.is_blocked(Vector2D(0, 0))       # wall
    assert scene.is_blocked(Vector2D(-1, 1))      # off the edge reads as wall
    assert not scene.is_blocked(Vector2D(1, 2))


def test_step_moves_or_reports_collision():
    scene = make_scene(ROOM)
    assert scene.step(Direction.RIGHT) is True
    assert scene.player.position == Vector2D(1, 2)
    assert scene.step(Direction.UP) is False      # into the top wall
    assert scene.player.position == Vector2D(1, 2)


def test_success_is_position_equality():
    scene = make_scene(ROOM)
    assert not scene.is_solved()
    scene.step(Direction.RIGHT)
    scene.step(Direction.RIGHT)
    assert scene.player.position == scene.key.position
    assert scene.is_solved()


def test_local_view_shape_and_padding():
    scene = make_scene(ROOM)
    view = scene.local_view(size=10)
    assert len(view) == 10 and all(len(row) == 10 for row in view)
    # Player sits at the (size-1)//2 offset...
    assert view[4][4] == Tile.PLAYER
    # ...and everything outside the 5x5 world reads as wall.
    assert view[0][0] == Tile.WALL
    assert view[9][9] == Tile.WALL
    # The key is two cells to the right of the player.
    assert view[4][6] == Tile.KEY


def test_state_observation_metadata():
    scene = make_scene(ROOM)
    state = State.observe(scene, step=3, last_outcome="x")
    assert state.player_position == Vector2D(1, 1)
    assert state.view_origin == Vector2D(-3, -3)
    assert state.world_size == (5, 5)
    assert state.key_visible is True
    assert "P" in state.to_text() and "K" in state.to_text()


def test_scene_round_trips_through_json(tmp_path):
    scene = make_scene(ROOM)
    scene.meta = {"theme": "test"}
    path = scene.save(tmp_path / "scene.json")
    restored = Scene.load(path)
    assert restored.to_rows() == scene.to_rows()
    assert restored.player.position == scene.player.position
    assert restored.key.position == scene.key.position
    assert restored.meta["theme"] == "test"


def test_move_parsing_is_tolerant():
    move = Move.from_structured_output({"reasoning": "r", "actions": ["UP", "d", "Left", "east"]})
    assert move.actions == [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    assert Move.from_structured_output({"actions": "up, up down"}).actions == [
        Direction.UP, Direction.UP, Direction.DOWN
    ]
