import random

from maze_harness.entities import Tile
from maze_harness.maze_utils import (
    bfs_path,
    connect_regions,
    ensure_border_walls,
    label_regions,
    open_ratio,
    parse_rows,
    procedural_maze,
)
from maze_harness.vector import Vector2D


def test_parse_rows_pads_and_truncates():
    grid = parse_rows(["..", "....", "."], width=3, height=4)
    assert len(grid) == 4 and all(len(row) == 3 for row in grid)
    assert list(grid[0]) == [Tile.EMPTY, Tile.EMPTY, Tile.WALL]   # short row padded with wall
    assert list(grid[1]) == [Tile.EMPTY] * 3                      # long row truncated
    assert list(grid[3]) == [Tile.WALL] * 3                       # missing row filled in


def test_parse_rows_treats_unknown_glyphs_as_wall():
    (row,) = parse_rows(["?~."], width=3, height=1)
    assert list(row) == [Tile.WALL, Tile.WALL, Tile.EMPTY]


def test_connect_regions_joins_every_pocket():
    rows = [
        "#########",
        "#..###..#",
        "#..###..#",
        "#########",
        "#..###..#",
        "#..###..#",
        "#########",
    ]
    terrain = parse_rows(rows, width=9, height=7)
    assert label_regions(terrain)[1] == 4
    carved = connect_regions(terrain)
    assert carved == 3                       # a spanning tree over 4 regions
    assert label_regions(terrain)[1] == 1


def test_connect_regions_never_breaches_the_border():
    rows = ["#####", "#...#", "#####", "#...#", "#####"]
    terrain = parse_rows(rows, width=5, height=5)
    connect_regions(terrain)
    assert all(terrain[0][c] == Tile.WALL and terrain[4][c] == Tile.WALL for c in range(5))
    assert all(terrain[r][0] == Tile.WALL and terrain[r][4] == Tile.WALL for r in range(5))
    assert label_regions(terrain)[1] == 1


def test_connect_regions_is_a_noop_when_already_connected():
    terrain = parse_rows(["###", "#.#", "###"], width=3, height=3)
    assert connect_regions(terrain) == 0


def test_procedural_maze_is_bordered_and_fully_connected():
    terrain = procedural_maze(41, 41, random.Random(0))
    ensure_border_walls(terrain)
    assert label_regions(terrain)[1] == 1
    assert 0.2 < open_ratio(terrain) < 0.7


def test_bfs_path_walks_from_start_to_goal():
    terrain = parse_rows(["#####", "#...#", "###.#", "#...#", "#####"], width=5, height=5)
    path = bfs_path(terrain, Vector2D(1, 1), Vector2D(3, 1))
    assert path is not None and len(path) == 6
    assert bfs_path(terrain, Vector2D(1, 1), Vector2D(1, 1)) == []


def test_bfs_path_returns_none_when_unreachable():
    terrain = parse_rows(["###", "#.#", "###", "#.#", "###"], width=3, height=5)
    assert bfs_path(terrain, Vector2D(1, 1), Vector2D(3, 1)) is None
