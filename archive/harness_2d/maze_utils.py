"""Grid utilities: tolerant parsing, connectivity repair, procedural fallback.

The connectivity repair matters because the maze bands come out of a language
model: a model that emits 100 rows of 100 characters will occasionally seal off a
pocket. Repair is a deterministic post-process, so a generated maze is *always*
solvable regardless of how the model performed.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Sequence

from .entities import GLYPH_TO_TILE, Tile
from .moves import Direction
from .vector import Vector2D

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------- parsing


def parse_rows(
    rows: Sequence[str], *, width: int, height: int | None = None, fill: int = Tile.WALL
) -> list[bytearray]:
    """Parse glyph rows into a terrain grid, padding/truncating to shape.

    Deliberately forgiving: models drop characters, add spaces, or emit one row
    too few. Anything unrecognised becomes `fill` (a wall), which is the safe
    direction — a spurious wall is repaired by `connect_regions`, whereas a
    spurious opening could not be detected at all.
    """
    grid: list[bytearray] = []
    for raw in rows:
        text = str(raw).strip()
        cells = bytearray(fill for _ in range(width))
        for i, ch in enumerate(text[:width]):
            cells[i] = GLYPH_TO_TILE.get(ch, fill)
        grid.append(cells)
    if height is not None:
        while len(grid) < height:
            grid.append(bytearray(fill for _ in range(width)))
        del grid[height:]
    return grid


def ensure_border_walls(terrain: list[bytearray]) -> None:
    """Seal the outer ring so the world edge is a wall like any other."""
    height, width = len(terrain), len(terrain[0])
    for c in range(width):
        terrain[0][c] = Tile.WALL
        terrain[height - 1][c] = Tile.WALL
    for r in range(height):
        terrain[r][0] = Tile.WALL
        terrain[r][width - 1] = Tile.WALL


def open_cells(terrain: list[bytearray]) -> list[Vector2D]:
    return [
        Vector2D(r, c)
        for r, row in enumerate(terrain)
        for c, cell in enumerate(row)
        if cell != Tile.WALL
    ]


def open_ratio(terrain: list[bytearray]) -> float:
    total = len(terrain) * len(terrain[0])
    return len(open_cells(terrain)) / total if total else 0.0


# ---------------------------------------------------------------- connectivity


def label_regions(terrain: list[bytearray]) -> tuple[list[list[int]], int]:
    """Flood-fill every open cell into a connected-component id (-1 for walls)."""
    height, width = len(terrain), len(terrain[0])
    labels = [[-1] * width for _ in range(height)]
    count = 0
    for r0 in range(height):
        for c0 in range(width):
            if terrain[r0][c0] == Tile.WALL or labels[r0][c0] != -1:
                continue
            queue = deque([(r0, c0)])
            labels[r0][c0] = count
            while queue:
                r, c = queue.popleft()
                for dr, dc in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and terrain[nr][nc] != Tile.WALL
                        and labels[nr][nc] == -1
                    ):
                        labels[nr][nc] = count
                        queue.append((nr, nc))
            count += 1
    return labels, count


def connect_regions(terrain: list[bytearray]) -> int:
    """Carve the cheapest set of corridors that makes every open cell reachable.

    Runs a multi-source 0-1 BFS that grows each region into the surrounding wall
    space (entering a wall costs 1, an open cell costs 0), which yields, for every
    cell, its nearest region and the wall-thickness to reach it. Adjacent cells
    owned by different regions become candidate corridors; a minimum spanning
    tree over those candidates picks which ones to actually carve. O(N log N).

    Returns the number of corridors carved.
    """
    labels, count = label_regions(terrain)
    if count <= 1:
        return 0

    height, width = len(terrain), len(terrain[0])
    inf = 1 << 30
    dist = [[inf] * width for _ in range(height)]
    owner = [[-1] * width for _ in range(height)]
    parent: list[list[tuple[int, int] | None]] = [[None] * width for _ in range(height)]

    def carvable(r: int, c: int) -> bool:
        """Border walls are load-bearing: never tunnel out through them."""
        if terrain[r][c] != Tile.WALL:
            return True
        return 0 < r < height - 1 and 0 < c < width - 1

    queue: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in range(width):
            if labels[r][c] != -1:
                dist[r][c] = 0
                owner[r][c] = labels[r][c]
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = dist[r][c]
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width) or not carvable(nr, nc):
                continue
            weight = 1 if terrain[nr][nc] == Tile.WALL else 0
            nd = d + weight
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                owner[nr][nc] = owner[r][c]
                parent[nr][nc] = (r, c)
                (queue.append if weight else queue.appendleft)((nr, nc))

    # Candidate corridors: any adjacency between cells owned by different regions.
    edges: list[tuple[int, int, int, tuple[int, int], tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if owner[r][c] == -1:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr >= height or nc >= width or owner[nr][nc] == -1:
                    continue
                if owner[r][c] != owner[nr][nc]:
                    edges.append(
                        (
                            dist[r][c] + dist[nr][nc],
                            owner[r][c],
                            owner[nr][nc],
                            (r, c),
                            (nr, nc),
                        )
                    )
    edges.sort(key=lambda e: e[0])

    root = list(range(count))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    def carve_back(cell: tuple[int, int]) -> None:
        """Walk the BFS parent chain back to its region, opening every wall on it."""
        node: tuple[int, int] | None = cell
        while node is not None and dist[node[0]][node[1]] > 0:
            terrain[node[0]][node[1]] = Tile.EMPTY
            node = parent[node[0]][node[1]]

    carved = 0
    for _weight, region_a, region_b, cell_a, cell_b in edges:
        ra, rb = find(region_a), find(region_b)
        if ra == rb:
            continue
        root[ra] = rb
        carve_back(cell_a)
        carve_back(cell_b)
        carved += 1
    return carved


def bfs_path(
    terrain: list[bytearray], start: Vector2D, goal: Vector2D
) -> list[Direction] | None:
    """Shortest directional path between two open cells, or None if unreachable."""
    if start == goal:
        return []
    height, width = len(terrain), len(terrain[0])
    seen = {start.as_tuple(): None}
    queue: deque[Vector2D] = deque([start])
    while queue:
        cur = queue.popleft()
        for direction in Direction:
            nxt = cur + direction.delta
            if not (0 <= nxt.row < height and 0 <= nxt.col < width):
                continue
            if terrain[nxt.row][nxt.col] == Tile.WALL or nxt.as_tuple() in seen:
                continue
            seen[nxt.as_tuple()] = (cur, direction)
            if nxt == goal:
                path: list[Direction] = []
                node = nxt
                while seen[node.as_tuple()] is not None:
                    prev, step = seen[node.as_tuple()]
                    path.append(step)
                    node = prev
                return list(reversed(path))
            queue.append(nxt)
    return None


# ------------------------------------------------------------------- placement


def random_open_cell(
    terrain: list[bytearray],
    rng: random.Random,
    *,
    exclude: Iterable[Vector2D] = (),
    min_distance: int = 0,
    anchor: Vector2D | None = None,
) -> Vector2D:
    """Pick a random open cell, preferring ones far from `anchor`."""
    banned = {p.as_tuple() for p in exclude}
    cells = [c for c in open_cells(terrain) if c.as_tuple() not in banned]
    if not cells:
        raise ValueError("maze has no open cells")
    if anchor is not None and min_distance > 0:
        far = [c for c in cells if c.manhattan(anchor) >= min_distance]
        if far:
            cells = far
    return rng.choice(cells)


# --------------------------------------------------------- procedural fallback


def procedural_maze(
    side_length: int, rng: random.Random, *, braid: float = 0.12
) -> list[bytearray]:
    """Randomised depth-first ("recursive backtracker") maze.

    Used by `--offline` and when an agent generation call fails, so a run never
    dies on one bad API response.
    """
    width = height = side_length
    terrain = [bytearray(int(Tile.WALL) for _ in range(width)) for _ in range(height)]
    terrain[1][1] = Tile.EMPTY
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        candidates = [
            (r + dr, c + dc)
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
            if 1 <= r + dr < height - 1
            and 1 <= c + dc < width - 1
            and terrain[r + dr][c + dc] == Tile.WALL
        ]
        if not candidates:
            stack.pop()
            continue
        nr, nc = rng.choice(candidates)
        terrain[(r + nr) // 2][(c + nc) // 2] = Tile.EMPTY
        terrain[nr][nc] = Tile.EMPTY
        stack.append((nr, nc))

    # Braiding: punch through some dead ends so the maze has loops. A perfect
    # maze is miserable to explore with a 10x10 window.
    if braid > 0:
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if terrain[r][c] != Tile.EMPTY:
                    continue
                exits = [
                    (r + dr, c + dc)
                    for dr, dc in _NEIGHBOURS
                    if terrain[r + dr][c + dc] == Tile.EMPTY
                ]
                if len(exits) == 1 and rng.random() < braid:
                    walls = [
                        (r + dr, c + dc)
                        for dr, dc in _NEIGHBOURS
                        if terrain[r + dr][c + dc] == Tile.WALL
                        and 0 < r + dr < height - 1
                        and 0 < c + dc < width - 1
                    ]
                    if walls:
                        wr, wc = rng.choice(walls)
                        terrain[wr][wc] = Tile.EMPTY
    ensure_border_walls(terrain)
    return terrain
