"""Rendering. Maps the integer tile matrix to colours on a Pygame surface.

Three renderers, all with the same `draw(...) -> keep_going` contract:

* `NullRenderer`     — no output (tests, batch runs)
* `TerminalRenderer` — one status line per event, for SSH / CI
* `PygameRenderer`   — the visual harness: fog-of-war world, the player's 10x10
  window outlined on it, and a sidebar of live episode stats

The Pygame view shows *what the agent knows*, not the ground truth, so you can
watch the map get uncovered. `--reveal` switches to the god view.
"""

from __future__ import annotations

import os
from pathlib import Path

from .entities import Tile
from .memory import FogMemory
from .scene import Scene
from .state import State
from .vector import Vector2D

# (r, g, b) for each tile, plus the chrome.
PALETTE = {
    Tile.EMPTY: (208, 210, 214),
    Tile.WALL: (34, 38, 46),
    Tile.KEY: (245, 197, 66),
    Tile.PLAYER: (86, 173, 235),
    Tile.UNKNOWN: (16, 18, 22),
}
BACKGROUND = (12, 13, 16)
PANEL = (22, 24, 29)
TEXT = (226, 228, 233)
MUTED = (138, 143, 152)
ACCENT = (86, 173, 235)
VIEWBOX = (245, 197, 66)


class NullRenderer:
    def draw(self, scene, state, memory, info) -> bool:
        return True

    def close(self) -> None:
        pass


class TerminalRenderer:
    """Periodic ASCII minimap of the fog-of-war map, for SSH / CI runs.

    The episode already logs a line per agent call, so this only adds the thing
    text logs cannot convey: the shape of what has been uncovered so far.
    """

    def __init__(self, every: int = 10, target_width: int = 50):
        self._every = max(1, every)
        self._target_width = target_width
        self._n = 0

    def draw(self, scene: Scene, state: State, memory: FogMemory | None, info: dict) -> bool:
        if info.get("phase") != "thinking" or memory is None:
            return True
        self._n += 1
        if (self._n - 1) % self._every:
            return True
        print(f"\n  --- map after {info['calls']} calls "
              f"({memory.explored_fraction():.0%} explored) ---")
        for line in self._minimap(scene, memory):
            print("  " + line)
        print()
        return True

    def _minimap(self, scene: Scene, memory: FogMemory) -> list[str]:
        block = max(1, -(-scene.width // self._target_width))  # ceil division
        player, key = scene.player.position, memory.key_position
        lines = []
        for r0 in range(0, scene.height, block):
            row = []
            for c0 in range(0, scene.width, block):
                cells = [memory.grid[r][c]
                         for r in range(r0, min(r0 + block, scene.height))
                         for c in range(c0, min(c0 + block, scene.width))]
                inside = (r0 <= player.row < r0 + block and c0 <= player.col < c0 + block)
                key_here = key is not None and (r0 <= key.row < r0 + block
                                                and c0 <= key.col < c0 + block)
                if inside:
                    row.append("P")
                elif key_here:
                    row.append("K")
                elif all(cell == Tile.UNKNOWN for cell in cells):
                    row.append(" ")
                elif any(cell == Tile.EMPTY for cell in cells):
                    row.append(".")
                else:
                    row.append("#")
            lines.append("".join(row))
        return lines

    def close(self) -> None:
        pass


class PygameRenderer:
    """Windowed (or offscreen) view of the episode."""

    def __init__(
        self,
        scene: Scene,
        *,
        cell: int = 6,
        sidebar: int = 320,
        reveal: bool = False,
        fps: int = 30,
        headless: bool = False,
        frame_dir: str | Path | None = None,
        frame_every: int = 1,
    ):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # imported lazily so the harness runs without a display

        self.pygame = pygame
        self.reveal = reveal
        self.cell = cell
        self.frame_dir = Path(frame_dir) if frame_dir else None
        self.frame_every = max(1, frame_every)
        self._frames = 0

        pygame.init()
        pygame.display.set_caption("maze harness")
        self.map_px = (scene.width * cell, scene.height * cell)
        self.size = (self.map_px[0] + sidebar, max(self.map_px[1], 520))
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 14)
        self.font_big = pygame.font.SysFont("dejavusansmono,monospace", 20, bold=True)

        if self.frame_dir:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

        # 1px-per-cell surface, scaled up on blit: one cheap redraw per reveal.
        self._tiles = pygame.Surface((scene.width, scene.height))
        self._tiles_revision = -1
        self._last_reasoning = ""

    # ------------------------------------------------------------------- draw

    def draw(self, scene: Scene, state: State, memory: FogMemory | None, info: dict) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False

        self.screen.fill(BACKGROUND)
        self._blit_map(scene, state, memory)
        self._blit_sidebar(scene, state, memory, info)
        pygame.display.flip()
        self._save_frame()
        self.clock.tick(self.fps)
        return True

    def _blit_map(self, scene: Scene, state: State, memory: FogMemory | None) -> None:
        pygame = self.pygame
        revision = -1 if self.reveal else (memory.revision if memory else 0)
        if revision != self._tiles_revision:
            self._paint_tiles(scene, memory)
            self._tiles_revision = revision

        surface = pygame.transform.scale(self._tiles, self.map_px)
        self.screen.blit(surface, (0, 0))

        # Key: always in the god view, otherwise only once it has been seen.
        key_pos = scene.key.position if self.reveal else (memory.key_position if memory else None)
        if key_pos is not None:
            self._cell_rect(key_pos, PALETTE[Tile.KEY], inset=1)
        self._cell_rect(scene.player.position, PALETTE[Tile.PLAYER])

        # Outline of the 10x10 window the agent actually observes.
        origin, span = state.view_origin, len(state.grid)
        rect = pygame.Rect(origin.col * self.cell, origin.row * self.cell,
                           span * self.cell, span * self.cell)
        pygame.draw.rect(self.screen, VIEWBOX, rect, 1)

    def _paint_tiles(self, scene: Scene, memory: FogMemory | None) -> None:
        source = scene.terrain if (self.reveal or memory is None) else memory.grid
        set_at = self._tiles.set_at
        for r, row in enumerate(source):
            for c, code in enumerate(row):
                set_at((c, r), PALETTE.get(Tile(code), PALETTE[Tile.UNKNOWN]))

    def _cell_rect(self, pos: Vector2D, colour, inset: int = 0) -> None:
        rect = self.pygame.Rect(pos.col * self.cell - inset, pos.row * self.cell - inset,
                                self.cell + 2 * inset, self.cell + 2 * inset)
        self.pygame.draw.rect(self.screen, colour, rect)

    def _blit_sidebar(self, scene: Scene, state: State, memory: FogMemory | None,
                      info: dict) -> None:
        pygame = self.pygame
        x0 = self.map_px[0]
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(x0, 0, self.size[0] - x0, self.size[1]))
        x = x0 + 14
        y = 14

        def line(text: str, colour=TEXT, font=None, gap: int = 18) -> None:
            nonlocal y
            self.screen.blit((font or self.font).render(text, True, colour), (x, y))
            y += gap

        theme = scene.meta.get("theme", scene.meta.get("generator", "maze"))
        line(str(theme)[:26], ACCENT, self.font_big, 30)
        line(f"policy     {info.get('policy', '—')}", MUTED)
        line(f"phase      {info.get('phase', '—')}", MUTED)
        y += 8
        line(f"step       {info['steps']} / {info['max_steps']}")
        line(f"agent call {info['calls']}")
        line(f"collisions {info['collisions']}")
        line(f"position   {state.player_position}")
        explored = memory.explored_fraction() if memory else 0.0
        line(f"explored   {explored:.1%}")
        key_known = memory.key_position if memory else None
        line(f"key        {key_known if key_known else 'not found yet'}")
        y += 10

        # Legend.
        for label, colour in (("you", PALETTE[Tile.PLAYER]), ("key", PALETTE[Tile.KEY]),
                              ("floor", PALETTE[Tile.EMPTY]), ("wall", PALETTE[Tile.WALL]),
                              ("unseen", PALETTE[Tile.UNKNOWN])):
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, y + 3, 10, 10))
            self.screen.blit(self.font.render(label, True, MUTED), (x + 18, y))
            y += 16
        y += 8
        self.screen.blit(self.font.render("yellow box = agent's 10x10 view", True, MUTED), (x, y))

    def _save_frame(self) -> None:
        if not self.frame_dir:
            return
        self._frames += 1
        if self._frames % self.frame_every:
            return
        self.pygame.image.save(self.screen, str(self.frame_dir / f"frame_{self._frames:06d}.png"))

    def close(self) -> None:
        self.pygame.quit()


def make_renderer(mode: str, scene: Scene, **kwargs):
    """`mode` is one of: none, terminal, window, headless."""
    if mode == "none":
        return NullRenderer()
    if mode == "terminal":
        return TerminalRenderer()
    if mode in ("window", "headless"):
        return PygameRenderer(scene, headless=(mode == "headless"), **kwargs)
    raise ValueError(f"unknown render mode {mode!r}")
