"""Rendering. Maps the integer tile matrix to colours on a Pygame surface.

`PygameRenderer` is the visual harness: a fog-of-war world, the player's 10x10
window outlined on it, and a sidebar of live episode stats. Its
`draw(...) -> keep_going` contract is what the engine drives.

The view shows *what the agent knows*, not the ground truth, so you can watch
the map get uncovered. The "god view" button in the window toggles the full-map
view on and off.
"""

from __future__ import annotations

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


class PygameRenderer:
    """Windowed (or offscreen) view of the episode.

    The window is user-resizable. Only the map area grows or shrinks with it —
    the sidebar on the right stays a fixed pixel width, so it never needs
    responsive layout of its own.
    """

    #: Smallest the map area is allowed to shrink to horizontally, so the
    #: sidebar can't eat the whole window and `cell` never hits zero.
    MIN_MAP_PX = 120
    #: Floor on the window's height, independent of the map — the sidebar's
    #: own content (stats, legend, controls hint) needs this much regardless
    #: of how small the map area gets.
    MIN_WINDOW_H = 516
    #: Lines of subtitle the base minimum window height already accommodates.
    SUBTITLE_LINES = 6
    SUBTITLE_LINE_HEIGHT = 15

    def __init__(
        self,
        scene: Scene,
        *,
        cell: int = 6,
        sidebar: int = 320,
        reveal: bool = False,
        fps: int = 30,
    ):
        import pygame  # imported lazily so the harness runs without a display

        self.pygame = pygame
        self.reveal = reveal
        self.scene_width = scene.width
        self.scene_height = scene.height
        self.sidebar = sidebar

        pygame.init()
        pygame.display.set_caption("maze harness")
        self._reveal_rect = pygame.Rect(
            0, 0, 0, 0
        )  # positioned for real by the first _blit_sidebar()
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 14)
        self.font_big = pygame.font.SysFont("dejavusansmono,monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("dejavusansmono,monospace", 12)
        description = " ".join(str(scene.meta.get("description") or "").split())
        subtitle_lines = self._wrap(description, self.font_small, sidebar - 28)
        self.min_window_h = (
            self.MIN_WINDOW_H
            + max(0, len(subtitle_lines) - self.SUBTITLE_LINES)
            * self.SUBTITLE_LINE_HEIGHT
        )
        initial_map_px = (scene.width * cell, scene.height * cell)
        self._resize((initial_map_px[0] + sidebar, initial_map_px[1]))
        self.clock = pygame.time.Clock()
        self.fps = fps

        # 1px-per-cell surface, scaled up on blit: one cheap redraw per reveal.
        self._tiles = pygame.Surface((scene.width, scene.height))
        # `None`, not -1: -1 is also the sentinel `_blit_map` uses for "god view is
        # on, don't bother diffing against fog revisions" — reusing it here would
        # make the very first god-view frame look identical to "nothing painted
        # yet" and the map would never get its first paint.
        self._tiles_revision = None
        #: Keys seen in the most recent `draw()`'s event batch — the only
        #: window into pygame's event queue callers other than `draw()` get,
        #: since `draw()` is what actually drains it.
        self.pending_keys: set[int] = set()

    # ----------------------------------------------------------------- layout

    def _resize(self, size: tuple[int, int]) -> None:
        """(Re)computes the map's pixel scale from a raw window size. The
        sidebar keeps its fixed width, right-anchored; the maze keeps its
        grid's aspect ratio (square cells) and is centered — letterboxed,
        never stretched — in whatever space is left."""
        width = max(self.sidebar + self.MIN_MAP_PX, size[0])
        height = max(self.min_window_h, size[1])
        self.size = (width, height)
        self.sidebar_x0 = width - self.sidebar

        available = (width - self.sidebar, height)
        self.cell = min(
            available[0] / self.scene_width, available[1] / self.scene_height
        )
        self.map_px = (
            round(self.scene_width * self.cell),
            round(self.scene_height * self.cell),
        )
        self.map_offset = (
            (available[0] - self.map_px[0]) // 2,
            (available[1] - self.map_px[1]) // 2,
        )

        self.screen = self.pygame.display.set_mode(self.size, self.pygame.RESIZABLE)

    # ------------------------------------------------------------------- draw

    def draw(
        self, scene: Scene, state: State, memory: FogMemory | None, info: dict
    ) -> bool:
        pygame = self.pygame
        self.pending_keys = set()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self._resize((event.w, event.h))
            if event.type == pygame.KEYDOWN:
                self.pending_keys.add(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._reveal_rect.collidepoint(event.pos):
                    self.reveal = not self.reveal

        self.screen.fill(BACKGROUND)
        self._blit_map(scene, state, memory)
        self._blit_sidebar(scene, state, memory, info)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _blit_map(self, scene: Scene, state: State, memory: FogMemory | None) -> None:
        pygame = self.pygame
        revision = -1 if self.reveal else (memory.revision if memory else 0)
        if revision != self._tiles_revision:
            self._paint_tiles(scene, memory)
            self._tiles_revision = revision

        surface = pygame.transform.scale(self._tiles, self.map_px)
        self.screen.blit(surface, self.map_offset)

        # Key: always in the god view, otherwise only once it has been seen.
        key_pos = (
            scene.key.position
            if self.reveal
            else (memory.key_position if memory else None)
        )
        if key_pos is not None:
            self._cell_rect(key_pos, PALETTE[Tile.KEY], inset=1)
        self._cell_rect(scene.player.position, PALETTE[Tile.PLAYER])

        # Outline of the 10x10 window the agent actually observes.
        origin, span = state.view_origin, len(state.grid)
        ox, oy = self.map_offset
        rect = pygame.Rect(
            ox + round(origin.col * self.cell),
            oy + round(origin.row * self.cell),
            round(span * self.cell),
            round(span * self.cell),
        )
        pygame.draw.rect(self.screen, VIEWBOX, rect, 1)

    def _paint_tiles(self, scene: Scene, memory: FogMemory | None) -> None:
        source = scene.terrain if (self.reveal or memory is None) else memory.grid
        set_at = self._tiles.set_at
        for r, row in enumerate(source):
            for c, code in enumerate(row):
                set_at((c, r), PALETTE.get(Tile(code), PALETTE[Tile.UNKNOWN]))

    def _cell_rect(self, pos: Vector2D, colour, inset: int = 0) -> None:
        ox, oy = self.map_offset
        rect = self.pygame.Rect(
            ox + round(pos.col * self.cell) - inset,
            oy + round(pos.row * self.cell) - inset,
            round(self.cell) + 2 * inset,
            round(self.cell) + 2 * inset,
        )
        self.pygame.draw.rect(self.screen, colour, rect)

    @staticmethod
    def _wrap(text: str, font, width: int) -> list[str]:
        """Greedy word wrap to `width` pixels.

        Measured with `font.size` rather than a character count: the sidebar's
        font is only monospace if the system actually has DejaVu Sans Mono, and
        SysFont falls back silently to a proportional face when it doesn't.
        """
        lines: list[str] = []
        current = ""
        for word in text.split():
            if font.size(word)[0] > width:
                if current:
                    lines.append(current)
                    current = ""
                while word:
                    end = len(word)
                    while font.size(word[:end])[0] > width:
                        end -= 1
                    lines.append(word[:end])
                    word = word[end:]
                continue
            candidate = f"{current} {word}" if current else word
            if current and font.size(candidate)[0] > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _blit_sidebar(
        self, scene: Scene, state: State, memory: FogMemory | None, info: dict
    ) -> None:
        pygame = self.pygame
        x0 = self.sidebar_x0
        pygame.draw.rect(
            self.screen, PANEL, pygame.Rect(x0, 0, self.size[0] - x0, self.size[1])
        )
        x = x0 + 14
        y = 14

        def line(text: str, colour=TEXT, font=None, gap: int = 18) -> None:
            nonlocal y
            self.screen.blit((font or self.font).render(text, True, colour), (x, y))
            y += gap

        theme = scene.meta.get("theme", scene.meta.get("generator", "maze"))
        line(str(theme)[:26], ACCENT, self.font_big, 30)

        # The generator's own blurb about the maze it drew, as a subtitle under
        # the theme. Collapsed to a single paragraph first: the model returns it
        # wrapped to its own line length, and those hard newlines are meaningless
        # at this width. Procedural mazes have no description and just skip it.
        description = " ".join(str(scene.meta.get("description") or "").split())
        if description:
            y -= 6  # tighten it up under the title it belongs to
            for text in self._wrap(description, self.font_small, self.sidebar - 28):
                line(text, MUTED, self.font_small, 15)
            y += 10

        # God-view toggle — its own filled row (not just a text line), so it
        # reads as a button and doesn't collide with the stats below it.
        self._reveal_rect = pygame.Rect(x, y, 112, 24)
        fill = ACCENT if self.reveal else (34, 38, 46)
        text_colour = BACKGROUND if self.reveal else MUTED
        pygame.draw.rect(self.screen, fill, self._reveal_rect, border_radius=4)
        label = self.font.render(
            f"god view: {'ON' if self.reveal else 'off'}", True, text_colour
        )
        self.screen.blit(
            label,
            (
                self._reveal_rect.centerx - label.get_width() // 2,
                self._reveal_rect.centery - label.get_height() // 2,
            ),
        )
        y += self._reveal_rect.height + 10

        line(f"policy     {info.get('policy', '—')}", MUTED)
        line(f"model      {info.get('agent_model') or '—'}", MUTED)
        line(f"effort     {info.get('agent_effort') or '—'}", MUTED)
        # The elapsed seconds are the only sign a long agent call is still
        # running: the window keeps redrawing throughout, so a hung call and a
        # slow one look identical without it.
        waiting = info.get("waiting_s") or 0.0
        phase = f"{info.get('phase', '—')}{f'  {waiting:.0f}s' if waiting else ''}"
        line(f"phase      {phase}", MUTED)
        y += 8
        budget = info["max_steps"]
        line(
            f"step       {info['steps']} / {'∞' if budget == float('inf') else budget}"
        )
        calls = "n/a" if info.get("policy") == "manual" else info["calls"]
        line(f"agent call {calls}")
        line(f"collisions {info['collisions']}")
        line(f"position   {state.player_position}")
        explored = memory.explored_fraction() if memory else 0.0
        line(f"explored   {explored:.1%}")
        key_known = memory.key_position if memory else None
        line(f"key        {key_known if key_known else 'not found yet'}")
        y += 10

        # Legend.
        for label, colour in (
            ("you", PALETTE[Tile.PLAYER]),
            ("key", PALETTE[Tile.KEY]),
            ("floor", PALETTE[Tile.EMPTY]),
            ("wall", PALETTE[Tile.WALL]),
            ("unseen", PALETTE[Tile.UNKNOWN]),
        ):
            pygame.draw.rect(self.screen, colour, pygame.Rect(x, y + 3, 10, 10))
            self.screen.blit(self.font.render(label, True, MUTED), (x + 18, y))
            y += 16
        y += 8
        span = len(state.grid)
        self.screen.blit(
            self.font.render(f"yellow box = agent's {span}x{span} view", True, MUTED),
            (x, y),
        )
        if info.get("phase") in ("idle", "replaying", "paused", "done"):
            y += 20
            self.screen.blit(self.font.render("SPACE play/pause", True, ACCENT), (x, y))

    def close(self) -> None:
        self.pygame.quit()
