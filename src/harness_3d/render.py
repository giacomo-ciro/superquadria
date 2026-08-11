"""Ursina integration — a renderer and an input loop, not the world state.

Everything authoritative (collision, visibility, success) already happened in
`SuperquadricHandler` and `Scene` before a frame is drawn. Ursina is handed
already-transformed vertices and never asked a question about the world, because
duplicating its Euler convention in authoritative geometry invites exactly the
kind of subtle disagreement that is impossible to debug later.

Ursina is imported lazily, inside the constructor: `generate` and every headless
check must run on a machine with no display.
"""

from __future__ import annotations

import time

from .geometry import look_angles
from .scene import Scene
from .state import State
from .superquadrics import KEY, Sensor

BACKGROUND = (0.05, 0.06, 0.08)
BOUNDS_COLOR = (0.35, 0.55, 0.75, 0.30)


class UrsinaRenderer:
    """First-person window onto an episode.

    All world primitives are rendered and the GPU handles visual culling and
    occlusion; `visible_from()` decides only what the *agent* is told and what
    the HUD reports. A shape vanishing because the sensor lost it would be a
    different game.
    """

    def __init__(self, scene: Scene, *, sensor: Sensor, mesh_resolution: int = 16,
                 size: tuple[int, int] = (1280, 720), mouse_look: bool = False):
        from ursina import Entity, Mesh, Text, Ursina, camera, color, window
        from ursina.lights import AmbientLight, DirectionalLight
        from ursina.shaders import lit_with_shadows_shader, unlit_shader

        self._camera = camera
        self.app = Ursina(title="superquadric harness", size=size, vsync=True,
                          development_mode=False, editor_ui_enabled=False, fullscreen=False,
                          show_ursina_splash=False)
        window.color = color.rgb(*BACKGROUND)

        camera.fov = sensor.fov
        # Collision keeps every surface at least one player radius away, so a
        # near plane well inside that can never clip a wall the player is
        # legally allowed to be next to.
        camera.clip_plane_near = 0.05
        camera.clip_plane_far = scene.bounds * 4

        #: One cached entity per primitive, built once. Static geometry is never
        #: rebuilt, and sensor visibility never toggles any of it: every shape
        #: renders, and the GPU handles culling and occlusion.
        self.entities = {}
        for prim in scene.primitives:
            mesh = scene.primitives.mesh_data(prim.id, mesh_resolution)
            self.entities[prim.id] = Entity(
                model=Mesh(vertices=mesh.vertices, triangles=mesh.triangles,
                           normals=mesh.normals, colors=[color.rgba(*c) for c in mesh.colors]),
                # The key reads as the goal, so it is lit by nothing and simply
                # glows; obstacles take the lit shader.
                shader=unlit_shader if prim.kind == KEY else lit_with_shadows_shader,
            )

        self._bounds_entity = Entity(model=self._bounds_mesh(Mesh, scene.bounds),
                                     shader=unlit_shader, color=color.rgba(*BOUNDS_COLOR))

        # Built after the geometry: the light sizes its shadow map to the bounds
        # of the scene as it stands when it is created.
        self.sun = DirectionalLight(shadows=True, rotation=(48, -34, 0))
        AmbientLight(color=color.rgba(0.28, 0.30, 0.36, 1.0))

        self.hud = Text(parent=camera.ui, text="", position=(-0.86, 0.47), scale=0.75,
                        color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.footer = Text(parent=camera.ui, text="", position=(-0.86, -0.44), scale=0.7,
                           color=color.rgba(0.55, 0.60, 0.68, 1.0), font="VeraMono.ttf")
        self.crosshair = Text(parent=camera.ui, text="+", position=(0, 0), origin=(0, 0),
                              scale=1.2, color=color.rgba(0.9, 0.9, 0.9, 0.5))

        self.theme = str(scene.meta.get("theme", scene.meta.get("source", "world")))
        self.mouse_look = mouse_look
        if mouse_look:
            from ursina import mouse
            mouse.locked = True
            self._mouse = mouse
        else:
            self._mouse = None

        self.pending_keys: set[str] = set()
        self.mouse_delta = (0.0, 0.0)
        self.dt = 1 / 60
        self.quit_requested = False
        self._closed = False
        self._pressed: set[str] = set()
        self._last_frame = time.monotonic()

    @staticmethod
    def _bounds_mesh(Mesh, bounds: float):
        """A wireframe cube marking the playable volume — a guide, not geometry."""
        h = bounds / 2
        corners = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if sum(corners[a][i] != corners[b][i] for i in range(3)) == 1]
        return Mesh(vertices=corners, triangles=edges, mode="line", thickness=2)

    # ------------------------------------------------------------------- input

    def held(self, key: str) -> bool:
        from ursina import held_keys

        return bool(held_keys[key])

    # -------------------------------------------------------------------- draw

    def draw(self, scene: Scene, state: State, memory, info: dict) -> bool:
        """Advance one frame. Returns False once the window is gone."""
        if self._closed:
            return False

        pitch, yaw = look_angles(scene.player.forward)
        self._camera.position = scene.player.position.as_tuple()
        self._camera.rotation = (pitch, yaw, 0)
        self.hud.text = self._hud_text(state, memory, info)
        self.footer.text = self._footer_text(info)

        try:
            self.app.step()
        except SystemExit:  # closing the window unwinds ShowBase through sys.exit
            self._closed = True
            return False

        now = time.monotonic()
        self.dt = min(0.1, now - self._last_frame)  # a stalled frame must not teleport anyone
        self._last_frame = now
        self._collect_input()
        return not self._closed

    def _collect_input(self) -> None:
        from ursina import held_keys

        pressed = {key for key, value in held_keys.items() if value}
        self.pending_keys = pressed - self._pressed
        self._pressed = pressed
        if "escape" in self.pending_keys or "q" in self.pending_keys:
            self.quit_requested = True
        if self._mouse is not None:
            velocity = self._mouse.velocity
            self.mouse_delta = (velocity[0], velocity[1])

    # --------------------------------------------------------------------- hud

    def _hud_text(self, state: State, memory, info: dict) -> str:
        waiting = info.get("waiting_s") or 0.0
        phase = f"{info.get('phase', '-')}{f'  {waiting:.0f}s' if waiting else ''}"
        max_calls = info.get("max_calls", 0)
        calls = "n/a" if info.get("policy") == "manual" else \
            f"{info.get('calls', 0)} / {'inf' if max_calls == float('inf') else int(max_calls)}"
        max_distance = info.get("max_distance", 0)
        distance = (f"{info.get('distance', 0):.0f} / "
                    f"{'inf' if max_distance == float('inf') else f'{max_distance:.0f}'}")
        key = memory.key if memory else None
        remembered = len(memory.primitives) if memory else 0
        lines = [
            self.theme[:34],
            "",
            f"phase       {phase}",
            f"policy      {info.get('policy', '-')}",
            f"model       {info.get('agent_model') or '-'}",
            "",
            f"call        {calls}",
            f"distance    {distance}",
            f"collisions  {info.get('collisions', 0)}",
            f"position    {state.player_position}",
            f"visible     {len(state.visible)} shapes",
            f"remembered  {remembered} shapes",
            f"key         {key.position if key else 'not found yet'}",
        ]
        if info.get("replay"):
            lines.append(f"replay      {info['replay']}")
        return "\n".join(lines)

    def _footer_text(self, info: dict) -> str:
        if info.get("phase") in ("idle", "replaying", "paused", "done"):
            return "SPACE play/pause    ESC quit"
        if info.get("policy") == "manual":
            return "WASD move    SPACE up    SHIFT down    mouse look    ESC quit"
        return "the agent is flying — close the window to stop"

    def release_mouse(self) -> None:
        """Hand the cursor back — replay is watched, not flown."""
        if self._mouse is not None:
            self._mouse.locked = False
            self._mouse = None
            self.mouse_delta = (0.0, 0.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._mouse is not None:
            self._mouse.locked = False
        try:
            self.app.destroy()
        except Exception:  # already torn down by the window closing
            pass
