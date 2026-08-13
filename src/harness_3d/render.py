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
from dataclasses import dataclass

from .geometry import ZERO, Vec3, basis, look_angles, look_vector
from .scene import Scene
from .state import State
from .superquadrics import KEY, Sensor

BACKGROUND = (0.05, 0.06, 0.08)
BOUNDS_COLOR = (0.35, 0.55, 0.75, 0.30)
#: The agent's own body, drawn only in third person. Unlit, so it reads as a
#: marker rather than as one more shape in the world.
AGENT_COLOR = (0.30, 0.85, 1.00, 1.00)

#: Degrees of orbit per unit of mouse travel. The window is a little under two
#: units wide in Ursina's screen space, so a full drag across it turns the camera
#: most of the way round the agent.
ORBIT_SENSITIVITY = 200.0
#: Pan per unit of mouse travel, as a multiple of the orbit radius — scaled by
#: distance so the scene keeps pace with the cursor however far out you are.
PAN_SPEED = 1.5
#: Multiplier per wheel notch, and how close the camera may be pulled in.
ZOOM_STEP = 0.85
MIN_ORBIT_DISTANCE = 2.0

#: Fraction of a primitive's own colour that survives on the side facing away
#: from the light. The world is an unlit void, so a surface that drops much
#: below this stops reading as a shape at all.
AMBIENT = 0.5
#: The key light, in *view* space — it rides the camera rather than the world,
#: so whatever the player is looking at is lit. A fixed world sun leaves half of
#: every episode silhouetted against the background.
LIGHT_DIRECTION = (-0.35, 0.55, 1.0)

#: How many past moves the history panel keeps on screen. Older ones scroll off
#: with a count of what was dropped — the episode JSON is the full record, the
#: window is for what just happened.
MOVE_HISTORY_ROWS = 12


def _clock(seconds: float) -> str:
    """mm:ss, growing an hours field only once there is one."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _matte_shader():
    """Lambert shading over the mesh's own vertex colours.

    Ursina's `lit_with_shadows_shader` is additive (`albedo + NdotL/pi`), so lit
    faces clip to white and the primitive's colour is lost exactly where the
    shape reads best. This multiplies instead, which keeps every colour true,
    and wraps the cosine term so the unlit side darkens without going black.
    """
    from ursina import Vec3
    from ursina.shader import Shader

    return Shader(name="matte_shader", language=Shader.GLSL, vertex="""
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
out vec3 view_normal;
out vec4 albedo;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    view_normal = p3d_NormalMatrix * p3d_Normal;
    albedo = p3d_Color;
}
""", fragment="""
#version 140
uniform vec3 light_direction;
uniform float ambient;
in vec3 view_normal;
in vec4 albedo;
out vec4 fragment_color;

void main() {
    float wrapped = 0.5 + 0.5 * dot(normalize(view_normal), normalize(light_direction));
    float light = ambient + (1.0 - ambient) * wrapped * wrapped;
    fragment_color = vec4(albedo.rgb * light, 1.0);
}
""", default_input={"light_direction": Vec3(*LIGHT_DIRECTION), "ambient": AMBIENT})


@dataclass
class Orbit:
    """Third-person camera state: a look direction and a radius about a pivot.

    The pivot is a fixed point in the world, moved only by panning. Nothing here
    tracks the agent — third person is an observation post you set up and the
    agent flies through, not a camera that chases it.
    """

    pitch: float
    yaw: float
    distance: float
    pivot: Vec3


class UrsinaRenderer:
    """Window onto an episode, first person or orbiting third person.

    All world primitives are rendered and the GPU handles visual culling and
    occlusion; `visible_from()` decides only what the *agent* is told and what
    the HUD reports. A shape vanishing because the sensor lost it would be a
    different game.
    """

    def __init__(self, scene: Scene, *, sensor: Sensor, mesh_resolution: int = 16,
                 size: tuple[int, int] = (1280, 720), mouse_look: bool = False):
        from ursina import Button, Entity, Mesh, Text, Ursina, camera, color, mouse, window
        from ursina.shaders import unlit_shader

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
        matte_shader = _matte_shader()
        for prim in scene.primitives:
            mesh = scene.primitives.mesh_data(prim.id, mesh_resolution)
            self.entities[prim.id] = Entity(
                model=Mesh(vertices=mesh.vertices, triangles=mesh.triangles,
                           normals=mesh.normals, colors=[color.rgba(*c) for c in mesh.colors]),
                # The key reads as the goal, so it is lit by nothing and simply
                # glows; obstacles take the shaded one.
                shader=unlit_shader if prim.kind == KEY else matte_shader,
            )

        self._bounds_entity = Entity(model=self._bounds_mesh(Mesh, scene.bounds),
                                     shader=unlit_shader, color=color.rgba(*BOUNDS_COLOR))

        #: The agent's body. It exists only for third person — in first person
        #: the camera sits inside it, and a sphere around the near plane is at
        #: best invisible and at worst a smear across the whole view.
        self.agent = Entity(model="sphere", scale=scene.player.radius * 2, enabled=False,
                            shader=unlit_shader, color=color.rgba(*AGENT_COLOR))
        # A stick along the agent's own +Z: where it is looking matters as much
        # as where it is, and a sphere alone says nothing about heading.
        Entity(parent=self.agent, shader=unlit_shader, color=color.rgba(*AGENT_COLOR),
               model=Mesh(vertices=[(0, 0, 0), (0, 0, 2.0)], mode="line", thickness=2))

        self.hud = Text(parent=camera.ui, text="", position=(-0.86, 0.47), scale=0.75,
                        color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.footer = Text(parent=camera.ui, text="", position=(-0.86, -0.44), scale=0.7,
                           color=color.rgba(0.55, 0.60, 0.68, 1.0), font="VeraMono.ttf")
        #: Move history and the episode clock, in the column the left HUD leaves
        #: free — below the view button, which owns the top right corner.
        self.moves = Text(parent=camera.ui, text="", position=(0.48, 0.40), scale=0.7,
                          color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.crosshair = Text(parent=camera.ui, text="+", position=(0, 0), origin=(0, 0),
                              scale=1.2, color=color.rgba(0.9, 0.9, 0.9, 0.5))
        self.view_button = Button(parent=camera.ui, text="view: first person", text_size=0.55,
                                  position=(0.63, 0.46), scale=(0.30, 0.05), radius=0.25,
                                  color=color.rgba(0.12, 0.14, 0.18, 0.85),
                                  on_click=self.toggle_view)
        #: Ursina drops wheel events before `held_keys`, so the only way to see a
        #: scroll is an entity input hook.
        self._input_entity = Entity(input=self._on_input)

        self.theme = str(scene.meta.get("theme", scene.meta.get("source", "world")))
        #: Whether first person flies by mouse — false for an agent episode and
        #: for replay, where the cursor stays free to click.
        self.mouse_look = mouse_look
        self._mouse = mouse
        mouse.locked = mouse_look

        self.third_person = False
        #: A camera that does not move on its own has to open on a framing that
        #: keeps the agent in shot for a whole episode, so it starts back far
        #: enough to hold the entire cube, looking slightly down along the
        #: agent's spawn heading. From then on it is the user's: dragging is the
        #: only thing that ever moves it, and toggling views never resets it.
        _, yaw = look_angles(scene.player.forward)
        self.orbit = Orbit(pitch=20.0, yaw=yaw, distance=scene.bounds, pivot=ZERO)
        self._max_orbit_distance = scene.bounds * 1.5

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
        self.agent.position = scene.player.position.as_tuple()
        self.agent.rotation = (pitch, yaw, 0)
        if self.third_person:
            self._place_orbit()
        else:
            self._camera.position = scene.player.position.as_tuple()
            self._camera.rotation = (pitch, yaw, 0)
        self.hud.text = self._hud_text(state, memory, info)
        self.moves.text = self._moves_text(info)
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
        # The button is unreachable while the cursor is locked to first-person
        # mouse look, so the key is the only way back out of it.
        if "v" in self.pending_keys:
            self.toggle_view()

        if self.third_person:
            self._drag_orbit()
            self.mouse_delta = (0.0, 0.0)  # the camera consumed it; the agent gets nothing
        elif self.mouse_look:
            velocity = self._mouse.velocity
            self.mouse_delta = (velocity[0], velocity[1])

    def _on_input(self, key: str) -> None:
        """Wheel zoom. Ursina calls this on every entity, for every key."""
        if not self.third_person:
            return
        if key == "scroll up":
            self._zoom(ZOOM_STEP)
        elif key == "scroll down":
            self._zoom(1 / ZOOM_STEP)

    def _zoom(self, factor: float) -> None:
        self.orbit.distance = max(MIN_ORBIT_DISTANCE,
                                  min(self._max_orbit_distance, self.orbit.distance * factor))

    def _drag_orbit(self) -> None:
        """Left drag rotates, right drag pans — the scene follows the cursor.

        A drag that started on the view button is ignored, so clicking it cannot
        also fling the camera.
        """
        if self._mouse.hovered_entity is self.view_button:
            return
        dx, dy = self._mouse.velocity[0], self._mouse.velocity[1]
        orbit = self.orbit
        if self._mouse.left:
            orbit.yaw += dx * ORBIT_SENSITIVITY
            # Same clamp as first-person look: at exactly +/-90 the frame the
            # camera is built from degenerates.
            orbit.pitch = max(-89.0, min(89.0, orbit.pitch - dy * ORBIT_SENSITIVITY))
        elif self._mouse.right:
            _, right, up = basis(look_vector(orbit.pitch, orbit.yaw))
            # The pivot moves against the drag, which is what makes the scene
            # move with it. Scaled by radius so panning feels the same at any zoom.
            step = PAN_SPEED * orbit.distance
            orbit.pivot = orbit.pivot - right * (dx * step) - up * (dy * step)

    def _place_orbit(self) -> None:
        orbit = self.orbit
        forward = look_vector(orbit.pitch, orbit.yaw)
        self._camera.position = (orbit.pivot - forward * orbit.distance).as_tuple()
        self._camera.rotation = (orbit.pitch, orbit.yaw, 0)

    def toggle_view(self) -> None:
        """Swap first person for the orbit camera and back.

        Third person hands the cursor back — it is the camera's now — which also
        parks manual flight, since the policy reads the same mouse and keys.
        """
        self.third_person = not self.third_person
        self.agent.enabled = self.third_person
        self.crosshair.enabled = not self.third_person
        self.view_button.text = f"view: {'third' if self.third_person else 'first'} person"
        self._mouse.locked = self.mouse_look and not self.third_person
        self.mouse_delta = (0.0, 0.0)

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

    def _moves_text(self, info: dict) -> str:
        """Total runtime, then one row per agent call.

        A move is one call — a whole batch of waypoints — which is also the only
        unit that has a generation time worth reporting. Manual flight issues a
        call per movement increment, so it gets the clock and no rows: thousands
        of sub-millisecond "moves" say nothing about how the flight is going.
        """
        elapsed = f"run {_clock(info.get('elapsed_s', 0.0))}"
        header = f"moves{elapsed:>19}"
        if info.get("policy") == "manual":
            return header

        history = info.get("history") or []
        rows = [self._move_row(row["call"], row["gen_s"], f"{row['flown']}/{row['planned']}",
                               row["result"]) for row in history[-MOVE_HISTORY_ROWS:]]
        # The call in flight has no row yet, and a long one is exactly when the
        # panel is being read, so it gets a provisional row with a live timer.
        if info.get("phase") == "thinking":
            rows.append(self._move_row(info.get("calls", 0) + 1, info.get("waiting_s") or 0.0,
                                       "..", "thinking", pending=True))
        if not rows:
            return f"{header}\n\n  no moves yet"

        dropped = len(history) - MOVE_HISTORY_ROWS
        lines = [header, "  #    gen  flown  result"]
        if dropped > 0:
            lines.append(f"  ...  {dropped} earlier")
        return "\n".join(lines + rows)

    @staticmethod
    def _move_row(call: int, gen_s: float, waypoints: str, result: str,
                  *, pending: bool = False) -> str:
        return f"{'>' if pending else ' '}{call:>2} {gen_s:>5.1f}s {waypoints:>5}  {result[:14]}"

    def _footer_text(self, info: dict) -> str:
        if self.third_person:
            return ("LEFT-drag rotate    RIGHT-drag pan    SCROLL zoom    "
                    "V first person    ESC quit")
        if info.get("phase") in ("idle", "replaying", "paused", "done"):
            return "SPACE play/pause    V third person    ESC quit"
        if info.get("policy") == "manual":
            return ("WASD move    SPACE up    SHIFT down    mouse look    "
                    "V third person    ESC quit")
        return "the agent is flying — V third person, close the window to stop"

    def release_mouse(self) -> None:
        """Hand the cursor back — replay is watched, not flown."""
        self.mouse_look = False
        self._mouse.locked = False
        self.mouse_delta = (0.0, 0.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mouse.locked = False
        try:
            self.app.destroy()
        except Exception:  # already torn down by the window closing
            pass
