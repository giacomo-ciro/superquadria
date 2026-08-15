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

import textwrap
import time
from dataclasses import dataclass
from typing import Any

from geometry import Vec3, basis, look_angles, look_vector
from geometry.superquadrics import DOOR, LOCK, OBJECT, OBSTACLE, PORTAL, Sensor, Superquadric, build_mesh
from navigation.state import State
from world.scene import Scene

BACKGROUND = (0.05, 0.06, 0.08)
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

#: How many past moves the history panel keeps on screen.
MOVE_HISTORY_ROWS = 6


def _clock(seconds: float) -> str:
    """mm:ss, growing an hours field only once there is one."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _wrap_text(text: str, width: int = 38, max_lines: int | None = None) -> list[str]:
    """Wrap text to fixed width and limit line count with ellipsis if max_lines is set."""
    if not text:
        return ["  (none)"]
    wrapped = textwrap.wrap(text.strip(), width=width)
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        if len(wrapped[-1]) > width - 3:
            wrapped[-1] = wrapped[-1][:width - 3] + "..."
        else:
            wrapped[-1] = wrapped[-1] + "..."
    return [f"  {line}" for line in wrapped]


def _pad_lines(lines: list[str], count: int) -> list[str]:
    """Pad lines with empty strings or truncate to exactly `count` items."""
    if len(lines) >= count:
        return lines[:count]
    return lines + [""] * (count - len(lines))


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
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec4 p3d_Color;
varying vec3 view_normal;
varying vec4 albedo;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    view_normal = p3d_NormalMatrix * p3d_Normal;
    albedo = p3d_Color;
}
""", fragment="""
#version 120
uniform vec3 light_direction;
uniform float ambient;
varying vec3 view_normal;
varying vec4 albedo;

void main() {
    float wrapped = 0.5 + 0.5 * dot(normalize(view_normal), normalize(light_direction));
    float light = ambient + (1.0 - ambient) * wrapped * wrapped;
    gl_FragColor = vec4(albedo.rgb * light, 1.0);
}
""", default_input={"light_direction": Vec3(*LIGHT_DIRECTION), "ambient": AMBIENT})


def _unlit_shader():
    """Unlit shading over the mesh's vertex colours or entity color scale."""
    from ursina.shader import Shader

    return Shader(name="unlit_shader", language=Shader.GLSL, vertex="""
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec4 p3d_Color;
varying vec4 vertex_color;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vertex_color = p3d_Color;
}
""", fragment="""
#version 120
uniform vec4 p3d_ColorScale;
varying vec4 vertex_color;

void main() {
    gl_FragColor = p3d_ColorScale * vertex_color;
}
""")


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


def _find_walls_without_doors(scene: Scene) -> set[int]:
    """Find IDs of all structural wall primitives that do not contain a doorway.

    In 3rd-person view, only walls that have a doorway are shown (to preserve
    doorway frames and locks), while solid walls (exterior walls and dividing
    walls without doors) are hidden for clear interior visibility.
    """
    doors = [p for p in scene.primitives if p.kind in (DOOR, PORTAL)]
    door_openings: list[tuple[int, float, float, float, float, float]] = []
    for d in doors:
        d_pos = d.position.as_tuple()
        if d.scale[0] <= d.scale[1] and d.scale[0] <= d.scale[2]:
            d_axis = 0
            d_h_min = d_pos[2] - d.scale[2]
            d_h_max = d_pos[2] + d.scale[2]
        elif d.scale[2] <= d.scale[0] and d.scale[2] <= d.scale[1]:
            d_axis = 2
            d_h_min = d_pos[0] - d.scale[0]
            d_h_max = d_pos[0] + d.scale[0]
        else:
            continue  # floor hole door
        d_coord = d_pos[d_axis]
        d_y_min = d_pos[1] - d.scale[1]
        d_y_max = d_pos[1] + d.scale[1]
        door_openings.append((d_axis, d_coord, d_h_min, d_h_max, d_y_min, d_y_max))

    hidden: set[int] = set()
    for p in scene.primitives:
        if p.kind != OBSTACLE or "wall" not in p.assembly:
            continue
        p_pos = p.position.as_tuple()
        w_axis = 0 if p.scale[0] < p.scale[2] else 2
        w_coord = p_pos[w_axis]
        w_h_min = p_pos[2] - p.scale[2] if w_axis == 0 else p_pos[0] - p.scale[0]
        w_h_max = p_pos[2] + p.scale[2] if w_axis == 0 else p_pos[0] + p.scale[0]
        w_y_min = p_pos[1] - p.scale[1]
        w_y_max = p_pos[1] + p.scale[1]

        has_door = False
        for (d_axis, d_coord, dh0, dh1, dy0, dy1) in door_openings:
            if d_axis != w_axis or abs(d_coord - w_coord) > 0.2:
                continue
            spans_door = (w_h_min <= dh0 + 0.1 and w_h_max >= dh1 - 0.1)
            left_jamb = abs(w_h_max - dh0) < 0.2 and not (w_y_max <= dy0 - 0.1 or w_y_min >= dy1 + 0.1)
            right_jamb = abs(w_h_min - dh1) < 0.2 and not (w_y_max <= dy0 - 0.1 or w_y_min >= dy1 + 0.1)

            if spans_door or left_jamb or right_jamb:
                has_door = True
                break
        if not has_door:
            hidden.add(p.id)

    return hidden


class UrsinaRenderer:
    """Window onto an episode, first person or orbiting third person.

    All world primitives are rendered and the GPU handles visual culling and
    occlusion; `visible_from()` decides only what the *agent* is told and what
    the HUD reports. A shape vanishing because the sensor lost it would be a
    different game.
    """

    def __init__(self, scene: Scene, *, sensor: Sensor, mesh_resolution: int = 16,
                 size: tuple[int, int] = (1280, 720), mouse_look: bool = False):
        from ursina import Entity, Mesh, Text, Ursina, camera, color, mouse, window

        self._camera = camera
        #: `Ursina` is a singleton proxy, so a checker sees none of `ShowBase`.
        self.app: Any = Ursina(title="superquadric harness", size=size, vsync=True,
                               development_mode=False, editor_ui_enabled=False, fullscreen=False,
                               show_ursina_splash=False)
        window.color = color.rgb(*BACKGROUND)  # pyright: ignore[reportCallIssue]

        camera.fov = sensor.fov  # pyright: ignore[reportAttributeAccessIssue]
        # Collision keeps every surface at least one player radius away, so a
        # near plane well inside that can never clip a wall the player is
        # legally allowed to be next to.
        camera.clip_plane_near = 0.05
        camera.clip_plane_far = scene.bounds * 4

        #: One cached entity per primitive, built once. Static geometry is never
        #: rebuilt, and sensor visibility never toggles any of it: every shape
        #: renders, and the GPU handles culling and occlusion.
        self.mesh_resolution = mesh_resolution
        #: `Any`, because Ursina annotates `Entity(shader=...)` as a class, not
        #: an instance, and every entity below would otherwise be an error.
        self._matte_shader: Any = _matte_shader()
        self._unlit_shader: Any = _unlit_shader()
        self._entity_keys: dict[int, tuple] = {}
        self.entities = {}
        for prim in scene.primitives:
            mesh = scene.primitives.mesh_data(prim.id, mesh_resolution)
            self.entities[prim.id] = Entity(
                model=Mesh(vertices=mesh.vertices, triangles=mesh.triangles,
                           normals=mesh.normals, colors=[color.rgba(*c) for c in mesh.colors]),
                # Every primitive is shaded, objects and the lock included: pure
                # colour with no lighting reads as a flat silhouette, and the
                # whole task is comparing *shape*.
                shader=self._matte_shader,
            )
            self._entity_keys[prim.id] = (prim.position.as_tuple(), prim.rotation, prim.scale, prim.exponents, prim.color)

        #: The agent's body: the driving model's own brand mark, or a plain
        #: sphere when there is no LLM policy to mark (manual play, tests). It
        #: exists only for third person — in first person the camera sits
        #: inside it. Built lazily on the first frame, once `info["policy"]`
        #: says which mark to use.
        self.agent: Entity | None = None
        self.carried_entity: Entity | None = None
        self._carried_prim_key = None

        self.show_hud = True
        self.hud = Text(parent=camera.ui, text="", position=(-0.86, 0.47), scale=0.65,
                        color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        self.footer = Text(parent=camera.ui, text="", position=(-0.86, -0.45), scale=0.65,
                           color=color.rgba(0.55, 0.60, 0.68, 1.0), font="VeraMono.ttf")
        #: Move history, agent reasoning and actions in the right column.
        self.moves = Text(parent=camera.ui, text="", position=(0.42, 0.47), scale=0.65,
                          color=color.rgba(0.90, 0.92, 0.96, 1.0), font="VeraMono.ttf")
        #: Last text assigned to each panel — Ursina rebuilds glyph meshes on
        #: every `.text =`, so skipping the assignment when nothing changed
        #: (e.g. a paused replay redrawing the same frame) avoids paying for
        #: that rebuild every frame.
        self._last_hud_text: str | None = None
        self._last_moves_text: str | None = None
        self._last_footer_text: str | None = None
        self.crosshair = Text(parent=camera.ui, text="+", position=(0, 0), origin=(0, 0),
                              scale=1.2, color=color.rgba(0.9, 0.9, 0.9, 0.5))
        #: Ursina drops wheel events before `held_keys`, so the only way to see a
        #: scroll is an entity input hook.
        self._input_entity = Entity(input=self._on_input)

        self.theme = str(scene.meta.get("theme", scene.meta.get("source", "world")))
        self.theme_description = str(scene.meta.get("description") or scene.meta.get("layout", {}).get("description") or "")
        #: Whether first person flies by mouse — false for an agent episode and
        #: for replay, where the cursor stays free to click.
        self.mouse_look = mouse_look
        self._mouse = mouse
        mouse.locked = mouse_look

        self.third_person = False
        self.agent_mind = True
        self._last_player_position = scene.player.position
        #: Third person opens framed on the agent inside the room.
        _, yaw = look_angles(scene.player.forward)
        self.orbit = Orbit(pitch=15.0, yaw=yaw, distance=6.0, pivot=scene.player.position)
        self._max_orbit_distance = scene.bounds * 1.5

        self._walls_without_doors = _find_walls_without_doors(scene)

        self.pending_keys: set[str] = set()
        self.mouse_delta = (0.0, 0.0)
        self.dt = 1 / 60
        self.quit_requested = False
        self._closed = False
        self._pressed: set[str] = set()
        self._last_frame = time.monotonic()

    def _build_agent(self, scene: Scene, policy_name: str | None) -> None:
        """Picks the agent's third-person body once `info["policy"]` is known."""
        from ursina import Entity, Mesh, color

        from geometry.agent_marks import MARKS_BY_POLICY

        mark = MARKS_BY_POLICY.get(policy_name) if policy_name else None
        if mark is None:
            self.agent = Entity(model="sphere", scale=scene.player.radius * 2, enabled=self.third_person,  # pyright: ignore[reportArgumentType]
                                shader=self._unlit_shader, color=color.rgba(*AGENT_COLOR))
        else:
            self.agent = Entity(
                model=Mesh(vertices=mark.vertices, triangles=mark.triangles,
                           normals=mark.normals, colors=[color.rgba(*c) for c in mark.colors]),
                scale=scene.player.radius, enabled=self.third_person, shader=self._unlit_shader,  # pyright: ignore[reportArgumentType]
            )
        self.carried_entity = Entity(parent=self.agent, enabled=False, shader=self._matte_shader)

    # ------------------------------------------------------------------- input

    def held(self, key: str) -> bool:
        from ursina import held_keys

        return bool(held_keys[key])  # pyright: ignore[reportIndexIssue]

    # -------------------------------------------------------------------- draw

    def draw(self, scene: Scene, state: State, memory, info: dict) -> bool:
        """Advance one frame. Returns False once the window is gone."""
        if self._closed:
            return False

        if self.agent is None:
            self._build_agent(scene, info.get("policy"))
        assert self.agent is not None and self.carried_entity is not None  # built just above

        self._last_player_position = scene.player.position

        # Ensure any newly created / re-added primitives also have an entity
        for prim in scene.primitives:
            if prim.id not in self.entities:
                from ursina import Entity, Mesh, color
                mesh = scene.primitives.mesh_data(prim.id, self.mesh_resolution)
                self.entities[prim.id] = Entity(
                    model=Mesh(vertices=mesh.vertices, triangles=mesh.triangles,
                               normals=mesh.normals, colors=[color.rgba(*c) for c in mesh.colors]),
                    shader=self._matte_shader,
                )
                self._entity_keys[prim.id] = (prim.position.as_tuple(), prim.rotation, prim.scale, prim.exponents, prim.color)

        # Sync primitive entities with scene state and agent memory.
        # In Agent Mind mode (default), only primitives the agent has seen are drawn.
        # In God Mode, all scene primitives are drawn.
        # The lock is part of the door assembly and renders whenever the door renders.
        mem_prims = memory.primitives if (memory and self.agent_mind) else None
        if mem_prims is not None:
            mem_prims_set = set(mem_prims)
            for task in scene.meta.get("task", {}).values():
                door_id = task.get("door_id")
                if door_id is not None and door_id in mem_prims_set:
                    for lid in task.get("lock_ids", []):
                        mem_prims_set.add(lid)
        else:
            mem_prims_set = None

        for prim_id, entity in self.entities.items():
            if prim_id not in scene.primitives:
                entity.enabled = False
            elif mem_prims_set is not None and prim_id not in mem_prims_set:
                entity.enabled = False
            else:
                prim = scene.primitives.get(prim_id)
                if prim.kind == PORTAL:
                    entity.enabled = False
                elif self.third_person and (prim_id in self._walls_without_doors or "outer wall" in prim.assembly):
                    entity.enabled = False
                else:
                    entity.enabled = True
                    geom_key = (prim.position.as_tuple(), prim.rotation, prim.scale, prim.exponents, prim.color)
                    if self._entity_keys.get(prim_id) != geom_key:
                        self._entity_keys[prim_id] = geom_key
                        from ursina import Mesh, color
                        mesh = scene.primitives.mesh_data(prim.id, self.mesh_resolution)
                        entity.model = Mesh(
                            vertices=mesh.vertices, triangles=mesh.triangles,
                            normals=mesh.normals, colors=[color.rgba(*c) for c in mesh.colors],
                        )

        # Update carried object display
        carrying = scene.player.carrying
        if carrying:
            self.carried_entity.enabled = self.third_person
            key = tuple((p.scale, p.exponents, p.color, p.position.as_tuple()) for p in carrying)
            if key != self._carried_prim_key:
                self._carried_prim_key = key
                from ursina import Mesh, color
                centroid = sum((p.position for p in carrying), Vec3(0.0, 0.0, 0.0)) * (1.0 / len(carrying))
                all_vertices = []
                all_triangles = []
                all_normals = []
                all_colors = []
                v_offset = 0
                max_radius = 0.5
                for p in carrying:
                    rel_pos = p.position - centroid
                    local_prim = Superquadric(
                        id=-1, kind=p.kind, assembly=p.assembly,
                        position=rel_pos, rotation=p.rotation,
                        scale=p.scale, exponents=p.exponents,
                        color=p.color,
                    )
                    m = build_mesh(local_prim, self.mesh_resolution)
                    all_vertices.extend(m.vertices)
                    all_normals.extend(m.normals)
                    all_colors.extend(m.colors)
                    all_triangles.extend((t[0] + v_offset, t[1] + v_offset, t[2] + v_offset) for t in m.triangles)
                    v_offset += len(m.vertices)
                    max_radius = max(max_radius, rel_pos.length() + max(p.scale))

                self.carried_entity.model = Mesh(
                    vertices=all_vertices, triangles=all_triangles,
                    normals=all_normals,
                    colors=[color.rgba(*c) for c in all_colors],
                )
                offset_z = scene.player.radius + max_radius + 0.1
                self.carried_entity.position = (0.0, 0.0, offset_z)
        else:
            self.carried_entity.enabled = False
            self._carried_prim_key = None

        pitch, yaw = look_angles(scene.player.forward)
        self.agent.position = scene.player.position.as_tuple()
        if self.third_person:
            self._place_orbit()
        else:
            self._camera.position = scene.player.position.as_tuple()
            self._camera.rotation = (pitch, yaw, 0)
        if self.show_hud:
            hud_text = self._hud_text(state, memory, info)
            moves_text = self._moves_text(info)
        else:
            hud_text = ""
            moves_text = ""
        if hud_text != self._last_hud_text:
            self.hud.text = self._last_hud_text = hud_text
        if moves_text != self._last_moves_text:
            self.moves.text = self._last_moves_text = moves_text
        footer_text = self._footer_text(info)
        if footer_text != self._last_footer_text:
            self.footer.text = self._last_footer_text = footer_text

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

        pressed = {key for key, value in held_keys.items() if value}  # pyright: ignore[reportAttributeAccessIssue]
        self.pending_keys = pressed - self._pressed
        self._pressed = pressed
        if "escape" in self.pending_keys or "q" in self.pending_keys:
            self.quit_requested = True
        # The button is unreachable while the cursor is locked to first-person
        # mouse look, so the key is the only way back out of it.
        if "v" in self.pending_keys:
            self.toggle_view()
        if "m" in self.pending_keys:
            self.toggle_agent_mind()
        if "tab" in self.pending_keys or "h" in self.pending_keys:
            self.toggle_hud()

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
        """Left drag rotates, right drag pans — the scene follows the cursor."""
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
        if self.agent is not None:
            self.agent.enabled = self.third_person
        self.crosshair.enabled = not self.third_person
        self._mouse.locked = self.mouse_look and not self.third_person
        self.mouse_delta = (0.0, 0.0)
        if self.third_person:
            self.orbit.pivot = self._last_player_position
            self.orbit.distance = min(self.orbit.distance, 8.0)

    def toggle_hud(self) -> None:
        """Toggle left and right HUD dashboard on and off."""
        self.show_hud = not self.show_hud
        self.hud.enabled = self.show_hud
        self.moves.enabled = self.show_hud

    def toggle_agent_mind(self) -> None:
        """Swap between Agent Mind (memory-only) and God Mode (full scene)."""
        self.agent_mind = not self.agent_mind

    # --------------------------------------------------------------------- hud

    def _hud_text(self, state: State, memory, info: dict) -> str:
        lines = [
            f"THEME: {self.theme}",
        ]
        if self.theme_description:
            lines.extend(_wrap_text(self.theme_description, width=38))
        lines.append("")
        lines.append(f"ROOM:  {state.room or 'unknown'}")
        if state.room_description:
            lines.extend(_wrap_text(state.room_description, width=38))

        lines.extend([
            "",
            "SPATIAL POSE",
            f" pos:  [{state.player_position.x:+5.1f}, {state.player_position.y:+5.1f}, {state.player_position.z:+5.1f}]",
            f" look: [{state.forward.x:+5.2f}, {state.forward.y:+5.2f}, {state.forward.z:+5.2f}]",
            "",
            "PUZZLE TARGET (LOCK)",
        ])
        if state.lock and "concept" in state.lock:
            lines.append(f" find:  {state.lock['concept']}")
        elif state.lock and "scale" in state.lock:
            sx, sy, sz = state.lock["scale"]
            e1, e2 = state.lock["exponents"]
            lines.append(f" scale: ({sx:.2f}, {sy:.2f}, {sz:.2f})")
            lines.append(f" exps:  ({e1:.2f}, {e2:.2f})")
        else:
            lines.append(" target: - (no active lock)")

        lines.append("")
        lines.append("INVENTORY (CARRYING)")
        if state.carrying:
            item_name = state.carrying.get("assembly", "object")
            parts = state.carrying.get("parts", 1)
            lines.append(f" item:  {item_name[:24]}")
            lines.append(f" parts: {parts} primitive(s)")
        else:
            lines.append(" item:  (none)")

        lines.append("")
        lines.append("SENSOR & MEMORY")
        n_vis = len(state.visible)
        n_obj = sum(1 for p in state.visible if p.kind == OBJECT)
        n_lock = sum(1 for p in state.visible if p.kind in (LOCK, DOOR))
        details = []
        if n_obj:
            details.append(f"{n_obj} obj")
        if n_lock:
            details.append(f"{n_lock} lock")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f" visible: {n_vis} shapes{suffix}")
        mem_shapes = len(memory.primitives) if memory else 0
        mem_poses = len(memory.poses) if memory else 0
        lines.append(f" memory:  {mem_shapes} shapes ({mem_poses} poses)")
        if self.third_person:
            lines.append(" *(3rd person: outer walls hidden for interior visibility)")

        lines.append("")
        lines.append("BUDGETS & PROGRESS")
        max_dist = info.get("max_distance", 0)
        dist_str = f"{info.get('distance', 0):.1f} / {'inf' if max_dist == float('inf') else f'{max_dist:.0f}'} m"
        lines.append(f" distance:  {dist_str}")
        max_coll = info.get("max_collisions", 0)
        coll_str = f"{info.get('collisions', 0)} / {'inf' if max_coll in (float('inf'), 0) else int(max_coll)}"
        lines.append(f" collision: {coll_str}")
        lines.append(f" rooms:     {info.get('rooms_cleared', 0)} / {info.get('rooms_total', 1)} cleared "
                     f"({info.get('failed_attempts', 0)} failed)")
        replay_str = f" replay:    {info['replay']}" if info.get("replay") else ""
        lines.append(replay_str)

        return "\n".join(lines)

    def _moves_text(self, info: dict) -> str:
        """Right column: active turn reasoning, action plan, and past turn history.

        Each section has fixed vertical slot allocation taking full column height
        so paragraphs stay in place without shifting as actions/reasoning progress.
        """
        elapsed = f"run {_clock(info.get('elapsed_s', 0.0))}"
        max_calls = info.get("max_calls", 0)
        call_str = f"{info.get('calls', 0)} / {'inf' if max_calls == float('inf') else int(max_calls)}"
        policy = info.get("policy", "-")
        model = info.get("agent_model")
        pol_str = f"{policy} ({model})" if model else policy

        if policy == "manual":
            mind_label = "god mode" if self.agent_mind else "agent mode"
            lines = [
                f"MANUAL PILOT{elapsed:>24}",
                f"policy  {policy}",
                "",
                "CONTROLS:",
                "  WASD        move horizontally",
                "  SPACE/SHIFT fly up / down",
                "  E           interact (pick/place)",
                "  MOUSE       look direction",
                f"  M           {mind_label}",
                "  TAB         toggle HUD",
            ]
            return "\n".join(_pad_lines(lines, 29))

        waiting = info.get("waiting_s") or 0.0
        phase = info.get("phase", "-")
        phase_str = f"{phase} ({waiting:.0f}s)" if waiting else phase

        # 1. Header (fixed 3 lines)
        header_lines = [
            f"TURN {call_str}{elapsed:>23}",
            f"policy  {pol_str}",
            f"phase   {phase_str}",
        ]

        # 2. Reasoning (fixed 5 content lines)
        active_turn = info.get("active_turn") or {}
        reasoning = active_turn.get("reasoning", "")
        raw_reasoning = _wrap_text(reasoning, width=38, max_lines=5)
        reasoning_slots = _pad_lines(raw_reasoning, 5)

        # 3. Actions (fixed 8 content lines)
        actions = active_turn.get("actions", [])
        action_lines = []
        if actions:
            if len(actions) > 8:
                for act in actions[:7]:
                    status = act.get("status", "pending")
                    icon = "[+]" if status == "done" else ("[>]" if status == "active" else ("[x]" if status == "failed" else "[ ]"))
                    atype = act.get("type", "MOVE")
                    target = act.get("target", [0, 0, 0])
                    t_str = f"[{target[0]:+4.1f}, {target[1]:+4.1f}, {target[2]:+4.1f}]"
                    tag = f" {act['info']}" if act.get("info") else (f" {act['dist']:.1f}m" if act.get("dist") else "")
                    action_lines.append(f" {icon} {act.get('idx', 1)}. {atype:<5} {t_str}{tag[:10]}")
                action_lines.append(f"     ... {len(actions) - 7} more")
            else:
                for act in actions:
                    status = act.get("status", "pending")
                    icon = "[+]" if status == "done" else ("[>]" if status == "active" else ("[x]" if status == "failed" else "[ ]"))
                    atype = act.get("type", "MOVE")
                    target = act.get("target", [0, 0, 0])
                    t_str = f"[{target[0]:+4.1f}, {target[1]:+4.1f}, {target[2]:+4.1f}]"
                    tag = f" {act['info']}" if act.get("info") else (f" {act['dist']:.1f}m" if act.get("dist") else "")
                    action_lines.append(f" {icon} {act.get('idx', 1)}. {atype:<5} {t_str}{tag[:10]}")
        else:
            action_lines.append("  (no actions planned)")
        action_slots = _pad_lines(action_lines, 8)

        # 4. History (fixed 6 content lines)
        history = info.get("history") or []
        history_lines = []
        if not history:
            history_lines.append("  (no completed turns yet)")
        else:
            for row in history[-6:]:
                history_lines.append(self._move_row(row["call"], row["gen_s"], f"{row['flown']}/{row['planned']}", row["result"]))
        history_slots = _pad_lines(history_lines, 6)

        lines = [
            *header_lines,
            "",
            "REASONING:",
            *reasoning_slots,
            "",
            f"ACTION BATCH ({len(actions)}):" if actions else "ACTIONS:",
            *action_slots,
            "",
            "PAST TURNS:",
            "  #    gen  flown  result",
            *history_slots,
        ]

        return "\n".join(lines)

    @staticmethod
    def _move_row(call: int, gen_s: float, waypoints: str, result: str,
                  *, pending: bool = False) -> str:
        return f"{'>' if pending else ' '}{call:>2} {gen_s:>5.1f}s {waypoints:>5}  {result[:14]}"

    def _footer_text(self, info: dict) -> str:
        view_toggle = f"M {'god mode' if self.agent_mind else 'agent mode'}   "
        speed = info.get("speed", 1.0)
        speed_hint = f"+/- speed ({speed:.2f}x)"
        if self.third_person:
            if info.get("phase") in ("idle", "replaying", "paused", "done"):
                return (f"SPACE play/pause   {speed_hint}   R restart   "
                        f"LEFT-drag rotate   SCROLL zoom   {view_toggle}TAB toggle HUD   V first person   ESC quit")
            return ("LEFT-drag rotate   RIGHT-drag pan   SCROLL zoom   "
                    f"{view_toggle}TAB toggle HUD   V first person   ESC quit")
        if info.get("phase") in ("idle", "replaying", "paused", "done"):
            return f"SPACE play/pause   {speed_hint}   R restart   {view_toggle}TAB toggle HUD   V third person   ESC quit"
        if info.get("policy") == "manual":
            return ("WASD move   E interact   SPACE up   SHIFT down   "
                    f"{view_toggle}TAB toggle HUD   V third person   ESC quit")
        return f"{view_toggle}TAB toggle HUD   V third person   ESC/close to quit"

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
