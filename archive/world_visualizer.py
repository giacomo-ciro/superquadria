"""Quick world screenshot/inspection tool.

Usage:
    python archive/world_visualizer.py [path_or_name_of_world]
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
WORLDS_DIR = ROOT_DIR / "worlds"

from engine.render import (  # noqa: E402
    BACKGROUND,
    MIN_ORBIT_DISTANCE,
    ORBIT_SENSITIVITY,
    PAN_SPEED,
    ZOOM_STEP,
    _find_walls_without_doors,
    _matte_shader,
    _silence_engine_logs,
)
from geometry import basis, look_angles, look_vector  # noqa: E402
from geometry.superquadrics import PORTAL  # noqa: E402
from world.scene import Scene  # noqa: E402


def main() -> None:
    # 1. Resolve world file
    if len(sys.argv) > 1:
        target = sys.argv[1]
        path = Path(target)
        if not path.exists():
            path = WORLDS_DIR / (
                target if target.endswith(".json") else f"{target}.json"
            )
        if not path.exists():
            print(f"Error: world file not found: {target}")
            return
    else:
        worlds = sorted(WORLDS_DIR.glob("*.json"))
        if not worlds:
            print(f"No worlds found under {WORLDS_DIR}/")
            return
        print("Available worlds:")
        for i, w in enumerate(worlds, 1):
            print(f"  {i:>2}. {w.stem}")
        try:
            choice = input(f"Select world [1-{len(worlds)}] (default 1): ").strip()
            idx = int(choice) - 1 if choice else 0
            path = worlds[idx]
        except (ValueError, IndexError):
            path = worlds[0]

    print(f"Loading {path}...")
    scene = Scene.load(path)
    theme = (
        scene.meta.get("theme")
        or scene.meta.get("layout", {}).get("theme")
        or path.stem
    )

    # 2. Setup Ursina renderer
    _silence_engine_logs()
    from ursina import Entity, Mesh, Ursina, application, camera, color, mouse, window

    app = Ursina(
        title=f"Inspect — {theme}",
        vsync=True,
        development_mode=False,
        editor_ui_enabled=False,
        fullscreen=False,
        show_ursina_splash=False,
    )
    window.color = color.rgb(*BACKGROUND)

    # Initial camera & orbit parameters
    camera.fov = 70
    camera.clip_plane_near = 0.05
    camera.clip_plane_far = scene.bounds * 15

    _, start_yaw = look_angles(scene.player.forward)
    pitch = 15.0
    yaw = start_yaw
    distance = 6.0
    pivot = scene.player.position
    max_orbit_distance = scene.bounds * 10

    # 3. Create primitive entities (no player/agent entity, no HUD)
    shader = _matte_shader()
    entities: dict[int, Entity] = {}
    mesh_res = 16

    for prim in scene.primitives:
        m = scene.primitives.mesh_data(prim.id, mesh_res)
        entities[prim.id] = Entity(
            model=Mesh(
                vertices=m.vertices,
                triangles=m.triangles,
                normals=m.normals,
                colors=[color.rgba(*c) for c in m.colors],
            ),
            shader=shader,
        )

    walls_without_doors = _find_walls_without_doors(scene)
    show_outer_walls = False

    def sync_visibility() -> None:
        for prim_id, entity in entities.items():
            prim = scene.primitives.get(prim_id)
            if prim.kind == PORTAL:
                entity.enabled = False
            elif not show_outer_walls and (
                prim_id in walls_without_doors or "outer wall" in prim.assembly
            ):
                entity.enabled = False
            else:
                entity.enabled = True

    sync_visibility()

    # 4. Input & Orbit Update
    def update() -> None:
        nonlocal pitch, yaw, pivot
        dx, dy = mouse.velocity[0], mouse.velocity[1]
        if mouse.left:
            yaw += dx * ORBIT_SENSITIVITY
            pitch = max(-89.0, min(89.0, pitch - dy * ORBIT_SENSITIVITY))
        elif mouse.right:
            _, right, up = basis(look_vector(pitch, yaw))
            step = PAN_SPEED * distance
            pivot = pivot - right * (dx * step) - up * (dy * step)

        forward = look_vector(pitch, yaw)
        camera.position = (pivot - forward * distance).as_tuple()
        camera.rotation = (pitch, yaw, 0)

    def input_handler(key: str) -> None:
        nonlocal show_outer_walls, distance
        if key in ("escape", "q"):
            application.quit()
        elif key == "o":
            show_outer_walls = not show_outer_walls
            sync_visibility()
            print(f"Outer walls: {'shown' if show_outer_walls else 'hidden'}")
        elif key in ("+", "=", "shift+=", "plus"):
            camera.fov = min(150.0, camera.fov + 10.0)
            print(f"FOV: {camera.fov:.0f}°")
        elif key in ("-", "_", "shift+-", "minus"):
            camera.fov = max(10.0, camera.fov - 10.0)
            print(f"FOV: {camera.fov:.0f}°")
        elif key == "scroll up":
            distance = max(MIN_ORBIT_DISTANCE, distance * ZOOM_STEP)
        elif key == "scroll down":
            distance = min(max_orbit_distance, distance * (1.0 / ZOOM_STEP))

    # Hook input and update via a controller entity
    Entity(update=update, input=input_handler)

    print("\n--- Controls ---")
    print("  Left-drag        : Rotate camera")
    print("  Right-drag       : Pan camera")
    print("  Scroll wheel     : Zoom in / out")
    print("  O                : Toggle outer walls")
    print("  + / =            : Increase FOV (+10°)")
    print("  - / _            : Decrease FOV (-10°)")
    print("  Q / ESC          : Quit\n")

    app.run()


if __name__ == "__main__":
    main()
