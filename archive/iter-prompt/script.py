"""Isolated single-room generation test harness for prompt and schema iteration.

Usage:
    uv run python iter-prompt/run.py             # codex agent, save world, & 3D render
    uv run python iter-prompt/run.py --offline   # procedural fallback (no agent calls)
    uv run python iter-prompt/run.py --concept guitar
    uv run python iter-prompt/run.py --replay    # pick from saved worlds and inspect
    uv run python iter-prompt/run.py --no-render # prompt & parse test only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

# Ensure root repository directory and src/ are in sys.path, and remove iter-prompt
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
ITER_PROMPT_DIR = Path(__file__).resolve().parent
WORLDS_DIR = ITER_PROMPT_DIR / "worlds"

while str(ITER_PROMPT_DIR) in sys.path:
    sys.path.remove(str(ITER_PROMPT_DIR))
while "" in sys.path:
    sys.path.remove("")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from engine.pipeline_log import PipelineLogger  # noqa: E402

# Import harness modules
from agents import ClaudeTmuxAgent, CodexTmuxAgent, TmuxAgent  # noqa: E402
from agents.base import Agent, AgentError  # noqa: E402
from engine.config import load_config  # noqa: E402
from geometry import FORWARD, Vec3, rotation_matrix  # noqa: E402
from geometry.superquadrics import (  # noqa: E402
    Sensor,
    Superquadric,
    SuperquadricHandler,
)
from navigation.policies import ManualPolicy  # noqa: E402
from navigation.state import State  # noqa: E402
from prompts.room import room_prompt  # noqa: E402
from world import layout as layout_module  # noqa: E402
from world import shell as shell_module  # noqa: E402
from world import task as task_module  # noqa: E402
from world.generation import (  # noqa: E402
    ROOM_SCHEMA,
    _aabb_inside,
    _aabb_overlaps,
    _parse_assembly_primitive,
    _procedural_object,
    _procedural_room,
    _room_reachable,
)
from world.layout import Door, Layout, Room, WorldConfig  # noqa: E402
from world.scene import Player, Scene  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs/3d.yaml"
AGENTS = {"claude": ClaudeTmuxAgent, "codex": CodexTmuxAgent}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_sample_layout(
    cfg: WorldConfig,
    *,
    room_id: str = "study",
    room_name: str = "The Scholar's Study",
    style: str = (
        "A scholarly Victorian study with dark walnut bookshelves, "
        "brass instruments, and reading furniture."
    ),
    key_concept: str = "telescope",
    wall_color: tuple[float, float, float] = (0.78, 0.74, 0.70),
    floor_color: tuple[float, float, float] = (0.38, 0.28, 0.20),
    ceiling_color: tuple[float, float, float] = (0.92, 0.90, 0.88),
    origin_cells: tuple[int, int] = (5, 5),
    size_cells: tuple[int, int] = (4, 4),
) -> Layout:
    """Creates a sample single-room layout with one exterior exit door."""
    raw = {
        "theme": room_name,
        "description": style,
        "levels": 1,
        "rooms": [
            {
                "id": room_id,
                "name": room_name,
                "style": style,
                "key_concept": key_concept,
                "palette": {
                    "wall": list(wall_color),
                    "floor": list(floor_color),
                    "ceiling": list(ceiling_color),
                },
                "level": 0,
                "origin": list(origin_cells),
                "size": list(size_cells),
            }
        ],
        "connections": [],
    }
    return layout_module.validate_and_repair(raw, cfg)


def instantiate(payload: dict, origin: Vec3, interior) -> list[Superquadric]:
    """`objects` + `placements` -> world-space primitives, with chatty rejects.

    Deliberately a copy of `world.generation._instantiate` rather than an import:
    this harness exists to hack on the payload shape, and it should be able to
    diverge from src/ mid-iteration without breaking the real pipeline.
    """
    objects = {
        str(obj["id"]): obj.get("primitives") or []
        for obj in payload.get("objects") or []
        if obj.get("id")
    }
    instantiated: list[Superquadric] = []
    counts: dict[str, int] = {}

    for idx, placement in enumerate(payload.get("placements") or []):
        object_id = str(placement.get("object", ""))
        primitives = objects.get(object_id)
        if not primitives:
            print(f"  [X] Placement #{idx} names unknown object '{object_id}'")
            continue
        try:
            asm_pos = Vec3.parse(placement.get("position", [0.0, 0.0, 0.0]))
            asm_rot = tuple(
                float(a) % 360.0
                for a in Vec3.parse(
                    placement.get("rotation", [0.0, 0.0, 0.0])
                ).as_tuple()
            )
            mult = float(placement.get("scale", 1.0))
        except (ValueError, TypeError, IndexError) as exc:
            print(f"  [X] Rejected placement #{idx} '{object_id}': {exc}")
            continue
        if not math.isfinite(mult) or mult <= 0:
            print(f"  [X] Rejected placement #{idx} '{object_id}': scale {mult}")
            continue

        label = f"{object_id}#{counts.get(object_id, 0)}"
        counts[object_id] = counts.get(object_id, 0) + 1
        r_asm = rotation_matrix(asm_rot)
        for part_idx, item in enumerate(primitives):
            try:
                instantiated.append(
                    _parse_assembly_primitive(
                        item,
                        label,
                        asm_pos,
                        r_asm,
                        origin,
                        len(instantiated),
                        interior,
                        scale_mult=mult,
                    )
                )
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                print(f"  [X] Rejected part #{part_idx} of '{label}': {exc}")
    return instantiated


def generate_single_room_furniture(
    agent: Agent | None,
    room: Room,
    doors: list[Door],
    cfg: WorldConfig,
    rng: random.Random,
    *,
    brief: str | None = None,
    log: PipelineLogger | None = None,
) -> tuple[list[Superquadric], dict, str]:
    """Generates and validates furniture for a single room using
    iter-prompt's room_prompt."""
    interior = room.interior(cfg)
    (x0, y0, z0), (x1, y1, z1) = interior
    origin = Vec3((x0 + x1) / 2, y0, (z0 + z1) / 2)

    prompt_text = room_prompt(room, doors, cfg, interior, origin, brief=brief)

    print("\n" + "=" * 70)
    print("=== PROMPT SENT TO ROOM AGENT ===")
    print("=" * 70)
    print(prompt_text)
    print("=" * 70 + "\n")

    raw_payload: dict = {}
    source = "procedural"
    if agent is not None:
        print(f"[*] Querying room agent ({getattr(agent, 'model', 'agent')})...")
        t0 = time.monotonic()
        try:
            raw_payload = agent.run(prompt_text, ROOM_SCHEMA)
            elapsed = time.monotonic() - t0
            print(f"[+] Agent responded in {elapsed:.1f}s")
            source = "agent"
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            elapsed = time.monotonic() - t0
            print(f"[!] Agent call failed after {elapsed:.1f}s: {exc}")
            print("[*] Falling back to procedural room...")
            raw_payload = _procedural_room(rng, interior, key_concept=room.key_concept)
            source = f"fallback:{type(exc).__name__}"
    else:
        print("[*] Generating offline / procedural furniture (zero agent calls)...")
        raw_payload = _procedural_room(rng, interior, key_concept=room.key_concept)
        source = "procedural"

    print("\n" + "=" * 70)
    print("=== RAW MODEL RESPONSE (JSON) ===")
    print("=" * 70)
    print(json.dumps(raw_payload, indent=2))
    print("=" * 70 + "\n")

    clearances = [shell_module.door_clearance(door, cfg) for door in doors]
    validated: list[Superquadric] = []

    n_objects = len(raw_payload.get("objects") or [])
    n_placements = len(raw_payload.get("placements") or [])
    print(f"[*] {n_objects} object(s) modelled, {n_placements} placement(s)...")

    instantiated = instantiate(raw_payload, origin, interior)
    print(f"[*] Processing {len(instantiated)} instantiated primitive(s)...")
    rejected_count = 0

    for prim in instantiated:
        lo, hi = prim.aabb()
        if not _aabb_inside(lo, hi, interior):
            print(
                f"  [X] Rejected part of '{prim.assembly}': "
                "outside room interior bounds"
            )
            rejected_count += 1
            continue
        if any(_aabb_overlaps(lo, hi, *clearance) for clearance in clearances):
            print(
                f"  [X] Rejected part of '{prim.assembly}': overlaps doorway clearance"
            )
            rejected_count += 1
            continue

        validated.append(prim)
        print(
            f"  [v] Accepted #{len(validated) - 1} '{prim.assembly}' "
            f"scale={prim.scale} pos={prim.position}"
        )

    print(
        f"\n[*] Total validated primitives: {len(validated)} "
        f"(rejected: {rejected_count})"
    )

    # Reachability flood test
    is_reachable = _room_reachable(
        validated,
        room,
        doors,
        cfg,
        collision_resolution=8,
        move_increment=0.5,
    )
    if not is_reachable:
        print(
            "[!] Warning: Reachability flood test failed! "
            "Doorways are blocked by furniture."
        )
    else:
        print("[+] Reachability check passed: all doorways are reachable.")

    return validated, raw_payload, source


def assemble_single_room_scene(
    layout: Layout,
    furniture: list[Superquadric],
    cfg: WorldConfig,
    rng: random.Random,
    *,
    source: str = "agent",
) -> Scene:
    """Builds the full single-room Scene with shell, lock, and spawn point."""
    shell_prims, _ = shell_module.build_shell(layout, cfg)
    handler = SuperquadricHandler(shell_prims, collision_resolution=8)

    room = layout.rooms[0]
    door = layout.doors[0]

    # Group primitives by assembly
    assemblies: dict[str, list[Superquadric]] = {}
    for prim in furniture:
        assemblies.setdefault(prim.assembly, []).append(prim)

    # Fallback key if none
    if not assemblies:
        interior = room.interior(cfg)
        (x0, y0, z0), (x1, y1, z1) = interior
        origin = Vec3((x0 + x1) / 2, y0, (z0 + z1) / 2)
        key_payload = {
            "objects": [_procedural_object(room.key_concept)],
            "placements": [
                {
                    "object": room.key_concept,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": 1.0,
                }
            ],
        }
        for p in instantiate(key_payload, origin, interior):
            furniture.append(p)
            assemblies.setdefault(p.assembly, []).append(p)

    # Identify key assembly
    key_orig_name = None
    for name in assemblies:
        if (
            room.key_concept.lower() in name.lower()
            or name.lower() in room.key_concept.lower()
        ):
            key_orig_name = name
            break
    if key_orig_name is None:
        key_orig_name = next(iter(assemblies))

    print(
        f"[*] Identified key assembly matching concept "
        f"'{room.key_concept}': '{key_orig_name}'"
    )

    # Anonymize assembly labels uniquely: assembly-0, assembly-1, ...
    orig_names = list(assemblies.keys())
    rng.shuffle(orig_names)
    anon_map = {}
    for idx, name in enumerate(orig_names):
        anon_map[name] = f"assembly-{idx}"

    for prim in furniture:
        prim.assembly = anon_map.get(prim.assembly, f"assembly-{len(orig_names)}")

    key_anon_name = anon_map[key_orig_name]
    key_prims = assemblies[key_orig_name]

    # Add furniture primitives to handler
    for prim in furniture:
        prim.id = handler.next_id()
        handler.add(prim)

    # Build embossed lock on door
    lock_prims = task_module.build_lock(door, room, key_prims, handler, cfg)
    print(f"[+] Embossed door lock created with {len(lock_prims)} primitive(s)")

    # Set up player spawn
    spawn = room.centre(cfg)
    spawn_forward = FORWARD

    tasks = {
        str(room.index): {
            "key_concept": room.key_concept,
            "key_assembly": key_anon_name,
            "door_id": door.id,
            "lock_ids": [p.id for p in lock_prims],
        }
    }

    meta = {
        "theme": room.name,
        "description": room.style,
        "source": source,
        "room_source": source,
        "layout": {
            "theme": layout.theme,
            "rooms": [
                {
                    "id": room.id,
                    "name": room.name,
                    "style": room.style,
                    "level": room.level,
                    "index": room.index,
                    "box": [list(lo) for lo in room.box(cfg)],
                }
            ],
            "doors": [{"a": door.a, "b": door.b, "id": door.id}],
        },
        "task": tasks,
    }

    return Scene(
        bounds=cfg.bounds,
        primitives=handler,
        player=Player(position=spawn, forward=spawn_forward, radius=cfg.player_radius),
        meta=meta,
    )


def save_room_world(scene: Scene, theme: str, concept: str, source: str) -> Path:
    """Saves the generated room world under iter-prompt/worlds/."""
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"room-{_slugify(theme)}-{_slugify(concept)}-{stamp}"
    path = WORLDS_DIR / f"{stem}.json"
    scene.save(path)
    print(f"[+] Saved generated world to {path}")
    return path


def list_saved_worlds() -> list[Path]:
    """Returns all saved json worlds sorted newest first."""
    if not WORLDS_DIR.exists():
        return []
    return sorted(
        WORLDS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )


def replay_saved_world(
    sensor: Sensor,
    mesh_resolution: int = 16,
    target_path: str | None = None,
    no_render: bool = False,
) -> int:
    """Prompts user to select a past saved world and inspects it."""
    saved_worlds = list_saved_worlds()
    if not saved_worlds:
        print(f"[replay] No saved worlds found in {WORLDS_DIR}/. Generate one first!")
        return 1

    selected_path: Path
    if target_path:
        p = Path(target_path)
        if p.exists():
            selected_path = p
        elif (WORLDS_DIR / target_path).exists():
            selected_path = WORLDS_DIR / target_path
        elif (WORLDS_DIR / f"{target_path}.json").exists():
            selected_path = WORLDS_DIR / f"{target_path}.json"
        elif target_path.isdigit() and 1 <= int(target_path) <= len(saved_worlds):
            selected_path = saved_worlds[int(target_path) - 1]
        else:
            print(f"[replay] Could not find world: {target_path}")
            return 1
    else:
        print(
            f"\n[replay] {len(saved_worlds)} saved room world(s) "
            f"available under {WORLDS_DIR}/:"
        )
        for idx, path in enumerate(saved_worlds, 1):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                theme = data.get("meta", {}).get("theme", "Unknown")
                prims_count = len(data.get("primitives", []))
                concept = "?"
                for t in data.get("meta", {}).get("task", {}).values():
                    concept = t.get("key_concept", concept)
                print(
                    f"  {idx:>2}. {path.name:<45} "
                    f"[{theme} | key: {concept} | {prims_count} prims]"
                )
            except Exception:
                print(f"  {idx:>2}. {path.name}")

        while True:
            try:
                choice = input(f"\nReplay which one? [1-{len(saved_worlds)}]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nAborted.")
                return 0
            if choice.isdigit() and 1 <= int(choice) <= len(saved_worlds):
                selected_path = saved_worlds[int(choice) - 1]
                break
            print("Invalid choice. Please enter a valid number.")

    print(f"\n[*] Loading scene from {selected_path}...")
    scene = Scene.load(selected_path, collision_resolution=8)
    print(
        f"[+] Loaded {len(scene.primitives)} primitives. "
        f"Theme: {scene.meta.get('theme')}"
    )

    if not no_render:
        inspect_scene_interactively(scene, sensor, mesh_resolution=mesh_resolution)
    return 0


def inspect_scene_interactively(
    scene: Scene, sensor: Sensor, mesh_resolution: int = 16
) -> None:
    """Launches the Ursina renderer for visual inspection."""
    from engine.render import UrsinaRenderer

    print("\n" + "=" * 70)
    print("=== LAUNCHING 3D VISUAL INSPECTION ===")
    print("=" * 70)
    print("Controls:")
    print("  - MOUSE LEFT-DRAG:  Orbit camera around room")
    print("  - MOUSE RIGHT-DRAG: Pan camera")
    print("  - SCROLL WHEEL:     Zoom in/out")
    print("  - V:                Toggle first-person / third-person view")
    print("  - WASD + SPACE/SHIFT (in 1st person): Fly around room")
    print("  - M:                Toggle God Mode / Agent Mind")
    print("  - TAB:              Toggle HUD")
    print("  - ESC / Q:          Close inspection window")
    print("=" * 70 + "\n")

    renderer = UrsinaRenderer(
        scene,
        sensor=sensor,
        mesh_resolution=mesh_resolution,
        mouse_look=False,
    )
    renderer._build_agent(scene, "inspection")
    # Default to third person orbit for convenient room inspection
    renderer.toggle_view()
    renderer.toggle_agent_mind()  # God mode by default for full inspection

    policy = ManualPolicy(renderer, scene, speed=10.0, sensitivity=40.0)

    try:
        while not renderer.quit_requested:
            state = State.observe(scene, sensor)
            if not renderer.third_person:
                try:
                    policy.act(state)
                except Exception:
                    break

            info = {
                "policy": "manual",
                "phase": "inspecting",
                "rooms_cleared": 0,
                "rooms_total": 1,
            }
            alive = renderer.draw(scene, state, memory=None, info=info)
            if not alive:
                break
    finally:
        renderer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Isolated room generation and prompt iteration harness.",
    )
    parser.add_argument(
        "--replay",
        nargs="?",
        const=True,
        default=False,
        help="Replay/inspect a previously saved world from iter-prompt/worlds/",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Procedural generation (no LLM calls)"
    )
    parser.add_argument(
        "--agent",
        choices=["codex", "claude"],
        default=None,
        help="Agent CLI to use (default from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override agent model name"
    )
    parser.add_argument(
        "--effort",
        type=str,
        default=None,
        help="Override agent effort level (low/medium/high/xhigh/max)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="telescope",
        help="Key concept object (default: telescope)",
    )
    parser.add_argument(
        "--room-name",
        type=str,
        default="The Scholar's Study",
        help="Name of the test room",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=(
            "A scholarly Victorian study with dark walnut bookshelves, "
            "brass instruments, and reading furniture."
        ),
        help="Room furnishing style brief",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="Room size in grid cells (default: 4 -> 16m x 16m)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not open 3D Ursina window (headless / prompt debug)",
    )
    parser.add_argument(
        "--save-world",
        type=str,
        default=None,
        help="Explicit path to save generated world JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for procedural elements"
    )

    # Support 'replay' positional argument if passed (e.g. `run.py replay`)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "replay":
        argv = ["--replay"] + argv[1:]

    args = parser.parse_args(argv)

    # Load base configuration
    config = load_config(CONFIG_PATH)
    sensor = Sensor(
        range=config.sensor.range,
        fov=config.sensor.fov,
        aspect=config.sensor.aspect,
        stride=config.sensor.stride,
    )
    mesh_res = int(config.world.mesh_resolution)

    # Handle Replay mode
    if args.replay:
        target = args.replay if isinstance(args.replay, str) else None
        return replay_saved_world(
            sensor,
            mesh_resolution=mesh_res,
            target_path=target,
            no_render=args.no_render,
        )

    rng = random.Random(args.seed)

    world_cfg = WorldConfig(
        bounds=float(config.world.bounds),
        grid_unit=float(config.world.grid_unit),
        level_height=float(config.world.level_height),
        wall_thickness=float(config.world.wall_thickness),
        door_width=float(config.world.door_width),
        door_height=float(config.world.door_height),
        player_radius=float(config.world.player_radius),
        max_levels=1,
        max_rooms=1,
    )

    # Build single room layout
    layout = build_sample_layout(
        world_cfg,
        room_name=args.room_name,
        style=args.style,
        key_concept=args.concept,
        size_cells=(args.size, args.size),
    )

    room = layout.rooms[0]
    doors = layout.doors

    # Initialize Agent if not offline
    agent = None
    if not args.offline:
        agent_name = args.agent or config.agent.name
        agent_cls = AGENTS[agent_name]
        model = args.model or config.agent.model
        effort = args.effort or config.agent.effort
        binary = config.agent.binary
        print(
            f"[*] Initializing {agent_name} agent (model={model}, effort={effort})..."
        )
        agent = agent_cls(
            model=model,
            binary=binary,
            timeout=float(config.agent.timeout),
            max_retries=int(config.agent.retries),
            session_name="general-intuition-prompt-iter",
            window_name="room-iter",
            effort=effort,
        )
        if isinstance(agent, TmuxAgent):
            agent.clean()

    # Generate furniture using iter-prompt room_prompt
    furniture, raw_payload, source = generate_single_room_furniture(
        agent,
        room,
        doors,
        world_cfg,
        rng,
        brief=config.generation.brief,
    )

    # Assemble scene
    scene = assemble_single_room_scene(layout, furniture, world_cfg, rng, source=source)

    # Automatically save generated world under iter-prompt/worlds/
    save_room_world(scene, args.room_name, args.concept, source)
    if args.save_world:
        explicit_path = Path(args.save_world)
        scene.save(explicit_path)
        print(f"[+] Also saved to {explicit_path}")

    # Launch renderer unless disabled
    if not args.no_render:
        inspect_scene_interactively(
            scene,
            sensor,
            mesh_resolution=mesh_res,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
