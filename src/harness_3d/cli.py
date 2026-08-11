"""Command line entrypoints: `generate`, `play`, `run`, `replay`.

All settings besides the command and `--offline` live in `configs/3d.yaml`, read
fresh on every run — the same convention as the 2D harness, and the same loader.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import signal
import sys
import time
from pathlib import Path

from omegaconf import DictConfig

from harness_2d.agents.claude_tmux import ClaudeTmuxAgent
from harness_2d.agents.codex_tmux import CodexTmuxAgent
from harness_2d.agents.tmux_agent import TmuxAgent
from harness_2d.config import load_config

from .engine import Budgets, Episode, Replay, load_episode, save_episode
from .generation import WorldGenerator
from .navigation import WaypointNavigator
from .policies import ManualPolicy, Policy
from .scene import Scene
from .superquadrics import Sensor

#: Worlds and episodes are separate artifacts: a world is generated once and can
#: be played many times, so every episode only records the path it ran on.
CONFIG_PATH = Path("configs/3d.yaml")
WORLDS_DIR = Path("worlds")
EPISODES_DIR = Path("episodes_3d")

#: Selected by `agent.name`, which comes from the `defaults: - agent:` group in
#: configs/3d.yaml — reused wholesale from the 2D harness, which is what
#: 3D_PLAN.md section 9 asks for until a shared package is justified.
AGENTS = {"claude": ClaudeTmuxAgent, "codex": CodexTmuxAgent}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness_3d",
        description="Agent harness that generates a 3D superquadric world and flies it. "
                    f"All other settings live in {CONFIG_PATH}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a world and save it under worlds/")
    gen.add_argument("--offline", action="store_true", help="procedural world, zero agent calls")

    sub.add_parser("play", help="pick a saved world under worlds/ and fly a fresh episode on it")

    run = sub.add_parser("run", help="generate a world under worlds/, then immediately fly it")
    run.add_argument("--offline", action="store_true",
                     help="procedural world plus manual play — zero agent calls")

    sub.add_parser("replay", help="pick a saved episode under episodes_3d/ and replay it")

    return parser


# --------------------------------------------------------------------- helpers


def _make_agent(cfg: DictConfig, role: str) -> TmuxAgent:
    if cfg.name not in AGENTS:
        raise RuntimeError(f"unknown agent {cfg.name!r} — expected one of {sorted(AGENTS)}")
    return AGENTS[cfg.name](
        model=cfg.model,
        binary=cfg.binary,
        timeout=cfg.timeout,
        max_retries=cfg.retries,
        # One session per role, so both can be watched at once — see the 2D README.
        session_name=f"general-intuition-3d-{role}",
        window_name=role,
        effort=cfg.effort,
    )


def _sensor(config: DictConfig) -> Sensor:
    s = config.sensor
    return Sensor(range=s.range, fov=s.fov, aspect=s.aspect)


def _generate(config: DictConfig, *, offline: bool) -> Scene:
    world, gen = config.world, config.generation
    generator = WorldGenerator(
        agent=None if offline else _make_agent(config.agent, role="generator"),
        bounds=world.bounds,
        player_radius=world.player_radius,
        move_increment=config.episode.move_increment,
        collision_resolution=world.collision_resolution,
        target_primitives=gen.target_primitives,
        min_primitives=gen.min_primitives,
        max_primitives=gen.max_primitives,
        exponent_range=tuple(gen.exponent_range),
        validation_spacing=gen.validation_spacing,
        min_free_voxels=gen.min_free_voxels,
        min_key_distance=gen.min_key_distance,
        sensor=_sensor(config),
    )
    started = time.monotonic()
    # `--offline` ignores the configured brief entirely: it directs agent
    # generation and there is no agent here to direct.
    scene = generator.generate_offline() if offline else generator.generate(gen.brief)
    scene.meta["generation_s"] = round(time.monotonic() - started, 1)
    print(f"[gen] {len(scene.primitives) - 1} primitives in a {scene.bounds:.0f}-unit cube "
          f"in {scene.meta['generation_s']:.1f}s | spawn {scene.player.position} -> "
          f"key {scene.key.position} | witness path {scene.meta['witness_path_length']:.0f} units")
    return scene


def _make_policy(config: DictConfig, renderer, scene: Scene, name: str) -> Policy:
    ep = config.episode
    if name == "manual":
        return ManualPolicy(renderer, scene, speed=ep.manual_speed,
                            sensitivity=ep.mouse_sensitivity)
    if name == "agent":
        return WaypointNavigator(
            _make_agent(config.agent, role="player"),
            sensor_range=config.sensor.range,
            max_waypoints=ep.max_waypoints,
            max_segment=ep.max_segment,
            call_budget=ep.max_calls,
            distance_budget=ep.max_distance,
            collision_budget=ep.max_collisions,
        )
    raise RuntimeError(f"unknown episode.policy {name!r} — expected agent or manual")


def _play(config: DictConfig, scene: Scene, world: Path, out: Path, *, offline: bool = False) -> int:
    from .render import UrsinaRenderer  # imported here so `generate` never touches Ursina

    ep = config.episode
    # `run --offline` means procedural generation followed by manual play, which
    # is what guarantees the whole command makes zero agent calls.
    policy_name = "manual" if offline else ep.policy
    sensor = _sensor(config)
    renderer = UrsinaRenderer(scene, sensor=sensor, mesh_resolution=config.world.mesh_resolution,
                              mouse_look=policy_name == "manual")
    try:
        policy = _make_policy(config, renderer, scene, policy_name)

        # A human at the controls gets no budget: manual play runs until the key
        # is reached or the window is closed.
        unlimited = policy.name == "manual"
        budgets = Budgets(
            max_calls=math.inf if unlimited else ep.max_calls,
            max_waypoints=ep.max_waypoints,
            max_segment=ep.max_segment,
            max_distance=math.inf if unlimited else ep.max_distance,
            max_collisions=math.inf if unlimited else ep.max_collisions,
            move_increment=ep.move_increment,
        )
        result = Episode(scene, policy, sensor=sensor, budgets=budgets, renderer=renderer,
                         step_delay=ep.step_delay).run()

        extra = {"policy": policy.name}
        agent = getattr(policy, "agent", None)
        if agent is not None:
            extra["agent_model"] = agent.model
            extra["agent_effort"] = agent.effort
            extra["agent_stats"] = agent.stats
            extra["agent_failures"] = str(getattr(policy, "failures", 0))
        save_episode(out, world, result, extra)

        print(f"\n{result.summary()}")
        print(f"witness path was {scene.meta.get('witness_path_length', '?')} units; "
              f"episode written to {out}")

        print("[replay] window stays open — SPACE play/pause, close the window to quit")
        renderer.release_mouse()
        Replay(scene, renderer, result, policy.name, sensor=sensor,
               agent_model=getattr(agent, "model", None),
               agent_effort=getattr(agent, "effort", None), step_delay=ep.step_delay).run()
        return 0 if result.solved else 1
    finally:
        renderer.close()


def _list_json(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), key=lambda p: p.name, reverse=True)


def _prompt_choice(paths: list[Path], verb: str) -> Path:
    while True:
        choice = input(f"{verb} which one? [1-{len(paths)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(paths):
            return paths[int(choice) - 1]
        print("invalid choice")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _world_path(scene: Scene, stamp: str) -> Path:
    """Where to write a freshly generated world.

    Named after the theme the generator gave it, so `worlds/` reads as a list of
    designs. A world the agent did not actually design keeps the timestamp.
    """
    theme = scene.meta.get("theme") if scene.meta.get("source") == "agent" else None
    stem = (_slugify(theme) if theme else "") or f"procedural-{stamp}"
    # A saved episode points at its world by path, so reusing a name would
    # silently repoint old episodes at different geometry.
    path = WORLDS_DIR / f"{stem}.json"
    nth = 2
    while path.exists():
        path = WORLDS_DIR / f"{stem}-{nth}.json"
        nth += 1
    return path


def _pick_world(config: DictConfig) -> tuple[Scene, Path] | None:
    worlds = _list_json(WORLDS_DIR)
    if not worlds:
        print(f"no worlds found under {WORLDS_DIR}/ — use `generate` (or `run`) first")
        return None
    print(f"[play] {len(worlds)} world(s) available under {WORLDS_DIR}/:")
    for i, path in enumerate(worlds, 1):
        print(f"  {i:>2}. {path.name}")
    path = _prompt_choice(worlds, "play")
    return Scene.load(path, collision_resolution=config.world.collision_resolution), path


def _replay(config: DictConfig) -> int:
    from .render import UrsinaRenderer

    ep = config.episode
    episodes = _list_json(EPISODES_DIR)
    if not episodes:
        print(f"no episodes found under {EPISODES_DIR}/ — use `run` or `play` first")
        return 1
    print(f"[replay] {len(episodes)} episode(s) available under {EPISODES_DIR}/:")
    for i, path in enumerate(episodes, 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verdict = "solved" if payload.get("solved") else "not solved"
            print(f"  {i:>2}. {path.name}  [{verdict}, {payload.get('calls', '?')} calls, "
                  f"on {payload.get('world', '?')}]")
        except (OSError, json.JSONDecodeError):
            print(f"  {i:>2}. {path.name}")
    episode_path = _prompt_choice(episodes, "replay")

    scene, result, policy_name, agent_model, agent_effort = load_episode(
        episode_path, collision_resolution=config.world.collision_resolution)
    sensor = _sensor(config)
    renderer = UrsinaRenderer(scene, sensor=sensor, mesh_resolution=config.world.mesh_resolution)
    try:
        print(f"[replay] {episode_path}: {result.summary()}")
        print("[replay] SPACE play/pause, close the window to quit")
        Replay(scene, renderer, result, policy_name, sensor=sensor, agent_model=agent_model,
               agent_effort=agent_effort, step_delay=ep.step_delay).run()
    finally:
        renderer.close()
    return 0 if result.solved else 1


# -------------------------------------------------------------------- commands


def _dispatch(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # one clean line, not a traceback
        parser.error(f"{CONFIG_PATH}: {exc}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    episode_path = EPISODES_DIR / f"episode-{stamp}.json"

    if args.command == "generate":
        scene = _generate(config, offline=args.offline)
        print(f"[gen] saved to {scene.save(_world_path(scene, stamp))}")
        return 0

    if args.command == "play":
        picked = _pick_world(config)
        if picked is None:
            return 1
        scene, world_path = picked
        print(f"[play] {len(scene.primitives) - 1} primitives, spawn {scene.player.position}")
        return _play(config, scene, world_path, episode_path)

    if args.command == "run":
        scene = _generate(config, offline=args.offline)
        world_path = scene.save(_world_path(scene, stamp))
        print(f"[gen] saved to {world_path}")
        return _play(config, scene, world_path, episode_path, offline=args.offline)

    if args.command == "replay":
        return _replay(config)

    raise AssertionError(f"unhandled command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    """Quitting has to unwind rather than stop the process where it stands: the
    agents interrupt their tmux turn on the way out, and that pane outlives us."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # 128 + SIGTERM
    try:
        return _dispatch(argv)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
