"""Command line entrypoints: `generate`, `play`, `run`, `replay`.

All settings besides the command and `--offline` live in `configs/3d.yaml`, read
fresh on every run — see `.config`.
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

from agents import ClaudeTmuxAgent, CodexTmuxAgent, TmuxAgent
from geometry.superquadrics import Sensor
from navigation import ManualPolicy, Policy, WaypointNavigator
from world import Scene, WorldGenerator
from .config import load_config
from .engine import Budgets, Episode, Replay, load_episode, save_episode
from .logger import Logger, logger

#: Worlds and episodes are separate artifacts: a world is generated once and can
#: be played many times, so every episode only records the path it ran on.
CONFIG_PATH = Path("configs/3d.yaml")
WORLDS_DIR = Path("worlds_3d")
EPISODES_DIR = Path("episodes_3d")

#: Selected by `agent.name`, which comes from the `defaults: - agent:` group in
#: configs/3d.yaml — so one line there swaps every agent call in the harness.
AGENTS = {"claude": ClaudeTmuxAgent, "codex": CodexTmuxAgent}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m engine",
        description="Agent harness that generates a 3D superquadric world and flies it. "
                    f"All other settings live in {CONFIG_PATH}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a world and save it under worlds_3d/")
    gen.add_argument("--offline", action="store_true", help="procedural world, zero agent calls")

    sub.add_parser("play", help="pick a saved world under worlds_3d/ and fly a fresh episode on it")

    run = sub.add_parser("run", help="generate a world under worlds_3d/, then immediately fly it")
    run.add_argument("--offline", action="store_true",
                     help="procedural world plus manual play — zero agent calls")

    sub.add_parser("replay", help="pick a saved episode under episodes_3d/ and replay it")

    return parser


# --------------------------------------------------------------------- helpers


def _log_config(config: DictConfig, command: str, *, offline: bool = False) -> None:
    agent_cfg = config.get("agent", {})
    gen_cfg = config.get("generation", {})
    ep_cfg = config.get("episode", {})

    logger.log(f"command: {command}{' --offline' if offline else ''}", stage="config")
    if not offline and agent_cfg:
        logger.log(
            f"agent: {agent_cfg.get('name', 'agent')} (model={agent_cfg.get('model', '?')}, "
            f"effort={agent_cfg.get('effort', '?')}, timeout={agent_cfg.get('timeout', 900):.0f}s)",
            stage="config",
        )
    if gen_cfg and command in ("generate", "run"):
        brief_str = f", brief='{gen_cfg.brief}'" if gen_cfg.get("brief") else ""
        logger.log(
            f"generation: rooms={gen_cfg.get('rooms', 1)}, levels={gen_cfg.get('levels', 1)}{brief_str}",
            stage="config",
        )
    if ep_cfg and command in ("play", "run"):
        pol = "manual" if offline else ep_cfg.get("policy", "agent")
        max_dist = ep_cfg.get("max_distance", 600.0)
        dist_str = "inf" if max_dist == float("inf") else f"{max_dist:.0f}m"
        logger.log(
            f"episode: policy={pol}, max_calls={ep_cfg.get('max_calls', 40)}, max_distance={dist_str}",
            stage="config",
        )


def _make_agent(cfg: DictConfig, role: str, *, session_name: str | None = None,
                window_name: str | None = None) -> TmuxAgent:
    if cfg.name not in AGENTS:
        raise RuntimeError(f"unknown agent {cfg.name!r} — expected one of {sorted(AGENTS)}")
    return AGENTS[cfg.name](
        model=cfg.model,
        binary=cfg.binary,
        timeout=cfg.timeout,
        max_retries=cfg.retries,
        session_name=session_name or "general-intuition-3d",
        window_name=window_name or role,
        effort=cfg.effort,
    )


def _generate(config: DictConfig, *, offline: bool) -> Scene:
    gen = config.generation
    generator_agent = None if offline else _make_agent(config.agent, role="generator", window_name="layout")
    if generator_agent is not None:
        generator_agent.clean()
    generator = WorldGenerator(
        agent=generator_agent,
        room_agent_factory=None if offline else (
            lambda room, idx: _make_agent(config.agent, role="generator", window_name=f"room{idx}")
        ),
        max_levels=gen.get("levels", 1),
        max_rooms=gen.get("rooms", 1),
    )
    started = time.monotonic()
    # `--offline` ignores the configured brief entirely: it directs agent
    # generation and there is no agent here to direct.
    scene = generator.generate_offline() if offline else generator.generate(gen.get("brief"))
    scene.meta["generation_s"] = round(time.monotonic() - started, 1)
    room0 = scene.meta["layout"]["rooms"][0]
    n_locks = len(scene.meta["task"])
    logger.log(f"{len(scene.primitives)} primitives, {len(scene.meta['layout']['rooms'])} room(s) "
               f"in a {scene.bounds:.0f}-unit building in {scene.meta['generation_s']:.1f}s | "
               f"spawn in '{room0['name']}' | {n_locks} locked door(s) to solve in order", stage="generation")
    return scene


def _make_policy(config: DictConfig, renderer, scene: Scene, name: str) -> Policy:
    ep = config.episode
    if name == "manual":
        return ManualPolicy(renderer, scene)
    if name == "agent":
        agent = _make_agent(config.agent, role="player")
        return WaypointNavigator(
            agent,
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
    logger.log(f"loading scene ({len(scene.primitives)} primitives)...", stage="render")
    sensor = Sensor()
    renderer = UrsinaRenderer(scene, sensor=sensor, mouse_look=policy_name == "manual")
    try:
        policy = _make_policy(config, renderer, scene, policy_name)

        # A human at the controls gets no budget: manual play runs until the key
        # is reached or the window is closed.
        unlimited = policy.name == "manual"
        budgets = Budgets(
            max_calls=math.inf if unlimited else ep.max_calls,
            max_distance=math.inf if unlimited else ep.max_distance,
            max_collisions=math.inf if unlimited else ep.max_collisions,
        )
        initial_scene_data = scene.to_dict()
        result = Episode(scene, policy, sensor=sensor, budgets=budgets, renderer=renderer,
                         step_delay=ep.get("step_delay", 0.0)).run()

        extra = {
            "policy": policy.name,
            "max_collisions": budgets.max_collisions,
            "max_calls": budgets.max_calls,
            "max_distance": budgets.max_distance,
        }
        agent = getattr(policy, "agent", None)
        if agent is not None:
            extra["agent_model"] = agent.model
            extra["agent_effort"] = agent.effort
            extra["agent_stats"] = agent.stats
            extra["agent_failures"] = str(getattr(policy, "failures", 0))
        save_episode(out, world, result, extra)

        logger.log(f"episode written to {out}", stage="play")
        logger.log("window stays open — SPACE play/pause, +/- speed, R restart, close the window to quit", stage="replay")
        renderer.release_mouse()
        replay_scene = Scene.from_dict(initial_scene_data)
        Replay(replay_scene, renderer, result, policy.name, sensor=sensor,
               agent_model=getattr(agent, "model", None),
               agent_effort=getattr(agent, "effort", None), step_delay=ep.get("step_delay", 0.0),
               max_collisions=budgets.max_collisions).run()
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

    Named after the theme the generator gave it, so `worlds_3d/` reads as a list of
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
        logger.log(f"no worlds found under {WORLDS_DIR}/ — use `generate` (or `run`) first", stage="play")
        return None
    logger.log(f"{len(worlds)} world(s) available under {WORLDS_DIR}/:", stage="play")
    for i, path in enumerate(worlds, 1):
        print(f"  {i:>2}. {path.name}")
    path = _prompt_choice(worlds, "play")
    return Scene.load(path), path


def _replay(config: DictConfig) -> int:
    from .render import UrsinaRenderer

    ep = config.episode
    episodes = _list_json(EPISODES_DIR)
    if not episodes:
        logger.log(f"no episodes found under {EPISODES_DIR}/ — use `run` or `play` first", stage="replay")
        return 1
    logger.log(f"{len(episodes)} episode(s) available under {EPISODES_DIR}/:", stage="replay")
    for i, path in enumerate(episodes, 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verdict = "solved" if payload.get("solved") else "not solved"
            print(f"  {i:>2}. {path.name}  [{verdict}, {payload.get('calls', '?')} calls, "
                  f"on {payload.get('world', '?')}]")
        except (OSError, json.JSONDecodeError):
            print(f"  {i:>2}. {path.name}")
    episode_path = _prompt_choice(episodes, "replay")

    scene, result, policy_name, agent_model, agent_effort, max_collisions = load_episode(episode_path)
    logger.log(f"loading scene ({len(scene.primitives)} primitives)...", stage="render")
    sensor = Sensor()
    renderer = UrsinaRenderer(scene, sensor=sensor)
    try:
        logger.log(f"{episode_path.name}: {result.summary()}", stage="replay")
        logger.log("SPACE play/pause, +/- speed, R restart, close the window to quit", stage="replay")
        Replay(scene, renderer, result, policy_name, sensor=sensor, agent_model=agent_model,
               agent_effort=agent_effort, step_delay=ep.get("step_delay", 0.0),
               max_collisions=max_collisions).run()
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

    _log_config(config, args.command, offline=getattr(args, "offline", False))

    if args.command == "generate":
        scene = _generate(config, offline=args.offline)
        world_path = scene.save(_world_path(scene, stamp))
        logger.log(f"saved to {world_path}", stage="generation")
        return 0

    if args.command == "play":
        picked = _pick_world(config)
        if picked is None:
            return 1
        scene, world_path = picked
        logger.log(f"{len(scene.primitives)} primitives, spawn {scene.player.position.rounded(1)}", stage="play")
        if config.episode.policy == "agent":
            TmuxAgent.clean_session_windows("general-intuition-3d")
        return _play(config, scene, world_path, episode_path)

    if args.command == "run":
        scene = _generate(config, offline=args.offline)
        world_path = scene.save(_world_path(scene, stamp))
        logger.log(f"saved to {world_path}", stage="generation")
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
