"""Command line entrypoints: `generate`, `play`, `run`, `replay`.

All settings live in `configs/config.yaml`, read fresh on every run — see `.config`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import signal
import sys
import time
from pathlib import Path

from omegaconf import DictConfig

from agents import Agent, Claude, Codex
from geometry.superquadrics import Sensor
from logger import logger
from navigation import ManualPolicy, Policy, WaypointNavigator
from world import Scene, WorldGenerator

from .config import load_config
from .engine import Budgets, Episode, Replay, load_episode, save_episode

#: Worlds and episodes are separate artifacts: a world is generated once and can
#: be played many times, so every episode only records the path it ran on.
CONFIG_PATH = Path("configs/config.yaml")
WORLDS_DIR = Path("worlds")
EPISODES_DIR = Path("episodes")

#: Selected by `agent.name`, which comes from the `defaults: - agent:` group in
#: configs/config.yaml — so one line there swaps every agent call in the harness.
AGENTS = {"claude": Claude, "codex": Codex}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m engine",
        description="Agent harness that generates a 3D superquadric world and "
        f"flies it. All other settings live in {CONFIG_PATH}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="generate a world and save it under worlds/")

    sub.add_parser(
        "play", help="pick a saved world under worlds/ and fly a fresh episode on it"
    )

    sub.add_parser(
        "run", help="generate a world under worlds/, then immediately fly it"
    )

    sub.add_parser("replay", help="pick a saved episode under episodes/ and replay it")

    return parser


# --------------------------------------------------------------------- helpers


def _log_config(config: DictConfig, command: str) -> None:
    agent_cfg = config.get("agent", {})
    gen_cfg = config.get("generation", {})
    ep_cfg = config.get("episode", {})

    logger.log(f"command: {command}", stage="config")
    if agent_cfg:
        logger.log(
            f"agent: {agent_cfg.get('name', 'agent')} "
            f"(model={agent_cfg.get('model', '?')}, "
            f"effort={agent_cfg.get('effort', '?')}, "
            f"timeout={agent_cfg.get('timeout', 900):.0f}s)",
            stage="config",
        )
    if gen_cfg and command in ("generate", "run"):
        brief_str = f", brief='{gen_cfg.brief}'" if gen_cfg.get("brief") else ""
        logger.log(
            f"generation: rooms={gen_cfg.get('rooms', 1)}, "
            f"levels={gen_cfg.get('levels', 1)}{brief_str}",
            stage="config",
        )
    if ep_cfg and command in ("play", "run"):
        pol = ep_cfg.get("policy", "agent")
        max_dist = ep_cfg.get("max_distance", 600.0)
        dist_str = "inf" if max_dist == float("inf") else f"{max_dist:.0f}m"
        logger.log(
            f"episode: policy={pol}, max_calls={ep_cfg.get('max_calls', 40)}, "
            f"max_distance={dist_str}",
            stage="config",
        )


def _make_agent(
    cfg: DictConfig,
    role: str,
    *,
    session_name: str | None = None,
    window_name: str | None = None,
) -> Agent:
    if cfg.name not in AGENTS:
        raise RuntimeError(
            f"unknown agent {cfg.name!r} — expected one of {sorted(AGENTS)}"
        )
    return AGENTS[cfg.name](
        model=cfg.model,
        binary=cfg.binary,
        timeout=cfg.timeout,
        max_retries=cfg.retries,
        session_name=session_name or "superquadria",
        window_name=window_name or role,
        effort=cfg.effort,
    )


def _generate(config: DictConfig) -> Scene:
    gen = config.generation
    generator_agent = _make_agent(config.agent, role="generator", window_name="layout")
    generator_agent.clean()
    generator = WorldGenerator(
        agent=generator_agent,
        room_agent_factory=(
            lambda room, idx: _make_agent(
                config.agent, role="generator", window_name=f"room{idx}"
            )
        ),
        max_levels=gen.get("levels", 1),
        max_rooms=gen.get("rooms", 1),
    )
    started = time.monotonic()
    scene = generator.generate(gen.get("brief"))
    scene.meta["generation_s"] = round(time.monotonic() - started, 1)
    room0 = scene.meta["layout"]["rooms"][0]
    n_locks = len(scene.meta["task"])
    logger.log(
        f"{len(scene.primitives)} primitives, "
        f"{len(scene.meta['layout']['rooms'])} room(s) "
        f"in a {scene.bounds:.0f}-unit building in {scene.meta['generation_s']:.1f}s | "
        f"spawn in '{room0['name']}' | {n_locks} locked door(s) to solve in order",
        stage="generation",
    )
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


def _play(config: DictConfig, scene: Scene, world: Path, out: Path) -> int:
    from .render import (
        UrsinaRenderer,  # imported here so `generate` never touches Ursina
    )

    ep = config.episode
    policy_name = ep.policy
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
        result = Episode(
            scene,
            policy,
            sensor=sensor,
            budgets=budgets,
            renderer=renderer,
            step_delay=ep.get("step_delay", 0.0),
        ).run()

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
        logger.log(
            "window stays open — SPACE play/pause, +/- speed, R restart, "
            "close the window to quit",
            stage="replay",
        )
        renderer.release_mouse()
        replay_scene = Scene.from_dict(initial_scene_data)
        Replay(
            replay_scene,
            renderer,
            result,
            policy.name,
            sensor=sensor,
            agent_model=getattr(agent, "model", None),
            agent_effort=getattr(agent, "effort", None),
            step_delay=ep.get("step_delay", 0.0),
            max_collisions=budgets.max_collisions,
        ).run()
        return 0 if result.solved else 1
    finally:
        renderer.close()


def _list_json(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        root.glob("*.json"),
        key=lambda p: [
            int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)
        ],
    )


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
    designs.
    """
    theme = scene.meta.get("theme")
    stem = (_slugify(theme) if theme else "") or f"world-{stamp}"
    # A saved episode points at its world by path, so reusing a name would
    # silently repoint old episodes at different geometry.
    path = WORLDS_DIR / f"{stem}.json"
    nth = 2
    while path.exists():
        path = WORLDS_DIR / f"{stem}-{nth}.json"
        nth += 1
    return path


def _world_name(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        theme = (
            meta.get("theme") or meta.get("layout", {}).get("theme") or meta.get("name")
        )
        if theme and str(theme).strip():
            return str(theme).strip()
    except (OSError, json.JSONDecodeError):
        pass
    return path.name


def _pick_world(config: DictConfig) -> tuple[Scene, Path] | None:
    worlds = _list_json(WORLDS_DIR)
    if not worlds:
        logger.log(
            f"no worlds found under {WORLDS_DIR}/ — use `generate` (or `run`) first",
            stage="play",
        )
        return None
    logger.log(f"{len(worlds)} world(s) available under {WORLDS_DIR}/:", stage="play")
    for i, path in enumerate(worlds, 1):
        print(f"  {i:>2}. {_world_name(path)}")
    path = _prompt_choice(worlds, "play")
    return Scene.load(path), path


def _replay(config: DictConfig) -> int:
    from .render import UrsinaRenderer

    ep = config.episode
    episodes = _list_json(EPISODES_DIR)
    if not episodes:
        logger.log(
            f"no episodes found under {EPISODES_DIR}/ — use `run` or `play` first",
            stage="replay",
        )
        return 1
    logger.log(
        f"{len(episodes)} episode(s) available under {EPISODES_DIR}/:", stage="replay"
    )
    for i, path in enumerate(episodes, 1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            verdict = "solved" if payload.get("solved") else "not solved"
            world_entry = payload.get("world", "?")
            world_name = _world_name(Path(world_entry)) if world_entry != "?" else "?"
            calls = payload.get("calls", "?")
            print(f"  {i:>2}. {path.name}  [{verdict}, {calls} calls, on {world_name}]")
        except (OSError, json.JSONDecodeError):
            print(f"  {i:>2}. {path.name}")
    episode_path = _prompt_choice(episodes, "replay")

    scene, result, policy_name, agent_model, agent_effort, max_collisions = (
        load_episode(episode_path)
    )
    logger.log(f"loading scene ({len(scene.primitives)} primitives)...", stage="render")
    sensor = Sensor()
    renderer = UrsinaRenderer(scene, sensor=sensor)
    try:
        logger.log(f"{episode_path.name}: {result.summary()}", stage="replay")
        logger.log(
            "close the window to quit",
            stage="replay",
        )
        Replay(
            scene,
            renderer,
            result,
            policy_name,
            sensor=sensor,
            agent_model=agent_model,
            agent_effort=agent_effort,
            step_delay=ep.get("step_delay", 0.0),
            max_collisions=max_collisions,
        ).run()
    finally:
        renderer.close()
    return 0 if result.solved else 1


# -------------------------------------------------------------------- commands


def _check_agent_available(config: DictConfig, command: str) -> str | None:
    """None if `command` doesn't need a live agent CLI, or if the configured one
    is on PATH. Otherwise the error message to show.

    Without this, a missing `tmux` or agent binary surfaces minutes later as a
    string of retried tmux failures, ending in either a logged one-liner
    (`generate`) or an uncaught traceback (`play`/`run`) — see `Agent.run`.
    """
    needs_agent = command in ("generate", "run") or (
        command == "play" and config.episode.policy == "agent"
    )
    if not needs_agent:
        return None
    if shutil.which("tmux") is None:
        return "tmux is required but was not found on PATH"
    binary = config.agent.binary
    if shutil.which(binary) is None:
        return (
            f"agent CLI '{binary}' (set via `agent` in {CONFIG_PATH}) was not "
            "found on PATH — install it or switch to the other agent"
        )
    return None


def _dispatch(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # one clean line, not a traceback
        parser.error(f"{CONFIG_PATH}: {exc}")
    if (err := _check_agent_available(config, args.command)) is not None:
        parser.error(err)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    episode_path = EPISODES_DIR / f"episode-{stamp}.json"

    _log_config(config, args.command)

    if args.command == "generate":
        try:
            scene = _generate(config)
        except Exception as exc:
            logger.log(f"generation failed: {exc}", stage="generation")
            return 1
        world_path = scene.save(_world_path(scene, stamp))
        logger.log(f"saved to {world_path}", stage="generation")
        return 0

    if args.command == "play":
        picked = _pick_world(config)
        if picked is None:
            return 1
        scene, world_path = picked
        logger.log(
            f"{len(scene.primitives)} primitives, "
            f"spawn {scene.player.position.rounded(1)}",
            stage="play",
        )
        if config.episode.policy == "agent":
            Agent.clean_session_windows("superquadria")
        return _play(config, scene, world_path, episode_path)

    if args.command == "run":
        try:
            scene = _generate(config)
        except Exception as exc:
            logger.log(f"generation failed: {exc}", stage="generation")
            return 1
        world_path = scene.save(_world_path(scene, stamp))
        logger.log(f"saved to {world_path}", stage="generation")
        return _play(config, scene, world_path, episode_path)

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
