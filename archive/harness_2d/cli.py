"""Command line entrypoints: `generate`, `play`, `run`.

All settings besides the command and `--offline` live in `configs/2d.yaml`, read
fresh on every run — see `harness_common.config`.
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

from harness_common.agents.claude_tmux import ClaudeTmuxAgent
from harness_common.agents.codex_tmux import CodexTmuxAgent
from harness_common.agents.tmux_agent import TmuxAgent
from harness_common.config import load_config

from .engine import Episode, Replay, load_episode, save_episode
from .generation import MazeGenerator
from .maze_utils import bfs_path
from .navigation import ClaudeNavigator
from .policies import FrontierPolicy, ManualPolicy, Policy
from .render import PygameRenderer
from .scene import Scene

#: Mazes and episodes are separate artifacts: a maze is generated once and can
#: be played many times, so every episode only records the path it ran on.
CONFIG_PATH = Path("configs/2d.yaml")
WORLDS_DIR = Path("worlds_2d")
EPISODES_DIR = Path("episodes_2d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness_2d",
        description="Agent harness that generates a 2D maze and navigates it. "
                     f"All other settings live in {CONFIG_PATH}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a maze and save it under worlds_2d/")
    gen.add_argument("--offline", action="store_true", help="procedural maze, zero agent calls")

    sub.add_parser("play", help="pick a saved maze under worlds_2d/ and play a fresh episode on it")

    run = sub.add_parser("run", help="generate a maze under worlds_2d/, then immediately play it")
    run.add_argument("--offline", action="store_true", help="procedural maze, zero agent calls")

    sub.add_parser("replay", help="pick a saved episode under episodes_2d/ and replay it in the pygame window")

    return parser


# --------------------------------------------------------------------- helpers


#: Selected by `agent.name`, which comes from the `defaults: - agent:` group in
#: configs/2d.yaml — so one line there swaps every agent call in the harness.
AGENTS = {"claude": ClaudeTmuxAgent, "codex": CodexTmuxAgent}


def _make_agent(cfg: DictConfig, role: str = "agent") -> TmuxAgent:
    if cfg.name not in AGENTS:
        raise RuntimeError(f"unknown agent {cfg.name!r} — expected one of {sorted(AGENTS)}")
    return AGENTS[cfg.name](
        model=cfg.model,
        binary=cfg.binary,
        timeout=cfg.timeout,
        max_retries=cfg.retries,
        # One session per role, not one session with a window per role: every client
        # attached to a session is forced to that session's single current window, so
        # two roles sharing one session cannot be watched side by side.
        session_name=f"general-intuition-{role}",
        window_name=role,
        effort=cfg.effort,
    )


def _generate(config: DictConfig, *, offline: bool) -> Scene:
    gen = config.generation
    generator = MazeGenerator(
        agent=None if offline else _make_agent(config.agent, role="generator"),
        side_length=gen.side_length,
        view_size=config.episode.view_size,
    )
    started = time.monotonic()
    scene = generator.generate_offline() if offline else generator.generate(gen.brief)
    elapsed = time.monotonic() - started

    path = bfs_path(scene.terrain, scene.player.position, scene.key.position)
    if path is None:  # no connectivity repair: an ungenerous model can produce this
        raise RuntimeError("generated maze is unsolvable — spawn and key are disconnected")
    scene.meta["shortest_path"] = len(path)
    scene.meta["generation_s"] = round(elapsed, 1)
    print(f"[gen] {scene.height}x{scene.width} maze in {elapsed:.1f}s | "
          f"spawn {scene.player.position} -> key {scene.key.position} | "
          f"shortest path {len(path)} steps")
    return scene


def _make_policy(config: DictConfig, renderer) -> Policy:
    ep = config.episode
    if ep.policy == "frontier":
        return FrontierPolicy()
    if ep.policy == "manual":
        return ManualPolicy(renderer)
    if ep.policy == "agent":
        return ClaudeNavigator(_make_agent(config.agent, role="player"), step_budget=ep.max_steps)
    raise RuntimeError(f"unknown episode.policy {ep.policy!r} — expected agent, frontier or manual")


def _play(config: DictConfig, scene: Scene, maze: Path, out: Path) -> int:
    ep = config.episode
    renderer = PygameRenderer(scene)
    try:
        policy = _make_policy(config, renderer)

        # A human at the keyboard gets no budget: manual play runs until the key
        # is reached or the window is closed.
        unlimited = policy.name == "manual"

        episode = Episode(
            scene,
            policy,
            max_steps=math.inf if unlimited else ep.max_steps,
            max_calls=math.inf if unlimited else ep.max_calls,
            renderer=renderer,
            step_delay=ep.step_delay,
            view_size=ep.view_size,
        )
        result = episode.run()

        extra = {"policy": policy.name}
        agent = getattr(policy, "agent", None)
        if agent is not None:
            extra["agent_model"] = agent.model
            extra["agent_effort"] = agent.effort
            extra["agent_stats"] = agent.stats
            extra["agent_failures"] = str(getattr(policy, "failures", 0))
        save_episode(out, maze, result, extra)

        print(f"\n{result.summary()}")
        print(f"optimal was {scene.meta.get('shortest_path', '?')} steps; episode written to {out}")

        print("[replay] window stays open — SPACE play/pause, close the window to quit")
        Replay(scene, renderer, result, policy.name,
               agent_model=getattr(agent, "model", None), agent_effort=getattr(agent, "effort", None),
               memory=getattr(policy, "memory", None), step_delay=ep.step_delay,
               view_size=ep.view_size).run()
    finally:
        renderer.close()

    return 0 if result.solved else 1


def _list_json(root: Path) -> list[Path]:
    """Saved `.json` artifacts under `root`, newest-looking name first."""
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), key=lambda p: p.name, reverse=True)


def _prompt_choice(paths: list[Path], verb: str) -> Path:
    while True:
        choice = input(f"{verb} which one? [1-{len(paths)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(paths):
            return paths[int(choice) - 1]
        print("invalid choice")


def _replay(config: DictConfig) -> int:
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
            print(f"  {i:>2}. {path.name}  [{verdict}, {payload.get('steps', '?')} steps, "
                  f"on {payload.get('maze', '?')}]")
        except (OSError, json.JSONDecodeError):
            print(f"  {i:>2}. {path.name}")
    episode_path = _prompt_choice(episodes, "replay")

    scene, result, policy_name, agent_model, agent_effort = load_episode(episode_path)
    renderer = PygameRenderer(scene)
    try:
        print(f"[replay] {episode_path}: {result.summary()}")
        print("[replay] SPACE play/pause, close the window to quit")
        Replay(scene, renderer, result, policy_name,
               agent_model=agent_model, agent_effort=agent_effort,
               step_delay=ep.step_delay, view_size=ep.view_size).run()
    finally:
        renderer.close()

    return 0 if result.solved else 1


def _slugify(text: str) -> str:
    """'The Pillared Hall' -> 'the-pillared-hall'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _maze_path(scene: Scene, stamp: str) -> Path:
    """Where to write a freshly generated maze.

    Named after the theme the generator gave it, so `worlds_2d/` reads as a list of
    designs rather than of timestamps. Terrain the agent did not actually draw —
    an offline run, or a failed agent call falling back to `procedural_maze` —
    has no real name to take, so it stays on the timestamp.
    """
    theme = scene.meta.get("theme") if scene.meta.get("source") == "agent" else None
    stem = (_slugify(theme) if theme else "") or f"procedural-{stamp}"
    # Two runs can land on the same theme, and a saved episode points at its
    # maze by path — reusing a name would silently repoint old episodes at
    # different terrain, so disambiguate instead of overwriting.
    path = WORLDS_DIR / f"{stem}.json"
    nth = 2
    while path.exists():
        path = WORLDS_DIR / f"{stem}-{nth}.json"
        nth += 1
    return path


def _pick_maze() -> tuple[Scene, Path] | None:
    """Let the user pick a previously generated maze from worlds_2d/. Maze files are
    written before any episode touches them, so the player is at its spawn."""
    mazes = _list_json(WORLDS_DIR)
    if not mazes:
        print(f"no mazes found under {WORLDS_DIR}/ — use `generate` (or `run`) first")
        return None
    print(f"[play] {len(mazes)} maze(s) available under {WORLDS_DIR}/:")
    for i, path in enumerate(mazes, 1):
        print(f"  {i:>2}. {path.name}")
    path = _prompt_choice(mazes, "play")
    return Scene.load(path), path


# -------------------------------------------------------------------- commands


def _dispatch(argv: list[str] | None = None) -> int:
    # Runs are long and mostly spent waiting on subprocesses; keep the log live
    # even when stdout is a pipe or a log file.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # malformed/missing config.yaml — fail with one clean line, not a traceback
        parser.error(f"{CONFIG_PATH}: {exc}")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    episode_path = EPISODES_DIR / f"episode-{stamp}.json"

    if args.command == "generate":
        scene = _generate(config, offline=args.offline)
        maze_path = scene.save(_maze_path(scene, stamp))
        print(f"[gen] saved to {maze_path}")
        return 0

    if args.command == "play":
        picked = _pick_maze()
        if picked is None:
            return 1
        scene, maze_path = picked
        if "shortest_path" not in scene.meta:
            path = bfs_path(scene.terrain, scene.player.position, scene.key.position)
            scene.meta["shortest_path"] = len(path) if path is not None else None
        print(f"[play] {scene.height}x{scene.width}, "
              f"spawn {scene.player.position} -> key {scene.key.position}")
        return _play(config, scene, maze_path, episode_path)

    if args.command == "run":
        scene = _generate(config, offline=args.offline)
        maze_path = scene.save(_maze_path(scene, stamp))
        print(f"[gen] saved to {maze_path}")
        return _play(config, scene, maze_path, episode_path)

    if args.command == "replay":
        return _replay(config)

    raise AssertionError(f"unhandled command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    """Quitting has to unwind, not stop the process where it stands: the agents
    interrupt their tmux turn on the way out, and that pane outlives us, so a turn
    left running would keep burning tokens on an answer nobody will read."""
    # SIGTERM (`kill`, a closing terminal) exits by default without raising, which
    # would skip that cleanup entirely. Turn it into an exception so it unwinds
    # the same way Ctrl+C does.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # 128 + SIGTERM
    try:
        return _dispatch(argv)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
