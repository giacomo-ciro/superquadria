"""Command line entrypoints: `generate`, `play`, `run`."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .agents.claude_cli import ClaudeSubprocessAgent
from .engine import Episode, save_run
from .generation import MazeGenerator
from .maze_utils import bfs_path
from .navigation import ClaudeNavigator
from .policies import FrontierPolicy, Policy
from .render import make_renderer
from .scene import Scene

DEFAULT_MODEL = "claude-sonnet-5"


# ------------------------------------------------------------------- arguments


def _default_render() -> str:
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    return "window" if has_display else "terminal"


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("agent")
    group.add_argument("--model", default=DEFAULT_MODEL, help="model id passed to `claude --model`")
    group.add_argument("--claude-bin", default="claude", help="path to the claude CLI")
    group.add_argument("--timeout", type=float, default=600.0, help="seconds per agent call")
    group.add_argument("--retries", type=int, default=2, help="retries per failed agent call")
    group.add_argument("--no-skip-permissions", action="store_true",
                       help="omit --dangerously-skip-permissions (needs an interactive approval)")


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("generation")
    group.add_argument("--width", type=int, default=100)
    group.add_argument("--height", type=int, default=100)
    group.add_argument("--band", type=int, default=10, help="rows per generation batch")
    group.add_argument("--workers", type=int, default=5, help="bands generated concurrently")
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--min-key-distance", type=int, default=None,
                       help="minimum Manhattan spawn-to-key distance "
                            "(default: (width + height) // 3)")
    group.add_argument("--offline", action="store_true",
                       help="procedural maze, zero agent calls")
    group.add_argument("--brief", default=None,
                       help="optional text command seeding the design step "
                            "(the MVP lets the agent decide for itself)")


def _add_play_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("episode")
    group.add_argument("--policy", choices=["claude", "frontier"], default="claude")
    group.add_argument("--max-steps", type=int, default=2000)
    group.add_argument("--max-calls", type=int, default=80)
    group.add_argument("--render", choices=["window", "headless", "terminal", "none"],
                       default=_default_render())
    group.add_argument("--reveal", action="store_true", help="god view instead of fog of war")
    group.add_argument("--cell", type=int, default=6, help="pixels per grid cell")
    group.add_argument("--fps", type=int, default=30)
    group.add_argument("--step-delay", type=float, default=0.0, help="seconds per executed step")
    group.add_argument("--frame-dir", default=None, help="write a PNG per frame here")
    group.add_argument("--run-dir", default=None, help="where to write scene.json/episode.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maze-harness",
        description="Agent harness that generates a 2D maze and navigates it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a maze and save it")
    _add_generate_args(gen)
    _add_agent_args(gen)
    gen.add_argument("--out", default=None, help="output scene JSON path")

    play = sub.add_parser("play", help="run an episode on a saved maze")
    play.add_argument("--maze", required=True, help="scene JSON produced by `generate`")
    _add_play_args(play)
    _add_agent_args(play)

    run = sub.add_parser("run", help="generate a maze, then immediately play it")
    _add_generate_args(run)
    _add_play_args(run)
    _add_agent_args(run)
    return parser


# --------------------------------------------------------------------- helpers


def _make_agent(args) -> ClaudeSubprocessAgent:
    return ClaudeSubprocessAgent(
        model=args.model,
        binary=args.claude_bin,
        timeout=args.timeout,
        max_retries=args.retries,
        skip_permissions=not args.no_skip_permissions,
    )


def _generate(args) -> Scene:
    generator = MazeGenerator(
        agent=None if args.offline else _make_agent(args),
        width=args.width,
        height=args.height,
        band_height=args.band,
        workers=args.workers,
        seed=args.seed,
        min_key_distance=args.min_key_distance,
    )
    started = time.monotonic()
    scene = generator.generate_offline() if args.offline else generator.generate(args.brief)
    elapsed = time.monotonic() - started

    path = bfs_path(scene.terrain, scene.player.position, scene.key.position)
    if path is None:  # connect_regions guarantees this cannot happen; assert it anyway
        raise RuntimeError("generated maze is unsolvable — connectivity repair failed")
    scene.meta["shortest_path"] = len(path)
    scene.meta["generation_s"] = round(elapsed, 1)
    print(f"[gen] {scene.height}x{scene.width} maze in {elapsed:.1f}s | "
          f"spawn {scene.player.position} -> key {scene.key.position} | "
          f"shortest path {len(path)} steps")
    return scene


def _make_policy(args) -> Policy:
    if args.policy == "frontier":
        return FrontierPolicy(seed=getattr(args, "seed", None))
    return ClaudeNavigator(_make_agent(args), step_budget=args.max_steps)


def _play(args, scene: Scene, run_dir: Path) -> int:
    policy = _make_policy(args)
    renderer_kwargs = {}
    if args.render in ("window", "headless"):
        renderer_kwargs = {
            "cell": args.cell,
            "reveal": args.reveal,
            "fps": args.fps,
            "frame_dir": args.frame_dir,
        }
    renderer = make_renderer(args.render, scene, **renderer_kwargs)

    episode = Episode(
        scene,
        policy,
        max_steps=args.max_steps,
        max_calls=args.max_calls,
        renderer=renderer,
        step_delay=args.step_delay,
    )
    try:
        result = episode.run()
    finally:
        renderer.close()

    extra = {"policy": policy.name}
    agent = getattr(policy, "agent", None)
    if agent is not None:
        extra["agent_stats"] = agent.stats
        extra["agent_failures"] = getattr(policy, "failures", 0)
    save_run(run_dir, scene, result, extra)

    print(f"\n{result.summary()}")
    print(f"optimal was {scene.meta.get('shortest_path', '?')} steps; run written to {run_dir}")
    return 0 if result.solved else 1


# -------------------------------------------------------------------- commands


def main(argv: list[str] | None = None) -> int:
    # Runs are long and mostly spent waiting on subprocesses; keep the log live
    # even when stdout is a pipe or a log file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.command == "generate":
        scene = _generate(args)
        out = Path(args.out or f"mazes/maze-{stamp}.json")
        scene.save(out)
        print(f"[gen] saved to {out}")
        return 0

    if args.command == "play":
        scene = Scene.load(args.maze)
        if "shortest_path" not in scene.meta:
            path = bfs_path(scene.terrain, scene.player.position, scene.key.position)
            scene.meta["shortest_path"] = len(path) if path is not None else None
        print(f"[play] {args.maze}: {scene.height}x{scene.width}, "
              f"spawn {scene.player.position} -> key {scene.key.position}")
        return _play(args, scene, Path(args.run_dir or f"runs/{stamp}"))

    if args.command == "run":
        run_dir = Path(args.run_dir or f"runs/{stamp}")
        scene = _generate(args)
        scene.save(run_dir / "scene.json")
        return _play(args, scene, run_dir)

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
