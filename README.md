# Infinite Environment Generation via an Agent Harness

An agent harness that **builds a 3D superquadric environment and then plays it**. Both halves
use the same abstraction — `Agent.run(prompt, schema) -> structured_output`, a wrapper
around an interactive coding CLI (`claude` or `codex`) driven inside tmux — checked by
geometry and collision code (`Scene`, `SuperquadricHandler`), never by looking at pixels.

Task spec: [`TASK.md`](./TASK.md) · conventions and architecture notes for
agents: [`AGENTS.md`](./AGENTS.md).

## Run it

```bash
uv sync                              # Python >=3.12

python -m engine generate --offline  # procedural building, zero agent calls
python -m engine run --offline       # procedural building + manual flight (smoke test)
python -m engine run                 # agent generates, agent flies
python -m engine play                # pick a saved world under worlds_3d/
python -m engine replay              # re-watch a saved episode
```

The CLI only takes a command (`generate` / `play` / `run` / `replay`) and `--offline`; everything
else — agent settings, world bounds, episode options, `policy: agent|manual` —
lives in [`configs/3d.yaml`](./configs/3d.yaml), read fresh on every run.

**Choosing the agent.** [`configs/3d.yaml`](./configs/3d.yaml) picks one with a
Hydra config group:

```yaml
defaults:
  - agent: codex    # or: claude
```

That composes [`configs/agent/claude.yaml`](./configs/agent/claude.yaml) or
[`configs/agent/codex.yaml`](./configs/agent/codex.yaml) — binary, model, effort — into
`agent`, alongside the agent-agnostic `timeout` / `retries` in the main config. Both are
driven interactively inside persistent tmux sessions; there is no non-interactive backend.

**Tmux.** All agents run inside a single persistent `tmux` session — `general-intuition-3d` —
with dedicated windows for each step (`layout`, `room1`, `room2`, ..., `player`).

The harness logs the attach command when it spawns a window; paste it into another
terminal to watch that agent live:
```bash
tmux attach -f ignore-size -t general-intuition-3d:layout
```

The window is pinned to 200x50 regardless of your terminal size, since the CLIs hard-wrap
JSON to the pane width.

**Worlds and episodes.** Worlds and episodes are separate artifacts:
- `generate` and `run` write worlds to `worlds_3d/<theme>.json` (or `worlds_3d/procedural-<timestamp>.json`).
- `python -m engine play` lists saved worlds under `worlds_3d/` and runs an episode on the one you pick.
- Episodes write to `episodes_3d/episode-<timestamp>.json` holding the world path and full trajectory.
- `python -m engine replay` replays a saved episode.

## How it fits together

```
                 ┌─────────────────────────────────────────┐
  layout+room  ──┤  Agent.run(prompt, schema) -> JSON      │──  claude / codex
  waypoints    ──┤  (TmuxAgent, agents/)                   │    in a tmux pane
                 └─────────────────────────────────────────┘

  generation.py ──> Scene ──> Episode ──> State (params) ──> policy ──> Trajectory(waypoints)
                     ▲          │                                           │
                     └──────────┴── fly increment by increment, halt on collision ┘
```

Two generation calls:
1. **Layout**: designs a floor plan (rooms, levels, connections); harness validates and repairs it, builds sealed rooms from superquadrics, and cuts doorways along the task path.
2. **Furnishing**: furnishes rooms with superquadrics. Harness places a lock on each room's door along the task path, plus that room's matching peg and decoys — the agent solves one room at a time to progress.

`superquadrics.py` meshes primitives in pure Python and owns collision, contact, and sensor visibility; [Ursina](https://www.ursinaengine.org/) renders those meshes and reads input, but is never asked a question about the world.

Movement is continuous: the navigation agent returns a batch of actions — `move` (fly to a target, camera turning to face it, straight segment flown in increments, halting at the first collision), `pick` (pick up a nearby object), `place` (place the carried object, or attempt the lock if placed at/near the door) — executed in order. Manual flight is `WASD` + `Space`/`Shift` + mouse look + `E` to pick/place (`V` toggles third-person inspection camera).

**Observation:** The navigation agent receives validated *parameters* of superquadrics detected by its sensor (camera frustum + line of sight, no range limit) plus its own pose, rather than pixels.

## Observations
- 3D Harness (Codex):
  - World generation works: follows the brief and generates plausible worlds in superquadrics.
  - Systematic exploration: closed circular loop at different levels to scan the whole volume, then navigates to target once observed.
  - codex uses an average of 3.4 actions per call (move, pick, place), even though limit is 8

## Future Work
Make room history important. E.g., the last room requires an element from the very first to be completed, so agent can remember everything and navigate back.