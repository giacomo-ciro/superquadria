# Infinite Environment Generation via an Agent Harness

An agent harness that **builds a 2D maze environment and then plays it**. Both halves
are the same abstraction — `Agent.run(prompt, schema) -> structured_output`, a wrapper
around an interactive coding CLI (`claude` or `codex`) driven inside tmux — checked by
code (position/terrain lookups in `Scene`), never by looking at pixels.

Task spec: [`TASK.md`](./TASK.md) · design: [`docs/2D_PLAN.md`](./docs/2D_PLAN.md) and
[`docs/3D_PLAN.md`](./docs/3D_PLAN.md) · conventions and architecture notes for
agents: [`AGENTS.md`](./AGENTS.md).

## Run it

```bash
uv sync                                        # Python >=3.12

python main.py run --offline                   # full loop, no API calls (smoke test)
python main.py run                             # agent generates, agent plays
```

The CLI only takes a command (`generate` / `play` / `run`) and `--offline`; everything
else — agent settings, maze size, episode options, `policy: agent|frontier` —
lives in [`configs/`](./configs/), read fresh on every run. Edit it directly, e.g. set
`episode.policy: frontier` for the zero-API-call baseline.

**Choosing the agent.** [`configs/config.yaml`](./configs/config.yaml) picks one with a
Hydra config group:

```yaml
defaults:
  - agent: claude    # or: codex
```

That composes [`configs/agent/claude.yaml`](./configs/agent/claude.yaml) or
[`configs/agent/codex.yaml`](./configs/agent/codex.yaml) — binary, model, effort — into
`agent`, alongside the agent-agnostic `timeout` / `retries` in the main config. Both are
driven interactively; there is no non-interactive backend.

**Tmux.** Each agent gets its own `tmux` session — `general-intuition-generator` and
`general-intuition-player` — because every client attached to a session is forced to that
session's current window, so one session per role is what lets you watch both at once.

The harness logs the attach command when it spawns a session; paste it into another
terminal to watch that agent live:
```bash
tmux attach -f ignore-size -t general-intuition-generator
```
`ignore-size` leaves the window's pinned dimensions unchanged, so the CLIs' wrapped JSON
is captured correctly. The session is interactive: use tmux copy mode (or the mouse, when
enabled) to scroll, but do not type or paste into the CLI composer. `Ctrl+b d` detaches.

The window is pinned to 200x50 regardless of your terminal size, since the CLIs hard-wrap
JSON to the pane width — so a smaller terminal sees a clipped view rather than resizing it.

In pieces (generation is the slow part, so generate once and play it many times while
iterating on navigation): `python main.py generate` writes `mazes/<theme>.json`, named
after the theme the generator gave it (`mazes/the-pillared-hall.json`); mazes the agent
did not design — `--offline`, or a failed generation call — are `mazes/procedural-<timestamp>.json`.
Then `python main.py play` lists what is in `mazes/` and runs a fresh episode on the one
you pick (toggle `episode.policy: frontier` there for the zero-API-call baseline).

Mazes and episodes are separate artifacts. `generate` and `run` both save the maze to
`mazes/`; every episode writes a single `episodes/episode-<timestamp>.json` holding the
`maze` path it was played on plus the trajectory (every plan, what executed, where it
collided). `python main.py replay` picks one of those and replays it.

## How it fits together

```
                 ┌─────────────────────────────────────────┐
  design+maze  ──┤  Agent.run(prompt, schema) -> JSON      │──  claude / codex
  each move    ──┤  (TmuxAgent, agents/)                   │    in a tmux pane
                 └─────────────────────────────────────────┘

  generation.py ──> Scene ──> Episode ──> State (10x10) ──> policy ──> Move(actions)
                     ▲          │                                           │
                     └──────────┴── execute step by step, halt on collision ┘
```

## The 3D harness

`harness_3d` is the sibling that does the same thing in three dimensions: an agent
designs a world made **entirely of superquadrics**, and an agent flies through it
looking for a key. Design: [`docs/3D_PLAN.md`](./docs/3D_PLAN.md).

```bash
python -m harness_3d generate --offline   # procedural world, zero agent calls
python -m harness_3d run --offline        # procedural world + manual flight (smoke test)
python -m harness_3d run                  # agent generates, agent flies
python -m harness_3d play                 # pick a saved world under worlds/
python -m harness_3d replay               # re-watch a saved episode
```

Settings live in [`configs/3d.yaml`](./configs/3d.yaml) — same loader, same `agent`
group, so `defaults: - agent: codex` picks the CLI here too. Worlds are written to
`worlds/`, episodes to `episodes_3d/`, and the two stay separate artifacts exactly
as mazes and episodes do.

One generation call returns a theme, a palette and a list of superquadric
parameters. The harness re-validates every one against hard numeric bounds, drops
whatever is malformed, and replaces the whole world with a procedural fallback if
too little survives. It then voxelises free space and proves — with the same
clearance test movement uses — that a route exists, before placing the spawn and a
reachable, initially unseen key. `superquadrics.py` meshes the primitives in pure
Python and owns collision, success and sensor visibility;
[Ursina](https://www.ursinaengine.org/) renders those meshes and reads input, but is
never asked a question about the world.

Movement is continuous rather than gridded: the agent returns a batch of absolute
target positions, the camera turns to face each one, and the harness flies the
straight segment in small increments, halting at the first that collides. Manual
play is `WASD` + `Space`/`Shift` + mouse look through the same collision and success
predicates.

**Where it departs from the 2D harness:** the 3D navigation agent is not a vision
policy, and `TASK.md`'s vision-policy context does not hold for it. It never sees
pixels. It receives the validated *parameters* of the superquadrics its sensor
(range + camera frustum + sampled line of sight) has detected, plus its own pose.
Once any part of a shape is detected it is handed the entire shape, so it knows more
than a screenshot of the same view would show. Meshes, unobserved primitives and the
generation-time reachability graph are never exposed to it.

## Observations
- Codex > Claude both for generation and navigation:
  - across all model sizes, codex is less verbose, more concise
  - for generation, codex uses python to generate the maze and don't get it wrong
  - when exploring, codex is more conservative and takes less action is the enviornment is not explored
- Model size matters: haiku gets the maze generation wrong, gpt luna always consumes all 24 moves even if only few tiles are visible.
- Terra is good at systemtaic exploration.
