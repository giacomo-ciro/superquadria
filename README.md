# Infinite Environment Generation via an Agent Harness

An agent harness that **builds a 2D maze environment and then plays it**. Both halves
are the same abstraction — `Agent.run(prompt, schema) -> structured_output`, a wrapper
around an interactive coding CLI (`claude` or `codex`) driven inside tmux — checked by
code (position/terrain lookups in `Scene`), never by looking at pixels.

Task spec: [`TASK.md`](./TASK.md) · design: [`PLAN.md`](./PLAN.md) · conventions and
architecture notes for agents: [`AGENTS.md`](./AGENTS.md).

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

## Observations
- Codex > Claude both for generation and navigation:
  - across all model sizes, codex is less verbose, more concise
  - for generation, codex uses python to generate the maze and don't get it wrong
  - when exploring, codex is more conservative and takes less action is the enviornment is not explored
- Model size matters: haiku gets the maze generation wrong, gpt luna always consumes all 24 moves even if only few tiles are visible.
- Terra is good at systemtaic exploration.
