An agent harness that **generates a 2D maze environment and then plays it**.
Challenge spec: [TASK.md](TASK.md) · user-facing docs: [README.md](README.md). Read those before changing behaviour.

## The idea

Both halves go through one abstraction — `Agent.run(prompt, schema) -> structured_output`,
implemented by driving an interactive coding CLI (`claude` or `codex`, chosen by the
`agent` config group) inside a persistent tmux session.

- **Generation.** One call for an entire maze generated one shot.
- **Navigation.** A second agent sees only a 10×10 window, returns a *batch* of moves, and
  the harness executes them one cell at a time, halting the moment one would hit a wall.
- **Objective.** Success is `player.position == key.position`; collision is a terrain
  lookup. Both are O(1) checks against the `Scene` — code-level, never pixel-inferred.

## Layout

The maze harness is the `harness_2d` package under [src/](src/); `harness_3d` is its
sibling, empty for now.

| Path | Role |
| --- | --- |
| [vector.py](src/harness_2d/vector.py), [entities.py](src/harness_2d/entities.py), [moves.py](src/harness_2d/moves.py) | grid primitives, tile codes, action vocabulary |
| [scene.py](src/harness_2d/scene.py) | world model; owns the collision and success predicates |
| [state.py](src/harness_2d/state.py) | the 10×10 observation handed to the agent |
| [agents/](src/harness_2d/agents/) | the `Agent` interface, the shared `TmuxAgent`, and the `claude`/`codex` backends |
| [generation.py](src/harness_2d/generation.py), [maze_utils.py](src/harness_2d/maze_utils.py) | brief + banded drawing; parsing, repair, procedural fallback |
| [memory.py](src/harness_2d/memory.py), [policies.py](src/harness_2d/policies.py), [navigation.py](src/harness_2d/navigation.py) | fog-of-war map, offline baseline policy, Claude policy |
| [engine.py](src/harness_2d/engine.py) | observe → plan → execute loop and the run log |
| [render.py](src/harness_2d/render.py), [cli.py](src/harness_2d/cli.py) | the Pygame view, `generate`/`play`/`run` commands |
| [config.py](src/harness_2d/config.py), [configs/](configs/) | Hydra/OmegaConf loader over `configs/config.yaml` and the `configs/agent/` group, the single source of defaults; the CLI takes only a command + `--offline`, everything else lives here |

## Commands

```bash
uv sync                                        # Python >=3.12, deps in pyproject.toml
python main.py run --offline                   # full loop, no API calls (smoke test)
python main.py run                             # agent generates, agent plays
```

## Conventions
- **Minimal docs.** Keep documentation (README.md, helps) minimal. We are still in MVP pahse, everything can change. README should just explain entry points and overall arch, nothing more.
- **Don't implement tests.** We are developing an MVP and we want to move fast.
- **Never trust model output.** Parse defensively, map anything unrecognised to *wall*,
  fall back procedurally — one bad call must not sink a run.
- **Walls are terrain, not entities** (deliberate departure from PLAN.md §3; see README).
- **The observation stays 10×10.** `FogMemory` is harness bookkeeping and must never hold
  a cell the player has not actually observed.
- README documents where the implementation departs from PLAN.md — update it when you add
  another departure.
- Running the loop consume agent usage credits. Do not run without asking the user or unless explicitely instructued.