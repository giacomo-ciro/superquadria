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

Three packages under [src/](src/): [harness_2d](src/harness_2d/) (the maze harness),
[harness_3d](src/harness_3d/) (its sibling — same loop in 3D, built entirely from
superquadrics), and [harness_common](src/harness_common/), which holds only what both
run on. Each harness is launched the same way, `python -m harness_2d|harness_3d`, and
reads its own config, `configs/2d.yaml` or `configs/3d.yaml`.

### Shared

| Path | Role |
| --- | --- |
| [agents/](src/harness_common/agents/) | the `Agent` interface, the shared `TmuxAgent`, and the `claude`/`codex` backends |
| [config.py](src/harness_common/config.py), [configs/](configs/) | Hydra/OmegaConf loader over each harness's config and the `configs/agent/` group, the single source of defaults; a CLI takes only a command + `--offline`, everything else lives here |
| [pipeline_log.py](src/harness_common/pipeline_log.py) | one legible line per event, for runs nobody is watching |

A module belongs here only when both harnesses need it unchanged — don't move
2D-specific code in to make it look symmetric.

### 2D

| Path | Role |
| --- | --- |
| [vector.py](src/harness_2d/vector.py), [entities.py](src/harness_2d/entities.py), [moves.py](src/harness_2d/moves.py) | grid primitives, tile codes, action vocabulary |
| [scene.py](src/harness_2d/scene.py) | world model; owns the collision and success predicates |
| [state.py](src/harness_2d/state.py) | the 10×10 observation handed to the agent |
| [generation.py](src/harness_2d/generation.py), [maze_utils.py](src/harness_2d/maze_utils.py) | brief + banded drawing; parsing, repair, procedural fallback |
| [memory.py](src/harness_2d/memory.py), [policies.py](src/harness_2d/policies.py), [navigation.py](src/harness_2d/navigation.py) | fog-of-war map, offline baseline policy, Claude policy |
| [engine.py](src/harness_2d/engine.py) | observe → plan → execute loop and the run log |
| [render.py](src/harness_2d/render.py), [cli.py](src/harness_2d/cli.py) | the Pygame view, `generate`/`play`/`run` commands |

### 3D

Design: [3D_PLAN.md](docs/3D_PLAN.md). Same `Agent.run` abstraction, from
`harness_common.agents`.

| Path | Role |
| --- | --- |
| [geometry.py](src/harness_3d/geometry.py), [superquadrics.py](src/harness_3d/superquadrics.py) | vectors and the one rotation order; the primitive, meshing, collision, sensor |
| [scene.py](src/harness_3d/scene.py) | bounds, player, key; owns the collision and success predicates |
| [generation.py](src/harness_3d/generation.py) | schema, defensive validation, procedural fallback, reachability witness |
| [state.py](src/harness_3d/state.py), [memory.py](src/harness_3d/memory.py) | the parameter observation and the observed-only map |
| [moves.py](src/harness_3d/moves.py), [policies.py](src/harness_3d/policies.py), [navigation.py](src/harness_3d/navigation.py) | target positions, manual flight, the agent policy |
| [engine.py](src/harness_3d/engine.py), [render.py](src/harness_3d/render.py), [cli.py](src/harness_3d/cli.py) | the loop and replay, the Ursina view, the commands |

Rules specific to it:
- **The agent sees parameters, not pixels.** This is a deliberate departure from
  TASK.md's vision-policy context — document it, don't quietly present it as
  equivalent. Once any part of a primitive is detected the agent receives the whole
  primitive.
- **`SpatialMemory` holds only primitives a sensor query actually returned.** Never
  the handler, never the generation-time voxel graph.
- **The handler is authoritative, Ursina is not.** Collision, success and visibility
  are computed from harness meshes, so generation and headless checks run without a
  display. Import Ursina lazily, inside the renderer only.
- **The harness places the spawn and key, not the generation agent** — after proving
  a route exists with the same clearance test movement uses.

## Commands

```bash
uv sync                              # Python >=3.12, deps in pyproject.toml
python -m harness_2d run --offline   # 2D: procedural maze + manual play
python -m harness_2d run             # 2D: agent generates, agent plays

python -m harness_3d run --offline   # 3D: procedural world + manual flight
python -m harness_3d run             # 3D: agent generates, agent flies
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