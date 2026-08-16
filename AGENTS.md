An agent harness that **generates a 3D superquadric environment and then plays it**.

Both halves go through one abstraction — `Agent.run(prompt, schema) -> structured_output`,
implemented by driving an interactive coding CLI (`claude` or `codex`, chosen by the
`agent` config group) inside a persistent tmux session.

- **Generation.** Two agent calls: layout designs a multi-room floor plan (assigning a `key_concept` per room); room call
  furnishes rooms with superquadric assemblies (including a mandatory key assembly). The harness builds sealed rooms,
  cuts doorways along the task path, and creates an embossed relief lock replica on each locked door.
- **Navigation.** A second agent observes superquadric parameters in its field of view (grouped into anonymized assemblies)
  and returns a batch of explicit actions — `move` (fly to a target, halting on collision), `pick` (pick up a nearby
  assembly, within reach), `place` (place the carried assembly, or attempt the lock if placed at/near the door) — executed in order.
- **Objective.** Success is identifying the matching key assembly from its geometric parameters and placing it at every
  room's locked door in turn, in path order, reaching the true exit. Collision, pickup and placement are code-level
  checks against the `Scene` / `SuperquadricHandler`, never pixel-inferred.

## Layout

Modular packages under [src/](src/):
- [agents/](src/agents/): the `Agent` base class, and the `claude`/`codex` backends
- [geometry/](src/geometry/): 3D vectors, superquadric primitives, meshing, collision, and sensor
- [world/](src/world/): floor plan layout, architectural shell, task objects, scene model, world generation
- [navigation/](src/navigation/): parameter observations, spatial memory, waypoints, and flight policies
- [prompts/](src/prompts/): decoupled prompt templates for layout, room furnishing, and navigation
- [engine/](src/engine/): observe-plan-fly loop, Ursina renderer, config loader, CLI
- [logger.py](src/logger.py): structured logging — a leaf module, since every package logs

The harness is launched via `python -m engine` and reads `configs/config.yaml`.

| Package | Role | Key Modules |
| --- | --- | --- |
| [agents/](src/agents/) | Agent CLI abstraction | `base.py`, `claude.py`, `codex.py` |
| [geometry/](src/geometry/) | Math & superquadrics | `geometry.py`, `superquadrics.py` |
| [world/](src/world/) | World & generation | `layout.py`, `shell.py`, `task.py`, `scene.py`, `generation.py` |
| [navigation/](src/navigation/) | Observation & policy | `state.py`, `memory.py`, `moves.py`, `policies.py`, `navigation.py` |
| [prompts/](src/prompts/) | Prompt engineering | `layout.py`, `room.py`, `navigation.py` |
| [engine/](src/engine/) | Execution & CLI | `engine.py`, `render.py`, `config.py`, `cli.py` |

## Commands

```bash
uv sync                              # Python >=3.12, deps in pyproject.toml
python -m engine generate            # agent generates world
python -m engine run                 # agent generates, agent flies
python -m engine play                # pick a saved world under worlds/
python -m engine replay              # re-watch a saved episode
```

## Conventions
- **Minimal docs.** Keep documentation (README.md, helps) minimal. We are still in MVP phase, everything can change.
- **Don't implement tests.** We are developing an MVP and we want to move fast.
- **Never trust model output.** Parse defensively — if generation fails, exit with an informative error rather than falling back procedurally.
- Running the loop consumes agent usage credits. Do not run without asking the user or unless explicitly instructed.