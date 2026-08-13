An agent harness that **generates a 3D superquadric environment and then plays it**.
Challenge spec: [TASK.md](TASK.md) · user-facing docs: [README.md](README.md). Read those before changing behaviour.

## The idea

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
- [agents/](src/agents/): the `Agent` interface, the shared `TmuxAgent`, and the `claude`/`codex` backends
- [geometry/](src/geometry/): 3D vectors, superquadric primitives, meshing, collision, and sensor
- [world/](src/world/): floor plan layout, architectural shell, task objects, scene model, world generation
- [navigation/](src/navigation/): parameter observations, spatial memory, waypoints, and flight policies
- [prompts/](src/prompts/): decoupled prompt templates for layout, room furnishing, and navigation
- [engine/](src/engine/): observe-plan-fly loop, Ursina renderer, config loader, pipeline logging, CLI

The harness is launched via `python -m engine` and reads `configs/3d.yaml`.

| Package | Role | Key Modules |
| --- | --- | --- |
| [agents/](src/agents/) | Agent CLI abstraction | `base.py`, `tmux_agent.py`, `claude_tmux.py`, `codex_tmux.py` |
| [geometry/](src/geometry/) | Math & superquadrics | `geometry.py`, `superquadrics.py` |
| [world/](src/world/) | World & generation | `layout.py`, `shell.py`, `task.py`, `scene.py`, `generation.py` |
| [navigation/](src/navigation/) | Observation & policy | `state.py`, `memory.py`, `moves.py`, `policies.py`, `navigation.py` |
| [prompts/](src/prompts/) | Prompt engineering | `layout.py`, `room.py`, `navigation.py` |
| [engine/](src/engine/) | Execution & CLI | `engine.py`, `render.py`, `config.py`, `pipeline_log.py`, `cli.py` |

Rules:
- **The agent sees parameters, not pixels.** This is a deliberate departure from
  TASK.md's vision-policy context — document it, don't quietly present it as
  equivalent. Once any part of a primitive is detected the agent receives the whole
  primitive.
- **`SpatialMemory` holds only primitives a sensor query actually returned.** Never
  the handler, never the generation-time voxel graph.
- **The handler is authoritative, Ursina is not.** Collision, success and visibility
  are computed from harness meshes, so generation and headless checks run without a
  display. Import Ursina lazily, inside the renderer only.
- **The harness places the spawn and objects (peg + decoys), not the generation agent**
  — after proving a route exists with the same clearance test movement uses.
- **The sensor has no range limit.** It sees everything in the camera frustum that
  nothing else is hiding. Measured as having zero cost difference from the old range
  inside sealed rooms.
- **Colour is generation-designed.** Generation agents color assemblies following
  each room's palette (walls, floor, ceiling). Only the lock is consistently gold.
- **Interaction is explicit `pick`/`place`, not collision-triggered.** Flying through an
  object or door does nothing by itself; the agent must issue a `pick`/`place` action
  within reach (`task.DEFAULT_REACH`, 2.5 units). `place` away from the door/lock drops
  the carried object back into the room — carrying a decoy does not force a trip to the
  door, only the single-slot inventory does.

## Commands

```bash
uv sync                              # Python >=3.12, deps in pyproject.toml
python -m engine generate --offline  # procedural world, zero agent calls
python -m engine run --offline       # procedural building + manual flight (smoke test)
python -m engine run                 # agent generates, agent flies
python -m engine play                # pick a saved world under worlds_3d/
python -m engine replay              # re-watch a saved episode
```

## Conventions
- **Minimal docs.** Keep documentation (README.md, helps) minimal. We are still in MVP phase, everything can change. README should just explain entry points and overall arch, nothing more.
- **Don't implement tests.** We are developing an MVP and we want to move fast.
- **Never trust model output.** Parse defensively, fall back procedurally — one bad call must not sink a run.
- Running the loop consumes agent usage credits. Do not run without asking the user or unless explicitly instructed.