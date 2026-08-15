# Generating and Navigating 3D Environments with Agents

An agent harness that **builds a 3D environment and then plays it**.

The world is made of rooms, connected in a feed-forward fashion (from a start room, one way toward a final room). The doors between rooms are initially locked. The goal of the agent is to find an object that unlocks the door, pick it up, place it on the door, and proceed. The more rooms the agent clears before running out of budget (collisions, calls, runtime), the higher the score.

The engine is divided into two parts, both agent-driven and taking only text input. Remarkably, no vision is used even though we are navigating a complex and expressive 3D environment. To accomplish this, we based the 3D world on a parametric yet very expressive geometric primitive: the superquadric. Each object in the scene is an assembly of superquadrics — the generation agent produces superquadric parameters, and the navigation agent consumes superquadric parameters. In this way we achieve expressive, lightweight, and efficient 3D generation and navigation from text-only commands.

The two main steps of the engine:
1. Generation: a two-step flow. A first agent generates the overall layout of the world (from a user prompt which provides arbitrary details on the environment, and constraints such as the max number of rooms, number of objects per room, etc.). Then, each room is furnished in parallel by a set of agents that take the brief and instructions and produce the actual shapes. After the engine checks for collisions, out-of-bounds placement, etc., the 3D world is rendered and ready to be played.
2. Navigation: the agent starts in the first room, with everything in front of it visible (i.e., visible shape parameters are listed in the navigation prompt). It is told the shape of the key it needs to find. It freely explores the room, picking up and placing objects, until it finds the key, places it on the door, and unlocks it. Once done, it proceeds to the next room and the process repeats. The game ends when the agent clears all rooms or reaches the maximum number of collisions or calls.

## Usage
The engine drives an interactive window where the world is rendered and the policy animated. Agent sessions run in a custom tmux session with one window per agent (one per layout, one per room, one per navigation), and you can attach directly from the terminal to inspect exactly what is happening: what the agent is receiving as input, how it's reasoning, and what it's returning as output.

```bash
uv sync                              # Python >=3.12

python -m engine generate --offline  # procedural building, zero agent calls
python -m engine run --offline       # procedural building + manual flight (smoke test)
python -m engine run                 # agent generates, agent flies
python -m engine play                # pick a saved world under worlds/
python -m engine replay              # re-watch a saved episode
```

The CLI is minimal by design. Agent settings, world bounds, episode budgets, and policy type (agent-driven, manual play) live in [`configs/config.yaml`](./configs/config.yaml).

The engine supports both Codex and Claude Code. Extensive experiments suggest Codex is superior across all models. It is less verbose, faster, and generally better at solving the tasks. We keep Claude for compatibility, but we strongly recommend using only Codex.

You can set the agent from the config:
```yaml
defaults:
  - agent: codex    # or: claude
```
Agent-specific settings can be specified in [`configs/agent/codex.yaml`](./configs/agent/codex.yaml), e.g., binary, model, effort.

**Tmux.** All agents run inside a single persistent `tmux` session — `general-intuition` —
with dedicated windows for each step (`layout`, `room1`, `room2`, ..., `player`).

The harness logs the attach command when it spawns a window; paste it into another
terminal to watch that agent live:
```bash
tmux attach -f ignore-size -t general-intuition:layout
```

**Play it yourself.** `policy: manual` flies the same scene with keyboard + mouse controls, so you can play a generated world.

## Discussion

Extensive experiments demonstrate Codex is superior at world generation and navigation, both in terms of speed, token usage, and quality. Codex uses an average of 3.4 actions per call (either move, pick, or place), even with a fixed limit of 8.

The hierarchical, multi-step generation is not only fancier or faster, but required for quality and realism. One-shot generation of the entire world is too heavy and gets the agent lost. Instead, first generating the layout ensures coherent structure and theme. Then each agent furnishes a room by first generating objects in a local frame, then placing them in the scene (position, scale, rotation). This grants reusability and ensures correct geometry — outputting absolute world coordinates for each part would be much harder for the agent.

## Acknowledgements

We use [Ursina](https://www.ursinaengine.org/) as the game engine, [Codex](https://openai.com/codex/) as the generation/navigation agent, and [Claude Code](https://claude.com/product/claude-code) and [Antigravity CLI](https://antigravity.google/product/antigravity-cli) as coding agents for developing the pipeline.

## Future Work

Make room history important, e.g., the last room requires an element from the very first to be completed, so the agent must remember everything and navigate back.
