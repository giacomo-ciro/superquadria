# Comprehensive Plan: Infinite Environment Generation via an Agent Harness

## 1. Project Overview
The goal is to build a hierarchical agent harness that generates and navigates through a static maze environment. The harness separates the environment creation process from the navigation loop to ensure clarity and scalability.

## 2. Game and Environment Mechanics
- **Environment Generation**: For the MVP, the generation agent itself decides the maze layout (no human-authored text prompt yet) and produces a 100x100 static maze defined by a fixed grid of walls and paths, generated completely in batches before the game begins. Accepting a user-supplied text command and extending generation beyond mazes is deferred to a future iteration (see Section 5).
- **Objective**: A key is hidden at random coordinates within the maze.
- **Player Interaction**: The player agent only observes a 10x10 local grid view of the maze at any given time.
- **Game Loop**: The agent receives the 10x10 state, proposes a batched move trajectory (multiple directional actions per call), and the harness executes it step by step. If a move in the trajectory would land on a wall cell, execution stops there and the remaining planned moves are discarded; the resulting state is then presented to the agent, and the loop repeats until collision-halt or successful task completion.
- **Verifiable Objective (code-level)**: Success is a direct equality check — player grid position equals the key entity's grid position. Collision is the player's next position landing on a wall-typed cell. Both are O(1) lookups against the `Scene`, not inferred from pixels/VLM, matching the code-level-objective motivation in TASK.md.

## 3. Code Structure
- **Internal Representation**: Modeled with a `Scene` class containing a list of `Entity` objects, using a `Vector2D` class with integer `(row, col)` coordinates to match the discretized grid state and directional (up/down/left/right) movement — no float precision needed for the MVP.
- **Agent Input**: A `State` object representing the 2D array of integers (the 10x10 local view), where each number corresponds to a specific element type (e.g., 0 for empty space, 1 for wall, 2 for key).
- **Rendering**: Maps the integer matrix to Pygame sprites or color codes for visual feedback.

## 4. Technology Stack
- **Language**: Python.
- **Game Engine/Renderer**: Pygame.
- **Agent Abstraction**: A custom class that receives the `State` object as input. It internally handles prompt collating, calling the agent (initially utilizing `claude -p` to enforce JSON structured output), and output parsing. It returns a structured `Move` object containing the sequence of directional actions.
- **Reference Implementation**: [claude.ts](https://github.com/giacomo-ciro/paperino/blob/main/src%2Fagent%2Fclaude.ts)

## 5. Agent Calling: Concrete `ClaudeSubprocessAgent` Implementation
The Agent Abstraction (Section 4) is an abstract interface — `run(prompt, schema) -> structured_output`. For the MVP, its first (and only) concrete implementation is a subprocess wrapper around the `claude` CLI, ported from the pattern in [claude.ts](https://github.com/giacomo-ciro/paperino/blob/main/src%2Fagent%2Fclaude.ts):

- **Invocation**: Spawn `claude` via Python's `subprocess.Popen` (own process group, so it can be killed cleanly on timeout) with:
  `--model <model> -p <prompt> --output-format stream-json --json-schema <schema> --allowedTools StructuredOutput --dangerously-skip-permissions`
  `--dangerously-skip-permissions` is required because the harness runs unattended with no TTY to approve the `StructuredOutput` tool call; this is safe because `--allowedTools StructuredOutput` restricts the agent to emitting text only (no bash/file-edit tools), even under prompt injection from generated content.
- **Structured output enforcement**: `--json-schema` pins the exact JSON Schema the harness expects (`Move` for the navigation agent, the maze-layout schema for the generation agent) — no free-text parsing or regex extraction.
- **Streaming JSONL parsing**: Read stdout line-by-line as newline-delimited JSON, tolerating partial chunks across reads. Watch for `type: "result"` events; the terminal event must have `subtype == "success"`, `is_error == False`, and a non-null `structured_output` field. Non-JSON lines (plain-text CLI diagnostics) are captured to a bounded tail buffer instead of dropped.
- **Error handling**: A non-zero exit code, a missing result event, `is_error: true`, or a missing `structured_output` field all raise a `ClaudeCallError` carrying exit code, captured stderr, the terminal result event, and the unparsed-stdout tail — enough to diagnose an unattended failure from one log line.
- **Timeout**: Process is killed (`SIGTERM` to the process group) if it exceeds a configured timeout, and the resulting `ClaudeCallError` reports the timeout explicitly rather than surfacing a generic non-zero exit.

## 6. Future Work (Out of Scope for MVP)
- **Persistent Interactive Agent via tmux**: Reimplement the Agent Abstraction to drive a long-lived Claude Code session through `tmux send-keys` instead of one-shot `claude -p` calls. This would let the agent hold a persistent, user-facing interactive chat across the full episode rather than re-invoking a stateless process per move, enabling mid-episode steering and richer conversational context. Not required for the MVP; noted here as the direction for the next iteration.
- **User-authored text commands**: Let the user (not the generation agent) supply the free-text environment spec, and extend generation beyond mazes to other environment types.
- **Structural proceduralism**: Vary the maze layout itself (not just the key position) across runs, moving toward genuinely infinite/procedural generation rather than one fixed maze per MVP run.
