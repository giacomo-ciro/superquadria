# Infinite Environment Generation via an Agent Harness

An agent harness that **builds a 2D maze environment and then plays it**. Both halves
are the same agent abstraction — one `run(prompt, schema) -> structured_output` call
against the `claude` CLI — and both are checked by code, not by looking at pixels.

- **Generation.** An agent invents a maze concept, then draws a 100×100 grid ten rows
  at a time across parallel calls. The harness stitches the batches and guarantees the
  result is solvable.
- **Navigation.** A second agent sees only a 10×10 window of that maze, proposes a
  *batch* of moves, and the harness executes them one cell at a time — halting the
  moment a move would walk into a wall.
- **Objective.** Success is `player.position == key.position`; collision is a terrain
  lookup. Both are O(1) checks against the `Scene` (`scene.py`), which is exactly the
  code-level-objective property the challenge is about.

Task spec: [`TASK.md`](./TASK.md) · Design: [`PLAN.md`](./PLAN.md)

---

## Run it

```bash
pip install -r requirements.txt        # pygame + pytest
```

**See the whole thing work in 10 seconds, with no API calls:**

```bash
python main.py run --offline --policy frontier
```

That generates a procedural maze and solves it with the built-in reference policy —
useful as a smoke test and as the baseline the agent is measured against.

**The real harness — agent generates, agent plays:**

```bash
python main.py run                     # 100x100, generation + navigation via `claude`
```

**In pieces**, which is what you want while iterating (generation is the slow part, so
generate once and replay the saved maze):

```bash
python main.py generate --out mazes/demo.json          # agent-authored maze -> JSON
python main.py play --maze mazes/demo.json             # agent navigates it
python main.py play --maze mazes/demo.json --reveal    # ...with the fog of war lifted
python main.py play --maze mazes/demo.json --policy frontier   # baseline, no API calls
```

Useful flags: `--render window|headless|terminal|none` (defaults to a window when
`$DISPLAY` is set, terminal otherwise), `--frame-dir frames/` to write a PNG per frame,
`--model`, `--seed`, `--width/--height`, `--max-steps`, `--max-calls`, `--step-delay`.
`python main.py --help` lists the rest.

Every run writes `runs/<timestamp>/scene.json` (the environment) and `episode.json`
(every plan, what executed, where it collided) so a run can be audited or replayed.

```bash
python -m pytest tests -q              # 38 tests, no API calls, ~1.5s
```

### What you see

The Pygame view renders **what the agent knows**, not the ground truth: dark cells are
unobserved, the yellow box is the 10×10 window it is actually being shown, and the map
uncovers as it explores. The sidebar tracks steps, agent calls, collisions, and
explored fraction. `--reveal` switches to the god view.

---

## How it fits together

```
                 ┌─────────────────────────────────────────┐
  design brief ──┤  Agent.run(prompt, schema) -> JSON       │──  claude -p
  band 0..9    ──┤  (ClaudeSubprocessAgent, agents/)        │    --json-schema
  each move    ──┘─────────────────────────────────────────┘    --output-format stream-json

  generation.py ──> Scene ──> Episode ──> State (10x10) ──> policy ──> Move(actions)
                     ▲          │                                          │
                     └──────────┴── execute step by step, halt on collision ┘
```

| Module | Role |
| --- | --- |
| `vector.py`, `entities.py`, `moves.py` | grid primitives, tile codes, the action vocabulary |
| `scene.py` | the world model; owns the collision and success predicates |
| `state.py` | the 10×10 observation handed to the agent |
| `agents/` | the `Agent` abstraction and its `claude -p` subprocess backend |
| `generation.py` | design brief + batched band drawing + stitching |
| `maze_utils.py` | tolerant parsing, connectivity repair, procedural fallback |
| `memory.py`, `policies.py`, `navigation.py` | fog-of-war map, baseline policy, Claude policy |
| `engine.py` | the observe → plan → execute loop, and the run log |
| `render.py` | Pygame / terminal / null views |

---

## The three problems worth reading about

### 1. A model cannot draw 10,000 cells in one call

The grid is cut into ten 10×100 bands, one call each. But bands drawn *independently*
have no reason to line up, and a naive fix — feed each band the previous band's last
row — serialises the whole thing into ten round trips.

Instead the harness makes the bands agree **without talking to each other**:

- One **design brief** call fixes the maze concept, and the brief prompt insists the
  rules be stated in absolute grid terms ("rows 0, 6, 12… are corridors"), never
  "continue from above". Every band then derives the same structure from its own row
  numbers.
- The harness — not the model — picks six **connector columns** per band seam, tells
  both neighbours about them, and then *enforces* them in code afterwards.

So bands run concurrently. In a real 40×40 run, four independent parallel calls
produced one continuous corridor grid needing exactly **one** repair corridor.

### 2. The maze must be solvable no matter what the model returns

Nothing about the generated output is trusted:

- `parse_rows` pads short rows, truncates long ones, and maps anything unrecognised to
  *wall* — the safe direction, since a spurious wall is repairable but a spurious
  opening is undetectable.
- A band that fails outright (API error, all-wall output) falls back to a procedural
  band; one bad call cannot sink a ten-call generation.
- `connect_regions` then labels every connected floor region and runs a multi-source
  0-1 BFS that grows each region into the surrounding wall space, giving every cell its
  nearest region and the wall thickness to reach it. A minimum spanning tree over the
  resulting candidate corridors picks the *cheapest set of carvings* that makes the map
  fully connected — O(N log N), and it never breaches the outer border.
- Finally the CLI asserts a BFS path exists from spawn to key before the episode starts,
  and records its length as the optimal-play baseline.

### 3. A stateless agent re-invoked per move cannot explore

The observation stays exactly what the spec says: the 10×10 window and nothing else.
But the agent is a fresh process every turn, so the harness keeps the **fog-of-war map**
for it (`memory.py`) — the union of everything it has already observed, replayed into
each prompt, with `?` for never-seen cells. It contains no cell the player has not
looked at, so it reveals nothing about the unexplored maze; it just stops the agent
re-deriving its own history 60 times per episode.

The prompt also tells it the rule that actually shapes good plans: *the first step into
a wall discards the rest of the batch*, so put the speculative step into `?` territory
at the end of a trajectory, not the start.

---

## What it actually does

Two real runs against the `claude` CLI (Haiku 4.5, 40×40 to keep them cheap):

**Generation** — four bands drawn by four *independent parallel* calls, from a brief the
agent wrote itself ("Archive Halls: a rigid grid of vault chambers connected by narrow
corridors… the repeating 5×5 chambers create natural landmarks"):

```
########################################
######.#####.#####.#####.#####.#####.###
######.#####.#####.#####.#####.#####.###
...
#......................................#      <- rows 6, 12, 18, 24, 30, 36
######.#####.#####.#####.#####.#####.###      <- columns 6, 12, 18, 24, 30, 36
```

The corridor lattice lines up across all four band boundaries, and connectivity repair
had to carve exactly **one** corridor to make the whole map reachable.

**Navigation** — spawn `(10, 17)`, key `(30, 12)`, optimal 27 steps:

| call | plan | result |
| --- | --- | --- |
| 1 | `right` + 19×`down` + 4×`left` | 24 steps, no collision — *"move right to reach column 18, which offers a clear vertical corridor"* |
| 2 | `down, left, left` | key reached |

**27 steps, 2 calls, 0 collisions, 12% of the map ever observed.** The first trajectory
is the interesting one: the agent read the corridor structure out of a 10×10 window,
committed 24 blind steps to it, and got every one right.

---

## Deliberate choices, and where they depart from `PLAN.md`

- **Walls are terrain, not entities.** `PLAN.md` §3 says `Scene` holds a list of
  `Entity`. Dynamic occupants (player, key) are entities; ~5,000 static walls live in a
  terrain grid instead, because they are only ever queried positionally and O(1) lookup
  is what the collision predicate needs.
- **Fog-of-war memory** (above) is harness bookkeeping added on top of the specified
  observation. `--policy frontier` and the tests exercise the same interface without it.
- **`--brief "..."`** optionally seeds the design step with a text command. The MVP has
  the agent decide for itself, per `PLAN.md` §2; this is the one-line hook that the
  "user-authored text commands" future-work item plugs into, and it is what makes the
  harness satisfy `TASK.md`'s "accepts text-based commands" today.
- **`FrontierPolicy`** (a greedy BFS explorer) is not in the plan. It costs nothing to
  run, gives the agent a number to be compared against, and lets the entire environment
  be tested without tokens.
- **`--dangerously-skip-permissions` is on by default**, as `PLAN.md` §5 requires for
  unattended runs, and is safe because `--allowedTools StructuredOutput` leaves the
  agent no capability but emitting text. `--no-skip-permissions` turns it off.

## Notes on cost and speed

Every turn is a fresh `claude -p` process, so **CLI startup dominates**, not inference —
on the Raspberry Pi this was developed on, a call costs ~60s regardless of model. The
agent is invoked with `--strict-mcp-config` and no `--mcp-config`, which loads no MCP
servers at all: nothing here needs them, and it trims both startup and the tool
definitions carried in every prompt.

Generation is 1 + `height / band` calls (11 for the default 100×100) but bands run
concurrently, so it costs roughly `--workers`-fold less wall-clock than that suggests.
Generate once, save the JSON, and replay it while iterating on navigation.

Navigation spends one call per trajectory of up to 24 steps, so budget `--max-calls`
accordingly: a 100×100 maze seen through a 10×10 window is a genuine search problem, and
the `frontier` baseline typically needs 60–100 batches to solve one. Making the agent
hold a persistent session instead of respawning per move (`PLAN.md` §6) is the change
that would matter most here.

## Known limits

- One maze layout per run. Varying the *structure* procedurally across runs
  (`PLAN.md` §6, "structural proceduralism") is the natural next step.
- The agent is re-invoked per move rather than held as a persistent session; the tmux
  approach in `PLAN.md` §6 would give it conversational continuity across an episode.
- 2D only, and integer-grid only. Nothing in the `Agent`/`Policy`/`Scene` split is
  2D-specific, but the renderer and the movement model are.
