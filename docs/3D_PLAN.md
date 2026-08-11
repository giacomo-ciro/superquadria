# 3D superquadric harness plan

Status: **final plan**.

The core claim is that a fully playable, realistic 3D game harness can be built entirely from
superquadrics: they are expressive enough to compose coherent worlds, lightweight enough to
store as compact parameters, and structured enough for an agent to understand directly.

"Fully playable" means a complete goal-directed loop: spawn in a validated world, observe it,
move freely in 3D, collide with obstacles and bounds, discover and collect a reachable key,
terminate with an explicit outcome, and replay the episode. Both an agent and a person can play.
It does not imply combat, inventory, NPCs, gravity, or general-purpose game mechanics.

"Realistic" means visually coherent real-time 3D scenes with credible scale, lighting,
occlusion, collision, and spatial composition. It does not mean photorealism or realistic
rigid-body physics. All generated world geometry, obstacles, landmarks, and collectibles are
represented by superquadric parameters and rendered from harness-generated superquadric meshes.
Cameras, lighting, UI, and non-solid boundary guides are rendering infrastructure, not world
geometry.

"Infinite environment generation" means an open-ended supply of distinct generated worlds,
not one spatially unbounded world. Each world is finite so reachability, collision, replay, and
completion remain verifiable.

## 1. Goal

Build a sibling `harness_3d` that:

1. asks the configured coding agent to generate a bounded 3D world made from coloured
   superquadrics, optionally directed by a world brief;
2. validates and stores that world as compact parameters, not as source code or model files;
3. places a reachable player and hidden key using harness-side code;
4. gives the navigation agent only the superquadric parameters it has observed, plus its
   own pose and episode bookkeeping;
5. supports agent waypoint navigation and manual first-person flight through the same collision
   and success predicates; and
6. renders the same parameterised world in Ursina and records a replayable episode.

Success remains a code-level predicate. No VLM judges collision, visibility, or key pickup.
The parameter observation is an explicit departure from TASK.md's vision-policy context.

## 2. Scope decisions

### Primitive equation

The MVP supports the complete useful shape range of the single standard superquadric equation
implemented by the referenced `PredictionHandler`:

```text
x = ax * signed_cos(eta, e1) * signed_cos(omega, e2)
y = ay * signed_cos(eta, e1) * signed_sin(omega, e2)
z = az * signed_sin(eta, e1)
```

where `eta` is latitude, `omega` is longitude, `scale=(ax, ay, az)`, and
`exponents=(e1, e2)`. Varying those parameters covers spherical, ellipsoidal, cuboid-like,
cylindrical, diamond-like, and pinched/concave forms without an enumerated shape catalogue.

In geometry literature this particular closed equation is more precisely the superellipsoid
equation, although computer-vision code commonly calls it simply "superquadric". This project
uses `Superquadric` consistently with the requested abstraction. It does not add separate
supertoroid or unbounded superhyperboloid equations.

The handler accepts every finite, non-degenerate parameter combination inside defensive
numeric bounds. Initial exponent bounds should be `[0.1, 4.0]`, covering common sharp through
pinched forms; they remain configuration values because extreme exponents require denser
meshes.

### World and entities

- The playable volume is a cube centred at world origin. Going outside it is a collision.
- Generated superquadrics are static, solid obstacles and visual landmarks.
- Several primitives may share an `assembly` label to form one visual composition.
- The harness, not the generation agent, places the spawn and key after validating free
  space. Otherwise a plausible-looking generation can be unwinnable.
- The key is itself a small gold superquadric, but is non-blocking.
- The player is a sphere with a configured radius. The camera is attached to its centre;
  render near-plane settings must prevent obvious clipping.
- There is no gravity, ground, jump, velocity, or inertia. This is controlled flight, not a
  physics simulator.
- Manual play uses `WASD` for horizontal movement, `Space`/`Shift` for up/down, and the mouse
  for look direction.

### Navigation actions

The agent returns only a sequence of absolute world-space target positions:

```json
{
  "positions": [
    [4.0, 7.5, -2.0],
    [7.0, 8.0, 1.0]
  ]
}
```

- Each position is the next destination. The requested movement vector and amount are
  `target - current_position`; separate direction and distance fields would be redundant.
- Before a segment, point the camera along `target - current_position`, record that
  observation, then fly the straight segment while holding that orientation.
- After arrival, retain the last travel direction until a new non-zero segment begins.
- A repeated/current position is a no-op and cannot rotate the camera. The agent must move to
  change its view direction; this is an intentional simplification.
- Each segment has a maximum length. Longer output is shortened along the requested vector.
- Each segment is subdivided into fixed collision increments. The first blocked increment
  halts the remaining batch, matching the 2D harness's failure semantics.
- Key pickup is checked after every increment, not only at target endpoints.
- The batch and distance caps prevent one bad response from consuming the episode.

Target positions are preferable to controller-like tokens here. The agent receives explicit
geometry and should be allowed to reason geometrically. Restricting it to six axis steps
would be a 3D grid with a fancy renderer, not complete 3D movement.

The initial agent action schema has no independent look action: each non-zero waypoint turns
the camera toward itself before movement begins. Manual play keeps mouse look independent of
movement and can change it while moving. An explicit agent `look_at` action is deferred.

Success is the same sphere-versus-mesh intersection used by collision, applied to the
non-solid key primitive. This avoids an arbitrary centre-distance threshold and keeps key
scale visually meaningful.

## 3. Canonical data model

All persisted values are plain JSON. Runtime engine objects never enter the saved world.

```text
Superquadric
  id: int                         # assigned by the harness, stable in observations
  kind: str                       # obstacle | key; only the harness may create a key
  assembly: str                   # short landmark/group label
  position: (float, float, float) # world-space centre
  rotation: (float, float, float) # XYZ Euler degrees, with one documented matrix order
  scale: (float, float, float)    # positive semiaxes
  exponents: (float, float)       # positive shape exponents
  color: (float, float, float)    # sRGB channels in [0, 1]

Player
  position: (float, float, float)
  forward: (float, float, float)   # camera look direction; initially +Z
  radius: float

Scene
  bounds: float                   # cube spans [-bounds/2, bounds/2] on every axis
  primitives: SuperquadricHandler
  player: Player
  key_id: int
  meta: dict
```

`kind` determines collision semantics: obstacles are solid and the key is collectible. A
separate `solid` flag would be redundant and could contradict `kind`.

Coordinates follow Ursina's convention: +X right, +Y up, +Z forward. Euler angles are easier
for a language model to generate and inspect than quaternions or a 3x3 matrix. For column
vectors, apply X, then Y, then Z (`R = Rz @ Ry @ Rx`). The handler converts angles once to an
orthonormal matrix and uses that exact matrix for meshing, collision, and visibility
calculations. Ursina must receive already transformed vertices; duplicating its Euler
convention in authoritative geometry invites subtle disagreement.

The generation schema should reduce colour noise by asking for a 3-6 colour palette and a
palette index per primitive. The parser resolves the index to RGB. The fixed key colour is
not part of that palette.

## 4. `SuperquadricHandler`

The referenced SuperDec `PredictionHandler` is inspiration for the API only. It is not
suitable for reuse: point clouds, assignment matrices, existence logits, Torch, Open3D,
Trimesh, and a batch dimension are inference artifacts, not game-world state.

The new handler owns a validated list of `Superquadric` values and provides:

```text
to_dict() / from_dict()
mesh_data(id, resolution) -> vertices, triangles, normals, colours
visible_from(pose, sensor) -> list[Superquadric]
is_blocked(position, player_radius) -> bool
segment_is_clear(start, end, player_radius, increment) -> collision result
```

Rules:

- Mesh generation is pure Python and engine-independent. It returns simple arrays/lists that
  Ursina's `Mesh(vertices=..., triangles=..., colors=..., normals=...)` can consume.
- Mesh results are cached by primitive ID and resolution. Static geometry must not be rebuilt
  per frame.
- Collision is authoritative in the handler, not inferred from pixels and not delegated to
  Ursina. Rendering remains optional, so generation and headless smoke checks work without a
  display.
- With an MVP cap below roughly one hundred primitives, an O(P) broad-phase scan is simpler
  and sufficient. A spatial tree is unjustified until profiling proves otherwise.
- A transformed axis-aligned bounding box rejects distant primitives before the expensive
  narrow-phase check.

### Collision approximation

Exact continuous collision between a sphere and an arbitrarily transformed superquadric is
not a small feature. The MVP should state its approximation honestly:

1. generate one fixed-resolution collision mesh from the same equation as the render mesh;
2. broad-phase against expanded primitive AABBs;
3. use the analytic implicit function to reject a player centre inside a primitive;
4. compute sphere-to-triangle distance against nearby surface triangles at each movement
   increment; and
5. reject the increment when that distance is at most the player radius.

This makes collision and the rendered surface agree up to the documented tessellation and
movement increment. Thin obstacle primitives are forbidden: every obstacle scale must exceed
the player diameter plus one movement increment. This reduces tunnelling risk within the stated
approximation; the key is exempt because it is non-blocking.

Ursina supports procedural `Mesh` objects and mesh colliders, but using its collider as the
source of truth would couple episode correctness to an active render window. Its manual
`Ursina.step()` API is useful for keeping the window responsive while an agent call runs.

References: [Ursina Mesh](https://www.ursinaengine.org/api_reference_v8_0_0/mesh.html),
[Entity/collider API](https://www.ursinaengine.org/api_reference_v8_0_0/entity.html), and
[current PyPI package](https://pypi.org/project/ursina/).

## 5. Defensive generation

### Agent output

Use one generation call for a coherent world, not independent spatial chunks whose seams and
palettes will disagree. `generation.brief` is optional: a non-empty brief directs agent
generation; without one, the agent invents a distinctive world. Offline generation ignores the
brief entirely. The response contains:

- theme and short description;
- palette;
- 3-6 concrete composition rules; and
- a bounded list of primitives with assembly labels.

The prompt requires distinct landmark assemblies, deliberate overlap inside assemblies,
open flight corridors between assemblies, coverage across all octants, and no primitive
crossing the world boundary. The configured primitive range should initially be modest
(approximately 32-80), then be tuned from measured rendering and prompt size.

### Validation

Never construct runtime geometry directly from model output. Parse every primitive with
these checks:

- exact field shape and finite numeric values (reject NaN and infinity);
- bounded position, scale, exponent, and rotation;
- valid palette index and short bounded strings;
- no duplicate IDs (agent IDs are ignored and reassigned);
- no primitive crossing the cube boundary after rotation;
- maximum primitive count enforced before meshing; and
- invalid individual primitives dropped, with a minimum valid count required for the world.

Unknown or malformed values do not become arbitrary geometry. If too little valid content
remains, or world validation fails, replace the whole result with the procedural fallback.
Partially "repairing" a failed composition can preserve neither its aesthetics nor its
navigability.

### Procedural fallback and reachability

The fallback creates a few palette-coherent assemblies from bounded random templates (arches,
rings, branching clusters, and offset columns). It is not uniform noise.

For both agent and fallback worlds:

1. voxelise free space at a configured validation spacing using the same player-clearance
   predicate as movement;
2. connect 26-neighbour voxel centres only when the complete segment between them passes that
   same player-clearance check;
3. find connected components in the edge-validated graph;
4. require a sufficiently large free component;
5. select spawn and key points from that component with a minimum graph distance;
6. require the key to be outside the initial sensor result; and
7. store the validation spacing, component size, and known witness path length in metadata.

The voxel graph is a generation-time proof witness only. Neither policy can access it, and
runtime movement remains continuous rather than snapping to it.

## 6. Observation and memory

The agent receives structured parameter observations only—never screenshots, mesh vertices,
collision triangles, hidden primitives, or the generation-time voxel graph. One observation
contains:

```text
call/waypoint/distance/collision budgets
player position + look direction
world bounds
last trajectory outcome
currently visible primitive parameters
remembered parameters for previously visible primitives
key_seen flag
```

Primitive positions are included in world coordinates plus a computed player-relative centre.
The latter prevents every navigation turn from wasting reasoning on coordinate subtraction.
Only stable, validated parameters are exposed; meshes and triangle lists are not.

`SpatialMemory` stores only primitives returned by an actual observation. It must never copy
the full handler or use the generation-time voxel map. It also keeps a coarse record of sensor
poses already sampled, so the agent can distinguish exploration from repeatedly looking at the
same volume.

### Sensor semantics

A primitive is observable when:

1. its bounding sphere intersects the configured range;
2. its bounding sphere intersects the camera frustum; and
3. at least one deterministic surface sample (the six local-axis extrema plus the collision
   mesh vertex nearest the player) has line of sight against collision meshes, ignoring the
   candidate primitive itself.

Returning the full parameters after any part becomes visible is the central abstraction: the
sensor does not reveal an unseen primitive, but it does reveal the complete fitted primitive
once detected. Rendering may show partial occlusion while the agent receives the full compact
shape. This information advantage over pixels must be documented, not disguised as equivalent
to a vision policy.

Visibility is computed at the initial pose, whenever camera orientation changes, and after every
movement increment, including intermediate points in a batched trajectory. For the agent, the
camera changes orientation before each non-zero segment; manual mouse look may change it while
moving. Memory therefore contains every actually traversed observation, not only turn
endpoints.

## 7. Episode loop

```text
observe -> integrate memory -> request target-position batch
        -> validate/limit batch
        -> orient and execute each segment incrementally
        -> stop on collision, key pickup, budget, or window close
        -> observe again
```

The episode records the requested positions, accepted/clamped positions, every executed
increment, collision primitive ID, observations gained, player poses, and the terminal reason.
Replay consumes only the saved world and episode JSON; it never calls an agent.

Budgets should be expressed separately:

- maximum agent calls;
- maximum target-position count per call;
- maximum distance per target position;
- maximum total travelled distance; and
- maximum collisions.

A generic `steps` counter is misleading once movement is continuous.

Policies:

- `agent`: parameter-observation prompt through the existing `Agent.run` abstraction;
- `manual`: first-person flight using `WASD`, `Space`/`Shift`, and mouse look.

`generate --offline` uses procedural generation and ignores the configured brief.
`run --offline` combines procedural generation with manual play, guaranteeing zero agent calls.
Normal `run` and `play` use the configured `agent` or `manual` policy.

## 8. Ursina renderer

Use Ursina as a renderer and input/event loop, not as authoritative world state.

- Create one cached `Entity` per primitive for the initial MVP. Render all world primitives and
  let the camera and GPU handle visual frustum and occlusion; `visible_from()` controls only the
  parameters disclosed to the agent and the sensor counts shown in the HUD. Batch or chunk
  meshes only after profiling shows entity overhead matters.
- Use smooth normals, a lit shader, one directional light, low ambient light, and the generated
  palette. A wireframe or translucent cube communicates world bounds.
- Render the key with a distinct gold/emissive treatment.
- Attach the camera to the player pose. Overlay phase, call count, travelled distance,
  collisions, visible/remembered primitive counts, and whether the key is known.
- Keep Ursina work on the main thread. Run blocking agent calls on a worker while the main
  thread repeatedly calls `app.step()` and redraws a "thinking" status.
- `generate` and pure-logic checks must not import or initialise Ursina. Import it lazily inside
  the renderer.
- Closing the window must cancel an in-flight tmux turn, as the 2D harness does.

The initial renderer displays the player's first-person view. A separate god/debug camera is
useful during development but is not agent input and should not be part of the MVP unless
requested.

## 9. Package and command layout

```text
src/harness_3d/
  __main__.py             # python -m harness_3d ...
  geometry.py             # Vec3 and rotation helpers
  superquadrics.py        # Superquadric, MeshData, SuperquadricHandler
  scene.py                # bounds, player, key, persistence, success
  generation.py           # schema, parsing, fallback, placement validation
  state.py                # current parameter observation
  memory.py               # observed-only SpatialMemory
  moves.py                # target-position schema and defensive parser
  policies.py             # policy interface and manual policy
  navigation.py           # agent prompt/policy
  engine.py               # episode and replay records
  render.py               # lazy Ursina integration
  cli.py                  # generate/play/run/replay

configs/3d.yaml
worlds/*.json
episodes_3d/*.json
```

For the prototype, `harness_3d` imports the existing `Agent` and tmux backends from
`harness_2d.agents`. It does not duplicate them or refactor the working 2D package into a new
shared package. Shared infrastructure can be extracted later if both harnesses justify it.

Keep `python main.py ...` as the existing 2D entrypoint. Use `python -m harness_3d ...` for the
new harness rather than breaking the current command shape.

Add an Ursina version compatible with the project's Python requirement and verify the resolved
constraint in the lockfile during implementation.

## 10. Implementation sequence and verification

No test files are added, per the project convention. Each stage still has a concrete check.

1. **Geometry, handler, persistence**  
   Verify: procedural primitives round-trip through JSON; generated meshes contain finite
   vertices, valid triangle indices, stable bounds, and deterministic output.

2. **Collision and world validation**  
   Verify: known inside/outside points, boundary crossings, blocked movement increments, and the
   edge-validated witness path behave correctly in a one-off smoke script.

3. **Generation and procedural fallback**  
   Verify: `python -m harness_3d generate --offline` creates a loadable, reachable world without
   importing Ursina or calling an agent.

4. **Observation, memory, and waypoint movement**  
   Verify: memory gains only IDs returned by sensor queries; scripted waypoint batches orient,
   move, collide, reveal primitives, and serialize without an agent call.

5. **Ursina rendering and manual policy**  
   Verify: a saved world opens, all meshes and colours render, the agreed keyboard and mouse
   controls respect collision, and closing the window exits cleanly.

6. **Agent generation/navigation and replay**  
   Verify: malformed scripted payloads fall back safely; then, only with explicit approval, run
   one real generated episode and replay its saved trajectory.

7. **Minimal docs update**  
   Add only the 3D entrypoint, architecture summary, and explicit parameter-policy departure to
   README.md and AGENTS.md.

## 11. Agreed architecture decisions

1. **Core claim:** a fully playable, visually coherent 3D game harness can use superquadrics
   for all generated world geometry because their parameters are expressive, compact, and
   directly understandable by an agent.
2. **Primitive equation:** support the full safe parameter range of the one standard closed
   superquadric equation used by the referenced handler; no separate toroid/hyperboloid types.
3. **Generation brief:** the configured brief is optional and used only for agent generation;
   without one, the agent invents the world. Offline generation ignores it.
4. **Observation:** range + camera frustum + sampled line of sight. Once any part is detected,
   the full validated parameters are revealed and may be remembered. Hidden geometry, meshes,
   screenshots, and the reachability graph are never exposed.
5. **Agent movement:** the agent returns batches of absolute target positions only. The camera
   faces the current segment target; the harness interpolates but never reroutes. Independent
   agent look actions are outside the initial prototype.
6. **Manual movement:** `WASD` controls horizontal movement, `Space`/`Shift` control vertical
   movement, and the mouse controls look direction independently.
7. **Collision:** deterministic sphere-versus-generated-mesh collision with an analytic
   inside check, configurable mesh resolution, and small movement increments.
8. **Key placement:** the harness selects a reachable, initially unseen position near an
   assembly after validating the generated world.
9. **Rendering:** all primitives render normally; parameter-sensor visibility never toggles
   rendered entities. The demo view is first-person only.
10. **Policies and offline mode:** the prototype has `agent` and `manual` policies only.
    `run --offline` means procedural generation followed by manual play.
11. **Package reuse:** `harness_3d` temporarily imports `harness_2d.agents`; shared-package
    extraction is deferred.
12. **World density:** initially 32-80 primitives arranged into sparse landmark assemblies.
