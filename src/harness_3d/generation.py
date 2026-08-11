"""Agent-driven world generation, defensive parsing, and reachability.

One agent call designs a whole world — theme, palette, composition rules and a
bounded list of superquadrics — because independent spatial chunks produce
disagreeing seams and palettes (3D_PLAN.md section 5).

Nothing the model returns becomes runtime geometry directly. Every primitive is
re-parsed against hard numeric bounds, anything malformed is dropped, and a world
that ends up too thin — or that no reachability witness can be found in — is
replaced wholesale by the procedural fallback. Partially repairing a failed
composition preserves neither its aesthetics nor its navigability.

Spawn and key are placed by the harness, never by the agent: a plausible-looking
generation is otherwise perfectly capable of being unwinnable.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

from harness_2d.agents.base import Agent, AgentError
from harness_2d.pipeline_log import PipelineLogger

from .geometry import FORWARD, Vec3, apply, rotation_matrix
from .scene import Player, Scene, key_primitive
from .superquadrics import OBSTACLE, Sensor, Superquadric, SuperquadricHandler

#: Integer grid coordinate of a free-space voxel in the reachability witness.
Cell = tuple[int, int, int]

WORLD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "description": "Short name for this world, e.g. 'Drifting Kilns'."},
        "description": {"type": "string", "description": "2-3 sentences on the spatial character of the world."},
        "palette": {
            "type": "array",
            "description": "3-6 hex colours, e.g. '#c8552a', used by every primitive via its index.",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "composition_rules": {
            "type": "array",
            "description": "3-6 concrete rules the primitives below actually follow.",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
        },
        "primitives": {
            "type": "array",
            "description": "The superquadrics that make up the world.",
            "items": {
                "type": "object",
                "properties": {
                    "assembly": {"type": "string", "description": "Short landmark label shared by the parts of one composition."},
                    "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "rotation": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "scale": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "exponents": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "color": {"type": "integer", "description": "Index into `palette`."},
                },
                "required": ["assembly", "position", "rotation", "scale", "exponents", "color"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["theme", "description", "palette", "composition_rules", "primitives"],
    "additionalProperties": False,
}

#: Palettes for the procedural fallback. It is not uniform noise, so it needs
#: coherent colour just as much as an agent-designed world does.
FALLBACK_PALETTES = [
    ("Rust Terraces", ["#b4552f", "#d98c4a", "#6b3f2a", "#e8c39e"]),
    ("Cold Foundry", ["#3f6d8c", "#7fa9c4", "#26323d", "#c3d5de"]),
    ("Moss Reliquary", ["#4f7a45", "#8fb573", "#2c3d2a", "#d3d9a8"]),
    ("Violet Kilns", ["#6b4b8a", "#a382c4", "#2f2338", "#e0d2ef"]),
]


def parse_color(text: str) -> tuple[float, float, float]:
    """'#c8552a' -> sRGB floats. Raises on anything else."""
    value = str(text).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        raise ValueError(f"not a hex colour: {text!r}")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@dataclass
class WorldGenerator:
    """Builds a validated, provably navigable `Scene`."""

    #: Only `generate_offline` may be called with agent=None.
    agent: Agent | None
    bounds: float = 60.0
    player_radius: float = 0.6
    move_increment: float = 0.5
    collision_resolution: int = 8
    target_primitives: int = 48
    min_primitives: int = 12
    max_primitives: int = 80
    exponent_range: tuple[float, float] = (0.1, 4.0)
    validation_spacing: float = 3.0
    min_free_voxels: int = 400
    min_key_distance: float = 24.0
    key_scale: float = 0.9
    sensor: Sensor = field(default_factory=Sensor)
    log: PipelineLogger = field(default_factory=lambda: PipelineLogger("gen3d"))
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        #: Nothing thinner than the player can be a solid obstacle: a sphere that
        #: does not fit could tunnel through it between two movement increments.
        self.min_scale = 2 * self.player_radius + self.move_increment
        self.max_scale = self.bounds / 4

    @property
    def half(self) -> float:
        return self.bounds / 2

    # --------------------------------------------------------------- top level

    def generate(self, brief: str | None = None) -> Scene:
        design, primitives, source = self._request(brief)
        self.log.info("generate", f"theme: {design['theme']} — {design['description']}")
        scene = self._assemble(design, primitives, source)
        if scene is not None:
            return scene
        # A world can parse perfectly and still be unnavigable. There is nothing
        # to repair here without wrecking the composition, so start over.
        self.log.info("generate", "no reachable free space in the generated world; using the fallback")
        return self._offline_scene("fallback:unreachable")

    def generate_offline(self) -> Scene:
        """Procedural world with no agent calls — for smoke tests and demos."""
        return self._offline_scene("procedural")

    def _offline_scene(self, source: str) -> Scene:
        for _ in range(8):
            design, primitives = self._procedural_world()
            scene = self._assemble(design, primitives, source)
            if scene is not None:
                return scene
        raise RuntimeError("procedural generation produced no navigable world in 8 attempts")

    # ------------------------------------------------------------------ agent

    def _request(self, brief: str | None) -> tuple[dict, list[Superquadric], str]:
        self.log.start("generate", f"requesting ~{self.target_primitives} superquadrics "
                                   f"in a {self.bounds:.0f}-unit cube")
        try:
            payload = self.agent.run(self._prompt(brief), WORLD_SCHEMA)
            design, primitives = self._parse(payload)
            self.log.end("generate", f"{len(primitives)} valid primitives across "
                                     f"{len({p.assembly for p in primitives})} assemblies")
            return design, primitives, "agent"
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            # One bad call must not sink the run — same contract as the 2D harness.
            self.log.info("generate", f"world generation fell back to procedural: {exc}")
            design, primitives = self._procedural_world()
            design["description"] = brief or design["description"]
            return design, primitives, f"fallback:{type(exc).__name__}"

    def _prompt(self, brief: str | None) -> str:
        if brief:
            intent = (f"The operator asked for this environment:\n\"{brief}\"\n"
                      "Turn that request into concrete superquadric parameters.")
        else:
            intent = ("Nobody has specified what this world should be. Decide for yourself, "
                      "and pick something with distinctive spatial structure rather than "
                      "scattered noise.")
        half = self.half
        lo, hi = self.exponent_range
        return f"""You are designing a 3D game world made entirely of superquadrics.

{intent}

The world is a cube centred on the origin: every axis runs from {-half:.0f} to {half:.0f}
(+X right, +Y up, +Z forward). A player flies through it freely — there is no
gravity and no ground plane — searching for a hidden key.

Each primitive is the standard superquadric surface, in its own local frame:

  x = ax * sign(cos eta)|cos eta|^e1 * sign(cos omega)|cos omega|^e2
  y = ay * sign(cos eta)|cos eta|^e1 * sign(sin omega)|sin omega|^e2
  z = az * sign(sin eta)|sin eta|^e1

so `scale = (ax, ay, az)` are the semiaxes and `exponents = (e1, e2)` set the
shape. Useful combinations:

  (1.0, 1.0)   ellipsoid          (0.2, 0.2)   box
  (0.2, 1.0)   cylinder along z   (1.0, 0.2)   rounded slab, square in xy
  (2.0, 2.0)   octahedral gem     (3.0, 1.0)   pinched, concave hourglass

`rotation` is XYZ Euler degrees; a primitive's local +z points along world +y
after a rotation of (-90, 0, 0), which is how you stand a cylinder upright.

BUDGET AND BOUNDS
- Return about {self.target_primitives} primitives (hard limits: {self.min_primitives}-{self.max_primitives}).
- Every `scale` component must be between {self.min_scale:.1f} and {self.max_scale:.0f}.
  Anything thinner than {self.min_scale:.1f} is not a solid the player can be stopped by.
- Every `exponents` value must be between {lo} and {hi}.
- No primitive may cross the cube boundary: its whole rotated extent must stay
  inside [{-half:.0f}, {half:.0f}] on every axis.
- `color` is an index into your own `palette`.

COMPOSITION
- Build 5-9 distinct named assemblies (arches, towers, rings, wrecks, reefs...).
  Every primitive carries the `assembly` label of the composition it belongs to.
- Inside an assembly, overlap primitives deliberately so they read as one solid
  object rather than as floating parts.
- Between assemblies, leave open flight corridors — a player must be able to fly
  from any assembly to any other without threading a gap narrower than 4 units.
- Spread the assemblies across all eight octants, including below y=0.
- The composition_rules you state must be the ones the primitives actually follow.
"""

    # -------------------------------------------------------------- validation

    def _parse(self, payload: dict) -> tuple[dict, list[Superquadric]]:
        """Rebuild the world from the model's JSON, dropping anything invalid.

        Every field is bounds-checked before it can become geometry. Agent IDs
        are ignored entirely and reassigned, so a duplicate or missing one cannot
        collide with the key or with an observation the agent already holds.
        """
        palette = [parse_color(c) for c in payload["palette"][:6]]
        if len(palette) < 3:
            raise ValueError("palette needs at least 3 colours")

        raw = payload["primitives"]
        if not isinstance(raw, list):
            raise TypeError("primitives is not a list")
        # The count cap is enforced here, before anything is meshed.
        kept: list[Superquadric] = []
        dropped = 0
        for item in raw[: self.max_primitives]:
            try:
                kept.append(self._parse_primitive(item, len(kept), palette))
            except (ValueError, KeyError, TypeError, IndexError):
                dropped += 1
        if len(kept) < self.min_primitives:
            raise ValueError(f"only {len(kept)} of {len(raw)} primitives survived validation")
        if dropped:
            self.log.info("generate", f"dropped {dropped} invalid primitive(s)")

        design = {
            "theme": str(payload["theme"])[:80],
            "description": str(payload["description"])[:600],
            "composition_rules": [str(r)[:200] for r in payload.get("composition_rules", [])[:6]],
            "palette": [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in palette],
        }
        return design, kept

    def _parse_primitive(self, item: dict, prim_id: int, palette: list) -> Superquadric:
        position = Vec3.parse(item["position"])
        rotation = tuple(float(a) % 360.0 for a in Vec3.parse(item["rotation"]).as_tuple())
        scale = Vec3.parse(item["scale"]).as_tuple()
        exponents = tuple(float(e) for e in item["exponents"])

        if any(not (self.min_scale <= a <= self.max_scale) for a in scale):
            raise ValueError("scale out of bounds")
        if any(not math.isfinite(e) or not (self.exponent_range[0] <= e <= self.exponent_range[1])
               for e in exponents):
            raise ValueError("exponent out of bounds")
        if max(abs(c) for c in position.as_tuple()) > self.half:
            raise ValueError("position outside the cube")

        prim = Superquadric(
            id=prim_id,
            kind=OBSTACLE,
            assembly=str(item.get("assembly", "unnamed"))[:32].strip() or "unnamed",
            position=position,
            rotation=rotation,
            scale=scale,
            exponents=exponents,
            color=palette[int(item["color"]) % len(palette)],
        )
        lo, hi = prim.aabb()
        if max(abs(c) for c in (*lo, *hi)) > self.half:
            raise ValueError("primitive crosses the world boundary")
        return prim

    # ------------------------------------------------------- procedural world

    def _procedural_world(self) -> tuple[dict, list[Superquadric]]:
        rng = self._rng
        theme, hexes = rng.choice(FALLBACK_PALETTES)
        palette = [parse_color(c) for c in hexes]
        templates = (self._arch, self._ring, self._cluster, self._columns)

        primitives: list[Superquadric] = []
        # One anchor per octant, jittered, so the fallback still covers the cube.
        octants = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        rng.shuffle(octants)
        reach = self.half * 0.55
        for index, (sx, sy, sz) in enumerate(octants[: rng.randint(6, 8)]):
            anchor = Vec3(sx * rng.uniform(0.35, 1.0) * reach,
                          sy * rng.uniform(0.25, 0.9) * reach,
                          sz * rng.uniform(0.35, 1.0) * reach)
            template = templates[index % len(templates)]
            for prim in template(anchor, f"{template.__name__.strip('_')}-{index}", palette):
                if len(primitives) >= self.max_primitives:
                    break
                prim.id = len(primitives)
                if self._fits(prim):
                    primitives.append(prim)

        design = {
            "theme": theme,
            "description": "Bounded random assemblies: arches, rings, branching clusters and "
                           "offset columns, placed one per octant.",
            "composition_rules": ["one assembly per occupied octant",
                                  "parts of an assembly overlap deliberately",
                                  "assemblies are separated by open flight corridors"],
            "palette": list(hexes),
        }
        return design, primitives

    def _fits(self, prim: Superquadric) -> bool:
        lo, hi = prim.aabb()
        return max(abs(c) for c in (*lo, *hi)) <= self.half

    def _make(self, assembly: str, position: Vec3, rotation, scale, exponents, palette) -> Superquadric:
        scale = tuple(max(self.min_scale, min(self.max_scale, a)) for a in scale)
        return Superquadric(id=0, kind=OBSTACLE, assembly=assembly, position=position,
                            rotation=tuple(float(a) for a in rotation), scale=scale,
                            exponents=exponents, color=self._rng.choice(palette))

    def _arch(self, anchor: Vec3, name: str, palette) -> list[Superquadric]:
        rng = self._rng
        span, height, thickness = rng.uniform(4, 8), rng.uniform(5, 9), rng.uniform(1.0, 1.8)
        legs = [self._make(name, anchor + Vec3(side * span, 0, 0), (-90, 0, 0),
                           (thickness, thickness, height), (0.2, rng.choice([0.3, 1.0])), palette)
                for side in (-1, 1)]
        # Local +z runs along world +x after a 90-degree turn about y, which is
        # what lays the lintel across the two legs.
        lintel = self._make(name, anchor + Vec3(0, height, 0), (0, 90, 0),
                            (thickness, thickness, span + thickness), (0.25, 0.25), palette)
        return [*legs, lintel]

    def _ring(self, anchor: Vec3, name: str, palette) -> list[Superquadric]:
        rng = self._rng
        count = rng.randint(5, 8)
        radius = rng.uniform(4, 8)
        tilt = rng.uniform(-50, 50)
        blob = rng.uniform(1.2, 2.4)
        matrix = rotation_matrix((tilt, 0.0, 0.0))
        parts = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            offset = Vec3(*apply(matrix, (math.cos(angle) * radius, 0.0, math.sin(angle) * radius)))
            parts.append(self._make(name, anchor + offset, (tilt, math.degrees(angle), 0),
                                    (blob, blob, blob * rng.uniform(1.0, 1.8)),
                                    (rng.uniform(0.4, 1.4), rng.uniform(0.4, 1.4)), palette))
        return parts

    def _cluster(self, anchor: Vec3, name: str, palette) -> list[Superquadric]:
        rng = self._rng
        core = rng.uniform(2.5, 4.5)
        parts = [self._make(name, anchor, (0, 0, 0), (core, core, core),
                            (rng.uniform(0.6, 1.6), rng.uniform(0.6, 1.6)), palette)]
        for _ in range(rng.randint(3, 5)):
            direction = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)).normalized()
            length = rng.uniform(3, 6)
            # Branches start inside the core, so the assembly reads as one solid.
            parts.append(self._make(name, anchor + direction * (core * 0.7),
                                    (rng.uniform(0, 180), rng.uniform(0, 180), rng.uniform(0, 180)),
                                    (self.min_scale * 1.2, self.min_scale * 1.2, length),
                                    (rng.uniform(0.2, 0.8), rng.uniform(0.2, 1.0)), palette))
        return parts

    def _columns(self, anchor: Vec3, name: str, palette) -> list[Superquadric]:
        rng = self._rng
        parts = []
        for _ in range(rng.randint(3, 5)):
            offset = Vec3(rng.uniform(-6, 6), rng.uniform(-3, 3), rng.uniform(-6, 6))
            width = rng.uniform(1.0, 2.2)
            parts.append(self._make(name, anchor + offset, (-90, rng.uniform(0, 90), 0),
                                    (width, width, rng.uniform(4, 9)),
                                    (rng.uniform(0.15, 0.4), rng.choice([0.2, 1.0])), palette))
        return parts

    # ------------------------------------------------- reachability + placement

    def _assemble(self, design: dict, primitives: list[Superquadric], source: str) -> Scene | None:
        """Turn validated primitives into a Scene with a proven route to the key.

        Returns None when no large enough free component exists, or when no
        candidate key position satisfies the distance and unseen requirements.
        """
        handler = SuperquadricHandler(primitives, collision_resolution=self.collision_resolution)
        graph = self._free_component(handler)
        if graph is None:
            return None
        nodes, component, distances, parents, spawn_cell = graph

        spawn = nodes[spawn_cell]
        key_cell = self._choose_key(handler, spawn, nodes, component, distances)
        if key_cell is None:
            return None

        key_node = nodes[key_cell]
        key_id = handler.next_id()
        handler.add(key_primitive(key_id, key_node, self.key_scale))
        witness = self._witness_length(nodes, parents, key_cell)

        scene = Scene(
            bounds=self.bounds,
            primitives=handler,
            player=Player(position=spawn, forward=FORWARD, radius=self.player_radius),
            key_id=key_id,
            meta={
                **design,
                "source": source,
                "primitives": len(primitives),
                "assemblies": sorted({p.assembly for p in primitives}),
                "validation_spacing": self.validation_spacing,
                "free_component": len(component),
                "witness_hops": distances[key_cell],
                "witness_path_length": round(witness, 1),
            },
        )
        self.log.end("generate", f"{len(primitives)} primitives, free component {len(component)} voxels, "
                                 f"spawn {spawn} -> key {key_node} ({witness:.0f} units of witness path)")
        return scene

    def _free_nodes(self, handler: SuperquadricHandler) -> dict[Cell, Vec3]:
        """Voxel centres where the player fits, using the movement clearance test.

        Keyed by integer cell so neighbour lookups are exact — float coordinates
        rebuilt by two different additions do not reliably compare equal.
        """
        limit = self.half - self.player_radius
        count = max(2, int((2 * limit) // self.validation_spacing) + 1)
        origin = -(count - 1) * self.validation_spacing / 2
        nodes: dict[Cell, Vec3] = {}
        for i in range(count):
            for j in range(count):
                for k in range(count):
                    point = Vec3(origin + i * self.validation_spacing,
                                 origin + j * self.validation_spacing,
                                 origin + k * self.validation_spacing)
                    if not handler.is_blocked(point, self.player_radius):
                        nodes[(i, j, k)] = point
        return nodes

    def _free_component(self, handler: SuperquadricHandler):
        """Largest free component found from a few random seeds, with distances.

        Edges are validated with the same clearance predicate movement uses, so
        the graph is a proof witness: every edge it keeps is a segment the player
        can actually fly. It is generation-time only — neither policy can see it,
        and runtime movement stays continuous rather than snapping to it.
        """
        nodes = self._free_nodes(handler)
        if len(nodes) < self.min_free_voxels:
            self.log.info("generate", f"only {len(nodes)} free voxels at spacing "
                                      f"{self.validation_spacing}")
            return None

        neighbourhood = [(dx, dy, dz)
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                         if (dx, dy, dz) != (0, 0, 0)]
        cells = list(nodes)
        best = None
        for _ in range(6):
            seed = self._rng.choice(cells)
            distances: dict[Cell, int] = {seed: 0}
            parents: dict[Cell, Cell | None] = {seed: None}
            component = [seed]
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                here = nodes[current]
                for dx, dy, dz in neighbourhood:
                    cell = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if cell in distances or cell not in nodes:
                        continue
                    if not handler.segment_is_clear(here, nodes[cell], self.player_radius,
                                                    self.move_increment):
                        continue
                    distances[cell] = distances[current] + 1
                    parents[cell] = current
                    component.append(cell)
                    queue.append(cell)
            if best is None or len(component) > len(best[0]):
                best = (component, distances, parents, seed)
            if len(component) >= self.min_free_voxels:
                break

        component, distances, parents, seed = best
        if len(component) < self.min_free_voxels:
            self.log.info("generate", f"largest free component is {len(component)} voxels, "
                                      f"below the required {self.min_free_voxels}")
            return None
        return nodes, component, distances, parents, seed

    def _choose_key(self, handler: SuperquadricHandler, spawn: Vec3, nodes: dict[Cell, Vec3],
                    component: list[Cell], distances: dict[Cell, int]) -> Cell | None:
        """A far, initially unseen voxel centre that sits close to an assembly."""
        min_hops = max(1, math.ceil(self.min_key_distance / self.validation_spacing))
        far = [cell for cell in component if distances[cell] >= min_hops]
        if not far:
            return None
        obstacles = handler.obstacles
        if obstacles:
            # "Near an assembly": a key floating in open space is neither
            # findable nor interesting.
            far.sort(key=lambda cell: min(nodes[cell].distance_to(o.position) - o.bounding_radius
                                          for o in obstacles))
        candidates = far[: max(12, len(far) // 8)]
        self._rng.shuffle(candidates)

        probe = handler.next_id()
        for cell in candidates:
            handler.add(key_primitive(probe, nodes[cell], self.key_scale))
            seen = any(p.id == probe for p in handler.visible_from(spawn, FORWARD, self.sensor))
            handler.remove(probe)
            if not seen:
                return cell
        return None

    def _witness_length(self, nodes: dict[Cell, Vec3], parents: dict[Cell, Cell | None],
                        key_cell: Cell) -> float:
        """Length of the voxel path the reachability proof actually found."""
        total = 0.0
        cell = key_cell
        while parents.get(cell) is not None:
            previous = parents[cell]
            total += nodes[cell].distance_to(nodes[previous])
            cell = previous
        return total
