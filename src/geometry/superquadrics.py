"""The one primitive the whole world is made of, and the handler that owns them.

Every piece of generated world geometry is a superquadric:

    x = ax * signed_cos(eta, e1) * signed_cos(omega, e2)
    y = ay * signed_cos(eta, e1) * signed_sin(omega, e2)
    z = az * signed_sin(eta, e1)

`eta` is latitude, `omega` longitude, `scale = (ax, ay, az)` and
`exponents = (e1, e2)`. Sweeping those covers spheres, ellipsoids, boxes,
cylinders, diamonds and pinched forms without an enumerated shape catalogue.

`SuperquadricHandler` is the authoritative geometry service: it meshes primitives
in pure Python (no engine import), answers collision from those meshes rather
than from pixels or from Ursina's colliders, and decides what the sensor can see.
Rendering is optional; everything here works headless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Matrix3, Vec3, apply, apply_inverse, basis, rotation_matrix

#: `kind` decides what touching a primitive means. Only the harness ever creates
#: a door, object, lock or portal; the generation agents can only make obstacles.
OBSTACLE = "obstacle"   # solid, blocks
DOOR = "door"           # solid, unlock attempt on touch
OBJECT = "object"       # not solid, pickable object primitive
LOCK = "lock"           # not solid, display only — the target parameters
PORTAL = "portal"       # not solid, display only — what an unlocked door becomes

STRUCTURAL_KEYWORDS = ("wall", "floor", "ceiling", "door", "frame", "portal", "lock", "shell")


def _is_structural(prim: Superquadric) -> bool:
    if prim.kind in (DOOR, PORTAL, LOCK):
        return True
    low = prim.assembly.lower().strip()
    return not low or any(k in low for k in STRUCTURAL_KEYWORDS)

Triple = tuple[float, float, float]


def _signed_pow(value: float, exponent: float) -> float:
    """`sign(v) * |v| ** e` — the signed power the superquadric equation needs."""
    if value == 0.0:
        return 0.0
    magnitude = abs(value) ** exponent
    return magnitude if value > 0.0 else -magnitude


@dataclass
class Superquadric:
    """One validated primitive. Positions are world-space; angles are XYZ degrees."""

    id: int
    kind: str
    assembly: str
    position: Vec3
    rotation: Triple
    scale: Triple
    exponents: tuple[float, float]
    color: Triple

    matrix: Matrix3 = field(init=False, repr=False)
    #: Radius of the sphere around `position` that contains the primitive. The
    #: parametric form never exceeds `scale` on any local axis, so the local
    #: box's corner bounds every shape the exponents can produce.
    bounding_radius: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.matrix = rotation_matrix(self.rotation)
        self.bounding_radius = math.sqrt(sum(a * a for a in self.scale))

    @property
    def is_solid(self) -> bool:
        """`kind` alone decides collision semantics — a separate `solid` flag
        could contradict it. A door blocks like a wall until it is unlocked."""
        return self.kind in (OBSTACLE, DOOR)

    # ----------------------------------------------------------------- geometry

    def to_local(self, point: Triple) -> Triple:
        p = (point[0] - self.position.x, point[1] - self.position.y, point[2] - self.position.z)
        return apply_inverse(self.matrix, p)

    def to_world(self, local: Triple) -> Triple:
        x, y, z = apply(self.matrix, local)
        return (x + self.position.x, y + self.position.y, z + self.position.z)

    def implicit(self, point: Triple) -> float:
        """Inside-outside function: < 1 inside, 1 on the surface, > 1 outside."""
        x, y, z = self.to_local(point)
        ax, ay, az = self.scale
        # |x| <= ax on every axis for any exponent, so the local box is an exact
        # early-out — and it keeps the fractional powers below from overflowing.
        if abs(x) > ax or abs(y) > ay or abs(z) > az:
            return math.inf
        e1, e2 = self.exponents
        radial = (abs(x / ax) ** (2.0 / e2) + abs(y / ay) ** (2.0 / e2)) ** (e2 / e1)
        return radial + abs(z / az) ** (2.0 / e1)

    def contains(self, point: Triple) -> bool:
        return self.implicit(point) <= 1.0

    def surface_point(self, eta: float, omega: float) -> Triple:
        ax, ay, az = self.scale
        e1, e2 = self.exponents
        cos_eta = _signed_pow(math.cos(eta), e1)
        return self.to_world((
            ax * cos_eta * _signed_pow(math.cos(omega), e2),
            ay * cos_eta * _signed_pow(math.sin(omega), e2),
            az * _signed_pow(math.sin(eta), e1),
        ))

    def axis_extrema(self) -> list[Triple]:
        """The six local-axis surface points, in world space.

        Exact for every exponent pair: each sits where one signed power is 1 and
        the others are 0.
        """
        ax, ay, az = self.scale
        return [self.to_world(local) for local in
                ((ax, 0.0, 0.0), (-ax, 0.0, 0.0), (0.0, ay, 0.0),
                 (0.0, -ay, 0.0), (0.0, 0.0, az), (0.0, 0.0, -az))]

    def aabb(self, margin: float = 0.0) -> tuple[Triple, Triple]:
        """World-space bounds of the rotated local box, optionally expanded.

        Used as the broad phase: a transformed AABB rejects distant primitives
        before any triangle is touched.
        """
        ax, ay, az = self.scale
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for sx in (-ax, ax):
            for sy in (-ay, ay):
                for sz in (-az, az):
                    corner = self.to_world((sx, sy, sz))
                    for axis in range(3):
                        lo[axis] = min(lo[axis], corner[axis])
                        hi[axis] = max(hi[axis], corner[axis])
        return ((lo[0] - margin, lo[1] - margin, lo[2] - margin),
                (hi[0] + margin, hi[1] + margin, hi[2] + margin))

    # ---------------------------------------------------------- (de)serialisation

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "assembly": self.assembly,
            # `float()` before rounding so a value that arrived as an int is
            # persisted as one type: the saved world is compared and reloaded.
            "position": list(self.position.rounded(3)),
            "rotation": [round(float(a), 2) for a in self.rotation],
            "scale": [round(float(a), 3) for a in self.scale],
            "exponents": [round(float(e), 3) for e in self.exponents],
            "color": [round(float(c), 3) for c in self.color],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Superquadric":
        return cls(
            id=int(data["id"]),
            kind=str(data["kind"]),
            assembly=str(data.get("assembly", "")),
            position=Vec3.parse(data["position"]),
            rotation=tuple(float(a) for a in data["rotation"]),
            scale=tuple(float(a) for a in data["scale"]),
            exponents=tuple(float(e) for e in data["exponents"]),
            color=tuple(float(c) for c in data["color"]),
        )


@dataclass(frozen=True)
class MeshData:
    """Engine-independent triangle soup: plain lists an `ursina.Mesh` can eat."""

    vertices: list[Triple]
    triangles: list[tuple[int, int, int]]
    normals: list[Triple]
    colors: list[tuple[float, float, float, float]]


def build_mesh(prim: Superquadric, resolution: int) -> MeshData:
    """Tessellate one primitive into world-space triangles.

    `resolution` is the number of latitude bands; longitude gets twice that. The
    two poles are single shared vertices, so no degenerate triangles reach the
    collision code. Normals are accumulated from face normals rather than taken
    from the analytic gradient, which is undefined at the creases sharp
    exponents produce.
    """
    bands = max(3, int(resolution))
    sectors = bands * 2

    vertices: list[Triple] = [prim.surface_point(-math.pi / 2, 0.0)]
    rings: list[list[int]] = []
    for i in range(1, bands):
        eta = -math.pi / 2 + math.pi * i / bands
        ring = []
        for j in range(sectors):
            omega = -math.pi + 2 * math.pi * j / sectors
            ring.append(len(vertices))
            vertices.append(prim.surface_point(eta, omega))
        rings.append(ring)
    north = len(vertices)
    vertices.append(prim.surface_point(math.pi / 2, 0.0))

    triangles: list[tuple[int, int, int]] = []
    first = rings[0]
    for j in range(sectors):
        triangles.append((0, first[(j + 1) % sectors], first[j]))
    for lower, upper in zip(rings, rings[1:]):
        for j in range(sectors):
            k = (j + 1) % sectors
            triangles.append((lower[j], lower[k], upper[j]))
            triangles.append((lower[k], upper[k], upper[j]))
    last = rings[-1]
    for j in range(sectors):
        triangles.append((north, last[j], last[(j + 1) % sectors]))

    normals = _smooth_normals(vertices, triangles, prim.position.as_tuple())
    # Wound above so the right-hand rule points out of the surface, which is what
    # makes the normals outward — but Ursina's world is left-handed (y-up-left),
    # so that same order rasterises clockwise and every front face is culled,
    # leaving the shapes looking hollow. Reversing here, after the normals are
    # taken, renders them solid without touching the normals. Collision is
    # winding-agnostic, so it reads either order identically.
    triangles = [(a, c, b) for a, b, c in triangles]
    color = (*prim.color, 1.0)
    return MeshData(vertices=vertices, triangles=triangles, normals=normals,
                    colors=[color] * len(vertices))


def _smooth_normals(vertices: list[Triple], triangles: list[tuple[int, int, int]],
                    centre: Triple) -> list[Triple]:
    sums = [[0.0, 0.0, 0.0] for _ in vertices]
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        for index in (i, j, k):
            sums[index][0] += n[0]
            sums[index][1] += n[1]
            sums[index][2] += n[2]
    normals: list[Triple] = []
    for index, (nx, ny, nz) in enumerate(sums):
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1e-12:
            normals.append((nx / length, ny / length, nz / length))
            continue
        # A vertex whose faces cancel out (a fully pinched crease): point away
        # from the centre so the shader still has something to light.
        vx, vy, vz = vertices[index]
        radial = (vx - centre[0], vy - centre[1], vz - centre[2])
        length = math.sqrt(sum(c * c for c in radial))
        normals.append(tuple(c / length for c in radial) if length > 1e-12 else (0.0, 1.0, 0.0))
    return normals


# ------------------------------------------------------------- triangle maths


def _closest_point_on_triangle(p: Triple, a: Triple, b: Triple, c: Triple) -> Triple:
    """Nearest point of triangle abc to p (Ericson, Real-Time Collision Detection)."""
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]
    d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2]
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = (p[0] - b[0], p[1] - b[1], p[2] - b[2])
    d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2]
    d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2]
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3) if d1 != d3 else 0.0
        return (a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t)

    cp = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2]
    d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2]
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6) if d2 != d6 else 0.0
        return (a[0] + ac[0] * t, a[1] + ac[1] * t, a[2] + ac[2] * t)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denominator = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / denominator if denominator != 0.0 else 0.0
        return (b[0] + (c[0] - b[0]) * t, b[1] + (c[1] - b[1]) * t, b[2] + (c[2] - b[2]) * t)

    denominator = va + vb + vc
    if denominator <= 0.0:  # degenerate triangle; a is as good as anything
        return a
    v, w = vb / denominator, vc / denominator
    return (a[0] + ab[0] * v + ac[0] * w,
            a[1] + ab[1] * v + ac[1] * w,
            a[2] + ab[2] * v + ac[2] * w)


def _segment_hits_triangle(origin: Triple, delta: Triple, a: Triple, b: Triple, c: Triple) -> bool:
    """Möller-Trumbore, restricted to the segment `origin -> origin + delta`."""
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    h = (delta[1] * e2[2] - delta[2] * e2[1],
         delta[2] * e2[0] - delta[0] * e2[2],
         delta[0] * e2[1] - delta[1] * e2[0])
    det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2]
    if -1e-12 < det < 1e-12:
        return False
    inv = 1.0 / det
    s = (origin[0] - a[0], origin[1] - a[1], origin[2] - a[2])
    u = inv * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2])
    if u < 0.0 or u > 1.0:
        return False
    q = (s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0])
    v = inv * (delta[0] * q[0] + delta[1] * q[1] + delta[2] * q[2])
    if v < 0.0 or u + v > 1.0:
        return False
    t = inv * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2])
    # Both ends are open: the ray starts at the eye and ends *on* the candidate's
    # own surface, and neither endpoint should count as an occlusion.
    return 1e-6 < t < 1.0 - 1e-6


def _ray_hits_aabb(origin: Triple, delta: Triple, lo: Triple, hi: Triple) -> bool:
    """Slab test for the segment `origin -> origin + delta` against a box.

    The occluder broad phase. A bounding *sphere* is a fine bound for a blob and
    a useless one for a slab — a 0.4 x 4 x 23 wall gets a 23-unit rejection
    sphere that covers the whole building, so no ray anywhere is ever rejected by
    it. A wall's AABB is thin even when its bounding sphere is enormous, which is
    exactly the case the sphere could not reject.
    """
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        d = delta[axis]
        if abs(d) < 1e-12:  # parallel to this slab: inside it, or missing entirely
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return False
            continue
        near, far = (lo[axis] - origin[axis]) / d, (hi[axis] - origin[axis]) / d
        if near > far:
            near, far = far, near
        t0, t1 = max(t0, near), min(t1, far)
        if t0 > t1:
            return False
    return True


def _nearest_surface_point(mesh: MeshData, p: Triple) -> Triple:
    """Closest point of a tessellated surface to `p`."""
    vertices = mesh.vertices
    best, best_distance = vertices[0], math.inf
    for i, j, k in mesh.triangles:
        q = _closest_point_on_triangle(p, vertices[i], vertices[j], vertices[k])
        distance = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
        if distance < best_distance:
            best, best_distance = q, distance
    return best


def _aabb_overlaps_point(lo: Triple, hi: Triple, p: Triple) -> bool:
    return (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2])


# -------------------------------------------------------------------- handler


@dataclass(frozen=True)
class Sensor:
    """What the parameter sensor can reach, and how often it is polled."""

    range: float = math.inf
    fov: float = 60.0
    aspect: float = 1.6
    #: Distance flown between sensor passes along a segment. A pass dominates the
    #: cost of a movement increment, and one every 0.5 units re-detects the same
    #: primitives from half a unit further along.
    stride: float = 2.0


class SuperquadricHandler:
    """Owns the validated primitives and answers every geometric question.

    Collision lives here, not in Ursina: rendering is optional, so generation and
    headless checks have to be able to run the same predicates the episode does.
    """

    def __init__(self, primitives=(), *, collision_resolution: int = 8):
        self._primitives: dict[int, Superquadric] = {}
        self.collision_resolution = collision_resolution
        self._meshes: dict[tuple[int, int], MeshData] = {}
        self._collision: dict[int, tuple[MeshData, Triple, Triple]] = {}
        for prim in primitives:
            self.add(prim)

    def __len__(self) -> int:
        return len(self._primitives)

    def __iter__(self):
        return iter(self._primitives.values())

    def __contains__(self, prim_or_id: int | Superquadric) -> bool:
        if isinstance(prim_or_id, Superquadric):
            return prim_or_id.id in self._primitives
        return prim_or_id in self._primitives

    def add(self, prim: Superquadric) -> Superquadric:
        self._primitives[prim.id] = prim
        self._forget(prim.id)
        return prim

    def remove(self, prim_id: int) -> None:
        self._primitives.pop(prim_id, None)
        self._forget(prim_id)

    def _forget(self, prim_id: int) -> None:
        """Drop cached geometry for one ID. Reusing an ID for different
        parameters — key placement probes candidate positions this way — must
        not leave the old mesh answering collision queries."""
        self._collision.pop(prim_id, None)
        for key in [k for k in self._meshes if k[0] == prim_id]:
            del self._meshes[key]

    def get(self, prim_id: int) -> Superquadric:
        return self._primitives[prim_id]

    def next_id(self) -> int:
        return max(self._primitives, default=-1) + 1

    @property
    def obstacles(self) -> list[Superquadric]:
        return [p for p in self._primitives.values() if p.is_solid]

    # ---------------------------------------------------------------- meshing

    def mesh_data(self, prim_id: int, resolution: int) -> MeshData:
        """Cached mesh for one primitive. Static geometry is never rebuilt."""
        key = (prim_id, resolution)
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = build_mesh(self._primitives[prim_id], resolution)
            self._meshes[key] = mesh
        return mesh

    def _collision_mesh(self, prim: Superquadric) -> tuple[MeshData, Triple, Triple]:
        """The mesh collision is measured against, plus its expanded AABB."""
        cached = self._collision.get(prim.id)
        if cached is None:
            mesh = self.mesh_data(prim.id, self.collision_resolution)
            lo, hi = prim.aabb()
            cached = (mesh, lo, hi)
            self._collision[prim.id] = cached
        return cached

    # -------------------------------------------------------------- collision

    def touches(self, prim: Superquadric, position: Vec3, radius: float) -> bool:
        """Does the player sphere intersect this primitive?

        Bounding volumes reject the far field, the analytic implicit function
        catches a centre that is inside, and sphere-to-triangle distance against
        the collision mesh decides the rest. Collision therefore agrees with the
        rendered surface up to the collision tessellation.
        """
        p = position.as_tuple()
        centre = prim.position
        reach = prim.bounding_radius + radius
        if (p[0] - centre.x) ** 2 + (p[1] - centre.y) ** 2 + (p[2] - centre.z) ** 2 > reach * reach:
            return False
        mesh, lo, hi = self._collision_mesh(prim)
        if not _aabb_overlaps_point((lo[0] - radius, lo[1] - radius, lo[2] - radius),
                                    (hi[0] + radius, hi[1] + radius, hi[2] + radius), p):
            return False
        if prim.contains(p):
            return True
        radius_squared = radius * radius
        vertices = mesh.vertices
        for i, j, k in mesh.triangles:
            q = _closest_point_on_triangle(p, vertices[i], vertices[j], vertices[k])
            if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2 <= radius_squared:
                return True
        return False

    def blocking_primitive(self, position: Vec3, radius: float) -> Superquadric | None:
        for prim in self._primitives.values():
            if prim.is_solid and self.touches(prim, position, radius):
                return prim
        return None

    def touching(self, position: Vec3, radius: float) -> list[Superquadric]:
        """Every primitive the player sphere intersects, solid or not.

        `blocking_primitive` answers "am I stopped", which is all movement needs.
        Contact-driven interaction asks the different question "what did I touch",
        and has to see all of it: one increment can brush a collectable and keep
        flying, and a non-solid primitive never stops anything at all.
        """
        return [prim for prim in self._primitives.values()
                if self.touches(prim, position, radius)]

    def is_blocked(self, position: Vec3, radius: float) -> bool:
        return self.blocking_primitive(position, radius) is not None

    def segment_is_clear(self, start: Vec3, end: Vec3, radius: float, increment: float) -> bool:
        """Can the player sphere fly `start -> end` without touching anything?

        The same predicate movement uses, sampled at the same increment — which
        is what lets generation prove a route the episode can actually fly.
        """
        delta = end - start
        length = delta.length()
        steps = max(1, math.ceil(length / increment))
        for i in range(steps + 1):
            if self.is_blocked(start + delta * (i / steps), radius):
                return False
        return True

    # ------------------------------------------------------------- visibility

    def _expand_assemblies(self, seen: list[Superquadric]) -> list[Superquadric]:
        """When at least one object which makes an assembly is seen, the whole
        assembly is visible. Assemblies are grouped and behave together."""
        seen_assemblies: set[str] = set()
        for p in seen:
            if not _is_structural(p) and p.assembly:
                seen_assemblies.add(p.assembly)

        if not seen_assemblies:
            return seen

        seen_ids = {p.id for p in seen}
        result = list(seen)
        for p in self._primitives.values():
            if p.id not in seen_ids and p.assembly in seen_assemblies:
                if p.kind == LOCK or p.assembly.startswith("lock-"):
                    continue
                seen_ids.add(p.id)
                result.append(p)
        return result

    def visible_from(self, position: Vec3, forward: Vec3, sensor: Sensor) -> list[Superquadric]:
        """Primitives the parameter sensor detects from this pose.

        Range, then camera frustum, then sampled line of sight against the
        collision meshes. Once any part of an assembly is detected the caller
        gets the *whole* assembly — that information advantage over a
        pixel policy is the point of the abstraction, not an accident of it.

        This is the genuine-occlusion check and it is expensive (a raycast per
        candidate against every solid primitive's collision mesh), which is
        fine for the one-off use it was built for — world generation proving a
        peg is hidden from the spawn pose (`world/task.py`) — but too slow to
        call every frame from the live sensor. That path uses
        `visible_cone_from` instead.
        """
        eye = position.as_tuple()
        candidates = self._cone_candidates(position, forward, sensor)
        seen = [prim for _, prim in candidates if self._has_line_of_sight(eye, prim)]
        return self._expand_assemblies(seen)

    def visible_cone_from(self, position: Vec3, forward: Vec3, sensor: Sensor) -> list[Superquadric]:
        """Primitives in sensor range and field of view, ignoring occlusion.

        When at least one object which makes an assembly is seen, the whole
        assembly is visible.

        No raycasting, so it is cheap enough for a live sensor called every
        frame — but on its own it sees straight through walls. It is safe for
        the live game loop only because `Scene.visible` additionally restricts
        the result to the player's current room, which is what actually keeps
        the agent from seeing into the next room; it is not a substitute for
        `visible_from`'s genuine occlusion test at world-generation time.
        """
        seen = [prim for _, prim in self._cone_candidates(position, forward, sensor)]
        return self._expand_assemblies(seen)

    def _cone_candidates(self, position: Vec3, forward: Vec3,
                         sensor: Sensor) -> list[tuple[float, Superquadric]]:
        """Primitives in range and field of view, nearest first. The shared
        broad phase behind both `visible_from` and `visible_cone_from`."""
        planes = self._frustum_planes(forward, sensor)
        candidates: list[tuple[float, Superquadric]] = []
        for prim in self._primitives.values():
            offset = prim.position - position
            distance = offset.length()
            reach = prim.bounding_radius
            if distance - reach > sensor.range:
                continue
            if any(normal.dot(offset) < -reach for normal in planes):
                continue
            candidates.append((distance, prim))

        candidates.sort(key=lambda item: item[0])
        return candidates

    @staticmethod
    def _frustum_planes(forward: Vec3, sensor: Sensor) -> list[Vec3]:
        """Inward normals of the four side planes, all through the eye."""
        f, right, up = basis(forward)
        half_h = math.radians(max(1.0, min(179.0, sensor.fov))) / 2
        half_v = math.atan(math.tan(half_h) / max(0.1, sensor.aspect))
        return [
            f * math.sin(half_h) + right * math.cos(half_h),
            f * math.sin(half_h) - right * math.cos(half_h),
            f * math.sin(half_v) + up * math.cos(half_v),
            f * math.sin(half_v) - up * math.cos(half_v),
        ]

    def _has_line_of_sight(self, eye: Triple, prim: Superquadric) -> bool:
        mesh, _, _ = self._collision_mesh(prim)
        # Nearest surface point first: it is the sample most likely to be clear,
        # so the usual case costs one ray instead of eight. The nearest *vertex*
        # alone was not enough — against a large slab the player is right up
        # against, it is a tessellation corner metres away that may itself be
        # occluded, every sample then fails, and the sensor reports no wall at
        # all while the agent flies into it.
        #
        # It is kept as well as, not instead of, the surface point: seen through
        # a doorway, a vertex further along a wall can be visible when the
        # nearest point on it is squarely behind the frame. Dropping it lost
        # primitives the old code detected.
        nearest_vertex = min(mesh.vertices, key=lambda v: (v[0] - eye[0]) ** 2
                             + (v[1] - eye[1]) ** 2 + (v[2] - eye[2]) ** 2)
        for sample in [_nearest_surface_point(mesh, eye), nearest_vertex,
                       *prim.axis_extrema()]:
            if self._sample_is_reachable(eye, sample, prim.id):
                return True
        return False

    def _sample_is_reachable(self, eye: Triple, sample: Triple, exclude: int) -> bool:
        delta = (sample[0] - eye[0], sample[1] - eye[1], sample[2] - eye[2])
        for other in self._primitives.values():
            if other.id == exclude or not other.is_solid:
                continue
            mesh, lo, hi = self._collision_mesh(other)
            if not _ray_hits_aabb(eye, delta, lo, hi):
                continue
            vertices = mesh.vertices
            for i, j, k in mesh.triangles:
                if _segment_hits_triangle(eye, delta, vertices[i], vertices[j], vertices[k]):
                    return False
        return True

    # ---------------------------------------------------------- (de)serialisation

    def to_list(self) -> list[dict]:
        return [prim.to_dict() for prim in self._primitives.values()]

    @classmethod
    def from_list(cls, data: list[dict], *, collision_resolution: int = 8) -> "SuperquadricHandler":
        return cls((Superquadric.from_dict(item) for item in data),
                   collision_resolution=collision_resolution)
