"""The one primitive the whole world is made of, and the handler that owns them.

Every piece of generated world geometry is a superquadric (3D_PLAN.md section 2):

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

#: Only the harness may create a key; the generation agent can only make obstacles.
OBSTACLE = "obstacle"
KEY = "key"

#: The key is not part of the generated palette — it has to read as "the goal".
KEY_COLOR = (1.0, 0.84, 0.28)

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
        could contradict it."""
        return self.kind == OBSTACLE

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


def _segment_point_distance_squared(origin: Triple, delta: Triple, p: Triple) -> float:
    length_squared = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2
    if length_squared <= 1e-18:
        return sum((p[i] - origin[i]) ** 2 for i in range(3))
    t = sum((p[i] - origin[i]) * delta[i] for i in range(3)) / length_squared
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return sum((origin[i] + delta[i] * t - p[i]) ** 2 for i in range(3))


def _aabb_overlaps_point(lo: Triple, hi: Triple, p: Triple) -> bool:
    return (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2])


# -------------------------------------------------------------------- handler


@dataclass(frozen=True)
class Sensor:
    """What the parameter sensor can reach: range, horizontal FOV, aspect."""

    range: float = 45.0
    fov: float = 70.0
    aspect: float = 1.6


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

        Approximated exactly as 3D_PLAN.md section 4 states: bounding volumes
        reject the far field, the analytic implicit function catches a centre
        that is inside, and sphere-to-triangle distance against the collision
        mesh decides the rest. Collision therefore agrees with the rendered
        surface up to the collision tessellation.
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

    def visible_from(self, position: Vec3, forward: Vec3, sensor: Sensor) -> list[Superquadric]:
        """Primitives the parameter sensor detects from this pose.

        Range, then camera frustum, then sampled line of sight against the
        collision meshes. Once any part of a primitive is detected the caller
        gets the *whole* validated primitive — that information advantage over a
        pixel policy is the point of the abstraction, not an accident of it.
        """
        eye = position.as_tuple()
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
        return [prim for _, prim in candidates if self._has_line_of_sight(eye, prim)]

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
        nearest = min(mesh.vertices,
                      key=lambda v: (v[0] - eye[0]) ** 2 + (v[1] - eye[1]) ** 2 + (v[2] - eye[2]) ** 2)
        # Nearest surface vertex first: it is the sample most likely to be clear,
        # so the usual case costs one ray instead of seven.
        for sample in [nearest, *prim.axis_extrema()]:
            if self._sample_is_reachable(eye, sample, prim.id):
                return True
        return False

    def _sample_is_reachable(self, eye: Triple, sample: Triple, exclude: int) -> bool:
        delta = (sample[0] - eye[0], sample[1] - eye[1], sample[2] - eye[2])
        for other in self._primitives.values():
            if other.id == exclude or not other.is_solid:
                continue
            reach = other.bounding_radius
            if _segment_point_distance_squared(eye, delta, other.position.as_tuple()) > reach * reach:
                continue
            mesh, _, _ = self._collision_mesh(other)
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
