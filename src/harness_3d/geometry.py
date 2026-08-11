"""Continuous 3D vectors and the one rotation convention the harness uses.

The 2D harness is fully discretised — one cell, one step. Here movement is
continuous: the player is a sphere at a float position flying straight segments,
so everything is float maths (3D_PLAN.md section 3).

Coordinates follow Ursina's: +X right, +Y up, +Z forward. Rotations are XYZ Euler
degrees applied to column vectors as `R = Rz @ Ry @ Rx`. That order is fixed here
and converted to a matrix exactly once per primitive, so meshing, collision and
visibility can never disagree about what a rotation meant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 3x3 row-major rotation matrix.
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        """Unit vector, or the zero vector if there is no direction to take.

        Callers that need a real direction (a look vector, a movement segment)
        check the length first; returning zero keeps the degenerate case from
        raising in the middle of a trajectory.
        """
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length) if length > 1e-12 else ZERO

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).length()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def rounded(self, digits: int = 2) -> tuple[float, float, float]:
        """Compact form for JSON and prompts — full float noise helps nobody.

        `float()` first, so a vector built from ints persists as the same JSON
        as the one reloaded from it.
        """
        return (round(float(self.x), digits), round(float(self.y), digits), round(float(self.z), digits))

    @classmethod
    def parse(cls, values) -> "Vec3":
        """Build from any 3-sequence of finite numbers, or raise.

        Used on the generation and navigation parsing paths, so it is strict:
        anything that is not three finite numbers is rejected outright rather
        than coerced into arbitrary geometry.
        """
        try:
            x, y, z = (float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"not a 3-vector: {values!r}") from exc
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError(f"non-finite 3-vector: {values!r}")
        return cls(x, y, z)

    def __repr__(self) -> str:  # shows up in prompts and logs
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


ZERO = Vec3(0.0, 0.0, 0.0)
UP = Vec3(0.0, 1.0, 0.0)
FORWARD = Vec3(0.0, 0.0, 1.0)


def rotation_matrix(rotation: tuple[float, float, float]) -> Matrix3:
    """XYZ Euler degrees -> `R = Rz @ Ry @ Rx` for column vectors."""
    rx, ry, rz = (math.radians(a) for a in rotation)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def apply(matrix: Matrix3, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """`matrix @ v` — local direction to world direction."""
    x, y, z = v
    return (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z)


def apply_inverse(matrix: Matrix3, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """`matrix.T @ v` — world direction to local direction.

    The transpose is the inverse because `rotation_matrix` is orthonormal by
    construction; no general inversion is needed anywhere in the harness.
    """
    x, y, z = v
    return (matrix[0][0] * x + matrix[1][0] * y + matrix[2][0] * z,
            matrix[0][1] * x + matrix[1][1] * y + matrix[2][1] * z,
            matrix[0][2] * x + matrix[1][2] * y + matrix[2][2] * z)


def look_angles(forward: Vec3) -> tuple[float, float]:
    """A look direction as (pitch, yaw) degrees in Ursina's convention."""
    f = forward.normalized()
    if f == ZERO:
        return (0.0, 0.0)
    yaw = math.degrees(math.atan2(f.x, f.z))
    pitch = math.degrees(-math.asin(max(-1.0, min(1.0, f.y))))
    return (pitch, yaw)


def look_vector(pitch: float, yaw: float) -> Vec3:
    """Inverse of `look_angles`."""
    p, y = math.radians(pitch), math.radians(yaw)
    cp = math.cos(p)
    return Vec3(math.sin(y) * cp, -math.sin(p), math.cos(y) * cp)


def basis(forward: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    """An orthonormal (forward, right, up) frame for a look direction.

    Left-handed, matching the coordinate convention: `right = up x forward`.
    Looking straight up or down has no unique roll, so the world +Z axis stands
    in for the reference vector there.
    """
    f = forward.normalized()
    if f == ZERO:
        f = FORWARD
    reference = UP if abs(f.dot(UP)) < 0.999 else FORWARD
    right = reference.cross(f).normalized()
    return f, right, f.cross(right).normalized()
