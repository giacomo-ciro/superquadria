"""Continuous 3D geometry, superquadric primitives, meshing, collision, and sensing."""

from .geometry import (
    FORWARD,
    UP,
    ZERO,
    Matrix3,
    Vec3,
    apply,
    apply_inverse,
    basis,
    look_angles,
    look_vector,
    rotation_matrix,
)
from .superquadrics import (
    MeshData,
    Sensor,
    Superquadric,
    SuperquadricHandler,
    build_mesh,
)

__all__ = [
    "FORWARD",
    "UP",
    "ZERO",
    "Matrix3",
    "MeshData",
    "Sensor",
    "Superquadric",
    "SuperquadricHandler",
    "Vec3",
    "apply",
    "apply_inverse",
    "basis",
    "build_mesh",
    "look_angles",
    "look_vector",
    "rotation_matrix",
]
