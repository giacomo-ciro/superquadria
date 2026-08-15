"""Agent-driven building generation: a floor plan, then one furnished room.

Two agent calls: one designs the whole building's floor plan (`LAYOUT_SCHEMA`),
the harness turns that into walls and doors with no agent involved, then a
second call furnishes the spawn room (`ROOM_SCHEMA`) in its own room-local
coordinate frame.

Nothing the model returns becomes runtime geometry directly. A layout is
validated and *repaired* — `layout.validate_and_repair` drops what is broken
and stitches the rest into one connected building — rather than thrown away
wholesale, because it is abstract enough to make that safe. Room content is
validated per primitive and, if the room's local flood cannot prove the door
still reachable, discarded wholesale: a bare sealed shell is still a valid
room, an unreachable door is not.

The harness creates and locks each door along the path with an embossed
replica of that room's key assembly: a plausible-looking generation is
otherwise perfectly capable of being unwinnable.
"""

from __future__ import annotations

import math
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from agents.base import Agent, AgentError
from agents.tmux_agent import TmuxAgent
from engine.pipeline_log import PipelineLogger
from geometry import FORWARD, ZERO, Matrix3, Vec3, apply, rotation_matrix
from geometry.superquadrics import OBSTACLE, Sensor, Superquadric, SuperquadricHandler
from prompts.layout import layout_prompt
from prompts.room import room_prompt

from . import layout as layout_module
from . import shell as shell_module
from . import task as task_module
from .layout import IN_PLANE, Door, Layout, Room, WorldConfig
from .scene import Player, Scene

LAYOUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "description": "Short name for this building, e.g. 'The Copper Works'."},
        "description": {"type": "string", "description": "2-3 sentences on what this building is."},
        "levels": {"type": "integer", "description": "How many storeys the building actually uses."},
        "rooms": {
            "type": "array",
            "description": "Every room's footprint on an integer grid, [0,0] at one corner of the floor plate.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique machine-facing slug, e.g. 'copper-galley'."},
                    "name": {"type": "string", "description": "Distinctive human-facing name, e.g. 'The Copper Galley'."},
                    "style": {"type": "string",
                             "description": "1-2 sentences: what the room is AND its forms/materials. The entire brief for furnishing it."},
                    "key_concept": {"type": "string",
                                   "description": "A recognizable object archetype (e.g. 'guitar', 'tree', 'chair', 'lamp', 'sword', 'telescope', 'chalice', 'clock', 'pencil') that unlocks this room's door."},
                    "palette": {
                        "type": "object",
                        "description": "RGB colors [r, g, b] (floats in range 0.0-1.0) for this room's surfaces.",
                        "properties": {
                            "wall": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "[r, g, b] wall color (0.0 to 1.0)"},
                            "floor": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "[r, g, b] floor color (0.0 to 1.0)"},
                            "ceiling": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "[r, g, b] ceiling color (0.0 to 1.0)"},
                        },
                        "required": ["wall", "floor", "ceiling"],
                        "additionalProperties": False,
                    },
                    "level": {"type": "integer"},
                    "origin": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "required": ["id", "name", "style", "key_concept", "palette", "level", "origin", "size"],
                "additionalProperties": False,
            },
        },
        "connections": {
            "type": "array",
            "description": "Room-id pairs that should be able to open into one another.",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                "required": ["from", "to"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["theme", "description", "levels", "rooms", "connections"],
    "additionalProperties": False,
}

#: Two-stage room content: `objects` are modelled once each in their own local
#: frame, `placements` instantiate them in the room. One object may be placed any
#: number of times — that repetition is what buys richness without re-modelling.
ROOM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "notes": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                  "description": "Brief layout reasoning: what the room is for and where things stand."},
        "objects": {
            "type": "array",
            "description": "Object recipes, each modelled in its own local frame. Placed by the `placements` array.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique slug for this object, e.g. 'ladderback_chair'. Exactly one object's id must contain the room's key_concept.",
                    },
                    "primitives": {
                        "type": "array",
                        "description": "The object's parts, in object-local coordinates: origin at the centre of its footprint, resting on Y=0, facing +Z.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "offset": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "[dx, dy, dz] centre of this part, in metres from the object origin.",
                                },
                                "rotation": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "[rx, ry, rz] degrees about this part's own centre, applied X then Y then Z (e.g. [90, 0, 0] stands a cylinder upright).",
                                },
                                "scale": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "[sx, sy, sz] half-extents in metres (a cylinder along local +Z is [r, r, half_length]).",
                                },
                                "exponents": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "[e1, e2] superquadric exponents: e2 shapes the XY cross-section, e1 the Z profile.",
                                },
                                "color": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "RGB color [r, g, b] (floats in range 0.0-1.0).",
                                },
                            },
                            "required": ["offset", "rotation", "scale", "exponents", "color"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "primitives"],
                "additionalProperties": False,
            },
        },
        "placements": {
            "type": "array",
            "description": "One entry per instance standing in the room. Repeat an object id to place it again.",
            "items": {
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "The `id` of the object being placed.",
                    },
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[px, py, pz] where the object's local origin lands, in room-local coordinates (py=0.0 rests it on the floor).",
                    },
                    "rotation": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[rx, ry, rz] degrees applied to the whole object about its origin. Use [0, yaw, 0] for anything standing on the floor.",
                    },
                    "scale": {
                        "type": "number",
                        "description": "Uniform size multiplier for this instance (1.0 = as modelled).",
                    },
                },
                "required": ["object", "position", "rotation", "scale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["notes", "objects", "placements"],
    "additionalProperties": False,
}


def _door_world_centre(door: Door, cfg: WorldConfig) -> Vec3:
    u_axis, v_axis = IN_PLANE[door.axis]
    pos = [0.0, 0.0, 0.0]
    pos[door.axis] = door.coord
    pos[u_axis] = (door.hole[0] + door.hole[2]) / 2
    pos[v_axis] = (door.hole[1] + door.hole[3]) / 2
    return Vec3(*pos)


def _matrix_multiply(a: Matrix3, b: Matrix3) -> Matrix3:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
         a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
         a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
         a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
         a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]),
        (a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
         a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
         a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]),
    )


def _matrix_to_euler_xyz(m: Matrix3) -> tuple[float, float, float]:
    """Extract XYZ Euler angles in degrees from a 3x3 rotation matrix R = Rz @ Ry @ Rx."""
    sy = -m[2][0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cos_ry = math.cos(ry)
    if abs(cos_ry) > 1e-6:
        rx = math.atan2(m[2][1], m[2][2])
        rz = math.atan2(m[1][0], m[0][0])
    else:
        rx = math.atan2(-m[1][2], m[1][1])
        rz = 0.0
    return (round(math.degrees(rx) % 360.0, 2),
            round(math.degrees(ry) % 360.0, 2),
            round(math.degrees(rz) % 360.0, 2))


def _parse_assembly_primitive(item: dict, assembly_name: str, asm_pos: Vec3, r_asm: Matrix3,
                             origin: Vec3, index: int, interior,
                             default_color: tuple[float, float, float] = (0.6, 0.55, 0.5),
                             scale_mult: float = 1.0) -> Superquadric:
    (x0, y0, z0), (x1, y1, z1) = interior
    max_scale = max(x1 - x0, y1 - y0, z1 - z0)
    offset_raw = item.get("offset", item.get("position", [0.0, 0.0, 0.0]))
    offset_vec = Vec3.parse(offset_raw) * scale_mult
    local_rot = tuple(float(a) % 360.0 for a in Vec3.parse(item.get("rotation", [0.0, 0.0, 0.0])).as_tuple())
    scale = tuple(a * scale_mult for a in Vec3.parse(item["scale"]).as_tuple())
    exponents = tuple(float(e) for e in item["exponents"])
    raw_color = item.get("color")
    if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
        try:
            color = (max(0.0, min(1.0, float(raw_color[0]))),
                     max(0.0, min(1.0, float(raw_color[1]))),
                     max(0.0, min(1.0, float(raw_color[2]))))
        except (ValueError, TypeError):
            color = default_color
    else:
        color = default_color

    if any(not math.isfinite(a) or a <= 0 or a > max_scale for a in scale):
        raise ValueError(f"scale {scale} must be positive and within room bound {max_scale:.1f}")
    if any(not math.isfinite(e) or e <= 0 for e in exponents):
        raise ValueError(f"exponents {exponents} must be positive")

    rot_offset = apply(r_asm, offset_vec.as_tuple())
    world_pos = Vec3(origin.x + asm_pos.x + rot_offset[0],
                     origin.y + asm_pos.y + rot_offset[1],
                     origin.z + asm_pos.z + rot_offset[2])

    r_prim = rotation_matrix(local_rot)
    r_world = _matrix_multiply(r_asm, r_prim)
    world_rot = _matrix_to_euler_xyz(r_world)

    return Superquadric(id=index, kind=OBSTACLE,
                        assembly=str(assembly_name)[:32].strip() or "furniture",
                        position=world_pos, rotation=world_rot, scale=scale,
                        exponents=exponents, color=color)


def _aabb_inside(lo, hi, box) -> bool:
    (x0, y0, z0), (x1, y1, z1) = box
    return x0 <= lo[0] and hi[0] <= x1 and y0 <= lo[1] and hi[1] <= y1 and z0 <= lo[2] and hi[2] <= z1


def _aabb_overlaps(lo1, hi1, lo2, hi2) -> bool:
    return all(lo1[a] < hi2[a] and lo2[a] < hi1[a] for a in range(3))


def _instantiate(payload: dict, origin: Vec3, interior, room_name: str,
                 log: PipelineLogger) -> list[Superquadric]:
    """Turn one `objects` + `placements` payload into world-space primitives.

    Objects are modelled once each in their own local frame; a placement stands
    one of them in the room at `position`, rotated by `rotation` and resized by
    the scalar `scale`. Uniform scale commutes with rotation, so it folds
    exactly into each part's offset and half-extents — which is why the
    placement scale is a scalar and not a vector: a per-axis scale applied
    outside an arbitrary part rotation is a shear, and a superquadric has no
    shear term to carry it.

    Each placement gets its own `object_id#N` assembly label. Sharing one label
    across instances would make `task.pick_at` gather every copy in the room
    into a single carried object.
    """
    objects = {str(obj["id"]): obj.get("primitives") or []
               for obj in payload.get("objects") or [] if obj.get("id")}
    instantiated: list[Superquadric] = []
    counts: dict[str, int] = {}

    for idx, placement in enumerate(payload.get("placements") or []):
        object_id = str(placement.get("object", ""))
        primitives = objects.get(object_id)
        if not primitives:
            log.info("room", f"{room_name}: placement #{idx} names unknown object '{object_id}'")
            continue
        try:
            asm_pos = Vec3.parse(placement.get("position", [0.0, 0.0, 0.0]))
            asm_rot = tuple(float(a) % 360.0 for a in
                            Vec3.parse(placement.get("rotation", [0.0, 0.0, 0.0])).as_tuple())
            mult = float(placement.get("scale", 1.0))
        except (ValueError, TypeError, IndexError) as exc:
            log.info("room", f"{room_name}: rejected placement #{idx} '{object_id}' ({exc})")
            continue
        if not math.isfinite(mult) or mult <= 0:
            log.info("room", f"{room_name}: rejected placement #{idx} '{object_id}' (scale {mult})")
            continue

        label = f"{object_id}#{counts.get(object_id, 0)}"
        counts[object_id] = counts.get(object_id, 0) + 1
        r_asm = rotation_matrix(asm_rot)
        for part_idx, item in enumerate(primitives):
            try:
                instantiated.append(_parse_assembly_primitive(
                    item, label, asm_pos, r_asm, origin, len(instantiated), interior,
                    scale_mult=mult))
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                log.info("room", f"{room_name}: rejected part #{part_idx} of '{label}' ({exc})")
    return instantiated


def _procedural_object(concept: str) -> dict:
    """One offline object recipe, in the same local frame the agent models in."""
    prims = []
    c = concept.lower()
    if "guitar" in c:
        prims.append({"offset": [0.0, 0.35, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.25, 0.35, 0.08], "exponents": [0.3, 0.8], "color": [0.72, 0.42, 0.18]})
        prims.append({"offset": [0.0, 0.9, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.04, 0.25, 0.04], "exponents": [0.1, 0.1], "color": [0.35, 0.22, 0.12]})
        prims.append({"offset": [0.0, 1.2, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.07, 0.07, 0.04], "exponents": [0.2, 0.2], "color": [0.25, 0.15, 0.08]})
    elif "tree" in c:
        prims.append({"offset": [0.0, 0.4, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.08, 0.08, 0.4], "exponents": [0.1, 1.0], "color": [0.45, 0.30, 0.18]})
        prims.append({"offset": [0.0, 1.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.4, 0.4, 0.4], "exponents": [1.0, 1.0], "color": [0.22, 0.55, 0.24]})
    elif "chair" in c:
        prims.append({"offset": [0.0, 0.45, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.25, 0.04, 0.25], "exponents": [0.2, 0.2], "color": [0.58, 0.38, 0.24]})
        prims.append({"offset": [0.0, 0.75, -0.2], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.25, 0.26, 0.04], "exponents": [0.2, 0.2], "color": [0.55, 0.35, 0.22]})
        for dx in (-0.18, 0.18):
            for dz in (-0.18, 0.18):
                prims.append({"offset": [dx, 0.2, dz], "rotation": [90.0, 0.0, 0.0],
                              "scale": [0.03, 0.03, 0.2], "exponents": [0.1, 1.0], "color": [0.30, 0.28, 0.26]})
    elif "lamp" in c:
        prims.append({"offset": [0.0, 0.02, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.18, 0.18, 0.02], "exponents": [0.1, 1.0], "color": [0.30, 0.30, 0.32]})
        prims.append({"offset": [0.0, 0.6, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.03, 0.03, 0.55], "exponents": [0.1, 1.0], "color": [0.65, 0.60, 0.40]})
        prims.append({"offset": [0.0, 1.25, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.2, 0.2, 0.12], "exponents": [0.4, 1.0], "color": [0.92, 0.88, 0.70]})
    elif "sword" in c:
        prims.append({"offset": [0.0, 0.6, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.04, 0.45, 0.02], "exponents": [0.1, 0.1], "color": [0.78, 0.80, 0.84]})
        prims.append({"offset": [0.0, 0.2, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.16, 0.03, 0.04], "exponents": [0.1, 0.1], "color": [0.75, 0.60, 0.20]})
        prims.append({"offset": [0.0, 0.08, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.03, 0.03, 0.08], "exponents": [0.1, 1.0], "color": [0.30, 0.18, 0.12]})
    elif "telescope" in c:
        prims.append({"offset": [0.0, 0.95, 0.0], "rotation": [30.0, 0.0, 0.0],
                      "scale": [0.08, 0.08, 0.45], "exponents": [0.1, 1.0], "color": [0.75, 0.62, 0.25]})
        prims.append({"offset": [0.0, 0.45, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.04, 0.04, 0.45], "exponents": [0.1, 1.0], "color": [0.25, 0.25, 0.28]})
        prims.append({"offset": [0.0, 0.02, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.2, 0.2, 0.02], "exponents": [0.1, 1.0], "color": [0.30, 0.30, 0.32]})
    elif "chalice" in c:
        prims.append({"offset": [0.0, 0.02, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.15, 0.15, 0.02], "exponents": [0.1, 1.0], "color": [0.85, 0.72, 0.25]})
        prims.append({"offset": [0.0, 0.2, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.03, 0.03, 0.15], "exponents": [0.1, 1.0], "color": [0.80, 0.68, 0.22]})
        prims.append({"offset": [0.0, 0.45, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.18, 0.18, 0.15], "exponents": [0.5, 1.0], "color": [0.90, 0.78, 0.30]})
    elif "clock" in c:
        prims.append({"offset": [0.0, 0.5, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.25, 0.35, 0.12], "exponents": [0.1, 0.1], "color": [0.50, 0.32, 0.18]})
        prims.append({"offset": [0.0, 0.55, 0.13], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.16, 0.16, 0.02], "exponents": [0.1, 1.0], "color": [0.92, 0.90, 0.85]})
    elif "anvil" in c:
        prims.append({"offset": [0.0, 0.15, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.22, 0.15, 0.18], "exponents": [0.1, 0.1], "color": [0.28, 0.28, 0.30]})
        prims.append({"offset": [0.0, 0.38, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.38, 0.08, 0.16], "exponents": [0.2, 0.2], "color": [0.38, 0.38, 0.42]})
    elif "pencil" in c:
        prims.append({"offset": [0.0, 0.45, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.04, 0.04, 0.38], "exponents": [0.1, 1.0], "color": [0.90, 0.75, 0.15]})
        prims.append({"offset": [0.0, 0.88, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.04, 0.08, 0.04], "exponents": [0.8, 0.8], "color": [0.25, 0.22, 0.20]})
    else:
        prims.append({"offset": [0.0, 0.04, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.18, 0.04, 0.18], "exponents": [0.1, 0.1], "color": [0.45, 0.45, 0.48]})
        prims.append({"offset": [0.0, 0.35, 0.0], "rotation": [90.0, 0.0, 0.0],
                      "scale": [0.04, 0.04, 0.3], "exponents": [0.1, 1.0], "color": [0.60, 0.58, 0.52]})
        prims.append({"offset": [0.0, 0.75, 0.0], "rotation": [0.0, 0.0, 0.0],
                      "scale": [0.14, 0.14, 0.14], "exponents": [1.0, 1.0], "color": [0.75, 0.50, 0.30]})
    return {"id": concept, "primitives": prims}


#: Offline stand-ins, in the same shape the agent returns.
_PROCEDURAL_FURNITURE: dict[str, list[dict]] = {
    "dining_table": [
        {"offset": [0.0, 0.73, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.6, 0.03, 0.4], "exponents": [0.1, 0.1], "color": [0.60, 0.42, 0.28]},
        {"offset": [-0.5, 0.35, -0.3], "rotation": [90.0, 0.0, 0.0], "scale": [0.03, 0.03, 0.35], "exponents": [0.1, 1.0], "color": [0.32, 0.25, 0.20]},
        {"offset": [0.5, 0.35, -0.3], "rotation": [90.0, 0.0, 0.0], "scale": [0.03, 0.03, 0.35], "exponents": [0.1, 1.0], "color": [0.32, 0.25, 0.20]},
        {"offset": [-0.5, 0.35, 0.3], "rotation": [90.0, 0.0, 0.0], "scale": [0.03, 0.03, 0.35], "exponents": [0.1, 1.0], "color": [0.32, 0.25, 0.20]},
        {"offset": [0.5, 0.35, 0.3], "rotation": [90.0, 0.0, 0.0], "scale": [0.03, 0.03, 0.35], "exponents": [0.1, 1.0], "color": [0.32, 0.25, 0.20]},
    ],
    "floor_lamp": [
        {"offset": [0.0, 0.02, 0.0], "rotation": [90.0, 0.0, 0.0], "scale": [0.18, 0.18, 0.02], "exponents": [0.1, 1.0], "color": [0.30, 0.30, 0.32]},
        {"offset": [0.0, 0.6, 0.0], "rotation": [90.0, 0.0, 0.0], "scale": [0.03, 0.03, 0.55], "exponents": [0.1, 1.0], "color": [0.65, 0.60, 0.40]},
        {"offset": [0.0, 1.25, 0.0], "rotation": [90.0, 0.0, 0.0], "scale": [0.2, 0.2, 0.12], "exponents": [0.4, 1.0], "color": [0.92, 0.88, 0.70]},
    ],
}


def _procedural_room(rng: random.Random, interior, key_concept: str = "key") -> dict:
    """An offline `objects` + `placements` payload, parsed by the same code path."""
    (x0, y0, z0), (x1, y1, z1) = interior
    half_x, half_z = (x1 - x0) / 2, (z1 - z0) / 2

    kpx = rng.uniform(-half_x * 0.6, half_x * 0.6)
    kpz = rng.uniform(-half_z * 0.6, half_z * 0.6)
    objects = [_procedural_object(key_concept)]
    placements = [{"object": key_concept, "position": [kpx, 0.0, kpz],
                   "rotation": [0.0, 0.0, 0.0], "scale": 1.0}]

    for name, prims in _PROCEDURAL_FURNITURE.items():
        if name.lower() in key_concept.lower() or key_concept.lower() in name.lower():
            continue
        fpx = rng.uniform(-half_x * 0.6, half_x * 0.6)
        fpz = rng.uniform(-half_z * 0.6, half_z * 0.6)
        if math.hypot(fpx - kpx, fpz - kpz) <= 1.5:
            continue
        objects.append({"id": name, "primitives": prims})
        placements.append({"object": name, "position": [fpx, 0.0, fpz],
                           "rotation": [0.0, 0.0, 0.0], "scale": 1.0})

    return {"notes": [], "objects": objects, "placements": placements}


def _room_reachable(furniture: list[Superquadric], room: Room, doors: list[Door], cfg: WorldConfig, *,
                    collision_resolution: int, move_increment: float) -> bool:
    """Local flood: with the furniture in place, is every one of the room's
    doorways still reachable at the player radius? Scoped to one room — its
    walls bound the volume by construction, so only the furniture inside needs
    to be flooded around. A room with an entrance and an exit must keep both
    reachable, not just one, or the path through it breaks."""
    if not furniture:
        return True
    handler = SuperquadricHandler(furniture, collision_resolution=collision_resolution)
    (x0, y0, z0), (x1, y1, z1) = room.interior(cfg)
    radius = cfg.player_radius
    spacing = max(1.0, cfg.grid_unit / 2)
    nx, ny, nz = (max(2, int((x1 - x0) / spacing)), max(2, int((y1 - y0) / spacing)),
                 max(2, int((z1 - z0) / spacing)))
    nodes: dict[tuple[int, int, int], Vec3] = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                p = Vec3(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny, z0 + (z1 - z0) * k / nz)
                if not handler.is_blocked(p, radius):
                    nodes[(i, j, k)] = p
    if not nodes:
        return False

    start = min(nodes, key=lambda c: nodes[c].distance_to(room.centre(cfg)))
    neighbourhood = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)]
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        here = nodes[cell]
        for dx, dy, dz in neighbourhood:
            nxt = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            if nxt in seen or nxt not in nodes:
                continue
            if not handler.segment_is_clear(here, nodes[nxt], radius, move_increment):
                continue
            seen.add(nxt)
            queue.append(nxt)

    for door in doors:
        clearance_lo, clearance_hi = shell_module.door_clearance(door, cfg)
        door_centre = Vec3(*((a + b) / 2 for a, b in zip(clearance_lo, clearance_hi)))
        nearest = min(nodes, key=lambda c: nodes[c].distance_to(door_centre))
        if nearest not in seen:
            return False
    return True


def generate_room(agent: Agent | None, room: Room, doors: list[Door], cfg: WorldConfig, rng: random.Random,
                  *, brief: str | None, collision_resolution: int, move_increment: float,
                  log: PipelineLogger) -> tuple[list[Superquadric], str]:
    """Furnish one room. A pure function of its inputs: same room, doors and
    agent response in, same validated primitives (or an empty list) out — which
    is what lets calling it once now, N times in a pool later, or in the
    background later still, be a driver change rather than a redesign.
    """
    interior = room.interior(cfg)
    (x0, y0, z0), (x1, y1, z1) = interior
    origin = Vec3((x0 + x1) / 2, y0, (z0 + z1) / 2)

    if agent is not None:
        try:
            payload = agent.run(room_prompt(room, doors, cfg, interior, origin, brief=brief),
                                ROOM_SCHEMA)
            source = "agent"
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            log.info("room", f"{room.name}: room generation fell back to procedural: {exc}")
            payload = _procedural_room(rng, interior, key_concept=room.key_concept)
            source = f"fallback:{type(exc).__name__}"
    else:
        payload = _procedural_room(rng, interior, key_concept=room.key_concept)
        source = "procedural"

    clearances = [shell_module.door_clearance(door, cfg) for door in doors]
    validated: list[Superquadric] = []

    for prim in _instantiate(payload, origin, interior, room.name, log):
        lo, hi = prim.aabb()
        if not _aabb_inside(lo, hi, interior):
            log.info("room", f"{room.name}: rejected part of '{prim.assembly}' (outside room interior bounds)")
            continue
        if any(_aabb_overlaps(lo, hi, *clearance) for clearance in clearances):
            log.info("room", f"{room.name}: rejected part of '{prim.assembly}' (overlaps door clearance)")
            continue
        validated.append(prim)

    if not _room_reachable(validated, room, doors, cfg, collision_resolution=collision_resolution,
                           move_increment=move_increment):
        log.info("room", f"{room.name}: local flood cannot reach every doorway with {len(validated)} primitives — shipped as a bare shell")
        return [], source
    return validated, source


# ============================================================== the pipeline


@dataclass
class WorldGenerator:
    """Builds a validated `Scene`: a floor plan, its shell, and furnished rooms."""

    #: Only `generate_offline` may be called with agent=None.
    agent: Agent | None = None
    room_agent_factory: Callable[[Room, int], Agent] | Callable[[Room], Agent] | None = None
    bounds: float = 60.0
    grid_unit: float = 4.0
    level_height: float = 5.0
    wall_thickness: float = 0.4
    door_width: float = 3.0
    door_height: float = 4.0
    player_radius: float = 0.6
    move_increment: float = 0.5
    collision_resolution: int = 8
    max_levels: int = 1
    max_rooms: int = 1
    tol_scale: float = 0.05
    tol_exp: float = 0.05
    sensor: Sensor = field(default_factory=Sensor)
    log: PipelineLogger = field(default_factory=lambda: PipelineLogger("gen3d"))
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.move_increment >= 2 * self.player_radius:
            raise ValueError(f"move_increment {self.move_increment} must stay under twice the "
                             f"player radius ({2 * self.player_radius}), or the player can pass "
                             f"through an obstacle between two collision samples")
        self.cfg = WorldConfig(bounds=self.bounds, grid_unit=self.grid_unit,
                               level_height=self.level_height, wall_thickness=self.wall_thickness,
                               door_width=self.door_width, door_height=self.door_height,
                               player_radius=self.player_radius, max_levels=self.max_levels,
                               max_rooms=self.max_rooms)

    # --------------------------------------------------------------- top level

    def generate(self, brief: str | None = None) -> Scene:
        if isinstance(self.agent, TmuxAgent):
            self.agent.clean()
        raw, source = self._request_layout(brief)
        try:
            built = layout_module.validate_and_repair(raw, self.cfg)
            scene = self._assemble(built, source=source, use_agent=(source == "agent"), brief=brief)
            if scene is not None:
                return scene
        except ValueError as exc:
            self.log.info("generate", f"layout repair failed: {exc}")
        self.log.info("generate", "falling back to the procedural layout")
        return self.generate_offline()

    def generate_offline(self) -> Scene:
        """Procedural building, zero agent calls — for smoke tests and demos."""
        for _ in range(8):
            raw = layout_module.procedural_layout(self._rng, self.cfg)
            try:
                built = layout_module.validate_and_repair(raw, self.cfg)
            except ValueError:
                continue
            scene = self._assemble(built, source="procedural", use_agent=False, brief=None)
            if scene is not None:
                return scene
        raise RuntimeError("procedural generation produced no valid world in 8 attempts")

    # ------------------------------------------------------------------ agent

    def _request_layout(self, brief: str | None) -> tuple[dict, str]:
        if self.agent is None:
            return layout_module.procedural_layout(self._rng, self.cfg), "procedural"
        self.log.start("layout", f"requesting a floor plan for a {self.bounds:.0f}-unit building")
        try:
            raw = self.agent.run(layout_prompt(self.cfg, brief), LAYOUT_SCHEMA)
            self.log.end("layout", f"{raw.get('theme', '?')}: {len(raw.get('rooms') or [])} "
                                   f"room(s) proposed")
            return raw, "agent"
        except (AgentError, ValueError, KeyError, TypeError) as exc:
            self.log.info("layout", f"layout generation fell back to procedural: {exc}")
            return layout_module.procedural_layout(self._rng, self.cfg), f"fallback:{type(exc).__name__}"

    def _get_room_agent(self, room: Room, idx: int = 1) -> Agent | None:
        if self.room_agent_factory is not None:
            try:
                return self.room_agent_factory(room, idx)
            except TypeError:
                return self.room_agent_factory(room)
        if isinstance(self.agent, TmuxAgent):
            return type(self.agent)(
                session_name=self.agent.session_name,
                window_name=f"room{idx}",
                binary=self.agent.binary,
                model=self.agent.model,
                timeout=self.agent.timeout,
                max_retries=self.agent.max_retries,
                retry_backoff=self.agent.retry_backoff,
                cwd=self.agent.cwd,
                effort=self.agent.effort,
            )
        return self.agent

    # ------------------------------------------------------------- assembly

    def _assemble(self, built: Layout, *, source: str, use_agent: bool,
                  brief: str | None) -> Scene | None:
        shell_prims, _ = shell_module.build_shell(built, self.cfg)
        handler = SuperquadricHandler(shell_prims, collision_resolution=self.collision_resolution)

        # Every room on the path has 1 or 2 doors touching it: an entrance (all
        # but the first), an outgoing hop (all but the last) or, on the last
        # room, the exit — `d.b is None` there, so it is never anyone's entrance.
        doors_by_room: dict[int, list[Door]] = {r.index: [] for r in built.rooms}
        for d in built.doors:
            doors_by_room[d.a].append(d)
            if d.b is not None:
                doors_by_room[d.b].append(d)

        room0 = built.spawn_room
        exit_room = built.rooms[built.path[-1]]

        spawn = room0.centre(self.cfg)
        away = spawn - _door_world_centre(doors_by_room[room0.index][0], self.cfg)
        spawn_forward = Vec3(away.x, 0.0, away.z).normalized()
        if spawn_forward == ZERO:
            spawn_forward = FORWARD

        # Every room on the path is furnished — the player flies through each
        # one on the way to the exit, so none of them can be left a bare shell.
        target_rooms = [(built.rooms[idx], doors_by_room[idx]) for idx in built.path]

        def _furnish_task(room_agent: Agent | None, room: Room, doors: list[Door],
                          rng: random.Random) -> tuple[Room, list[Superquadric], str]:
            prims, r_source = generate_room(
                room_agent, room, doors, self.cfg, rng, brief=brief,
                collision_resolution=self.collision_resolution, move_increment=self.move_increment,
                log=self.log)
            return room, prims, r_source

        room_agents = [self._get_room_agent(r, i + 1) if use_agent else None for i, (r, _) in enumerate(target_rooms)]
        room_results: list[tuple[Room, list[Superquadric], str]] = []
        try:
            with ThreadPoolExecutor(max_workers=len(target_rooms)) as executor:
                futures = [
                    executor.submit(_furnish_task, r_agent, r, d, random.Random(self._rng.randint(0, 10**9)))
                    for r_agent, (r, d) in zip(room_agents, target_rooms)
                ]
                for f in futures:
                    room_results.append(f.result())
        except BaseException:
            for r_agent in room_agents:
                if r_agent is not None:
                    r_agent.cancel()
            raise

        furniture_by_room = {r.index: f for r, f, _ in room_results}
        room0_source = next((s for r, _, s in room_results if r.index == room0.index), "procedural")

        # One task per room on the path: each room locks its own outgoing door
        # with an embossed replica of that room's key object assembly.
        tasks: dict[str, dict] = {}
        global_assembly_id = 0
        for room_index, door in zip(built.path, built.doors):
            room = built.rooms[room_index]
            room_furniture = furniture_by_room.get(room_index, [])

            # Group primitives by original assembly label
            assemblies: dict[str, list[Superquadric]] = {}
            for prim in room_furniture:
                assemblies.setdefault(prim.assembly, []).append(prim)

            # If no assemblies were generated at all, generate procedural key assembly
            if not assemblies:
                interior = room.interior(self.cfg)
                (x0, y0, z0), (x1, y1, z1) = interior
                origin = Vec3((x0 + x1) / 2, y0, (z0 + z1) / 2)
                key_payload = {
                    "objects": [_procedural_object(room.key_concept)],
                    "placements": [{"object": room.key_concept, "position": [0.0, 0.0, 0.0],
                                    "rotation": [0.0, 0.0, 0.0], "scale": 1.0}],
                }
                for p in _instantiate(key_payload, origin, interior, room.name, self.log):
                    room_furniture.append(p)
                    assemblies.setdefault(p.assembly, []).append(p)

            # Identify the key assembly
            key_orig_name = None
            for name in assemblies:
                if room.key_concept.lower() in name.lower() or name.lower() in room.key_concept.lower():
                    key_orig_name = name
                    break
            if key_orig_name is None:
                key_orig_name = next(iter(assemblies))

            # Anonymize assembly labels uniquely: assembly-0, assembly-1, ...
            orig_names = list(assemblies.keys())
            self._rng.shuffle(orig_names)
            anon_map = {}
            for name in orig_names:
                anon_map[name] = f"assembly-{global_assembly_id}"
                global_assembly_id += 1

            for prim in room_furniture:
                prim.assembly = anon_map.get(prim.assembly, f"assembly-{global_assembly_id}")

            key_anon_name = anon_map[key_orig_name]
            key_prims = assemblies[key_orig_name]

            # Add primitives to handler with next available IDs
            for prim in room_furniture:
                prim.id = handler.next_id()
                handler.add(prim)

            # Build the embossed lock on the door
            lock_prims = task_module.build_lock(door, room, key_prims, handler, self.cfg)
            tasks[str(room_index)] = {
                "key_concept": room.key_concept,
                "key_assembly": key_anon_name,
                "door_id": door.id,
                "lock_ids": [p.id for p in lock_prims],
            }

        # Add furniture for any rooms not on the path (if any)
        path_set = set(built.path)
        for room, furniture, _ in room_results:
            if room.index not in path_set:
                orig_names = list({p.assembly for p in furniture})
                self._rng.shuffle(orig_names)
                anon_map = {}
                for name in orig_names:
                    anon_map[name] = f"assembly-{global_assembly_id}"
                    global_assembly_id += 1
                for prim in furniture:
                    prim.assembly = anon_map.get(prim.assembly, f"assembly-{global_assembly_id}")
                    prim.id = handler.next_id()
                    handler.add(prim)

        meta = {
            "theme": built.theme, "description": built.description, "source": source,
            "room_source": room0_source,
            "match_tolerance": {"scale": self.tol_scale, "exponents": self.tol_exp},
            "layout": self._layout_meta(built),
            "task": tasks,
        }
        scene = Scene(bounds=self.bounds, primitives=handler,
                      player=Player(position=spawn, forward=spawn_forward, radius=self.player_radius),
                      meta=meta)
        self.log.end("generate", f"{built.theme}: {len(built.rooms)} room(s), {len(handler)} "
                                 f"primitives, spawn in {room0.name}, exit in {exit_room.name}")
        return scene

    def _layout_meta(self, built: Layout) -> dict:
        return {
            "theme": built.theme, "description": built.description, "path": built.path,
            "rooms": [{"id": r.id, "name": r.name, "style": r.style, "level": r.level,
                      "index": r.index, "box": [list(lo), list(hi)],
                      "palette": {"wall": list(r.wall_color), "floor": list(r.floor_color), "ceiling": list(r.ceiling_color)}}
                     for r in built.rooms for lo, hi in (r.box(self.cfg),)],
            "doors": [{"a": d.a, "b": d.b, "id": d.id} for d in built.doors],
        }
