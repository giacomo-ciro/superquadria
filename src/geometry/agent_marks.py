"""The agent's own body: a Claude or Codex brand mark, rendered as a coin.

Geometry was pixel-scanned from the two reference logos and built offline with
numpy/trimesh/manifold3d (a one-off tool, not a runtime dependency of this
package) — Claude's blocky mark as ten exact axis-aligned boxes, Codex's cloud
as six unioned circles with the chevron/dash boolean-cut through as real holes.
The result is checked in as plain JSON under `assets/agent_marks/` and loaded
here into the same `MeshData` shape `superquadrics.build_mesh` produces, so
the renderer treats a mark exactly like any other mesh.

Both are built in Ursina's own convention (+X right, +Y up, +Z forward) at a
fixed unit size — roughly 2 units wide — so a caller sizes one for the scene
with a plain `Entity(scale=...)` rather than baking any world scale in here.
"""

from __future__ import annotations

import json
from pathlib import Path

from .superquadrics import MeshData

_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "agent_marks"


def _load(name: str) -> MeshData:
    data = json.loads((_ASSETS / f"{name}.json").read_text())
    return MeshData(
        vertices=[tuple(v) for v in data["vertices"]],
        triangles=[tuple(t) for t in data["triangles"]],
        normals=[tuple(n) for n in data["normals"]],
        colors=[tuple(c) for c in data["colors"]],
    )


CLAUDE_MARK = _load("claude")
CODEX_MARK = _load("codex")

#: `WaypointNavigator.name` (src/navigation/navigation.py) is the driving
#: agent's `binary` — "claude" or "codex" — so this is the render-time lookup
#: from `info["policy"]` straight to the mark to show.
MARKS_BY_POLICY = {"claude": CLAUDE_MARK, "codex": CODEX_MARK}
