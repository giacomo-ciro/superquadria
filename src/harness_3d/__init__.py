"""An agent harness that generates a 3D superquadric world and then flies it.

Sibling of `harness_2d`; the `Agent` abstraction and tmux backends both harnesses
run on live in `harness_common`. Every piece of generated world geometry is a
superquadric, stored as compact parameters rather than as source code or model
files, and the navigation agent observes those parameters directly instead of
pixels.

Layout:
  geometry / superquadrics      vectors, the primitive, meshing, collision, sensor
  scene                         bounds, player, key, persistence, the win check
  generation                    schema, validation, procedural fallback, reachability
  state / memory                the parameter observation and the observed-only map
  moves / policies / navigation target positions, manual flight, the agent policy
  engine                        the observe -> plan -> fly loop, and replay
  render                        the lazily-imported Ursina view
"""

from .engine import Budgets, Episode, EpisodeResult, Replay
from .generation import WorldGenerator
from .geometry import Vec3
from .memory import SpatialMemory
from .moves import Trajectory
from .navigation import WaypointNavigator
from .policies import ManualPolicy, Policy
from .scene import Player, Scene
from .state import State
from .superquadrics import MeshData, Sensor, Superquadric, SuperquadricHandler

__all__ = [
    "Budgets", "Episode", "EpisodeResult", "Replay", "WorldGenerator", "Vec3",
    "SpatialMemory", "Trajectory", "WaypointNavigator", "ManualPolicy", "Policy",
    "Player", "Scene", "State", "MeshData", "Sensor", "Superquadric", "SuperquadricHandler",
]
