"""An agent harness that generates a 2D maze environment and navigates it.

The 3D counterpart lives in the sibling `harness_3d` package; what the two share
— the `Agent` abstraction, the config loader, the pipeline log — lives in
`harness_common`.

Layout:
  vector / entities / moves   grid primitives and the action vocabulary
  scene                       the world model + the verifiable win/collision checks
  state                       the 10x10 observation handed to the agent
  generation                  batched, agent-authored maze construction
  memory / policies / navigation   fog-of-war map, offline baseline, Claude policy
  engine                      the observe -> plan -> execute loop
  render                      the Pygame view
"""

from .engine import Episode, EpisodeResult
from .entities import Entity, Tile
from .generation import MazeGenerator
from .memory import FogMemory
from .moves import Direction, Move
from .navigation import ClaudeNavigator
from .policies import FrontierPolicy, Policy
from .scene import VIEW_SIZE, Scene
from .state import State
from .vector import Vector2D

__all__ = [
    "Episode", "EpisodeResult", "Entity", "Tile", "MazeGenerator", "FogMemory",
    "Direction", "Move", "ClaudeNavigator", "FrontierPolicy", "Policy",
    "Scene", "VIEW_SIZE", "State", "Vector2D",
]
