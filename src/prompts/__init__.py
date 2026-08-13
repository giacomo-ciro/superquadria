"""Top-level prompt modules.

One module per prompt to keep prompt engineering clean, readable, and decoupled
from game logic, geometry algorithms, and physics.
"""

from .layout import layout_prompt
from .navigation import NAV_SYSTEM_PROMPT, nav_user_prompt
from .room import room_prompt

__all__ = [
    "layout_prompt",
    "room_prompt",
    "NAV_SYSTEM_PROMPT",
    "nav_user_prompt",
]
