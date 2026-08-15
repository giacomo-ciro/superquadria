"""Policy interface and the human policy.

The prototype has exactly two policies: `agent`
(navigation.py) and `manual`. There is no offline baseline navigator — the 2D
harness's frontier policy has no honest 3D analogue that does not read the
reachability graph the agent is forbidden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import math

from geometry import UP, Vec3, look_angles, look_vector
from geometry.superquadrics import DOOR, LOCK, OBJECT
from world.task import DEFAULT_REACH

from .memory import SpatialMemory
from .moves import MOVE, PICK, PLACE, Action, Trajectory
from .state import State


class PolicyQuit(Exception):
    """Raised by an interactive policy when the user quits mid-turn."""


class Policy(ABC):
    """Turns an observation into a batch of target positions."""

    name = "policy"

    #: Interactive policies read the render window, so the engine polls their
    #: `act` between frames instead of calling it once and waiting on it.
    main_thread_only = False

    #: Populated by `observe`; the renderer draws its counts.
    memory: SpatialMemory | None = None

    @abstractmethod
    def act(self, state: State) -> Trajectory | None:
        """The next batch. Only a `main_thread_only` policy may return None,
        meaning "nothing to do yet — ask again next frame"."""

    def cancel(self) -> None:
        """Abandon an in-flight `act`, called from the render thread when the
        user closes the window. Default is a no-op."""

    def observe(self, state: State) -> SpatialMemory:
        """Fold one observation into memory.

        The engine calls this before `act` and again after every movement
        increment, so memory holds every pose actually traversed rather than
        only the endpoints of a batch. Both happen on the main thread, which
        keeps every write on the thread that renders it.
        """
        if self.memory is None:
            self.memory = SpatialMemory()
        self.memory.integrate(state)
        return self.memory


class ManualPolicy(Policy):
    """First-person flight: WASD horizontally, Space/Shift vertically, mouse look, E interact.

    Returns None until a movement/interact key is pressed rather than waiting for one, so the
    renderer keeps stepping Ursina between polls. Mouse look is applied on every
    poll — independently of movement, which is the whole point of it — by writing
    straight into the player's look direction.
    """

    name = "manual"
    main_thread_only = True

    def __init__(self, renderer, scene, *, speed: float = 12.0, sensitivity: float = 40.0):
        self.renderer = renderer
        self.scene = scene
        self.speed = speed
        self.sensitivity = sensitivity
        self.pitch, self.yaw = look_angles(scene.player.forward)

    def act(self, state: State) -> Trajectory | None:
        renderer = self.renderer
        if renderer.quit_requested:
            raise PolicyQuit()
        # Third person is a camera, not a controller: the mouse orbits and the
        # movement keys are ignored until the view comes back.
        if renderer.third_person:
            return None

        dx, dy = renderer.mouse_delta
        self.yaw += dx * self.sensitivity
        # Clamped short of vertical: at exactly +/-90 the look direction has no
        # horizontal component left to steer WASD by.
        self.pitch = max(-89.0, min(89.0, self.pitch - dy * self.sensitivity))
        self.scene.player.forward = look_vector(self.pitch, self.yaw)

        # 'E' key: interact (pick up or place / unlock)
        if "e" in renderer.pending_keys:
            player_pos = self.scene.player.position
            if not self.scene.player.carrying:
                from world.task import is_pickable
                # Find nearest pickable object/furniture within reach
                best_obj, best_d = None, math.inf
                for prim in self.scene.primitives:
                    if is_pickable(prim):
                        d = (prim.position - player_pos).length()
                        if d < best_d:
                            best_obj, best_d = prim, d
                if best_obj is not None and best_d <= DEFAULT_REACH:
                    return Trajectory([Action(type=PICK, target=best_obj.position)], steer=False)
            else:
                # Place on door/lock if near, otherwise place in front of player
                door_or_lock = None
                for prim in self.scene.primitives:
                    if prim.kind in (DOOR, LOCK):
                        if (prim.position - player_pos).length() <= DEFAULT_REACH:
                            door_or_lock = prim
                            break
                if door_or_lock is not None:
                    return Trajectory([Action(type=PLACE, target=door_or_lock.position)], steer=False)
                drop_pos = player_pos + self.scene.player.forward * 1.5
                return Trajectory([Action(type=PLACE, target=drop_pos)], steer=False)

        direction = self._movement_vector()
        if direction == Vec3(0.0, 0.0, 0.0):
            return None
        step = direction.normalized() * (self.speed * max(1e-3, renderer.dt))
        return Trajectory([Action(type=MOVE, target=self.scene.player.position + step)], steer=False)

    def _movement_vector(self) -> Vec3:
        held = self.renderer.held
        # Built from yaw alone, so WASD stays horizontal however far up or down
        # the camera is pointing — and stays exact at the pitch limits, where a
        # frame derived from the look vector degenerates.
        ground = look_vector(0.0, self.yaw)
        side = Vec3(ground.z, 0.0, -ground.x)
        move = Vec3(0.0, 0.0, 0.0)
        if held("w"):
            move = move + ground
        if held("s"):
            move = move - ground
        if held("d"):
            move = move + side
        if held("a"):
            move = move - side
        if held("space"):
            move = move + UP
        if held("shift") or held("left shift"):
            move = move - UP
        return move
