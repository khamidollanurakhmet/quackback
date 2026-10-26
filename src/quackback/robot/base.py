"""What ask_human needs from a robot, whichever robot it is.

Positions are in the robot's base frame: metres from where it started, x forward, y to its left.
Every failure is a RobotError, so the caller can log it and carry on without the robot.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# The robot never stops closer than this to the human (CLAUDE.md: stop distance >= 40 cm).
HUMAN_STOP_DISTANCE = 0.40  # m


class RobotError(RuntimeError):
    """The robot could not do what was asked. Never fatal: ask_human goes on without it."""


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float  # radians, counter-clockwise, 0 = facing +x

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)


@dataclass(frozen=True)
class CameraFrame:
    """A picture from the duck's head camera, and where the camera pointed when it took it."""

    image: NDArray[np.uint8]  # height x width x 3, RGB
    vfov: float  # vertical field of view, radians
    heading: float  # where the optical axis points in the duck's world, radians, counter-clockwise
    pitch: float  # the optical axis's angle above level, radians
    # How far the picture is turned about the optical axis, radians, positive when its left side
    # is up: a head that turns while it looks up tilts the picture.
    roll: float = 0.0

    @property
    def hfov(self) -> float:
        """Horizontal field of view, radians."""
        height, width = self.image.shape[:2]
        return 2 * math.atan(math.tan(self.vfov / 2) * width / height)

    def bearing(self, u: float, v: float = 0.5) -> float:
        """The heading in the duck's world of the picture's point (u, v): fractions of its width
        from the left and of its height from the top."""
        height, width = self.image.shape[:2]
        focal = (height / 2) / math.tan(self.vfov / 2)
        right, up = (u - 0.5) * width, (0.5 - v) * height
        # The point's ray in the camera's own frame, rolled level: ahead, left, up.
        c, s = math.cos(self.roll), math.sin(self.roll)
        left, up = -right * c - up * s, -right * s + up * c
        # Pitched, its level part: how far ahead and to the left of the optical axis it goes.
        ahead = focal * math.cos(self.pitch) - up * math.sin(self.pitch)
        return self.heading + math.atan2(left, ahead)


class RobotBackend(Protocol):
    async def wait_ready(self, max_wait: float) -> bool:
        """True once the robot is connected and knows where its base is."""
        ...

    async def go_to(self, x: float, y: float, stop_short: float = 0.0) -> None:
        """Walk to a point of the base frame, or stop `stop_short` metres before it.

        Raises RobotError if it cannot get there.
        """
        ...

    @property
    def can_find_person(self) -> bool:
        """True if the robot can look for the human itself (approach_person)."""
        ...

    async def approach_person(self, search_timeout: float) -> bool:
        """Look around for a person, walk up to them and face them, stopping when the depth
        sensor has them HUMAN_STOP_DISTANCE away plus a margin.

        False if nobody came into view within `search_timeout` seconds. Raises RobotError if it
        cannot get there.
        """
        ...

    async def cue(self) -> None:
        """Get the human's attention before the question."""
        ...

    async def show_text(self, text: str | None) -> None:
        """Put the question where the human watches the robot, if there is such a place."""
        ...

    def send_home(self) -> None:
        """Start walking back to base, without waiting. A newer go_to takes over."""
        ...

    async def emergency_stop(self) -> None:
        """Stop now, and refuse to move until resume(). Never raises."""
        ...

    def resume(self) -> None:
        """Allow motion again after an emergency stop (each new question calls this)."""
        ...
