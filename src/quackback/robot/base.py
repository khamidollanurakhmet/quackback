"""What ask_human needs from a robot, whichever robot it is.

Positions are in the robot's base frame: metres from where it started, x forward, y to its left.
Every failure is a RobotError, so the caller can log it and carry on without the robot.
"""

import math
from dataclasses import dataclass
from typing import Protocol

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


class RobotBackend(Protocol):
    async def wait_ready(self, max_wait: float) -> bool:
        """True once the robot is connected and knows where its base is."""
        ...

    async def go_to(self, x: float, y: float, stop_short: float = 0.0) -> None:
        """Walk to a point of the base frame, or stop `stop_short` metres before it.

        Raises RobotError if it cannot get there.
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
