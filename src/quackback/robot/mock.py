"""Stand-in robots for tests: MockBackend records what ask_human asked of the robot, MockLink is
a small kinematic duck for exercising MicroduckBackend's steering without a simulator. It has
eyes: a camera that draws the `person` as a bright bar, which MockDetector finds, and a depth
sensor that sees the person as a wide pole, and any `obstacles` as thin pillars.
"""

import math

import anyio
import numpy as np
from numpy.typing import NDArray

from quackback.robot.base import CameraFrame, Pose, RobotError
from quackback.robot.depth import NO_TARGET, DepthFrame, zone_direction
from quackback.robot.microduck import TOF_AHEAD, Gait, wrap_angle
from quackback.vision.person_detector import Box

DEADMAN_S = 0.5
# The mock camera: a small portrait picture with the simulated duck's field of view.
PICTURE = (160, 90)  # height, width
VFOV = math.radians(72.7)
REST_PITCH = math.radians(-11)
PERSON_RADIUS = 0.25  # m, as the depth sensor sees them
OBSTACLE_RADIUS = 0.05  # m: obstacles are thin pillars
PERSON_HEIGHT = 1.7
SENSOR_HEIGHT = 0.27


class MockBackend:
    def __init__(
        self,
        *,
        ready: bool = True,
        fail: bool = False,
        walk_time: float = 0.0,
        sees: bool = False,
        finds: bool = True,
    ) -> None:
        self.ready = ready
        self.fail = fail
        self.walk_time = walk_time
        self.can_find_person = sees
        self.finds = finds
        self.calls: list[str] = []

    async def wait_ready(self, max_wait: float) -> bool:
        return self.ready

    async def go_to(self, x: float, y: float, stop_short: float = 0.0) -> None:
        self.calls.append(f"go_to {x:g},{y:g}" + (f" short {stop_short:g}" if stop_short else ""))
        await anyio.sleep(self.walk_time)
        if self.fail:
            raise RobotError("a leg fell off")

    async def approach_person(self, search_timeout: float) -> bool:
        self.calls.append(f"find person {search_timeout:g}")
        await anyio.sleep(self.walk_time)
        if self.fail:
            raise RobotError("a leg fell off")
        return self.finds

    async def cue(self) -> None:
        self.calls.append("cue")

    async def show_text(self, text: str | None) -> None:
        self.calls.append(f"text {text}")

    def send_home(self) -> None:
        self.calls.append("home")

    async def emergency_stop(self) -> None:
        self.calls.append("stop")

    def resume(self) -> None:
        self.calls.append("resume")


class MockLink:
    """A unicycle that obeys a Gait: no walking below min_speed, and no turning without walking
    when the gait cannot turn in place. Commands expire like the real deadman. `time_scale`
    runs its world faster than the clock, so tests do not wait for a duck's pace.
    """

    def __init__(
        self,
        gait: Gait | None = None,
        start: Pose | None = None,
        *,
        time_scale: float = 1.0,
        fail_open: bool = False,
        fail_after_moves: int | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        open_delay: float = 0.0,
        stop_delay: float = 0.0,
        person: tuple[float, float] | None = None,
        head_gain: float = 1.0,
        obstacles: tuple[tuple[float, float], ...] = (),
        speed_scale: float = 1.0,
        turn_scale: float = 1.0,
    ) -> None:
        self.gait = gait or Gait(max_speed=0.25, max_turn=1.0)
        self.person = person
        self.obstacles = list(obstacles)  # poles only the depth sensor sees
        # A policy that does less than it is told, as the simulator's does: 0.1 m/s for 0.25,
        # 0.58 rad/s for 1.0.
        self._speed_scale = speed_scale
        self._turn_scale = turn_scale
        self._head_gain = head_gain  # how far the camera turns per radian the head is told to
        self.head: tuple[float, float] | None = None  # (up, left) radians; None is at rest
        self.looks: list[tuple[float, float] | None] = []
        self.bounds = bounds
        self._open_delay = open_delay
        self._stop_delay = stop_delay
        self._pose = start or Pose(0.0, 0.0, 0.0)
        self._time_scale = time_scale
        self._fail_open = fail_open
        self._fail_after_moves = fail_after_moves
        self._command = (0.0, 0.0, 0.0)
        self._command_at = -math.inf
        self._updated_at: float | None = None
        self.moves: list[tuple[float, float, float]] = []
        self.stops = 0
        self.quacks = 0
        self.texts: list[str | None] = []
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        await anyio.sleep(self._open_delay)
        if self._fail_open:
            raise ConnectionError("simulator did not start")
        self.opened = True

    async def move(self, vx: float, vy: float, vyaw: float) -> None:
        self._advance()
        if self._fail_after_moves is not None and len(self.moves) >= self._fail_after_moves:
            raise ConnectionError("browser closed")
        self._command = (vx, vy, vyaw)
        self._command_at = anyio.current_time()
        self.moves.append(self._command)

    async def stop(self) -> None:
        self._advance()
        self._command = (0.0, 0.0, 0.0)
        self.stops += 1
        await anyio.sleep(self._stop_delay)  # a slow round trip to the duck

    @property
    def stopped(self) -> bool:
        return self._command == (0.0, 0.0, 0.0)

    async def pose(self) -> Pose:
        self._advance()
        return self._pose

    async def quack(self) -> None:
        self.quacks += 1

    async def show_text(self, text: str | None) -> None:
        self.texts.append(text)

    async def close(self) -> None:
        self.closed = True

    def teleport(self, pose: Pose) -> None:
        self._pose = pose

    # Eyes.

    async def look(self, up: float | None, left: float = 0.0) -> None:
        self.head = None if up is None else (up, left)
        self.looks.append(self.head)

    def _aim(self) -> tuple[float, float]:
        up, left = self.head or (REST_PITCH, 0.0)
        # Wrapped, like the simulator's headings.
        return wrap_angle(self._pose.yaw + self._head_gain * left), up

    async def camera_frame(self) -> CameraFrame:
        self._advance()
        heading, pitch = self._aim()
        height, width = PICTURE
        image = np.zeros((height, width, 3), np.uint8)
        seen = (
            None if self.person is None else _seen(self.person, PERSON_RADIUS, self._pose, heading)
        )
        if seen is not None:
            bearing, half = seen
            focal = (height / 2) / math.tan(VFOV / 2) * math.cos(pitch)
            if abs(bearing) < math.radians(80):
                left = 0.5 - math.tan(bearing + half) * focal / width
                right = 0.5 - math.tan(bearing - half) * focal / width
                first, last = max(0, round(left * width)), min(width, round(right * width))
                if first < last:
                    image[height // 4 : height * 3 // 4, first:last] = 255
        return CameraFrame(image, VFOV, heading, pitch)

    async def depth_frame(self) -> DepthFrame:
        self._advance()
        heading, pitch = self._aim()
        # In the head, ahead of the neck: it swings round with the head.
        sensor = (
            self._pose.x + TOF_AHEAD * math.cos(heading),
            self._pose.y + TOF_AHEAD * math.sin(heading),
        )
        frame = DepthFrame(8, 8, (0,) * 64, (NO_TARGET,) * 64, pitch, heading)
        mm, status = [0] * 64, [NO_TARGET] * 64
        things = [(thing, OBSTACLE_RADIUS) for thing in self.obstacles]
        if self.person is not None:
            things.append((self.person, PERSON_RADIUS))
        for thing, radius in things:
            bearing, half = _seen(thing, radius, Pose(*sensor, 0.0), heading)
            distance = math.hypot(thing[0] - sensor[0], thing[1] - sensor[1])
            top = math.atan2(PERSON_HEIGHT - SENSOR_HEIGHT, distance)
            bottom = math.atan2(-SENSOR_HEIGHT, distance)
            for i in range(64):
                zone_bearing, elevation = zone_direction(frame, i // 8, i % 8)
                if abs(wrap_angle(zone_bearing - bearing)) > half:
                    continue
                if bottom <= elevation <= top:
                    slant = round(1000 * distance / math.cos(elevation))
                    if status[i] == NO_TARGET or slant < mm[i]:
                        mm[i], status[i] = slant, 5
        return DepthFrame(8, 8, tuple(mm), tuple(status), pitch, heading)

    def _advance(self) -> None:
        now = anyio.current_time()
        if self._updated_at is None:
            self._updated_at = now
            return
        dt = (now - self._updated_at) * self._time_scale
        self._updated_at = now
        vx, _vy, vyaw = self._command  # a duck does not strafe
        if now - self._command_at > DEADMAN_S:
            vx = vyaw = 0.0
        walking = abs(vx) >= max(self.gait.min_speed, 1e-9)
        vx, vyaw = vx * self._speed_scale, vyaw * self._turn_scale
        if not walking:
            vx = 0.0
            if not self.gait.turns_in_place:
                vyaw = 0.0
        p = self._pose
        yaw = p.yaw + vyaw * dt
        self._pose = Pose(p.x + vx * math.cos(yaw) * dt, p.y + vx * math.sin(yaw) * dt, yaw)


def _seen(
    thing: tuple[float, float], radius: float, at: Pose, heading: float
) -> tuple[float, float]:
    """A pole's bearing relative to `heading` and half its width, seen from `at`."""
    dx, dy = thing[0] - at.x, thing[1] - at.y
    distance = max(math.hypot(dx, dy), radius * 1.01)
    return wrap_angle(math.atan2(dy, dx) - heading), math.asin(radius / distance)


class MockDetector:
    """Finds the bright bar MockLink's camera draws for the person."""

    def __init__(self, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.loaded = False
        self.pictures = 0

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("no model")
        self.loaded = True

    def detect(self, image: NDArray[np.uint8]) -> list[Box]:
        self.pictures += 1
        bright = image.max(axis=2) > 200
        columns, rows = np.flatnonzero(bright.any(axis=0)), np.flatnonzero(bright.any(axis=1))
        if columns.size == 0:
            return []
        height, width = bright.shape
        return [
            Box(
                columns[0] / width,
                rows[0] / height,
                (columns[-1] + 1) / width,
                (rows[-1] + 1) / height,
                0.9,
            )
        ]
