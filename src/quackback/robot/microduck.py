"""MicroduckBackend: walking a Microduck to a point and back, steered from the PC.

The duck itself only takes a velocity twist (robotd's `robot.move {vx, vy, vyaw}`), stops when
twists stop arriving (its 500 ms deadman), and reports where it is (odometry). Going somewhere
is a closed loop on the PC: read the pose, steer toward the target, resend at 10 Hz. A DuckLink
carries those few primitives to one kind of duck: the simulator website today (sandbox.py), the
JSON-RPC socket of a real one later.

A duck with eyes (a head camera and the depth sensor beside it, DuckEyes) can also find the
human itself: look around with its head until the person detector sees someone, walk toward
them, stop well before HUMAN_STOP_DISTANCE, and face them. Every walk of a duck with eyes
watches the depth sensor and stops for anything close in its way.

Only the newest motion request ever moves the duck: each go_to / approach_person / send_home /
emergency_stop bumps a generation, a running motion is cancelled on the spot, and a queued one
checks its generation the moment it gets its turn. An emergency stop holds until resume(), which
the next question calls.
"""

import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Protocol, Self, TypeVar, runtime_checkable

import anyio
from anyio.abc import TaskGroup

from quackback.config import Settings
from quackback.robot.base import HUMAN_STOP_DISTANCE, CameraFrame, Pose, RobotError
from quackback.robot.depth import DepthFrame, nearest, points
from quackback.vision.person_detector import Box, Detector

log = logging.getLogger(__name__)

CONTROL_HZ = 10.0  # well inside the duck's 500 ms deadman
ARRIVE_RADIUS = 0.15  # m
# Stopping short of a point (the human) also allows for the policy's lag and an arc's overshoot.
STOP_MARGIN = 0.10  # m
TURN_GAIN = 2.2  # rad/s of turn per rad of heading error, before the cap
SMOOTHING = 0.5  # share of the way to the new turn rate per tick
SLOW_DOWN_WITHIN = 0.3  # m: a duck that can stand still eases off near the target
# Give up on a point after this long plus the time it takes at a crawl...
NAV_TIMEOUT_BASE_S = 20.0
NAV_CRAWL_SPEED = 0.05  # m/s, slower than any duck that is really walking
# ...or as soon as it has not got 5 cm closer for a while: a wall, a prop, going in circles.
STALL_WINDOW_S = 12.0
STALL_PROGRESS = 0.05  # m
WALL_MARGIN = 0.2  # m kept from a world's walls

# A walk with eyes stops when the depth sensor has anything this close in the duck's way. It is
# the sensor's own reading, so the 40 cm hold even at its tip; the margin covers a reading up to
# a tick old and the 5-8 cm the walk coasts on after a stop (measured in the simulator).
WALK_STOP = HUMAN_STOP_DISTANCE + 0.10
IN_THE_WAY = math.radians(12)  # half the width of the duck's path, seen from the sensor
TOF_AHEAD = 0.08  # m: the depth sensor sits this far ahead of the body's centre
# A duck that turns only in arcs looks about with the depth sensor before it turns round (the
# head level, turned to each of these, degrees left of the body), and takes only an arc that
# keeps its body this far from everything seen: its nose 40 cm away, and a little for wobble.
SCAN_PITCH = 0.0
SCAN_HEAD = (50.0, 100.0, -50.0, -100.0, 0.0)
SCAN_S = 0.4
ARC_CLEAR = HUMAN_STOP_DISTANCE + TOF_AHEAD + 0.02

# Finding the human. The camera looks this far up while searching and walking: a standing
# person's upper body is then in view from about 0.6 m out to 4 m.
SEARCH_PITCH = math.radians(30)
FACE_PITCH = math.radians(45)  # looking up at them, once there
HEAD_SETTLE_S = 1.0
# Looking around: the head turns the camera this much further each step, a little less than
# its 45 degree view so the views overlap, out to this far each way, looking a moment each time.
SWEEP_STEP = math.radians(35)
SWEEP_REACH = math.radians(110)
LOOK_S = 0.8
# How far the camera turns per unit of head command, until the pictures show it: the most the
# simulator's does (with the head up), so the first step is never too big.
FIRST_HEAD_GAIN = 2.0
# ...then the body turns this far, and the head looks around again.
SEARCH_TURN_ANGLE = math.radians(110)
SEARCH_TURN = 0.5  # share of the turn cap, turning on the spot
LOST_AFTER_S = 2.0  # not seen for this long where the camera should see them: look again
# Walking up to the person ends a little before WALK_STOP would stop the duck anyway, by the
# depth sensor or, while they are out of its view, by where they were seen. A duck that turns
# only in arcs stops further out, by its turning radius, so that it has room to turn round and
# leave (see ARC_CLEAR).
PERSON_STOP = WALK_STOP + 0.05
UNKNOWN_RANGE = 1.5  # m: where a person is taken to be when the depth sensor has no reading
PERSON_HALF_WIDTH = 0.25  # m, taken until a whole person is in a picture
EDGE = 0.01  # a box this close to the side of a picture is cut off by it
FACE_WITHIN = math.radians(10)
FACE_TIMEOUT_S = 10.0
SETTLE_S = 3.0  # at most, for the body to come to rest after walking or turning
AIM_TRIES = 3

T = TypeVar("T")


@dataclass(frozen=True)
class Gait:
    """How a duck's walking policy responds to commands."""

    max_speed: float  # m/s
    max_turn: float  # rad/s
    # Below this forward command the policy stands still instead of walking.
    min_speed: float = 0.0
    # False: the policy only turns while walking forward, so it steers in arcs...
    turns_in_place: bool = True
    # ...and this is the tightest circle it walks, m (as measured: the commanded speed over the
    # commanded turn rate may say otherwise).
    turn_radius: float = 0.0


class DuckLink(Protocol):
    gait: Gait
    # The walls of the duck's world in the link's frame, (x_min, x_max, y_min, y_max), if any.
    bounds: tuple[float, float, float, float] | None

    async def open(self) -> None: ...
    async def move(self, vx: float, vy: float, vyaw: float) -> None: ...
    async def stop(self) -> None: ...
    async def pose(self) -> Pose: ...
    async def quack(self) -> None: ...
    async def show_text(self, text: str | None) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class DuckEyes(Protocol):
    """What a duck with a head camera and a depth sensor adds to its DuckLink."""

    async def camera_frame(self) -> CameraFrame: ...
    async def depth_frame(self) -> DepthFrame: ...

    async def look(self, up: float | None, left: float = 0.0) -> None:
        """Point the camera `up` radians above level and `left` radians left of the body, as
        far as the neck goes; None puts the head back at rest."""
        ...


@dataclass(frozen=True)
class Sighting:
    """Where the person is taken to be, in the link's frame."""

    x: float
    y: float
    half_width: float  # radians, as seen from where they were seen
    at: float  # anyio clock, when the camera last saw them
    radius: float = PERSON_HALF_WIDTH  # m, half their width


@dataclass(frozen=True)
class _Look:
    """One look: a picture, the most certain person in it, and the depth frame and pose taken
    together with it."""

    frame: CameraFrame
    box: Box | None
    depth: DepthFrame
    pose: Pose


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MicroduckBackend:
    """A RobotBackend. Use as `async with MicroduckBackend(link, settings) as robot:`."""

    def __init__(
        self, link: DuckLink, settings: Settings, detector: Detector | None = None
    ) -> None:
        self._link = link
        self._eyes = link if isinstance(link, DuckEyes) else None
        self._detector = detector if self._eyes is not None else None
        self._sees = False
        gait = link.gait
        self._max_speed = min(settings.max_speed or gait.max_speed, gait.max_speed)
        self._max_turn = min(settings.max_turn or gait.max_turn, gait.max_turn)
        self._gait = gait
        # The circle an arc-walking duck really walks: as measured, and as much larger as the
        # caps make the commanded one.
        commanded = self._max_speed / self._max_turn
        measured = gait.turn_radius / (gait.max_speed / gait.max_turn) if gait.turn_radius else 1.0
        self._radius = 0.0 if gait.turns_in_place else commanded * measured
        self._person_stop = PERSON_STOP + self._radius
        self._head: tuple[float | None, float] = (None, 0.0)  # where it was last told to look
        self._head_gains: dict[float, float] = {}  # learnt, for each pitch it looks up at
        self._base: Pose | None = None
        self._ready = anyio.Event()
        self.error: str | None = None
        self._tg: TaskGroup | None = None
        self._motion_lock = anyio.Lock()
        self._motion: anyio.CancelScope | None = None
        self._generation = 0
        self._halted = False

    async def __aenter__(self) -> Self:
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        self._tg.start_soon(self._connect)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        assert self._tg is not None
        self._tg.cancel_scope.cancel()
        with anyio.CancelScope(shield=True):
            await self._stop_quietly()
            try:
                await self._link.close()
            except Exception:
                log.debug("could not close the robot link", exc_info=True)
        return await self._tg.__aexit__(exc_type, exc, tb)

    async def _connect(self) -> None:
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._load_detector)
                await self._link.open()
                self._base = await self._link.pose()
            log.info("robot ready, base at (%.2f, %.2f)", self._base.x, self._base.y)
        except Exception as exc:
            exc = _first(exc)
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("robot unavailable, questions go without it: %s", self.error)
        finally:
            self._ready.set()

    async def _load_detector(self) -> None:
        if self._detector is None:
            return
        try:
            await anyio.to_thread.run_sync(self._detector.load)
            self._sees = True
        except Exception as exc:
            log.warning("no person detection, the robot walks to robot_target: %s", exc)

    @property
    def can_find_person(self) -> bool:
        return self._sees

    async def wait_ready(self, max_wait: float) -> bool:
        with anyio.move_on_after(max_wait):
            await self._ready.wait()
        return self._ready.is_set() and self._base is not None

    # The base frame (where the duck started) to the link's own frame, and back.

    def _to_link(self, x: float, y: float) -> tuple[float, float]:
        base = self._require_base()
        c, s = math.cos(base.yaw), math.sin(base.yaw)
        return base.x + c * x - s * y, base.y + s * x + c * y

    def to_base(self, pose: Pose) -> Pose:
        base = self._require_base()
        dx, dy = pose.x - base.x, pose.y - base.y
        c, s = math.cos(base.yaw), math.sin(base.yaw)
        return Pose(c * dx + s * dy, -s * dx + c * dy, wrap_angle(pose.yaw - base.yaw))

    def _require_base(self) -> Pose:
        if self._base is None:
            raise RobotError(f"the robot is not connected ({self.error or 'still starting'})")
        return self._base

    # Motions.

    def resume(self) -> None:
        """Lift an emergency stop, so the next motion may move the duck again."""
        self._halted = False

    async def go_to(self, x: float, y: float, stop_short: float = 0.0) -> None:
        """Walk to (x, y) of the base frame, or to `stop_short` metres from it.

        Whatever the duck was doing, or was about to do, is cancelled first.
        """
        generation = self._supersede()
        await self._run_motion(partial(self._drive_to, x, y, stop_short), generation)

    async def approach_person(self, search_timeout: float) -> bool:
        """Find the person and walk up to them. False if nobody came into view in time."""
        generation = self._supersede()
        return await self._run_motion(partial(self._find_person, search_timeout), generation)

    def send_home(self) -> None:
        if self._tg is None or self._base is None or self._halted:
            return
        generation = self._supersede()
        self._tg.start_soon(self._go_home, generation)

    async def emergency_stop(self) -> None:
        self._halted = True
        self._supersede()
        await self._stop_quietly()
        log.warning("robot emergency stop")

    def _supersede(self) -> int:
        """A new motion request: whatever moves the duck now, or is waiting to, is outdated."""
        self._generation += 1
        if self._motion is not None:
            self._motion.cancel()
        return self._generation

    async def _go_home(self, generation: int) -> None:
        try:
            await self._run_motion(partial(self._drive_to, 0.0, 0.0, 0.0), generation)
        except RobotError as exc:
            log.info("robot did not make it home: %s", exc)

    async def _run_motion(self, drive: Callable[[], Awaitable[T]], generation: int) -> T:
        async with self._motion_lock:
            # No await between this check and publishing the scope: from here on, a newer
            # request or an emergency stop cancels this motion through the scope.
            if self._halted:
                raise RobotError("held by an emergency stop")
            if generation != self._generation:
                raise RobotError("superseded by a newer motion")
            with anyio.CancelScope() as scope:
                self._motion = scope
                try:
                    result = await drive()
                finally:
                    self._motion = None
                    with anyio.CancelScope(shield=True):
                        await self._stop_quietly()
            if scope.cancelled_caught:
                raise RobotError(
                    "emergency stop" if self._halted else "interrupted by a newer motion"
                )
            return result

    def _check_speed_cap(self) -> None:
        if self._max_speed < self._gait.min_speed:
            raise RobotError(
                f"the speed cap ({self._max_speed} m/s) is below the slowest this duck "
                f"walks ({self._gait.min_speed} m/s)"
            )

    async def _drive_to(self, x: float, y: float, stop_short: float) -> None:
        self._check_speed_cap()
        target_x, target_y = self._to_link(x, y)
        self._check_inside_walls(x, y, target_x, target_y)
        arrive = max(ARRIVE_RADIUS, stop_short + STOP_MARGIN) if stop_short > 0 else ARRIVE_RADIUS
        try:
            if self._eyes is not None:
                await self._point_head(self._eyes, None)
            pose = await self._link.pose()
            distance = pose.distance_to(target_x, target_y)
            turn = 0
            behind = abs(_error(pose, target_x, target_y)) > math.pi / 2
            if self._radius and self._eyes is not None and behind and distance > arrive:
                turn = await self._turn_round_side(self._eyes, target_x, target_y)
            now = anyio.current_time()
            deadline = now + NAV_TIMEOUT_BASE_S + distance / NAV_CRAWL_SPEED
            best, best_at = distance, now
            vyaw = 0.0
            while distance > arrive:
                now = anyio.current_time()
                if now > deadline:
                    raise RobotError(f"did not reach ({x:.2f}, {y:.2f}): {distance:.2f} m short")
                if turn and abs(_error(pose, target_x, target_y)) <= math.pi / 2:
                    # Turned round: from here on it steers as usual, and progress counts.
                    turn = 0
                    best, best_at = distance, now
                if distance < best - STALL_PROGRESS:
                    best, best_at = distance, now
                elif now - best_at > STALL_WINDOW_S:
                    raise RobotError(f"stuck {distance:.2f} m from ({x:.2f}, {y:.2f})")
                vx, want_vyaw = self._steer(pose, target_x, target_y, distance, turn)
                if self._eyes is not None and vx > 0:  # turning on the spot hits nothing
                    ahead = self._ahead(await self._eyes.depth_frame(), pose)
                    if stop_short > 0 and ahead is not None and ahead <= self._person_stop:
                        # Walking up to the human, and there they are.
                        await self._link.stop()
                        log.info("stopped %.2f m short (depth sensor)", ahead)
                        return
                    if ahead is not None and ahead <= WALK_STOP:
                        await self._link.stop()
                        raise RobotError(f"stopped for something {ahead:.2f} m ahead")
                # A turn round follows the arc it was cleared for. Otherwise only the turn is
                # smoothed: easing the speed toward a gait's minimum would leave it just below,
                # where the policy stands still.
                vyaw = want_vyaw if turn else vyaw + SMOOTHING * (want_vyaw - vyaw)
                await self._link.move(vx, 0.0, vyaw)
                await anyio.sleep(1 / CONTROL_HZ)
                pose = await self._link.pose()
                distance = pose.distance_to(target_x, target_y)
        except RobotError:
            raise
        except Exception as exc:
            raise RobotError(f"lost the robot: {type(exc).__name__}: {exc}") from exc
        log.info("robot reached (%.2f, %.2f)", x, y)

    def _ahead(self, depth: DepthFrame, pose: Pose) -> float | None:
        """The depth sensor's nearest reading along the duck's way: straight ahead, or anywhere
        in its view for a duck whose way curves."""
        return nearest(depth) if self._radius else _in_the_way(depth, pose)

    async def _turn_round_side(self, eyes: DuckEyes, x: float, y: float) -> int:
        """Which way an arc-walking duck turns round to head for (x, y), +1 left or -1 right: the
        shorter way, unless the depth sensor, looking about, sees something too close to that
        arc. Raises RobotError if both ways are."""
        seen = await self._scan(eyes)
        await self._point_head(eyes, None)
        pose = await self._link.pose()
        shorter = 1 if _error(pose, x, y) >= 0 else -1
        for side in (shorter, -shorter):
            if _arc_clearance(pose, side, self._radius, math.pi, seen) >= ARC_CLEAR:
                return side
        raise RobotError("no room to turn round here")

    async def _scan(self, eyes: DuckEyes) -> list[tuple[float, float]]:
        """Where the depth sensor sees things around the front of the duck, the head level."""
        seen: list[tuple[float, float]] = []
        for degrees in SCAN_HEAD:
            await self._point_head(eyes, SCAN_PITCH, math.radians(degrees), settle=False)
            await anyio.sleep(SCAN_S)
            seen += _points(await eyes.depth_frame(), await self._link.pose())
        return seen

    def _check_inside_walls(self, x: float, y: float, link_x: float, link_y: float) -> None:
        if not self._inside_walls(link_x, link_y):
            raise RobotError(f"({x:.2f}, {y:.2f}) is outside the walls of the duck's world")

    def _inside_walls(self, link_x: float, link_y: float) -> bool:
        bounds = self._link.bounds
        if bounds is None:
            return True
        x_min, x_max, y_min, y_max = bounds
        inside_x = x_min + WALL_MARGIN <= link_x <= x_max - WALL_MARGIN
        inside_y = y_min + WALL_MARGIN <= link_y <= y_max - WALL_MARGIN
        return inside_x and inside_y

    # Finding the human.

    async def _find_person(self, search_timeout: float) -> bool:
        eyes, detector = self._eyes, self._detector
        if eyes is None or detector is None or not self._sees:
            raise RobotError("this robot cannot look for people")
        self._check_speed_cap()
        try:
            person = await self._walk_up_to_person(eyes, detector, search_timeout)
            if person is None:
                await self._point_head(eyes, None, settle=False)
                return False
            await self._face(eyes, detector, person)
        except RobotError:
            raise
        except Exception as exc:
            raise RobotError(f"lost the robot: {type(exc).__name__}: {exc}") from exc
        return True

    async def _walk_up_to_person(
        self, eyes: DuckEyes, detector: Detector, search_timeout: float
    ) -> Sighting | None:
        """Look around for the person, then walk toward them until they are close.

        Where they are is estimated again from every picture they are in: the bearing from the
        camera, the range from the depth sensor's zones in that direction (both sit in the
        head). While they are out of view, the duck steers by the last estimate and its own
        odometry, and that estimate also decides when it is close enough.
        """
        person: Sighting | None = None
        best, best_at = math.inf, anyio.current_time()
        while True:
            if person is None:
                person = await self._look_around(eyes, detector, search_timeout)
                if person is None:
                    return None
                log.info("found a person, walking over")
                # Ahead again, before walking: the depth sensor watches the way.
                await self._point_head(eyes, SEARCH_PITCH)
                best, best_at = math.inf, anyio.current_time()
            tick = anyio.current_time() + 1 / CONTROL_HZ
            look = await self._look(eyes, detector)
            now, pose = anyio.current_time(), look.pose
            if not self._inside_walls(pose.x, pose.y):
                raise RobotError("too close to the walls of the duck's world")
            if look.box is not None:
                person = self._place(look, person, now)
            elif now - person.at > LOST_AFTER_S and (
                self._in_view(look, person) or now - person.at > 3 * LOST_AFTER_S
            ):
                log.info("lost sight of the person, looking around again")
                await self._link.stop()
                person = None
                continue
            sensor_x, sensor_y = _sensor(pose)
            toward = math.atan2(person.y - sensor_y, person.x - sensor_x) - look.depth.heading
            toward = wrap_angle(toward)
            around = nearest(look.depth, toward + person.half_width, toward - person.half_width)
            to_person = math.hypot(person.x - sensor_x, person.y - sensor_y)
            if around is not None:
                to_person = min(to_person, around)
            if to_person <= self._person_stop:
                await self._link.stop()
                log.info("stopped %.2f m from the person", to_person)
                return person
            anything = nearest(look.depth)
            if anything is not None and anything <= WALK_STOP:
                await self._link.stop()
                log.info("stopped for something %.2f m away, short of the person", anything)
                return person
            distance = pose.distance_to(person.x, person.y)
            if distance < best - STALL_PROGRESS:
                best, best_at = distance, now
            elif now - best_at > STALL_WINDOW_S:
                raise RobotError(f"stuck {distance:.2f} m from the person")
            vx, vyaw = self._steer(pose, person.x, person.y, distance)
            await self._link.move(vx, 0.0, vyaw)
            await anyio.sleep(max(0.0, tick - anyio.current_time()))

    async def _look_around(
        self, eyes: DuckEyes, detector: Detector, search_timeout: float
    ) -> Sighting | None:
        """Stand and look around with the head; between looks, turn the body: on the spot if it
        can, else in an arc to a side where the depth sensor just saw room for one."""
        give_up = anyio.current_time() + search_timeout
        while anyio.current_time() < give_up:
            seen: list[tuple[float, float]] = []  # what the depth sensor saw, all around
            # Ahead, then step by step to each side: each step turns the camera SWEEP_STEP past
            # where the last picture shows it pointed, however far the head turns it.
            for side in (0, 1, -1):
                reached, command = 0.0, 0.0
                while reached < SWEEP_REACH and anyio.current_time() < give_up:
                    if side:
                        gain = self._head_gains.get(SEARCH_PITCH, FIRST_HEAD_GAIN)
                        command = side * (reached + SWEEP_STEP) / gain
                    # No waiting for the head: each picture says where the camera pointed.
                    await self._point_head(eyes, SEARCH_PITCH, command, settle=False)
                    look = await self._look_a_while(eyes, detector, seen, give_up)
                    if look.box is not None:
                        look = await self._center_on(eyes, detector, look)
                        return self._place(look, None, anyio.current_time())
                    turned = side * wrap_angle(look.frame.heading - look.pose.yaw)
                    if not side or turned < reached + math.radians(5):
                        break  # straight ahead looked at, or the neck turns no further
                    reached = turned
                    self._learn_head_gain(SEARCH_PITCH, look, command)
            await self._point_head(eyes, SEARCH_PITCH)  # ahead before turning: watch the way
            if anyio.current_time() >= give_up:
                break
            person = await self._turn_to_look(eyes, detector, seen, give_up)
            if person is not None:
                return person
        log.info("nobody came into view in %.0f s", search_timeout)
        return None

    async def _look_a_while(
        self,
        eyes: DuckEyes,
        detector: Detector,
        seen: list[tuple[float, float]],
        give_up: float,
    ) -> _Look:
        """Pictures for LOOK_S, while the head gets where it was told: the first with a person in
        it, or the last. What the depth sensor saw goes into `seen`."""
        until = min(anyio.current_time() + LOOK_S, give_up)
        while True:
            look = await self._look(eyes, detector)
            seen += _points(look.depth, look.pose)
            if look.box is not None or anyio.current_time() >= until:
                return look
            await anyio.sleep(1 / CONTROL_HZ)

    def _learn_head_gain(self, up: float, look: _Look, command: float) -> float:
        gain = _head_gain(look, command)
        if gain != 1.0 or abs(command) > math.radians(3):
            self._head_gains[up] = gain
        return gain

    async def _turn_to_look(
        self,
        eyes: DuckEyes,
        detector: Detector,
        seen: list[tuple[float, float]],
        give_up: float,
    ) -> Sighting | None:
        """Turn the body SEARCH_TURN_ANGLE, watching the camera on the way."""
        if self._gait.turns_in_place:
            twist = (0.0, 0.0, SEARCH_TURN * self._max_turn)
        else:
            pose = await self._link.pose()
            sides = [
                s
                for s in (1, -1)
                if _arc_clearance(pose, s, self._radius, SEARCH_TURN_ANGLE, seen) >= ARC_CLEAR
            ]
            if not sides:
                log.info("no room to turn around here, looking with the head only")
                await anyio.sleep(LOOK_S)
                return None
            twist = (self._max_speed, 0.0, sides[0] * self._max_turn)
        start = (await self._link.pose()).yaw
        stop_at = min(give_up, anyio.current_time() + 3 * SEARCH_TURN_ANGLE / abs(twist[2]))
        try:
            while anyio.current_time() < stop_at:
                tick = anyio.current_time() + 1 / CONTROL_HZ
                look = await self._look(eyes, detector)
                if look.box is not None:
                    await self._link.stop()
                    look = await self._center_on(eyes, detector, look)
                    return self._place(look, None, anyio.current_time())
                if abs(wrap_angle(look.pose.yaw - start)) >= SEARCH_TURN_ANGLE:
                    break
                if not self._inside_walls(look.pose.x, look.pose.y):
                    raise RobotError("too close to the walls of the duck's world")
                anything = nearest(look.depth)
                if twist[0] > 0 and anything is not None and anything <= WALK_STOP:
                    log.info("stopped turning for something %.2f m away", anything)
                    break
                await self._link.move(*twist)
                await anyio.sleep(max(0.0, tick - anyio.current_time()))
        finally:
            with anyio.CancelScope(shield=True):
                await self._stop_quietly()
        return None

    async def _center_on(self, eyes: DuckEyes, detector: Detector, look: _Look) -> _Look:
        """Turn the head to put a person just seen in the middle of the picture: a person cut
        off at its edge puts the middle of their box off their middle, and the depth sensor
        (beside the camera) ranges best what is in the middle of its view. A look from there,
        or the first one if they are gone from it."""
        for _ in range(2):
            off = wrap_angle(_box_bearing(look) - look.frame.heading)
            if abs(off) < math.radians(5):
                break
            command = self._head[1]
            gain = self._learn_head_gain(SEARCH_PITCH, look, command)
            await self._point_head(eyes, SEARCH_PITCH, command + off / gain)
            again = await self._look(eyes, detector)
            if again.box is None:
                break
            look = again
        return look

    async def _look(self, eyes: DuckEyes, detector: Detector) -> _Look:
        frame = await eyes.camera_frame()
        depth = await eyes.depth_frame()
        pose = await self._link.pose()
        boxes = await anyio.to_thread.run_sync(detector.detect, frame.image)
        return _Look(frame, boxes[0] if boxes else None, depth, pose)

    def _place(self, look: _Look, before: Sighting | None, now: float) -> Sighting:
        frame, box, depth, pose = look.frame, look.box, look.depth, look.pose
        assert box is not None
        # Ranged first, by the depth zones over the part of them the picture shows...
        middle = (box.top + box.bottom) / 2
        left, right = frame.bearing(box.left, middle), frame.bearing(box.right, middle)
        shown = max(0.0, wrap_angle(left - right) / 2)
        toward = wrap_angle(_box_bearing(look) - depth.heading)
        reading = nearest(depth, toward + shown, toward - shown)
        if reading is not None:
            distance = reading + TOF_AHEAD
        else:  # the depth sensor lost them: keep the last range
            distance = UNKNOWN_RANGE if before is None else pose.distance_to(before.x, before.y)
        # ...then, if the picture cuts them off, how wide they are at that range tells where
        # their middle is.
        radius = PERSON_HALF_WIDTH if before is None else before.radius
        if _whole(box):
            half_width = shown
            radius = distance * math.sin(half_width)
        else:
            half_width = _half_width(radius, distance)
        bearing = _seen_bearing(look, half_width)
        if bearing is None:  # they fill the picture: keep the last direction
            bearing = frame.heading if before is None else _direction(pose, before)
        return Sighting(
            pose.x + distance * math.cos(bearing),
            pose.y + distance * math.sin(bearing),
            half_width,
            now,
            radius,
        )

    def _in_view(self, look: _Look, person: Sighting) -> bool:
        """Whether the camera should be seeing the person, from where they are taken to be."""
        pose = look.pose
        bearing = math.atan2(person.y - pose.y, person.x - pose.x)
        return abs(wrap_angle(bearing - look.frame.heading)) < 0.4 * look.frame.hfov

    async def _face(self, eyes: DuckEyes, detector: Detector, person: Sighting) -> None:
        """Turn to the person: the body if it can turn on the spot, then the head."""
        pose = await self._settle()
        if self._gait.turns_in_place:
            error = _bearing(pose, person)
            give_up = anyio.current_time() + FACE_TIMEOUT_S
            while abs(error) > FACE_WITHIN and anyio.current_time() < give_up:
                vyaw = max(-self._max_turn, min(self._max_turn, TURN_GAIN * error))
                await self._link.move(0.0, 0.0, vyaw)
                await anyio.sleep(1 / CONTROL_HZ)
                pose = await self._link.pose()
                error = _bearing(pose, person)
            await self._link.stop()
            await self._settle()
        await self._aim_head(eyes, detector, person)

    async def _aim_head(self, eyes: DuckEyes, detector: Detector, person: Sighting) -> None:
        """Point the camera at the person: where the picture shows them, or else where they
        were taken to be. And check where it really points: how far the head turns it sideways
        depends on how far up it looks (twice as far at 45 degrees up as level, in the
        simulator)."""
        left = _bearing(await self._link.pose(), person) / self._head_gains.get(FACE_PITCH, 1.0)
        for _ in range(AIM_TRIES):
            await self._point_head(eyes, FACE_PITCH, left)
            look = await self._look(eyes, detector)
            pose = look.pose
            got = wrap_angle(look.frame.heading - pose.yaw)
            want = _bearing(pose, person)
            if look.box is not None:
                half = _half_width(person.radius, pose.distance_to(person.x, person.y))
                seen = _seen_bearing(look, half)
                if seen is not None:
                    want = wrap_angle(seen - pose.yaw)
            if abs(want - got) < FACE_WITHIN / 2:
                break
            left = want / self._learn_head_gain(FACE_PITCH, look, left)
        log.info("facing the person, camera %.0f degrees to the side", math.degrees(got))

    async def _point_head(
        self, eyes: DuckEyes, up: float | None, left: float = 0.0, settle: bool = True
    ) -> None:
        """Point the head (see DuckEyes.look) and, with `settle`, give it time to get there."""
        if self._head == (up, left):
            return
        await eyes.look(up, left)
        self._head = (up, left)
        if settle:
            await anyio.sleep(HEAD_SETTLE_S)

    async def _settle(self) -> Pose:
        """The pose once the body has come to rest: the walking policy coasts after a stop."""
        pose = await self._link.pose()
        give_up = anyio.current_time() + SETTLE_S
        while anyio.current_time() < give_up:
            await anyio.sleep(0.25)
            now = await self._link.pose()
            turned = abs(wrap_angle(now.yaw - pose.yaw))
            if now.distance_to(pose.x, pose.y) < 0.005 and turned < math.radians(0.5):
                return now
            pose = now
        return pose

    def _steer(
        self, pose: Pose, x: float, y: float, distance: float, turn: int = 0
    ) -> tuple[float, float]:
        error = wrap_angle(math.atan2(y - pose.y, x - pose.x) - pose.yaw)
        vyaw = max(-self._max_turn, min(self._max_turn, TURN_GAIN * error))
        if not self._gait.turns_in_place:
            # This policy only steers while walking: keep walking, turn in an arc.
            if turn and abs(error) > math.pi / 2:
                return self._max_speed, turn * self._max_turn  # turning round, the chosen way
            # A target inside the turning circle would be orbited forever: straighten out first.
            if distance < 2 * self._radius * abs(math.sin(error)):
                return self._max_speed, 0.0
            return self._max_speed, vyaw
        # Turn toward the target first, then walk; ease off on the last stretch.
        vx = self._max_speed * max(0.0, math.cos(error)) * min(1.0, distance / SLOW_DOWN_WITHIN)
        if vx > 0:
            vx = max(vx, self._gait.min_speed)
        return vx, vyaw

    async def cue(self) -> None:
        try:
            await self._link.quack()
        except Exception as exc:
            raise RobotError(f"could not quack: {exc}") from exc

    async def show_text(self, text: str | None) -> None:
        try:
            await self._link.show_text(text)
        except Exception as exc:
            raise RobotError(f"could not show the question: {exc}") from exc

    async def position(self) -> Pose:
        """Where the duck is, in the base frame."""
        try:
            return self.to_base(await self._link.pose())
        except RobotError:
            raise
        except Exception as exc:
            raise RobotError(f"lost the robot: {type(exc).__name__}: {exc}") from exc

    async def _stop_quietly(self) -> None:
        try:
            await self._link.stop()
        except Exception:
            log.debug("could not send stop", exc_info=True)


def _bearing(pose: Pose, person: Sighting) -> float:
    """Where the person is, relative to the way the body faces."""
    return _error(pose, person.x, person.y)


def _head_gain(look: _Look, command: float) -> float:
    """How far the camera turns sideways per unit of head command, from where it points in a
    look taken with that command (1 when the head is too near the middle to tell)."""
    got = wrap_angle(look.frame.heading - look.pose.yaw)
    if abs(command) > math.radians(3) and abs(got) > math.radians(1):
        return min(3.0, max(0.3, got / command))
    return 1.0


def _points(depth: DepthFrame, pose: Pose) -> list[tuple[float, float]]:
    """What a depth frame saw, in the duck's world (the sensor turns with the head)."""
    x = pose.x + TOF_AHEAD * math.cos(depth.heading)
    y = pose.y + TOF_AHEAD * math.sin(depth.heading)
    return points(depth, x, y)


def _arc_clearance(
    pose: Pose, side: int, radius: float, angle: float, seen: list[tuple[float, float]]
) -> float:
    """How close the body comes to any of the points `seen` while it walks an arc of `radius`
    from `pose`, turning `angle` radians to `side` (+1 left, -1 right)."""
    cx = pose.x - side * radius * math.sin(pose.yaw)
    cy = pose.y + side * radius * math.cos(pose.yaw)
    start = math.atan2(pose.y - cy, pose.x - cx)
    end_x, end_y = (
        cx + radius * math.cos(start + side * angle),
        cy + radius * math.sin(start + side * angle),
    )
    closest = math.inf
    for px, py in seen:
        around = side * wrap_angle(math.atan2(py - cy, px - cx) - start)
        if 0 <= around <= angle:  # beside the arc: as far as from its circle
            gap = abs(math.hypot(px - cx, py - cy) - radius)
        else:  # past either end
            gap = min(math.hypot(px - pose.x, py - pose.y), math.hypot(px - end_x, py - end_y))
        closest = min(closest, gap)
    return closest


def _error(pose: Pose, x: float, y: float) -> float:
    """How far the body would have to turn to face (x, y)."""
    return wrap_angle(math.atan2(y - pose.y, x - pose.x) - pose.yaw)


def _direction(pose: Pose, person: Sighting) -> float:
    return math.atan2(person.y - pose.y, person.x - pose.x)


def _whole(box: Box) -> bool:
    """Whether the picture shows both sides of the person in the box."""
    return box.left > EDGE and box.right < 1 - EDGE


def _half_width(radius: float, distance: float) -> float:
    """The angle half a person of this half-width fills at this distance."""
    return math.asin(min(1.0, radius / max(distance, radius)))


def _seen_bearing(look: _Look, half_width: float) -> float | None:
    """Where the middle of the person in a look is, in the duck's world: the middle of their
    box; or, when the picture cuts them off on one side, half their width in from the side it
    shows; or None when it shows neither (up close, looking up, a person fills the picture)."""
    box, frame = look.box, look.frame
    assert box is not None
    middle = (box.top + box.bottom) / 2
    cut_left, cut_right = box.left <= EDGE, box.right >= 1 - EDGE
    if cut_left and cut_right:
        return None
    if cut_left:
        return frame.bearing(box.right, middle) + half_width
    if cut_right:
        return frame.bearing(box.left, middle) - half_width
    return frame.bearing((box.left + box.right) / 2, middle)


def _box_bearing(look: _Look) -> float:
    """Where the person in a look is, in the duck's world: the middle of their box."""
    box = look.box
    assert box is not None
    return look.frame.bearing((box.left + box.right) / 2, (box.top + box.bottom) / 2)


def _in_the_way(depth: DepthFrame, pose: Pose) -> float | None:
    """The depth sensor's nearest reading along the way the body faces, however the head is
    turned (None if it cannot see that way)."""
    ahead = wrap_angle(pose.yaw - depth.heading)
    return nearest(depth, ahead + IN_THE_WAY, ahead - IN_THE_WAY)


def _sensor(pose: Pose) -> tuple[float, float]:
    """Where the depth sensor is, the head looking ahead."""
    return pose.x + TOF_AHEAD * math.cos(pose.yaw), pose.y + TOF_AHEAD * math.sin(pose.yaw)


def _first(exc: BaseException) -> BaseException:
    """The error inside a task group's exception group."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc
