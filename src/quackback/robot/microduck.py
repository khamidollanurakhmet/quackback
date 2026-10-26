"""MicroduckBackend: walking a Microduck to a point and back, steered from the PC.

The duck itself only takes a velocity twist (robotd's `robot.move {vx, vy, vyaw}`), stops when
twists stop arriving (its 500 ms deadman), and reports where it is (odometry). Going somewhere
is a closed loop on the PC: read the pose, steer toward the target, resend at 10 Hz. A DuckLink
carries those few primitives to one kind of duck: the simulator website today (sandbox.py), the
JSON-RPC socket of a real one later.

Only the newest motion request ever moves the duck: each go_to / send_home / emergency_stop bumps
a generation, a running motion is cancelled on the spot, and a queued one checks its generation
the moment it gets its turn. An emergency stop holds until resume(), which the next question calls.
"""

import logging
import math
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import anyio
from anyio.abc import TaskGroup

from quackback.config import Settings
from quackback.robot.base import Pose, RobotError

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


@dataclass(frozen=True)
class Gait:
    """How a duck's walking policy responds to commands."""

    max_speed: float  # m/s
    max_turn: float  # rad/s
    # Below this forward command the policy stands still instead of walking.
    min_speed: float = 0.0
    # False: the policy only turns while walking forward, so it steers in arcs.
    turns_in_place: bool = True


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


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class MicroduckBackend:
    """A RobotBackend. Use as `async with MicroduckBackend(link, settings) as robot:`."""

    def __init__(self, link: DuckLink, settings: Settings) -> None:
        self._link = link
        gait = link.gait
        self._max_speed = min(settings.max_speed or gait.max_speed, gait.max_speed)
        self._max_turn = min(settings.max_turn or gait.max_turn, gait.max_turn)
        self._gait = gait
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
            await self._link.open()
            self._base = await self._link.pose()
            log.info("robot ready, base at (%.2f, %.2f)", self._base.x, self._base.y)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("robot unavailable, questions go without it: %s", self.error)
        finally:
            self._ready.set()

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
        await self._run_motion(x, y, stop_short, generation)

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
            await self._run_motion(0.0, 0.0, 0.0, generation)
        except RobotError as exc:
            log.info("robot did not make it home: %s", exc)

    async def _run_motion(self, x: float, y: float, stop_short: float, generation: int) -> None:
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
                    await self._drive_to(x, y, stop_short)
                finally:
                    self._motion = None
                    with anyio.CancelScope(shield=True):
                        await self._stop_quietly()
            if scope.cancelled_caught:
                raise RobotError(
                    "emergency stop" if self._halted else "interrupted by a newer motion"
                )

    async def _drive_to(self, x: float, y: float, stop_short: float) -> None:
        if self._max_speed < self._gait.min_speed:
            raise RobotError(
                f"the speed cap ({self._max_speed} m/s) is below the slowest this duck "
                f"walks ({self._gait.min_speed} m/s)"
            )
        target_x, target_y = self._to_link(x, y)
        self._check_inside_walls(x, y, target_x, target_y)
        arrive = max(ARRIVE_RADIUS, stop_short + STOP_MARGIN) if stop_short > 0 else ARRIVE_RADIUS
        try:
            pose = await self._link.pose()
            distance = pose.distance_to(target_x, target_y)
            now = anyio.current_time()
            deadline = now + NAV_TIMEOUT_BASE_S + distance / NAV_CRAWL_SPEED
            best, best_at = distance, now
            vyaw = 0.0
            while distance > arrive:
                now = anyio.current_time()
                if now > deadline:
                    raise RobotError(f"did not reach ({x:.2f}, {y:.2f}): {distance:.2f} m short")
                if distance < best - STALL_PROGRESS:
                    best, best_at = distance, now
                elif now - best_at > STALL_WINDOW_S:
                    raise RobotError(f"stuck {distance:.2f} m from ({x:.2f}, {y:.2f})")
                vx, want_vyaw = self._steer(pose, target_x, target_y, distance)
                # Only the turn is smoothed: easing the speed toward a gait's minimum would
                # leave it just below, where the policy stands still.
                vyaw += SMOOTHING * (want_vyaw - vyaw)
                await self._link.move(vx, 0.0, vyaw)
                await anyio.sleep(1 / CONTROL_HZ)
                pose = await self._link.pose()
                distance = pose.distance_to(target_x, target_y)
        except RobotError:
            raise
        except Exception as exc:
            raise RobotError(f"lost the robot: {type(exc).__name__}: {exc}") from exc
        log.info("robot reached (%.2f, %.2f)", x, y)

    def _check_inside_walls(self, x: float, y: float, link_x: float, link_y: float) -> None:
        bounds = self._link.bounds
        if bounds is None:
            return
        x_min, x_max, y_min, y_max = bounds
        inside_x = x_min + WALL_MARGIN <= link_x <= x_max - WALL_MARGIN
        inside_y = y_min + WALL_MARGIN <= link_y <= y_max - WALL_MARGIN
        if not (inside_x and inside_y):
            raise RobotError(f"({x:.2f}, {y:.2f}) is outside the walls of the duck's world")

    def _steer(self, pose: Pose, x: float, y: float, distance: float) -> tuple[float, float]:
        error = wrap_angle(math.atan2(y - pose.y, x - pose.x) - pose.yaw)
        vyaw = max(-self._max_turn, min(self._max_turn, TURN_GAIN * error))
        if not self._gait.turns_in_place:
            # This policy only steers while walking: keep walking, turn in an arc. A target
            # inside the turning circle would be orbited forever, so straighten out first.
            radius = self._max_speed / self._max_turn
            if distance < 2 * radius * abs(math.sin(error)):
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
        return self.to_base(await self._link.pose())

    async def _stop_quietly(self) -> None:
        try:
            await self._link.stop()
        except Exception:
            log.debug("could not send stop", exc_info=True)
