"""Stand-in robots for tests: MockBackend records what ask_human asked of the robot, MockLink is
a small kinematic duck for exercising MicroduckBackend's steering without a simulator.
"""

import math

import anyio

from quackback.robot.base import Pose, RobotError
from quackback.robot.microduck import Gait

DEADMAN_S = 0.5


class MockBackend:
    def __init__(self, *, ready: bool = True, fail: bool = False, walk_time: float = 0.0) -> None:
        self.ready = ready
        self.fail = fail
        self.walk_time = walk_time
        self.calls: list[str] = []

    async def wait_ready(self, max_wait: float) -> bool:
        return self.ready

    async def go_to(self, x: float, y: float, stop_short: float = 0.0) -> None:
        self.calls.append(f"go_to {x:g},{y:g}" + (f" short {stop_short:g}" if stop_short else ""))
        await anyio.sleep(self.walk_time)
        if self.fail:
            raise RobotError("a leg fell off")

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
    ) -> None:
        self.gait = gait or Gait(max_speed=0.25, max_turn=1.0)
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
        if not walking:
            vx = 0.0
            if not self.gait.turns_in_place:
                vyaw = 0.0
        p = self._pose
        yaw = p.yaw + vyaw * dt
        self._pose = Pose(p.x + vx * math.cos(yaw) * dt, p.y + vx * math.sin(yaw) * dt, yaw)
