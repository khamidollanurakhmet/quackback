import math

import anyio
import pytest

from quackback.config import Settings
from quackback.robot.base import Pose, RobotError
from quackback.robot.microduck import Gait, MicroduckBackend, wrap_angle
from quackback.robot.mock import MockLink

pytestmark = pytest.mark.anyio

# The simulator website's walking policy, as measured: walks only at 0.25 m/s, turns in arcs.
SANDBOX = Gait(max_speed=0.25, max_turn=1.0, min_speed=0.25, turns_in_place=False)
NIMBLE = Gait(max_speed=0.3, max_turn=1.0)


@pytest.fixture(autouse=True)
def _quick_head(monkeypatch: pytest.MonkeyPatch) -> None:
    # The mock duck has eyes: an arc-walking one looks about before it turns round.
    from quackback.robot import microduck

    monkeypatch.setattr(microduck, "HEAD_SETTLE_S", 0.0)
    monkeypatch.setattr(microduck, "SCAN_S", 0.0)


def _backend(link: MockLink, **settings: object) -> MicroduckBackend:
    return MicroduckBackend(link, Settings(**settings))  # type: ignore[arg-type]


async def _ready(robot: MicroduckBackend) -> None:
    assert await robot.wait_ready(5)


@pytest.mark.parametrize("gait", [SANDBOX, NIMBLE], ids=["arcs", "turns-in-place"])
@pytest.mark.parametrize(
    "target",
    [(1.0, 0.0), (0.6, 0.6), (-0.5, 0.2), (0.1, 0.3), (0.0, -0.4), (0.2, 0.0)],
    ids=["ahead", "diagonal", "behind", "near-left", "beside-right", "just-ahead"],
)
async def test_walks_to_a_point(gait: Gait, target: tuple[float, float]) -> None:
    link = MockLink(gait, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(30):
            await robot.go_to(*target)
        assert (await robot.position()).distance_to(*target) < 0.2
        assert link.stopped  # stopped on arrival, before shutdown


async def test_arcs_with_a_tight_turn_cap_still_arrive() -> None:
    # 0.25 m/s at 0.4 rad/s is a 0.6 m turning radius: these targets start inside the circle.
    link = MockLink(SANDBOX, time_scale=4)
    async with _backend(link, max_turn=0.4) as robot:
        await _ready(robot)
        for target in [(0.6, 0.6), (0.0, 0.0), (0.3, -0.5)]:
            with anyio.fail_after(60):
                await robot.go_to(*target)
            assert (await robot.position()).distance_to(*target) < 0.2


async def test_stops_short_of_the_human() -> None:
    link = MockLink(SANDBOX, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(30):
            await robot.go_to(1.5, 0.0, stop_short=0.4)
        distance = (await robot.position()).distance_to(1.5, 0.0)
        assert 0.4 <= distance <= 0.6


async def test_base_frame_follows_the_start_pose() -> None:
    # Starting at (2, 1) facing +y: one metre forward is (2, 2) in the link's frame.
    link = MockLink(NIMBLE, start=Pose(2.0, 1.0, math.pi / 2), time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(20):
            await robot.go_to(1.0, 0.0)
        assert (await link.pose()).distance_to(2.0, 2.0) < 0.2


async def test_speed_caps_hold() -> None:
    link = MockLink(NIMBLE, time_scale=4)
    async with _backend(link, max_speed=0.1, max_turn=0.4) as robot:
        await _ready(robot)
        with anyio.fail_after(30):
            await robot.go_to(0.5, 0.3)
    assert max(abs(vx) for vx, _, _ in link.moves) <= 0.1 + 1e-9
    assert max(abs(vyaw) for _, _, vyaw in link.moves) <= 0.4 + 1e-9


async def test_a_cap_below_the_gait_minimum_is_an_error() -> None:
    async with _backend(MockLink(SANDBOX), max_speed=0.15) as robot:
        await _ready(robot)
        with pytest.raises(RobotError, match="slowest"):
            await robot.go_to(1.0, 0.0)


async def test_points_outside_the_walls_are_refused() -> None:
    # Like the simulator: walls at +-1.5 m, the duck starting at (-0.6, 0) facing +x.
    link = MockLink(SANDBOX, start=Pose(-0.6, 0.0, 0.0), bounds=(-1.5, 1.5, -1.5, 1.5))
    async with _backend(link) as robot:
        await _ready(robot)
        with pytest.raises(RobotError, match="outside the walls"):
            await robot.go_to(2.5, 0.0)
        assert link.moves == []


async def test_unreachable_point_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from quackback.robot import microduck

    monkeypatch.setattr(microduck, "NAV_TIMEOUT_BASE_S", 0.3)
    monkeypatch.setattr(microduck, "NAV_CRAWL_SPEED", 100.0)
    link = MockLink(NIMBLE, time_scale=0.0)  # a frozen world: the duck never gets anywhere
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(5), pytest.raises(RobotError, match="did not reach"):
            await robot.go_to(1.0, 0.0)


async def test_no_progress_is_stuck(monkeypatch: pytest.MonkeyPatch) -> None:
    from quackback.robot import microduck

    monkeypatch.setattr(microduck, "STALL_WINDOW_S", 0.3)
    link = MockLink(NIMBLE, time_scale=0.0)  # pushing against a wall
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(5), pytest.raises(RobotError, match="stuck"):
            await robot.go_to(1.0, 0.0)
        assert link.stopped


async def test_cancelled_walk_stops_the_duck() -> None:
    link = MockLink(NIMBLE)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.move_on_after(0.4):
            await robot.go_to(3.0, 0.0)
        assert link.moves  # it was moving
        assert link.stopped  # and was stopped when the walk was cancelled


async def test_lost_link_is_a_robot_error_and_a_stop() -> None:
    link = MockLink(NIMBLE, fail_after_moves=3)
    async with _backend(link) as robot:
        await _ready(robot)
        with pytest.raises(RobotError, match="lost the robot"):
            await robot.go_to(1.0, 0.0)
        assert link.stops >= 1


async def test_failed_connection_is_not_ready() -> None:
    async with _backend(MockLink(fail_open=True)) as robot:
        assert not await robot.wait_ready(1)
        assert robot.error is not None
        assert "did not start" in robot.error
        with pytest.raises(RobotError, match="not connected"):
            await robot.go_to(1.0, 0.0)
        robot.send_home()  # a no-op, not a crash


async def test_send_home_walks_back_in_the_background() -> None:
    link = MockLink(SANDBOX, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(20):
            await robot.go_to(0.8, 0.0)
        robot.send_home()
        for _ in range(200):  # up to 20 s
            if (await robot.position()).distance_to(0.0, 0.0) < 0.2:
                break
            await anyio.sleep(0.1)
        assert (await robot.position()).distance_to(0.0, 0.0) < 0.2


async def test_a_new_walk_takes_over_from_going_home() -> None:
    link = MockLink(NIMBLE, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(20):
            await robot.go_to(1.0, 0.0)
        robot.send_home()
        await anyio.sleep(0.2)  # the home trip is under way
        closest_to_home = math.inf

        async def watch() -> None:
            nonlocal closest_to_home
            while True:
                closest_to_home = min(closest_to_home, (await robot.position()).distance_to(0, 0))
                await anyio.sleep(0.02)

        async with anyio.create_task_group() as tg:
            tg.start_soon(watch)
            with anyio.fail_after(20):
                await robot.go_to(1.0, 0.5)
            tg.cancel_scope.cancel()
        assert (await robot.position()).distance_to(1.0, 0.5) < 0.2
        assert closest_to_home > 0.5  # it turned back; it did not finish the trip home first


async def test_a_queued_home_trip_never_overtakes_a_newer_walk() -> None:
    link = MockLink(NIMBLE, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)
        with anyio.fail_after(20):
            await robot.go_to(1.0, 0.0)
        robot.send_home()  # scheduled, not yet running...
        with anyio.fail_after(20):
            await robot.go_to(1.0, 0.5)  # ...and superseded right away
        await anyio.sleep(0.5)
        assert (await robot.position()).distance_to(1.0, 0.5) < 0.2  # it stayed


async def test_emergency_stop_interrupts_and_holds() -> None:
    link = MockLink(NIMBLE, time_scale=4)
    async with _backend(link) as robot:
        await _ready(robot)

        async def walk() -> None:
            with pytest.raises(RobotError, match="emergency stop"):
                await robot.go_to(5.0, 0.0)

        async with anyio.create_task_group() as tg:
            tg.start_soon(walk)
            await anyio.sleep(0.5)
            await robot.emergency_stop()
        held_at = await robot.position()
        moves = len(link.moves)
        assert held_at.distance_to(0.0, 0.0) > 0.3  # away from home, so a trip home would show
        robot.send_home()  # ignored while stopped
        with pytest.raises(RobotError, match="emergency stop"):
            await robot.go_to(0.0, 0.0)  # refused until resume()
        await anyio.sleep(0.5)
        assert len(link.moves) == moves
        assert (await robot.position()).distance_to(held_at.x, held_at.y) < 0.01
        robot.resume()
        with anyio.fail_after(20):
            await robot.go_to(0.0, 0.0)


async def test_emergency_stop_also_cancels_a_walk_waiting_its_turn() -> None:
    # A slow stop() keeps the first walk holding the motion lock while the second one queues.
    link = MockLink(NIMBLE, time_scale=4, stop_delay=0.3)
    async with _backend(link) as robot:
        await _ready(robot)
        refused: list[str] = []

        async def walk(x: float, y: float) -> None:
            try:
                await robot.go_to(x, y)
            except RobotError as exc:
                refused.append(str(exc))

        async with anyio.create_task_group() as tg:
            tg.start_soon(walk, 2.0, 0.0)
            await anyio.sleep(0.3)
            tg.start_soon(walk, 0.0, 2.0)  # supersedes the first, waits for the lock
            await anyio.sleep(0.05)
            await robot.emergency_stop()
            moves = len(link.moves)
        assert len(refused) == 2
        assert len(link.moves) == moves  # nothing moved after the stop


async def test_emergency_stop_during_start_up_holds_the_first_walk() -> None:
    from quackback.ask import Asker
    from tests.fakes import FakeKeyboard, FakeVoice

    settings = Settings(robot="sandbox")
    link = MockLink(NIMBLE, time_scale=4, open_delay=0.5)  # still booting when asked
    async with _backend(link) as robot:
        asker = Asker(settings, voice=FakeVoice(["fine"]), keyboard=FakeKeyboard(), robot=robot)
        result = None

        async def ask() -> None:
            nonlocal result
            result = await asker.ask("Q?")

        async with anyio.create_task_group() as tg:
            tg.start_soon(ask)
            await anyio.sleep(0.2)
            assert await asker.stop_robot()
        assert result is not None
        assert result["answer"] == "fine"  # the question still went ahead
        assert link.moves == []  # but the duck never moved


async def test_shutdown_stops_and_closes() -> None:
    link = MockLink(NIMBLE)
    async with _backend(link) as robot:
        await _ready(robot)
    assert link.closed
    assert link.stops >= 1


async def test_cue_and_text_reach_the_link() -> None:
    link = MockLink()
    async with _backend(link) as robot:
        await _ready(robot)
        await robot.show_text("Which database?")
        await robot.cue()
        await robot.show_text(None)
    assert link.quacks == 1
    assert link.texts == ["Which database?", None]


def test_wrap_angle() -> None:
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)
