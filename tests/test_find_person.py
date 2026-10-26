import math
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import numpy as np
import pytest

from quackback.config import Settings
from quackback.robot.base import CameraFrame, Pose, RobotError
from quackback.robot.depth import DepthFrame
from quackback.robot.microduck import (
    ARRIVE_RADIUS,
    CONTROL_HZ,
    FACE_PITCH,
    SEARCH_PITCH,
    TOF_AHEAD,
    WALK_STOP,
    Gait,
    MicroduckBackend,
    _Look,
    wrap_angle,
)
from quackback.robot.mock import MockDetector, MockLink
from quackback.vision.person_detector import Box

pytestmark = pytest.mark.anyio

# The mock walks exactly as told, so its arcs are 0.25 m / 1 rad/s = 0.25 m in radius.
SANDBOX = Gait(max_speed=0.25, max_turn=1.0, min_speed=0.25, turns_in_place=False, turn_radius=0.25)
NIMBLE = Gait(max_speed=0.3, max_turn=1.0)
# Like the simulator's: told 0.25 m/s and 1 rad/s, it walks 0.1 m/s and turns 0.58 rad/s, so its
# circles are 0.17 m, not the 0.25 m the commands make.
SIMLIKE = Gait(max_speed=0.25, max_turn=1.0, min_speed=0.25, turns_in_place=False, turn_radius=0.17)


def _simlike(**options: Any) -> MockLink:
    return MockLink(SIMLIKE, time_scale=SCALE, speed_scale=0.4, turn_scale=0.58, **options)


GAITS = pytest.mark.parametrize("gait", [SANDBOX, NIMBLE], ids=["arcs", "turns-in-place"])
# Twice as fast as life: quick, while a slow test machine still leaves the stops a margin.
SCALE = 2


@pytest.fixture(autouse=True)
def _quick_head(monkeypatch: pytest.MonkeyPatch) -> None:
    from quackback.robot import microduck

    monkeypatch.setattr(microduck, "HEAD_SETTLE_S", 0.0)
    monkeypatch.setattr(microduck, "LOOK_S", 0.05)
    monkeypatch.setattr(microduck, "SCAN_S", 0.0)


def _backend(link: Any, detector: MockDetector | None = None) -> MicroduckBackend:
    return MicroduckBackend(link, Settings(), MockDetector() if detector is None else detector)


def _sensor_distance(link: MockLink, pose: Pose) -> float:
    """From the person to the duck's nose (the depth sensor, ahead of the body's centre),
    whichever way the duck faces: the closest the duck itself comes to them."""
    assert link.person is not None
    return pose.distance_to(*link.person) - TOF_AHEAD


def _nearest_thing(link: MockLink, pose: Pose) -> float:
    """Like _sensor_distance, to the nearest of the person and the obstacles."""
    things = ([link.person] if link.person else []) + link.obstacles
    return min((pose.distance_to(*thing) - TOF_AHEAD for thing in things), default=math.inf)


async def _closest_during(
    link: MockLink, action: Callable[[], Awaitable[Any]]
) -> tuple[Any, float]:
    """Run `action`, watching how close the duck ever gets to the person (or an obstacle).
    The action's RobotError, if any, is returned, not raised."""
    closest = math.inf

    async def watch() -> None:
        nonlocal closest
        while True:
            closest = min(closest, _nearest_thing(link, await link.pose()))
            await anyio.sleep(0.005)

    async with anyio.create_task_group() as tg:
        tg.start_soon(watch)
        try:
            result: Any = await action()
        except RobotError as exc:
            result = exc
        tg.cancel_scope.cancel()
    return result, closest


async def _gaze_error(link: MockLink) -> float:
    """How far the camera points from the person, radians."""
    assert link.person is not None
    pose = await link.pose()
    heading = (await link.camera_frame()).heading
    return abs(wrap_angle(heading - math.atan2(link.person[1] - pose.y, link.person[0] - pose.x)))


def test_camera_bearings() -> None:
    frame = CameraFrame(np.zeros((640, 360, 3), np.uint8), math.radians(72.7), 1.0, 0.0)
    assert frame.bearing(0.5) == pytest.approx(1.0)
    assert frame.hfov == pytest.approx(math.radians(45), abs=0.01)
    assert frame.bearing(0.0) == pytest.approx(1.0 + frame.hfov / 2)
    assert frame.bearing(1.0) == pytest.approx(1.0 - frame.hfov / 2)
    looking_up = CameraFrame(frame.image, frame.vfov, 1.0, math.radians(30))
    assert looking_up.bearing(0.0) > frame.bearing(0.0)  # a tilted view takes in more sideways


@pytest.mark.parametrize("roll", [0.0, math.radians(20), math.radians(-25)])
def test_camera_bearings_allow_for_a_tilted_picture(roll: float) -> None:
    pitch = math.radians(45)
    frame = CameraFrame(np.zeros((640, 360, 3), np.uint8), math.radians(72.7), 0.3, pitch, roll)
    # The camera's axes in the world, with the heading taken out: ahead, left, up.
    ahead = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
    left0, up0 = np.array([0.0, 1.0, 0.0]), np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
    left = math.cos(roll) * left0 + math.sin(roll) * up0
    up = -math.sin(roll) * left0 + math.cos(roll) * up0
    focal = 320 / math.tan(frame.vfov / 2)
    for bearing, rise in [(0.0, 1.6), (0.2, 1.0), (-0.15, 0.5)]:
        ray = np.array([math.cos(bearing), math.sin(bearing), rise])
        x, y, z = ray @ ahead, ray @ left, ray @ up
        u, v = 0.5 - focal * y / x / 360, 0.5 - focal * z / x / 640
        assert frame.bearing(u, v) == pytest.approx(0.3 + bearing, abs=1e-9)


@pytest.mark.parametrize("bearing", [math.radians(15), math.radians(-8), 0.0])
async def test_the_mock_camera_and_detector_agree(bearing: float) -> None:
    link = MockLink(NIMBLE, person=(2 * math.cos(bearing), 2 * math.sin(bearing)))
    await link.look(SEARCH_PITCH)
    frame = await link.camera_frame()
    (box,) = MockDetector().detect(frame.image)
    middle = (box.top + box.bottom) / 2
    seen = frame.bearing((box.left + box.right) / 2, middle)
    assert seen == pytest.approx(bearing, abs=math.radians(3))


@GAITS
@pytest.mark.parametrize(
    "person",
    [(1.5, 0.0), (1.2, -0.8), (0.6, 1.1), (-1.0, 0.5), (-1.2, -0.05), (0.1, 0.6)],
    ids=["ahead", "right", "left", "behind", "right-behind", "beside"],
)
async def test_finds_walks_up_to_and_faces_the_person(
    gait: Gait, person: tuple[float, float]
) -> None:
    link = MockLink(gait, time_scale=SCALE, person=person)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        assert robot.can_find_person
        with anyio.fail_after(60):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=30)
            )
        assert found is True
        assert link.stopped
        assert closest >= 0.4  # never within 40 cm of the depth sensor, all the way
        assert _sensor_distance(link, await link.pose()) <= 0.85  # it did walk up
        assert link.head is not None
        assert link.head[0] == pytest.approx(FACE_PITCH)
        assert await _gaze_error(link) < math.radians(6)
        if gait.turns_in_place:
            assert abs(link.head[1]) < math.radians(12)  # the body turned, not just the head


@pytest.mark.parametrize("head_gain", [0.6, 1.0, 2.0])
async def test_the_head_is_aimed_by_where_the_camera_really_points(head_gain: float) -> None:
    # Right beside the duck: close enough to stop at once, so only the head can face them.
    link = MockLink(SANDBOX, time_scale=SCALE, person=(0.1, 0.62), head_gain=head_gain)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        assert (await link.pose()).distance_to(0.0, 0.0) < 0.01  # it never had to walk
        assert await _gaze_error(link) < math.radians(6)
        assert link.head is not None
        # The head was told to turn as far as it takes for this head to turn the camera there.
        assert link.head[1] * head_gain == pytest.approx(math.radians(81), abs=math.radians(6))


async def test_nobody_there_gives_up_and_rests_the_head() -> None:
    link = MockLink(SANDBOX, time_scale=SCALE)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        started = anyio.current_time()
        with anyio.fail_after(10):
            assert not await robot.approach_person(search_timeout=1.0)
        assert anyio.current_time() - started >= 1.0
        assert link.head is None  # back at rest
        sideways = [left for up, left in filter(None, link.looks) if up == SEARCH_PITCH]
        assert max(sideways) >= math.radians(100)  # it looked around, both ways
        assert min(sideways) <= -math.radians(100)
        assert any(vx > 0 for vx, _, _ in link.moves)  # and turned (in arcs) between looks
        assert link.stopped


class Blindfolded(MockDetector):
    """Sees nobody, ever: a seated person, a person it does not recognise, a pillar."""

    def detect(self, image: np.ndarray) -> list[Box]:
        return []


@pytest.mark.parametrize(
    "thing",
    [(0.0, 0.5), (0.0, -0.55), (0.53, 0.0), (0.3, 0.45)],
    ids=["left", "right", "ahead", "front-left"],
)
async def test_looking_around_never_walks_into_what_it_cannot_recognise(
    thing: tuple[float, float],
) -> None:
    link = MockLink(SANDBOX, time_scale=SCALE, person=thing)
    async with _backend(link, Blindfolded()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(20):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=3.0)
            )
        assert found is False
        assert closest >= 0.4


@pytest.mark.parametrize(
    "things",
    [((0.53, 0.0),), ((0.1, 0.52), (0.1, -0.52))],
    ids=["in-front", "either-side"],
)
async def test_with_no_room_to_turn_it_only_looks(things: tuple[tuple[float, float], ...]) -> None:
    link = MockLink(SANDBOX, time_scale=SCALE, obstacles=things)
    async with _backend(link, Blindfolded()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(10):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=1.0)
            )
        assert found is False
        assert not any(vx > 0 for vx, _, _ in link.moves)
        assert closest >= 0.4


class Intruder(MockLink):
    """Someone steps in front of the duck the moment it starts to walk."""

    async def move(self, vx: float, vy: float, vyaw: float) -> None:
        if vx > 0 and not self.obstacles:
            pose = await self.pose()
            ahead = pose.yaw + 0.5  # a little to the side it turns to, half a metre out
            self.obstacles.append(
                (pose.x + 0.58 * math.cos(ahead), pose.y + 0.58 * math.sin(ahead))
            )
        await super().move(vx, vy, vyaw)


async def test_someone_stepping_into_a_search_turn_stops_it() -> None:
    link = Intruder(SANDBOX, time_scale=SCALE)
    async with _backend(link, Blindfolded()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(20):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=2.0)
            )
        assert found is False
        assert link.obstacles  # they did step in
        assert closest >= 0.4


async def test_losing_the_person_means_looking_again(monkeypatch: pytest.MonkeyPatch) -> None:
    from quackback.robot import microduck

    monkeypatch.setattr(microduck, "LOST_AFTER_S", 0.3)
    link = MockLink(NIMBLE, time_scale=SCALE, person=(2.5, 0.0))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        looks_before = 0

        async def vanish() -> None:
            nonlocal looks_before
            await anyio.sleep(0.5)
            looks_before = len(link.looks)
            link.person = None

        started = anyio.current_time()
        async with anyio.create_task_group() as tg:
            tg.start_soon(vanish)
            with anyio.fail_after(15):
                assert not await robot.approach_person(search_timeout=1.5)
        # Last seen at 0.4 s at the earliest, lost 0.3 s later, then a whole new search.
        assert anyio.current_time() - started >= 0.4 + 0.3 + 1.5
        again = [look for look in link.looks[looks_before:] if look and look[0] == SEARCH_PITCH]
        assert any(left > math.radians(30) for _, left in again)  # it looked around again
        assert link.stopped


class ShortSighted(MockDetector):
    """Loses the person once they fill much of the picture, as a real detector can up close."""

    def detect(self, image: np.ndarray) -> list[Box]:
        boxes = super().detect(image)
        return [b for b in boxes if b.right - b.left < 0.35]


@GAITS
async def test_the_depth_sensor_stops_the_duck_when_the_camera_cannot_tell(gait: Gait) -> None:
    link = MockLink(gait, time_scale=SCALE, person=(1.5, 0.0))
    async with _backend(link, ShortSighted()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=5)
            )
        assert found is True
        assert closest >= 0.4


@pytest.mark.parametrize(
    "person",
    [(1.2, 0.0), (0.9, 0.3), (0.8, -0.4), (0.3, 0.9)],
    ids=["ahead", "left", "right", "far-left"],
)
async def test_leaving_turns_away_and_keeps_its_distance(person: tuple[float, float]) -> None:
    link = MockLink(SANDBOX, time_scale=SCALE, person=person)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        with anyio.fail_after(60):
            error, closest = await _closest_during(link, lambda: robot.go_to(0.0, 0.0))
        assert error is None  # home, and
        assert closest >= 0.4  # never closer than 40 cm on the way


async def test_leaving_someone_beside_it_turns_the_other_way() -> None:
    # Found right beside it: no walking up, just the head. The way out is behind, so the arc
    # must swing away from them, not through them.
    link = MockLink(SANDBOX, time_scale=SCALE, person=(0.1, 0.62))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        start = _sensor_distance(link, await link.pose())
        with anyio.fail_after(60):
            error, closest = await _closest_during(link, lambda: robot.go_to(-0.6, 0.0))
        assert error is None
        assert closest >= start - 0.05


@pytest.mark.parametrize(
    "pillar",
    [(0.9, 0.05), (0.9, 0.13)],
    ids=["in-line", "beside-the-line"],  # the second never looks like part of the person
)
async def test_an_obstacle_on_the_way_to_the_person_stops_the_walk(
    pillar: tuple[float, float],
) -> None:
    link = MockLink(NIMBLE, time_scale=SCALE, person=(1.8, 0.0), obstacles=(pillar,))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=5)
            )
        assert found is True  # as close as it can get
        assert closest >= 0.4  # and not through the obstacle


async def test_a_pillar_in_the_way_of_an_arc_stops_the_walk() -> None:
    # An arc toward someone off to the left sweeps over a pillar that is nowhere near the
    # direction of the person: only the check for anything in the duck's way can stop it.
    link = MockLink(SANDBOX, time_scale=SCALE, person=(0.6, 1.2), obstacles=((0.52, 0.30),))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            found, closest = await _closest_during(
                link, lambda: robot.approach_person(search_timeout=5)
            )
        assert found is True
        assert closest >= 0.4


class Fidget(MockLink):
    """The person steps aside as soon as the duck stops in front of them."""

    stepped = False

    async def stop(self) -> None:
        if self.person is not None and not self.stepped and any(vx > 0 for vx, _, _ in self.moves):
            self.stepped = True
            pose = await self.pose()
            aside = pose.yaw - math.pi / 2  # 20 cm to the duck's right
            self.person = (
                self.person[0] + 0.2 * math.cos(aside),
                self.person[1] + 0.2 * math.sin(aside),
            )
        await super().stop()


@GAITS
async def test_it_faces_the_person_where_they_are_now(gait: Gait) -> None:
    link = Fidget(gait, time_scale=SCALE, person=(1.4, 0.0))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        assert link.stepped
        assert await _gaze_error(link) < math.radians(6)


async def test_a_walk_stops_for_someone_in_the_way() -> None:
    link = MockLink(NIMBLE, time_scale=SCALE, person=(1.0, 0.0))
    async with MicroduckBackend(link, Settings()) as robot:  # eyes, but no detector
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            error, closest = await _closest_during(link, lambda: robot.go_to(2.0, 0.0))
        assert isinstance(error, RobotError)
        assert "stopped for something" in str(error)
        assert closest >= 0.4
        assert _sensor_distance(link, await link.pose()) <= WALK_STOP + 0.05


@GAITS
async def test_walking_up_to_the_human_without_a_detector_stops_by_depth(gait: Gait) -> None:
    link = MockLink(gait, time_scale=SCALE, person=(1.5, 0.0))
    async with MicroduckBackend(link, Settings()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            error, closest = await _closest_during(
                link, lambda: robot.go_to(1.5, 0.0, stop_short=0.4)
            )
        assert error is None  # arrived: the human is where it was told
        assert closest >= 0.4
        # Stopped by the depth sensor, at the distance it stops at for a person (further out
        # for an arc-walking duck, to have room to turn round and leave), give or take two
        # ticks of walking on a slow machine; not 0.42 m from the nose, where it would stop
        # on its own, 0.5 m short of the target.
        two_ticks = 2 * gait.max_speed * SCALE / CONTROL_HZ
        assert _sensor_distance(link, await link.pose()) >= robot._person_stop - two_ticks
        with anyio.fail_after(60):
            error, closest = await _closest_during(link, lambda: robot.go_to(0.0, 0.0))
        assert error is None
        assert closest >= 0.4


@GAITS
@pytest.mark.parametrize(
    "person", [(1.3, 0.5), (1.2, 0.5), (1.3, -0.5)], ids=["left", "near-left", "right"]
)
async def test_leaving_someone_beside_where_it_stopped_keeps_its_distance(
    gait: Gait, person: tuple[float, float]
) -> None:
    # No detector: it walks to robot_target (1.5, 0) while the person stands beside it.
    link = MockLink(gait, time_scale=SCALE, person=person)
    async with MicroduckBackend(link, Settings()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(60):
            out, closest_out = await _closest_during(
                link, lambda: robot.go_to(1.5, 0.0, stop_short=0.4)
            )
            back, closest_back = await _closest_during(link, lambda: robot.go_to(0.0, 0.0))
        assert out is None
        # Home, or stopped for want of room: never through them.
        assert back is None or "no room" in str(back) or "stopped for something" in str(back)
        assert min(closest_out, closest_back) >= 0.4


@pytest.mark.parametrize("person", [(1.05, 0.0), (0.75, 0.75), (0.8, -0.7), (1.0, 0.25)])
async def test_a_short_trip_home_after_an_approach_gets_home(
    person: tuple[float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    from quackback.robot import microduck

    # Turning round takes it about 2.7 s here; any real progress home then comes within a second.
    monkeypatch.setattr(microduck, "STALL_WINDOW_S", 3.0)
    link = _simlike(person=person)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        with anyio.fail_after(60):
            error, closest = await _closest_during(link, lambda: robot.go_to(0.0, 0.0))
        assert error is None
        assert (await robot.position()).distance_to(0.0, 0.0) <= ARRIVE_RADIUS + 0.05
        assert closest >= 0.4


def test_arcs_go_by_the_circle_the_duck_really_walks() -> None:
    robot = MicroduckBackend(MockLink(SIMLIKE), Settings())
    # 0.3 m off, 60 degrees to the left: outside the 0.17 m circle it really walks, so it can
    # arc in, though inside the 0.25 m one its commands alone describe (it would go straight).
    bearing = math.radians(60)
    x, y = 0.3 * math.cos(bearing), 0.3 * math.sin(bearing)
    vx, vyaw = robot._steer(Pose(0.0, 0.0, 0.0), x, y, 0.3)
    assert vx > 0
    assert vyaw > 0


async def test_the_head_sweep_leaves_no_gap_however_far_the_head_turns_the_camera() -> None:
    # Head up, this head turns the camera twice as far as told: steps calibrated at rest would
    # look past someone standing 2 m away, 37 degrees to the left. It turns on the spot to
    # search, and must not need to.
    link = MockLink(NIMBLE, time_scale=SCALE, person=(1.6, 1.2), head_gain=2.0)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
        assert not any(vx == 0 and vyaw > 0 for vx, _, vyaw in link.moves)  # no search turn


async def test_a_person_seen_at_the_edge_is_put_in_the_middle_of_the_picture() -> None:
    # Found 50 degrees to the left at the edge of a look 35 degrees round; this head turns the
    # camera twice as far as told, so half the offset is what centres them.
    link = MockLink(NIMBLE, time_scale=SCALE, person=(1.29, 1.53), head_gain=2.0)
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=10)
    searching = [left for up, left in filter(None, link.looks) if up == SEARCH_PITCH]
    back_ahead = next(i for i, left in enumerate(searching) if i and left == 0.0)
    centred = searching[back_ahead - 1] * 2.0  # where the camera pointed, before walking
    assert centred == pytest.approx(math.radians(50), abs=math.radians(6))


async def test_an_arc_stops_for_a_pillar_beside_its_way() -> None:
    # Walking a curve to the left: the pillar never sits straight ahead, where a straight walk
    # would look for it, until the duck is on top of it.
    link = MockLink(SANDBOX, time_scale=SCALE, obstacles=((0.52, 0.30),))
    async with MicroduckBackend(link, Settings()) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            error, closest = await _closest_during(link, lambda: robot.go_to(0.6, 1.2))
        assert "stopped for something" in str(error)
        assert closest >= 0.4


def test_a_sighting_across_the_half_turn_line_keeps_its_range() -> None:
    # Facing almost straight back, the camera's heading and the depth sensor's (read a moment
    # apart) can fall either side of +-180 degrees.
    image = np.zeros((160, 90, 3), np.uint8)
    image[40:120, 30:60] = 255
    frame = CameraFrame(image, math.radians(72.7), 3.13, 0.0)
    rows = [[(0, 255)] * 8 for _ in range(8)]
    for row in (3, 4):  # the rows about level: their slant range is the level one
        rows[row][3] = rows[row][4] = (600, 5)
    depth = DepthFrame(
        8,
        8,
        tuple(mm for r in rows for mm, _ in r),
        tuple(s for r in rows for _, s in r),
        0.0,
        -3.13,
    )
    (box,) = MockDetector().detect(image)
    robot = MicroduckBackend(MockLink(NIMBLE), Settings())
    sighting = robot._place(_Look(frame, box, depth, Pose(0.0, 0.0, 3.13)), None, 0.0)
    assert math.hypot(sighting.x, sighting.y) == pytest.approx(0.6 + TOF_AHEAD, abs=0.01)


async def test_emergency_stop_interrupts_the_approach() -> None:
    link = MockLink(NIMBLE, person=(3.0, 0.0))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)

        async def approach() -> None:
            with pytest.raises(RobotError, match="emergency stop"):
                await robot.approach_person(search_timeout=5)

        async with anyio.create_task_group() as tg:
            tg.start_soon(approach)
            await anyio.sleep(0.6)
            await robot.emergency_stop()
        assert link.moves
        assert link.stopped
        with pytest.raises(RobotError, match="emergency stop"):
            await robot.approach_person(search_timeout=1)


async def test_a_detector_that_fails_to_load_means_walking_blind() -> None:
    link = MockLink(NIMBLE, person=(1.0, 0.0))
    async with _backend(link, MockDetector(fail_load=True)) as robot:
        assert await robot.wait_ready(5)
        assert not robot.can_find_person
        with pytest.raises(RobotError, match="cannot look"):
            await robot.approach_person(search_timeout=1)


class Blind:
    """A DuckLink with no camera."""

    def __init__(self) -> None:
        self._link = MockLink(NIMBLE)
        self.gait, self.bounds = self._link.gait, None

    async def open(self) -> None: ...
    async def move(self, vx: float, vy: float, vyaw: float) -> None: ...
    async def stop(self) -> None: ...
    async def quack(self) -> None: ...
    async def show_text(self, text: str | None) -> None: ...
    async def close(self) -> None: ...

    async def pose(self) -> Pose:
        return Pose(0.0, 0.0, 0.0)


async def test_a_duck_without_eyes_cannot_find_people() -> None:
    detector = MockDetector()
    async with MicroduckBackend(Blind(), Settings(), detector) as robot:
        assert await robot.wait_ready(5)
        assert not robot.can_find_person
        assert not detector.loaded


async def test_walking_somewhere_puts_the_head_back() -> None:
    link = MockLink(NIMBLE, time_scale=SCALE, person=(1.2, 0.3))
    async with _backend(link) as robot:
        assert await robot.wait_ready(5)
        with anyio.fail_after(30):
            assert await robot.approach_person(search_timeout=5)
            await robot.go_to(-0.3, 0.0)
        assert link.head is None
