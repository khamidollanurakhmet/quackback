"""Against the real simulator website, in a headless browser.

Opt-in (needs a network and a Chromium-family browser): QUACKBACK_TEST_SANDBOX=1 uv run pytest
tests/test_sandbox_live.py
"""

import contextlib
import math
import os

import anyio
import pytest

from quackback.config import Settings
from quackback.robot.base import HUMAN_STOP_DISTANCE, RobotError
from quackback.robot.microduck import ARRIVE_RADIUS, MicroduckBackend
from quackback.robot.sandbox import SandboxLink

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.sandbox,
    pytest.mark.skipif(
        os.environ.get("QUACKBACK_TEST_SANDBOX") != "1", reason="set QUACKBACK_TEST_SANDBOX=1"
    ),
]


async def test_walks_out_and_back_in_the_simulator() -> None:
    settings = Settings(robot="sandbox", sandbox_headless=True)
    link = SandboxLink(settings.sandbox_url, headless=True)
    async with MicroduckBackend(link, settings) as robot:
        assert await robot.wait_ready(240), robot.error
        await robot.show_text("Which database should the cache use?")
        with anyio.fail_after(90):
            await robot.go_to(1.0, 0.0)
        assert (await robot.position()).distance_to(1.0, 0.0) <= ARRIVE_RADIUS + 0.05
        await robot.cue()
        with anyio.fail_after(90):
            await robot.go_to(0.0, 0.0)
        assert (await robot.position()).distance_to(0.0, 0.0) <= ARRIVE_RADIUS + 0.05
        with anyio.fail_after(90):
            await robot.go_to(0.6, 0.6)
        assert (await robot.position()).distance_to(0.6, 0.6) <= ARRIVE_RADIUS + 0.05
        # Toward a human 1.5 m ahead of the start: never closer than 40 cm.
        with anyio.fail_after(90):
            await robot.go_to(1.5, 0.0, stop_short=HUMAN_STOP_DISTANCE)
        await anyio.sleep(1.0)  # let it settle after the stop
        distance = (await robot.position()).distance_to(1.5, 0.0)
        assert HUMAN_STOP_DISTANCE <= distance <= HUMAN_STOP_DISTANCE + 0.2


async def test_finds_faces_and_leaves_the_stand_in_person() -> None:
    pytest.importorskip("mediapipe")
    from quackback.robot.base import Pose
    from quackback.robot.microduck import TOF_AHEAD, wrap_angle
    from quackback.vision.person_detector import PersonDetector

    person = (1.3, -0.7)  # off to the right, out of view from the start: it has to look around
    settings = Settings(robot="sandbox", sandbox_headless=True)
    link = SandboxLink(settings.sandbox_url, headless=True, person=person)
    async with MicroduckBackend(link, settings, PersonDetector(settings)) as robot:
        assert await robot.wait_ready(240), robot.error
        assert robot.can_find_person
        closest = math.inf

        async def watch() -> None:
            # The duck's nose (its depth sensor) is TOF_AHEAD ahead of the body's centre.
            nonlocal closest
            while True:
                with contextlib.suppress(RobotError):  # the page reloading under it
                    pose = await robot.position()
                    closest = min(closest, pose.distance_to(*person) - TOF_AHEAD)
                await anyio.sleep(0.05)

        async with anyio.create_task_group() as tg:
            tg.start_soon(watch)
            with anyio.fail_after(150):
                assert await robot.approach_person(search_timeout=60)
            await anyio.sleep(1.5)  # let the head settle
            pose = await robot.position()
            assert pose.distance_to(*person) - TOF_AHEAD <= 0.8  # it walked up to them
            camera = robot.to_base(Pose(0.0, 0.0, (await link.camera_frame()).heading))
            toward = math.atan2(person[1] - pose.y, person[0] - pose.x)
            assert abs(wrap_angle(camera.yaw - toward)) < math.radians(12)
            with anyio.fail_after(150):
                await robot.go_to(0.0, 0.0)  # and home, turning away from them
            await anyio.sleep(1.0)  # it coasts after the stop
            # Someone reloads the page: the duck starts over at its spawn point, and quackback
            # installs itself again, the stand-in person included.
            await link._page.reload()
            with anyio.fail_after(240):
                assert await robot.approach_person(search_timeout=60)
            tg.cancel_scope.cancel()
        assert closest >= HUMAN_STOP_DISTANCE
