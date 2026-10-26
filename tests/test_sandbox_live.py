"""Against the real simulator website, in a headless browser.

Opt-in (needs a network and a Chromium-family browser): QUACKBACK_TEST_SANDBOX=1 uv run pytest
tests/test_sandbox_live.py
"""

import os

import anyio
import pytest

from quackback.config import Settings
from quackback.robot.base import HUMAN_STOP_DISTANCE
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
