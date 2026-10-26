import math

import pytest

from quackback.robot.sandbox import GAIT, browser_channels, pose_from_qpos, with_boot_flag


def test_boot_flag_is_added_once() -> None:
    url = "https://pollen-robotics-microduck-simulator.hf.space/"
    assert with_boot_flag(url) == url + "?boot=1"
    assert with_boot_flag(url + "?lite=1") == url + "?lite=1&boot=1"
    assert with_boot_flag(url + "?boot=1") == url + "?boot=1"


def test_pose_from_free_joint() -> None:
    half = math.pi / 4  # a quarter turn left about z
    pose = pose_from_qpos([0.5, -0.2, 0.12, math.cos(half), 0.0, 0.0, math.sin(half)])
    assert (pose.x, pose.y) == (0.5, -0.2)
    assert pose.yaw == pytest.approx(math.pi / 2)


def test_browser_order() -> None:
    assert browser_channels("", platform="win32") == ["msedge", None]
    assert browser_channels("", platform="linux") == ["chrome", None]
    assert browser_channels("chrome", platform="win32") == ["chrome"]


def test_gait_matches_the_measured_policy() -> None:
    assert GAIT.min_speed == GAIT.max_speed == 0.25
    assert not GAIT.turns_in_place
