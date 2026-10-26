import base64
import math
from pathlib import Path

import numpy as np
import pytest

from quackback.robot.base import RobotError
from quackback.robot.sandbox import (
    _EYES,
    DEFAULT_PERSON_IMAGE,
    GAIT,
    browser_channels,
    decode_jpeg,
    head_command,
    image_data_url,
    pose_from_qpos,
    with_boot_flag,
)


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
    assert GAIT.turn_radius == pytest.approx(0.17)  # 33 degrees/s at about 0.1 m/s


def test_head_commands_follow_the_measured_table() -> None:
    assert head_command(None) == [0.0, 0.0, 0.0, 0.0]
    assert head_command(math.radians(30.4)) == pytest.approx([0.0, -1.2, 0.0, 0.0])
    assert head_command(math.radians(17.4))[1] == pytest.approx(-0.75)  # between two rows
    assert head_command(math.radians(80))[1] == pytest.approx(-2.5)  # as far up as it goes
    assert head_command(math.radians(-40))[1] == 0.0  # it does not look down from rest
    assert head_command(0.0, math.radians(28))[2] == pytest.approx(0.5)
    assert head_command(0.0, -10.0)[2] == -2.5  # the page's command range


def test_the_eyes_script_and_the_person_ship_with_the_package() -> None:
    assert _EYES.lstrip().startswith("async (standIn) =>")
    assert DEFAULT_PERSON_IMAGE.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_person_pictures_become_data_urls(tmp_path: Path) -> None:
    jpeg = tmp_path / "me.JPG"
    jpeg.write_bytes(b"\xff\xd8 pretend")
    url = image_data_url(jpeg)
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8 pretend"
    assert image_data_url(DEFAULT_PERSON_IMAGE).startswith("data:image/png;base64,")
    with pytest.raises(RobotError, match="cannot read"):
        image_data_url(tmp_path / "missing.png")


def test_camera_pictures_are_decoded_to_rgb() -> None:
    cv2 = pytest.importorskip("cv2")
    bgr = np.zeros((64, 36, 3), np.uint8)
    bgr[:, :, 2] = 255  # red, in OpenCV's order
    ok, jpeg = cv2.imencode(".jpg", bgr)
    assert ok
    rgb = decode_jpeg(jpeg.tobytes())
    assert rgb.shape == (64, 36, 3)
    assert rgb[..., 0].mean() > 240  # red first
    assert rgb[..., 2].mean() < 15
    with pytest.raises(RobotError, match="not a JPEG"):
        decode_jpeg(b"nope")
