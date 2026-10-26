import math

import pytest

from quackback.robot.depth import (
    NO_TARGET,
    DepthFrame,
    column_bearing,
    nearest,
    row_elevation,
    zone_direction,
)


def _frame(fill: int = 0, status: int = NO_TARGET) -> list[list[tuple[int, int]]]:
    return [[(fill, status) for _ in range(8)] for _ in range(8)]


def _pack(grid: list[list[tuple[int, int]]], pitch: float = 0.0) -> DepthFrame:
    """A frame of upright surfaces at the grid's level distances, as the sensor measures them."""
    distance_mm, status = [], []
    for row, zones in enumerate(grid):
        stretch = 1 / math.cos(row_elevation(row) + pitch)
        for mm, s in zones:
            distance_mm.append(round(mm * stretch))
            status.append(s)
    return DepthFrame(8, 8, tuple(distance_mm), tuple(status), pitch)


def test_geometry() -> None:
    assert column_bearing(0) == pytest.approx(math.radians(22.5 - 2.8125))
    assert column_bearing(7) == pytest.approx(-math.radians(22.5 - 2.8125))
    assert row_elevation(0) > 0 > row_elevation(7)


def test_nothing_in_view() -> None:
    assert nearest(_pack(_frame())) is None


def test_a_wall_ahead() -> None:
    assert nearest(_pack(_frame(1200, 5))) == pytest.approx(1.2, abs=0.001)


def test_one_flier_does_not_decide() -> None:
    grid = _frame()
    grid[2][3] = (300, 5)
    assert nearest(_pack(grid)) is None  # a single zone is not enough
    grid[2][4] = (450, 5)
    assert nearest(_pack(grid)) == pytest.approx(0.45, abs=0.001)  # the second closest


def test_crosstalk_and_untrusted_statuses_are_ignored() -> None:
    grid = _frame(2000, 5)
    for col in range(8):
        grid[1][col] = (60, 5)  # under 10 cm: the cover glass
        grid[2][col] = (300, 3)  # status 3: not believed
    assert nearest(_pack(grid)) == pytest.approx(2.0, abs=0.001)


def test_moving_target_statuses_count() -> None:
    grid = _frame()
    grid[3][3] = (500, 12)  # a target blurred by an edge
    grid[3][4] = (520, 4)  # a consistency failure of a moving target
    assert nearest(_pack(grid)) == pytest.approx(0.52, abs=0.001)


def test_floor_rows_are_ignored() -> None:
    grid = _frame()
    grid[7] = [(350, 5)] * 8  # the floor, just ahead of the feet
    assert nearest(_pack(grid)) is None


def test_bearing_window() -> None:
    grid = _frame()
    grid[2][0] = grid[3][0] = (400, 5)  # something far left
    grid[2][4] = grid[3][4] = (900, 5)  # the person, just right of centre
    assert nearest(_pack(grid)) == pytest.approx(0.4, abs=0.001)
    person = column_bearing(4)
    assert nearest(_pack(grid), left=person, right=person) == pytest.approx(0.9, abs=0.001)


def test_looking_up_turns_floor_rows_into_a_view_ahead() -> None:
    grid = _frame()
    grid[7] = [(500, 5)] * 8  # the lowest row
    assert nearest(_pack(grid)) is None  # at rest it looks 20 degrees down: floor
    assert nearest(_pack(grid, pitch=math.radians(30))) == pytest.approx(0.5, abs=0.001)


def test_slant_ranges_are_made_level() -> None:
    zones = 64
    frame = DepthFrame(8, 8, (1000,) * zones, (5,) * zones, pitch=math.radians(30))
    # The top zones look most steeply up (50 degrees): the same range is nearest on the level.
    expected = math.cos(row_elevation(0) + math.radians(30))
    assert nearest(frame) == pytest.approx(expected, abs=0.002)


def test_zone_directions_follow_the_pitch() -> None:
    level = _pack(_frame())
    assert zone_direction(level, 2, 1) == pytest.approx((column_bearing(1), row_elevation(2)))
    up = _pack(_frame(), pitch=math.radians(30))
    bearing, elevation = zone_direction(up, 3, 3)
    assert elevation == pytest.approx(row_elevation(3) + math.radians(30), abs=0.002)
    # Looking up, the side columns spread wider: a pitched frustum is wider at the top.
    assert bearing > column_bearing(3)
    assert zone_direction(up, 0, 0)[0] > zone_direction(up, 7, 0)[0]
