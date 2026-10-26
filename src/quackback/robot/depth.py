"""The duck's 8x8 time-of-flight depth sensor, and how far away the thing in front of it is.

The frame is what robotd's tofd publishes (`tof.frame`): 8x8 zones over 45 x 45 degrees, row 0 at
the top and column 0 on the left, a slant range in millimetres and an ST status byte per zone.
The rules for believing it come from upstream's theremin (kinematics/src/hand.rs): the status
set a moving target needs (not only ST's two "valid" codes, which lose a hand at about 30 cm),
nothing under 10 cm (the cover glass's crosstalk), and at least two zones agreeing, because
the single nearest zone of this sensor is regularly a flier.

The sensor sits in the head, so where it points depends on the head: `heading` and `pitch` say
where it pointed for this frame. Distances are made level with them, and zones looking down at
the floor are left out.
"""

import math
from dataclasses import dataclass

FOV = math.radians(45.0)
# ST status bytes whose distance upstream believes for a moving target.
BELIEVED_STATUSES = frozenset({4, 5, 6, 9, 10, 12, 13})
NO_TARGET = 255
NEAREST_BELIEVABLE_M = 0.10
MIN_ZONES = 2
# Zones looking further down than this see the floor a few decimetres ahead, not a person.
FLOOR_BELOW = math.radians(-10.0)


@dataclass(frozen=True)
class DepthFrame:
    rows: int
    cols: int
    distance_mm: tuple[int, ...]  # row-major, row 0 at the top, column 0 on the left
    status: tuple[int, ...]
    pitch: float = 0.0  # the sensor's angle above level, radians
    heading: float = 0.0  # where it points in the duck's world, radians, counter-clockwise

    def zone(self, row: int, col: int) -> tuple[int, int]:
        i = row * self.cols + col
        return self.distance_mm[i], self.status[i]


def column_bearing(col: int, cols: int = 8) -> float:
    """Horizontal angle of a column's centre relative to the sensor, radians, positive left."""
    return FOV / 2 - (col + 0.5) * FOV / cols


def row_elevation(row: int, rows: int = 8) -> float:
    """Vertical angle of a row's centre relative to the sensor, radians, positive up."""
    return FOV / 2 - (row + 0.5) * FOV / rows


def zone_direction(frame: DepthFrame, row: int, col: int) -> tuple[float, float]:
    """(bearing, elevation) of a zone's centre once the sensor's pitch is taken into account:
    the bearing relative to the sensor's heading, positive left, and the angle above level."""
    az, el = column_bearing(col, frame.cols), row_elevation(row, frame.rows)
    ahead, left, up = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    c, s = math.cos(frame.pitch), math.sin(frame.pitch)
    level_ahead, level_up = ahead * c - up * s, ahead * s + up * c
    return math.atan2(left, level_ahead), math.asin(max(-1.0, min(1.0, level_up)))


def nearest(frame: DepthFrame, left: float = FOV / 2, right: float = -FOV / 2) -> float | None:
    """Level distance in metres to the nearest thing between two bearings, or None.

    `left` and `right` are bearings relative to the sensor's heading in radians (positive to
    the left); a zone counts when its centre lies within half a zone of that span. The answer
    is the second-closest believable zone, so one flier never decides.
    """
    half_zone = FOV / frame.cols / 2
    distances = [
        level
        for bearing, level in _believed(frame)
        if right - half_zone <= bearing <= left + half_zone
    ]
    if len(distances) < MIN_ZONES:
        return None
    return sorted(distances)[MIN_ZONES - 1]


def points(frame: DepthFrame, x: float, y: float) -> list[tuple[float, float]]:
    """Where in the duck's world the believable zones found something, the sensor at (x, y)."""
    return [
        (
            x + level * math.cos(frame.heading + bearing),
            y + level * math.sin(frame.heading + bearing),
        )
        for bearing, level in _believed(frame)
    ]


def _believed(frame: DepthFrame) -> list[tuple[float, float]]:
    """(bearing, level distance in metres) of every zone believed, the floor left out."""
    found = []
    for row in range(frame.rows):
        for col in range(frame.cols):
            bearing, elevation = zone_direction(frame, row, col)
            if elevation < FLOOR_BELOW:
                continue
            mm, status = frame.zone(row, col)
            slant = mm / 1000
            if status in BELIEVED_STATUSES and slant >= NEAREST_BELIEVABLE_M:
                found.append((bearing, slant * math.cos(elevation)))
    return found
