"""Driver seat ROI checker — determines whether a detected object is near the driver."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DriverProximityChecker:
    """Check whether a bounding box center falls within the driver seat ROI.

    ROI is defined as normalized [0, 1] frame coordinates in thresholds.yaml
    under the driver_seat_roi key. Default values assume a left-hand-drive
    vehicle (Turkey) with the driver on the left side of the frame.
    """

    def __init__(self, roi_config: dict) -> None:
        self._x_min: float = float(roi_config.get("x_min", 0.20))
        self._y_min: float = float(roi_config.get("y_min", 0.10))
        self._x_max: float = float(roi_config.get("x_max", 0.75))
        self._y_max: float = float(roi_config.get("y_max", 0.85))

    def is_in_driver_roi(
        self,
        xyxy: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Return True if the bbox center (normalized) is inside the driver ROI."""
        if frame_w <= 0 or frame_h <= 0:
            return False
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2.0) / frame_w
        cy = ((y1 + y2) / 2.0) / frame_h
        return (
            self._x_min <= cx <= self._x_max
            and self._y_min <= cy <= self._y_max
        )
