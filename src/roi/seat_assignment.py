"""Assign COCO person bounding boxes to competition yolcular seat labels."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Normalized y-boundary between front and rear seats
_FRONT_REAR_Y_SPLIT = 0.55
# Normalized x-boundary between left (driver side) and right (passenger side)
_LEFT_RIGHT_X_SPLIT = 0.50


class SeatAssigner:
    """Map detected person bboxes to competition yolcular labels.

    Rules (LHD vehicle — Turkey, driver on the left):
    - Person whose center is inside the driver ROI → skip (that's the driver).
    - x_center > 0.50 AND y_center < 0.55 → on_koltuk  (front passenger)
    - x_center < 0.50 AND y_center > 0.55 → arka_koltuk_1  (rear left)
    - x_center > 0.50 AND y_center > 0.55 → arka_koltuk_2  (rear right)

    Each label is emitted at most once per video (tracked via seen_seat_labels).
    Confidence is fixed at 0.75 (rule-based assignment, no per-frame score).
    """

    def assign(
        self,
        person_bboxes: list[tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
        time_seconds: float,
        seen_seat_labels: set[str],
        driver_roi: dict,
    ) -> list[dict]:
        """Return new yolcular events for unseen seats only."""
        if frame_w <= 0 or frame_h <= 0:
            return []

        roi_x_min = float(driver_roi.get("x_min", 0.20))
        roi_y_min = float(driver_roi.get("y_min", 0.10))
        roi_x_max = float(driver_roi.get("x_max", 0.75))
        roi_y_max = float(driver_roi.get("y_max", 0.85))

        events: list[dict] = []

        for x1, y1, x2, y2 in person_bboxes:
            cx = ((x1 + x2) / 2.0) / frame_w
            cy = ((y1 + y2) / 2.0) / frame_h

            # Skip if this is likely the driver
            if roi_x_min <= cx <= roi_x_max and roi_y_min <= cy <= roi_y_max:
                logger.debug("seat_assigner: person at cx=%.2f cy=%.2f → driver, skipping", cx, cy)
                continue

            label = _classify_seat(cx, cy)
            if label is None or label in seen_seat_labels:
                continue

            logger.info(
                "[t=%.2fs] yolcular/%s cx=%.2f cy=%.2f",
                time_seconds, label, cx, cy,
            )
            events.append({
                "zaman_saniye": round(time_seconds, 2),
                "kategori": "yolcular",
                "etiket": label,
                "confidence_score": 0.75,
            })

        return events


def _classify_seat(cx: float, cy: float) -> str | None:
    """Return seat label based on normalized bbox center, or None if unclassifiable."""
    if cy < _FRONT_REAR_Y_SPLIT:
        if cx > _LEFT_RIGHT_X_SPLIT:
            return "on_koltuk"
        return None  # front-left → driver region, already skipped above
    else:
        if cx <= _LEFT_RIGHT_X_SPLIT:
            return "arka_koltuk_1"
        return "arka_koltuk_2"
