"""MediaPipe head yaw estimator → arkaya_bakma / etrafa_bakinma detection."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices for yaw estimation
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263


def _estimate_yaw_degrees(landmarks: list) -> float | None:
    """Estimate head yaw from outer eye corner landmarks.

    Returns approximate yaw in degrees [-180, 180].
    Negative = turned left, positive = turned right.
    Returns None if landmarks are unavailable or malformed.
    """
    try:
        left = landmarks[_LEFT_EYE_OUTER]
        right = landmarks[_RIGHT_EYE_OUTER]
        dx = right.x - left.x
        dy = right.y - left.y
        yaw = math.degrees(math.atan2(dy, dx))
        return yaw
    except (IndexError, AttributeError) as exc:
        logger.debug("head_pose: yaw estimation failed (%s)", exc)
        return None


class HeadPoseDetector:
    """Detect arkaya_bakma and etrafa_bakinma from MediaPipe face landmarks.

    arkaya_bakma: head yaw exceeds threshold for a sustained number of frames.
    etrafa_bakinma: head oscillates left↔right ≥ min_oscillations times
                    within a rolling window.
    """

    def __init__(
        self,
        yaw_back_threshold: float = 75.0,
        arkaya_min_frames: int = 4,
        yaw_oscillation_threshold: float = 30.0,
        window_frames: int = 18,
        min_oscillations: int = 2,
        cooldown_seconds: float = 3.0,
    ) -> None:
        self._yaw_back_threshold = yaw_back_threshold
        self._arkaya_min_frames = arkaya_min_frames
        self._yaw_osc_threshold = yaw_oscillation_threshold
        self._window = deque(maxlen=window_frames)
        self._min_oscillations = min_oscillations
        self._cooldown = cooldown_seconds

        self._back_frames: int = 0
        self._last_emitted_arkaya: float = -999.0
        self._last_emitted_etrafa: float = -999.0

    def update(self, face_result: Any, time_seconds: float) -> list[dict]:
        """Update state with the latest FaceLandmarkerResult.

        Returns a list of tespitler-compatible event dicts (may be empty).
        """
        events: list[dict] = []

        if face_result is None or not face_result.face_landmarks:
            self._back_frames = 0
            self._window.append(None)
            return events

        yaw = _estimate_yaw_degrees(face_result.face_landmarks[0])
        if yaw is None:
            self._back_frames = 0
            self._window.append(None)
            return events

        self._window.append(yaw)

        # --- arkaya_bakma ---
        if abs(yaw) > self._yaw_back_threshold:
            self._back_frames += 1
            if (
                self._back_frames >= self._arkaya_min_frames
                and time_seconds - self._last_emitted_arkaya > self._cooldown
            ):
                self._last_emitted_arkaya = time_seconds
                self._back_frames = 0
                conf = round(min(0.95, 0.55 + (abs(yaw) - self._yaw_back_threshold) / 90.0), 2)
                events.append({
                    "zaman_saniye": round(time_seconds, 2),
                    "kategori": "sofor_eylemi",
                    "etiket": "arkaya_bakma",
                    "confidence_score": conf,
                })
        else:
            self._back_frames = 0

        # --- etrafa_bakinma ---
        if time_seconds - self._last_emitted_etrafa > self._cooldown:
            valid_yaws = [y for y in self._window if y is not None]
            if len(valid_yaws) >= self._window.maxlen // 2:
                oscillations = _count_oscillations(valid_yaws, self._yaw_osc_threshold)
                if oscillations >= self._min_oscillations:
                    self._last_emitted_etrafa = time_seconds
                    events.append({
                        "zaman_saniye": round(time_seconds, 2),
                        "kategori": "sofor_eylemi",
                        "etiket": "etrafa_bakinma",
                        "confidence_score": round(min(0.90, 0.50 + oscillations * 0.12), 2),
                    })

        return events


def _count_oscillations(yaws: list[float], threshold: float) -> int:
    """Count left↔right direction changes exceeding threshold amplitude."""
    if len(yaws) < 3:
        return 0

    direction: int | None = None
    changes = 0
    peak = yaws[0]

    for y in yaws[1:]:
        if direction is None:
            if y - peak > threshold:
                direction = 1
                peak = y
            elif peak - y > threshold:
                direction = -1
                peak = y
        else:
            if direction == 1 and peak - y > threshold:
                direction = -1
                peak = y
                changes += 1
            elif direction == -1 and y - peak > threshold:
                direction = 1
                peak = y
                changes += 1
            elif direction == 1 and y > peak:
                peak = y
            elif direction == -1 and y < peak:
                peak = y

    return changes
