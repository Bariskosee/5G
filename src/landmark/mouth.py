"""Mouth aspect ratio (MAR) → esneme (yawning) stateful detector."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MediaPipe face mesh indices used for MAR
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_CORNER = 61
_RIGHT_CORNER = 291


def mouth_aspect_ratio(landmarks: list) -> float:
    """Compute MAR from a MediaPipe face landmark list (normalized coords).

    MAR = vertical_opening / horizontal_width. Higher → more open mouth.
    """
    try:
        vertical = abs(landmarks[_LOWER_LIP].y - landmarks[_UPPER_LIP].y)
        horizontal = abs(landmarks[_RIGHT_CORNER].x - landmarks[_LEFT_CORNER].x) + 1e-6
        return vertical / horizontal
    except (IndexError, AttributeError) as exc:
        logger.debug("mouth_aspect_ratio: failed (%s)", exc)
        return 0.0


class EsnemeDetector:
    """Emit one esneme event per sustained open-mouth sequence.

    Tracks how many consecutive frames have MAR above threshold and
    emits an event dict (matching the competition tespitler schema) once
    the sequence reaches min_duration_frames. A cooldown prevents repeated
    events from the same yawn.
    """

    def __init__(
        self,
        mar_threshold: float = 0.60,
        min_duration_frames: int = 8,
        cooldown_seconds: float = 3.0,
    ) -> None:
        self._mar_threshold = mar_threshold
        self._min_duration_frames = min_duration_frames
        self._cooldown_seconds = cooldown_seconds
        self._open_frames: int = 0
        self._last_emitted_at: float = -999.0

    def update(self, face_result: Any, time_seconds: float) -> dict | None:
        """Update state with the latest FaceLandmarkerResult.

        Returns a tespitler-compatible event dict on detection, else None.
        """
        if face_result is None or not face_result.face_landmarks:
            self._open_frames = 0
            return None

        mar = mouth_aspect_ratio(face_result.face_landmarks[0])

        if mar > self._mar_threshold:
            self._open_frames += 1
            if (
                self._open_frames >= self._min_duration_frames
                and time_seconds - self._last_emitted_at > self._cooldown_seconds
            ):
                self._last_emitted_at = time_seconds
                self._open_frames = 0
                return {
                    "zaman_saniye": round(time_seconds, 2),
                    "kategori": "sofor_eylemi",
                    "etiket": "esneme",
                    "confidence_score": round(min(0.95, mar * 0.85), 2),
                }
        else:
            self._open_frames = 0

        return None
