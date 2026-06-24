"""HSV + Lab vehicle color classifier — no training required."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Hue ranges in OpenCV [0, 180] mapped to competition color labels
_HUE_RANGES: list[tuple[tuple[int, int], str]] = [
    ((0, 10), "kirmizi"),
    ((166, 180), "kirmizi"),
    ((11, 25), "turuncu"),
    ((26, 34), "sari"),
    ((35, 85), "yesil"),
    ((86, 130), "mavi"),
    ((131, 165), "kahverengi"),
]

_DEFAULT_COLOR = "beyaz"


def classify_color(
    frame: np.ndarray,
    bbox: Sequence[int],
    *,
    sample_inset: float = 0.25,
    low_sat_threshold: int = 30,
    lab_white_min: int = 200,
    lab_black_max: int = 60,
) -> tuple[str, float]:
    """Classify vehicle color from a BGR frame region.

    Returns (color_label, confidence) where confidence is in [0.0, 1.0].
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return _DEFAULT_COLOR, 0.0

    dx = max(1, int(w * sample_inset))
    dy = max(1, int(h * sample_inset))
    crop = frame[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    if crop.size == 0:
        return _DEFAULT_COLOR, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))

    if mean_sat < low_sat_threshold:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        mean_l = float(np.mean(lab[:, :, 0]))
        if mean_l >= lab_white_min:
            return "beyaz", 0.75
        if mean_l <= lab_black_max:
            return "siyah", 0.75
        return "gri", 0.70

    hue_flat = hsv[:, :, 0].flatten()
    sat_flat = hsv[:, :, 1].flatten().astype(float)
    votes: dict[str, float] = {}
    for h_val, s_val in zip(hue_flat, sat_flat):
        label = _hue_to_label(int(h_val))
        votes[label] = votes.get(label, 0.0) + s_val

    if not votes:
        return _DEFAULT_COLOR, 0.20

    best = max(votes, key=lambda k: votes[k])
    total = sum(votes.values())
    confidence = min(0.95, votes[best] / total + 0.25) if total > 0 else 0.30
    return best, confidence


def _hue_to_label(hue: int) -> str:
    for (lo, hi), label in _HUE_RANGES:
        if lo <= hue <= hi:
            return label
    return "kahverengi"
