"""Visualization utilities: draw bounding boxes and labels on frames.

Example::

    from src.utils.visualization import draw_detections, draw_behaviors

    annotated = draw_detections(frame, detections)
    annotated = draw_behaviors(annotated, behaviors)
    cv2.imshow("result", annotated)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color palette (BGR) — one color per class label (cycles if > palette size)
_PALETTE: list[tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 0, 255),  # purple
    (0, 128, 255),  # orange
]

_BEHAVIOR_COLOR: tuple[int, int, int] = (0, 0, 255)  # red for behaviors


def _get_color(label: str) -> tuple[int, int, int]:
    """Return a consistent color for *label*."""
    return _PALETTE[hash(label) % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw vehicle detection bounding boxes and labels on a frame.

    Parameters
    ----------
    frame:
        BGR image array (will *not* be modified in-place — a copy is made).
    detections:
        List of detection dicts, each with keys
        ``{"bbox": [x1,y1,x2,y2], "class": str, "confidence": float}``.
    thickness:
        Box border thickness in pixels.
    font_scale:
        OpenCV font scale for the label text.

    Returns
    -------
    np.ndarray
        Annotated BGR frame copy.
    """
    output = frame.copy()

    for det in detections:
        bbox = det.get("bbox")
        label = det.get("class", "vehicle")
        conf = det.get("confidence", 0.0)

        if bbox is None or len(bbox) != 4:
            logger.warning("draw_detections: skipping detection with invalid bbox")
            continue

        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = _get_color(label)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Background rectangle for readability
        cv2.rectangle(
            output, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1
        )
        cv2.putText(
            output,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def draw_behaviors(
    frame: np.ndarray,
    behaviors: list[dict],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw driver behavior warnings on a frame.

    Parameters
    ----------
    frame:
        BGR image array (a copy is made).
    behaviors:
        List of behavior dicts, each with keys
        ``{"behavior": str, "confidence": float, "bbox": list | None}``.
    thickness, font_scale:
        Same as :func:`draw_detections`.

    Returns
    -------
    np.ndarray
        Annotated BGR frame copy.
    """
    output = frame.copy()

    for idx, beh in enumerate(behaviors):
        label = beh.get("behavior", "unknown")
        conf = beh.get("confidence", 0.0)
        bbox = beh.get("bbox")
        text = f"[!] {label} {conf:.2f}"

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), _BEHAVIOR_COLOR, thickness)
            cv2.putText(
                output,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                _BEHAVIOR_COLOR,
                thickness,
                cv2.LINE_AA,
            )
        else:
            # No bbox — display as an overlay in the top-left corner
            y_pos = 30 + idx * 30
            cv2.putText(
                output,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                _BEHAVIOR_COLOR,
                thickness,
                cv2.LINE_AA,
            )

    return output
