"""Video reading and frame sampling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def get_video_id(video_path: str | Path) -> str:
    """Return the video basename used in FTR output."""
    return Path(video_path).name


def iter_video_frames(
    video_path: str | Path,
    frame_stride: int = 10,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield sampled frames as (frame_index, time_seconds, frame)."""
    video_path = Path(video_path)
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    yielded = 0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                yield frame_index, frame_index / fps, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()
