"""Video utility: frame extraction from video files using OpenCV.

Example::

    from src.utils.video import extract_frames

    paths = extract_frames("data/raw/sample.mp4", "data/frames/sample", interval=1)
    print(f"Extracted {len(paths)} frames")
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval: float = 1.0,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract frames from a video file at a given time interval.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_dir:
        Directory where extracted frames (PNG) will be saved.
        Created automatically if it does not exist.
    interval:
        Time interval in seconds between extracted frames.
        Use ``1`` for one frame per second, ``0.5`` for two per second, etc.
    max_frames:
        Maximum number of frames to extract.  ``None`` means no limit.

    Returns
    -------
    list[Path]
        Sorted list of paths to the extracted frame images.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    ValueError
        If *interval* is not positive.
    RuntimeError
        If OpenCV cannot open the video file.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, int(round(fps * interval)))

    saved_paths: list[Path] = []
    frame_idx = 0
    saved_count = 0

    logger.info(
        "Extracting frames from '%s' (fps=%.2f, interval=%.2fs, skip=%d)",
        video_path,
        fps,
        interval,
        frame_skip,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            out_path = output_dir / f"frame_{saved_count:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
            saved_count += 1
            logger.debug("Saved frame %d → %s", frame_idx, out_path)

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames to '%s'", len(saved_paths), output_dir)
    return sorted(saved_paths)


def load_frame(frame_path: str | Path) -> np.ndarray:
    """Load a single frame image as a BGR numpy array.

    Parameters
    ----------
    frame_path:
        Path to a PNG / JPEG frame image.

    Returns
    -------
    np.ndarray
        BGR image array with dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If *frame_path* does not exist.
    RuntimeError
        If OpenCV cannot read the image.
    """
    frame_path = Path(frame_path)
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {frame_path}")
    return img
