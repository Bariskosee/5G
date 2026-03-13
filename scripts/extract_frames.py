"""CLI script: extract frames from a video file.

Usage::

    python scripts/extract_frames.py \\
        --video data/raw/sample.mp4 \\
        --output data/frames/sample \\
        --interval 1 \\
        [--max-frames 500]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.video import extract_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file at a fixed time interval."
    )
    parser.add_argument(
        "--video", "-v", required=True, type=Path, help="Path to input video file."
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Directory for extracted frames."
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Time interval in seconds between extracted frames (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        paths = extract_frames(
            video_path=args.video,
            output_dir=args.output,
            interval=args.interval,
            max_frames=args.max_frames,
        )
        print(f"Extracted {len(paths)} frames to '{args.output}'")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("Error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
