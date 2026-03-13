"""CLI script: run the full detection pipeline on a video file.

Usage::

    python scripts/run_detection.py \\
        --video data/raw/sample.mp4 \\
        [--config configs/default.yaml] \\
        [--output outputs] \\
        [--no-annotated]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the road-safety detection pipeline on a video file."
    )
    parser.add_argument(
        "--video", "-v", required=True, type=Path, help="Path to input video file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs).",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip saving annotated frame images.",
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

    pipeline = Pipeline(config_path=args.config, output_dir=args.output)
    try:
        results = pipeline.run(
            video_path=args.video,
            save_annotated=not args.no_annotated,
        )
        print(results["summary"])
        print(f"Log saved to: {results['log_path']}")
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
