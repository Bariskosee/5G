"""FTR Docker entry point for the current inference skeleton."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.detection.plate_detector import PlateDetector, crop_plate, select_best_detection
from src.ocr.plate_reader import PlateReader
from src.output.result_builder import build_result, write_results_json
from src.output.schema import PLATE_FALLBACK
from src.utils.plate_normalizer import choose_best_plate
from src.utils.video_reader import get_video_id, iter_video_frames

DEFAULT_INPUT = "/app/data/input/video.mp4"
DEFAULT_OUTPUT = "/app/data/output/results.json"
DEFAULT_PLATE_MODEL = "/app/models/model_b_plate/best.pt"

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output results.json path")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL, help="YOLO plate model path")
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max sampled frames")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Plate confidence threshold")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR and use fallback plate")
    return parser


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.plate_model)

    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        return 1
    if not model_path.exists():
        logger.error("Plate model not found: %s", model_path)
        return 1
    if args.frame_stride <= 0:
        logger.error("--frame-stride must be positive")
        return 1
    if args.max_frames is not None and args.max_frames < 0:
        logger.error("--max-frames must be non-negative")
        return 1

    video_id = get_video_id(input_path)
    plate_candidates: list[tuple[str, float]] = []
    sampled_frames = 0
    detected_plates = 0

    try:
        detector = PlateDetector(
            model_path=model_path,
            device="auto",
            conf_threshold=args.conf_threshold,
        )
    except Exception as exc:
        logger.error("Could not initialize plate detector: %s", exc)
        return 1

    reader = PlateReader(languages=["en"], enabled=not args.disable_ocr)

    try:
        for frame_index, time_seconds, frame in iter_video_frames(
            input_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        ):
            sampled_frames += 1
            try:
                detections = detector.detect(frame)
                best_detection = select_best_detection(detections)
                if best_detection is None:
                    continue

                detected_plates += 1
                plate_crop = crop_plate(frame, best_detection)
                for raw_text, ocr_confidence in reader.read_plate(plate_crop):
                    combined_confidence = best_detection.confidence * ocr_confidence
                    plate_candidates.append((raw_text, combined_confidence))
            except Exception as exc:
                logger.warning(
                    "Frame %s at %.2fs failed; continuing. Details: %s",
                    frame_index,
                    time_seconds,
                    exc,
                )
    except Exception as exc:
        logger.warning("Video processing failed; writing fallback JSON. Details: %s", exc)

    plate, plate_confidence = choose_best_plate(plate_candidates)
    if plate is None:
        logger.warning(
            "No valid OCR plate found; using temporary fallback '%s' with confidence 0.01",
            PLATE_FALLBACK,
        )
        plate = PLATE_FALLBACK
        plate_confidence = 0.01

    output_data = build_result(
        video_id=video_id,
        vehicle_type="sedan",
        plate=plate,
        color="beyaz",
        vehicle_confidence=plate_confidence,
        events=[],
    )
    write_results_json(output_data, output_path)

    logger.info("Sampled frames: %d", sampled_frames)
    logger.info("Frames with plate detections: %d", detected_plates)
    logger.info("Wrote FTR results: %s", output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
