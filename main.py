"""FTR Docker entry point for the AI inference pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.color.hsv_lab import classify_color
from src.detection.model_a import ModelADetector
from src.detection.plate_detector import PlateDetector, crop_plate, select_best_detection
from src.landmark.face import FaceLandmarker
from src.landmark.mouth import EsnemeDetector
from src.ocr.plate_reader import PlateReader
from src.output.result_builder import build_result, write_results_json
from src.output.schema import PLATE_FALLBACK
from src.utils.plate_normalizer import choose_best_plate
from src.utils.video_reader import get_video_id, iter_video_frames

DEFAULT_INPUT = "/app/data/input/video.mp4"
DEFAULT_OUTPUT = "/app/data/output/results.json"
DEFAULT_PLATE_MODEL = "/app/models/model_b_plate/best.pt"
DEFAULT_MEDIAPIPE_MODEL = "/app/models/mediapipe/face_landmarker.task"

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output results.json path")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL, help="YOLO plate model path")
    parser.add_argument(
        "--mediapipe-model",
        default=DEFAULT_MEDIAPIPE_MODEL,
        help="MediaPipe FaceLandmarker .task file path",
    )
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max sampled frames")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Plate confidence threshold")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR and use fallback plate")
    return parser


def _pick_best_label(votes: list[tuple[str, float]], default: str) -> tuple[str, float]:
    """Return the highest confidence-weighted label from per-frame votes."""
    if not votes:
        return default, 0.01
    totals: dict[str, float] = {}
    for label, conf in votes:
        totals[label] = totals.get(label, 0.0) + conf
    best = max(totals, key=lambda k: totals[k])
    avg_conf = min(0.95, totals[best] / len(votes))
    return best, avg_conf


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
    vehicle_type_votes: list[tuple[str, float]] = []
    color_votes: list[tuple[str, float]] = []
    events: list[dict[str, Any]] = []
    sampled_frames = 0
    detected_plates = 0

    try:
        plate_detector = PlateDetector(
            model_path=model_path,
            device="auto",
            conf_threshold=args.conf_threshold,
        )
    except Exception as exc:
        logger.error("Could not initialize plate detector: %s", exc)
        return 1

    reader = PlateReader(languages=["tr", "en"], enabled=not args.disable_ocr)
    vehicle_detector = ModelADetector(model_path=None, confidence=0.45, device="auto")
    landmarker = FaceLandmarker(model_path=args.mediapipe_model)
    esneme_detector = EsnemeDetector(mar_threshold=0.60, min_duration_frames=8)

    try:
        for frame_index, time_seconds, frame in iter_video_frames(
            input_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        ):
            sampled_frames += 1
            try:
                # Plate detection + OCR
                detections = plate_detector.detect(frame)
                best_plate = select_best_detection(detections)
                if best_plate is not None:
                    detected_plates += 1
                    plate_crop = crop_plate(frame, best_plate)
                    for raw_text, ocr_conf in reader.read_plate(plate_crop):
                        plate_candidates.append((raw_text, best_plate.confidence * ocr_conf))

                # Vehicle type (COCO YOLOv8m)
                vtype, vconf = vehicle_detector.detect_vehicle_type(frame)
                vehicle_type_votes.append((vtype, vconf))

                # Color — sample lower half of frame as vehicle proxy
                fh, fw = frame.shape[:2]
                vehicle_region = [int(fw * 0.05), int(fh * 0.25), int(fw * 0.95), int(fh * 0.95)]
                clr, clr_conf = classify_color(frame, vehicle_region, sample_inset=0.10)
                color_votes.append((clr, clr_conf))

                # Face landmarks → esneme
                face_result = landmarker.detect(frame)
                event = esneme_detector.update(face_result, time_seconds)
                if event is not None:
                    events.append(event)
                    logger.info(
                        "[t=%.2fs] sofor_eylemi/esneme conf=%.2f",
                        event["zaman_saniye"],
                        event["confidence_score"],
                    )

            except Exception as exc:
                logger.warning(
                    "Frame %d at %.2fs failed; continuing. Details: %s",
                    frame_index,
                    time_seconds,
                    exc,
                )
    except Exception as exc:
        logger.warning("Video processing failed; writing fallback JSON. Details: %s", exc)

    plate, plate_confidence = choose_best_plate(plate_candidates)
    if plate is None:
        logger.warning("No valid OCR plate found; using fallback '%s'", PLATE_FALLBACK)
        plate = PLATE_FALLBACK
        plate_confidence = 0.01

    vehicle_type, vt_conf = _pick_best_label(vehicle_type_votes, "sedan")
    vehicle_color, vc_conf = _pick_best_label(color_votes, "beyaz")
    overall_confidence = max(plate_confidence, vt_conf, vc_conf)

    output_data = build_result(
        video_id=video_id,
        vehicle_type=vehicle_type,
        plate=plate,
        color=vehicle_color,
        vehicle_confidence=overall_confidence,
        events=events,
    )
    write_results_json(output_data, output_path)

    logger.info("Sampled frames: %d", sampled_frames)
    logger.info("Frames with plate detections: %d", detected_plates)
    logger.info("Detected events: %d", len(events))
    logger.info("Wrote FTR results: %s", output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
