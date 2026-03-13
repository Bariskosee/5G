"""CLI script: evaluate detection results against ground truth labels.

Reads a predictions JSON file and a ground truth labels JSON file, then
prints precision, recall, IoU, and accuracy metrics.

Both files must be lists of objects with at minimum:
``{"bbox": [x1,y1,x2,y2], "class": str}``

Usage::

    python scripts/evaluate.py \\
        --predictions outputs/predictions.json \\
        --labels data/labels/sample.json \\
        [--iou-threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_iou,
    calculate_precision,
    calculate_recall,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detection predictions against ground truth labels."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        type=Path,
        help="Path to predictions JSON file.",
    )
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        type=Path,
        help="Path to ground truth labels JSON file.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for true-positive matching (default: 0.5).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in '{path}', got {type(data).__name__}")
    return data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        predictions = _load_json(args.predictions)
        ground_truth = _load_json(args.labels)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logging.error("Failed to load files: %s", exc)
        return 1

    iou_threshold = args.iou_threshold

    # Compute TP / FP / FN
    matched_gt: set[int] = set()
    tp = 0
    for pred in predictions:
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred.get("class") != gt.get("class"):
                continue
            try:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
            except (ValueError, KeyError):
                continue
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                break

    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    accuracy = calculate_accuracy(predictions, ground_truth, iou_threshold)

    print(f"Predictions : {len(predictions)}")
    print(f"Ground truth: {len(ground_truth)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
