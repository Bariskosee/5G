"""Draw bounding-box annotations on random sample images for visual sanity check.

Selects N random images per class from a specified split and saves annotated
copies (bbox + class label) under output/<classname>/.

Usage:
    python scripts/visualize_bbox_samples.py \
        --data datasets/processed/model_a_vehicle_types/data.yaml \
        --split train \
        --n 20 \
        --output outputs/vehicle_type_visual_check \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yolo_dataset import (
    YoloDatasetError,
    discover_images,
    load_yolo_data_config,
    read_yolo_label_file,
)

# Distinct BGR colours per class index (cycles if > 7 classes)
PALETTE = [
    (255, 56, 56),    # red
    (56, 255, 56),    # green
    (56, 56, 255),    # blue
    (255, 165, 0),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (200, 200, 0),    # yellow
]


def _draw_annotations(image_path: Path, label_path: Path, class_names: dict[int, str]) -> None | tuple:
    """Return (image_bgr, annotations) or None if reading fails."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    try:
        annotations = read_yolo_label_file(label_path, class_count=len(class_names))
    except Exception:
        return None

    for ann in annotations:
        color = PALETTE[ann.class_id % len(PALETTE)]
        x1 = int((ann.x_center - ann.width / 2) * w)
        y1 = int((ann.y_center - ann.height / 2) * h)
        x2 = int((ann.x_center + ann.width / 2) * w)
        y2 = int((ann.y_center + ann.height / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[ann.class_id]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img, annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--split", default="train", help="Split to sample from (default: train)")
    parser.add_argument("--n", type=int, default=20, help="Samples per class (default: 20)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_yolo_data_config(args.data)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.split not in config.split_paths:
        print(f"ERROR: split '{args.split}' not in {args.data}", file=sys.stderr)
        return 1

    images, errors = discover_images(config.split_paths[args.split])
    if errors:
        for err in errors:
            print(f"WARNING: {err}", file=sys.stderr)

    # Build index: class_id → list of (image_path, label_path)
    class_images: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for img in images:
        lbl = img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            continue
        try:
            annotations = read_yolo_label_file(lbl, class_count=config.class_count)
        except Exception:
            continue
        seen_classes = {ann.class_id for ann in annotations}
        for cid in seen_classes:
            class_images[cid].append((img, lbl))

    output_dir = Path(args.output)
    rng = random.Random(args.seed)
    total_saved = 0

    for class_id in sorted(config.names):
        class_name = config.names[class_id]
        available = class_images.get(class_id, [])
        sample = rng.sample(available, min(args.n, len(available)))

        class_out = output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        saved = 0
        for img_path, lbl_path in sample:
            result = _draw_annotations(img_path, lbl_path, config.names)
            if result is None:
                continue
            annotated_img, _ = result
            out_file = class_out / img_path.name
            cv2.imwrite(str(out_file), annotated_img)
            saved += 1

        print(f"  {class_name:12s}: {saved}/{args.n} images → {class_out}")
        total_saved += saved

    print(f"\nTotal saved: {total_saved} annotated images under {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
