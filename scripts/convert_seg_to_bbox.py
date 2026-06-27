"""Convert YOLO segmentation labels to bounding-box detection labels.

Roboflow sometimes exports segmentation (polygon) labels even for classification
datasets. This script converts each polygon annotation to its axis-aligned
bounding box so the dataset can be used for standard YOLO detection training.

Usage:
    python scripts/convert_seg_to_bbox.py \
        --data "datasets/raw/Vehicle Classification V2.v3i.yolov8/data.yaml" \
        --output datasets/raw/vehicle_classification_bbox
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yolo_dataset import SPLITS, YoloDatasetError, discover_images, load_yolo_data_config


def _seg_line_to_bbox(fields: list[str], class_count: int) -> str | None:
    """Convert a segmentation line to a YOLO bbox line.

    Returns None if the line is malformed or class_id is out of range.
    """
    if len(fields) < 7:
        return None
    try:
        class_id = int(fields[0])
    except ValueError:
        return None
    if not (0 <= class_id < class_count):
        return None

    coords = fields[1:]
    if len(coords) % 2 != 0:
        return None
    try:
        xs = [float(coords[i]) for i in range(0, len(coords), 2)]
        ys = [float(coords[i]) for i in range(1, len(coords), 2)]
    except ValueError:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml")
    parser.add_argument("--output", required=True, help="Output directory for converted dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        source_config = load_yolo_data_config(args.data)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    total_images = 0
    total_converted = 0
    total_skipped_lines = 0
    processed_splits: list[str] = []

    for split in SPLITS:
        if split not in source_config.split_paths:
            continue

        processed_splits.append(split)
        images, errors = discover_images(source_config.split_paths[split])
        if errors:
            for err in errors:
                print(f"WARNING: {err}", file=sys.stderr)

        for image_path in images:
            total_images += 1
            rel = image_path.relative_to(source_config.split_paths[split][0])
            out_image = output_dir / "images" / split / rel
            out_label = output_dir / "labels" / split / rel.with_suffix(".txt")

            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_image)

            label_path = image_path.parent.parent / "labels" / rel.with_suffix(".txt")
            if not label_path.exists():
                out_label.write_text("", encoding="utf-8")
                continue

            bbox_lines: list[str] = []
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) == 5:
                    bbox_lines.append(line)
                    total_converted += 1
                elif len(fields) >= 7:
                    converted = _seg_line_to_bbox(fields, source_config.class_count)
                    if converted:
                        bbox_lines.append(converted)
                        total_converted += 1
                    else:
                        total_skipped_lines += 1
                else:
                    total_skipped_lines += 1

            text = "\n".join(bbox_lines)
            if text:
                text += "\n"
            out_label.write_text(text, encoding="utf-8")

    # Write output data.yaml with corrected split paths
    names_list = [source_config.names[i] for i in sorted(source_config.names)]
    data_yaml: dict = {"nc": source_config.class_count, "names": names_list}
    for split in processed_splits:
        data_yaml[split] = f"{split}/images"

    (output_dir / "data.yaml").write_text(
        yaml.dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Source dataset : {source_config.data_yaml}")
    print(f"Output dataset : {output_dir}")
    print(f"Splits processed: {processed_splits}")
    print(f"Images copied   : {total_images}")
    print(f"Annotations kept: {total_converted}")
    print(f"Lines skipped   : {total_skipped_lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
