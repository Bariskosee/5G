"""Print statistics for a YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yolo_dataset import (  # noqa: E402
    SPLITS,
    YoloDatasetError,
    YoloLabelError,
    discover_images,
    label_path_for_image,
    load_yolo_data_config,
    read_yolo_label_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument(
        "--min-count",
        type=int,
        default=100,
        help="Report classes with fewer than this many bounding boxes.",
    )
    parser.add_argument("--output-json", help="Optional path to save statistics as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.min_count < 0:
        print("ERROR: --min-count must be non-negative", file=sys.stderr)
        return 1

    try:
        config = load_yolo_data_config(args.data)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter({class_id: 0 for class_id in config.names})
    split_stats: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        if split not in config.split_paths:
            continue

        images, image_errors = discover_images(config.split_paths[split])
        errors.extend(f"{split}: {message}" for message in image_errors)

        label_files = 0
        boxes = 0
        for image_path in images:
            label_path = label_path_for_image(image_path)
            if not label_path.exists():
                warnings.append(f"{split}: missing label for image {image_path}")
                continue

            label_files += 1
            try:
                annotations = read_yolo_label_file(label_path, class_count=config.class_count)
            except YoloLabelError as exc:
                errors.append(str(exc))
                continue

            for annotation in annotations:
                class_counts[annotation.class_id] += 1
                boxes += 1

        split_stats[split] = {
            "images": len(images),
            "label_files": label_files,
            "boxes": boxes,
        }

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    low_count_classes = {
        config.names[class_id]: count
        for class_id, count in class_counts.items()
        if count < args.min_count
    }

    print(f"Dataset: {config.data_yaml}")
    print("\nSplits:")
    for split, stats in split_stats.items():
        print(
            f"  {split}: images={stats['images']} "
            f"label_files={stats['label_files']} boxes={stats['boxes']}"
        )

    print("\nBounding boxes per class:")
    for class_id, class_name in config.names.items():
        print(f"  {class_id:>2} {class_name}: {class_counts[class_id]}")

    print(f"\nClasses with fewer than {args.min_count} examples:")
    if low_count_classes:
        for class_name, count in low_count_classes.items():
            print(f"  {class_name}: {count}")
    else:
        print("  none")

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    output: dict[str, Any] = {
        "data": str(config.data_yaml),
        "classes": {str(class_id): name for class_id, name in config.names.items()},
        "splits": split_stats,
        "class_counts": {
            config.names[class_id]: class_counts[class_id] for class_id in config.names
        },
        "low_count_classes": low_count_classes,
        "warnings": warnings,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON stats to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
