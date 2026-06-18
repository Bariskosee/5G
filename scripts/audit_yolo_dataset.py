"""Audit a YOLO-format dataset before training."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

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
        "--missing-labels",
        choices=("warn", "error"),
        default="warn",
        help="Treat missing label files as warnings or fatal errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_yaml = Path(args.data)

    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter()
    split_summaries: dict[str, dict[str, int]] = {}

    try:
        config = load_yolo_data_config(data_yaml)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not config.split_paths:
        print("ERROR: data.yaml does not declare any train/val/test split", file=sys.stderr)
        return 1

    for class_id in config.names:
        class_counts[class_id] = 0

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
                message = f"{split}: missing label for image {image_path}"
                if args.missing_labels == "error":
                    errors.append(message)
                else:
                    warnings.append(message)
                continue

            label_files += 1
            if label_path.stat().st_size == 0:
                warnings.append(f"{split}: empty label file: {label_path}")

            try:
                annotations = read_yolo_label_file(label_path, class_count=config.class_count)
            except YoloLabelError as exc:
                errors.append(str(exc))
                continue

            if not annotations and label_path.stat().st_size > 0:
                warnings.append(f"{split}: label file has no non-empty labels: {label_path}")

            for annotation in annotations:
                class_counts[annotation.class_id] += 1
                boxes += 1

        split_summaries[split] = {
            "images": len(images),
            "label_files": label_files,
            "boxes": boxes,
        }

    print(f"Dataset: {config.data_yaml}")
    print(f"Classes: {config.class_count}")
    print("\nSplits:")
    for split, summary in split_summaries.items():
        print(
            f"  {split}: images={summary['images']} "
            f"label_files={summary['label_files']} boxes={summary['boxes']}"
        )

    print("\nClass distribution:")
    for class_id, class_name in config.names.items():
        print(f"  {class_id:>2} {class_name}: {class_counts[class_id]}")

    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  WARNING: {warning}", file=sys.stderr)

    if errors:
        print("\nErrors:", file=sys.stderr)
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s).", file=sys.stderr)
        return 1

    print(f"\nOK: audit passed with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
