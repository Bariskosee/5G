"""Split a single-split YOLO dataset into train / valid / test.

Works in-place: files are moved from the existing split (default: train) into
new valid/ and test/ subdirectories.  Background images (empty labels) stay in
train only.

Usage:
    python scripts/split_yolo_dataset.py \
        --data datasets/processed/model_a_vehicle_types/data.yaml \
        --train 0.8 --val 0.1 --test 0.1 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yolo_dataset import YoloDatasetError, discover_images, load_yolo_data_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to source YOLO data.yaml")
    parser.add_argument("--source-split", default="train", help="Name of split to restructure (default: train)")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def _label_path(image_path: Path) -> Path:
    """Return the label .txt path corresponding to an image path."""
    return image_path.parent.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")


def _move_pair(image_path: Path, target_images_dir: Path, target_labels_dir: Path) -> None:
    label_path = _label_path(image_path)
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), target_images_dir / image_path.name)
    if label_path.exists():
        shutil.move(str(label_path), target_labels_dir / label_path.name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"ERROR: --train + --val + --test must sum to 1.0 (got {total:.3f})", file=sys.stderr)
        return 1

    try:
        config = load_yolo_data_config(args.data)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.source_split not in config.split_paths:
        print(f"ERROR: split '{args.source_split}' not found in {args.data}", file=sys.stderr)
        return 1

    source_image_dir = config.split_paths[args.source_split][0]
    images, errors = discover_images([source_image_dir])
    if errors:
        for err in errors:
            print(f"WARNING: {err}", file=sys.stderr)

    labeled: list[Path] = []
    background: list[Path] = []
    for img in images:
        lbl = _label_path(img)
        if lbl.exists() and lbl.stat().st_size > 0:
            labeled.append(img)
        else:
            background.append(img)

    print(f"Source split   : {args.source_split} ({source_image_dir})")
    print(f"Total images   : {len(images)}")
    print(f"Labeled        : {len(labeled)}")
    print(f"Background     : {len(background)}")

    rng = random.Random(args.seed)
    rng.shuffle(labeled)

    n_val = int(len(labeled) * args.val)
    n_test = int(len(labeled) * args.test)

    val_images = labeled[:n_val]
    test_images = labeled[n_val:n_val + n_test]
    # rest of labeled + all background stay in train (nothing to move for them)

    dataset_dir = config.data_yaml.parent

    def images_dir(split: str) -> Path:
        return dataset_dir / "images" / split

    def labels_dir(split: str) -> Path:
        return dataset_dir / "labels" / split

    print(f"\nMoving {len(val_images)} images → valid ...")
    for img in val_images:
        _move_pair(img, images_dir("valid"), labels_dir("valid"))

    print(f"Moving {len(test_images)} images → test ...")
    for img in test_images:
        _move_pair(img, images_dir("test"), labels_dir("test"))

    # Rename train split dir if it was flat (images/train → images/train stays)
    # The remaining files in images/train are the new train set.
    n_train = len(labeled) - n_val - n_test + len(background)
    print(f"Train kept     : {n_train} images")

    # Update data.yaml
    names_list = [config.names[i] for i in sorted(config.names)]
    data_yaml_content: dict = {
        "nc": config.class_count,
        "names": names_list,
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
    }
    config.data_yaml.write_text(
        yaml.dump(data_yaml_content, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nUpdated        : {config.data_yaml}")
    print("Split complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
