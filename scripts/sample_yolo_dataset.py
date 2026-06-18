"""Create a small sample subset from a YOLO-format dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.yolo_dataset import (  # noqa: E402
    SPLITS,
    YoloDatasetError,
    discover_images,
    label_path_for_image,
    load_yolo_data_config,
    output_relative_image_path,
    write_yolo_data_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to source YOLO data.yaml")
    parser.add_argument("--output", required=True, help="Output sample dataset directory")
    parser.add_argument("--n", type=int, required=True, help="Number of images to sample")
    parser.add_argument(
        "--mode",
        choices=("per-split", "total"),
        default="per-split",
        help="Sample N images per split or N images total across all declared splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser


def copy_sample(image_path: Path, output_dir: Path, split: str, split_roots: list[Path]) -> bool:
    relative_image = output_relative_image_path(image_path, split_roots)
    output_image = output_dir / "images" / split / relative_image
    output_label = output_dir / "labels" / split / relative_image.with_suffix(".txt")

    output_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_image)

    label_path = label_path_for_image(image_path)
    if label_path.exists():
        output_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, output_label)
        return True

    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.n < 0:
        print("ERROR: --n must be non-negative", file=sys.stderr)
        return 1

    try:
        config = load_yolo_data_config(args.data)
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    output_dir = Path(args.output)
    split_images: dict[str, list[Path]] = {}
    errors: list[str] = []

    for split in SPLITS:
        if split not in config.split_paths:
            continue
        images, image_errors = discover_images(config.split_paths[split])
        split_images[split] = images
        errors.extend(f"{split}: {message}" for message in image_errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    selected: dict[str, list[Path]] = {split: [] for split in split_images}
    if args.mode == "per-split":
        for split, images in split_images.items():
            selected[split] = rng.sample(images, min(args.n, len(images)))
    else:
        all_images = [
            (split, image_path)
            for split, images in split_images.items()
            for image_path in images
        ]
        for split, image_path in rng.sample(all_images, min(args.n, len(all_images))):
            selected[split].append(image_path)

    copied_images = 0
    copied_labels = 0
    for split, images in selected.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image_path in images:
            has_label = copy_sample(
                image_path,
                output_dir,
                split,
                config.split_paths[split],
            )
            copied_images += 1
            if has_label:
                copied_labels += 1

    write_yolo_data_yaml(output_dir, config.names, selected.keys())

    print(f"Source dataset: {config.data_yaml}")
    print(f"Sample dataset: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Images copied: {copied_images}")
    print(f"Labels copied: {copied_labels}")
    print("\nImages per split:")
    for split in SPLITS:
        if split in selected:
            print(f"  {split}: {len(selected[split])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
