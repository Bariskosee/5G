"""Remap YOLO class ids from a source dataset into project target ids."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

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
    output_relative_image_path,
    read_yolo_label_file,
    write_yolo_data_yaml,
)

TARGET_MODEL_CONFIGS = {
    "model_a": REPO_ROOT / "configs" / "model_a_classes.yaml",
    "model_b": REPO_ROOT / "configs" / "model_b_classes.yaml",
    "model_a_unified": REPO_ROOT / "configs" / "model_a_classes.yaml",
    "model_b_plate": REPO_ROOT / "configs" / "model_b_classes.yaml",
    "model_a_vehicle_types": REPO_ROOT / "configs" / "model_a_vehicle_type_classes.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to source YOLO data.yaml")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--mapping", required=True, help="Class-name remapping YAML file")
    return parser


def load_mapping(mapping_path: str | Path) -> tuple[str, dict[str, str]]:
    path = Path(mapping_path)
    if not path.exists():
        raise YoloDatasetError(f"mapping YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise YoloDatasetError(f"{path}: expected a YAML mapping")

    target_model = data.get("target_model")
    if target_model not in TARGET_MODEL_CONFIGS:
        allowed = ", ".join(sorted(TARGET_MODEL_CONFIGS))
        raise YoloDatasetError(f"{path}: target_model must be one of: {allowed}")

    mapping_raw = data.get("source_to_target")
    if not isinstance(mapping_raw, dict) or not mapping_raw:
        raise YoloDatasetError(f"{path}: source_to_target must be a non-empty mapping")

    mapping = {str(source): str(target) for source, target in mapping_raw.items()}
    return str(target_model), mapping


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        source_config = load_yolo_data_config(args.data)
        target_model, source_to_target = load_mapping(args.mapping)
        target_config = load_yolo_data_config(TARGET_MODEL_CONFIGS[target_model])
    except YoloDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    target_name_to_id = target_config.names_by_value
    unknown_targets = sorted(set(source_to_target.values()) - set(target_name_to_id))
    if unknown_targets:
        print(
            f"ERROR: mapping targets are not valid {target_model} classes: {unknown_targets}",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output)
    errors: list[str] = []
    copied_images = 0
    missing_labels = 0
    kept_labels = 0
    skipped_labels = 0
    skipped_by_source: Counter[str] = Counter()
    kept_by_target: Counter[str] = Counter()
    processed_splits: list[str] = []

    for split in SPLITS:
        if split not in source_config.split_paths:
            continue

        processed_splits.append(split)
        images, image_errors = discover_images(source_config.split_paths[split])
        errors.extend(f"{split}: {message}" for message in image_errors)

        for image_path in images:
            relative_image = output_relative_image_path(image_path, source_config.split_paths[split])
            output_image = output_dir / "images" / split / relative_image
            output_label = output_dir / "labels" / split / relative_image.with_suffix(".txt")

            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            copied_images += 1

            source_label = label_path_for_image(image_path)
            if not source_label.exists():
                missing_labels += 1
                write_label(output_label, [])
                continue

            try:
                annotations = read_yolo_label_file(
                    source_label,
                    class_count=source_config.class_count,
                )
            except YoloLabelError as exc:
                errors.append(str(exc))
                write_label(output_label, [])
                continue

            remapped_lines: list[str] = []
            for annotation in annotations:
                source_name = source_config.names[annotation.class_id]
                target_name = source_to_target.get(source_name)
                if target_name is None:
                    skipped_labels += 1
                    skipped_by_source[source_name] += 1
                    continue

                target_class_id = target_name_to_id[target_name]
                remapped = replace(annotation, class_id=target_class_id)
                remapped_lines.append(remapped.to_yolo_line())
                kept_labels += 1
                kept_by_target[target_name] += 1

            write_label(output_label, remapped_lines)

    write_yolo_data_yaml(output_dir, target_config.names, processed_splits)

    print(f"Source dataset: {source_config.data_yaml}")
    print(f"Output dataset: {output_dir}")
    print(f"Target model: {target_model}")
    print(f"Images copied: {copied_images}")
    print(f"Labels kept: {kept_labels}")
    print(f"Labels skipped: {skipped_labels}")
    print(f"Images with missing source labels: {missing_labels}")

    print("\nKept labels by target class:")
    if kept_by_target:
        for target_name, count in sorted(kept_by_target.items()):
            print(f"  {target_name}: {count}")
    else:
        print("  none")

    print("\nSkipped labels by source class:")
    if skipped_by_source:
        for source_name, count in sorted(skipped_by_source.items()):
            print(f"  {source_name}: {count}")
    else:
        print("  none")

    if errors:
        print("\nErrors:", file=sys.stderr)
        for error in errors:
            print(f"  ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
