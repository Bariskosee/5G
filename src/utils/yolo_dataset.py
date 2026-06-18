"""Shared helpers for YOLO-format dataset preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


class YoloDatasetError(ValueError):
    """Raised when a YOLO dataset configuration is invalid."""


class YoloLabelError(ValueError):
    """Raised when a YOLO label line is invalid."""


@dataclass(frozen=True)
class YoloAnnotation:
    """One YOLO bbox annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        """Return the annotation as a normalized YOLO bbox line."""
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass(frozen=True)
class YoloDataConfig:
    """Parsed data.yaml information needed by dataset utilities."""

    data_yaml: Path
    root: Path
    names: dict[int, str]
    split_paths: dict[str, list[Path]]
    raw: dict[str, Any]

    @property
    def class_count(self) -> int:
        return len(self.names)

    @property
    def names_by_value(self) -> dict[str, int]:
        return {name: class_id for class_id, name in self.names.items()}


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and ensure the top-level value is a mapping."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise YoloDatasetError(f"YAML file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise YoloDatasetError(f"{yaml_path}: expected a YAML mapping")
    return data


def parse_class_names(data: dict[str, Any], source: str | Path = "data.yaml") -> dict[int, str]:
    """Parse YOLO class names from list or id-to-name mapping format."""
    source = str(source)
    names_raw = data.get("names")
    nc_raw = data.get("nc")

    if names_raw is None:
        if isinstance(nc_raw, int) and nc_raw >= 0:
            return {class_id: f"class_{class_id}" for class_id in range(nc_raw)}
        raise YoloDatasetError(f"{source}: missing required 'names' field")

    if isinstance(names_raw, list):
        names = {idx: str(name) for idx, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        names = {}
        for key, value in names_raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise YoloDatasetError(
                    f"{source}: class id '{key}' in names is not an integer"
                ) from exc
            if class_id < 0:
                raise YoloDatasetError(f"{source}: class id {class_id} cannot be negative")
            names[class_id] = str(value)
    else:
        raise YoloDatasetError(f"{source}: 'names' must be a list or mapping")

    if not names:
        raise YoloDatasetError(f"{source}: 'names' must contain at least one class")

    expected_ids = set(range(max(names) + 1))
    missing_ids = expected_ids - names.keys()
    if missing_ids:
        raise YoloDatasetError(
            f"{source}: names must use contiguous class ids; missing {sorted(missing_ids)}"
        )

    if nc_raw is not None:
        try:
            nc = int(nc_raw)
        except (TypeError, ValueError) as exc:
            raise YoloDatasetError(f"{source}: 'nc' must be an integer") from exc
        if nc != len(names):
            raise YoloDatasetError(
                f"{source}: nc={nc} does not match number of classes ({len(names)})"
            )

    return dict(sorted(names.items()))


def load_yolo_data_config(data_yaml: str | Path) -> YoloDataConfig:
    """Load a YOLO data.yaml file and resolve declared split paths."""
    yaml_path = Path(data_yaml).resolve()
    data = load_yaml(yaml_path)
    names = parse_class_names(data, yaml_path)

    root_raw = data.get("path")
    if root_raw is None:
        root = yaml_path.parent
    else:
        root = Path(str(root_raw))
        if not root.is_absolute():
            root = yaml_path.parent / root
    root = root.resolve()

    split_paths: dict[str, list[Path]] = {}
    for split in SPLITS:
        if split not in data or data[split] is None:
            continue

        raw_value = data[split]
        raw_paths: Iterable[Any]
        if isinstance(raw_value, list):
            raw_paths = raw_value
        else:
            raw_paths = [raw_value]

        paths: list[Path] = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, (str, Path)):
                raise YoloDatasetError(
                    f"{yaml_path}: split '{split}' path must be a string or list of strings"
                )
            split_path = Path(str(raw_path))
            if not split_path.is_absolute():
                split_path = root / split_path
            paths.append(split_path.resolve())

        split_paths[split] = paths

    return YoloDataConfig(
        data_yaml=yaml_path,
        root=root,
        names=names,
        split_paths=split_paths,
        raw=data,
    )


def _images_from_list_file(list_path: Path) -> tuple[list[Path], list[str]]:
    images: list[Path] = []
    errors: list[str] = []

    for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            continue

        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = list_path.parent / image_path
        image_path = image_path.resolve()

        if not image_path.exists():
            errors.append(f"{list_path}:{line_number}: image file not found: {image_path}")
        elif not image_path.is_file() or not is_image_file(image_path):
            errors.append(f"{list_path}:{line_number}: not a supported image file: {image_path}")
        else:
            images.append(image_path)

    return images, errors


def discover_images(paths: Iterable[Path]) -> tuple[list[Path], list[str]]:
    """Discover image files from split directories, image files, or list files."""
    images: list[Path] = []
    errors: list[str] = []

    for path in paths:
        if not path.exists():
            errors.append(f"split path does not exist: {path}")
            continue

        if path.is_dir():
            images.extend(
                sorted(
                    item.resolve()
                    for item in path.rglob("*")
                    if item.is_file() and is_image_file(item)
                )
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            listed_images, list_errors = _images_from_list_file(path)
            images.extend(listed_images)
            errors.extend(list_errors)
        elif path.is_file() and is_image_file(path):
            images.append(path.resolve())
        else:
            errors.append(f"unsupported split path type: {path}")

    unique_images = sorted(dict.fromkeys(images))
    return unique_images, errors


def label_path_for_image(image_path: str | Path) -> Path:
    """Return the conventional YOLO label path for an image path."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    image_dirs = [idx for idx, part in enumerate(parts) if part == "images"]

    if image_dirs:
        idx = image_dirs[-1]
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def output_relative_image_path(image_path: Path, split_roots: Iterable[Path]) -> Path:
    """Choose a stable relative path for copying an image into a split folder."""
    image_path = image_path.resolve()

    for root in split_roots:
        root = root.resolve()
        if root.is_file():
            root = root.parent
        try:
            return image_path.relative_to(root)
        except ValueError:
            continue

    parts = list(image_path.parts)
    image_dirs = [idx for idx, part in enumerate(parts) if part == "images"]
    if image_dirs:
        idx = image_dirs[-1]
        tail = parts[idx + 1 :]
        if tail and tail[0] in SPLITS:
            tail = tail[1:]
        if tail:
            return Path(*tail)

    return Path(image_path.name)


def parse_yolo_label_line(
    line: str,
    class_count: int | None = None,
    source: str | Path = "<memory>",
    line_number: int = 1,
) -> YoloAnnotation:
    """Parse and validate one YOLO bbox label line."""
    location = f"{source}:{line_number}"
    parts = line.strip().split()

    if not parts:
        raise YoloLabelError(f"{location}: empty label line")
    if len(parts) != 5:
        raise YoloLabelError(
            f"{location}: expected 5 fields '<class_id> <x_center> <y_center> <width> <height>', "
            f"got {len(parts)}"
        )

    class_token = parts[0]
    if not class_token.lstrip("-").isdigit():
        raise YoloLabelError(f"{location}: class id must be an integer, got '{class_token}'")

    class_id = int(class_token)
    if class_id < 0:
        raise YoloLabelError(f"{location}: class id must be non-negative, got {class_id}")
    if class_count is not None and class_id >= class_count:
        raise YoloLabelError(
            f"{location}: class id {class_id} is outside valid range 0..{class_count - 1}"
        )

    values: list[float] = []
    for field_name, token in zip(("x_center", "y_center", "width", "height"), parts[1:]):
        try:
            value = float(token)
        except ValueError as exc:
            raise YoloLabelError(
                f"{location}: bbox {field_name} must be numeric, got '{token}'"
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise YoloLabelError(
                f"{location}: bbox {field_name}={value} must be between 0 and 1"
            )
        values.append(value)

    return YoloAnnotation(class_id, values[0], values[1], values[2], values[3])


def read_yolo_label_file(label_path: str | Path, class_count: int | None = None) -> list[YoloAnnotation]:
    """Read and validate all non-empty YOLO bbox lines in a label file."""
    label_path = Path(label_path)
    annotations: list[YoloAnnotation] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        annotations.append(
            parse_yolo_label_line(
                line,
                class_count=class_count,
                source=label_path,
                line_number=line_number,
            )
        )

    return annotations


def names_for_yaml(names: dict[int, str]) -> dict[int, str]:
    """Return class names in stable id order for YAML output."""
    return {class_id: names[class_id] for class_id in sorted(names)}


def write_yolo_data_yaml(output_dir: str | Path, names: dict[int, str], splits: Iterable[str]) -> Path:
    """Write a compact YOLO data.yaml in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_set = set(splits)
    data: dict[str, Any] = {"path": "."}
    for split in SPLITS:
        if split in split_set:
            data[split] = f"images/{split}"
    data["nc"] = len(names)
    data["names"] = names_for_yaml(names)

    data_yaml = output_dir / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

    return data_yaml
