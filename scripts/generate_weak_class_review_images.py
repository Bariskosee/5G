"""
Generate annotated review images for weak YOLO classes (hatchback, panelvan).

Reads ground-truth label files from the dataset — NOT model predictions.
Draws all bounding boxes; highlights target-class boxes with a distinct color.

Usage:
    python scripts/generate_weak_class_review_images.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("datasets/processed/model_a_vehicle_types")
SPLITS = ["train", "valid", "test"]

TARGET_CLASSES = {2: "hatchback", 5: "panelvan"}

CLASS_NAMES = {
    0: "sedan",
    1: "suv",
    2: "hatchback",
    3: "pickup",
    4: "minibus",
    5: "panelvan",
    6: "kamyon",
}

# BGR colors — one per class
CLASS_COLORS = {
    0: (200, 200, 200),   # sedan      — light grey
    1: (255, 200, 0),     # suv        — cyan-ish
    2: (0, 80, 255),      # hatchback  — vivid red (BGR)
    3: (0, 200, 50),      # pickup     — green
    4: (200, 100, 0),     # minibus    — blue
    5: (0, 0, 255),       # panelvan   — pure red (BGR)
    6: (150, 0, 200),     # kamyon     — purple
}

REVIEW_ROOT = Path("teknofest_backup/outputs/review")
CSV_OUT = REVIEW_ROOT / "review.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
TARGET_THICKNESS = 3   # thicker box for hatchback / panelvan
LABEL_PAD = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cls_id, cx, cy, bw, bh))
    return boxes


def yolo_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1)


def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             color: tuple[int, int, int], label: str, is_target: bool) -> None:
    thick = TARGET_THICKNESS if is_target else THICKNESS
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    bg_y1 = max(y1 - th - 2 * LABEL_PAD, 0)
    bg_y2 = bg_y1 + th + 2 * LABEL_PAD
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 2 * LABEL_PAD, bg_y2), color, -1)

    txt_color = (255, 255, 255) if is_target else (0, 0, 0)
    cv2.putText(img, label, (x1 + LABEL_PAD, bg_y2 - LABEL_PAD - baseline // 2),
                FONT, FONT_SCALE, txt_color, 1, cv2.LINE_AA)


def annotate_image(img_path: Path,
                   boxes: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    # draw non-target classes first (background layer)
    for cls_id, cx, cy, bw, bh in boxes:
        if cls_id in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_id, (180, 180, 180))
        draw_box(img, x1, y1, x2, y2, color, label, is_target=False)

    # draw target classes on top
    for cls_id, cx, cy, bw, bh in boxes:
        if cls_id not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
        label = TARGET_CLASSES[cls_id].upper()
        color = CLASS_COLORS.get(cls_id, (0, 0, 255))
        draw_box(img, x1, y1, x2, y2, color, label, is_target=True)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ensure output dirs exist
    for cls_name in TARGET_CLASSES.values():
        (REVIEW_ROOT / cls_name).mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    errors = 0

    for split in SPLITS:
        labels_dir = DATASET_ROOT / "labels" / split
        images_dir = DATASET_ROOT / "images" / split

        if not labels_dir.exists():
            print(f"[SKIP] No labels dir for split '{split}'")
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"[{split}] scanning {len(label_files)} label files …")

        for label_path in label_files:
            boxes = parse_label_file(label_path)
            if not boxes:
                continue

            box_class_ids = {b[0] for b in boxes}
            hit_targets = box_class_ids & set(TARGET_CLASSES)
            if not hit_targets:
                continue

            img_path = find_image(label_path, images_dir)
            if img_path is None:
                print(f"  [WARN] No image for label: {label_path.name}")
                errors += 1
                continue

            try:
                annotated = annotate_image(img_path, boxes)
            except IOError as e:
                print(f"  [WARN] {e}")
                errors += 1
                continue

            bbox_count = len(boxes)

            for cls_id in hit_targets:
                cls_name = TARGET_CLASSES[cls_id]
                out_dir = REVIEW_ROOT / cls_name
                out_path = out_dir / img_path.name

                # avoid silent collision — add split prefix if filename exists
                if out_path.exists():
                    out_path = out_dir / f"{split}_{img_path.name}"

                cv2.imwrite(str(out_path), annotated)

                csv_rows.append({
                    "image": str(out_path),
                    "class": cls_name,
                    "split": split,
                    "bbox_count": bbox_count,
                })
                counts[cls_name][split] += 1

    # write CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "split", "bbox_count"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # summary
    print("\n" + "=" * 55)
    print("GENERATION SUMMARY")
    print("=" * 55)
    total = 0
    for cls_name in sorted(TARGET_CLASSES.values()):
        by_split = counts[cls_name]
        subtotal = sum(by_split.values())
        total += subtotal
        split_str = "  ".join(f"{s}={by_split[s]}" for s in SPLITS if by_split[s])
        print(f"  {cls_name:10s}  {split_str}  → total={subtotal}")
    print(f"  {'TOTAL':10s}  {total} review images generated")
    if errors:
        print(f"  [WARN] {errors} images skipped due to read errors")
    print("=" * 55)
    print(f"\nCSV  → {CSV_OUT}")
    print(f"Images → {REVIEW_ROOT}/hatchback/  and  {REVIEW_ROOT}/panelvan/")


if __name__ == "__main__":
    main()
