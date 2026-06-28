"""
Interactive local reviewer for weak YOLO classes (hatchback, panelvan).

Usage:
    python scripts/review_weak_classes_local.py

Keys (press in the image window):
    g  — good  (copy to review_good/<class>/)
    b  — bad   (copy to review_bad/<class>/)
    s  — skip
    q  — quit

Output:
    teknofest_backup/outputs/review/review_decisions.csv
"""

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REVIEW_ROOT = Path("teknofest_backup/outputs/review")
REVIEW_GOOD = Path("teknofest_backup/outputs/review_good")
REVIEW_BAD = Path("teknofest_backup/outputs/review_bad")
CSV_IN = REVIEW_ROOT / "review.csv"
CSV_OUT = REVIEW_ROOT / "review_decisions.csv"

REQUIRED_CSV_COLS = {"image", "class", "split", "bbox_count"}

WINDOW_NAME = "Review — g=good  b=bad  s=skip  q=quit"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay_label(img, text: str, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    cv2.putText(img, text, (10, h - 15), FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, h - 15), FONT, 0.7, color, 2, cv2.LINE_AA)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        print("  Run the training / review-image generation script first.")
        sys.exit(1)

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_CSV_COLS - set(reader.fieldnames or [])
        if missing:
            print(f"[ERROR] CSV is missing columns: {missing}")
            sys.exit(1)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[WARN] CSV is empty — nothing to review.")
        sys.exit(0)

    return rows


def write_decisions(decisions: list[dict]) -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "class", "split", "bbox_count", "decision"]
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    print(f"\nDecisions written → {CSV_OUT}")


def copy_image(src: Path, decision: str, cls: str) -> None:
    dest_root = REVIEW_GOOD if decision == "good" else REVIEW_BAD
    dest_dir = dest_root / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        # avoid silent overwrites — add a counter suffix
        stem, suffix = src.stem, src.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(src, dest)


def run_review(rows: list[dict]) -> list[dict]:
    decisions: list[dict] = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    total = len(rows)
    for idx, row in enumerate(rows):
        img_path = Path(row["image"])
        cls = row["class"]

        if not img_path.exists():
            print(f"[WARN] Image not found, skipping: {img_path}")
            decisions.append({**row, "decision": "skipped_missing"})
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image, skipping: {img_path}")
            decisions.append({**row, "decision": "skipped_unreadable"})
            continue

        display = img.copy()
        label = f"[{idx + 1}/{total}]  class={cls}  bboxes={row['bbox_count']}  split={row['split']}"
        overlay_label(display, label, (255, 255, 255))

        cv2.imshow(WINDOW_NAME, display)
        cv2.resizeWindow(WINDOW_NAME, min(display.shape[1], 1200), min(display.shape[0], 800))

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("g"):
                decision = "good"
                copy_image(img_path, decision, cls)
                print(f"  GOOD  {img_path.name}")
                break
            elif key == ord("b"):
                decision = "bad"
                copy_image(img_path, decision, cls)
                print(f"  BAD   {img_path.name}")
                break
            elif key == ord("s"):
                decision = "skip"
                print(f"  SKIP  {img_path.name}")
                break
            elif key == ord("q"):
                decision = "quit"
                print(f"  QUIT at {img_path.name}")
                decisions.append({**row, "decision": "skip"})
                cv2.destroyAllWindows()
                return decisions
            else:
                # ignore unknown keys
                continue

        decisions.append({**row, "decision": decision})

    cv2.destroyAllWindows()
    return decisions


def print_summary(decisions: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"good": 0, "bad": 0, "skipped": 0})

    for d in decisions:
        cls = d["class"]
        dec = d["decision"]
        if dec == "good":
            counts[cls]["good"] += 1
        elif dec == "bad":
            counts[cls]["bad"] += 1
        else:
            counts[cls]["skipped"] += 1

    print("\n" + "=" * 50)
    print("REVIEW SUMMARY")
    print("=" * 50)
    for cls in sorted(counts):
        c = counts[cls]
        total = c["good"] + c["bad"] + c["skipped"]
        print(f"  {cls:12s}  good={c['good']}  bad={c['bad']}  skipped={c['skipped']}  (total={total})")
    print("=" * 50)


def main() -> None:
    rows = load_csv(CSV_IN)
    print(f"Loaded {len(rows)} images from {CSV_IN}")
    print("Controls: g=good  b=bad  s=skip  q=quit\n")

    decisions = run_review(rows)
    write_decisions(decisions)
    print_summary(decisions)


if __name__ == "__main__":
    main()
