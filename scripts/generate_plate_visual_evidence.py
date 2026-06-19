"""Generate annotated YOLO plate detection videos and label files for FTR evidence.

Runs YOLOv8 predict on each .mp4 in the input directory and saves annotated
output videos, per-frame label .txt files, and a visual_summary.csv.

Usage
-----
python scripts/generate_plate_visual_evidence.py \\
  --input-dir /tmp/5g_ftr_videos \\
  --output-dir /tmp/5g_ftr_outputs/yolo_visual \\
  --model models/model_b_plate/best.pt \\
  --imgsz 640 \\
  --conf 0.25 \\
  --vid-stride 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_predict(
    video: Path,
    model_path: Path,
    output_root: Path,
    imgsz: int,
    conf: float,
    vid_stride: int,
) -> dict:
    name = video.stem
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(video),
            imgsz=imgsz,
            conf=conf,
            vid_stride=vid_stride,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_root),
            name=name,
            verbose=False,
        )

        out_dir = output_root / name
        label_files = list((out_dir / "labels").glob("*.txt")) if (out_dir / "labels").exists() else []
        annotated = out_dir / video.name
        annotated_path = str(annotated) if annotated.exists() else ""

        logger.info(
            "%s → %d label files, annotated video: %s",
            video.name,
            len(label_files),
            annotated_path or "(not found)",
        )
        return {
            "video_id": video.name,
            "status": "ok",
            "output_dir": str(out_dir),
            "label_file_count": len(label_files),
            "annotated_video_path": annotated_path,
        }

    except Exception as exc:
        logger.error("Failed on %s: %s", video.name, exc)
        return {
            "video_id": video.name,
            "status": f"failed: {exc}",
            "output_dir": "",
            "label_file_count": 0,
            "annotated_video_path": "",
        }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory containing .mp4 files")
    parser.add_argument("--output-dir", default="/tmp/5g_ftr_outputs/yolo_visual", help="Annotated output root")
    parser.add_argument("--model", default="models/model_b_plate/best.pt", help="YOLO model path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--vid-stride", type=int, default=10)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).resolve()
    model_path = (repo_root / args.model).resolve()

    if not input_dir.exists():
        logger.error("input-dir not found: %s", input_dir)
        return 1
    if not model_path.exists():
        logger.error("model not found: %s", model_path)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        logger.error("No .mp4 files found in %s", input_dir)
        return 1

    logger.info("Running YOLO predict on %d video(s)...", len(videos))
    rows: list[dict] = []
    for video in videos:
        row = _run_predict(video, model_path, output_root, args.imgsz, args.conf, args.vid_stride)
        rows.append(row)

    summary_csv = output_root / "visual_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Visual summary: %s", summary_csv)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nResult: {ok}/{len(rows)} videos processed successfully.")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
