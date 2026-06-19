"""Reproducible batch video test runner for the FTR inference skeleton.

Copies videos to a work directory (handles extensionless files), checks OpenCV
readability, runs main.py on each readable video, validates every results.json,
and writes summary.csv + video_metadata.csv.

Usage
-----
python scripts/run_ftr_video_tests.py \\
  --input-dir datasets/raw/turkish_number_plates/dataset \\
  --output-dir /tmp/5g_ftr_outputs \\
  --plate-model models/model_b_plate/best.pt \\
  --frame-stride 10 \\
  --disable-ocr \\
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _discover_videos(input_dir: Path) -> list[Path]:
    """Return all files that look like videos (by extension or absence of extension).

    Hidden files (names starting with '.') are always skipped.
    """
    candidates: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            candidates.append(path)
        elif path.suffix == "":
            candidates.append(path)
    return candidates


def _normalize_to_work_dir(
    video: Path,
    work_dir: Path,
    overwrite: bool,
) -> Path:
    """Copy video into work_dir, adding .mp4 extension if it has none."""
    stem = video.stem if video.suffix else video.name
    target = work_dir / (stem + ".mp4" if not video.suffix else video.name)
    if target.exists() and not overwrite:
        return target
    shutil.copy2(video, target)
    return target


def _probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    first_frame_ok = False
    if opened:
        first_frame_ok, _ = cap.read()
    cap.release()
    duration = round(n / fps, 3) if fps > 0 else 0.0
    size_mb = round(path.stat().st_size / 1024**2, 2) if path.exists() else 0.0
    return {
        "opened": opened,
        "first_frame_read": bool(first_frame_ok),
        "width": w,
        "height": h,
        "source_fps": round(float(fps), 3),
        "frame_count": n,
        "duration_s": duration,
        "file_size_mb": size_mb,
    }


def _run_inference(
    normalized_path: Path,
    output_json: Path,
    log_path: Path,
    model_path: Path,
    frame_stride: int,
    max_frames: int | None,
    conf_threshold: float,
    disable_ocr: bool,
    python_exe: str,
    repo_root: Path,
) -> tuple[int, float]:
    cmd = [
        python_exe,
        "main.py",
        "--input",
        str(normalized_path),
        "--output",
        str(output_json),
        "--plate-model",
        str(model_path),
        "--frame-stride",
        str(frame_stride),
        "--conf-threshold",
        str(conf_threshold),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if disable_ocr:
        cmd.append("--disable-ocr")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    elapsed = time.perf_counter() - t0

    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    return result.returncode, elapsed


def _validate_json(json_path: Path, python_exe: str, repo_root: Path) -> bool:
    if not json_path.exists():
        return False
    r = subprocess.run(
        [python_exe, "scripts/validate_results_json.py", str(json_path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    return r.returncode == 0


def _parse_json_fields(json_path: Path) -> tuple[str | None, float | None, int | None]:
    if not json_path.exists():
        return None, None, None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        plate = data.get("arac_bilgisi", {}).get("plaka")
        conf = data.get("arac_bilgisi", {}).get("confidence_score")
        n_det = len(data.get("tespitler", []))
        return plate, conf, n_det
    except Exception:
        return None, None, None


def _parse_inference_log(log_path: Path) -> tuple[int, int]:
    """Return (sampled_frames, frames_with_plate_detection) by parsing run.log stderr."""
    sampled, detected = 0, 0
    if not log_path.exists():
        return sampled, detected
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "Sampled frames:" in line:
            try:
                sampled = int(line.split("Sampled frames:")[-1].strip())
            except ValueError:
                pass
        elif "Frames with plate detections:" in line:
            try:
                detected = int(line.split("Frames with plate detections:")[-1].strip())
            except ValueError:
                pass
    return sampled, detected


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--output-dir", default="/tmp/5g_ftr_outputs", help="Output root directory")
    parser.add_argument("--work-dir", default="/tmp/5g_ftr_videos", help="Temp directory for normalized video copies")
    parser.add_argument("--plate-model", default="models/model_b_plate/best.pt", help="YOLO plate model path")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing work-dir copies")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    model_path = (repo_root / args.plate_model).resolve()

    if not input_dir.exists():
        logger.error("input-dir not found: %s", input_dir)
        return 1
    if not model_path.exists():
        logger.error("plate-model not found: %s", model_path)
        return 1

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_videos = _discover_videos(input_dir)
    logger.info("Found %d candidate video files in %s", len(raw_videos), input_dir)

    metadata_rows: list[dict] = []
    summary_rows: list[dict] = []

    for video in raw_videos:
        normalized = _normalize_to_work_dir(video, work_dir, args.overwrite)
        probe = _probe_video(normalized)

        meta_row = {
            "video_id": normalized.name,
            "original_path": str(video),
            "normalized_path": str(normalized),
            "original_extension": video.suffix or "(none)",
            **probe,
        }
        metadata_rows.append(meta_row)

        readable = probe["opened"] and probe["first_frame_read"]
        if not readable:
            logger.warning("Skipping unreadable video: %s", normalized.name)
            summary_rows.append({
                "video_id": normalized.name,
                "status": "skipped_unreadable",
                "resolution": f"{probe['width']}x{probe['height']}",
                "source_fps": probe["source_fps"],
                "frames": probe["frame_count"],
                "duration_s": probe["duration_s"],
                "runtime_s": "",
                "effective_video_fps": "",
                "processed_sampled_frames": "",
                "frames_with_plate_detection": "",
                "plate_detection_frame_ratio": "",
                "json_valid": False,
                "return_code": "",
                "plate_output": "",
                "vehicle_confidence": "",
                "tespitler_count": "",
                "output_json": "",
                "log": "",
            })
            continue

        stem = normalized.stem
        out_dir = output_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        output_json = out_dir / "results.json"
        log_path = out_dir / "run.log"

        logger.info("=== %s (%dx%d, %.1fs) ===", normalized.name, probe["width"], probe["height"], probe["duration_s"])

        rc, elapsed = _run_inference(
            normalized_path=normalized,
            output_json=output_json,
            log_path=log_path,
            model_path=model_path,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            conf_threshold=args.conf_threshold,
            disable_ocr=args.disable_ocr,
            python_exe=args.python,
            repo_root=repo_root,
        )

        json_valid = _validate_json(output_json, args.python, repo_root)
        plate, conf, n_det = _parse_json_fields(output_json)
        sampled_frames, frames_detected = _parse_inference_log(log_path)
        dur = probe["duration_s"]
        eff_fps = round(dur / elapsed, 3) if elapsed > 0 else 0.0
        det_ratio = round(frames_detected / sampled_frames, 4) if sampled_frames > 0 else 0.0

        status = "ok" if rc == 0 and json_valid else "failed"
        logger.info(
            "  status=%s rc=%s runtime=%.2fs valid=%s plate=%s sampled=%d detected=%d",
            status, rc, elapsed, json_valid, plate, sampled_frames, frames_detected,
        )

        summary_rows.append({
            "video_id": normalized.name,
            "status": status,
            "resolution": f"{probe['width']}x{probe['height']}",
            "source_fps": probe["source_fps"],
            "frames": probe["frame_count"],
            "duration_s": dur,
            "runtime_s": round(elapsed, 3),
            "effective_video_fps": eff_fps,
            "processed_sampled_frames": sampled_frames,
            "frames_with_plate_detection": frames_detected,
            "plate_detection_frame_ratio": det_ratio,
            "json_valid": json_valid,
            "return_code": rc,
            "plate_output": plate,
            "vehicle_confidence": conf,
            "tespitler_count": n_det,
            "output_json": str(output_json),
            "log": str(log_path),
        })

    meta_csv = output_dir / "video_metadata.csv"
    if metadata_rows:
        with meta_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
        logger.info("Video metadata: %s", meta_csv)

    summary_csv = output_dir / "summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Summary: %s", summary_csv)

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    total = len(summary_rows)
    print(f"\nResult: {ok_count}/{total} videos passed.\n")
    for row in summary_rows:
        print(f"  {row['video_id']}: {row['status']}  plate={row.get('plate_output', '')}  runtime={row.get('runtime_s', '')}s")

    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
