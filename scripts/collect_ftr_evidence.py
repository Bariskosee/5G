"""Collect FTR report evidence artifacts into a single local folder.

Copies summary.csv, video_metadata.csv, per-video results.json, and
visual_summary.csv. Annotated videos are listed in evidence_manifest.txt
(not copied by default — use --include-annotated-videos to copy them).

Usage
-----
python scripts/collect_ftr_evidence.py \\
  --test-output-dir /tmp/5g_ftr_outputs \\
  --evidence-dir /tmp/5g_ftr_report_evidence

python scripts/collect_ftr_evidence.py \\
  --test-output-dir /tmp/5g_ftr_outputs \\
  --evidence-dir /tmp/5g_ftr_report_evidence \\
  --include-annotated-videos
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        logger.warning("Missing: %s", src)
        return False
    shutil.copy2(src, dst)
    logger.info("Copied: %s → %s", src.name, dst)
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-output-dir", required=True, help="Root output dir from run_ftr_video_tests.py")
    parser.add_argument("--evidence-dir", required=True, help="Destination folder for evidence artifacts")
    parser.add_argument("--include-annotated-videos", action="store_true", help="Also copy annotated .mp4 files")
    args = parser.parse_args(argv)

    test_out = Path(args.test_output_dir).resolve()
    evidence = Path(args.evidence_dir).resolve()
    evidence.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for name in ("summary.csv", "video_metadata.csv"):
        src = test_out / name
        if _copy(src, evidence / name):
            copied.append(name)
        else:
            missing.append(name)

    visual_summary = test_out / "yolo_visual" / "visual_summary.csv"
    if _copy(visual_summary, evidence / "visual_summary.csv"):
        copied.append("visual_summary.csv")
    else:
        missing.append("visual_summary.csv (expected at yolo_visual/visual_summary.csv)")

    for results_json in sorted(test_out.glob("*/results.json")):
        stem = results_json.parent.name
        dest = evidence / f"{stem}_results.json"
        if _copy(results_json, dest):
            copied.append(dest.name)
        else:
            missing.append(str(results_json))

    manifest_lines: list[str] = ["# FTR Evidence Manifest\n"]
    manifest_lines.append("## Copied files\n")
    manifest_lines.extend(f"  {c}\n" for c in copied)

    annotated_videos = sorted(test_out.glob("yolo_visual/*/*.mp4"))
    if annotated_videos:
        manifest_lines.append("\n## Annotated video paths (YOLO visual evidence)\n")
        for v in annotated_videos:
            if args.include_annotated_videos:
                dest = evidence / f"annotated_{v.name}"
                if _copy(v, dest):
                    copied.append(dest.name)
                manifest_lines.append(f"  COPIED → {dest}\n")
            else:
                manifest_lines.append(f"  {v}\n")

    if missing:
        manifest_lines.append("\n## Missing (not found)\n")
        manifest_lines.extend(f"  {m}\n" for m in missing)

    manifest = evidence / "evidence_manifest.txt"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")
    logger.info("Manifest: %s", manifest)

    print(f"\nEvidence collected to: {evidence}")
    print(f"Copied {len(copied)} file(s). Missing: {len(missing)}.")
    if missing:
        print("Missing files (warnings above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
