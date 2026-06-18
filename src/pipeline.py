"""Orchestrates one video → results.json (FTR pipeline).

Per CLAUDE.md §4, the pipeline runs in this order for each sampled frame:
  1. Frame sampling (every N frames via src.utils.video)
  2. Model A (YOLOv8m) — vehicle types, driver objects, general objects, person
  3. Model B (YOLOv8s) — license_plate bbox → lazy EasyOCR → plate text
  4. MediaPipe FaceLandmarker — esneme, arkaya_bakma, etrafa_bakinma
  5. Vehicle bbox tracker — slalom detection
  6. HSV+Lab color analyzer — arac_bilgisi.renk
  7. ROI mapper — on_koltuk / arka_koltuk_1 / arka_koltuk_2

After all frames: aggregation (src.output.aggregator) → formatter
(src.output.formatter) → writes /app/data/output/results.json.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "src/pipeline.py is a placeholder. Implement Pipeline before running."
    )
