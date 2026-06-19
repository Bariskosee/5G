# FTR Testing Workflow

Operational guide for running local video tests, generating evidence, and
verifying Docker packaging before the FTR submission.

---

## 1. Completed Scope

| Component | Status |
|---|---|
| Model B — YOLOv8s license plate bbox detector | Trained, weights at `models/model_b_plate/best.pt` |
| FTR inference skeleton (`main.py`) | Implemented, schema-valid output |
| Local 4K/50 FPS video smoke test | 3 videos pass |
| OCR wrapper (EasyOCR) | Non-fatal fallback; final packaging pending |

Pending modules: vehicle type, color, driver actions, passengers, slalom,
MediaPipe landmarks, T4/Linux Docker final validation.

---

## 2. Local Video Test

Runs `main.py` on every video in a directory, validates each `results.json`,
and writes `summary.csv` + `video_metadata.csv`.

```bash
python scripts/run_ftr_video_tests.py \
  --input-dir /path/to/ftr/videos \
  --output-dir /tmp/5g_ftr_outputs \
  --work-dir /tmp/5g_ftr_videos \
  --plate-model models/model_b_plate/best.pt \
  --frame-stride 10 \
  --disable-ocr \
  --overwrite
```

> Replace `/path/to/ftr/videos` with the actual path to your FTR smoke-test videos.
> The Turkish Number Plates training dataset (`datasets/raw/turkish_number_plates/`) is for
> Model B training and is **not** the FTR smoke-test video source.

**Arguments:**

| Argument | Default | Notes |
|---|---|---|
| `--input-dir` | *(required)* | Directory with video files |
| `--output-dir` | `/tmp/5g_ftr_outputs` | Root for per-video outputs |
| `--work-dir` | `/tmp/5g_ftr_videos` | Temp dir for normalized copies |
| `--plate-model` | `models/model_b_plate/best.pt` | |
| `--frame-stride` | `10` | |
| `--max-frames` | `None` | |
| `--conf-threshold` | `0.25` | |
| `--disable-ocr` | `False` | Skip EasyOCR, use fallback plate |
| `--overwrite` | `False` | Overwrite existing work-dir copies |

**Outputs under `--output-dir`:**
- `video_metadata.csv` — resolution, FPS, frame count, duration for each video
- `summary.csv` — runtime, validity, plate output per video
- `<video_stem>/results.json` — schema-valid output
- `<video_stem>/run.log` — stdout + stderr from `main.py`

> **video_1 note:** The original `video_1` file has no `.mp4` extension but is a valid
> ISO MP4 file. The script copies it as `video_1.mp4` automatically.

---

## 3. YOLO Plate Detection Visual Evidence

Generates annotated videos and per-frame label `.txt` files for the FTR report.

```bash
python scripts/generate_plate_visual_evidence.py \
  --input-dir /tmp/5g_ftr_videos \
  --output-dir /tmp/5g_ftr_outputs/yolo_visual \
  --model models/model_b_plate/best.pt \
  --imgsz 640 \
  --conf 0.25 \
  --vid-stride 10
```

**Outputs under `--output-dir`:**
- `<video_stem>/video_*.mp4` — annotated video with plate bboxes
- `<video_stem>/labels/*.txt` — YOLO format `class cx cy w h conf` per frame
- `visual_summary.csv` — label file count and annotated video path per video

Label format: `0 <cx> <cy> <w> <h> <conf>` (class 0 = license_plate)

---

## 4. Evidence Collection

Copies key artifacts into a single folder for the FTR report.

```bash
python scripts/collect_ftr_evidence.py \
  --test-output-dir /tmp/5g_ftr_outputs \
  --evidence-dir /tmp/5g_ftr_report_evidence
```

With `--include-annotated-videos` the annotated `.mp4` files are also copied.
Default behaviour writes their paths to `evidence_manifest.txt` only (saves disk space).

**Evidence folder contents (default):**
- `summary.csv`
- `video_metadata.csv`
- `visual_summary.csv`
- `video_1_results.json`, `video_2_results.json`, `video_3_results.json`
- `evidence_manifest.txt`

---

## 5. JSON Schema Validation

```bash
python scripts/validate_results_json.py outputs/results.json
```

Expected output: `OK: <path> is valid.`

Every `results.json` must pass this validator before Docker build. The validator
checks exact JSON keys, allowed label values, confidence ranges, and Turkish plate format.

Validate all evidence JSONs at once:

```bash
for f in /tmp/5g_ftr_report_evidence/*_results.json; do
  echo "=== $f ===" && python scripts/validate_results_json.py "$f"
done
```

---

## 6. Docker Packaging

### Official paths (hardcoded by grader)

| Path | Value |
|---|---|
| Input video | `/app/data/input/video.mp4` |
| Output JSON | `/app/data/output/results.json` |
| Model weights | `/app/models/model_b_plate/best.pt` |

### Static packaging check (no Docker daemon needed)

```bash
python scripts/check_docker_packaging.py
```

Verifies:
- Dockerfile exists with correct base image (`nvidia/cuda:12.1.0-base-ubuntu22.04`)
- Dockerfile CMD is `python3 main.py`
- `.dockerignore` does NOT exclude `models/` or `*.pt`
- `models/model_b_plate/best.pt` exists locally
- `best.pt` is gitignored

### Build command (Linux x86_64 only)

```bash
docker build -t teknofest/5g-road-safety:latest .
```

### Run command

```bash
docker run --rm --gpus all \
  -v /absolute/path/to/video.mp4:/app/data/input/video.mp4 \
  -v /absolute/path/to/output:/app/data/output \
  teknofest/5g-road-safety:latest
```

---

## 7. Mac ARM64 Docker Limitation

The Dockerfile installs PyTorch via `--index-url https://download.pytorch.org/whl/cu121`.
This index only provides Linux x86_64 wheels. **Mac ARM64 builds fail** with:

```
ERROR: Could not find a version that satisfies the requirement torch==2.3.1
```

This is expected. The Dockerfile is correct for the competition grader environment:
**Linux x86_64 + NVIDIA Tesla T4 (CUDA 12.1)**.

Final Docker validation must be done on a real Linux x86_64 + NVIDIA GPU machine before
the FTR submission deadline (28 June 2026, 17:00 Turkey time).

---

## 8. FTR Report Evidence Checklist

Items to include in the Final Design Report (§3.3 Çözüm Detayları, §4 Çözümün Sınanması):

| Evidence | Source |
|---|---|
| Model B validation metrics table | From training run (P=0.984, R=0.958, mAP50=0.989) |
| Model B test metrics table | From training run (P=0.973, R=0.971, mAP50=0.992) |
| Video metadata table (resolution, FPS, duration) | `video_metadata.csv` |
| Runtime / effective FPS table | `summary.csv` |
| JSON validation results | stdout of `validate_results_json.py` |
| YOLO annotated frame screenshots | Frames from annotated videos in `yolo_visual/` |
| Sample `results.json` output | Any `*_results.json` in evidence folder |

**Pending items (note in report):**
- Real OCR plate reading (EasyOCR model files packaging)
- Vehicle type, color, driver actions, passengers
- T4/Linux Docker final validation

---

## 9. Quick Commands Reference

```bash
# Pre-flight
python -m compileall src scripts main.py -q
pytest tests/ -q
python scripts/validate_results_json.py tests/fixtures/dummy_results.json

# Docker static check
python scripts/check_docker_packaging.py

# Full local test
python scripts/run_ftr_video_tests.py \
  --input-dir /path/to/ftr/videos \
  --output-dir /tmp/5g_ftr_outputs \
  --plate-model models/model_b_plate/best.pt \
  --frame-stride 10 --disable-ocr --overwrite

# Visual evidence
python scripts/generate_plate_visual_evidence.py \
  --input-dir /tmp/5g_ftr_videos \
  --output-dir /tmp/5g_ftr_outputs/yolo_visual \
  --model models/model_b_plate/best.pt

# Collect evidence
python scripts/collect_ftr_evidence.py \
  --test-output-dir /tmp/5g_ftr_outputs \
  --evidence-dir /tmp/5g_ftr_report_evidence
```
