# TEKNOFEST 2026 — 5G & AI Smart Road Safety

## Project Title
**TEKNOFEST 2026 — 5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması**

## Description
AI pipeline for the TEKNOFEST 2026 road safety competition. Processes offline video
files and produces a competition-schema `results.json` for the FTR (Final Design Report)
auto-grader.

---

## Current Completed Scope

| Component | Status |
|---|---|
| Model B — YOLOv8s license plate bbox detector | ✅ Trained, local weights available |
| FTR-compatible inference skeleton (`main.py`) | ✅ Implemented |
| Schema-valid `results.json` output | ✅ Tested with 3 local videos |
| JSON schema validator (`scripts/validate_results_json.py`) | ✅ |
| Local 4K/50 FPS video smoke test | ✅ All 3 videos pass |

## Pending Modules

| Module | Notes |
|---|---|
| Model A — vehicle type, driver behaviors, objects | Not yet trained |
| EasyOCR final packaging | Detector bbox works; plate string requires OCR model files baked into Docker |
| Vehicle color inference (HSV+Lab) | Not yet implemented |
| Driver action landmarks (MediaPipe) | Not yet implemented |
| Passenger seat ROI | Not yet implemented |
| Slalom tracking | Not yet implemented |
| T4/Linux x86_64 Docker final validation | Mac ARM64 build expected to fail (see Docker section) |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Bariskosee/5G.git
cd 5G

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) install the package in editable mode
pip install -e .
```

---

## Project Structure

```
├── configs/                    # Model class names, thresholds, label mappings
├── datasets/                   # Competition datasets (gitignored)
├── docs/                       # Project documentation
├── models/                     # Trained weights (gitignored — see note below)
├── notebooks/                  # Exploratory Jupyter notebooks
├── outputs/                    # Runtime outputs (gitignored)
├── reports/                    # Reports and submission drafts
├── scripts/                    # Validation, testing, and utility entry points
├── src/
│   ├── detection/              # YOLO plate and vehicle detector wrappers
│   ├── ocr/                    # EasyOCR wrapper (best-effort, non-fatal)
│   ├── landmark/               # MediaPipe face landmark module (stub)
│   ├── tracking/               # Object tracking module (stub)
│   ├── color/                  # Color inference module (stub)
│   ├── roi/                    # Region-of-interest utilities (stub)
│   ├── output/                 # Result schema, builder, and writer
│   ├── utils/                  # Video iteration, plate normalization, logging
│   ├── pipeline.py             # Pipeline orchestration (stub — not called by main.py)
│   └── predict.py              # Legacy placeholder (stub — not called by main.py)
├── main.py                     # FTR Docker entry point (CMD target)
└── tests/                      # Unit tests
```

> **Note:** `main.py` is the Docker `CMD` target and the active FTR inference entry
> point. `src/predict.py` and `src/pipeline.py` are stubs for future full-pipeline
> integration and are **not called by `main.py`**.

---

## Model B — Plate Detector Metrics

Model B is a YOLOv8s license plate bbox detector trained on Turkish plate images.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation | 0.984 | 0.958 | 0.989 | 0.851 |
| Test | 0.973 | 0.971 | 0.992 | 0.851 |

- Weights: `models/model_b_plate/best.pt` (local only, gitignored)
- 1 class: `license_plate`

---

## FTR Inference Skeleton

`main.py` is the active inference entry point. Run local inference with the trained
plate detector:

```bash
python main.py \
  --input /path/to/video.mp4 \
  --output outputs/results.json \
  --plate-model models/model_b_plate/best.pt \
  --frame-stride 10
```

| Argument | Default (Docker) | Notes |
|---|---|---|
| `--input` | `/app/data/input/video.mp4` | Input video path |
| `--output` | `/app/data/output/results.json` | Output JSON path |
| `--plate-model` | `/app/models/model_b_plate/best.pt` | YOLO plate model |
| `--frame-stride` | `10` | Sample 1 frame out of N |
| `--max-frames` | `None` | Optional frame cap |
| `--conf-threshold` | `0.25` | Plate detection threshold |
| `--disable-ocr` | `False` | Skip OCR, use fallback plate |

When EasyOCR model files are not packaged, inference continues and writes a
schema-valid fallback:

```json
"plaka": "tespit_edilemedi"
```

with `confidence_score: 0.01`. This is a temporary fallback — final scoring-oriented
OCR should produce a normalized Turkish plate string when available.

---

## Local FTR Video Test Results

Tested on 3 local 4K/50 FPS videos (`--frame-stride 10`, OCR disabled):

| video_id | Resolution | Source FPS | Duration | Runtime | Plate Frames | JSON Valid |
|---|---|---|---|---|---|---|
| video_1.mp4 | 3840×2160 | 50 | 8.46 s | 6.3 s | 21 / 43 | ✅ |
| video_2.mp4 | 3840×2160 | 50 | 9.14 s | 6.4 s | 21 / 46 | ✅ |
| video_3.mp4 | 3840×2160 | 50 | 7.66 s | 5.8 s | 9 / 39 | ✅ |

Run reproducibly with:

```bash
python scripts/run_ftr_video_tests.py \
  --input-dir /path/to/videos \
  --output-dir /tmp/5g_ftr_outputs \
  --plate-model models/model_b_plate/best.pt \
  --frame-stride 10 \
  --disable-ocr
```

---

## Docker

The FTR Docker entry point defaults are:

| Path | Value |
|---|---|
| Input video | `/app/data/input/video.mp4` |
| Output JSON | `/app/data/output/results.json` |
| Model weights | `/app/models/model_b_plate/best.pt` |

**Prerequisites:** `models/model_b_plate/best.pt` must exist locally before building.
The `.dockerignore` intentionally does **not** exclude `models/` so that `best.pt` is
copied into the image during `docker build` even though it is gitignored.

**Build (Linux x86_64 with NVIDIA GPU):**
```bash
docker build -t teknofest/5g-road-safety:latest .
```

**Run:**
```bash
docker run --rm --gpus all \
  -v /absolute/path/to/video.mp4:/app/data/input/video.mp4 \
  -v /absolute/path/to/output:/app/data/output \
  teknofest/5g-road-safety:latest
```

### Mac ARM64 Limitation

The Dockerfile targets `nvidia/cuda:12.1.0-base-ubuntu22.04` with CUDA 12.1-compatible
PyTorch wheels from `download.pytorch.org/whl/cu121`. These wheels are only available
for Linux x86_64. **Mac ARM64 Docker builds are expected to fail** with:

```
ERROR: Could not find a version that satisfies the requirement torch==2.3.1
```

This is a local development limitation only. The official competition grader runs on
**Linux x86_64 + NVIDIA Tesla T4**, where the Dockerfile works correctly.
Final Docker validation must be performed in a real Linux x86_64 + NVIDIA GPU environment.

Run static Docker packaging checks (no Docker daemon required):

```bash
python scripts/check_docker_packaging.py
```

---

## Dataset Preparation Pipeline

```bash
# Audit a YOLO-format dataset before training
python scripts/audit_yolo_dataset.py --data datasets/processed/model_b_plate/data.yaml

# Print class statistics
python scripts/dataset_stats.py \
  --data datasets/processed/model_a_unified/data.yaml --min-count 100

# Remap source classes to project target classes
python scripts/remap_yolo_labels.py \
  --data datasets/raw/some_dataset/data.yaml \
  --output datasets/processed/model_a_unified \
  --mapping configs/remap_driver_behavior.yaml

# Create a small inspection sample
python scripts/sample_yolo_dataset.py \
  --data datasets/raw/some_dataset/data.yaml \
  --output datasets/samples/some_dataset \
  --n 100
```

---

## Validate and Test

```bash
# Validate a results.json against the competition schema
python scripts/validate_results_json.py outputs/results.json

# Run unit tests
pytest tests/

# Generate YOLO plate detection visual evidence (annotated videos + labels)
python scripts/generate_plate_visual_evidence.py \
  --input-dir /tmp/5g_ftr_videos \
  --output-dir /tmp/5g_ftr_outputs/yolo_visual \
  --model models/model_b_plate/best.pt

# Collect FTR report evidence into one folder
python scripts/collect_ftr_evidence.py \
  --test-output-dir /tmp/5g_ftr_outputs \
  --evidence-dir /tmp/5g_ftr_report_evidence
```

See [docs/FTR_TESTING.md](docs/FTR_TESTING.md) for the full testing workflow.

---

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Explore video data and class distributions |
| 02 | `02_vehicle_detection.ipynb` | Prototype vehicle detection |
| 03 | `03_plate_recognition.ipynb` | Prototype plate OCR |
| 04 | `04_driver_behavior.ipynb` | Prototype behavior analysis |
| 05 | `05_evaluation.ipynb` | Compute evaluation metrics |

---

## Team Members

| Name | Role |
|---|---|
| *(placeholder)* | Team Lead |
| *(placeholder)* | ML Engineer |
| *(placeholder)* | CV Engineer |
| *(placeholder)* | Backend Developer |

---

## License

Developed solely for the TEKNOFEST 2026 competition.
