# TEKNOFEST 2026 — 5G & AI Smart Road Safety

## Project Title
**TEKNOFEST 2026 — 5G & AI Smart Road Safety (Phase 1)**

## Description
This project is developed for the TEKNOFEST 2026 "5G & AI Smart Road Safety" competition.  
**Phase 1** focuses on building an offline AI pipeline that processes video files to detect road-safety targets:

- 🚗 **Vehicle Detection** — YOLO-based object detection
- 🔡 **License Plate Recognition** — Detection + OCR pipeline
- 🧠 **Driver Behavior Analysis** — Phone use, smoking, drowsiness, seatbelt detection
- 📊 **Evaluation** — Precision, recall, IoU metrics

> No mobile app or 5G integration in this phase.

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
├── configs/default.yaml        # Model & pipeline configuration
├── data/
│   ├── raw/                    # Competition video files (gitignored)
│   ├── frames/                 # Extracted frames (gitignored)
│   ├── labels/                 # Ground truth labels (gitignored)
│   └── README.md
├── models/                     # Trained weights (gitignored)
├── notebooks/                  # Exploratory Jupyter notebooks
├── src/
│   ├── detection/              # Vehicle detection module
│   ├── plate/                  # Plate recognition module
│   ├── behavior/               # Driver behavior analysis
│   ├── evaluation/             # Metrics
│   ├── utils/                  # Video, visualisation, logging helpers
│   └── pipeline.py             # End-to-end pipeline
├── scripts/                    # CLI entry points
├── reports/                    # PDR drafts
├── outputs/                    # Results, logs, visualisations (gitignored)
└── tests/                      # Unit tests
```

---

## Usage

### 1. Extract frames from a video
```bash
python scripts/extract_frames.py --video data/raw/sample.mp4 --output data/frames/sample --interval 1
```

### 2. Run the full detection pipeline
```bash
python scripts/run_detection.py --video data/raw/sample.mp4 --config configs/default.yaml
```

### 3. Evaluate against ground truth
```bash
python scripts/evaluate.py --predictions outputs/predictions.json --labels data/labels/sample.json
```

### 4. Run unit tests
```bash
pytest tests/
```

---

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_dataset_exploration.ipynb` | Explore video data and class distributions |
| 02 | `02_vehicle_detection.ipynb` | Prototype vehicle detection |
| 03 | `03_plate_recognition.ipynb` | Prototype plate OCR |
| 04 | `04_driver_behavior.ipynb` | Prototype behavior analysis |
| 05 | `05_evaluation.ipynb` | Compute evaluation metrics |

---

## Team Members
| Name | Role |
|------|------|
| *(placeholder)* | Team Lead |
| *(placeholder)* | ML Engineer |
| *(placeholder)* | CV Engineer |
| *(placeholder)* | Backend Developer |

---

## License
This project is developed solely for the TEKNOFEST 2026 competition.