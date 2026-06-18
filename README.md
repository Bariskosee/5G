# TEKNOFEST 2026 - 5G & AI Smart Road Safety

## Project Title
**TEKNOFEST 2026 - 5G & AI Smart Road Safety**

## Description
This project is developed for the TEKNOFEST 2026 "5G & AI Smart Road Safety" competition.  
The repository is organized for the FTR inference architecture: video/frame utilities, vehicle detection, OCR, landmark detection, tracking, color inference, ROI processing, and final output validation.

- **Vehicle Detection** - model-specific vehicle/object detection components
- **OCR** - text extraction components for plate and scene text outputs
- **Landmark** - landmark/keypoint inference components
- **Tracking** - temporal association across frames
- **Color** - vehicle color inference components
- **ROI** - region-of-interest extraction and filtering
- **Output** - final result formatting and schema validation

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
├── datasets/                   # Competition datasets and local inputs (gitignored)
├── docs/                       # Project documentation
├── models/                     # Trained weights (gitignored)
├── notebooks/                  # Exploratory Jupyter notebooks
├── outputs/                    # Runtime outputs, logs, and visualisations (gitignored)
├── reports/                    # Reports and submission drafts
├── scripts/                    # Repository validation and utility entry points
├── src/
│   ├── detection/              # Vehicle detection module
│   ├── ocr/                    # OCR module
│   ├── landmark/               # Landmark inference module
│   ├── tracking/               # Object tracking module
│   ├── color/                  # Color inference module
│   ├── roi/                    # Region-of-interest utilities
│   ├── output/                 # Result formatting module
│   ├── utils/                  # Video, visualisation, logging helpers
│   ├── pipeline.py             # Pipeline orchestration
│   └── predict.py              # Inference entry point
└── tests/                      # Unit tests
```

---

## Usage

### Dataset Preparation Pipeline

Audit a YOLO-format dataset before training:
```bash
python scripts/audit_yolo_dataset.py --data datasets/processed/model_b_plate/data.yaml
```

Print dataset statistics and low-count classes:
```bash
python scripts/dataset_stats.py --data datasets/processed/model_a_unified/data.yaml --min-count 100
```

Remap source YOLO classes into the project target classes:
```bash
python scripts/remap_yolo_labels.py --data datasets/raw/some_dataset/data.yaml --output datasets/processed/model_a_unified --mapping configs/remap_driver_behavior.yaml
```

Create a small inspection sample:
```bash
python scripts/sample_yolo_dataset.py --data datasets/raw/some_dataset/data.yaml --output datasets/samples/some_dataset --n 100
```

### Validate a results JSON fixture
```bash
python scripts/validate_results_json.py tests/fixtures/dummy_results.json
```

### Run unit tests
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
