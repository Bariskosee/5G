# TEKNOFEST 2026 — Penta Tech Deliverables Package

**Competition:** TEKNOFEST 2026 — 5G & Yapay Zeka ile Akilli Yol Guvenligi Yarisması  
**Team:** Penta Tech  
**Phase:** Phase 2 — Final Tasarim Raporu (FTR), deadline 28 June 2026  

---

## Directory Structure

```
deliverables/
├── models/
│   ├── plate_best.pt       — Trained YOLOv8s plate detector
│   └── vehicle_best.pt     — YOLOv8m COCO pretrained (vehicle type backbone)
├── configs/
│   ├── vehicle_data.yaml   — Vehicle type dataset config
│   ├── plate_data.yaml     — Plate detection dataset config
│   ├── model_a_classes.yaml            — Unified detector (14 classes)
│   ├── model_a_vehicle_type_classes.yaml — Vehicle-type-only (7 classes)
│   ├── model_b_classes.yaml            — Plate detector (1 class)
│   ├── final_label_mapping.yaml        — Model class → competition JSON label
│   ├── thresholds.yaml                 — All runtime parameters
│   └── class_lists.md                  — Human-readable class reference
└── results/
    ├── plate/
    │   ├── results.csv      — Per-epoch training metrics (24 epochs)
    │   ├── results.png      — Training overview curves
    │   ├── P_curve.png      — Precision curve
    │   ├── R_curve.png      — Recall curve
    │   ├── PR_curve.png     — mAP50 + mAP50-95 curves
    │   ├── F1_curve.png     — F1 score curve
    │   └── examples/        — 10 annotated inference images
    ├── vehicle/
    │   └── examples/        — 10 annotated inference images (COCO classes)
    └── behavior/            — No .pt model (MediaPipe + rule-based)
```

---

## Model Summary

### Model B — Plate Detector (`plate_best.pt`)

| Property        | Value                                         |
|-----------------|-----------------------------------------------|
| File            | `deliverables/models/plate_best.pt`           |
| Architecture    | YOLOv8s                                       |
| Task            | Object detection                              |
| Classes         | 1 — `license_plate`                           |
| Image size      | 640 × 640                                     |
| Training epochs | 24 (early-stopped at patience=8 / max 40)     |
| Batch size      | 16                                            |
| Optimizer       | Auto (SGD/Adam via Ultralytics auto)          |
| Pretrained from | `yolov8s.pt` (COCO)                          |
| Trained on      | Turkish Number Plates (full split, 1 class)   |
| Training date   | 2026-06-18                                    |
| Device          | NVIDIA GPU (device=0)                         |
| Ultralytics ver | 8.4.70                                        |

**Best validation metrics (epoch 16):**

| Metric         | Value   |
|----------------|---------|
| Precision      | 0.9840  |
| Recall         | 0.9583  |
| mAP@50         | 0.9893  |
| mAP@50–95      | 0.8517  |
| Val Box Loss   | 0.5706  |
| Val Cls Loss   | 0.3714  |

**Dataset:** `datasets/processed/model_b_plate/turkish_number_plates_full/`  
**Config:** `deliverables/configs/plate_data.yaml`  
**Training curves:** `deliverables/results/plate/results.png`

---

### Model A (Vehicle Type) — `vehicle_best.pt`

| Property        | Value                                                  |
|-----------------|--------------------------------------------------------|
| File            | `deliverables/models/vehicle_best.pt`                  |
| Architecture    | YOLOv8m                                                |
| Task            | Object detection                                       |
| Classes (total) | 80 (COCO) — 3 used at runtime: car, bus, truck         |
| Image size      | 640 × 640                                              |
| Training        | COCO pretrained — no project fine-tuning yet           |
| Remapping       | car→sedan, bus→minibus, truck→kamyon                   |
| Planned         | Fine-tune on `model_a_vehicle_types` (7 classes) if time permits |

**Note:** This is the COCO pretrained baseline. The fine-tuned version
(`model_a_best.pt`, 7 competition-specific classes) is planned for
training on `datasets/processed/model_a_vehicle_types/`. Until then,
COCO remapping provides functional vehicle-type detection.

**Dataset (for planned fine-tuning):** `datasets/processed/model_a_vehicle_types/`  
**Config:** `deliverables/configs/vehicle_data.yaml`  
**Metrics:** Not available (pretrained COCO baseline, no project training run)

---

### Behavior Detection — No `.pt` Model

| Property   | Value                                              |
|------------|----------------------------------------------------|
| File       | `models/mediapipe/face_landmarker.task` (3.6 MB)   |
| Framework  | MediaPipe FaceLandmarker                           |
| Detections | esneme (yawning), arkaya_bakma (looking back), etrafa_bakinma (glancing around) |
| + YOLO     | phone, cigarette, drink, no_seatbelt (from Model A) |
| + Tracker  | slalom (vehicle bbox temporal oscillation)         |

Behavior detection is intentionally not a trained YOLO model — data for
these classes is too scarce for reliable training. MediaPipe provides
landmark-based detection that generalizes without labeled examples.

---

## Dataset Information

| Dataset                                   | Classes | Split   | License    |
|-------------------------------------------|---------|---------|------------|
| Turkish Number Plates (full)              | 1       | train/val/test | Roboflow |
| Turkish Number Plates (100-sample subset) | 1       | train/val/test | Roboflow |
| Vehicle Classification V2 (Roboflow)     | 21      | train/val/test | CC BY 4.0  |
| Vehicle Classification bbox              | 21      | train   | —          |
| Model A Vehicle Types (processed)        | 7       | train/val/test | Various    |

---

## Inference Examples

### Plate model (`plate_best.pt`) — 10 annotated images
Location: `deliverables/results/plate/examples/`  
Images selected randomly from test split (seed=42).  
Each image shows detected `license_plate` bounding boxes with confidence scores.

### Vehicle model (`vehicle_best.pt`) — 10 annotated images
Location: `deliverables/results/vehicle/examples/`  
Images selected randomly from vehicle type test split (seed=42).  
Detections shown for COCO classes: car, bus, truck.

### Behavior model — No inference images
MediaPipe-based detection requires video frames with visible driver faces.
See `docs/FTR_TESTING.md` for the full video test workflow.

---

## How to Reproduce Inference

```bash
# Plate detection
python -c "
from ultralytics import YOLO
model = YOLO('deliverables/models/plate_best.pt')
model.predict('path/to/image.jpg', conf=0.25, save=True)
"

# Vehicle type detection
python -c "
from ultralytics import YOLO
model = YOLO('deliverables/models/vehicle_best.pt')
model.predict('path/to/image.jpg', conf=0.45, classes=[2,5,7], save=True)
"

# Full pipeline (video → results.json)
python main.py \
  --input /app/data/input/video.mp4 \
  --output /app/data/output/results.json \
  --plate-model models/model_b_plate/best.pt

# Validate output
python scripts/validate_results_json.py outputs/results.json
```

---

## Missing / Not Yet Available

| Artifact               | Status                                              |
|------------------------|-----------------------------------------------------|
| `behavior_best.pt`     | Not applicable — MediaPipe, no .pt file             |
| `model_a_best.pt`      | Not yet trained (fine-tuning planned post-FTR)      |
| Confusion matrices     | Not generated (no local training run recorded)      |
| `labels.jpg`           | Not generated (training was done on Colab/Drive)    |
| Behavior training metrics | N/A — rule-based, no gradient-based training     |

---

*Generated: 2026-06-28 | Penta Tech — TEKNOFEST 2026 FTR Package*
