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
│   ├── vehicle_best.pt     — YOLOv8m COCO pretrained (vehicle type backbone)
│   └── behavior_best.pt    — YOLOv8m COCO pretrained (shared with vehicle; covers person + phone classes)
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
    │   ├── results.csv          — Per-epoch training metrics (24 epochs)
    │   ├── results.png          — Training overview curves
    │   ├── P_curve.png          — Precision curve
    │   ├── R_curve.png          — Recall curve
    │   ├── PR_curve.png         — mAP50 + mAP50-95 curves
    │   ├── F1_curve.png         — F1 score curve
    │   ├── confusion_matrix.png — Val-set confusion matrix (100 images)
    │   ├── labels.jpg           — Val-set label distribution (408 instances)
    │   └── examples/            — 10 annotated inference images
    ├── vehicle/
    │   ├── confusion_matrix.png — Val-set confusion matrix (80 images, COCO remap)
    │   ├── labels.jpg           — Val-set label distribution (242 instances)
    │   └── examples/            — 10 annotated inference images
    └── behavior/
        └── examples/            — 10 annotated images (person + cell phone, street scenes)
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

### Model A (Behavior) — `behavior_best.pt`

| Property        | Value                                                          |
|-----------------|----------------------------------------------------------------|
| File            | `deliverables/models/behavior_best.pt`                         |
| Architecture    | YOLOv8m (same weights as `vehicle_best.pt`)                    |
| Task            | Object detection — COCO pretrained baseline                    |
| COCO classes used | 0 (person), 67 (cell phone)                                 |
| Project classes | phone→telefonla_konusma, person→passenger ROI assignment       |
| Image size      | 640 × 640                                                      |
| Training        | COCO pretrained — unified fine-tune (model_a_best.pt) planned  |
| Metrics         | No project training run yet (COCO pretrained baseline)         |

**Additional behavior pipeline (non-YOLO):**

| Property   | Value                                              |
|------------|----------------------------------------------------|
| File       | `models/mediapipe/face_landmarker.task` (3.6 MB)   |
| Framework  | MediaPipe FaceLandmarker                           |
| Detections | esneme (yawning), arkaya_bakma, etrafa_bakinma     |
| + Tracker  | slalom (vehicle bbox temporal oscillation)         |

The full unified Model A (phone + cigarette + drink + no_seatbelt + person)
will be fine-tuned once the `model_a_unified` dataset is assembled. Until
then, COCO pretrained covers person and phone at minimum.

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

| Artifact                    | Status                                                      |
|-----------------------------|-------------------------------------------------------------|
| `model_a_best.pt`           | Not yet trained — fine-tuning on 7-class vehicle dataset planned |
| Behavior YOLO training metrics | N/A — COCO pretrained baseline used; fine-tune TBD      |
| Behavior confusion matrix   | N/A — behavior YOLO not yet fine-tuned; MediaPipe non-YOLO |
| `behavior_best.pt` (custom) | Planned: unified Model A fine-tune after FTR submission     |

---

*Generated: 2026-06-28 | Penta Tech — TEKNOFEST 2026 FTR Package*
