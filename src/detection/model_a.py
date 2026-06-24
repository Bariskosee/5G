"""Model A wrapper — COCO pretrained YOLOv8 mapped to competition vehicle labels.

COCO class → competition label:
  car (2)        → sedan
  bus (5)        → minibus
  truck (7)      → kamyon
  cell phone (67)→ telefonla_konusma  (sofor_eylemi, requires driver ROI check)
  bottle (39)    → su_icme            (sofor_eylemi, requires driver ROI check)
  laptop (63)    → bilgisayar         (nesneler)
  person (0)     → yolcular source    (ROI assignment, not emitted directly)

Fine-tuned model A weights are not yet available; this module uses pretrained
YOLOv8m COCO weights as a quality baseline until training is complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_COCO_TO_COMPETITION: dict[str, str] = {
    "car": "sedan",
    "bus": "minibus",
    "truck": "kamyon",
}

# COCO driver-object classes yielding sofor_eylemi or nesneler events
_COCO_DRIVER_OBJECTS: set[str] = {"cell phone", "bottle", "laptop"}

_DEFAULT_VEHICLE_TYPE = "sedan"


@dataclass
class ModelADetections:
    """All detections from a single YOLO pass on one frame."""

    vehicle_type: str = _DEFAULT_VEHICLE_TYPE
    vehicle_conf: float = 0.01
    vehicle_bbox: tuple[int, int, int, int] | None = None
    # List of (coco_class_name, confidence, xyxy) for driver-relevant objects
    driver_objects: list[tuple[str, float, tuple[int, int, int, int]]] = field(
        default_factory=list
    )
    # List of (xyxy, confidence) for all detected persons
    person_bboxes: list[tuple[tuple[int, int, int, int], float]] = field(
        default_factory=list
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class ModelADetector:
    """Detect vehicle type and driver-relevant objects using COCO pretrained YOLOv8m."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.45,
        device: str = "auto",
    ) -> None:
        self._confidence = confidence
        self._device = _resolve_device(device)
        self._model: Any = None
        self._load(model_path)

    def _load(self, model_path: str | Path | None) -> None:
        try:
            from ultralytics import YOLO

            weights = str(model_path) if model_path else "yolov8m.pt"
            self._model = YOLO(weights)
            self._model.to(self._device)
            logger.info("ModelADetector: loaded '%s' on %s", weights, self._device)
        except Exception as exc:
            logger.warning(
                "ModelADetector: could not load model (%s) — vehicle type will default to '%s'.",
                exc,
                _DEFAULT_VEHICLE_TYPE,
            )
            self._model = None

    # ------------------------------------------------------------------
    # Primary API: full single-pass detection
    # ------------------------------------------------------------------

    def detect_all(self, frame: np.ndarray) -> ModelADetections:
        """Run a single YOLO pass and return vehicle type, driver objects, and persons."""
        result = ModelADetections()

        if self._model is None or frame is None or frame.size == 0:
            return result

        try:
            yolo_results = self._model(
                frame, conf=self._confidence, verbose=False, device=self._device
            )
        except Exception as exc:
            logger.debug("ModelADetector: inference failed (%s)", exc)
            return result

        best_vehicle_conf = 0.0

        for res in yolo_results:
            boxes = res.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name: str = res.names.get(cls_id, "")
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                xyxy: tuple[int, int, int, int] = (x1, y1, x2, y2)

                # Vehicle type
                comp_label = _COCO_TO_COMPETITION.get(cls_name)
                if comp_label is not None:
                    if conf > best_vehicle_conf:
                        best_vehicle_conf = conf
                        result.vehicle_type = comp_label
                        result.vehicle_conf = max(conf, 0.01)
                        result.vehicle_bbox = xyxy
                    continue

                # Driver-relevant objects
                if cls_name in _COCO_DRIVER_OBJECTS:
                    result.driver_objects.append((cls_name, conf, xyxy))
                    continue

                # Persons
                if cls_name == "person":
                    result.person_bboxes.append((xyxy, conf))

        return result

    # ------------------------------------------------------------------
    # Backward-compatible API (used by older code paths)
    # ------------------------------------------------------------------

    def detect_vehicle_type(self, frame: np.ndarray) -> tuple[str, float]:
        """Return (vehicle_type_label, confidence).

        Falls back to ('sedan', 0.01) when no vehicle is detected or model unavailable.
        """
        det = self.detect_all(frame)
        return det.vehicle_type, det.vehicle_conf
