"""Model A wrapper — COCO pretrained YOLOv8 mapped to competition vehicle labels.

COCO class → competition label:
  car (2)   → sedan
  bus (5)   → minibus
  truck (7) → kamyon

Fine-tuned model A weights are not yet available; this module uses pretrained
YOLOv8m COCO weights as a quality baseline until training is complete.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_COCO_TO_COMPETITION: dict[str, str] = {
    "car": "sedan",
    "bus": "minibus",
    "truck": "kamyon",
}

_DEFAULT_VEHICLE_TYPE = "sedan"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class ModelADetector:
    """Detect vehicle type using COCO pretrained YOLOv8m."""

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

    def detect_vehicle_type(self, frame: np.ndarray) -> tuple[str, float]:
        """Return (vehicle_type_label, confidence).

        Falls back to ('sedan', 0.01) when no vehicle is detected or model unavailable.
        """
        if self._model is None or frame is None or frame.size == 0:
            return _DEFAULT_VEHICLE_TYPE, 0.01

        try:
            results = self._model(frame, conf=self._confidence, verbose=False, device=self._device)
        except Exception as exc:
            logger.debug("ModelADetector: inference failed (%s)", exc)
            return _DEFAULT_VEHICLE_TYPE, 0.01

        best_label = _DEFAULT_VEHICLE_TYPE
        best_conf = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = result.names.get(cls_id, "")
                comp_label = _COCO_TO_COMPETITION.get(cls_name)
                if comp_label is None:
                    continue
                conf = float(box.conf[0].item())
                if conf > best_conf:
                    best_conf = conf
                    best_label = comp_label

        return best_label, max(best_conf, 0.01)
