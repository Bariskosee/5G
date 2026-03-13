"""Vehicle detection module using a pretrained YOLO model.

This module provides the :class:`VehicleDetector` class which wraps an
Ultralytics YOLO model and exposes a simple ``detect()`` interface.

Example::

    from src.detection.vehicle_detector import VehicleDetector

    detector = VehicleDetector(confidence=0.5)
    detections = detector.detect(frame)          # frame is a numpy BGR array
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO class names for the vehicle categories we care about
_VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Detect vehicles in a single video frame using YOLO.

    Parameters
    ----------
    model_path:
        Path to a ``*.pt`` / ``*.onnx`` YOLO weights file.  When *None* the
        default YOLOv8n pretrained weights are used (downloaded on first run).
    confidence:
        Minimum confidence threshold for returned detections.
    device:
        Inference device, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.confidence = confidence
        self.device = device
        self._model: Any = None
        self._model_path = model_path

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the YOLO model.  Falls back to YOLOv8n if no path given."""
        try:
            from ultralytics import YOLO  # type: ignore[import]

            weights = str(self._model_path) if self._model_path else "yolov8n.pt"
            self._model = YOLO(weights)
            self._model.to(self.device)
            logger.info("VehicleDetector: loaded model from '%s'", weights)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "VehicleDetector: could not load YOLO model (%s). "
                "Running in placeholder mode — detect() will return [].",
                exc,
            )
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run vehicle detection on a single BGR frame.

        Parameters
        ----------
        frame:
            A BGR image as a NumPy ``uint8`` array with shape ``(H, W, 3)``.

        Returns
        -------
        list[dict]
            Each entry is::

                {
                    "bbox":       [x1, y1, x2, y2],   # pixel coordinates
                    "confidence": float,
                    "class":      str,                 # e.g. "car"
                }

            Returns an empty list when no vehicle is detected or the model
            is not available.
        """
        if self._model is None:
            logger.debug("VehicleDetector: model not loaded; returning []")
            return []

        results = self._model(
            frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in _VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class": _VEHICLE_CLASSES[cls_id],
                    }
                )

        return detections
