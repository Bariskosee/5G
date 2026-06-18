"""YOLO-based license plate detector for Model B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.output.schema import clamp_confidence


@dataclass(frozen=True)
class PlateDetection:
    """A single license plate detection in pixel coordinates."""

    xyxy: tuple[int, int, int, int]
    confidence: float


class PlateDetector:
    """Run YOLOv8 license plate detection on video frames."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Plate model not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.conf_threshold = clamp_confidence(conf_threshold)
        self.iou_threshold = clamp_confidence(iou_threshold)
        self._model = self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for PlateDetector") from exc

        model = YOLO(str(self.model_path))
        try:
            model.to(self.device)
        except Exception:
            pass
        return model

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """Detect license plates in one BGR frame."""
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections: list[PlateDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item()) if box.cls is not None else 0
                if class_id != 0:
                    continue

                confidence = clamp_confidence(float(box.conf[0].item()))
                if confidence < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    PlateDetection(
                        xyxy=(x1, y1, x2, y2),
                        confidence=confidence,
                    )
                )

        return sorted(detections, key=lambda item: item.confidence, reverse=True)


def crop_plate(
    frame: np.ndarray,
    detection: PlateDetection,
    padding_ratio: float = 0.08,
) -> np.ndarray:
    """Crop a detected plate region with small contextual padding."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()


def select_best_detection(detections: list[PlateDetection]) -> PlateDetection | None:
    """Return the highest-confidence plate detection."""
    if not detections:
        return None
    return max(detections, key=lambda item: item.confidence)
