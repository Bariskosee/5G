"""Driver behavior analysis module.

Detects unsafe driver behaviors from a video frame:

* phone usage
* smoking
* drowsiness (eye closure / head nodding)
* seatbelt violation

This is a functional placeholder.  Replace ``_placeholder_analysis`` with a
trained classifier or pose-estimation pipeline.

Example::

    from src.behavior.driver_analyzer import DriverAnalyzer

    analyzer = DriverAnalyzer()
    behaviors = analyzer.analyze(frame)
    # [{"behavior": "phone_usage", "confidence": 0.85}, ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Supported behavior labels
BEHAVIOR_LABELS: list[str] = [
    "phone_usage",
    "smoking",
    "drowsiness",
    "seatbelt_violation",
]


class DriverAnalyzer:
    """Analyze a frame for unsafe driver behaviors.

    Parameters
    ----------
    model_path:
        Path to a classification / detection model weights file.
        When *None* the analyser operates in placeholder mode (returns empty
        results).
    confidence:
        Minimum confidence threshold for reported behaviors.
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

        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str | Path | None) -> None:
        """Load the behavior detection model."""
        if model_path is None:
            logger.info(
                "DriverAnalyzer: no model_path provided; running in placeholder mode"
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(str(model_path))
            self._model.to(self.device)
            logger.info("DriverAnalyzer: loaded model from '%s'", model_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("DriverAnalyzer: could not load model (%s)", exc)

    def _placeholder_analysis(self, frame: np.ndarray) -> list[dict]:  # noqa: ARG002
        """Return an empty detection list (placeholder).

        Replace this method with real inference once a model is available.
        """
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> list[dict]:
        """Detect unsafe driver behaviors in a single frame.

        Parameters
        ----------
        frame:
            A BGR image as a NumPy ``uint8`` array with shape ``(H, W, 3)``.

        Returns
        -------
        list[dict]
            Each entry is::

                {
                    "behavior":   str,    # e.g. "phone_usage"
                    "confidence": float,
                    "bbox":       [x1, y1, x2, y2] | None,
                }

            Returns an empty list when no behavior is detected or the model
            is not available.
        """
        if self._model is None:
            return self._placeholder_analysis(frame)

        results = self._model(frame, conf=self.confidence, verbose=False)
        behaviors: list[dict] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = (
                    BEHAVIOR_LABELS[cls_id]
                    if cls_id < len(BEHAVIOR_LABELS)
                    else f"behavior_{cls_id}"
                )
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                behaviors.append(
                    {
                        "behavior": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return behaviors
