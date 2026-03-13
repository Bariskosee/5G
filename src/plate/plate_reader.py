"""License plate detection and OCR module.

This module provides the :class:`PlateReader` class which:

1. Detects a license plate region inside a vehicle crop (``detect_plate``).
2. Reads the text from a plate crop using OCR (``read_text``).

Both steps are functional placeholders — replace the ``detect_plate``
implementation with a trained plate-detection model and the ``read_text``
implementation with EasyOCR / Tesseract as needed.

Example::

    from src.plate.plate_reader import PlateReader

    reader = PlateReader()
    plate_info = reader.detect_plate(vehicle_crop)   # numpy BGR array
    text = reader.read_text(plate_info["crop"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PlateReader:
    """Detect and read license plates from vehicle image crops.

    Parameters
    ----------
    model_path:
        Path to a plate-detection model weights file.  When *None* a simple
        centre-crop heuristic is used as a placeholder.
    confidence:
        Minimum confidence threshold for plate detections.
    ocr_language:
        Language code passed to the OCR engine (default ``"tr"`` for Turkish).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.6,
        ocr_language: str = "tr",
    ) -> None:
        self.confidence = confidence
        self.ocr_language = ocr_language
        self._model: Any = None
        self._ocr: Any = None

        self._load_model(model_path)
        self._load_ocr()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str | Path | None) -> None:
        """Load plate-detection model or fall back to heuristic placeholder."""
        if model_path is None:
            logger.info("PlateReader: no model_path given; using heuristic placeholder")
            return

        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(str(model_path))
            logger.info("PlateReader: loaded detection model from '%s'", model_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("PlateReader: could not load model (%s)", exc)

    def _load_ocr(self) -> None:
        """Load EasyOCR reader; fall back gracefully if not installed."""
        try:
            import easyocr  # type: ignore[import]

            self._ocr = easyocr.Reader([self.ocr_language], gpu=False, verbose=False)
            logger.info("PlateReader: EasyOCR loaded (lang=%s)", self.ocr_language)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "PlateReader: EasyOCR not available (%s). "
                "read_text() will return an empty string.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_plate(self, vehicle_crop: np.ndarray) -> dict:
        """Detect the license plate region inside a vehicle image crop.

        Parameters
        ----------
        vehicle_crop:
            BGR numpy array of the cropped vehicle region.

        Returns
        -------
        dict
            ::

                {
                    "bbox":       [x1, y1, x2, y2],   # within vehicle_crop
                    "confidence": float,
                    "crop":       np.ndarray,          # BGR plate sub-image
                }

            ``bbox`` and ``confidence`` are ``None`` / ``0.0`` when detection
            is unavailable.
        """
        if self._model is not None:
            results = self._model(vehicle_crop, conf=self.confidence, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    crop = vehicle_crop[y1:y2, x1:x2]
                    return {"bbox": [x1, y1, x2, y2], "confidence": conf, "crop": crop}

        # Heuristic placeholder: assume plate is in the lower-centre of the crop
        h, w = vehicle_crop.shape[:2]
        x1 = w // 4
        x2 = 3 * w // 4
        y1 = 2 * h // 3
        y2 = h
        crop = vehicle_crop[y1:y2, x1:x2]
        logger.debug("PlateReader: using heuristic plate region")
        return {"bbox": [x1, y1, x2, y2], "confidence": 0.0, "crop": crop}

    def read_text(self, plate_crop: np.ndarray) -> str:
        """Run OCR on a plate image crop and return the recognised text.

        Parameters
        ----------
        plate_crop:
            BGR numpy array of the plate region.

        Returns
        -------
        str
            Recognised plate text, or an empty string on failure.
        """
        if plate_crop is None or plate_crop.size == 0:
            return ""

        if self._ocr is not None:
            try:
                results = self._ocr.readtext(plate_crop, detail=0)
                text = " ".join(results).strip().upper()
                logger.debug("PlateReader OCR result: '%s'", text)
                return text
            except Exception as exc:  # pragma: no cover
                logger.warning("PlateReader: OCR failed (%s)", exc)

        logger.debug("PlateReader: OCR not available; returning ''")
        return ""
