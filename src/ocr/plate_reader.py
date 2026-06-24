"""Best-effort OCR wrapper for cropped license plates."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import cv2
import numpy as np

from src.output.schema import clamp_confidence

logger = logging.getLogger(__name__)

# Restrict OCR output to plate-legal characters only.
# This prevents EasyOCR from returning Turkish characters or punctuation.
_PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PlateReader:
    """Read license plate text from crops using EasyOCR when available."""

    def __init__(
        self,
        languages: Sequence[str] = ("tr", "en"),
        use_gpu: bool = True,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.available = False
        self._reader: Any = None

        if not enabled:
            logger.info("PlateReader: OCR disabled by configuration")
            return

        try:
            import easyocr

            self._reader = easyocr.Reader(
                list(languages),
                gpu=use_gpu,
                download_enabled=False,
                model_storage_directory="/app/easyocr_models",
                verbose=False,
            )
            self.available = True
        except Exception as exc:
            logger.warning(
                "PlateReader: EasyOCR unavailable or model files missing; "
                "continuing without OCR. Details: %s  "
                "To enable OCR: install easyocr ('pip install easyocr') and ensure "
                "EasyOCR language model files are present (baked into Docker image at build time).",
                exc,
            )
            self._reader = None
            self.available = False

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if height > 0 and height < 64:
            scale = 64.0 / height
            gray = cv2.resize(
                gray,
                (max(1, int(width * scale)), 64),
                interpolation=cv2.INTER_CUBIC,
            )

        _, thresholded = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return thresholded

    def read_plate(self, crop: np.ndarray) -> list[tuple[str, float]]:
        """Return raw OCR candidates as (text, confidence)."""
        if not self.enabled or not self.available or self._reader is None:
            return []
        if crop is None or crop.size == 0:
            return []

        try:
            processed = self._preprocess(crop)
            results = self._reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist=_PLATE_ALLOWLIST,
            )
        except Exception as exc:
            logger.warning("PlateReader: OCR failed for crop; continuing. Details: %s", exc)
            return []

        candidates: list[tuple[str, float]] = []
        for result in results:
            if len(result) < 3:
                continue
            text = str(result[1]).strip()
            confidence = clamp_confidence(result[2])
            if text:
                candidates.append((text, confidence))

        return candidates
