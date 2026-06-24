"""MediaPipe FaceLandmarker wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/app/models/mediapipe/face_landmarker.task"


class FaceLandmarker:
    """Run MediaPipe FaceLandmarker on BGR frames.

    Gracefully degrades to a no-op when the model file is absent.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._delegate: Any = None
        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "FaceLandmarker: model not found at '%s' — landmark detection disabled.",
                self._model_path,
            )
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self._delegate = vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker: loaded from '%s'", self._model_path)
        except Exception as exc:
            logger.warning("FaceLandmarker: init failed (%s) — landmark detection disabled.", exc)
            self._delegate = None

    @property
    def available(self) -> bool:
        return self._delegate is not None

    def detect(self, bgr_frame: np.ndarray) -> Any | None:
        """Return MediaPipe FaceLandmarkerResult, or None if unavailable/no face detected."""
        if self._delegate is None or bgr_frame is None or bgr_frame.size == 0:
            return None
        try:
            import mediapipe as mp

            rgb = bgr_frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._delegate.detect(mp_image)
            return result if result.face_landmarks else None
        except Exception as exc:
            logger.debug("FaceLandmarker: detect failed (%s)", exc)
            return None
