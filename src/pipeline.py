"""End-to-end detection pipeline.

Chains together:
1. Frame extraction from a video file
2. Vehicle detection per frame
3. License plate reading per detected vehicle
4. Driver behavior analysis per frame
5. Detection logging (JSON Lines)
6. Optional evaluation against ground truth labels

Example::

    from src.pipeline import Pipeline

    pipeline = Pipeline(config_path="configs/default.yaml")
    results = pipeline.run("data/raw/sample.mp4")
    print(results["summary"])
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.behavior.driver_analyzer import DriverAnalyzer
from src.detection.vehicle_detector import VehicleDetector
from src.plate.plate_reader import PlateReader
from src.utils.logger import DetectionLogger
from src.utils.video import extract_frames, load_frame
from src.utils.visualization import draw_behaviors, draw_detections

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load YAML config; return empty dict on failure."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Pipeline: could not load config '%s' (%s)", config_path, exc)
        return {}


class Pipeline:
    """Full road-safety detection pipeline.

    Parameters
    ----------
    config_path:
        Path to ``configs/default.yaml`` or a custom YAML config.
    output_dir:
        Where to write annotated frames and the detection log.
        Defaults to ``outputs/``.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/default.yaml",
        output_dir: str | Path = "outputs",
    ) -> None:
        self._cfg = _load_config(config_path)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = self._cfg.get("model", {})
        video_cfg = self._cfg.get("video", {})
        logging_cfg = self._cfg.get("logging", {})
        paths_cfg = self._cfg.get("paths", {})

        # Frame extraction settings
        self._frame_interval: float = float(video_cfg.get("frame_interval", 1))
        self._max_frames: int | None = video_cfg.get("max_frames")

        # Sub-modules
        vd_cfg = model_cfg.get("vehicle_detector", {})
        self._vehicle_detector = VehicleDetector(
            model_path=vd_cfg.get("weights"),
            confidence=float(vd_cfg.get("confidence", 0.5)),
            device=str(vd_cfg.get("device", "cpu")),
        )

        pr_cfg = model_cfg.get("plate_reader", {})
        self._plate_reader = PlateReader(
            confidence=float(pr_cfg.get("confidence", 0.6)),
            ocr_language=str(pr_cfg.get("ocr_language", "tr")),
        )

        da_cfg = model_cfg.get("driver_analyzer", {})
        self._driver_analyzer = DriverAnalyzer(
            confidence=float(da_cfg.get("confidence", 0.5)),
        )

        # Logger
        log_dir = Path(logging_cfg.get("log_dir", self._output_dir / "logs"))
        log_file = logging_cfg.get("log_file", "detections.json")
        self._log_path = log_dir / log_file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | Path,
        save_annotated: bool = True,
    ) -> dict[str, Any]:
        """Run the full pipeline on a single video file.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        save_annotated:
            Whether to save annotated frames to *output_dir/annotated/*.

        Returns
        -------
        dict
            A results dict with the following keys:

            * ``"video_path"`` — input video path (str)
            * ``"frames_processed"`` — number of frames analysed
            * ``"vehicle_detections"`` — total vehicle detections across all frames
            * ``"plate_readings"`` — list of recognised plate texts
            * ``"behavior_detections"`` — total behavior detections
            * ``"log_path"`` — path to the JSON log file (str)
            * ``"summary"`` — human-readable summary string
        """
        video_path = Path(video_path)
        logger.info("Pipeline: starting on '%s'", video_path)

        # 1. Extract frames to a temporary directory
        with tempfile.TemporaryDirectory(prefix="pipeline_frames_") as tmp_dir:
            frame_paths = extract_frames(
                video_path=video_path,
                output_dir=tmp_dir,
                interval=self._frame_interval,
                max_frames=self._max_frames,
            )

            annotated_dir = self._output_dir / "annotated" / video_path.stem
            if save_annotated:
                annotated_dir.mkdir(parents=True, exist_ok=True)

            total_vehicles = 0
            total_behaviors = 0
            plate_texts: list[str] = []

            with DetectionLogger(self._log_path) as dlog:
                for frame_idx, frame_path in enumerate(frame_paths):
                    frame = load_frame(frame_path)

                    # 2. Vehicle detection
                    vehicles = self._vehicle_detector.detect(frame)
                    dlog.log_many(
                        frame_id=frame_idx,
                        module="vehicle",
                        detections=vehicles,
                        video_path=str(video_path),
                    )
                    total_vehicles += len(vehicles)

                    # 3. Plate reading for each detected vehicle
                    for veh in vehicles:
                        x1, y1, x2, y2 = veh["bbox"]
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        plate_info = self._plate_reader.detect_plate(crop)
                        plate_text = self._plate_reader.read_text(plate_info["crop"])
                        if plate_text:
                            plate_texts.append(plate_text)
                        dlog.log(
                            frame_id=frame_idx,
                            module="plate",
                            detection={**plate_info, "text": plate_text, "crop": None},
                            video_path=str(video_path),
                        )

                    # 4. Driver behavior analysis
                    behaviors = self._driver_analyzer.analyze(frame)
                    dlog.log_many(
                        frame_id=frame_idx,
                        module="behavior",
                        detections=behaviors,
                        video_path=str(video_path),
                    )
                    total_behaviors += len(behaviors)

                    # 5. Save annotated frame
                    if save_annotated:
                        annotated = draw_detections(frame, vehicles)
                        annotated = draw_behaviors(annotated, behaviors)
                        import cv2  # local import to keep top-level light

                        cv2.imwrite(
                            str(annotated_dir / f"frame_{frame_idx:06d}.png"),
                            annotated,
                        )

        summary = (
            f"Processed {len(frame_paths)} frames from '{video_path.name}': "
            f"{total_vehicles} vehicle detections, "
            f"{len(plate_texts)} plate readings, "
            f"{total_behaviors} behavior detections."
        )
        logger.info("Pipeline: %s", summary)

        return {
            "video_path": str(video_path),
            "frames_processed": len(frame_paths),
            "vehicle_detections": total_vehicles,
            "plate_readings": plate_texts,
            "behavior_detections": total_behaviors,
            "log_path": str(self._log_path),
            "summary": summary,
        }
