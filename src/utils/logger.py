"""Detection logger: records every detection to a JSON Lines log file.

Each log entry captures:
- ISO-8601 timestamp
- frame_id
- module name (vehicle / plate / behavior)
- model output (raw detection dict)
- confidence score

This log is required for automated proof-of-analysis in the competition.

Example::

    from src.utils.logger import DetectionLogger

    dlog = DetectionLogger(log_path="outputs/logs/detections.json")
    dlog.log(frame_id=42, module="vehicle", detection={"class": "car", "confidence": 0.91})
    dlog.close()

    # Or use as a context manager:
    with DetectionLogger("outputs/logs/detections.json") as dlog:
        dlog.log(frame_id=0, module="vehicle", detection={...})
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DetectionLogger:
    """Append-only JSON Lines logger for detection events.

    Each call to :meth:`log` writes a single JSON object (one per line) to
    the log file.  The file is flushed after every write so that partial
    results survive crashes.

    Parameters
    ----------
    log_path:
        Path to the output ``.json`` / ``.jsonl`` file.
        Parent directories are created automatically.
    """

    def __init__(self, log_path: str | Path = "outputs/logs/detections.json") -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._log_path.open("a", encoding="utf-8")
        logger.info("DetectionLogger: logging to '%s'", self._log_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        frame_id: int,
        module: str,
        detection: dict,
        video_path: str | None = None,
    ) -> None:
        """Write a single detection event to the log.

        Parameters
        ----------
        frame_id:
            Zero-based frame index within the video.
        module:
            Name of the module that produced the detection
            (e.g. ``"vehicle"``, ``"plate"``, ``"behavior"``).
        detection:
            Raw detection dict as returned by the corresponding module.
        video_path:
            Optional source video file path for cross-referencing.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "frame_id": frame_id,
            "module": module,
            "video_path": video_path,
            "confidence": detection.get("confidence"),
            "detection": detection,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        logger.debug(
            "DetectionLogger: frame=%d module=%s confidence=%s",
            frame_id,
            module,
            entry["confidence"],
        )

    def log_many(
        self,
        frame_id: int,
        module: str,
        detections: list[dict],
        video_path: str | None = None,
    ) -> None:
        """Log a list of detections for the same frame and module.

        Parameters
        ----------
        frame_id, module, video_path:
            Same as :meth:`log`.
        detections:
            List of detection dicts to log.
        """
        for det in detections:
            self.log(frame_id=frame_id, module=module, detection=det, video_path=video_path)

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("DetectionLogger: closed '%s'", self._log_path)

    @staticmethod
    def read_log(log_path: str | Path) -> list[dict]:
        """Read all entries from a JSON Lines log file.

        Parameters
        ----------
        log_path:
            Path to the log file written by :class:`DetectionLogger`.

        Returns
        -------
        list[dict]
            Parsed log entries in the order they were written.
        """
        log_path = Path(log_path)
        entries: list[dict] = []
        if not log_path.exists():
            return entries
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
