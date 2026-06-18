"""Build and write FTR results JSON objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.output.schema import (
    PLATE_FALLBACK,
    VALID_COLORS,
    VALID_VEHICLE_TYPES,
    clamp_confidence,
    validate_output_schema,
)
from src.utils.plate_normalizer import is_valid_turkish_plate

DEFAULT_VEHICLE_TYPE = "sedan"
DEFAULT_COLOR = "beyaz"
DEFAULT_FALLBACK_CONFIDENCE = 0.01


def _safe_vehicle_type(vehicle_type: str | None) -> str:
    return vehicle_type if vehicle_type in VALID_VEHICLE_TYPES else DEFAULT_VEHICLE_TYPE


def _safe_color(color: str | None) -> str:
    return color if color in VALID_COLORS else DEFAULT_COLOR


def _safe_plate(plate: str | None) -> str:
    if plate == PLATE_FALLBACK:
        return PLATE_FALLBACK
    if plate and is_valid_turkish_plate(plate):
        return plate
    return PLATE_FALLBACK


def build_default_result(video_id: str) -> dict[str, Any]:
    """Build a valid temporary fallback result for the current skeleton."""
    return build_result(
        video_id=video_id,
        vehicle_type=DEFAULT_VEHICLE_TYPE,
        plate=PLATE_FALLBACK,
        color=DEFAULT_COLOR,
        vehicle_confidence=DEFAULT_FALLBACK_CONFIDENCE,
        events=[],
    )


def build_result(
    video_id: str,
    vehicle_type: str | None,
    plate: str | None,
    color: str | None,
    vehicle_confidence: float,
    events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build an official-schema FTR result object."""
    safe_plate = _safe_plate(plate)
    if safe_plate == PLATE_FALLBACK:
        confidence = DEFAULT_FALLBACK_CONFIDENCE
    else:
        confidence = clamp_confidence(vehicle_confidence)

    return {
        "video_id": video_id,
        "arac_bilgisi": {
            "tip": _safe_vehicle_type(vehicle_type),
            "plaka": safe_plate,
            "renk": _safe_color(color),
            "confidence_score": confidence,
        },
        "tespitler": events or [],
    }


def write_results_json(output_data: dict[str, Any], output_path: str | Path) -> None:
    """Validate and write an FTR results JSON file."""
    errors = validate_output_schema(output_data)
    if errors:
        joined_errors = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid FTR output schema:\n{joined_errors}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
