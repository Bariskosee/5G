"""Turkish license plate text normalization helpers."""

from __future__ import annotations

import re
from typing import Iterable

from src.output.schema import TURKISH_PLATE_RE, clamp_confidence

NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")
NUMERIC_OCR_MAP = str.maketrans({"O": "0", "I": "1", "S": "5"})


def _clean_text(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.upper())


def _normalize_numeric_segment(value: str) -> str:
    return value.translate(NUMERIC_OCR_MAP)


def is_valid_turkish_plate(plate: str) -> bool:
    """Return True for normalized Turkish plates with province code 01-81."""
    return bool(isinstance(plate, str) and TURKISH_PLATE_RE.fullmatch(plate))


def _plate_candidates(cleaned: str) -> Iterable[str]:
    if len(cleaned) < 5:
        return

    province = _normalize_numeric_segment(cleaned[:2])
    body = cleaned[2:]

    for letter_count in (1, 2, 3):
        letters = body[:letter_count]
        numbers = _normalize_numeric_segment(body[letter_count:])
        if not letters or not numbers:
            continue
        if letters.isalpha() and numbers.isdigit():
            yield f"{province}{letters}{numbers}"


def normalize_plate_text(text: str | None) -> str | None:
    """Normalize OCR output into a valid Turkish plate string when possible."""
    if text is None:
        return None

    cleaned = _clean_text(text)
    if not cleaned:
        return None

    for candidate in _plate_candidates(cleaned):
        if is_valid_turkish_plate(candidate):
            return candidate

    return None


def choose_best_plate(candidates: list[tuple[str, float]]) -> tuple[str | None, float]:
    """Choose the highest-confidence valid normalized plate candidate."""
    best_plate: str | None = None
    best_confidence = 0.0

    for raw_text, confidence in candidates:
        normalized = normalize_plate_text(raw_text)
        if normalized is None:
            continue

        normalized_confidence = clamp_confidence(confidence)
        if best_plate is None or normalized_confidence > best_confidence:
            best_plate = normalized
            best_confidence = normalized_confidence

    return best_plate, best_confidence
