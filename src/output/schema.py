"""Official FTR output schema helpers."""

from __future__ import annotations

import math
import re
from typing import Any

VALID_VEHICLE_TYPES = [
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "minibus",
    "panelvan",
    "kamyon",
]

VALID_COLORS = [
    "beyaz",
    "siyah",
    "gri",
    "kirmizi",
    "mavi",
    "sari",
    "yesil",
    "turuncu",
    "kahverengi",
]

VALID_CATEGORIES = [
    "sofor_eylemi",
    "nesneler",
    "yolcular",
]

VALID_LABELS_BY_CATEGORY = {
    "sofor_eylemi": [
        "arkaya_bakma",
        "esneme",
        "sigara_icme",
        "su_icme",
        "telefonla_konusma",
        "slalom",
        "etrafa_bakinma",
        "emniyet_kemeri_ihlali",
    ],
    "nesneler": [
        "teknocan",
        "bilgisayar",
    ],
    "yolcular": [
        "arka_koltuk_1",
        "arka_koltuk_2",
        "on_koltuk",
    ],
}

PLATE_FALLBACK = "tespit_edilemedi"
TOP_LEVEL_KEYS = {"video_id", "arac_bilgisi", "tespitler"}
ARAC_BILGISI_KEYS = {"tip", "plaka", "renk", "confidence_score"}
TESPIT_KEYS = {"zaman_saniye", "kategori", "etiket", "confidence_score"}

ASCII_SAFE_LOWER_RE = re.compile(r"^[a-z0-9_]+$")
TURKISH_PLATE_RE = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"[A-Z]{1,3}"
    r"[0-9]{2,5}$"
)


def clamp_confidence(value: Any) -> float:
    """Clamp a confidence-like value into the official [0.0, 1.0] range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(confidence) or math.isinf(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def is_ascii_safe_lower(value: Any) -> bool:
    """Return True when value only contains lowercase ASCII letters, digits, or underscores."""
    return isinstance(value, str) and bool(ASCII_SAFE_LOWER_RE.fullmatch(value))


def _check_exact_keys(
    errors: list[str],
    obj: dict[str, Any],
    expected_keys: set[str],
    path: str,
) -> None:
    missing = expected_keys - obj.keys()
    extra = obj.keys() - expected_keys
    if missing:
        errors.append(f"{path}: missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"{path}: unexpected extra keys: {sorted(extra)}")


def _check_confidence(errors: list[str], value: Any, path: str) -> None:
    if not isinstance(value, (int, float)):
        errors.append(f"{path}: confidence_score must be numeric")
        return

    confidence = float(value)
    if math.isnan(confidence) or math.isinf(confidence) or not 0.0 <= confidence <= 1.0:
        errors.append(f"{path}: confidence_score must be between 0 and 1")


def _check_plate(errors: list[str], value: Any, path: str) -> None:
    if not isinstance(value, str):
        errors.append(f"{path}: plaka must be a string")
        return

    if value == PLATE_FALLBACK:
        return

    if not TURKISH_PLATE_RE.fullmatch(value):
        errors.append(
            f"{path}: plaka must be a normalized Turkish plate or '{PLATE_FALLBACK}'"
        )


def validate_output_schema(output: dict[str, Any]) -> list[str]:
    """Validate an FTR results object and return a list of schema errors."""
    errors: list[str] = []

    if not isinstance(output, dict):
        return ["top-level output must be an object"]

    _check_exact_keys(errors, output, TOP_LEVEL_KEYS, "top-level")

    video_id = output.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        errors.append("video_id: must be a non-empty string")

    arac = output.get("arac_bilgisi")
    if not isinstance(arac, dict):
        errors.append("arac_bilgisi: must be an object")
    else:
        _check_exact_keys(errors, arac, ARAC_BILGISI_KEYS, "arac_bilgisi")

        tip = arac.get("tip")
        if not is_ascii_safe_lower(tip) or tip not in VALID_VEHICLE_TYPES:
            errors.append("arac_bilgisi.tip: invalid vehicle type")

        renk = arac.get("renk")
        if not is_ascii_safe_lower(renk) or renk not in VALID_COLORS:
            errors.append("arac_bilgisi.renk: invalid color")

        _check_plate(errors, arac.get("plaka"), "arac_bilgisi.plaka")

        if "confidence_score" in arac:
            _check_confidence(
                errors,
                arac["confidence_score"],
                "arac_bilgisi.confidence_score",
            )

    tespitler = output.get("tespitler")
    if not isinstance(tespitler, list):
        errors.append("tespitler: must be a list")
    else:
        for idx, tespit in enumerate(tespitler):
            path = f"tespitler[{idx}]"
            if not isinstance(tespit, dict):
                errors.append(f"{path}: must be an object")
                continue

            _check_exact_keys(errors, tespit, TESPIT_KEYS, path)

            zaman = tespit.get("zaman_saniye")
            if not isinstance(zaman, (int, float)) or float(zaman) < 0:
                errors.append(f"{path}.zaman_saniye: must be a non-negative number")

            kategori = tespit.get("kategori")
            if not is_ascii_safe_lower(kategori) or kategori not in VALID_CATEGORIES:
                errors.append(f"{path}.kategori: invalid category")

            etiket = tespit.get("etiket")
            allowed_labels = VALID_LABELS_BY_CATEGORY.get(str(kategori), [])
            if not is_ascii_safe_lower(etiket) or etiket not in allowed_labels:
                errors.append(f"{path}.etiket: invalid label for category")

            if "confidence_score" in tespit:
                _check_confidence(
                    errors,
                    tespit["confidence_score"],
                    f"{path}.confidence_score",
                )

    return errors
