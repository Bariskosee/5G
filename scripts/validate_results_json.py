"""
validate_results_json.py
========================
Validates a results.json file against the TEKNOFEST 2026 competition schema.

The competition auto-grader will reject any submission where:
  - JSON keys differ from the exact schema (e.g. "score" instead of "confidence_score")
  - Etiket/kategori values are not in the allowed enum
  - Turkish characters (ş, ç, ğ, ü, ö, ı) appear in labels
  - Confidence scores are outside [0.0, 1.0]
  - Required fields are missing

Run this BEFORE every Docker build and BEFORE final KYS submission.

Usage:
    python validate_results_json.py <path/to/results.json>

Exit codes:
    0 - Valid
    1 - Invalid (errors printed to stderr)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema definitions (must match competition FTR document exactly)
# ---------------------------------------------------------------------------

ALLOWED_VEHICLE_TYPES = {
    "sedan", "suv", "hatchback", "pickup",
    "minibus", "panelvan", "kamyon",
}

ALLOWED_COLORS = {
    "beyaz", "siyah", "gri", "kirmizi", "mavi",
    "sari", "yesil", "turuncu", "kahverengi",
}

ALLOWED_KATEGORI = {"sofor_eylemi", "nesneler", "yolcular"}

ALLOWED_ETIKET_BY_KATEGORI: dict[str, set[str]] = {
    "sofor_eylemi": {
        "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
        "telefonla_konusma", "slalom", "etrafa_bakinma",
        "emniyet_kemeri_ihlali",
    },
    "nesneler": {"teknocan", "bilgisayar"},
    "yolcular": {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"},
}

# Any string field must contain only these characters
ASCII_LOWER_PATTERN = re.compile(r"^[a-z0-9_]+$")

# Turkish license plate regex (loose): 2 digits + 1-3 letters + 2-5 digits
PLATE_PATTERN = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"([A-Z]{1,3})"
    r"(\d{2,5})$"
)

PLATE_PLACEHOLDER = "tespit_edilemedi"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def check_confidence(self, value: Any, path: str) -> None:
        if not isinstance(value, (int, float)):
            self.err(f"{path}: confidence_score must be a number, got {type(value).__name__}")
            return
        if not (0.0 <= float(value) <= 1.0):
            self.err(f"{path}: confidence_score {value} must be in [0.0, 1.0]")

    def check_ascii_lower(self, value: Any, path: str) -> None:
        if not isinstance(value, str):
            self.err(f"{path}: expected string, got {type(value).__name__}")
            return
        if not ASCII_LOWER_PATTERN.match(value):
            self.err(
                f"{path}: '{value}' contains non-ASCII or uppercase characters. "
                "Only [a-z0-9_] allowed."
            )

    def validate_arac_bilgisi(self, arac: Any) -> None:
        if not isinstance(arac, dict):
            self.err("arac_bilgisi: must be an object")
            return

        expected_keys = {"tip", "plaka", "renk", "confidence_score"}
        missing = expected_keys - arac.keys()
        if missing:
            self.err(f"arac_bilgisi: missing keys: {sorted(missing)}")
        extra = arac.keys() - expected_keys
        if extra:
            self.warn(f"arac_bilgisi: unexpected extra keys: {sorted(extra)}")

        # tip
        tip = arac.get("tip")
        if tip is not None:
            self.check_ascii_lower(tip, "arac_bilgisi.tip")
            if tip not in ALLOWED_VEHICLE_TYPES:
                self.err(
                    f"arac_bilgisi.tip: '{tip}' not in allowed set "
                    f"{sorted(ALLOWED_VEHICLE_TYPES)}"
                )

        # renk
        renk = arac.get("renk")
        if renk is not None:
            self.check_ascii_lower(renk, "arac_bilgisi.renk")
            if renk not in ALLOWED_COLORS:
                self.err(
                    f"arac_bilgisi.renk: '{renk}' not in allowed set "
                    f"{sorted(ALLOWED_COLORS)}"
                )

        # plaka
        plaka = arac.get("plaka")
        if plaka is not None:
            if not isinstance(plaka, str):
                self.err("arac_bilgisi.plaka: must be a string")
            elif plaka != PLATE_PLACEHOLDER and not PLATE_PATTERN.match(plaka):
                self.warn(
                    f"arac_bilgisi.plaka: '{plaka}' does not match Turkish plate "
                    f"regex. Use '{PLATE_PLACEHOLDER}' if OCR failed."
                )

        # confidence
        if "confidence_score" in arac:
            self.check_confidence(arac["confidence_score"], "arac_bilgisi.confidence_score")

    def validate_tespit(self, tespit: Any, idx: int) -> None:
        path = f"tespitler[{idx}]"
        if not isinstance(tespit, dict):
            self.err(f"{path}: must be an object")
            return

        expected_keys = {"zaman_saniye", "kategori", "etiket", "confidence_score"}
        missing = expected_keys - tespit.keys()
        if missing:
            self.err(f"{path}: missing keys: {sorted(missing)}")
        extra = tespit.keys() - expected_keys
        if extra:
            self.warn(f"{path}: unexpected extra keys: {sorted(extra)}")

        # zaman_saniye
        zaman = tespit.get("zaman_saniye")
        if zaman is not None and not isinstance(zaman, (int, float)):
            self.err(f"{path}.zaman_saniye: must be a number")
        elif isinstance(zaman, (int, float)) and zaman < 0:
            self.err(f"{path}.zaman_saniye: must be >= 0")

        # kategori + etiket (must be checked together)
        kategori = tespit.get("kategori")
        etiket = tespit.get("etiket")

        if kategori is not None:
            self.check_ascii_lower(kategori, f"{path}.kategori")
            if kategori not in ALLOWED_KATEGORI:
                self.err(
                    f"{path}.kategori: '{kategori}' not in {sorted(ALLOWED_KATEGORI)}"
                )

        if etiket is not None:
            self.check_ascii_lower(etiket, f"{path}.etiket")
            if kategori in ALLOWED_ETIKET_BY_KATEGORI:
                allowed = ALLOWED_ETIKET_BY_KATEGORI[kategori]
                if etiket not in allowed:
                    self.err(
                        f"{path}.etiket: '{etiket}' not valid for kategori "
                        f"'{kategori}'. Allowed: {sorted(allowed)}"
                    )

        # confidence
        if "confidence_score" in tespit:
            self.check_confidence(tespit["confidence_score"], f"{path}.confidence_score")

    def validate(self, data: Any) -> None:
        if not isinstance(data, dict):
            self.err("Top-level JSON must be an object")
            return

        expected_top = {"arac_bilgisi", "tespitler"}
        missing = expected_top - data.keys()
        if missing:
            self.err(f"Top-level: missing keys: {sorted(missing)}")
        extra = data.keys() - expected_top
        if extra:
            self.warn(f"Top-level: unexpected extra keys: {sorted(extra)}")

        if "arac_bilgisi" in data:
            self.validate_arac_bilgisi(data["arac_bilgisi"])

        if "tespitler" in data:
            tespitler = data["tespitler"]
            if not isinstance(tespitler, list):
                self.err("tespitler: must be a list")
            else:
                for i, t in enumerate(tespitler):
                    self.validate_tespit(t, i)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_results_json.py <path/to/results.json>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}", file=sys.stderr)
        return 1

    v = Validator()
    v.validate(data)

    for w in v.warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    if v.errors:
        for e in v.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(v.errors)} error(s), {len(v.warnings)} warning(s). FAILED.", file=sys.stderr)
        return 1

    print(f"OK: {path} is valid ({len(v.warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
