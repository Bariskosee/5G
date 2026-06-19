"""Tests for PlateReader OCR wrapper — no EasyOCR download required."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.ocr.plate_reader import PlateReader


def _blank_crop() -> np.ndarray:
    return np.zeros((30, 100, 3), dtype=np.uint8)


def test_disabled_returns_empty() -> None:
    """PlateReader with enabled=False must return [] without any error."""
    reader = PlateReader(enabled=False)
    assert reader.read_plate(_blank_crop()) == []


def test_disabled_available_flag() -> None:
    reader = PlateReader(enabled=False)
    assert not reader.available


def test_none_crop_returns_empty() -> None:
    """read_plate must handle None input gracefully."""
    reader = PlateReader(enabled=False)
    assert reader.read_plate(None) == []  # type: ignore[arg-type]


def test_empty_crop_returns_empty() -> None:
    """read_plate must handle zero-size array gracefully."""
    reader = PlateReader(enabled=False)
    assert reader.read_plate(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_unavailable_easyocr_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """When easyocr import fails PlateReader must not crash and must return []."""
    monkeypatch.setitem(sys.modules, "easyocr", None)

    import importlib

    import src.ocr.plate_reader as module

    importlib.reload(module)

    reader = module.PlateReader(enabled=True)
    assert not reader.available
    assert reader.read_plate(_blank_crop()) == []
