"""Unit tests for src.evaluation.metrics.

Run with::

    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_iou,
    calculate_precision,
    calculate_recall,
)


# ---------------------------------------------------------------------------
# calculate_iou
# ---------------------------------------------------------------------------


class TestCalculateIoU:
    """Tests for calculate_iou."""

    def test_perfect_overlap(self) -> None:
        """Identical boxes → IoU == 1.0."""
        box = [0, 0, 10, 10]
        assert calculate_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes → IoU == 0.0."""
        box1 = [0, 0, 5, 5]
        box2 = [10, 10, 20, 20]
        assert calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Boxes overlapping by 25 sq-px, union 175 sq-px → IoU ≈ 1/7."""
        box1 = [0, 0, 10, 10]   # area = 100
        box2 = [5, 5, 15, 15]   # area = 100, intersection = 5×5 = 25
        expected = 25 / (100 + 100 - 25)  # 25/175 ≈ 0.1428…
        assert calculate_iou(box1, box2) == pytest.approx(expected, rel=1e-5)

    def test_contained_box(self) -> None:
        """Smaller box fully inside larger box."""
        outer = [0, 0, 10, 10]   # area = 100
        inner = [2, 2, 8, 8]     # area = 36, intersection = 36
        expected = 36 / (100 + 36 - 36)  # 36/100 = 0.36
        assert calculate_iou(outer, inner) == pytest.approx(expected)

    def test_touching_edges_no_overlap(self) -> None:
        """Boxes that share only an edge have zero area intersection."""
        box1 = [0, 0, 5, 5]
        box2 = [5, 0, 10, 5]
        assert calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        """Return type must be float."""
        result = calculate_iou([0, 0, 2, 2], [1, 1, 3, 3])
        assert isinstance(result, float)

    def test_zero_area_box_raises(self) -> None:
        """Boxes with zero area should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_iou([0, 0, 0, 5], [0, 0, 5, 5])  # zero-width box

    def test_symmetric(self) -> None:
        """IoU(A, B) == IoU(B, A)."""
        box1 = [0, 0, 6, 6]
        box2 = [3, 3, 9, 9]
        assert calculate_iou(box1, box2) == pytest.approx(calculate_iou(box2, box1))

    def test_float_coordinates(self) -> None:
        """Float-valued bounding box coordinates should work."""
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 5.0, 15.0, 15.0]
        result = calculate_iou(box1, box2)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# calculate_precision
# ---------------------------------------------------------------------------


class TestCalculatePrecision:
    """Tests for calculate_precision."""

    def test_perfect_precision(self) -> None:
        assert calculate_precision(tp=10, fp=0) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        assert calculate_precision(tp=0, fp=10) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        assert calculate_precision(tp=8, fp=2) == pytest.approx(0.8)

    def test_no_predictions(self) -> None:
        """No predictions at all → 0.0 (avoid division by zero)."""
        assert calculate_precision(tp=0, fp=0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(calculate_precision(tp=1, fp=1), float)

    def test_large_values(self) -> None:
        assert calculate_precision(tp=1000, fp=1000) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# calculate_recall
# ---------------------------------------------------------------------------


class TestCalculateRecall:
    """Tests for calculate_recall."""

    def test_perfect_recall(self) -> None:
        assert calculate_recall(tp=10, fn=0) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        assert calculate_recall(tp=0, fn=10) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        assert calculate_recall(tp=7, fn=3) == pytest.approx(0.7)

    def test_no_ground_truth(self) -> None:
        """No ground truth → 0.0 (avoid division by zero)."""
        assert calculate_recall(tp=0, fn=0) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(calculate_recall(tp=3, fn=7), float)


# ---------------------------------------------------------------------------
# calculate_accuracy
# ---------------------------------------------------------------------------


class TestCalculateAccuracy:
    """Tests for calculate_accuracy."""

    def _make_box(self, cls: str, x1: int, y1: int, x2: int, y2: int) -> dict:
        return {"class": cls, "bbox": [x1, y1, x2, y2]}

    def test_empty_both(self) -> None:
        """No predictions and no GT → perfect accuracy."""
        assert calculate_accuracy([], []) == pytest.approx(1.0)

    def test_empty_predictions(self) -> None:
        """No predictions but there are GT boxes → 0.0."""
        gt = [self._make_box("car", 0, 0, 10, 10)]
        assert calculate_accuracy([], gt) == pytest.approx(0.0)

    def test_empty_ground_truth(self) -> None:
        """Predictions exist but no GT → 0.0."""
        preds = [self._make_box("car", 0, 0, 10, 10)]
        assert calculate_accuracy(preds, []) == pytest.approx(0.0)

    def test_perfect_match(self) -> None:
        """One prediction exactly matches one GT box."""
        box = self._make_box("car", 0, 0, 10, 10)
        assert calculate_accuracy([box], [box]) == pytest.approx(1.0)

    def test_class_mismatch(self) -> None:
        """Same bbox but different class labels → 0.0."""
        pred = self._make_box("car", 0, 0, 10, 10)
        gt = self._make_box("truck", 0, 0, 10, 10)
        assert calculate_accuracy([pred], [gt]) == pytest.approx(0.0)

    def test_iou_below_threshold(self) -> None:
        """Prediction with IoU below threshold → not matched."""
        pred = self._make_box("car", 0, 0, 5, 5)
        gt = self._make_box("car", 4, 4, 14, 14)  # very small overlap
        # iou = 1/(25+100-1) ≈ 0.008 → below default threshold 0.5
        assert calculate_accuracy([pred], [gt], iou_threshold=0.5) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        box = self._make_box("car", 0, 0, 10, 10)
        result = calculate_accuracy([box], [box])
        assert isinstance(result, float)

    def test_partial_match(self) -> None:
        """Two predictions, one matches GT, one does not → 0.5."""
        gt = [self._make_box("car", 0, 0, 10, 10)]
        preds = [
            self._make_box("car", 0, 0, 10, 10),   # perfect match
            self._make_box("car", 50, 50, 60, 60),  # far away, no match
        ]
        # TP=1, total=max(2,1)=2 → 0.5
        assert calculate_accuracy(preds, gt) == pytest.approx(0.5)
