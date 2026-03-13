"""Evaluation metrics for object detection.

Provides standard detection-quality metrics:

* :func:`calculate_iou` — Intersection over Union for two bounding boxes.
* :func:`calculate_precision` — Precision from TP / FP counts.
* :func:`calculate_recall` — Recall from TP / FN counts.
* :func:`calculate_accuracy` — Frame-level accuracy from prediction lists.

All functions are pure Python / NumPy — no ML framework required.

Example::

    from src.evaluation.metrics import calculate_iou, calculate_precision

    iou = calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])   # 0.142...
    precision = calculate_precision(tp=8, fp=2)             # 0.8
"""

from __future__ import annotations


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate Intersection over Union (IoU) for two axis-aligned bounding boxes.

    Parameters
    ----------
    box1, box2:
        Bounding boxes in ``[x1, y1, x2, y2]`` format (pixel coordinates).
        ``x1 < x2`` and ``y1 < y2`` are assumed.

    Returns
    -------
    float
        IoU value in the range ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If either box has non-positive area.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 <= 0 or area2 <= 0:
        raise ValueError(
            f"Bounding boxes must have positive area. Got area1={area1}, area2={area2}"
        )

    union = area1 + area2 - intersection
    return float(intersection / union)


def calculate_precision(tp: int, fp: int) -> float:
    """Calculate precision.

    .. math::

        \\text{Precision} = \\frac{TP}{TP + FP}

    Parameters
    ----------
    tp:
        Number of true positives.
    fp:
        Number of false positives.

    Returns
    -------
    float
        Precision in ``[0.0, 1.0]``.  Returns ``0.0`` when ``tp + fp == 0``.
    """
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def calculate_recall(tp: int, fn: int) -> float:
    """Calculate recall.

    .. math::

        \\text{Recall} = \\frac{TP}{TP + FN}

    Parameters
    ----------
    tp:
        Number of true positives.
    fn:
        Number of false negatives.

    Returns
    -------
    float
        Recall in ``[0.0, 1.0]``.  Returns ``0.0`` when ``tp + fn == 0``.
    """
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def calculate_accuracy(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate frame-level detection accuracy.

    A prediction is counted as a *true positive* when its IoU with any
    ground-truth box of the same class exceeds *iou_threshold*.

    Parameters
    ----------
    predictions:
        List of prediction dicts, each with keys
        ``{"bbox": [x1,y1,x2,y2], "class": str}``.
    ground_truth:
        List of ground-truth dicts in the same format.
    iou_threshold:
        Minimum IoU to consider a detection as correct.

    Returns
    -------
    float
        Accuracy in ``[0.0, 1.0]``.  Returns ``1.0`` when both lists are
        empty (nothing to detect, nothing predicted).
    """
    if not predictions and not ground_truth:
        return 1.0

    if not predictions or not ground_truth:
        return 0.0

    matched_gt: set[int] = set()
    tp = 0

    for pred in predictions:
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred.get("class") != gt.get("class"):
                continue
            try:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
            except (ValueError, KeyError):
                continue
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                break

    total = max(len(predictions), len(ground_truth))
    return float(tp / total)
