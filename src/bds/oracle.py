"""Oracle@k candidate-set diagnostics."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .retained_scoring import accuracy, mean_iou


def _topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError("k must satisfy 1 <= k <= num_classes")
    order = np.argsort(logits, axis=1)
    return order[:, -k:]


def topk_hit(logits: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Return whether target is in top-k for each row."""
    target = np.asarray(target)
    topk = _topk_indices(logits, k)
    if target.ndim != 1 or len(target) != topk.shape[0]:
        raise ValueError("target must have shape (N,)")
    return np.any(topk == target[:, None], axis=1)


def oracle_predictions(logits: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Use argmax unless the target appears in top-k, then use the target."""
    logits = np.asarray(logits)
    target = np.asarray(target)
    pred = np.argmax(logits, axis=1)
    hit = topk_hit(logits, target, k)
    pred = pred.copy()
    pred[hit] = target[hit]
    return pred


def oracle_accuracy(logits: np.ndarray, target: np.ndarray, k: int) -> float:
    """Compute accuracy after Oracle@k replacement."""
    return accuracy(oracle_predictions(logits, target, k), target)


def oracle_miou(
    logits: np.ndarray,
    target: np.ndarray,
    k: int,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute mIoU after Oracle@k replacement."""
    pred = oracle_predictions(logits, target, k)
    return mean_iou(pred, target, num_classes=num_classes, ignore_index=ignore_index)
