"""Scoring utilities for retained-support diagnostics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def average_duplicate_logits(
    logits: np.ndarray, original_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Average logits for duplicate original point indices."""
    logits = np.asarray(logits, dtype=float)
    original_indices = np.asarray(original_indices)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if original_indices.ndim != 1 or len(original_indices) != len(logits):
        raise ValueError("original_indices must have shape (N,)")

    unique, inverse = np.unique(original_indices, return_inverse=True)
    summed = np.zeros((len(unique), logits.shape[1]), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    np.add.at(summed, inverse, logits)
    np.add.at(counts, inverse, 1.0)
    return unique, summed / counts[:, None]


def mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute class-mean IoU over classes with non-empty union."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    mask = np.ones_like(target, dtype=bool)
    if ignore_index is not None:
        mask &= target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_c, target_c).sum()
        ious.append(inter / union)
    if not ious:
        return float("nan")
    return float(np.mean(ious))


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute simple accuracy."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if target.size == 0:
        return float("nan")
    return float(np.mean(pred == target))
