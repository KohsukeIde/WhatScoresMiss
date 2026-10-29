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


def unique_retained_shape_part_iou(
    logits: np.ndarray,
    target: np.ndarray,
    original_indices: np.ndarray,
    valid_part_ids: np.ndarray,
) -> float:
    """Score one shape using the PointGPT rebuttal's retained-point protocol.

    Average duplicate logits by original point ID, restrict predictions to
    the shape category's valid parts, and average IoU over all valid parts.
    A part absent from both prediction and target has IoU 1.
    """
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target)
    original_indices = np.asarray(original_indices)
    valid_part_ids = np.asarray(valid_part_ids)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have nonempty shape (N, C)")
    if target.ndim != 1 or target.size != logits.shape[0]:
        raise ValueError("target must have shape (N,)")
    if original_indices.ndim != 1 or original_indices.size != logits.shape[0]:
        raise ValueError("original_indices must have shape (N,)")
    if not np.issubdtype(original_indices.dtype, np.integer):
        raise ValueError("original_indices must contain integer IDs")
    if valid_part_ids.ndim != 1 or valid_part_ids.size == 0:
        raise ValueError("valid_part_ids must be a nonempty 1D array")
    if not np.issubdtype(valid_part_ids.dtype, np.integer):
        raise ValueError("valid_part_ids must contain integers")
    if np.unique(valid_part_ids).size != valid_part_ids.size:
        raise ValueError("valid_part_ids must be unique")
    if np.any(valid_part_ids < 0) or np.any(valid_part_ids >= logits.shape[1]):
        raise ValueError("valid_part_ids are outside the logit dimension")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    if not np.all(np.isin(target, valid_part_ids)):
        raise ValueError("target contains a part outside the category")

    unique_ids, averaged = average_duplicate_logits(logits, original_indices)
    unique_target = []
    for original_id in unique_ids:
        labels = np.unique(target[original_indices == original_id])
        if labels.size != 1:
            raise ValueError("one original point ID has inconsistent targets")
        unique_target.append(labels[0])
    unique_target = np.asarray(unique_target)
    pred = valid_part_ids[np.argmax(averaged[:, valid_part_ids], axis=1)]
    part_ious = []
    for part_id in valid_part_ids:
        union = np.logical_or(unique_target == part_id, pred == part_id).sum()
        intersection = np.logical_and(unique_target == part_id, pred == part_id).sum()
        part_ious.append(1.0 if union == 0 else float(intersection / union))
    return float(np.mean(part_ious))
