"""Pure, identity-aware primitives for the R7 resampling probe.

This module deliberately has no PointGPT, CUDA, checkpoint, or dataset imports so
that the protocol invariants can be tested on tiny synthetic arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


CURRENT = "current_all_replace"
SUPPORT_PRESERVING = "support_preserving"
UNIQUE = "unique_no_padding"
RESAMPLING_PATHS = (CURRENT, SUPPORT_PRESERVING, UNIQUE)
PREREGISTERED_FRACTIONS = (0.8, 0.5, 0.2)
RESAMPLED_2048_WITH_DUPLICATES = "resampled_2048_with_duplicates"
UNIQUE_RETAINED_DEDUPED = "unique_retained_deduped"
FULL_CLOUD_NN_PROPAGATION = "full_cloud_nn_propagation"
CLASS_MEAN_IOU_OVER_50 = "class_mean_iou_over_50"
INSTANCE_SCORING_METRICS = (
    RESAMPLED_2048_WITH_DUPLICATES,
    UNIQUE_RETAINED_DEDUPED,
    FULL_CLOUD_NN_PROPAGATION,
)
SCORING_METRICS = INSTANCE_SCORING_METRICS + (CLASS_MEAN_IOU_OVER_50,)


def keep_count(npoint: int, fraction: float) -> int:
    if npoint <= 0:
        raise ValueError("npoint must be positive")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, int(round(float(fraction) * int(npoint))))


def nested_supports(
    priority: np.ndarray, fractions: Sequence[float]
) -> Dict[float, np.ndarray]:
    """Return prefix supports from one priority; smaller supports are nested."""
    priority = np.asarray(priority, dtype=np.int64)
    if priority.ndim != 1 or priority.size == 0:
        raise ValueError("priority must be a non-empty 1D array")
    if np.unique(priority).size != priority.size:
        raise ValueError("priority must be a permutation without duplicates")
    out = {}
    for fraction in fractions:
        out[float(fraction)] = priority[: keep_count(priority.size, fraction)].copy()
    ordered = sorted(out, key=lambda x: keep_count(priority.size, x))
    for small, large in zip(ordered, ordered[1:]):
        if not set(out[small].tolist()).issubset(set(out[large].tolist())):
            raise AssertionError("retained supports are not nested")
    return out


def resampling_map(
    retained_size: int,
    npoint: int,
    rng: np.random.RandomState,
    path: str = CURRENT,
) -> np.ndarray:
    """Map model-input positions to positions in the retained support."""
    retained_size = int(retained_size)
    npoint = int(npoint)
    if retained_size <= 0 or npoint <= 0:
        raise ValueError("retained_size and npoint must be positive")
    if path == CURRENT:
        return rng.choice(
            retained_size,
            npoint,
            replace=retained_size < npoint,
        ).astype(np.int64, copy=False)
    if path == SUPPORT_PRESERVING:
        if retained_size >= npoint:
            return rng.choice(retained_size, npoint, replace=False).astype(
                np.int64, copy=False
            )
        extras = rng.choice(retained_size, npoint - retained_size, replace=True)
        combined = np.concatenate([np.arange(retained_size, dtype=np.int64), extras])
        # Avoid making input position zero or the early FPS candidates special.
        return combined[rng.permutation(npoint)]
    if path == UNIQUE:
        if retained_size > npoint:
            raise ValueError("unique variable-N path expects retained_size <= npoint")
        return np.arange(retained_size, dtype=np.int64)
    raise ValueError("unknown resampling path: %s" % path)


def apply_retention_and_resampling(
    points: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
    retained_positions: np.ndarray,
    model_to_retained: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply both maps while preserving exact original-point identity."""
    points = np.asarray(points)
    target = np.asarray(target)
    original_ids = np.asarray(original_ids)
    retained_positions = np.asarray(retained_positions, dtype=np.int64)
    model_to_retained = np.asarray(model_to_retained, dtype=np.int64)
    if points.shape[0] != target.shape[0] or points.shape[0] != original_ids.shape[0]:
        raise ValueError(
            "points, target, and original_ids must share their first dimension"
        )
    if retained_positions.ndim != 1 or model_to_retained.ndim != 1:
        raise ValueError("maps must be 1D")
    if retained_positions.size == 0:
        raise ValueError("retained support is empty")
    if retained_positions.min() < 0 or retained_positions.max() >= points.shape[0]:
        raise IndexError("retained position is out of range")
    if (
        model_to_retained.min() < 0
        or model_to_retained.max() >= retained_positions.size
    ):
        raise IndexError("resampling map is out of range")
    input_positions = retained_positions[model_to_retained]
    return (
        points[input_positions],
        target[input_positions],
        original_ids[input_positions],
    )


def multiplicity_summary(original_ids: np.ndarray) -> Dict[str, object]:
    original_ids = np.asarray(original_ids, dtype=np.int64)
    unique_ids, counts = np.unique(original_ids, return_counts=True)
    hist_values, hist_counts = np.unique(counts, return_counts=True)
    return {
        "unique_ids": unique_ids.tolist(),
        "counts": counts.tolist(),
        "num_input_positions": int(original_ids.size),
        "num_unique_original_ids": int(unique_ids.size),
        "occupancy": float(unique_ids.size / max(1, original_ids.size)),
        "multiplicity_histogram": {
            str(int(value)): int(count)
            for value, count in zip(hist_values.tolist(), hist_counts.tolist())
        },
        "max_multiplicity": int(counts.max()) if counts.size else 0,
    }


def average_duplicate_logits(
    logits: np.ndarray, original_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduped scorer helper: mean logits by tracked original-point ID."""
    logits = np.asarray(logits, dtype=np.float64)
    original_ids = np.asarray(original_ids, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if original_ids.ndim != 1 or original_ids.size != logits.shape[0]:
        raise ValueError("original_ids must have shape (N,)")
    unique_ids, inverse = np.unique(original_ids, return_inverse=True)
    summed = np.zeros((unique_ids.size, logits.shape[1]), dtype=np.float64)
    counts = np.zeros(unique_ids.size, dtype=np.int64)
    np.add.at(summed, inverse, logits)
    np.add.at(counts, inverse, 1)
    return unique_ids, summed / counts[:, None]


def average_targets_by_id(
    target: np.ndarray, original_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one target per ID and fail if duplicate IDs disagree."""
    target = np.asarray(target)
    original_ids = np.asarray(original_ids, dtype=np.int64)
    unique_ids = np.unique(original_ids)
    values = []
    for original_id in unique_ids:
        candidate = np.unique(target[original_ids == original_id])
        if candidate.size != 1:
            raise AssertionError("one original ID has inconsistent targets")
        values.append(candidate[0])
    return unique_ids, np.asarray(values, dtype=target.dtype)


def deduped_pred_target_over_50(
    logits: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate one shape, then predict over all 50 global part classes."""
    logit_ids, averaged = average_duplicate_logits(logits, original_ids)
    target_ids, unique_target = average_targets_by_id(target, original_ids)
    if not np.array_equal(logit_ids, target_ids):
        raise AssertionError("logit and target identity sets differ")
    if averaged.shape[1] != 50:
        raise ValueError("class_mean_iou_over_50 requires exactly 50 logit classes")
    return averaged.argmax(axis=1), unique_target


def class_intersection_union_over_50(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sufficient statistics for cross-shape pooled class-mean IoU."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    intersections = np.zeros(50, dtype=np.int64)
    unions = np.zeros(50, dtype=np.int64)
    for cls in range(50):
        pred_c = pred == cls
        target_c = target == cls
        unions[cls] = np.logical_or(pred_c, target_c).sum()
        intersections[cls] = np.logical_and(pred_c, target_c).sum()
    return intersections, unions


def class_mean_iou_from_intersection_union(
    intersections: np.ndarray,
    unions: np.ndarray,
) -> float:
    """Average class IoUs with non-empty union, matching bds.mean_iou."""
    intersections = np.asarray(intersections)
    unions = np.asarray(unions)
    if intersections.shape != (50,) or unions.shape != (50,):
        raise ValueError("intersections and unions must each have shape (50,)")
    nonempty = unions != 0
    if not np.any(nonempty):
        return float("nan")
    return float(np.mean(intersections[nonempty] / unions[nonempty]))


def class_mean_iou_over_50(pred: np.ndarray, target: np.ndarray) -> float:
    """Class-mean IoU over pooled points, exactly as bds.mean_iou(..., 50)."""
    intersections, unions = class_intersection_union_over_50(pred, target)
    return class_mean_iou_from_intersection_union(intersections, unions)


def _mean_part_iou(
    pred: np.ndarray,
    target: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 1:
        raise ValueError("pred and target must have the same 1D shape")
    part_ious = []
    for part_id in np.asarray(valid_part_ids, dtype=np.int64):
        union = np.logical_or(target == part_id, pred == part_id).sum()
        part_ious.append(
            1.0
            if union == 0
            else float(np.logical_and(target == part_id, pred == part_id).sum() / union)
        )
    return float(np.mean(part_ious))


def resampled_shape_part_iou(
    logits: np.ndarray,
    target: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    """Original support-stress score over every resampled input position."""
    logits = np.asarray(logits)
    target = np.asarray(target)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if target.ndim != 1 or target.size != logits.shape[0]:
        raise ValueError("target must have shape (N,)")
    valid_part_ids = np.asarray(valid_part_ids, dtype=np.int64)
    pred = valid_part_ids[np.argmax(logits[:, valid_part_ids], axis=1)]
    return _mean_part_iou(pred, target, valid_part_ids)


def unique_retained_shape_part_iou(
    logits: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    """Instance IoU after duplicate-logit averaging on actual entering IDs."""
    logit_ids, averaged = average_duplicate_logits(logits, original_ids)
    target_ids, unique_target = average_targets_by_id(target, original_ids)
    if not np.array_equal(logit_ids, target_ids):
        raise AssertionError("logit and target identity sets differ")
    valid_part_ids = np.asarray(valid_part_ids, dtype=np.int64)
    pred = valid_part_ids[np.argmax(averaged[:, valid_part_ids], axis=1)]
    return _mean_part_iou(pred, unique_target, valid_part_ids)


def full_cloud_nn_propagated_labels(
    logits: np.ndarray,
    full_points: np.ndarray,
    retained_indices: np.ndarray,
    valid_part_ids: Sequence[int],
) -> np.ndarray:
    """Propagate retained predictions to every pre-drop point by xyz 1-NN.

    ``retained_indices`` maps every model-input logit back to its position in
    ``full_points``. Repeated input positions are reduced by mean logits before
    their part label is chosen. A single-threaded KD-tree query keeps the
    nearest-neighbor fill deterministic.
    """
    logits = np.asarray(logits)
    full_points = np.asarray(full_points)
    retained_indices = np.asarray(retained_indices, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if full_points.ndim != 2 or full_points.shape[1] < 3 or full_points.shape[0] == 0:
        raise ValueError("full_points must have non-empty shape (M, >=3)")
    if retained_indices.ndim != 1 or retained_indices.size != logits.shape[0]:
        raise ValueError("retained_indices must have shape (N,)")
    if retained_indices.size == 0:
        raise ValueError("retained_indices must not be empty")
    if retained_indices.min() < 0 or retained_indices.max() >= full_points.shape[0]:
        raise IndexError("retained index is out of range for the full cloud")

    unique_indices, averaged_logits = average_duplicate_logits(logits, retained_indices)
    valid_part_ids = np.asarray(valid_part_ids, dtype=np.int64)
    retained_labels = valid_part_ids[
        np.argmax(averaged_logits[:, valid_part_ids], axis=1)
    ]
    retained_xyz = np.asarray(full_points[unique_indices, :3], dtype=np.float64)
    full_xyz = np.asarray(full_points[:, :3], dtype=np.float64)
    nearest = cKDTree(retained_xyz).query(full_xyz, k=1, workers=1)[1]
    return retained_labels[np.asarray(nearest, dtype=np.int64)]


def full_cloud_nn_propagation_shape_part_iou(
    logits: np.ndarray,
    full_points: np.ndarray,
    full_target: np.ndarray,
    retained_indices: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    """Instance part mIoU on the full pre-drop cloud after retained-label 1-NN."""
    full_target = np.asarray(full_target)
    if full_target.ndim != 1 or full_target.size != np.asarray(full_points).shape[0]:
        raise ValueError("full_target must have shape (M,)")
    propagated = full_cloud_nn_propagated_labels(
        logits,
        full_points,
        retained_indices,
        valid_part_ids,
    )
    return _mean_part_iou(propagated, full_target, valid_part_ids)


def score_both_shape_part_ious(
    logits: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
    valid_part_ids: Sequence[int],
) -> Dict[str, float]:
    """Compute both protocol scorings from one set of per-position logits."""
    return {
        RESAMPLED_2048_WITH_DUPLICATES: resampled_shape_part_iou(
            logits, target, valid_part_ids
        ),
        UNIQUE_RETAINED_DEDUPED: unique_retained_shape_part_iou(
            logits, target, original_ids, valid_part_ids
        ),
    }


def score_all_shape_part_ious(
    logits: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
    valid_part_ids: Sequence[int],
    full_points: np.ndarray,
    full_target: np.ndarray,
    retained_indices: np.ndarray,
) -> Dict[str, float]:
    """Compute all three score spaces from one set of per-position logits."""
    scores = score_both_shape_part_ious(
        logits,
        target,
        original_ids,
        valid_part_ids,
    )
    scores[FULL_CLOUD_NN_PROPAGATION] = full_cloud_nn_propagation_shape_part_iou(
        logits,
        full_points,
        full_target,
        retained_indices,
        valid_part_ids,
    )
    return scores


def shape_part_iou(
    logits: np.ndarray,
    target: np.ndarray,
    original_ids: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    """Backward-compatible name for unique-retained deduped scoring."""
    return unique_retained_shape_part_iou(logits, target, original_ids, valid_part_ids)


def clean_subset_iou(
    clean_logits: np.ndarray,
    clean_target: np.ndarray,
    clean_original_ids: np.ndarray,
    actual_input_ids: np.ndarray,
    valid_part_ids: Sequence[int],
) -> float:
    """Score clean logits on exactly the unique IDs that entered a stressed arm."""
    clean_ids, averaged = average_duplicate_logits(clean_logits, clean_original_ids)
    target_ids, unique_target = average_targets_by_id(clean_target, clean_original_ids)
    if not np.array_equal(clean_ids, target_ids):
        raise AssertionError("clean logit and target identity sets differ")
    wanted = np.unique(np.asarray(actual_input_ids, dtype=np.int64))
    mask = np.isin(clean_ids, wanted)
    if not np.array_equal(clean_ids[mask], wanted):
        missing = np.setdiff1d(wanted, clean_ids)
        raise AssertionError(
            "stressed IDs missing from clean input: %s" % missing.tolist()
        )
    return shape_part_iou(
        averaged[mask],
        unique_target[mask],
        clean_ids[mask],
        valid_part_ids,
    )


def dip_contrast(miou_by_fraction: Mapping[float, float]) -> float:
    needed = set(PREREGISTERED_FRACTIONS)
    if not needed.issubset(set(float(x) for x in miou_by_fraction)):
        raise ValueError("dip contrast requires 80%, 50%, and 20%")
    return float(
        miou_by_fraction[0.8] - 0.5 * (miou_by_fraction[0.5] + miou_by_fraction[0.2])
    )


def reproduction_match_report(
    reproduced: Mapping[str, Mapping[str, float]],
    expected: Mapping[str, float],
    tolerance_pp: float,
) -> Dict[str, object]:
    """Record PASS/NO-MATCH per score space and pass when any matches."""
    tolerance_pp = float(tolerance_pp)
    if tolerance_pp < 0.0:
        raise ValueError("tolerance_pp must be non-negative")
    per_metric = {}
    matched_metrics = []
    for metric in SCORING_METRICS:
        if metric not in reproduced:
            raise KeyError("missing reproduction metric: %s" % metric)
        differences = {
            condition: float(reproduced[metric][condition] - expected_value)
            for condition, expected_value in expected.items()
        }
        matched = all(
            abs(difference) <= tolerance_pp for difference in differences.values()
        )
        per_metric[metric] = {
            "status": "PASS" if matched else "NO-MATCH",
            "matched": bool(matched),
            "difference_pp": differences,
        }
        if matched:
            matched_metrics.append(metric)
    return {
        "passed": bool(matched_metrics),
        "status": "PASS" if matched_metrics else "NO-MATCH",
        "matched_metrics": matched_metrics,
        "per_metric": per_metric,
        "tolerance_pp": tolerance_pp,
    }


def paired_summary(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty paired sample")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": std,
        "se": float(std / np.sqrt(values.size)),
        "n": int(values.size),
    }


def group_mode_viability(
    clean_miou: float,
    minimum_clean_miou: Optional[float],
    canonical: bool,
) -> Dict[str, object]:
    """Guard against calling a secondary-mode floor collapse an artifact."""
    if canonical:
        return {
            "viable": True,
            "reason": "canonical fps_knn is the primary estimand",
            "minimum_clean_miou": minimum_clean_miou,
        }
    if minimum_clean_miou is None:
        raise ValueError("secondary group modes require --min_clean_miou")
    viable = float(clean_miou) >= float(minimum_clean_miou)
    return {
        "viable": bool(viable),
        "reason": (
            "secondary clean performance meets the user-specified floor"
            if viable
            else "floor-effect guard: artifact interpretation is forbidden"
        ),
        "minimum_clean_miou": float(minimum_clean_miou),
    }


def assert_artifact_reading_allowed(
    viability_rows: Sequence[Mapping[str, object]],
) -> None:
    failed = [row for row in viability_rows if not bool(row["viable"])]
    if failed:
        raise AssertionError(
            "floor-effect guard failed; collapsed OOD grouping cannot support an artifact reading"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
