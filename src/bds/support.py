"""Support perturbation utilities for synthetic smoke tests."""

from __future__ import annotations

import numpy as np


def validate_keep_fraction(keep_fraction: float) -> None:
    """Require 0 < keep_fraction <= 1."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must satisfy 0 < keep_fraction <= 1")


def _keep_count(n_points: int, keep_fraction: float) -> int:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    validate_keep_fraction(keep_fraction)
    return max(1, int(round(n_points * keep_fraction)))


def random_keep_indices(n_points: int, keep_fraction: float, seed: int) -> np.ndarray:
    """Uniformly retain approximately keep_fraction of points."""
    keep_n = _keep_count(n_points, keep_fraction)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=keep_n, replace=False))


def structured_keep_by_grid(
    xyz: np.ndarray, keep_fraction: float, seed: int, grid_size: int = 8
) -> np.ndarray:
    """Retain whole shuffled grid cells until the target support size is reached."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    keep_n = _keep_count(len(xyz), keep_fraction)
    rng = np.random.default_rng(seed)

    mins = xyz.min(axis=0)
    spans = np.maximum(xyz.max(axis=0) - mins, 1e-12)
    norm = (xyz - mins) / spans
    bins = np.minimum((norm * grid_size).astype(int), grid_size - 1)
    cell_ids = bins[:, 0] * grid_size * grid_size + bins[:, 1] * grid_size + bins[:, 2]

    cells = np.unique(cell_ids)
    rng.shuffle(cells)
    kept: list[int] = []
    for cell in cells:
        kept.extend(np.flatnonzero(cell_ids == cell).tolist())
        if len(kept) >= keep_n:
            break
    if len(kept) > keep_n:
        kept = rng.choice(np.asarray(kept), size=keep_n, replace=False).tolist()
    return np.sort(np.asarray(kept, dtype=int))


def structured_keep_by_anchor(
    xyz: np.ndarray, keep_fraction: float, seed: int
) -> np.ndarray:
    """Choose a random anchor and retain the farthest points from it."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    keep_n = _keep_count(len(xyz), keep_fraction)
    rng = np.random.default_rng(seed)
    anchor = xyz[rng.integers(0, len(xyz))]
    dist = np.linalg.norm(xyz - anchor, axis=1)
    return np.sort(np.argsort(dist)[-keep_n:])


def semantic_keep_indices(labels: np.ndarray, keep_fraction: float, seed: int) -> np.ndarray:
    """Retain approximately keep_fraction within each semantic label."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    validate_keep_fraction(keep_fraction)
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        keep_n = max(1, int(round(len(idx) * keep_fraction)))
        kept.extend(rng.choice(idx, size=keep_n, replace=False).tolist())
    return np.sort(np.asarray(kept, dtype=int))
