import numpy as np
import pytest

from bds.support import (
    random_keep_indices,
    semantic_keep_indices,
    structured_keep_by_anchor,
    structured_keep_by_grid,
    validate_keep_fraction,
)


def test_random_keep_indices():
    idx = random_keep_indices(100, 0.2, seed=0)
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20


def test_structured_keep_by_grid():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(100, 3))
    idx = structured_keep_by_grid(xyz, 0.3, seed=1, grid_size=4)
    assert 1 <= len(idx) <= 30
    assert len(np.unique(idx)) == len(idx)


def test_structured_keep_by_anchor():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(50, 3))
    idx = structured_keep_by_anchor(xyz, 0.4, seed=2)
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20


def test_semantic_keep_indices():
    labels = np.array([0] * 10 + [1] * 20)
    idx = semantic_keep_indices(labels, 0.5, seed=0)
    assert (labels[idx] == 0).sum() == 5
    assert (labels[idx] == 1).sum() == 10


def test_validate_keep_fraction():
    with pytest.raises(ValueError):
        validate_keep_fraction(0.0)
    with pytest.raises(ValueError):
        validate_keep_fraction(1.1)
