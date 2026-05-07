#!/usr/bin/env bash
set -euo pipefail

pytest -q

python - <<'PY'
import numpy as np

from bds.closure import gap_recovery, target_swap_sensitivity
from bds.oracle import oracle_accuracy
from bds.support import random_keep_indices, semantic_keep_indices, structured_keep_by_anchor

rng = np.random.default_rng(0)
xyz = rng.normal(size=(32, 3))
labels = np.array([0] * 16 + [1] * 16)
logits = rng.normal(size=(32, 3))
target = rng.integers(0, 3, size=32)

assert len(random_keep_indices(32, 0.5, 0)) == 16
assert len(structured_keep_by_anchor(xyz, 0.25, 1)) == 8
assert len(semantic_keep_indices(labels, 0.5, 2)) == 16
assert 0.0 <= oracle_accuracy(logits, target, 2) <= 1.0
assert target_swap_sensitivity(5.0, [6.0, 7.0]) == 40.0
assert gap_recovery(47.94, 40.64, 40.10) > 0.0

print("bds smoke test passed")
PY
