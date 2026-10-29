"""Compare retained-point scoring against the original PointGPT R7 scorer."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bds.retained_scoring import unique_retained_shape_part_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to original rebuttal_eval/r7.py")
    parser.add_argument("--reference-csv", required=True, help="Path to original replay support-curve CSV")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("pointgpt_reference", args.reference)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load reference scorer")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(2026)
    for _ in range(200):
        valid = np.sort(rng.choice(50, size=int(rng.integers(2, 7)), replace=False))
        count = int(rng.integers(2, 80))
        ids = rng.integers(0, max(2, count // 2), size=count)
        original_target = rng.choice(valid, size=int(ids.max()) + 1)
        target = original_target[ids]
        logits = rng.normal(size=(count, 50))
        actual = unique_retained_shape_part_iou(logits, target, ids, valid)
        expected = reference.unique_retained_shape_part_iou(logits, target, ids, valid)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shape.npz"
        np.savez(path, logits=logits, target=target, original_indices=ids, valid_part_ids=valid)
        result = subprocess.check_output(
            [sys.executable, "-m", "reproduce.support.score_retained_points", "--logits-npz", str(path)],
            text=True,
        )
        expected_text = f"instance_miou_percent={100.0 * expected:.8f} shapes=1"
        if result.strip() != expected_text:
            raise AssertionError(f"CLI mismatch: {result.strip()} != {expected_text}")
    try:
        unique_retained_shape_part_iou(
            np.ones((2, 3)), np.array([0, 1]), np.array([4, 4]), np.array([0, 1])
        )
    except ValueError as exc:
        if "inconsistent targets" not in str(exc):
            raise
    else:
        raise AssertionError("conflicting labels for one original ID were accepted")

    def rows(path):
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    original = rows(args.reference_csv)
    published = [r for r in rows("results/paper_reported_results/support_shapenetpart_curves.csv")
                 if r["model"] == "PointGPT-S official"]
    if len(original) != 13 or len(published) != 13:
        raise AssertionError("expected 13 PointGPT conditions")
    original = {r["condition"]: r for r in original}
    for row in published:
        source = original[row["condition"]]
        for key in ("instance_miou_percent", "clean_subset_instance_miou_percent", "damage_pp"):
            np.testing.assert_allclose(float(row[key]), float(source[key]), rtol=0, atol=1e-12)
    ledger = [r for r in rows("results/ledgers/shapenetpart_support_eval_runs.csv")
              if r["model"] == "PointGPT-S official"]
    if len(ledger) != 13:
        raise AssertionError("expected 13 PointGPT ledger rows")
    for row in ledger:
        np.testing.assert_allclose(float(row["value"]),
                                   float(original[row["condition"]]["instance_miou_percent"]),
                                   rtol=0, atol=1e-12)
    print("PASS: 200 shapes match the original scorer; CLI and all 13 replay rows match")


if __name__ == "__main__":
    main()
