"""Object-level support perturbations for point clouds stored in NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bds.support import random_keep_indices, semantic_keep_indices, structured_keep_by_anchor


def keep_indices(xyz: np.ndarray, labels: np.ndarray, perturbation: str, keep_fraction: float, seed: int) -> np.ndarray:
    if perturbation == "random_keep":
        return random_keep_indices(len(xyz), keep_fraction, seed)
    if perturbation == "structured_keep":
        return structured_keep_by_anchor(xyz, keep_fraction, seed)
    if perturbation == "semantic_keep":
        return semantic_keep_indices(labels, keep_fraction, seed)
    if perturbation == "clean":
        return np.arange(len(xyz))
    raise ValueError(f"unknown perturbation: {perturbation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True, help="NPZ with xyz and labels arrays.")
    parser.add_argument("--perturbation", default="random_keep")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    xyz = data["xyz"]
    labels = data["labels"]
    idx = keep_indices(xyz, labels, args.perturbation, args.keep_fraction, args.seed)
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, xyz=xyz[idx], labels=labels[idx], original_indices=idx)
    print(f"wrote {args.output_npz} with {len(idx)} retained points")


if __name__ == "__main__":
    main()
