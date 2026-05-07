"""Scene-level support perturbations for normalized XYZ arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bds.support import random_keep_indices, structured_keep_by_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True, help="NPZ with xyz and labels arrays.")
    parser.add_argument("--perturbation", choices=["clean", "random_keep", "structured_keep"], default="structured_keep")
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    xyz = data["xyz"]
    labels = data["labels"]
    if args.perturbation == "clean":
        idx = np.arange(len(xyz))
    elif args.perturbation == "random_keep":
        idx = random_keep_indices(len(xyz), args.keep_fraction, args.seed)
    else:
        idx = structured_keep_by_grid(xyz, args.keep_fraction, args.seed)
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, xyz=xyz[idx], labels=labels[idx], original_indices=idx)
    print(f"wrote {args.output_npz} with {len(idx)} retained points")


if __name__ == "__main__":
    main()
