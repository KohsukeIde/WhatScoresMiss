"""Nearest-neighbor propagation from retained scene points to full scenes."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-npz", required=True, help="NPZ with retained xyz and pred arrays.")
    parser.add_argument("--full-npz", required=True, help="NPZ with full xyz and labels arrays.")
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    retained = np.load(args.retained_npz)
    full = np.load(args.full_npz)
    tree = cKDTree(retained["xyz"])
    _, nn = tree.query(full["xyz"], k=1)
    pred_full = retained["pred"][nn]
    np.savez(args.output_npz, pred=pred_full, target=full["labels"], xyz=full["xyz"])
    print(f"wrote propagated predictions to {args.output_npz}")


if __name__ == "__main__":
    main()
