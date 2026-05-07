"""Score segmentation predictions on unique retained original points."""

from __future__ import annotations

import argparse

import numpy as np

from bds.retained_scoring import average_duplicate_logits, mean_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-npz", required=True, help="NPZ with logits, target, and original_indices arrays.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=None)
    args = parser.parse_args()

    data = np.load(args.logits_npz)
    unique_idx, avg_logits = average_duplicate_logits(data["logits"], data["original_indices"])
    target_full = data["target"]
    target = target_full[unique_idx]
    pred = avg_logits.argmax(axis=1)
    print(mean_iou(pred, target, args.num_classes, args.ignore_index))


if __name__ == "__main__":
    main()
