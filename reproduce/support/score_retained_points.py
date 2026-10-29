"""Compute ShapeNetPart instance mIoU from per-shape retained-point logits."""

from __future__ import annotations

import argparse

import numpy as np

from bds.retained_scoring import unique_retained_shape_part_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logits-npz", nargs="+", required=True,
        help="One NPZ per shape with logits, target, original_indices, and valid_part_ids.",
    )
    args = parser.parse_args()
    scores = []
    for path in args.logits_npz:
        with np.load(path, allow_pickle=False) as data:
            scores.append(unique_retained_shape_part_iou(
                data["logits"], data["target"], data["original_indices"], data["valid_part_ids"]
            ))
    print(f"instance_miou_percent={100.0 * np.mean(scores):.8f} shapes={len(scores)}")


if __name__ == "__main__":
    main()
