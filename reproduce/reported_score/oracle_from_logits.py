"""Compute Oracle@k metrics from logits, or aggregate sanitized oracle provenance summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bds.oracle import oracle_accuracy, oracle_miou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-npz", default=None, help="Optional NPZ with logits and target arrays.")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ledger", default="results/ledgers/scannet_oracle_eval_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/oracle_from_ledger.csv")
    args = parser.parse_args()

    if args.logits_npz:
        data = np.load(args.logits_npz)
        logits = data["logits"]
        target = data["target"]
        if target.ndim == 1:
            print({"oracle2_acc": oracle_accuracy(logits, target, 2), "oracle5_acc": oracle_accuracy(logits, target, 5)})
        else:
            if args.num_classes is None:
                raise ValueError("--num-classes is required for mIoU")
            print(
                {
                    "oracle2_miou": oracle_miou(logits, target, 2, args.num_classes),
                    "oracle5_miou": oracle_miou(logits, target, 5, args.num_classes),
                }
            )
        return

    df = pd.read_csv(args.ledger)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
