"""Class-pair diagnostic aggregation and optional feature-probe computation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bds.class_pair import pairwise_balanced_accuracy, target_to_confusion_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-npz", default=None, help="Optional NPZ with features, labels, pred arrays.")
    parser.add_argument("--class-a", type=int, default=None)
    parser.add_argument("--class-b", type=int, default=None)
    parser.add_argument("--ledger", default="results/ledgers/class_pair_diagnostic_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/class_pair_from_ledger.csv")
    args = parser.parse_args()

    if args.features_npz:
        data = np.load(args.features_npz)
        if args.class_a is None or args.class_b is None:
            raise ValueError("--class-a and --class-b are required with --features-npz")
        print(
            {
                "pairwise_balanced_accuracy": pairwise_balanced_accuracy(
                    data["features"], data["labels"], args.class_a, args.class_b
                ),
                "target_to_confusion_rate": target_to_confusion_rate(
                    data["pred"], data["labels"], args.class_a, args.class_b
                ),
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
