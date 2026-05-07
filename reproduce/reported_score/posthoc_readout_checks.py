"""Aggregate post-hoc readout diagnostic provenance summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="results/ledgers/scannet_oracle_eval_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/posthoc_readout_from_ledger.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.ledger)
    df = df[df["metric"].isin(["top1_miou", "best_posthoc_miou"])]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
