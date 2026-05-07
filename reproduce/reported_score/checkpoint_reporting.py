"""Aggregate checkpoint selection protocol ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="results/ledgers/scanobjectnn_checkpoint_selection_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/checkpoint_reporting_from_ledger.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.ledger)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
