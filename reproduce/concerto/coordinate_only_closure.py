"""Coordinate-only closure aggregation for Concerto pretext diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bds.closure import coordinate_only_closure


def aggregate(ledger: Path, output: Path) -> pd.DataFrame:
    df = pd.read_csv(ledger)
    value_col = "loss" if "loss" in df.columns else "value"
    rows = []
    for dataset, group in df.groupby("dataset", sort=False):
        clean = float(group.loc[group["condition"] == "clean", value_col].iloc[0])
        mean_swap = float(group.loc[group["condition"] == "mean_swap", value_col].iloc[0])
        coord = float(group.loc[group["condition"] == "coordinate_only", value_col].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "clean_loss": clean,
                "mean_swap_loss": mean_swap,
                "coordinate_only_loss": coord,
                "xyz_closure_percent": coordinate_only_closure(clean, mean_swap, coord),
                "source_ledger": str(ledger),
            }
        )
    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reproduce/concerto/configs/scannet_target_swap.yaml")
    parser.add_argument("--closure-ledger", default="results/ledgers/concerto_closure_fit_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/concerto_coordinate_closure_from_ledger.csv")
    args = parser.parse_args()
    print(aggregate(Path(args.closure_ledger), Path(args.output)).to_string(index=False))


if __name__ == "__main__":
    main()
