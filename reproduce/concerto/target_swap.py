"""Concerto target-swap diagnostic aggregation.

The full target extraction/inference loop depends on an external
Concerto-compatible codebase. This script implements the paper diagnostic
aggregation from a loss provenance summary and prints the exact external
command when the model code is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate(loss_ledger: Path, output: Path) -> pd.DataFrame:
    records = pd.read_csv(loss_ledger)
    value_col = "loss" if "loss" in records.columns else "value"
    rows = []
    for dataset, group in records.groupby("dataset", sort=False):
        clean = float(group.loc[group["condition"] == "clean", value_col].iloc[0])
        swaps = group[group["condition"] != "clean"]
        max_row = swaps.loc[swaps[value_col].idxmax()]
        rows.append(
            {
                "dataset": dataset,
                "clean_loss": clean,
                "max_swap_condition": max_row["condition"],
                "max_swap_loss": float(max_row[value_col]),
                "target_sensitivity_percent": 100.0 * (float(max_row[value_col]) - clean) / clean,
                "source_ledger": str(loss_ledger),
            }
        )
    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reproduce/concerto/configs/scannet_target_swap.yaml")
    parser.add_argument("--loss-ledger", default="results/ledgers/concerto_loss_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/concerto_target_swap_from_ledger.csv")
    parser.add_argument("--external-code-root", default=None)
    args = parser.parse_args()

    out = aggregate(Path(args.loss_ledger), Path(args.output))
    print(out.to_string(index=False))
    if args.external_code_root:
        print("External run entry point:")
        print(
            "python <external_concerto>/tools/target_swap_eval.py "
            f"--config {args.config} --output-root $OUTPUT_ROOT"
        )
    else:
        print("External Concerto code not bundled; aggregation ran from the sanitized provenance summary.")


if __name__ == "__main__":
    main()
