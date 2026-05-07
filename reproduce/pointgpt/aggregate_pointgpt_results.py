"""Aggregate PointGPT mask/order relaxation ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretext-ledger", default="results/ledgers/pointgpt_pretext_logs.csv")
    parser.add_argument("--downstream-ledger", default="results/ledgers/pointgpt_downstream_eval_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/pointgpt_mask_order_from_ledgers.csv")
    args = parser.parse_args()
    pre = pd.read_csv(args.pretext_ledger)
    down = pd.read_csv(args.downstream_ledger)
    rows = []
    for _, row in down.iterrows():
        variant = row["condition"]
        pre_row = pre[pre["condition"] == variant]
        rows.append(
            {
                "condition": variant,
                "dataset": row["dataset"],
                "metric": row["metric"],
                "value": row["value"],
                "pretext_loss_drop_percent": None if pre_row.empty else pre_row["value"].iloc[0],
                "checkpoint_source": row["checkpoint_source"],
                "seed": row["seed"],
            }
        )
    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
