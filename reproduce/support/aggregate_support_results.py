"""Aggregate scene/object support ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ledger", default="results/ledgers/scannet_support_eval_runs.csv")
    parser.add_argument("--object-ledger", default="results/ledgers/shapenetpart_support_eval_runs.csv")
    parser.add_argument("--output", default="results/generated_tables/support_from_ledgers.csv")
    args = parser.parse_args()
    frames = [pd.read_csv(args.scene_ledger), pd.read_csv(args.object_ledger)]
    out = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
