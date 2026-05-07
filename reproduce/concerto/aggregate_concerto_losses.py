"""Build Concerto diagnostic summary tables from sanitized ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

from reproduce.concerto.coordinate_only_closure import aggregate as aggregate_closure
from reproduce.concerto.target_swap import aggregate as aggregate_swap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-ledger", default="results/ledgers/concerto_loss_runs.csv")
    parser.add_argument("--closure-ledger", default="results/ledgers/concerto_closure_fit_runs.csv")
    parser.add_argument("--output-dir", default="results/generated_tables")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    aggregate_swap(Path(args.loss_ledger), out_dir / "concerto_target_swap_from_ledger.csv")
    aggregate_closure(Path(args.closure_ledger), out_dir / "concerto_coordinate_closure_from_ledger.csv")
    print(f"Wrote Concerto aggregates to {out_dir}")


if __name__ == "__main__":
    main()
