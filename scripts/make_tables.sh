#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/generated_tables

python - <<'PY'
from pathlib import Path

import pandas as pd

from bds.io import dataframe_to_markdown

src = Path("results/paper_reported_results")
dst = Path("results/generated_tables")
dst.mkdir(parents=True, exist_ok=True)

for csv_path in sorted(src.glob("*.csv")):
    df = pd.read_csv(csv_path)
    out = dst / f"{csv_path.stem}.md"
    out.write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(f"wrote {out}")
PY
