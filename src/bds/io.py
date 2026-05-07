"""Small I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple GitHub-flavored markdown table without extra deps."""
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append(["" if pd.isna(v) else str(v) for v in row.tolist()])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataframe_to_markdown(df), encoding="utf-8")
