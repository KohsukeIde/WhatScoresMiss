"""Checkpoint selection and reporting protocol helpers."""

from __future__ import annotations

import pandas as pd


def select_checkpoint(
    records: pd.DataFrame, selection_split: str, metric: str, mode: str = "max"
) -> pd.Series:
    """Select the best checkpoint by split and metric."""
    if mode not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")
    subset = records[records["split"] == selection_split]
    if subset.empty:
        raise ValueError(f"no records for selection_split={selection_split!r}")
    idx = subset[metric].idxmax() if mode == "max" else subset[metric].idxmin()
    return subset.loc[idx]


def report_selected(
    records: pd.DataFrame, selected_epoch: int, report_split: str, metric: str
) -> float:
    """Report a metric for the selected epoch on report_split."""
    subset = records[(records["epoch"] == selected_epoch) & (records["split"] == report_split)]
    if subset.empty:
        raise ValueError("no record for selected epoch and report split")
    if len(subset) > 1:
        subset = subset.sort_values("epoch").iloc[:1]
        return float(subset[metric].iloc[0])
    return float(subset.iloc[0][metric])


def compare_selection_protocols(
    records: pd.DataFrame,
    standard_split: str,
    heldout_split: str,
    report_split: str,
    metric: str,
) -> dict:
    """Compare target-as-selection and held-out selection protocols."""
    standard = select_checkpoint(records, standard_split, metric)
    heldout = select_checkpoint(records, heldout_split, metric)
    standard_score = report_selected(records, int(standard["epoch"]), report_split, metric)
    heldout_score = report_selected(records, int(heldout["epoch"]), report_split, metric)
    return {
        "standard_epoch": int(standard["epoch"]),
        "heldout_epoch": int(heldout["epoch"]),
        "standard_report_score": standard_score,
        "heldout_report_score": heldout_score,
        "delta": standard_score - heldout_score,
    }
