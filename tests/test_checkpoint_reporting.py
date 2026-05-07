import pandas as pd

from bds.checkpoint_reporting import compare_selection_protocols, select_checkpoint


def test_select_checkpoint():
    records = pd.DataFrame(
        [
            {"epoch": 1, "split": "val", "acc": 0.5},
            {"epoch": 2, "split": "val", "acc": 0.7},
            {"epoch": 2, "split": "test", "acc": 0.6},
        ]
    )
    row = select_checkpoint(records, "val", "acc")
    assert int(row["epoch"]) == 2


def test_compare_selection_protocols():
    records = pd.DataFrame(
        [
            {"epoch": 1, "split": "test", "acc": 0.9},
            {"epoch": 2, "split": "test", "acc": 0.8},
            {"epoch": 1, "split": "heldout", "acc": 0.4},
            {"epoch": 2, "split": "heldout", "acc": 0.7},
        ]
    )
    out = compare_selection_protocols(records, "test", "heldout", "test", "acc")
    assert out["standard_epoch"] == 1
    assert out["heldout_epoch"] == 2
    assert out["standard_report_score"] == 0.9
    assert out["heldout_report_score"] == 0.8
