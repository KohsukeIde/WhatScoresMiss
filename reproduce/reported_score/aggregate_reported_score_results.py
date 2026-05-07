"""Build reported-score diagnostic outputs from sanitized ledgers."""

from __future__ import annotations

from reproduce.reported_score.checkpoint_reporting import main as checkpoint_main
from reproduce.reported_score.class_pair_diagnostics import main as class_pair_main
from reproduce.reported_score.oracle_from_logits import main as oracle_main
from reproduce.reported_score.posthoc_readout_checks import main as posthoc_main


if __name__ == "__main__":
    oracle_main()
    class_pair_main()
    posthoc_main()
    checkpoint_main()
