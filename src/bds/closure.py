"""Pretext and representation-side scalar diagnostics."""

from collections.abc import Sequence


def target_swap_sensitivity(clean_loss: float, swap_losses: Sequence[float]) -> float:
    """Return 100 * max(swap_loss - clean_loss) / clean_loss."""
    if clean_loss == 0:
        raise ValueError("clean_loss must be non-zero")
    if not swap_losses:
        raise ValueError("swap_losses must be non-empty")
    return 100.0 * (max(swap_losses) - clean_loss) / clean_loss


def coordinate_only_closure(
    clean_loss: float, mean_swap_loss: float, coord_loss: float
) -> float:
    """Return the percent of swap-induced loss increase closed by coord-only loss."""
    denom = mean_swap_loss - clean_loss
    if denom <= 0:
        raise ValueError("mean_swap_loss must be greater than clean_loss")
    return 100.0 * (mean_swap_loss - coord_loss) / denom


def within_run_loss_drop(first_loss: float, final_loss: float) -> float:
    """Return percent loss decrease from first to final loss."""
    if first_loss == 0:
        raise ValueError("first_loss must be non-zero")
    return 100.0 * (first_loss - final_loss) / first_loss


def gap_recovery(
    full_score: float, candidate_score: float, lower_reference_score: float
) -> float:
    """Return candidate recovery of the lower-reference to full-score gap."""
    denom = full_score - lower_reference_score
    if denom == 0:
        raise ValueError("full_score and lower_reference_score must differ")
    return 100.0 * (candidate_score - lower_reference_score) / denom
