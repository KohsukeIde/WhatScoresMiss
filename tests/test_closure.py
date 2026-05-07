import pytest

from bds.closure import (
    coordinate_only_closure,
    gap_recovery,
    target_swap_sensitivity,
    within_run_loss_drop,
)


def test_target_swap_sensitivity():
    assert target_swap_sensitivity(10.0, [11.0, 13.0]) == pytest.approx(30.0)


def test_coordinate_only_closure():
    assert coordinate_only_closure(5.0, 8.0, 6.5) == pytest.approx(50.0)


def test_coordinate_only_closure_bad_denominator():
    with pytest.raises(ValueError):
        coordinate_only_closure(5.0, 4.0, 4.5)


def test_within_run_loss_drop():
    assert within_run_loss_drop(10.0, 2.5) == pytest.approx(75.0)


def test_gap_recovery():
    assert gap_recovery(47.94, 40.64, 40.10) == pytest.approx(6.8877551)
