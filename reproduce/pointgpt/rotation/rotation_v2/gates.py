"""Reusable fail-closed clean and identity replay gates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare canonical contiguous tensor bytes, including signed zero bits."""
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != torch.strided
        or right.layout != torch.strided
    ):
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def apply_rotation_xyz(
    points: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply active ``R`` to row-vector XYZ as ``points @ R.T``."""
    if not torch.is_tensor(points) or not torch.is_tensor(rotation):
        raise TypeError("points and rotation must be torch tensors")
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must end in XYZ shape 3")
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if points.dtype != rotation.dtype or points.device != rotation.device:
        raise TypeError("points and rotation must have identical dtype and device")
    if not torch.is_floating_point(points):
        raise TypeError("points must use a floating dtype")
    if not bool(torch.isfinite(points).all()) or not bool(
        torch.isfinite(rotation).all()
    ):
        raise FloatingPointError("points and rotation must be finite")
    return torch.matmul(points, rotation.transpose(0, 1))


def _rng_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cpu": torch.random.get_rng_state().clone()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        snapshot["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return snapshot


def _rng_restore(snapshot: dict[str, Any]) -> None:
    torch.random.set_rng_state(snapshot["cpu"])
    if "cuda" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _all_modules_eval(model: torch.nn.Module) -> bool:
    return all(not module.training for module in model.modules())


def assert_dual_clean_gate(
    *,
    model: torch.nn.Module,
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    identity_matrix: torch.Tensor,
    logits_atol: float = 0.0,
    logits_rtol: float = 0.0,
) -> dict[str, object]:
    """Compare a bypass forward with explicit identity under the same RNG.

    Input equality is always bitwise. Logits are bitwise when both tolerances
    are zero. A nonzero tolerance must be preregistered by the caller.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not _all_modules_eval(model):
        raise AssertionError("dual clean gate requires eval() on every module")
    if logits_atol < 0.0 or logits_rtol < 0.0:
        raise ValueError("logit tolerances must be non-negative")
    identity_points = apply_rotation_xyz(points, identity_matrix)
    input_bitwise_equal = tensor_bitwise_equal(points, identity_points)
    if not input_bitwise_equal:
        raise AssertionError("identity replay changed input bytes")

    initial_rng = _rng_snapshot()
    after_bypass_rng: dict[str, Any] | None = None
    try:
        with torch.inference_mode():
            bypass_logits = forward_logits(points)
        after_bypass_rng = _rng_snapshot()
        _rng_restore(initial_rng)
        with torch.inference_mode():
            identity_logits = forward_logits(identity_points)
    finally:
        _rng_restore(after_bypass_rng if after_bypass_rng is not None else initial_rng)
    if not torch.is_tensor(bypass_logits) or not torch.is_tensor(identity_logits):
        raise TypeError("forward_logits must return a torch tensor")
    if bypass_logits.shape != identity_logits.shape:
        raise AssertionError("clean and identity logits have different shapes")
    if not bool(torch.isfinite(bypass_logits).all()) or not bool(
        torch.isfinite(identity_logits).all()
    ):
        raise FloatingPointError("dual clean gate logits must be finite")

    bitwise_logits = tensor_bitwise_equal(bypass_logits, identity_logits)
    if logits_atol == 0.0 and logits_rtol == 0.0:
        if not bitwise_logits:
            raise AssertionError("clean and identity logits are not bitwise equal")
        maximum_absolute_error = 0.0
    else:
        if not torch.allclose(
            bypass_logits,
            identity_logits,
            atol=logits_atol,
            rtol=logits_rtol,
        ):
            raise AssertionError("clean and identity logits exceed tolerance")
        maximum_absolute_error = float(
            torch.max(torch.abs(bypass_logits - identity_logits)).item()
        )
    return {
        "input_bitwise_equal": True,
        "logits_bitwise_equal": bitwise_logits,
        "logits_atol": float(logits_atol),
        "logits_rtol": float(logits_rtol),
        "maximum_absolute_logit_error": maximum_absolute_error,
        "same_forward_rng_replayed": True,
    }
