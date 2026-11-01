"""Controlled attention-locality interventions for PTv3 serialization.

This module intentionally operates on the padded attention layout only.  The
geometric ``serialized_code/order/inverse`` tensors remain the source of truth
for SerializedPooling.
"""

from __future__ import annotations

import hashlib

import torch


ATTENTION_ORDER_MODES = {
    "off",
    "z_only",
    "hilbert_only",
    "locality_preserving",
    "locality_destroying",
}
LOCALITY_MODES = {"locality_preserving", "locality_destroying"}


def normalize_attention_order_mode(mode):
    if mode is None:
        return "off"
    mode = str(mode).lower().replace("-", "_")
    aliases = {
        "none": "off",
        "baseline": "off",
        "original": "off",
        "forced_z_only": "z_only",
        "forced_hilbert_only": "hilbert_only",
        "preserve": "locality_preserving",
        "destroy": "locality_destroying",
    }
    mode = aliases.get(mode, mode)
    if mode not in ATTENTION_ORDER_MODES:
        raise ValueError(
            f"Unsupported attention_order_mode={mode!r}; "
            f"expected one of {sorted(ATTENTION_ORDER_MODES)}"
        )
    return mode


def resolve_serialization_order(order, mode):
    """Resolve the valid geometric family used by each ladder condition."""
    mode = normalize_attention_order_mode(mode)
    order = [order] if isinstance(order, str) else list(order)
    if mode == "z_only":
        return ["z"]
    if mode == "hilbert_only":
        return ["hilbert"]
    return order


def deterministic_family_permutation(num_orders, seed, stage):
    """Return a local-generator permutation without consuming global RNG."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_mix_seed(seed, stage, num_orders, 0x5E71A1))
    return torch.randperm(num_orders, generator=generator)


def _mix_seed(*values):
    payload = ":".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & 0x7FFFFFFFFFFFFFFF


def _sample_fingerprint(grid_coord, point_ids):
    coord = (
        grid_coord.index_select(0, point_ids)
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
    )
    return int.from_bytes(
        hashlib.blake2b(coord.numpy().tobytes(), digest_size=8).digest(), "little"
    ) & 0x7FFFFFFFFFFFFFFF


def _randperm(length, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator).to(device=device)


def _window_ids(length, cu_seqlens, device):
    window_ids = torch.empty(length, dtype=torch.long, device=device)
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    for window_index, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        window_ids[begin:end] = window_index
    return window_ids


def _mean_within_window_distance(grid_coord, order, cu_seqlens):
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    distance_sum = grid_coord.new_zeros((), dtype=torch.float64)
    pair_count = 0
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        size = end - begin
        if size < 2:
            continue
        coord = grid_coord.index_select(0, order[begin:end]).to(torch.float64)
        distance_sum = distance_sum + torch.pdist(coord).sum()
        pair_count += size * (size - 1) // 2
    if pair_count == 0:
        return 0.0
    return (distance_sum / pair_count).item()


def _same_window_membership(base_order, changed_order, cu_seqlens):
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        base = torch.sort(base_order[begin:end]).values
        changed = torch.sort(changed_order[begin:end]).values
        if not torch.equal(base, changed):
            return False
    return True


@torch.no_grad()
def build_attention_layout(
    *,
    grid_coord,
    batch,
    base_order,
    serialized_inverse,
    pad,
    unpad,
    cu_seqlens,
    mode,
    seed,
    stage,
    order_index,
):
    """Build a deterministic padded attention order and its exact inverse.

    ``base_order`` and ``serialized_inverse`` are never modified.  Padding can
    duplicate tokens, so a canonical-occurrence mask is permuted alongside the
    padded order; the returned inverse always selects exactly one occurrence
    for every unpadded point.
    """
    mode = normalize_attention_order_mode(mode)
    if mode not in LOCALITY_MODES:
        raise ValueError(f"build_attention_layout requires a locality mode, got {mode}")

    device = base_order.device
    padded_order = base_order.index_select(0, pad)
    num_points = base_order.numel()
    canonical_positions = unpad.index_select(0, serialized_inverse)
    canonical = torch.zeros(padded_order.numel(), dtype=torch.bool, device=device)
    canonical[canonical_positions] = True

    source_index = torch.arange(padded_order.numel(), device=device)
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()

    if mode == "locality_preserving":
        sample_window_counts = {}
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            length = end - begin
            if length <= 1:
                continue
            batch_index = int(batch[padded_order[begin]].item())
            sample_window_index = sample_window_counts.get(batch_index, 0)
            sample_window_counts[batch_index] = sample_window_index + 1
            sample_points = torch.nonzero(
                batch == batch_index, as_tuple=False
            ).flatten()
            sample_seed = _mix_seed(
                seed,
                stage,
                order_index,
                sample_window_index,
                _sample_fingerprint(grid_coord, sample_points),
            )
            perm = _randperm(length, sample_seed, device)
            if torch.equal(perm, torch.arange(length, device=device)):
                perm = torch.roll(perm, shifts=1)
            source_index[begin:end] = source_index[begin:end].index_select(0, perm)
    else:
        padded_batch = batch.index_select(0, padded_order)
        _, sample_counts = torch.unique_consecutive(
            padded_batch, return_counts=True
        )
        sample_begin = 0
        for sample_count in (
            sample_counts.detach().to(device="cpu", dtype=torch.long).tolist()
        ):
            sample_end = sample_begin + sample_count
            sample_points = padded_order[sample_begin:sample_end].unique(sorted=True)
            sample_seed = _mix_seed(
                seed,
                stage,
                order_index,
                _sample_fingerprint(grid_coord, sample_points),
            )
            perm = _randperm(sample_count, sample_seed, device)
            candidate = source_index[sample_begin:sample_end].index_select(0, perm)

            # A random draw can very rarely leave all window memberships intact.
            # Rotate across a boundary to make the destructive condition genuine.
            if sample_count > 1:
                trial_index = source_index.clone()
                trial_index[sample_begin:sample_end] = candidate
                trial_order = padded_order.index_select(0, trial_index)
                sample_windows = sum(
                    sample_begin <= begin < sample_end for begin in boundaries[:-1]
                )
                if (
                    sample_windows > 1
                    and _same_window_membership(
                        padded_order[sample_begin:sample_end],
                        trial_order[sample_begin:sample_end],
                        cu_seqlens[
                            (cu_seqlens >= sample_begin)
                            & (cu_seqlens <= sample_end)
                        ]
                        - sample_begin,
                    )
                ):
                    candidate = torch.roll(candidate, shifts=1)
            source_index[sample_begin:sample_end] = candidate
            sample_begin = sample_end

    changed_order = padded_order.index_select(0, source_index)
    changed_canonical = canonical.index_select(0, source_index)
    changed_positions = torch.arange(
        changed_order.numel(), device=device, dtype=torch.long
    )[changed_canonical]
    changed_points = changed_order[changed_canonical]
    inverse = torch.full((num_points,), -1, device=device, dtype=torch.long)
    inverse[changed_points] = changed_positions

    if changed_points.numel() != num_points:
        raise AssertionError("attention permutation lost canonical point occurrences")
    if torch.any(inverse < 0):
        raise AssertionError("attention inverse is not a complete bijection")
    if not torch.equal(
        changed_order.index_select(0, inverse),
        torch.arange(num_points, device=device),
    ):
        raise AssertionError("attention order/inverse consistency failed")
    if not torch.equal(
        batch.index_select(0, padded_order),
        batch.index_select(0, changed_order),
    ):
        raise AssertionError("attention permutation mixed batch elements")

    window_ids = _window_ids(changed_order.numel(), cu_seqlens, device)
    base_point_window = torch.empty(num_points, dtype=torch.long, device=device)
    base_point_window[padded_order[canonical]] = window_ids[canonical]
    changed_point_window = torch.empty(num_points, dtype=torch.long, device=device)
    changed_point_window[changed_points] = window_ids[changed_canonical]
    same_membership = _same_window_membership(
        padded_order, changed_order, cu_seqlens
    )
    changed_fraction = (
        (base_point_window != changed_point_window).to(torch.float64).mean().item()
        if num_points
        else 0.0
    )
    base_distance = _mean_within_window_distance(
        grid_coord, padded_order, cu_seqlens
    )
    changed_distance = _mean_within_window_distance(
        grid_coord, changed_order, cu_seqlens
    )
    destruction_possible = False
    for batch_index in torch.unique(batch).detach().to(device="cpu").tolist():
        batch_mask = batch == batch_index
        if torch.unique(base_point_window[batch_mask]).numel() > 1:
            destruction_possible = True
            break
    distance_tolerance = max(abs(base_distance), 1.0) * 1e-12
    locality_verified = (
        same_membership
        and abs(changed_distance - base_distance) <= distance_tolerance
        if mode == "locality_preserving"
        else (
            destruction_possible
            and not same_membership
            and changed_distance > base_distance + distance_tolerance
        )
    )
    if mode == "locality_preserving" and not same_membership:
        raise AssertionError("locality-preserving control changed window membership")
    if mode == "locality_destroying" and destruction_possible and same_membership:
        raise AssertionError("locality-destroying control preserved every window")
    diagnostic = {
        "mode": mode,
        "stage": int(stage),
        "order_index": int(order_index),
        "num_points": int(num_points),
        "num_padded_tokens": int(changed_order.numel()),
        "num_windows": int(cu_seqlens.numel() - 1),
        "destruction_possible": bool(destruction_possible),
        "locality_verified": bool(locality_verified),
        "window_membership_preserved": bool(same_membership),
        "canonical_window_changed_fraction": float(changed_fraction),
        "base_mean_within_window_distance": float(base_distance),
        "changed_mean_within_window_distance": float(changed_distance),
        "within_window_distance_ratio": float(
            changed_distance / base_distance if base_distance > 0 else 1.0
        ),
    }
    return changed_order, inverse, diagnostic
