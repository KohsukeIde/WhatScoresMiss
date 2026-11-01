#!/usr/bin/env python3
"""Read-only CPU auditor for the official Point-MAE ``part_seg.pth``.

The complete report is emitted as JSON on stdout. Exit status is 0 only for a
proven full finetuned 50-part model, 2 for a recognized layout that is not a
complete finetuned model, and 3 for UNKNOWN or an audit error. No checkpoint
content is ever written.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import pickle
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


OFFICIAL_COMMIT = "7445a68068d707da6d959b66773bfef84c4e3e32"
OFFICIAL_PT_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    + OFFICIAL_COMMIT
    + "/segmentation/models/pt.py"
)
OFFICIAL_POINTNET2_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    + OFFICIAL_COMMIT
    + "/segmentation/models/pointnet2_utils.py"
)
OFFICIAL_MAIN_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    + OFFICIAL_COMMIT
    + "/segmentation/main.py"
)
FULL_STATUS = "FULL_FINETUNED_HEAD"
ABSENT_STATUS = "NO_FULL_FINETUNED_HEAD"
UNKNOWN_STATUS = "UNKNOWN"
OFFICIAL_FINETUNED_TOP_LEVEL_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "train_acc",
        "test_acc",
        "class_avg_iou",
        "inctance_avg_iou",
    }
)
OFFICIAL_FINETUNED_METADATA_KEYS = (
    "epoch",
    "train_acc",
    "test_acc",
    "class_avg_iou",
    "inctance_avg_iou",
)
OFFICIAL_RELEASE_METADATA = {
    "epoch": 205,
    "class_avg_iou": 0.8418,
    "inctance_avg_iou": 0.8607,
}
OFFICIAL_STATE_PREFIX_COUNTS = {
    "blocks": 132,
    "encoder": 18,
    "pos_embed": 4,
    "norm": 2,
    "bns1": 5,
    "bns2": 5,
    "convs1": 2,
    "convs2": 2,
    "convs3": 2,
    "conv_fuse": 6,
    "label_conv": 6,
    "propagation_0": 14,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_batch_norm(
    result: dict[str, tuple[int, ...]],
    prefix: str,
    channels: int,
) -> None:
    for name in ("weight", "bias", "running_mean", "running_var"):
        result[prefix + "." + name] = (channels,)
    result[prefix + ".num_batches_tracked"] = ()


def official_expected_state_shapes() -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
]:
    """Return the pinned official backbone and segmentation-head state shapes.

    The backbone mirrors Point-MAE ``segmentation/models/pt.py`` lines 47-61
    and 121-193 at:
    https://github.com/Pang-Yatian/Point-MAE/blob/7445a68068d707da6d959b66773bfef84c4e3e32/segmentation/models/pt.py#L47-L193

    The active head mirrors ``pt.py`` lines 194-205 and the feature-propagation
    ``ModuleList`` construction in ``segmentation/models/pointnet2_utils.py``
    lines 244-253:
    https://github.com/Pang-Yatian/Point-MAE/blob/7445a68068d707da6d959b66773bfef84c4e3e32/segmentation/models/pt.py#L194-L205
    https://github.com/Pang-Yatian/Point-MAE/blob/7445a68068d707da6d959b66773bfef84c4e3e32/segmentation/models/pointnet2_utils.py#L244-L253

    The released ``part_seg.pth`` contains the six ``conv_fuse`` tensors from
    the official segmentation model lineage in addition to the active head.
    The pinned public forward does not consume that module. The exact released
    198-key template is therefore the binding strict-load template.
    """
    backbone: dict[str, tuple[int, ...]] = {
        "encoder.first_conv.0.weight": (128, 3, 1),
        "encoder.first_conv.0.bias": (128,),
        "encoder.first_conv.3.weight": (256, 128, 1),
        "encoder.first_conv.3.bias": (256,),
        "encoder.second_conv.0.weight": (512, 512, 1),
        "encoder.second_conv.0.bias": (512,),
        "encoder.second_conv.3.weight": (384, 512, 1),
        "encoder.second_conv.3.bias": (384,),
        "pos_embed.0.weight": (128, 3),
        "pos_embed.0.bias": (128,),
        "pos_embed.2.weight": (384, 128),
        "pos_embed.2.bias": (384,),
        "norm.weight": (384,),
        "norm.bias": (384,),
    }
    _add_batch_norm(backbone, "encoder.first_conv.1", 128)
    _add_batch_norm(backbone, "encoder.second_conv.1", 512)
    for block_index in range(12):
        prefix = "blocks.blocks.%d" % block_index
        backbone.update(
            {
                prefix + ".norm1.weight": (384,),
                prefix + ".norm1.bias": (384,),
                prefix + ".attn.qkv.weight": (1152, 384),
                prefix + ".attn.proj.weight": (384, 384),
                prefix + ".attn.proj.bias": (384,),
                prefix + ".norm2.weight": (384,),
                prefix + ".norm2.bias": (384,),
                prefix + ".mlp.fc1.weight": (1536, 384),
                prefix + ".mlp.fc1.bias": (1536,),
                prefix + ".mlp.fc2.weight": (384, 1536),
                prefix + ".mlp.fc2.bias": (384,),
            }
        )

    head: dict[str, tuple[int, ...]] = {
        "conv_fuse.0.weight": (1024, 1152, 1),
        "label_conv.0.weight": (64, 16, 1),
        "propagation_0.mlp_convs.0.weight": (1536, 1155, 1),
        "propagation_0.mlp_convs.0.bias": (1536,),
        "propagation_0.mlp_convs.1.weight": (1024, 1536, 1),
        "propagation_0.mlp_convs.1.bias": (1024,),
        "convs1.weight": (512, 3392, 1),
        "convs1.bias": (512,),
        "convs2.weight": (256, 512, 1),
        "convs2.bias": (256,),
        "convs3.weight": (50, 256, 1),
        "convs3.bias": (50,),
    }
    _add_batch_norm(head, "conv_fuse.1", 1024)
    _add_batch_norm(head, "label_conv.1", 64)
    _add_batch_norm(head, "propagation_0.mlp_bns.0", 1536)
    _add_batch_norm(head, "propagation_0.mlp_bns.1", 1024)
    _add_batch_norm(head, "bns1", 512)
    _add_batch_norm(head, "bns2", 256)
    return backbone, head


def _state_prefix_counts(state: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return dict(
        sorted(Counter(key.split(".", 1)[0] for key in state).items())
    )


def _primitive_number(value: object) -> int | float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _release_metadata_report(checkpoint: Mapping[object, object]) -> dict[str, object]:
    actual_keys = {
        key for key in checkpoint if isinstance(key, str)
    }
    non_string_keys = sorted(
        (repr(key) for key in checkpoint if not isinstance(key, str))
    )
    missing = sorted(OFFICIAL_FINETUNED_TOP_LEVEL_KEYS - actual_keys)
    unexpected = sorted(actual_keys - OFFICIAL_FINETUNED_TOP_LEVEL_KEYS)
    values: dict[str, object] = {}
    errors: list[str] = []
    for key in OFFICIAL_FINETUNED_METADATA_KEYS:
        if key not in checkpoint:
            continue
        normalized = _primitive_number(checkpoint[key])
        if normalized is None:
            errors.append("%s is not a finite primitive numeric scalar" % key)
        else:
            values[key] = normalized
    if "epoch" in values and not isinstance(values["epoch"], int):
        errors.append("epoch is not an integer")
    if non_string_keys:
        errors.append("container has non-string top-level keys")
    exact_layout = not missing and not unexpected and not non_string_keys
    valid = exact_layout and not errors
    return {
        "allowlisted_keys": sorted(OFFICIAL_FINETUNED_TOP_LEVEL_KEYS),
        "actual_keys": sorted(actual_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "non_string_keys": non_string_keys,
        "recorded_values": values,
        "reference_release_values": dict(OFFICIAL_RELEASE_METADATA),
        "validation_errors": errors,
        "exact_allowlisted_layout": exact_layout,
        "valid": valid,
    }


def _key_text(key: object) -> str:
    if isinstance(key, str):
        return key
    return repr(key)


def _child_path(parent: str, key: object) -> str:
    key_text = _key_text(key)
    return key_text if not parent else parent + "." + key_text


def tensor_inventory(value: object) -> list[dict[str, object]]:
    """Inventory every reachable tensor without copying or modifying it."""
    result: list[dict[str, object]] = []
    active: set[int] = set()

    def visit(node: object, path: str) -> None:
        if torch.is_tensor(node):
            result.append(
                {
                    "key": path or "<root>",
                    "shape": [int(size) for size in node.shape],
                    "dtype": str(node.dtype),
                    "device": str(node.device),
                    "layout": str(node.layout),
                    "numel": int(node.numel()),
                }
            )
            return
        if isinstance(node, Mapping):
            object_id = id(node)
            if object_id in active:
                return
            active.add(object_id)
            try:
                for key in sorted(node, key=lambda item: _key_text(item)):
                    visit(node[key], _child_path(path, key))
            finally:
                active.remove(object_id)
            return
        if isinstance(node, (list, tuple)):
            object_id = id(node)
            if object_id in active:
                return
            active.add(object_id)
            try:
                for index, child in enumerate(node):
                    visit(child, "%s[%d]" % (path, index))
            finally:
                active.remove(object_id)

    visit(value, "")
    return result


def structural_dump(
    value: object,
    *,
    max_depth: int = 4,
    max_children: int = 512,
) -> dict[str, object]:
    """Return a safe type-and-key dump without tensor values."""
    active: set[int] = set()

    def describe(node: object, depth: int) -> dict[str, object]:
        if torch.is_tensor(node):
            return {
                "type": "torch.Tensor",
                "shape": [int(size) for size in node.shape],
                "dtype": str(node.dtype),
                "device": str(node.device),
                "layout": str(node.layout),
            }
        if isinstance(node, Mapping):
            keys = sorted(node, key=lambda item: _key_text(item))
            summary: dict[str, object] = {
                "type": type(node).__name__,
                "length": len(node),
                "keys": [_key_text(key) for key in keys],
                "value_type_counts": dict(
                    sorted(
                        Counter(type(item).__name__ for item in node.values()).items()
                    )
                ),
            }
            object_id = id(node)
            if object_id in active:
                summary["cycle"] = True
                return summary
            if depth >= max_depth:
                summary["children_omitted_at_depth"] = depth
                return summary
            active.add(object_id)
            try:
                child_keys = keys[:max_children]
                summary["children"] = [
                    {
                        "key": _key_text(key),
                        "key_type": type(key).__name__,
                        "value": describe(node[key], depth + 1),
                    }
                    for key in child_keys
                ]
                if len(keys) > max_children:
                    summary["truncated_child_count"] = len(keys) - max_children
            finally:
                active.remove(object_id)
            return summary
        if isinstance(node, (list, tuple)):
            summary = {
                "type": type(node).__name__,
                "length": len(node),
                "value_type_counts": dict(
                    sorted(Counter(type(item).__name__ for item in node).items())
                ),
            }
            object_id = id(node)
            if object_id in active:
                summary["cycle"] = True
                return summary
            if depth >= max_depth:
                summary["children_omitted_at_depth"] = depth
                return summary
            active.add(object_id)
            try:
                children = node[:max_children]
                summary["children"] = [describe(child, depth + 1) for child in children]
                if len(node) > max_children:
                    summary["truncated_child_count"] = len(node) - max_children
            finally:
                active.remove(object_id)
            return summary
        if node is None or isinstance(node, (bool, int, float, str)):
            primitive: dict[str, object] = {"type": type(node).__name__}
            if isinstance(node, str):
                primitive["length"] = len(node)
                primitive["preview"] = node[:160]
            elif isinstance(node, float) and not math.isfinite(node):
                primitive["value_repr"] = repr(node)
            else:
                primitive["value"] = node
            return primitive
        return {"type": type(node).__name__, "repr": repr(node)[:160]}

    return describe(value, 0)


def _tensor_state_mapping(
    value: object,
) -> tuple[
    dict[str, torch.Tensor] | None,
    list[str],
]:
    if not isinstance(value, Mapping) or not value:
        return None, ["state container is not a non-empty mapping"]
    if any(not isinstance(key, str) for key in value):
        return None, ["state container has a non-string key"]
    non_tensors = sorted(
        key for key, item in value.items() if not torch.is_tensor(item)
    )
    if non_tensors:
        return None, ["state container has non-tensor entries: %s" % non_tensors[:20]]
    non_cpu = sorted(key for key, item in value.items() if item.device.type != "cpu")
    if non_cpu:
        return None, [
            "state container has tensors that did not load onto CPU: %s" % non_cpu[:20]
        ]
    non_strided = sorted(
        key for key, item in value.items() if item.layout != torch.strided
    )
    if non_strided:
        return None, ["state container has non-strided tensors: %s" % non_strided[:20]]
    return dict(value), []


def _strip_prefixes(
    state: Mapping[str, torch.Tensor],
    *,
    primary_prefixes: tuple[str, ...],
    secondary_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Tensor] | None, dict[str, int], list[str]]:
    """Apply only declared leading-prefix rewrites and reject collisions.

    The secondary ``MAE_encoder.`` and ``base_model.`` rewrites mirror the
    official preload path in Point-MAE ``segmentation/models/pt.py`` lines
    214-226:
    https://github.com/Pang-Yatian/Point-MAE/blob/7445a68068d707da6d959b66773bfef84c4e3e32/segmentation/models/pt.py#L214-L226
    """
    result: dict[str, torch.Tensor] = {}
    counts = {
        "none": 0,
        **{prefix: 0 for prefix in primary_prefixes + secondary_prefixes},
    }
    errors: list[str] = []
    for source_key, tensor in state.items():
        target_key = source_key
        applied: list[str] = []
        for prefix in primary_prefixes:
            if target_key.startswith(prefix):
                target_key = target_key[len(prefix) :]
                applied.append(prefix)
                break
        for prefix in secondary_prefixes:
            if target_key.startswith(prefix):
                target_key = target_key[len(prefix) :]
                applied.append(prefix)
                break
        if target_key in result:
            errors.append(
                "prefix rewrite collision from %s to %s" % (source_key, target_key)
            )
            continue
        result[target_key] = tensor
        if not applied:
            counts["none"] += 1
        else:
            for prefix in applied:
                counts[prefix] += 1
    if errors:
        return None, counts, errors
    return result, counts, errors


def _coverage(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, tuple[int, ...]],
    *,
    include_unexpected: bool,
) -> dict[str, object]:
    actual_keys = set(actual)
    expected_keys = set(expected)
    shared = sorted(actual_keys & expected_keys)
    missing = sorted(expected_keys - actual_keys)
    shape_mismatches = [
        {
            "key": key,
            "expected_shape": list(expected[key]),
            "checkpoint_shape": [int(size) for size in actual[key].shape],
        }
        for key in shared
        if tuple(actual[key].shape) != tuple(expected[key])
    ]
    mismatch_keys = {item["key"] for item in shape_mismatches}
    shape_matched = [key for key in shared if key not in mismatch_keys]
    result: dict[str, object] = {
        "expected_key_count": len(expected_keys),
        "present_key_count": len(shared),
        "shape_matched_key_count": len(shape_matched),
        "coverage_fraction": (
            float(len(shape_matched) / len(expected_keys)) if expected_keys else 1.0
        ),
        "missing_keys": missing,
        "shape_mismatches": shape_mismatches,
        "complete": not missing and not shape_mismatches,
    }
    if include_unexpected:
        unexpected = sorted(actual_keys - expected_keys)
        result["unexpected_keys"] = unexpected
        result["exact_key_set_and_shapes"] = (
            not missing and not unexpected and not shape_mismatches
        )
    return result


def _load_checkpoint(path: Path) -> tuple[object, str]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        return (
            torch.load(path, map_location="cpu"),
            "legacy_torch_without_weights_only_argument",
        )
    try:
        return (
            torch.load(path, map_location="cpu", weights_only=True),
            "weights_only=True",
        )
    except pickle.UnpicklingError:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            raise
        scalar_constructor = np.core.multiarray.scalar
        numpy_scalar_types = [
            type(np.dtype(dtype))
            for dtype in (
                np.bool_,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            )
        ]
        allowlist = list(
            dict.fromkeys([scalar_constructor, np.dtype, *numpy_scalar_types])
        )
        with safe_globals(allowlist):
            return (
                torch.load(path, map_location="cpu", weights_only=True),
                "weights_only=True+numpy_scalar_safe_globals",
            )


def _base_report(path: Path) -> dict[str, object]:
    return {
        "schema": "rotation_matrix_v2_pointmae_partseg_audit_v2",
        "checkpoint": {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size) if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "official_reference": {
            "commit": OFFICIAL_COMMIT,
            "model_definition": OFFICIAL_PT_URL,
            "feature_propagation_definition": OFFICIAL_POINTNET2_URL,
            "training_and_save_driver": OFFICIAL_MAIN_URL,
            "num_categories": 16,
            "num_parts": 50,
            "released_state_key_count": 198,
            "released_state_prefix_counts": dict(OFFICIAL_STATE_PREFIX_COUNTS),
            "released_container_allowlist": sorted(
                OFFICIAL_FINETUNED_TOP_LEVEL_KEYS
            ),
        },
    }


def _unknown_verdict(notes: list[str]) -> dict[str, object]:
    return {
        "status": UNKNOWN_STATUS,
        "full_finetuned_head": None,
        "missing": [],
        "notes": notes,
    }


def audit_checkpoint(path: Path) -> dict[str, object]:
    """Audit a checkpoint without importing Point-MAE or allocating CUDA.

    The known finetuned ``model_state_dict`` container mirrors the official
    save path in ``segmentation/main.py`` lines 256-269:
    https://github.com/Pang-Yatian/Point-MAE/blob/7445a68068d707da6d959b66773bfef84c4e3e32/segmentation/main.py#L256-L269
    """
    checkpoint_path = Path(path)
    report = _base_report(checkpoint_path)
    if not checkpoint_path.is_file():
        report["verdict"] = _unknown_verdict(["checkpoint path is not a regular file"])
        return report
    before_hash = report["checkpoint"]["sha256"]
    try:
        checkpoint, load_mode = _load_checkpoint(checkpoint_path)
    except Exception as error:
        report["torch_load"] = {
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        report["verdict"] = _unknown_verdict(
            ["checkpoint could not be safely loaded on CPU"]
        )
        return report
    if file_sha256(checkpoint_path) != before_hash:
        report["verdict"] = _unknown_verdict(
            ["checkpoint bytes changed during the audit"]
        )
        return report

    report["torch_load"] = {"status": "OK", "mode": load_mode}
    report["container_structure"] = structural_dump(checkpoint)
    report["tensor_inventory"] = tensor_inventory(checkpoint)
    report["tensor_count"] = len(report["tensor_inventory"])
    if isinstance(checkpoint, Mapping):
        report["top_level_keys"] = sorted(_key_text(key) for key in checkpoint)
        report["top_level_key_inventory"] = sorted(
            (
                {
                    "key": _key_text(key),
                    "key_type": type(key).__name__,
                }
                for key in checkpoint
            ),
            key=lambda item: (item["key"], item["key_type"]),
        )
    else:
        report["top_level_keys"] = []
        report["top_level_key_inventory"] = []

    if not isinstance(checkpoint, Mapping):
        report["container_layout"] = "unknown_non_mapping"
        report["coverage"] = None
        report["verdict"] = _unknown_verdict(
            ["unknown container layout; safe structural dump is included"]
        )
        return report

    report["release_container_metadata"] = _release_metadata_report(checkpoint)

    present_weight_containers = [
        key
        for key in ("model_state_dict", "base_model")
        if key in checkpoint and checkpoint[key] is not None
    ]
    if len(present_weight_containers) != 1:
        report["container_layout"] = "unknown_or_ambiguous"
        report["coverage"] = None
        report["verdict"] = _unknown_verdict(
            [
                "expected exactly one known weight container, found %s"
                % present_weight_containers,
                "safe structural dump is included",
            ]
        )
        return report

    container_key = present_weight_containers[0]
    state, state_errors = _tensor_state_mapping(checkpoint[container_key])
    report["container_layout"] = (
        "official_finetuned_model_state_dict"
        if container_key == "model_state_dict"
        else "official_pretraining_base_model"
    )
    report["selected_state_container"] = container_key
    if state is None:
        report["coverage"] = None
        report["verdict"] = _unknown_verdict(
            state_errors + ["known container name has an invalid payload"]
        )
        return report

    if container_key == "model_state_dict":
        normalized = dict(state)
        rewrite_counts = {"none": len(state)}
        rewrite_errors: list[str] = []
    else:
        normalized, rewrite_counts, rewrite_errors = _strip_prefixes(
            state,
            primary_prefixes=("module.",),
            secondary_prefixes=("MAE_encoder.", "base_model."),
        )
    report["key_normalization"] = {
        "policy": (
            "no_rewrites_exact_released_keys"
            if container_key == "model_state_dict"
            else "official_pretraining_prefix_allowlist"
        ),
        "rewrite_counts": rewrite_counts,
        "errors": rewrite_errors,
    }
    if normalized is None:
        report["coverage"] = None
        report["verdict"] = _unknown_verdict(
            rewrite_errors + ["key normalization was ambiguous"]
        )
        return report

    backbone_expected, head_expected = official_expected_state_shapes()
    classifier_expected = {**backbone_expected, **head_expected}
    classifier_coverage = _coverage(
        normalized,
        classifier_expected,
        include_unexpected=True,
    )
    backbone_coverage = _coverage(
        normalized,
        backbone_expected,
        include_unexpected=False,
    )
    head_coverage = _coverage(
        normalized,
        head_expected,
        include_unexpected=False,
    )
    final_projection = {
        "weight_key": "convs3.weight",
        "bias_key": "convs3.bias",
        "expected_weight_shape": [50, 256, 1],
        "expected_bias_shape": [50],
        "weight_present": "convs3.weight" in normalized,
        "bias_present": "convs3.bias" in normalized,
        "weight_shape_matches": (
            "convs3.weight" in normalized
            and tuple(normalized["convs3.weight"].shape) == (50, 256, 1)
        ),
        "bias_shape_matches": (
            "convs3.bias" in normalized
            and tuple(normalized["convs3.bias"].shape) == (50,)
        ),
    }
    report["coverage"] = {
        "official_classifier_full_model": classifier_coverage,
        "backbone": backbone_coverage,
        "segmentation_head": head_coverage,
        "final_50_part_projection": final_projection,
        "state_prefix_counts": {
            "expected": dict(OFFICIAL_STATE_PREFIX_COUNTS),
            "actual": _state_prefix_counts(normalized),
            "exact": (
                _state_prefix_counts(normalized)
                == OFFICIAL_STATE_PREFIX_COUNTS
            ),
        },
    }
    exact_full_model = bool(classifier_coverage["exact_key_set_and_shapes"])
    exact_prefix_inventory = bool(
        report["coverage"]["state_prefix_counts"]["exact"]
    )
    exact_release_container = bool(
        report["release_container_metadata"]["valid"]
    )
    full_finetuned_head = (
        container_key == "model_state_dict"
        and exact_full_model
        and exact_prefix_inventory
        and exact_release_container
        and bool(head_coverage["complete"])
        and bool(final_projection["weight_shape_matches"])
        and bool(final_projection["bias_shape_matches"])
    )
    missing = list(classifier_coverage["missing_keys"])
    notes: list[str] = []
    if container_key == "base_model":
        notes.append(
            "base_model is the official pretrained-backbone input layout, "
            "not the finetuned best_model.pth save layout"
        )
    if classifier_coverage["shape_mismatches"]:
        notes.append("one or more official classifier tensors have incompatible shapes")
    if classifier_coverage["unexpected_keys"]:
        notes.append("strict full-model loading would reject unexpected state keys")
    if not exact_prefix_inventory:
        notes.append("released state prefix inventory does not match exactly")
    if container_key == "model_state_dict" and not exact_release_container:
        notes.append(
            "top-level container keys or metadata types do not match the "
            "released checkpoint allowlist"
        )
    if not head_coverage["complete"]:
        notes.append("the complete official segmentation head is absent")
    if full_finetuned_head:
        notes.append(
            "all 198 released classifier keys and shapes, prefix counts, and "
            "allowlisted container metadata match exactly"
        )
    else:
        notes.append(
            "this audit does not authorize evaluation as an official "
            "finetuned 50-part system"
        )
    report["verdict"] = {
        "status": FULL_STATUS if full_finetuned_head else ABSENT_STATUS,
        "full_finetuned_head": full_finetuned_head,
        "missing": missing,
        "notes": notes,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only CPU audit of a Point-MAE ShapeNetPart checkpoint. "
            "The report is printed as JSON."
        )
    )
    parser.add_argument("--ckpt", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_checkpoint(args.ckpt)
    except Exception as error:
        report = _base_report(args.ckpt)
        report["audit_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        report["verdict"] = _unknown_verdict(
            ["auditor failed closed before reaching a verdict"]
        )
    print(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    status = report["verdict"]["status"]
    if status == FULL_STATUS:
        return 0
    if status == ABSENT_STATUS:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
