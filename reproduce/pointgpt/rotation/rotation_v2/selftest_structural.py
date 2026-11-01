#!/usr/bin/env python3
"""CPU-only structural self-tests for rotation_matrix_v2 P1."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotation_v2.adapters.strict_loading import (  # noqa: E402
    StrictLoadError,
    state_dict_sha256,
    strict_load_state_dict,
)
from rotation_v2.audit_pointmae_partseg_ckpt import (  # noqa: E402
    ABSENT_STATUS,
    FULL_STATUS,
    UNKNOWN_STATUS,
    audit_checkpoint,
    official_expected_state_shapes,
)
from rotation_v2.frozen_bank import (  # noqa: E402
    ARTIFACT_FILENAMES,
    assert_rotation_matrices,
    canonical_json_bytes,
    generate_frozen_bank,
    load_frozen_bank,
    read_example_id_list,
)
from rotation_v2.gates import assert_dual_clean_gate  # noqa: E402
from rotation_v2.shapenetpart import (  # noqa: E402
    category_restricted_predictions,
    official_sample_indices,
    official_shape_instance_ious,
)


def _test_bank_order_independence() -> dict[str, object]:
    ids_a = ["scene010", "2", "alpha", "001", "zeta", "object/7"]
    ids_b = ["zeta", "001", "object/7", "alpha", "scene010", "2"]
    with tempfile.TemporaryDirectory(prefix="rotation_v2_bank_test_") as root:
        root_path = Path(root)
        directory_a = root_path / "bank_a"
        directory_b = root_path / "bank_b"
        id_list = root_path / "example_ids.json"
        id_list.write_bytes(canonical_json_bytes(ids_a))
        parsed_ids = read_example_id_list(id_list)
        bank_a = generate_frozen_bank(
            directory_a,
            parsed_ids,
            global_seed=20260728,
        )
        bank_b = generate_frozen_bank(
            directory_b,
            ids_b,
            global_seed=20260728,
        )
        filenames = (*ARTIFACT_FILENAMES, "manifest.json")
        for filename in filenames:
            if (directory_a / filename).read_bytes() != (
                directory_b / filename
            ).read_bytes():
                raise AssertionError(
                    "shuffled ID order changed frozen artifact %s" % filename
                )
        if bank_a.example_ids != bank_b.example_ids:
            raise AssertionError("canonical example ID order changed")
        if not np.array_equal(bank_a.matrices, bank_b.matrices):
            raise AssertionError("shuffled ID order changed rotation matrices")
        if not np.array_equal(bank_a.quaternions, bank_b.quaternions):
            raise AssertionError("shuffled ID order changed quaternions")
        if bank_a.matrices.flags.writeable or bank_a.quaternions.flags.writeable:
            raise AssertionError("loaded frozen bank arrays must be read-only")
        reloaded = load_frozen_bank(
            directory_a,
            expected_manifest_sha256=str(bank_a.manifest["manifest_sha256"]),
        )
        return {
            "manifest_sha256": bank_a.manifest["manifest_sha256"],
            "content_sha256": bank_a.manifest["content_sha256"],
            "num_examples": len(reloaded.example_ids),
            "num_conditions": len(reloaded.conditions),
            "byte_identical_after_id_shuffle": True,
        }


class _TinyCleanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 5, bias=True, dtype=torch.float64)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.projection(points).mean(dim=1)


def _test_dual_clean_gate() -> dict[str, object]:
    torch.manual_seed(17)
    model = _TinyCleanModel().eval()
    points = torch.randn(4, 11, 3, dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    gate = assert_dual_clean_gate(
        model=model,
        forward_logits=model,
        points=points,
        identity_matrix=identity,
    )
    if not gate["input_bitwise_equal"] or not gate["logits_bitwise_equal"]:
        raise AssertionError("dual clean gate did not enforce bitwise equality")
    return gate


def _test_strict_loader() -> dict[str, object]:
    torch.manual_seed(23)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    source = {
        "module." + key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    record = strict_load_state_dict(
        model,
        source,
        allowed_prefix_rewrites=("module.",),
    )
    loaded_hash = state_dict_sha256(model)
    if loaded_hash != record.loaded_state_sha256:
        raise AssertionError("strict loader record has a stale state hash")

    mismatched = dict(source)
    original_key = sorted(mismatched)[0]
    mismatched["module.not_a_model_key"] = mismatched.pop(original_key)
    try:
        strict_load_state_dict(
            model,
            mismatched,
            allowed_prefix_rewrites=("module.",),
        )
    except StrictLoadError:
        rejected = True
    else:
        rejected = False
    if not rejected:
        raise AssertionError("strict loader accepted a key-mismatched state dict")
    return {
        "positive_strict_load": True,
        "key_mismatch_rejected": True,
        "model_key_count": record.model_key_count,
        "loaded_state_sha256": loaded_hash,
    }


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(deltas, axis=-1)


def _test_geometry() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rotation_v2_geometry_") as root:
        bank = generate_frozen_bank(
            Path(root) / "bank",
            ["geometry-example"],
            global_seed=99,
        )
        rotation = np.asarray(
            bank.matrix("geometry-example", "so3_01"),
            dtype=np.float64,
        )
        geometry = assert_rotation_matrices(rotation)
        rng = np.random.default_rng(3)
        points = rng.normal(size=(19, 3))
        vectors = rng.normal(size=(13, 3))
        rotated_points = points @ rotation.T
        rotated_vectors = vectors @ rotation.T
        norm_error = float(
            np.max(
                np.abs(
                    np.linalg.norm(rotated_vectors, axis=1)
                    - np.linalg.norm(vectors, axis=1)
                )
            )
        )
        dot_error = float(
            np.max(np.abs(rotated_vectors @ rotated_vectors.T - vectors @ vectors.T))
        )
        distance_error = float(
            np.max(
                np.abs(
                    _pairwise_distances(rotated_points) - _pairwise_distances(points)
                )
            )
        )
        determinant = float(np.linalg.det(rotation))
        tolerance = 2.0e-12
        if max(norm_error, dot_error, distance_error) > tolerance:
            raise AssertionError("rotation failed a geometry preservation check")
        if abs(determinant - 1.0) > tolerance:
            raise AssertionError("rotation determinant is not one")
        return {
            **geometry,
            "maximum_norm_error": norm_error,
            "maximum_dot_product_error": dot_error,
            "maximum_pairwise_distance_error": distance_error,
            "determinant": determinant,
        }


def _test_auditor_unknown_layout() -> dict[str, object]:
    backbone_shapes, head_shapes = official_expected_state_shapes()
    classifier_shapes = {**backbone_shapes, **head_shapes}
    if (len(backbone_shapes), len(head_shapes), len(classifier_shapes)) != (
        156,
        42,
        198,
    ):
        raise AssertionError("official Point-MAE key inventory count changed")
    if classifier_shapes.get("convs3.weight") != (50, 256, 1):
        raise AssertionError("official Point-MAE 50-part projection changed")
    with tempfile.TemporaryDirectory(prefix="rotation_v2_auditor_") as root:
        checkpoint = Path(root) / "unknown.pth"
        torch.save(
            {
                "state_dict": {"weight": torch.ones(2, 3)},
                "nonfinite_scalar_probe": float("nan"),
                "numpy_metric_probe": np.float64(0.861),
            },
            checkpoint,
        )
        report = audit_checkpoint(checkpoint)
        json.dumps(report, sort_keys=True, allow_nan=False)
        if report["verdict"]["status"] != UNKNOWN_STATUS:
            raise AssertionError("auditor guessed an unknown container layout")
        if report["verdict"]["full_finetuned_head"] is not None:
            raise AssertionError("UNKNOWN auditor verdict must use null head status")
        partial_checkpoint = Path(root) / "partial_finetune.pth"
        torch.save(
            {
                "model_state_dict": {
                    "convs3.weight": torch.ones(50, 256, 1),
                    "convs3.bias": torch.ones(50),
                }
            },
            partial_checkpoint,
        )
        partial_report = audit_checkpoint(partial_checkpoint)
        if partial_report["verdict"]["status"] != ABSENT_STATUS:
            raise AssertionError("auditor accepted an incomplete finetuned model")
        if partial_report["verdict"]["full_finetuned_head"]:
            raise AssertionError("partial finetuned model received a true verdict")
        return {
            "unknown_layout_status": report["verdict"]["status"],
            "safe_tensor_inventory_count": report["tensor_count"],
            "partial_model_status": partial_report["verdict"]["status"],
            "expected_classifier_key_count": len(classifier_shapes),
            "expected_head_key_count": len(head_shapes),
        }


def _test_sampling_freeze_determinism() -> dict[str, object]:
    raw_counts = np.asarray([1, 3, 17, 2049], dtype=np.int64)
    first = official_sample_indices(raw_counts, seed=20260728)
    second = official_sample_indices(raw_counts, seed=20260728)
    different = official_sample_indices(raw_counts, seed=20260729)
    if not np.array_equal(first, second):
        raise AssertionError("official replacement sampling did not replay")
    if np.array_equal(first, different):
        raise AssertionError("sampling seed did not affect frozen indices")
    if first.shape != (4, 2048) or first.dtype != np.int64:
        raise AssertionError("frozen sampling has an unexpected schema")
    if np.any(first < 0) or np.any(first >= raw_counts[:, None]):
        raise AssertionError("frozen sampling produced an out-of-range index")
    if np.unique(first[1]).size >= first.shape[1]:
        raise AssertionError("replacement sampling did not retain duplicates")
    return {
        "same_seed_byte_identical": True,
        "different_seed_changes_indices": True,
        "replacement_duplicates_present": True,
        "indices_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "shape": list(first.shape),
    }


def _test_metric_category_restriction() -> dict[str, object]:
    logits = np.full((2, 4, 50), -10.0, dtype=np.float64)
    logits[:, :, 49] = 100.0
    logits[0, 0, 0] = 1.0
    logits[0, 1, 1] = 1.0
    logits[0, 2, 0] = 1.0
    logits[0, 3, 1] = 1.0
    logits[1, 0, 4] = 1.0
    logits[1, 1, 4] = 1.0
    logits[1, 2, 5] = 1.0
    logits[1, 3, 5] = 1.0
    class_labels = np.asarray([0, 1], dtype=np.int64)
    targets = np.asarray(
        [
            [0, 1, 1, 1],
            [4, 5, 5, 5],
        ],
        dtype=np.int64,
    )
    predictions = category_restricted_predictions(logits, class_labels)
    expected_predictions = np.asarray(
        [
            [0, 1, 0, 1],
            [4, 4, 5, 5],
        ],
        dtype=np.int64,
    )
    if not np.array_equal(predictions, expected_predictions):
        raise AssertionError("category-valid argmax restriction is incorrect")
    if np.any(predictions == 49):
        raise AssertionError("global invalid-part maximum escaped restriction")
    shape_ious = official_shape_instance_ious(
        predictions,
        targets,
        class_labels,
    )
    expected_ious = np.asarray([19.0 / 24.0, 7.0 / 12.0])
    if not np.allclose(shape_ious, expected_ious, rtol=0.0, atol=1.0e-15):
        raise AssertionError("official synthetic per-shape IoU is incorrect")
    return {
        "invalid_global_maximum_rejected": True,
        "shape_instance_ious": shape_ious.tolist(),
        "absent_both_parts_score_one": True,
    }


def _expanded_exact_state(
    shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    scalar = torch.zeros((), dtype=torch.float32)
    return {
        key: scalar if not shape else scalar.expand(shape)
        for key, shape in shapes.items()
    }


def _test_auditor_exact_released_template() -> dict[str, object]:
    backbone_shapes, head_shapes = official_expected_state_shapes()
    exact_state = _expanded_exact_state({**backbone_shapes, **head_shapes})
    with tempfile.TemporaryDirectory(prefix="rotation_v2_auditor_exact_") as root:
        checkpoint = Path(root) / "part_seg_synthetic_exact.pth"
        container = {
            "model_state_dict": exact_state,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "epoch": 205,
            "train_acc": 0.9783,
            "test_acc": 0.9458,
            "class_avg_iou": 0.8418,
            "inctance_avg_iou": 0.8607,
        }
        torch.save(container, checkpoint)
        report = audit_checkpoint(checkpoint)
        if report["verdict"]["status"] != FULL_STATUS:
            raise AssertionError("auditor rejected the exact released template")
        if report["verdict"]["full_finetuned_head"] is not True:
            raise AssertionError("exact released template did not receive true verdict")
        if report["coverage"]["official_classifier_full_model"][
            "expected_key_count"
        ] != 198:
            raise AssertionError("auditor did not require all 198 released keys")
        if not report["release_container_metadata"]["valid"]:
            raise AssertionError("auditor rejected allowlisted release metadata")

        extra_checkpoint = Path(root) / "part_seg_synthetic_extra.pth"
        torch.save({**container, "optimizer": {}}, extra_checkpoint)
        extra_report = audit_checkpoint(extra_checkpoint)
        if extra_report["verdict"]["status"] != ABSENT_STATUS:
            raise AssertionError("auditor accepted extra top-level metadata")
        if extra_report["verdict"]["full_finetuned_head"]:
            raise AssertionError("extra checkpoint metadata received a true verdict")
        return {
            "exact_198_key_template_accepted": True,
            "allowlisted_metadata_recorded": report[
                "release_container_metadata"
            ]["recorded_values"],
            "extra_metadata_rejected": True,
            "prefix_counts_exact": report["coverage"]["state_prefix_counts"][
                "exact"
            ],
        }


TESTS: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
    ("bank_order_independence", _test_bank_order_independence),
    ("dual_clean_identity_gate", _test_dual_clean_gate),
    ("strict_loader_rejection", _test_strict_loader),
    ("geometry_preservation", _test_geometry),
    ("auditor_unknown_layout", _test_auditor_unknown_layout),
    ("sampling_freeze_determinism", _test_sampling_freeze_determinism),
    ("metric_category_restriction", _test_metric_category_restriction),
    ("auditor_exact_released_template", _test_auditor_exact_released_template),
)


def main() -> int:
    results: list[dict[str, object]] = []
    failed = False
    for name, test in TESTS:
        try:
            details = test()
        except Exception as error:
            failed = True
            results.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "status": "PASS",
                    "details": details,
                }
            )
    report = {
        "schema": "rotation_matrix_v2_structural_selftest_v2",
        "status": "FAIL" if failed else "PASS",
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tests": results,
    }
    print(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
