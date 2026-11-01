#!/usr/bin/env python3
"""Fail-closed ShapeNetPart rotation_matrix_v2 evaluation driver."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal_eval.determinism import (  # noqa: E402
    configure_deterministic_runtime,
    isolated_forward_rng,
)
from rotation_v2.adapters.base import FrozenSegmentationAdapter  # noqa: E402
from rotation_v2.adapters.pointgpt_seg import PointGPTSegAdapter  # noqa: E402
from rotation_v2.adapters.pointmae_seg import PointMAESegAdapter  # noqa: E402
from rotation_v2.adapters.strict_loading import file_sha256  # noqa: E402
from rotation_v2.frozen_bank import (  # noqa: E402
    FrozenRotationBank,
    canonical_json_bytes,
    canonical_json_sha256,
    generate_frozen_bank,
    load_frozen_bank,
    logical_array_sha256,
)
from rotation_v2.gates import tensor_bitwise_equal  # noqa: E402
from rotation_v2.shapenetpart import (  # noqa: E402
    CATEGORY_NAMES,
    LOCAL_EVALUATOR_MIRROR,
    NPOINT,
    NUM_PARTS,
    OFFICIAL_EVALUATOR_URL,
    FrozenShapeNetPartPlan,
    category_restricted_predictions,
    freeze_shapenetpart_sample_plan,
    load_shapenetpart_sample_plan,
    official_shape_instance_ious,
    paired_shape_bootstrap_ci,
)


RESULT_SCHEMA = "rotation_matrix_v2_shapenetpart_cell_v1"
RUN_SCHEMA = "rotation_matrix_v2_shapenetpart_run_v1"
POINTMAE_RELEASE_INSTANCE_MIOU_PERCENT = 86.07
POINTMAE_CLEAN_TOLERANCE_PERCENTAGE_POINTS = 0.3
POINTGPT_R7C_CONTEXT_PERCENT = 85.53


def _derived_seed(base_seed: int, *fields: object) -> int:
    payload = canonical_json_bytes([int(base_seed), *fields])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _stream_digest(
    *,
    dtype: str,
    shape: Sequence[int],
    namespace: str,
) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "namespace": namespace,
                "dtype": dtype,
                "shape": [int(size) for size in shape],
                "order": "C",
            }
        )
    )
    return digest


class _ConditionAccumulator:
    def __init__(self, plan: FrozenShapeNetPartPlan) -> None:
        self.plan = plan
        self.shape_ious = np.empty(len(plan.example_ids), dtype=np.float64)
        self.confusion = np.zeros((NUM_PARTS, NUM_PARTS), dtype=np.int64)
        self.correct = 0
        self.total = 0
        self.prediction_digest = _stream_digest(
            dtype="uint8",
            shape=(len(plan.example_ids), NPOINT),
            namespace="category_restricted_predictions",
        )
        self.input_digest = _stream_digest(
            dtype=np.dtype(np.float32).str,
            shape=(len(plan.example_ids), NPOINT, 3),
            namespace="model_ready_xyz",
        )

    def add(
        self,
        start: int,
        logits: torch.Tensor,
        model_input: torch.Tensor,
    ) -> None:
        end = start + int(logits.shape[0])
        logits_numpy = logits.detach().cpu().numpy()
        target = np.asarray(self.plan.targets[start:end], dtype=np.int64)
        labels = np.asarray(self.plan.class_labels[start:end], dtype=np.int64)
        predictions = category_restricted_predictions(logits_numpy, labels)
        shape_ious = official_shape_instance_ious(predictions, target, labels)
        self.shape_ious[start:end] = shape_ious
        np.add.at(
            self.confusion,
            (target.reshape(-1), predictions.reshape(-1)),
            1,
        )
        self.correct += int(np.sum(predictions == target))
        self.total += int(target.size)
        self.prediction_digest.update(
            np.ascontiguousarray(predictions, dtype=np.uint8).tobytes()
        )
        self.input_digest.update(
            np.ascontiguousarray(
                model_input.detach().cpu().numpy(),
                dtype=np.float32,
            ).tobytes()
        )

    def finalize(self) -> dict[str, object]:
        if self.total != len(self.plan.example_ids) * NPOINT:
            raise AssertionError("condition accumulator is incomplete")
        per_category: dict[str, float] = {}
        labels = np.asarray(self.plan.class_labels, dtype=np.int64)
        for class_index, category in enumerate(CATEGORY_NAMES):
            mask = labels == class_index
            if not np.any(mask):
                raise AssertionError("full test set is missing category %s" % category)
            per_category[category] = float(np.mean(self.shape_ious[mask]))
        instance = float(np.mean(self.shape_ious))
        class_average = float(np.mean(list(per_category.values())))
        accuracy = float(self.correct / self.total)
        return {
            "primary_metric": "official_category_valid_per_shape_instance_miou",
            "accuracy_fraction": accuracy,
            "accuracy_percent": 100.0 * accuracy,
            "instance_miou_fraction": instance,
            "instance_miou_percent": 100.0 * instance,
            "class_avg_iou_fraction": class_average,
            "class_avg_iou_percent": 100.0 * class_average,
            "per_category_iou_fraction": per_category,
            "per_shape_instance_iou_fraction": self.shape_ious.tolist(),
            "confusion_50x50": self.confusion.tolist(),
            "prediction_stream_sha256": self.prediction_digest.hexdigest(),
            "model_input_stream_sha256": self.input_digest.hexdigest(),
            "metric_contract": {
                "argmax": "restricted_to_category_valid_parts",
                "part_iou_absent_from_prediction_and_target": 1.0,
                "shape_score": "mean_over_category_valid_parts",
                "dataset_score": "mean_over_shapes",
                "official_source": OFFICIAL_EVALUATOR_URL,
                "local_mirror": LOCAL_EVALUATOR_MIRROR,
            },
        }


def _forward(
    adapter: FrozenSegmentationAdapter,
    points: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    first_state = next(adapter.model.parameters(), None)
    if first_state is None:
        first_state = next(adapter.model.buffers(), None)
    uses_cuda = bool(first_state is not None and first_state.is_cuda)
    with isolated_forward_rng(seed, uses_cuda=uses_cuda):
        logits = adapter.forward_logits(points, labels)
    return logits


def _dual_clean(
    adapter: FrozenSegmentationAdapter,
    plan: FrozenShapeNetPartPlan,
    *,
    batch_size: int,
    forward_seed: int,
    logits_atol: float,
    logits_rtol: float,
) -> tuple[dict[str, object], dict[str, object]]:
    accumulator = _ConditionAccumulator(plan)
    maximum_error = 0.0
    all_input_equal = True
    all_logits_bitwise = True
    for start in range(0, len(plan.example_ids), batch_size):
        end = min(start + batch_size, len(plan.example_ids))
        points = torch.from_numpy(
            np.array(plan.points[start:end], dtype=np.float32, copy=True)
        )
        labels = torch.from_numpy(
            np.array(plan.class_labels[start:end], dtype=np.int64, copy=True)
        ).long()
        identity = torch.eye(3, dtype=points.dtype, device=points.device)
        identity_points = torch.matmul(points, identity.transpose(0, 1))
        input_equal = tensor_bitwise_equal(points, identity_points)
        all_input_equal = all_input_equal and input_equal
        if not input_equal:
            raise AssertionError("explicit identity changed ShapeNetPart input bytes")
        seed = _derived_seed(forward_seed, "clean_identity", start)
        bypass_logits = _forward(adapter, points, labels, seed=seed)
        identity_logits = _forward(adapter, identity_points, labels, seed=seed)
        if bypass_logits.shape != identity_logits.shape:
            raise AssertionError("bypass and identity logits have different shapes")
        bitwise = tensor_bitwise_equal(bypass_logits, identity_logits)
        all_logits_bitwise = all_logits_bitwise and bitwise
        error = float(
            torch.max(torch.abs(bypass_logits - identity_logits)).item()
        )
        maximum_error = max(maximum_error, error)
        if logits_atol == 0.0 and logits_rtol == 0.0:
            if not bitwise:
                raise AssertionError("bypass and identity logits differ bitwise")
        elif not torch.allclose(
            bypass_logits,
            identity_logits,
            atol=logits_atol,
            rtol=logits_rtol,
        ):
            raise AssertionError("bypass and identity logits exceed tolerance")
        accumulator.add(start, bypass_logits, points)
    gate = {
        "input_bitwise_equal": all_input_equal,
        "logits_bitwise_equal": all_logits_bitwise,
        "logits_atol": float(logits_atol),
        "logits_rtol": float(logits_rtol),
        "maximum_absolute_logit_error": maximum_error,
        "same_forward_seed_replayed_per_batch": True,
        "all_shapes_covered": True,
    }
    return accumulator.finalize(), gate


def _geometry_probe(
    original: torch.Tensor,
    rotated: torch.Tensor,
    matrices_float64: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float]:
    probe_count = min(32, int(original.shape[1]))
    before = original[:, :probe_count].double()
    after = rotated[:, :probe_count].double()
    before_norm = torch.linalg.vector_norm(before, dim=-1)
    after_norm = torch.linalg.vector_norm(after, dim=-1)
    norm_error = float(torch.max(torch.abs(before_norm - after_norm)).item())
    before_dot = torch.bmm(before, before.transpose(1, 2))
    after_dot = torch.bmm(after, after.transpose(1, 2))
    dot_error = float(torch.max(torch.abs(before_dot - after_dot)).item())
    before_distance = torch.cdist(before, before)
    after_distance = torch.cdist(after, after)
    distance_error = float(
        torch.max(torch.abs(before_distance - after_distance)).item()
    )
    determinants = np.linalg.det(np.asarray(matrices_float64, dtype=np.float64))
    determinant_error = float(np.max(np.abs(determinants - 1.0)))
    if max(norm_error, dot_error, distance_error, determinant_error) > tolerance:
        raise AssertionError("rotated XYZ failed a geometry preservation gate")
    return {
        "probe_points_per_shape": probe_count,
        "maximum_norm_error": norm_error,
        "maximum_dot_product_error": dot_error,
        "maximum_pairwise_distance_error": distance_error,
        "maximum_determinant_error": determinant_error,
        "tolerance": float(tolerance),
    }


def _run_rotation_condition(
    adapter: FrozenSegmentationAdapter,
    plan: FrozenShapeNetPartPlan,
    bank: FrozenRotationBank,
    bank_indices: np.ndarray,
    condition: Mapping[str, object],
    *,
    batch_size: int,
    forward_seed: int,
    geometry_atol: float,
) -> tuple[dict[str, object], dict[str, object], str]:
    condition_index = int(condition["index"])
    matrices_float64 = np.asarray(
        bank.matrices[bank_indices, condition_index],
        dtype=np.float64,
    )
    matrices_float32 = np.asarray(matrices_float64, dtype=np.float32)
    accumulator = _ConditionAccumulator(plan)
    geometry_maxima = {
        "probe_points_per_shape": 32,
        "maximum_norm_error": 0.0,
        "maximum_dot_product_error": 0.0,
        "maximum_pairwise_distance_error": 0.0,
        "maximum_determinant_error": 0.0,
        "tolerance": float(geometry_atol),
    }
    for start in range(0, len(plan.example_ids), batch_size):
        end = min(start + batch_size, len(plan.example_ids))
        points = torch.from_numpy(
            np.array(plan.points[start:end], dtype=np.float32, copy=True)
        )
        labels = torch.from_numpy(
            np.array(plan.class_labels[start:end], dtype=np.int64, copy=True)
        ).long()
        matrices = torch.from_numpy(
            np.array(matrices_float32[start:end], copy=True)
        )
        rotated = torch.bmm(points, matrices.transpose(1, 2))
        geometry = _geometry_probe(
            points,
            rotated,
            matrices_float64[start:end],
            tolerance=geometry_atol,
        )
        for key in (
            "maximum_norm_error",
            "maximum_dot_product_error",
            "maximum_pairwise_distance_error",
            "maximum_determinant_error",
        ):
            geometry_maxima[key] = max(geometry_maxima[key], geometry[key])
        seed = _derived_seed(forward_seed, str(condition["name"]), start)
        logits = _forward(adapter, rotated, labels, seed=seed)
        accumulator.add(start, logits, rotated)
    return (
        accumulator.finalize(),
        geometry_maxima,
        logical_array_sha256(matrices_float64),
    )


def _bootstrap_for_metrics(
    metrics: Mapping[str, object],
    clean_metrics: Mapping[str, object],
    *,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    candidate = np.asarray(
        metrics["per_shape_instance_iou_fraction"],
        dtype=np.float64,
    )
    clean = np.asarray(
        clean_metrics["per_shape_instance_iou_fraction"],
        dtype=np.float64,
    )
    return paired_shape_bootstrap_ci(
        candidate,
        clean,
        seed=seed,
        replicates=replicates,
        scale=100.0,
    )


def _aggregate_conditions(
    condition_records: Sequence[Mapping[str, object]],
    clean_metrics: Mapping[str, object],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, object]:
    yaw = [
        record for record in condition_records if record["condition"]["family"] == "yaw"
    ]
    so3 = [
        record for record in condition_records if record["condition"]["family"] == "so3"
    ]
    if len(yaw) != 7 or len(so3) != 8:
        raise AssertionError("full rotation condition family is incomplete")
    yaw_shape_values = np.asarray(
        [
            record["metrics"]["per_shape_instance_iou_fraction"]
            for record in yaw
        ],
        dtype=np.float64,
    )
    yaw_per_shape_mean = yaw_shape_values.mean(axis=0)
    yaw_scores = [
        float(record["metrics"]["instance_miou_percent"]) for record in yaw
    ]
    worst_index = int(np.argmin(yaw_scores))
    yaw_aggregate_metrics = {
        "per_shape_instance_iou_fraction": yaw_per_shape_mean.tolist()
    }
    yaw_bootstrap = _bootstrap_for_metrics(
        yaw_aggregate_metrics,
        clean_metrics,
        seed=_derived_seed(bootstrap_seed, "yaw_aggregate"),
        replicates=bootstrap_replicates,
    )

    so3_shape_values = np.asarray(
        [
            record["metrics"]["per_shape_instance_iou_fraction"]
            for record in so3
        ],
        dtype=np.float64,
    )
    so3_per_shape_mean = so3_shape_values.mean(axis=0)
    so3_repeat_scores = [
        float(record["metrics"]["instance_miou_percent"]) for record in so3
    ]
    so3_aggregate_metrics = {
        "per_shape_instance_iou_fraction": so3_per_shape_mean.tolist()
    }
    so3_bootstrap = _bootstrap_for_metrics(
        so3_aggregate_metrics,
        clean_metrics,
        seed=_derived_seed(bootstrap_seed, "so3_aggregate"),
        replicates=bootstrap_replicates,
    )
    return {
        "yaw_circular_grid_identity_excluded": {
            "condition_values_percent": {
                str(record["condition"]["name"]): float(
                    record["metrics"]["instance_miou_percent"]
                )
                for record in yaw
            },
            "mean_percent": float(np.mean(yaw_scores)),
            "per_shape_mean_instance_iou_fraction": yaw_per_shape_mean.tolist(),
            "worst_angle": {
                "condition": yaw[worst_index]["condition"]["name"],
                "degrees": yaw[worst_index]["condition"]["degrees"],
                "instance_miou_percent": yaw_scores[worst_index],
            },
            "paired_shape_bootstrap": yaw_bootstrap,
        },
        "so3_haar": {
            "aggregation_order": "mean_within_shape_over_8_then_across_shapes",
            "repeat_values_percent": {
                str(record["condition"]["name"]): float(
                    record["metrics"]["instance_miou_percent"]
                )
                for record in so3
            },
            "mean_within_shape_then_dataset_percent": float(
                100.0 * np.mean(so3_per_shape_mean)
            ),
            "repeat_score_mean_percent": float(np.mean(so3_repeat_scores)),
            "repeat_score_median_percent": float(np.median(so3_repeat_scores)),
            "repeat_score_range_percent": [
                float(np.min(so3_repeat_scores)),
                float(np.max(so3_repeat_scores)),
            ],
            "per_shape_mean_instance_iou_fraction": so3_per_shape_mean.tolist(),
            "paired_shape_bootstrap": so3_bootstrap,
        },
    }


def _reproduction_gate(
    model_name: str,
    clean_metrics: Mapping[str, object],
    adapter_metadata: Mapping[str, object],
) -> dict[str, object]:
    measured = float(clean_metrics["instance_miou_percent"])
    if model_name == "pointmae":
        metadata = adapter_metadata["checkpoint_release_metadata"]
        stored_values = metadata["recorded_values"]
        stored_fraction = float(stored_values["inctance_avg_iou"])
        stored_matches = (
            abs(stored_fraction - 0.8607242789186563) <= 1.0e-12
        )
        difference = measured - POINTMAE_RELEASE_INSTANCE_MIOU_PERCENT
        score_matches = (
            abs(difference) <= POINTMAE_CLEAN_TOLERANCE_PERCENTAGE_POINTS
        )
        return {
            "status": "PASS" if stored_matches and score_matches else "FAIL",
            "enforced": True,
            "reference": {
                "source": "part_seg.pth inctance_avg_iou metadata",
                "stored_fraction": stored_fraction,
                "percent": POINTMAE_RELEASE_INSTANCE_MIOU_PERCENT,
            },
            "measured_percent": measured,
            "difference_percentage_points": difference,
            "tolerance_percentage_points": (
                POINTMAE_CLEAN_TOLERANCE_PERCENTAGE_POINTS
            ),
            "checkpoint_metadata_matches_bound_release": stored_matches,
            "confusion_gate": {
                "available_in_source_artifact": False,
                "enforced": False,
                "reason": (
                    "the official release container records aggregate IoUs but "
                    "does not contain a source confusion matrix"
                ),
            },
        }
    if model_name == "pointgpt":
        return {
            "status": "PASS",
            "enforced": True,
            "gate": "this_driver_clean_bypass_equals_identity_replay",
            "measured_percent": measured,
            "r7c_context": {
                "value_percent": POINTGPT_R7C_CONTEXT_PERCENT,
                "enforced_as_equality": False,
                "reason": (
                    "R7C uses its own retention-oriented frozen sampling and "
                    "ID accounting. v2 uses one official-style replacement "
                    "sample shared across rotations and models. Cross-protocol "
                    "equality would be invalid."
                ),
            },
        }
    raise ValueError("unknown model name")


def _evaluate_cell(
    model_name: str,
    adapter: FrozenSegmentationAdapter,
    plan: FrozenShapeNetPartPlan,
    bank: FrozenRotationBank,
    *,
    batch_size: int,
    forward_seed: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    clean_only: bool,
    logits_atol: float,
    logits_rtol: float,
    geometry_atol: float,
    run_config: Mapping[str, object],
) -> dict[str, object]:
    print("%s: loading frozen system" % model_name, flush=True)
    adapter.load_frozen_system()
    print("%s: dual clean gate" % model_name, flush=True)
    clean_metrics, dual_gate = _dual_clean(
        adapter,
        plan,
        batch_size=batch_size,
        forward_seed=forward_seed,
        logits_atol=logits_atol,
        logits_rtol=logits_rtol,
    )
    adapter_metadata = adapter.metadata()
    reproduction = _reproduction_gate(
        model_name,
        clean_metrics,
        adapter_metadata,
    )
    clean_record: dict[str, object] = {
        "condition": {
            "name": "clean_bypass",
            "family": "clean",
            "rotation_helper_used": False,
        },
        "metrics": clean_metrics,
        "paired_shape_bootstrap": _bootstrap_for_metrics(
            clean_metrics,
            clean_metrics,
            seed=_derived_seed(bootstrap_seed, "clean"),
            replicates=bootstrap_replicates,
        ),
    }
    identity_condition = bank.conditions[0]
    identity_record: dict[str, object] = {
        "condition": dict(identity_condition),
        "metrics": copy.deepcopy(clean_metrics),
        "matrix_logical_sha256": logical_array_sha256(
            np.asarray(
                [
                    bank.matrix(example_id, "identity_control")
                    for example_id in plan.example_ids
                ],
                dtype=np.float64,
            )
        ),
        "dual_clean_gate": dual_gate,
        "paired_shape_bootstrap": _bootstrap_for_metrics(
            clean_metrics,
            clean_metrics,
            seed=_derived_seed(bootstrap_seed, "identity_control"),
            replicates=bootstrap_replicates,
        ),
    }
    condition_records: list[dict[str, object]] = [clean_record, identity_record]
    invalid_reasons: list[str] = []
    if reproduction["status"] != "PASS":
        invalid_reasons.append("clean reproduction gate failed")

    bank_indices = np.asarray(
        [bank.example_index(example_id) for example_id in plan.example_ids],
        dtype=np.int64,
    )
    if set(bank.example_ids) != set(plan.example_ids):
        raise AssertionError("rotation bank and sample plan ID populations differ")
    if not clean_only and not invalid_reasons:
        for condition in bank.conditions[1:]:
            print(
                "%s: evaluating %s" % (model_name, condition["name"]),
                flush=True,
            )
            metrics, geometry, matrix_hash = _run_rotation_condition(
                adapter,
                plan,
                bank,
                bank_indices,
                condition,
                batch_size=batch_size,
                forward_seed=forward_seed,
                geometry_atol=geometry_atol,
            )
            record = {
                "condition": dict(condition),
                "metrics": metrics,
                "geometry_gate": geometry,
                "matrix_logical_sha256": matrix_hash,
                "paired_shape_bootstrap": _bootstrap_for_metrics(
                    metrics,
                    clean_metrics,
                    seed=_derived_seed(bootstrap_seed, condition["name"]),
                    replicates=bootstrap_replicates,
                ),
            }
            condition_records.append(record)
    aggregates = (
        None
        if clean_only or invalid_reasons
        else _aggregate_conditions(
            condition_records,
            clean_metrics,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
    )
    final_adapter_metadata = adapter.metadata()
    if final_adapter_metadata.get("parameter_hash_invariant") is not True:
        invalid_reasons.append("parameter hash invariance gate failed")

    plan_artifacts = plan.manifest["artifacts"]
    cell: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "INVALID" if invalid_reasons else "VALID",
        "invalid_reasons": invalid_reasons,
        "model": model_name,
        "task": "ShapeNetPart_part_segmentation",
        "clean_only": bool(clean_only),
        "system_tuple": {
            "data_population": "official ShapeNetPart test split",
            "sample_plan_manifest_sha256": plan.manifest["manifest_sha256"],
            "rotation_bank_manifest_sha256": bank.manifest["manifest_sha256"],
            "normal_channel": False,
            "npoint": NPOINT,
            "metric": "official_category_valid_per_shape_instance_miou",
            "adapter": final_adapter_metadata,
        },
        "hashes": {
            "checkpoint_sha256": final_adapter_metadata["checkpoint_sha256"],
            "sample_plan_manifest_sha256": plan.manifest["manifest_sha256"],
            "dataset_manifest_sha256": plan.manifest[
                "dataset_manifest_sha256"
            ],
            "sampled_indices_logical_sha256": plan_artifacts[
                "sampled_indices.npy"
            ]["logical_sha256"],
            "model_ready_points_logical_sha256": plan_artifacts["points.npy"][
                "logical_sha256"
            ],
            "targets_logical_sha256": plan_artifacts["targets.npy"][
                "logical_sha256"
            ],
            "category_part_mapping_logical_sha256": plan_artifacts[
                "category_part_mapping.json"
            ]["logical_sha256"],
            "rotation_bank_content_sha256": bank.manifest["content_sha256"],
            "evaluation_config_sha256": canonical_json_sha256(run_config),
            "driver_sha256": file_sha256(Path(__file__).resolve()),
            "metric_and_sampling_source_sha256": file_sha256(
                Path(__file__).with_name("shapenetpart.py")
            ),
        },
        "input_contract": {
            "consumed_features": ["xyz", "category_one_hot_16"],
            "xyz_dtype": "float32",
            "normal_channel": False,
            "sampled_once_with_replacement": True,
            "shared_across_rotations_and_models": True,
            "no_r7c_deduplication": True,
        },
        "clean_reproduction_gate": reproduction,
        "dual_clean_gate": dual_gate,
        "conditions": condition_records,
        "aggregates": aggregates,
        "claim_boundary": (
            "task-conditional full-system rotation sensitivity; no cross-model "
            "absolute robustness ranking and no representation-invariance claim"
        ),
        "adapter_metadata": final_adapter_metadata,
    }
    cell["payload_sha256"] = canonical_json_sha256(cell)
    return cell


def _asset_freeze(args: argparse.Namespace) -> int:
    plan_directory = Path(args.sample_plan_dir)
    if plan_directory.exists():
        plan = load_shapenetpart_sample_plan(
            plan_directory,
            dataset_root=Path(args.dataset_root),
        )
        if int(plan.manifest["sampling"]["seed"]) != int(args.sample_seed):
            raise ValueError("existing sample plan uses a different seed")
    else:
        plan = freeze_shapenetpart_sample_plan(
            Path(args.dataset_root),
            plan_directory,
            seed=int(args.sample_seed),
        )
    bank_directory = Path(args.rotation_bank_dir)
    if bank_directory.exists():
        bank = load_frozen_bank(bank_directory)
        if int(bank.manifest["global_seed"]) != int(args.rotation_seed):
            raise ValueError("existing rotation bank uses a different seed")
    else:
        bank = generate_frozen_bank(
            bank_directory,
            plan.example_ids,
            global_seed=int(args.rotation_seed),
        )
    if set(bank.example_ids) != set(plan.example_ids):
        raise AssertionError("frozen asset populations differ")
    print(
        json.dumps(
            {
                "status": "FROZEN_AND_VERIFIED",
                "sample_plan_directory": str(plan.directory),
                "sample_plan_manifest_sha256": plan.manifest["manifest_sha256"],
                "rotation_bank_directory": str(bank.directory),
                "rotation_bank_manifest_sha256": bank.manifest["manifest_sha256"],
                "num_shapes": len(plan.example_ids),
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShapeNetPart rotation_matrix_v2 segmentation evaluation"
    )
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--sample_plan_dir", required=True, type=Path)
    parser.add_argument("--rotation_bank_dir", required=True, type=Path)
    parser.add_argument("--freeze_assets_only", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--rotation_seed", type=int, default=20260728)
    parser.add_argument(
        "--expected_sample_manifest_sha256",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expected_rotation_manifest_sha256",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pointgpt", "pointmae"),
        default=("pointgpt", "pointmae"),
    )
    parser.add_argument("--pointgpt_ckpt", type=Path, default=None)
    parser.add_argument("--pointmae_ckpt", type=Path, default=None)
    parser.add_argument("--pointmae_snapshots_root", type=Path, default=None)
    parser.add_argument(
        "--expected_pointgpt_checkpoint_sha256",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expected_pointmae_checkpoint_sha256",
        type=str,
        default=None,
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--forward_seed", type=int, default=0)
    parser.add_argument("--bootstrap_seed", type=int, default=20260728)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--identity_logits_atol", type=float, default=0.0)
    parser.add_argument("--identity_logits_rtol", type=float, default=0.0)
    parser.add_argument("--geometry_atol", type=float, default=5.0e-6)
    return parser.parse_args()


def _validate_eval_args(args: argparse.Namespace) -> tuple[str, ...]:
    models = tuple(args.models)
    if len(models) != len(set(models)):
        raise ValueError("--models must not contain duplicates")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")
    if int(args.bootstrap_replicates) < 2:
        raise ValueError("--bootstrap_replicates must be at least 2")
    for value in (
        args.sample_seed,
        args.rotation_seed,
        args.forward_seed,
        args.bootstrap_seed,
    ):
        if int(value) < 0:
            raise ValueError("all seeds must be non-negative")
    for value in (
        args.identity_logits_atol,
        args.identity_logits_rtol,
        args.geometry_atol,
    ):
        if float(value) < 0.0:
            raise ValueError("all tolerances must be non-negative")
    if args.output_dir is None:
        raise ValueError("--output_dir is required for evaluation")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite output directory")
    if not args.expected_sample_manifest_sha256:
        raise ValueError("--expected_sample_manifest_sha256 is required")
    if not args.expected_rotation_manifest_sha256:
        raise ValueError("--expected_rotation_manifest_sha256 is required")
    for model_name in models:
        checkpoint = getattr(args, "%s_ckpt" % model_name)
        expected_hash = getattr(
            args,
            "expected_%s_checkpoint_sha256" % model_name,
        )
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError("missing %s checkpoint" % model_name)
        if not expected_hash:
            raise ValueError(
                "--expected_%s_checkpoint_sha256 is required" % model_name
            )
        actual_hash = file_sha256(Path(checkpoint))
        if actual_hash != expected_hash:
            raise ValueError("%s checkpoint differs from preregistration" % model_name)
    return models


def _write_results(
    output_directory: Path,
    cells: Sequence[Mapping[str, object]],
    *,
    run_config: Mapping[str, object],
) -> None:
    target = Path(output_directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=target.name + ".partial.", dir=target.parent)
    )
    try:
        cell_files: dict[str, object] = {}
        for cell in cells:
            filename = "%s.json" % cell["model"]
            path = staging / filename
            path.write_bytes(canonical_json_bytes(cell))
            cell_files[str(cell["model"])] = {
                "filename": filename,
                "status": cell["status"],
                "payload_sha256": cell["payload_sha256"],
                "file_sha256": file_sha256(path),
            }
        run_manifest: dict[str, object] = {
            "schema": RUN_SCHEMA,
            "status": (
                "VALID"
                if all(cell["status"] == "VALID" for cell in cells)
                else "INVALID"
            ),
            "run_config": run_config,
            "run_config_sha256": canonical_json_sha256(run_config),
            "cell_files": cell_files,
        }
        run_manifest["payload_sha256"] = canonical_json_sha256(run_manifest)
        (staging / "run_manifest.json").write_bytes(
            canonical_json_bytes(run_manifest)
        )
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    args = parse_args()
    if args.freeze_assets_only:
        return _asset_freeze(args)
    models = _validate_eval_args(args)
    configure_deterministic_runtime()
    plan = load_shapenetpart_sample_plan(
        args.sample_plan_dir,
        dataset_root=args.dataset_root,
        expected_manifest_sha256=args.expected_sample_manifest_sha256,
    )
    bank = load_frozen_bank(
        args.rotation_bank_dir,
        expected_manifest_sha256=args.expected_rotation_manifest_sha256,
    )
    if set(bank.example_ids) != set(plan.example_ids):
        raise AssertionError("frozen bank and sample plan populations differ")
    if int(plan.manifest["sampling"]["seed"]) != int(args.sample_seed):
        raise ValueError("sample seed differs from frozen plan")
    if int(bank.manifest["global_seed"]) != int(args.rotation_seed):
        raise ValueError("rotation seed differs from frozen bank")

    run_config = {
        "protocol": "rotation_matrix_v2",
        "models": list(models),
        "sample_seed": int(args.sample_seed),
        "rotation_seed": int(args.rotation_seed),
        "forward_seed": int(args.forward_seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "batch_size": int(args.batch_size),
        "clean_only": bool(args.clean_only),
        "device": str(args.device),
        "identity_logits_atol": float(args.identity_logits_atol),
        "identity_logits_rtol": float(args.identity_logits_rtol),
        "geometry_atol": float(args.geometry_atol),
        "sample_plan_manifest_sha256": args.expected_sample_manifest_sha256,
        "rotation_bank_manifest_sha256": args.expected_rotation_manifest_sha256,
        "checkpoint_selection": {
            model: {
                "path": str(Path(getattr(args, "%s_ckpt" % model)).resolve()),
                "sha256": getattr(
                    args,
                    "expected_%s_checkpoint_sha256" % model,
                ),
            }
            for model in models
        },
    }
    cells: list[dict[str, object]] = []
    for model_name in models:
        if model_name == "pointgpt":
            adapter: FrozenSegmentationAdapter = PointGPTSegAdapter(
                checkpoint_path=args.pointgpt_ckpt,
                device=args.device,
            )
        else:
            adapter = PointMAESegAdapter(
                checkpoint_path=args.pointmae_ckpt,
                snapshots_root=args.pointmae_snapshots_root,
                device=args.device,
            )
        cell = _evaluate_cell(
            model_name,
            adapter,
            plan,
            bank,
            batch_size=int(args.batch_size),
            forward_seed=int(args.forward_seed),
            bootstrap_seed=int(args.bootstrap_seed),
            bootstrap_replicates=int(args.bootstrap_replicates),
            clean_only=bool(args.clean_only),
            logits_atol=float(args.identity_logits_atol),
            logits_rtol=float(args.identity_logits_rtol),
            geometry_atol=float(args.geometry_atol),
            run_config=run_config,
        )
        cells.append(cell)
    _write_results(args.output_dir, cells, run_config=run_config)
    status = "VALID" if all(cell["status"] == "VALID" for cell in cells) else "INVALID"
    print(
        "%s output_dir=%s" % (status, Path(args.output_dir).resolve()),
        flush=True,
    )
    return 0 if status == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
