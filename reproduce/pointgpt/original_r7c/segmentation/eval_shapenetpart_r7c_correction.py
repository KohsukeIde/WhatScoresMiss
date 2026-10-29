#!/usr/bin/env python
"""Fail-closed R7C ShapeNetPart retention correction driver.

The driver evaluates only the newly declared PointGPT protocol. It freezes a
single 2048-point base, consumes only a verified R7C plan after freezing,
executes deterministic batched forwards, and derives both CSV views from one
canonical JSON payload.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if import_root in sys.path:
        sys.path.remove(import_root)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(REPO_ROOT))

from rebuttal_eval.determinism import (  # noqa: E402
    configure_deterministic_model,
    configure_deterministic_runtime,
    deterministic_model_forward,
)
from rebuttal_eval.r7 import (  # noqa: E402
    class_intersection_union_over_50,
    class_mean_iou_from_intersection_union,
    clean_subset_iou,
    deduped_pred_target_over_50,
    shape_part_iou,
)
from rebuttal_eval.r7c import (  # noqa: E402
    BASE_SAMPLING_ORIGINAL_REPLACEMENT,
    BASE_SAMPLING_UNIQUE_RAW_ID,
    BASE_SAMPLINGS,
    FRACTIONS,
    NPOINT,
    PERTURBATIONS,
    array_sha256,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    input_map_name,
    load_plan_array,
    materialize_frozen_plan,
    model_label_for_base_sampling,
    original_replacement_base_indices,
    retained_counts_name,
    retained_original_ids_name,
    retained_positions_name,
    result_schema_for_base_sampling,
    stamp_result_payload,
    verify_frozen_plan,
    verify_result_payload,
    write_canonical_json,
    write_long_csv_from_json,
    write_wide_csv_from_long,
)
from dataset import PartNormalDataset  # noqa: E402
from main import get_model_loss, seg_classes, to_categorical  # noqa: E402


DATASET_MANIFEST_SCHEMA = "pointgpt_r7c_dataset_content_manifest_v1"
SUPERSEDED_SOURCE_RESULT = (
    "3D-NEPA/results/pointgpt_unique_retained/"
    "shapenetpart_official_support_unique.json"
)
MODEL_MODULE = "pt"
MODEL_NAME = "PointGPT_S"
GROUP_MODE = "fps_knn"
GROUP_RADIUS = 0.22
GROUP_VOXEL_GRID = 6
NUM_CATEGORIES = 16
NUM_PARTS = 50


def trace_schema_for_base_sampling(base_sampling: str) -> str:
    if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
        return "pointgpt_r7c_original_sampling_replay_object_trace_v1"
    return "pointgpt_r7c_object_trace_v1"


def selection_protocol_for_base_sampling(base_sampling: str) -> str:
    if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
        return (
            "pointgpt_r7c_original_replacement_support_preserving_v1"
        )
    return "pointgpt_r7c_unique_raw_id_support_preserving_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "R7C PointGPT ShapeNetPart retention correction"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--trace_jsonl_gz", required=True)
    parser.add_argument("--long_csv", required=True)
    parser.add_argument("--wide_csv", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--base_sampling",
        choices=BASE_SAMPLINGS,
        required=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Dict[str, Path]:
    if int(args.seed) < 0:
        raise ValueError("--seed must be non-negative")
    if int(args.replicates) != 2:
        raise ValueError("R7C is locked to exactly two replicates")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")

    checkpoint = Path(args.ckpt).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    plan_dir = Path(args.plan_dir).expanduser().resolve()
    outputs = {
        "output_json": Path(args.output_json).expanduser().resolve(),
        "trace_jsonl_gz": Path(args.trace_jsonl_gz).expanduser().resolve(),
        "long_csv": Path(args.long_csv).expanduser().resolve(),
        "wide_csv": Path(args.wide_csv).expanduser().resolve(),
    }
    if not checkpoint.is_file():
        raise FileNotFoundError("missing checkpoint: %s" % checkpoint)
    if not root.is_dir():
        raise FileNotFoundError("missing ShapeNetPart root: %s" % root)
    if plan_dir.exists() and not plan_dir.is_dir():
        raise NotADirectoryError("R7C plan path is not a directory: %s" % plan_dir)
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("R7C output paths must be distinct")
    if plan_dir in outputs.values():
        raise ValueError("R7C plan directory cannot also be an output file")
    for name, path in outputs.items():
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "refusing to overwrite existing %s artifact: %s" % (name, path)
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint": checkpoint,
        "root": root,
        "plan_dir": plan_dir,
        **outputs,
    }


def normalize_xyz(points: np.ndarray) -> np.ndarray:
    """Match segmentation/pointnet_util.py:15-20 on float32 xyz."""
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 3:
        raise ValueError("raw xyz must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(value)):
        raise ValueError("raw xyz contains a non-finite value")
    centroid = np.mean(value, axis=0)
    centered = value - centroid
    radius = float(np.max(np.sqrt(np.sum(centered**2, axis=1))))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("degenerate ShapeNetPart object has zero normalization radius")
    normalized = centered / radius
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized xyz contains a non-finite value")
    return np.asarray(normalized, dtype=np.float32)


def _relative_file(path: Path, root: Path) -> Tuple[Path, str]:
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise AssertionError(
            "dataset file resolves outside the declared root: %s" % resolved
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError("dataset path is not a file: %s" % resolved)
    return resolved, relative.as_posix()


def build_dataset_manifest(
    root: Path, dataset: PartNormalDataset
) -> Dict[str, object]:
    """Hash every evaluated object file and every loader metadata input."""
    root = Path(root).resolve(strict=True)
    if len(dataset) <= 0:
        raise ValueError("ShapeNetPart test split is empty")

    metadata_relatives = (
        "synsetoffset2category.txt",
        "train_test_split/shuffled_train_file_list.json",
        "train_test_split/shuffled_val_file_list.json",
        "train_test_split/shuffled_test_file_list.json",
    )
    metadata_files = []
    for relative in metadata_relatives:
        path, normalized_relative = _relative_file(root / relative, root)
        metadata_files.append(
            {
                "relative_path": normalized_relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )

    evaluated_files = []
    seen = set()
    for object_index, (category, filename) in enumerate(dataset.datapath):
        path, relative = _relative_file(Path(filename), root)
        if relative in seen:
            raise AssertionError("duplicate test object path: %s" % relative)
        seen.add(relative)
        if category not in dataset.classes or category not in seg_classes:
            raise AssertionError("unbound ShapeNetPart category: %s" % category)
        evaluated_files.append(
            {
                "object_index": int(object_index),
                "category": str(category),
                "class_label": int(dataset.classes[category]),
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    if len(evaluated_files) != len(dataset):
        raise AssertionError("dataset manifest object count mismatch")

    content_binding = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "split": "test",
        "num_objects": int(len(dataset)),
        "normal_channel": False,
        "npoint": int(NPOINT),
        "file_parser": "numpy.loadtxt_then_float32",
        "normalization": (
            "float32 xyz centroid subtraction followed by division by maximum "
            "Euclidean radius, matching segmentation/pointnet_util.py:15-20"
        ),
        "metadata_files": metadata_files,
        "evaluated_files": evaluated_files,
    }
    manifest = {"root": str(root), **content_binding}
    manifest["content_sha256"] = canonical_json_sha256(content_binding)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _load_raw_object(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path).astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
        raise ValueError("invalid ShapeNetPart object array: %s" % path)
    if not np.all(np.isfinite(data)):
        raise ValueError("ShapeNetPart object contains non-finite data: %s" % path)
    raw_labels = data[:, -1]
    rounded = np.rint(raw_labels)
    if not np.array_equal(raw_labels, rounded):
        raise ValueError("ShapeNetPart target labels are not integral: %s" % path)
    return normalize_xyz(data[:, :3]), rounded.astype(np.int64)


def freeze_base_arrays(
    dataset: PartNormalDataset,
    dataset_manifest: Mapping[str, object],
    seed: int,
    base_sampling: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Freeze either declared R7C base without using stochastic ``__getitem__``.

    ``original_replacement`` mirrors
    ``segmentation/eval_shapenetpart_r7_resampling.py:545-566`` exactly:
    one shared ``np.random.RandomState(seed)`` draws 2048 raw rows with
    replacement and then consumes ``permutation(2048)`` for each object. The
    underlying provider operation is ``segmentation/dataset.py:154-159``.
    """
    num_objects = len(dataset)
    points = np.empty((num_objects, NPOINT, 3), dtype=np.float32)
    targets = np.empty((num_objects, NPOINT), dtype=np.int64)
    original_ids = np.empty((num_objects, NPOINT), dtype=np.int64)
    class_labels = np.empty((num_objects, 1), dtype=np.int64)
    object_keys: List[Dict[str, object]] = []
    manifest_files = dataset_manifest["evaluated_files"]
    if len(manifest_files) != num_objects:
        raise AssertionError("full dataset manifest is not aligned to the test split")

    from rebuttal_eval.r7c import frozen_base_indices

    replay_rng = (
        np.random.RandomState(int(seed))
        if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT
        else None
    )
    root = Path(str(dataset_manifest["root"]))
    for object_index in tqdm(
        range(num_objects), desc="R7C freeze 2048-point base", unit="object"
    ):
        category, filename = dataset.datapath[object_index]
        path, relative = _relative_file(Path(filename), root)
        record = manifest_files[object_index]
        expected_file_sha256 = str(record["sha256"])
        if (
            int(record["object_index"]) != object_index
            or str(record["category"]) != category
            or str(record["relative_path"]) != relative
            or file_sha256(path) != expected_file_sha256
        ):
            raise AssertionError("dataset changed while freezing the R7C base")
        raw_points, raw_targets = _load_raw_object(path)
        if file_sha256(path) != expected_file_sha256:
            raise AssertionError(
                "dataset file changed while reading the R7C frozen base"
            )
        raw_count = int(raw_targets.size)
        legacy_priority_sha256 = None
        if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
            choice, legacy_priority = original_replacement_base_indices(
                raw_count,
                NPOINT,
                replay_rng,
            )
            selection_mode = (
                "partnormal_2048_with_replacement_shared_randomstate"
            )
            selection_seed = int(seed)
            legacy_priority_sha256 = array_sha256(legacy_priority)
        else:
            choice, selection_mode, selection_seed = frozen_base_indices(
                raw_count, NPOINT, seed, object_index
            )
        class_label = int(dataset.classes[category])
        valid_parts = np.asarray(seg_classes[category], dtype=np.int64)
        if class_label != int(record["class_label"]):
            raise AssertionError("dataset class-label binding changed during freeze")
        invalid = np.setdiff1d(np.unique(raw_targets), valid_parts)
        if invalid.size:
            raise AssertionError(
                "target label is invalid for category %s: %s"
                % (category, invalid.tolist())
            )
        points[object_index] = raw_points[choice]
        targets[object_index] = raw_targets[choice]
        original_ids[object_index] = choice
        class_labels[object_index, 0] = class_label
        unique_count = int(np.unique(choice).size)
        if (
            base_sampling == BASE_SAMPLING_UNIQUE_RAW_ID
            and unique_count != min(raw_count, NPOINT)
        ):
            raise AssertionError("frozen base did not preserve its unique raw-ID universe")
        key = {
            "object_index": int(object_index),
            "category": str(category),
            "class_label": class_label,
            "valid_part_ids": valid_parts.tolist(),
            "relative_path": relative,
            "raw_file_sha256": expected_file_sha256,
            "raw_count": raw_count,
            "base_selection_mode": selection_mode,
            "base_selection_seed": int(selection_seed),
            "base_unique_raw_id_count": unique_count,
        }
        if legacy_priority_sha256 is not None:
            key["legacy_priority_sha256"] = legacy_priority_sha256
        object_keys.append(key)
    return points, targets, original_ids, class_labels, object_keys


def load_or_materialize_plan(
    plan_dir: Path,
    dataset: PartNormalDataset,
    dataset_manifest: Mapping[str, object],
    seed: int,
    replicates: int,
    base_sampling: str,
) -> Dict[str, object]:
    manifest_path = plan_dir / "manifest.json"
    if manifest_path.is_file():
        return verify_frozen_plan(
            plan_dir,
            expected_seed=seed,
            expected_replicates=replicates,
            expected_dataset_content_sha256=str(
                dataset_manifest["content_sha256"]
            ),
            expected_npoint=NPOINT,
            expected_base_sampling=base_sampling,
        )
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise FileExistsError(
            "R7C plan directory is non-empty but has no manifest: %s" % plan_dir
        )
    arrays = freeze_base_arrays(
        dataset,
        dataset_manifest,
        seed,
        base_sampling,
    )
    return materialize_frozen_plan(
        directory=plan_dir,
        points=arrays[0],
        targets=arrays[1],
        original_ids=arrays[2],
        class_labels=arrays[3],
        object_keys=arrays[4],
        dataset_manifest=dataset_manifest,
        seed=seed,
        replicates=replicates,
        fractions=FRACTIONS,
        base_sampling=base_sampling,
    )


def build_model_strict(
    checkpoint: Path,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Strictly load the official architecture and record both key lists."""
    checkpoint_sha256 = file_sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise AssertionError("checkpoint changed while it was being loaded")
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint container is not a mapping")
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        state_dict_source = "model_state_dict"
    elif state and all(torch.is_tensor(value) for value in state.values()):
        state_dict = state
        state_dict_source = "direct_state_dict"
    else:
        raise KeyError("checkpoint has no unambiguous model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint model_state_dict is empty or invalid")
    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError("checkpoint state key is not a string")
    non_tensors = sorted(
        key for key, value in state_dict.items() if not torch.is_tensor(value)
    )
    if non_tensors:
        raise TypeError(
            "checkpoint model_state_dict contains non-tensors: %s"
            % non_tensors[:10]
        )

    module = importlib.import_module(MODEL_MODULE)
    model_args = SimpleNamespace(
        model_name=MODEL_NAME,
        group_mode=GROUP_MODE,
        group_radius=GROUP_RADIUS,
        group_voxel_grid=GROUP_VOXEL_GRID,
    )
    model, criterion = get_model_loss(module, model_args, num_part=NUM_PARTS)
    del criterion
    model_state = model.state_dict()
    checkpoint_keys = sorted(state_dict)
    model_keys = sorted(model_state)
    missing_keys = sorted(set(model_keys) - set(checkpoint_keys))
    unexpected_keys = sorted(set(checkpoint_keys) - set(model_keys))
    shared_keys = sorted(set(model_keys) & set(checkpoint_keys))
    shape_mismatches = [
        {
            "key": key,
            "model_shape": list(model_state[key].shape),
            "checkpoint_shape": list(state_dict[key].shape),
        }
        for key in shared_keys
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    ]
    load_record = {
        "state_dict_source": state_dict_source,
        "checkpoint_keys": checkpoint_keys,
        "model_keys": model_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(model_keys),
        "checkpoint_key_list_sha256": canonical_json_sha256(checkpoint_keys),
        "model_key_list_sha256": canonical_json_sha256(model_keys),
        "key_transform": "none",
        "strict": True,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise AssertionError(
            "strict checkpoint compatibility failed: missing=%d unexpected=%d "
            "shape_mismatches=%d"
            % (len(missing_keys), len(unexpected_keys), len(shape_mismatches))
        )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("strict checkpoint loader returned incompatible keys")
    configure_deterministic_model(model)
    return model, load_record


def normalize_logits(logits: torch.Tensor) -> np.ndarray:
    value = logits.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[1] == NUM_PARTS:
        value = np.transpose(value, (0, 2, 1))
    if value.ndim != 3 or value.shape[-1] != NUM_PARTS:
        raise AssertionError(
            "unexpected segmentation logit shape: %s" % (value.shape,)
        )
    value = np.ascontiguousarray(value)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("segmentation logits contain a non-finite value")
    return value


def _forward_seed(seed: int, batch_start: int, condition: str) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes([int(seed), int(batch_start), str(condition)])
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def run_forward(
    model: torch.nn.Module,
    points: np.ndarray,
    class_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    point_tensor = (
        torch.from_numpy(np.array(points, dtype=np.float32, copy=True))
        .cuda()
        .transpose(2, 1)
    )
    label_tensor = torch.from_numpy(
        np.array(class_labels, dtype=np.int64, copy=True)
    ).long().cuda()
    one_hot = to_categorical(label_tensor, NUM_CATEGORIES)
    logits = deterministic_model_forward(
        model, point_tensor, seed, cls_label=one_hot
    )
    return normalize_logits(logits)


def _update_array_digest(digest: "hashlib._Hash", value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes())


def _part_counts(
    targets: np.ndarray, valid_parts: Sequence[int]
) -> Dict[str, int]:
    target = np.asarray(targets, dtype=np.int64)
    return {
        str(int(part)): int(np.sum(target == int(part)))
        for part in valid_parts
    }


def _multiplicity_record(original_ids: np.ndarray) -> Dict[str, object]:
    _, counts = np.unique(np.asarray(original_ids, dtype=np.int64), return_counts=True)
    values, frequencies = np.unique(counts, return_counts=True)
    return {
        "num_input_positions": int(np.asarray(original_ids).size),
        "num_unique_raw_ids": int(counts.size),
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
        "histogram": {
            str(int(value)): int(frequency)
            for value, frequency in zip(values, frequencies)
        },
    }


def _summary(values: np.ndarray, scale: float = 1.0) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64) * float(scale)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize an empty or non-finite vector")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def _temp_path(target: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=target.name + ".partial.",
        dir=target.parent,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _commit_without_overwrite(
    staged: Sequence[Tuple[Path, Path]]
) -> None:
    committed: List[Path] = []
    try:
        for temporary, target in staged:
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    "output appeared during R7C execution: %s" % target
                )
            os.link(temporary, target)
            committed.append(target)
        for temporary, _ in staged:
            temporary.unlink()
    except Exception:
        for target in committed:
            if target.is_file():
                target.unlink()
        raise


def _cell_key(
    perturbation: str, fraction: float, replicate: int
) -> Tuple[str, float, int]:
    return perturbation, float(fraction), int(replicate)


def evaluate(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    dataset_manifest: Mapping[str, object],
    plan_manifest: Mapping[str, object],
    model: torch.nn.Module,
    checkpoint_load: Mapping[str, object],
) -> Dict[str, object]:
    plan_dir = paths["plan_dir"]
    trace_schema = trace_schema_for_base_sampling(args.base_sampling)
    result_schema = result_schema_for_base_sampling(args.base_sampling)
    model_label = model_label_for_base_sampling(args.base_sampling)
    selection_protocol = selection_protocol_for_base_sampling(
        args.base_sampling
    )
    points = load_plan_array(plan_dir, "base_points")
    targets = load_plan_array(plan_dir, "base_targets")
    original_ids = load_plan_array(plan_dir, "base_original_ids")
    class_labels = load_plan_array(plan_dir, "class_labels")
    object_keys = json.loads(
        (plan_dir / "object_keys.json").read_text(encoding="utf-8")
    )
    num_objects = int(plan_manifest["num_objects"])
    if num_objects != int(dataset_manifest["num_objects"]):
        raise AssertionError("R7C plan does not cover the full test population")
    if points.shape != (num_objects, NPOINT, 3):
        raise AssertionError("R7C frozen base has an unexpected shape")
    if targets.shape != (num_objects, NPOINT):
        raise AssertionError("R7C frozen target array has an unexpected shape")
    if original_ids.shape != (num_objects, NPOINT):
        raise AssertionError("R7C frozen ID array has an unexpected shape")
    if class_labels.shape != (num_objects, 1):
        raise AssertionError("R7C frozen category array has an unexpected shape")

    plan_arrays: Dict[Tuple[str, float, int], Dict[str, np.ndarray]] = {}
    metric_arrays: Dict[Tuple[str, float, int], Dict[str, np.ndarray]] = {}
    class_stats: Dict[Tuple[str, float, int], Dict[str, np.ndarray]] = {}
    for perturbation in PERTURBATIONS:
        for fraction in FRACTIONS:
            for replicate in range(int(args.replicates)):
                key = _cell_key(perturbation, fraction, replicate)
                plan_arrays[key] = {
                    "positions": load_plan_array(
                        plan_dir,
                        retained_positions_name(
                            perturbation, fraction, replicate
                        ),
                    ),
                    "retained_ids": load_plan_array(
                        plan_dir,
                        retained_original_ids_name(
                            perturbation, fraction, replicate
                        ),
                    ),
                    "counts": load_plan_array(
                        plan_dir,
                        retained_counts_name(
                            perturbation, fraction, replicate
                        ),
                    ),
                    "maps": load_plan_array(
                        plan_dir,
                        input_map_name(perturbation, fraction, replicate),
                    ),
                }
                metric_arrays[key] = {
                    "instance_miou": np.full(num_objects, np.nan),
                    "clean_subset_instance_miou": np.full(num_objects, np.nan),
                    "damage": np.full(num_objects, np.nan),
                    "full_clean_minus_stressed": np.full(num_objects, np.nan),
                    "unique_raw_id_count": np.full(num_objects, -1, dtype=np.int64),
                    "base_present_part_count": np.full(
                        num_objects, -1, dtype=np.int64
                    ),
                    "retained_present_part_count": np.full(
                        num_objects, -1, dtype=np.int64
                    ),
                    "maximum_input_multiplicity": np.full(
                        num_objects, -1, dtype=np.int64
                    ),
                }
                class_stats[key] = {
                    "intersections": np.zeros(NUM_PARTS, dtype=np.int64),
                    "unions": np.zeros(NUM_PARTS, dtype=np.int64),
                }

    clean_ious = np.full(num_objects, np.nan)
    clean_class_intersections = np.zeros(NUM_PARTS, dtype=np.int64)
    clean_class_unions = np.zeros(NUM_PARTS, dtype=np.int64)
    clean_digest_a = hashlib.sha256()
    clean_digest_b = hashlib.sha256()
    trace_count = 0
    trace_temporary = _temp_path(paths["trace_jsonl_gz"])

    try:
        with trace_temporary.open("wb") as raw_trace:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_trace,
                mtime=0,
            ) as trace_handle:
                with torch.no_grad():
                    for start in tqdm(
                        range(0, num_objects, int(args.batch_size)),
                        desc="R7C deterministic 26-config batches",
                        unit="batch",
                    ):
                        stop = min(num_objects, start + int(args.batch_size))
                        batch_points = np.asarray(points[start:stop])
                        batch_targets = np.asarray(targets[start:stop])
                        batch_ids = np.asarray(original_ids[start:stop])
                        batch_labels = np.asarray(class_labels[start:stop])
                        clean_seed = _forward_seed(args.seed, start, "clean")
                        clean_logits = run_forward(
                            model, batch_points, batch_labels, clean_seed
                        )
                        repeat_logits = run_forward(
                            model, batch_points, batch_labels, clean_seed
                        )
                        if (
                            clean_logits.shape != repeat_logits.shape
                            or clean_logits.dtype != repeat_logits.dtype
                            or clean_logits.tobytes() != repeat_logits.tobytes()
                        ):
                            raise AssertionError(
                                "clean repeat is not byte-identical at batch %d"
                                % start
                            )
                        _update_array_digest(clean_digest_a, clean_logits)
                        _update_array_digest(clean_digest_b, repeat_logits)

                        trace_rows: List[Dict[str, object]] = []
                        for row, object_index in enumerate(range(start, stop)):
                            key_record = object_keys[object_index]
                            valid_parts = np.asarray(
                                key_record["valid_part_ids"], dtype=np.int64
                            )
                            if (
                                int(class_labels[object_index, 0])
                                != int(key_record["class_label"])
                            ):
                                raise AssertionError(
                                    "frozen class label changed during evaluation"
                                )
                            clean_score = shape_part_iou(
                                clean_logits[row],
                                batch_targets[row],
                                batch_ids[row],
                                valid_parts,
                            )
                            clean_ious[object_index] = clean_score
                            clean_pred, clean_target = (
                                deduped_pred_target_over_50(
                                    clean_logits[row],
                                    batch_targets[row],
                                    batch_ids[row],
                                )
                            )
                            intersections, unions = (
                                class_intersection_union_over_50(
                                    clean_pred, clean_target
                                )
                            )
                            clean_class_intersections += intersections
                            clean_class_unions += unions
                            _, first_positions = np.unique(
                                batch_ids[row], return_index=True
                            )
                            base_unique_target = batch_targets[row, first_positions]
                            trace_rows.append(
                                {
                                    "schema": trace_schema,
                                    **(
                                        {
                                            "base_sampling": (
                                                args.base_sampling
                                            )
                                        }
                                        if args.base_sampling
                                        == BASE_SAMPLING_ORIGINAL_REPLACEMENT
                                        else {}
                                    ),
                                    "object_index": int(object_index),
                                    "relative_path": str(
                                        key_record["relative_path"]
                                    ),
                                    "category": str(key_record["category"]),
                                    "class_label": int(
                                        key_record["class_label"]
                                    ),
                                    "valid_part_ids": valid_parts.tolist(),
                                    "raw_count": int(key_record["raw_count"]),
                                    "base_selection_mode": str(
                                        key_record["base_selection_mode"]
                                    ),
                                    "base_selection_seed": int(
                                        key_record["base_selection_seed"]
                                    ),
                                    "base_unique_raw_id_count": int(
                                        key_record[
                                            "base_unique_raw_id_count"
                                        ]
                                    ),
                                    "base_part_unique_raw_id_counts": _part_counts(
                                        base_unique_target, valid_parts
                                    ),
                                    "clean": {
                                        "instance_miou_percent": float(
                                            100.0 * clean_score
                                        ),
                                        "audit_class_mean_iou_over_50_percent": float(
                                            100.0
                                            * class_mean_iou_from_intersection_union(
                                                intersections, unions
                                            )
                                        ),
                                        "audit_class_intersections_50": intersections.tolist(),
                                        "audit_class_unions_50": unions.tolist(),
                                        "logits_sha256": array_sha256(
                                            clean_logits[row]
                                        ),
                                    },
                                    "cells": [],
                                }
                            )

                        for perturbation in PERTURBATIONS:
                            for fraction in FRACTIONS:
                                for replicate in range(int(args.replicates)):
                                    key = _cell_key(
                                        perturbation, fraction, replicate
                                    )
                                    plan = plan_arrays[key]
                                    stressed_points = []
                                    stressed_targets = []
                                    stressed_ids = []
                                    retained_positions_rows = []
                                    retained_ids_rows = []
                                    for row, object_index in enumerate(
                                        range(start, stop)
                                    ):
                                        count = int(
                                            plan["counts"][object_index]
                                        )
                                        retained_positions = np.asarray(
                                            plan["positions"][
                                                object_index, :count
                                            ],
                                            dtype=np.int64,
                                        )
                                        retained_ids = np.asarray(
                                            plan["retained_ids"][
                                                object_index, :count
                                            ],
                                            dtype=np.int64,
                                        )
                                        model_map = np.asarray(
                                            plan["maps"][object_index],
                                            dtype=np.int64,
                                        )
                                        input_positions = retained_positions[
                                            model_map
                                        ]
                                        input_ids = np.asarray(
                                            original_ids[
                                                object_index, input_positions
                                            ],
                                            dtype=np.int64,
                                        )
                                        if (
                                            np.unique(retained_ids).size != count
                                            or np.unique(input_ids).size != count
                                            or set(np.unique(input_ids).tolist())
                                            != set(retained_ids.tolist())
                                        ):
                                            raise AssertionError(
                                                "support-preserving consumer lost a retained raw ID"
                                            )
                                        stressed_points.append(
                                            np.asarray(
                                                points[
                                                    object_index,
                                                    input_positions,
                                                ]
                                            )
                                        )
                                        stressed_targets.append(
                                            np.asarray(
                                                targets[
                                                    object_index,
                                                    input_positions,
                                                ]
                                            )
                                        )
                                        stressed_ids.append(input_ids)
                                        retained_positions_rows.append(
                                            retained_positions
                                        )
                                        retained_ids_rows.append(retained_ids)

                                    condition = "%s_%s_rep%d" % (
                                        perturbation,
                                        str(int(round(100.0 * fraction))),
                                        replicate,
                                    )
                                    logits = run_forward(
                                        model,
                                        np.stack(stressed_points),
                                        batch_labels,
                                        _forward_seed(
                                            args.seed, start, condition
                                        ),
                                    )
                                    metrics = metric_arrays[key]
                                    for row, object_index in enumerate(
                                        range(start, stop)
                                    ):
                                        valid_parts = np.asarray(
                                            object_keys[object_index][
                                                "valid_part_ids"
                                            ],
                                            dtype=np.int64,
                                        )
                                        score = shape_part_iou(
                                            logits[row],
                                            stressed_targets[row],
                                            stressed_ids[row],
                                            valid_parts,
                                        )
                                        subset_score = clean_subset_iou(
                                            clean_logits[row],
                                            batch_targets[row],
                                            batch_ids[row],
                                            stressed_ids[row],
                                            valid_parts,
                                        )
                                        damage = subset_score - score
                                        full_gap = (
                                            clean_ious[object_index] - score
                                        )
                                        retained_target = np.asarray(
                                            targets[
                                                object_index,
                                                retained_positions_rows[row],
                                            ],
                                            dtype=np.int64,
                                        )
                                        _, base_first = np.unique(
                                            batch_ids[row], return_index=True
                                        )
                                        base_present = np.unique(
                                            batch_targets[row, base_first]
                                        )
                                        retained_present = np.unique(
                                            retained_target
                                        )
                                        multiplicity = _multiplicity_record(
                                            stressed_ids[row]
                                        )
                                        metrics["instance_miou"][
                                            object_index
                                        ] = score
                                        metrics[
                                            "clean_subset_instance_miou"
                                        ][object_index] = subset_score
                                        metrics["damage"][
                                            object_index
                                        ] = damage
                                        metrics[
                                            "full_clean_minus_stressed"
                                        ][object_index] = full_gap
                                        metrics["unique_raw_id_count"][
                                            object_index
                                        ] = len(retained_ids_rows[row])
                                        metrics["base_present_part_count"][
                                            object_index
                                        ] = len(base_present)
                                        metrics[
                                            "retained_present_part_count"
                                        ][object_index] = len(retained_present)
                                        metrics[
                                            "maximum_input_multiplicity"
                                        ][object_index] = int(
                                            multiplicity["maximum"]
                                        )
                                        pooled_pred, pooled_target = (
                                            deduped_pred_target_over_50(
                                                logits[row],
                                                stressed_targets[row],
                                                stressed_ids[row],
                                            )
                                        )
                                        intersections, unions = (
                                            class_intersection_union_over_50(
                                                pooled_pred, pooled_target
                                            )
                                        )
                                        class_stats[key][
                                            "intersections"
                                        ] += intersections
                                        class_stats[key]["unions"] += unions
                                        trace_rows[row]["cells"].append(
                                            {
                                                "perturbation": perturbation,
                                                "keep_fraction": float(
                                                    fraction
                                                ),
                                                "replicate": int(replicate),
                                                "instance_miou_percent": float(
                                                    100.0 * score
                                                ),
                                                "clean_subset_instance_miou_percent": float(
                                                    100.0 * subset_score
                                                ),
                                                "damage_pp": float(
                                                    100.0 * damage
                                                ),
                                                "full_clean_minus_stressed_pp": float(
                                                    100.0 * full_gap
                                                ),
                                                "retained_unique_raw_id_count": int(
                                                    len(
                                                        retained_ids_rows[
                                                            row
                                                        ]
                                                    )
                                                ),
                                                "retained_part_unique_raw_id_counts": _part_counts(
                                                    retained_target,
                                                    valid_parts,
                                                ),
                                                "retained_present_part_ids": retained_present.tolist(),
                                                "input_multiplicity": multiplicity,
                                                "retained_positions_sha256": array_sha256(
                                                    retained_positions_rows[
                                                        row
                                                    ]
                                                ),
                                                "retained_original_ids_sha256": array_sha256(
                                                    retained_ids_rows[row]
                                                ),
                                                "input_original_ids_sha256": array_sha256(
                                                    stressed_ids[row]
                                                ),
                                                "logits_sha256": array_sha256(
                                                    logits[row]
                                                ),
                                                "audit_class_mean_iou_over_50_percent": float(
                                                    100.0
                                                    * class_mean_iou_from_intersection_union(
                                                        intersections,
                                                        unions,
                                                    )
                                                ),
                                                "audit_class_intersections_50": intersections.tolist(),
                                                "audit_class_unions_50": unions.tolist(),
                                            }
                                        )

                        for trace_row in trace_rows:
                            if len(trace_row["cells"]) != (
                                len(PERTURBATIONS)
                                * len(FRACTIONS)
                                * int(args.replicates)
                            ):
                                raise AssertionError(
                                    "per-object trace has an incomplete condition set"
                                )
                            trace_handle.write(
                                canonical_json_bytes(trace_row)
                            )
                            trace_count += 1
    except Exception:
        if trace_temporary.exists():
            trace_temporary.unlink()
        raise

    if trace_count != num_objects:
        trace_temporary.unlink(missing_ok=True)
        raise AssertionError("R7C per-object trace count mismatch")
    clean_sha_a = clean_digest_a.hexdigest()
    clean_sha_b = clean_digest_b.hexdigest()
    if clean_sha_a != clean_sha_b:
        trace_temporary.unlink(missing_ok=True)
        raise AssertionError("clean repeat aggregate byte hashes differ")
    if not np.all(np.isfinite(clean_ious)):
        trace_temporary.unlink(missing_ok=True)
        raise FloatingPointError("clean per-shape score vector is incomplete")

    cells = []
    for perturbation in PERTURBATIONS:
        for fraction in FRACTIONS:
            replicate_rows = []
            for replicate in range(int(args.replicates)):
                key = _cell_key(perturbation, fraction, replicate)
                metrics = metric_arrays[key]
                for metric_name, values in metrics.items():
                    if metric_name.endswith("count") or metric_name.endswith(
                        "multiplicity"
                    ):
                        if np.any(values < 0):
                            trace_temporary.unlink(missing_ok=True)
                            raise AssertionError(
                                "R7C integer metric is incomplete: %s"
                                % metric_name
                            )
                    elif not np.all(np.isfinite(values)):
                        trace_temporary.unlink(missing_ok=True)
                        raise FloatingPointError(
                            "R7C score metric is incomplete: %s"
                            % metric_name
                        )
                replicate_rows.append(
                    {
                        "replicate": int(replicate),
                        "instance_miou_percent": float(
                            100.0 * metrics["instance_miou"].mean()
                        ),
                        "per_shape_instance_miou_percent": _summary(
                            metrics["instance_miou"], scale=100.0
                        ),
                        "clean_subset_instance_miou_percent": float(
                            100.0
                            * metrics[
                                "clean_subset_instance_miou"
                            ].mean()
                        ),
                        "damage_pp": float(
                            100.0 * metrics["damage"].mean()
                        ),
                        "full_clean_minus_stressed_pp": float(
                            100.0
                            * metrics[
                                "full_clean_minus_stressed"
                            ].mean()
                        ),
                        "audit_class_mean_iou_over_50_percent": float(
                            100.0
                            * class_mean_iou_from_intersection_union(
                                class_stats[key]["intersections"],
                                class_stats[key]["unions"],
                            )
                        ),
                        "unique_raw_id_count_per_shape": _summary(
                            metrics["unique_raw_id_count"]
                        ),
                        "base_present_part_count_per_shape": _summary(
                            metrics["base_present_part_count"]
                        ),
                        "retained_present_part_count_per_shape": _summary(
                            metrics["retained_present_part_count"]
                        ),
                        "all_base_present_parts_retained_shape_fraction": float(
                            np.mean(
                                metrics["base_present_part_count"]
                                == metrics[
                                    "retained_present_part_count"
                                ]
                            )
                        ),
                        "maximum_input_multiplicity_per_shape": _summary(
                            metrics["maximum_input_multiplicity"]
                        ),
                        "num_shapes": int(num_objects),
                    }
                )

            def across(name: str) -> Tuple[float, float]:
                values = np.asarray(
                    [float(row[name]) for row in replicate_rows],
                    dtype=np.float64,
                )
                return float(values.mean()), float(values.std(ddof=1))

            instance_mean, instance_sd = across(
                "instance_miou_percent"
            )
            subset_mean, subset_sd = across(
                "clean_subset_instance_miou_percent"
            )
            damage_mean_raw, damage_sd = across("damage_pp")
            full_gap_mean, full_gap_sd = across(
                "full_clean_minus_stressed_pp"
            )
            class_mean, class_sd = across(
                "audit_class_mean_iou_over_50_percent"
            )
            damage_mean = subset_mean - instance_mean
            if not np.isclose(
                damage_mean, damage_mean_raw, rtol=0.0, atol=1e-10
            ):
                trace_temporary.unlink(missing_ok=True)
                raise AssertionError(
                    "paired replicate damage does not match its components"
                )
            cells.append(
                {
                    "perturbation": perturbation,
                    "keep_fraction": float(fraction),
                    "replicates": replicate_rows,
                    "replicate_aggregate": {
                        "rule": (
                            "mean per-shape category-restricted instance mIoU "
                            "within each replicate, then mean and sample SD "
                            "across the two replicate estimates"
                        ),
                        "num_replicates": int(args.replicates),
                        "instance_miou_percent_mean": instance_mean,
                        "instance_miou_percent_sd": instance_sd,
                        "clean_subset_instance_miou_percent_mean": subset_mean,
                        "clean_subset_instance_miou_percent_sd": subset_sd,
                        "damage_pp_mean": damage_mean,
                        "damage_pp_sd": damage_sd,
                        "full_clean_minus_stressed_pp_mean": full_gap_mean,
                        "full_clean_minus_stressed_pp_sd": full_gap_sd,
                        "audit_class_mean_iou_over_50_percent_mean": class_mean,
                        "audit_class_mean_iou_over_50_percent_sd": class_sd,
                    },
                }
            )

    categories: Dict[str, List[float]] = {}
    for object_index, key_record in enumerate(object_keys):
        categories.setdefault(str(key_record["category"]), []).append(
            float(clean_ious[object_index])
        )
    clean_category_means = {
        category: float(100.0 * np.mean(values))
        for category, values in sorted(categories.items())
    }
    try:
        final_plan_manifest = verify_frozen_plan(
            plan_dir,
            expected_seed=args.seed,
            expected_replicates=args.replicates,
            expected_dataset_content_sha256=str(
                dataset_manifest["content_sha256"]
            ),
            expected_npoint=NPOINT,
            expected_base_sampling=args.base_sampling,
        )
    except Exception:
        trace_temporary.unlink(missing_ok=True)
        raise
    if (
        final_plan_manifest["manifest_sha256"]
        != plan_manifest["manifest_sha256"]
    ):
        trace_temporary.unlink(missing_ok=True)
        raise AssertionError("R7C frozen plan changed during evaluation")
    if (
        file_sha256(paths["checkpoint"])
        != checkpoint_load["checkpoint_sha256"]
    ):
        trace_temporary.unlink(missing_ok=True)
        raise AssertionError("checkpoint changed during R7C evaluation")
    trace_sha256 = file_sha256(trace_temporary)
    code_paths = {
        "driver": Path(__file__).resolve(),
        "r7c_helpers": REPO_ROOT / "rebuttal_eval" / "r7c.py",
        "r7_scorer": REPO_ROOT / "rebuttal_eval" / "r7.py",
        "determinism": REPO_ROOT / "rebuttal_eval" / "determinism.py",
        "segmentation_factory": SCRIPT_DIR / "main.py",
        "segmentation_model": SCRIPT_DIR / "models" / "pt.py",
    }
    code_sha256 = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for name, path in code_paths.items()
    }
    result = {
        "schema": result_schema,
        "model": model_label,
        "estimand": (
            "ShapeNetPart per-shape category-restricted instance mIoU on "
            "unique retained raw IDs for the official PointGPT-S checkpoint"
        ),
        "protocol": {
            "selection_protocol": selection_protocol,
            **(
                {"base_sampling": args.base_sampling}
                if args.base_sampling
                == BASE_SAMPLING_ORIGINAL_REPLACEMENT
                else {}
            ),
            "declaration_status": "newly_declared_pointgpt_only_protocol",
            "base_universe": (
                (
                    "one frozen 2048-point base per object, sampled with "
                    "replacement by one shared numpy.random.RandomState(seed), "
                    "including the original inter-object RNG advancement"
                )
                if args.base_sampling
                == BASE_SAMPLING_ORIGINAL_REPLACEMENT
                else (
                    "one frozen 2048-point base per object, sampled without "
                    "replacement when raw_count >= 2048; short raw clouds "
                    "include every raw ID once before replacement-only padding"
                )
            ),
            "identity_space": "unique raw row IDs within each evaluated object file",
            "fractions": list(FRACTIONS),
            "perturbations": list(PERTURBATIONS),
            "replicates": int(args.replicates),
            "input_map": (
                "support-preserving fixed-2048 map; every retained position "
                "enters at least once"
            ),
            "scorer": (
                "mean duplicate logits by raw ID, restrict predictions to "
                "parts valid for the frozen category, compute per-shape part "
                "mIoU, then average shapes"
            ),
            "class_mean_over_50": (
                "audit-only pooled class mean with argmax over all 50 parts"
            ),
            "damage_pp": (
                "within-cell clean-subset instance mIoU minus stressed "
                "instance mIoU on the same retained raw IDs"
            ),
            "full_clean_minus_stressed_pp": (
                "companion difference across different evaluation populations"
            ),
            "monotonicity": "measured values are reported as-is and are not acceptance-gated",
            "forward_configurations_per_object": 26,
        },
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": str(checkpoint_load["checkpoint_sha256"]),
            "load": dict(checkpoint_load),
        },
        "dataset": dict(dataset_manifest),
        "frozen_plan": {
            "path": str(plan_dir),
            "schema": str(plan_manifest["schema"]),
            "manifest_sha256": str(
                plan_manifest["manifest_sha256"]
            ),
            "dataset_content_sha256": str(
                plan_manifest["dataset_content_sha256"]
            ),
            "semantic_verification_on_load": True,
            "replicate_support_sha256": plan_manifest[
                "replicate_support_sha256"
            ],
        },
        "runtime": {
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": str(torch.version.cuda),
            "cuda_device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "tf32_matmul": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "clean": {
            "instance_miou_percent": float(100.0 * clean_ious.mean()),
            "per_shape_instance_miou_percent": _summary(
                clean_ious, scale=100.0
            ),
            "per_category_instance_miou_percent": clean_category_means,
            "audit_class_mean_iou_over_50_percent": float(
                100.0
                * class_mean_iou_from_intersection_union(
                    clean_class_intersections, clean_class_unions
                )
            ),
            "clean_repeat": {
                "byte_identical": True,
                "first_logits_stream_sha256": clean_sha_a,
                "repeat_logits_stream_sha256": clean_sha_b,
                "same_frozen_inputs_and_batching": True,
            },
            "num_shapes": int(num_objects),
        },
        "cells": cells,
        "artifact_paths": {
            "output_json": str(paths["output_json"]),
            "trace_jsonl_gz": str(paths["trace_jsonl_gz"]),
            "long_csv": str(paths["long_csv"]),
            "wide_csv": str(paths["wide_csv"]),
            "plan_dir": str(paths["plan_dir"]),
        },
        "trace": {
            "schema": trace_schema,
            "num_object_records": int(trace_count),
            "jsonl_gz_sha256": trace_sha256,
            "gzip_mtime": 0,
        },
        "source_code_sha256": code_sha256,
        "superseded_provenance": {
            "recorded_contaminated_source_result": SUPERSEDED_SOURCE_RESULT,
            "preservation": (
                "this driver writes only new R7C paths and never reads, "
                "modifies, or overwrites the superseded artifact"
            ),
            "comparator_rows": (
                "retain the paper's original pipeline and are not modified "
                "by this PointGPT-only correction"
            ),
        },
    }
    stamped = stamp_result_payload(result)
    verify_result_payload(stamped)

    json_temporary = _temp_path(paths["output_json"])
    long_temporary = _temp_path(paths["long_csv"])
    wide_temporary = _temp_path(paths["wide_csv"])
    staged = [
        (trace_temporary, paths["trace_jsonl_gz"]),
        (long_temporary, paths["long_csv"]),
        (wide_temporary, paths["wide_csv"]),
        (json_temporary, paths["output_json"]),
    ]
    try:
        write_canonical_json(json_temporary, stamped)
        write_long_csv_from_json(json_temporary, long_temporary)
        wide_row = write_wide_csv_from_long(
            long_temporary, wide_temporary
        )
        stamped["derived_artifacts"] = {
            "derivation_order": (
                "canonical JSON payload -> long CSV -> R=0.5 wide CSV row"
            ),
            "canonical_json": {
                "path": str(paths["output_json"]),
                "canonical_payload_sha256": stamped[
                    "canonical_payload_sha256"
                ],
                "note": (
                    "the final JSON file hash is printed after commit because "
                    "a file cannot contain its own SHA256"
                ),
            },
            "trace_jsonl_gz": {
                "path": str(paths["trace_jsonl_gz"]),
                "sha256": trace_sha256,
            },
            "long_csv": {
                "path": str(paths["long_csv"]),
                "sha256": file_sha256(long_temporary),
            },
            "wide_csv": {
                "path": str(paths["wide_csv"]),
                "sha256": file_sha256(wide_temporary),
                "row": wide_row,
            },
        }
        verify_result_payload(stamped)
        write_canonical_json(json_temporary, stamped)
        loaded = json.loads(json_temporary.read_text(encoding="utf-8"))
        verify_result_payload(loaded)
        if json_temporary.read_bytes() != canonical_json_bytes(loaded):
            raise AssertionError("R7C result JSON is not canonical")
        if (
            file_sha256(long_temporary)
            != stamped["derived_artifacts"]["long_csv"]["sha256"]
            or file_sha256(wide_temporary)
            != stamped["derived_artifacts"]["wide_csv"]["sha256"]
        ):
            raise AssertionError("derived CSV changed after hashing")
        _commit_without_overwrite(staged)
    except Exception:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
        raise
    return stamped


def main() -> None:
    args = parse_args()
    paths = validate_args(args)
    configure_deterministic_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("R7C official evaluation requires CUDA")

    dataset = PartNormalDataset(
        root=str(paths["root"]),
        npoints=NPOINT,
        split="test",
        normal_channel=False,
    )
    dataset_manifest = build_dataset_manifest(paths["root"], dataset)
    plan_manifest = load_or_materialize_plan(
        paths["plan_dir"],
        dataset,
        dataset_manifest,
        args.seed,
        args.replicates,
        args.base_sampling,
    )
    model, checkpoint_load = build_model_strict(paths["checkpoint"])
    result = evaluate(
        args,
        paths,
        dataset_manifest,
        plan_manifest,
        model,
        checkpoint_load,
    )
    print(
        "R7C COMPLETE output_json=%s payload_sha256=%s file_sha256=%s"
        % (
            paths["output_json"],
            result["canonical_payload_sha256"],
            file_sha256(paths["output_json"]),
        )
    )
    print(
        "R7C ARTIFACTS trace_sha256=%s long_sha256=%s wide_sha256=%s"
        % (
            result["derived_artifacts"]["trace_jsonl_gz"]["sha256"],
            result["derived_artifacts"]["long_csv"]["sha256"],
            result["derived_artifacts"]["wide_csv"]["sha256"],
        )
    )


if __name__ == "__main__":
    main()
