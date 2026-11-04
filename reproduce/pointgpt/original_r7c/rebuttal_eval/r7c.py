"""Pure protocol, frozen-plan, and artifact helpers for the R7C correction.

This module intentionally has no Torch, CUDA, checkpoint, or dataset-loader
dependency. The GPU driver owns model construction and forward execution. This
module owns the declared support operators, unique-raw-ID accounting, frozen
plan materialization, semantic verification on load, and mechanical result
views.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


BASE_SAMPLING_UNIQUE_RAW_ID = "unique_raw_id"
BASE_SAMPLING_ORIGINAL_REPLACEMENT = "original_replacement"
BASE_SAMPLINGS = (
    BASE_SAMPLING_UNIQUE_RAW_ID,
    BASE_SAMPLING_ORIGINAL_REPLACEMENT,
)
FROZEN_PLAN_SCHEMA = "pointgpt_r7c_frozen_plan_v1"
ORIGINAL_REPLACEMENT_FROZEN_PLAN_SCHEMA = (
    "pointgpt_r7c_original_sampling_replay_frozen_plan_v1"
)
RESULT_SCHEMA = "pointgpt_r7c_correction_result_v1"
ORIGINAL_REPLACEMENT_RESULT_SCHEMA = (
    "pointgpt_r7c_original_sampling_replay_v1"
)
FRACTIONS = (0.8, 0.5, 0.2, 0.1)
RANDOM_KEEP = "random_keep"
STRUCTURED_KEEP = "structured_keep"
SEMANTIC_KEEP = "semantic_keep_per_part"
PERTURBATIONS = (RANDOM_KEEP, STRUCTURED_KEEP, SEMANTIC_KEEP)
NPOINT = 2048
MODEL_LABEL = "PointGPT-S official (corrected)"
ORIGINAL_REPLACEMENT_MODEL_LABEL = (
    "PointGPT-S official (corrected, original sampling)"
)

LONG_CSV_COLUMNS = (
    "model",
    "selection_protocol",
    "task",
    "metric_scope",
    "logit_aggregation",
    "perturbation",
    "keep_fraction",
    "condition",
    "instance_miou_percent",
    "clean_subset_instance_miou_percent",
    "damage_pp",
    "source_result",
)
WIDE_CSV_COLUMNS = (
    "model",
    "clean_inst_miou",
    "random50_inst_miou",
    "structured50_inst_miou",
    "semantic50_inst_miou",
)


def validate_base_sampling(base_sampling: str) -> str:
    value = str(base_sampling)
    if value not in BASE_SAMPLINGS:
        raise ValueError(
            "base_sampling must be one of %s" % (BASE_SAMPLINGS,)
        )
    return value


def frozen_plan_schema_for_base_sampling(base_sampling: str) -> str:
    value = validate_base_sampling(base_sampling)
    if value == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
        return ORIGINAL_REPLACEMENT_FROZEN_PLAN_SCHEMA
    return FROZEN_PLAN_SCHEMA


def result_schema_for_base_sampling(base_sampling: str) -> str:
    value = validate_base_sampling(base_sampling)
    if value == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
        return ORIGINAL_REPLACEMENT_RESULT_SCHEMA
    return RESULT_SCHEMA


def model_label_for_base_sampling(base_sampling: str) -> str:
    value = validate_base_sampling(base_sampling)
    if value == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
        return ORIGINAL_REPLACEMENT_MODEL_LABEL
    return MODEL_LABEL


def canonical_json_bytes(value: object) -> bytes:
    """Serialize finite JSON deterministically for hashes and canonical output."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_keep_fraction(keep_fraction: float) -> None:
    """Mirror WhatScoresMiss/src/bds/support.py:8-11."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must satisfy 0 < keep_fraction <= 1")


def keep_count(n_points: int, keep_fraction: float) -> int:
    """Mirror WhatScoresMiss/src/bds/support.py:14-18."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    validate_keep_fraction(keep_fraction)
    return max(1, int(round(n_points * keep_fraction)))


def random_keep_indices(
    n_points: int, keep_fraction: float, seed: int
) -> np.ndarray:
    """Mirror WhatScoresMiss/src/bds/support.py:21-25 exactly."""
    keep_n = keep_count(n_points, keep_fraction)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=keep_n, replace=False))


def structured_keep_by_anchor(
    xyz: np.ndarray, keep_fraction: float, seed: int
) -> np.ndarray:
    """Mirror WhatScoresMiss/src/bds/support.py:58-69 exactly."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if xyz.shape[0] == 0:
        raise ValueError("xyz must contain at least one point")
    keep_n = keep_count(len(xyz), keep_fraction)
    rng = np.random.default_rng(seed)
    anchor = xyz[rng.integers(0, len(xyz))]
    dist = np.linalg.norm(xyz - anchor, axis=1)
    return np.sort(np.argsort(dist)[-keep_n:])


def semantic_keep_indices(
    labels: np.ndarray, keep_fraction: float, seed: int
) -> np.ndarray:
    """Mirror WhatScoresMiss/src/bds/support.py:72-84 exactly."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0:
        raise ValueError("labels must contain at least one point")
    validate_keep_fraction(keep_fraction)
    rng = np.random.default_rng(seed)
    kept: List[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        keep_n = max(1, int(round(len(idx) * keep_fraction)))
        kept.extend(rng.choice(idx, size=keep_n, replace=False).tolist())
    return np.sort(np.asarray(kept, dtype=int))


def keep_indices(
    xyz: np.ndarray,
    labels: np.ndarray,
    perturbation: str,
    keep_fraction: float,
    seed: int,
) -> np.ndarray:
    """Mirror reproduce/support/apply_object_support.py:13-22 exactly."""
    if perturbation == "random_keep":
        return random_keep_indices(len(xyz), keep_fraction, seed)
    if perturbation == "structured_keep":
        return structured_keep_by_anchor(xyz, keep_fraction, seed)
    if perturbation == "semantic_keep":
        return semantic_keep_indices(labels, keep_fraction, seed)
    if perturbation == "clean":
        return np.arange(len(xyz))
    raise ValueError("unknown perturbation: %s" % perturbation)


def _validate_fractions(fractions: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in fractions)
    if values != FRACTIONS:
        raise ValueError("R7C fractions must be exactly %s" % (FRACTIONS,))
    if len(values) != len(set(values)):
        raise ValueError("R7C fractions must be unique")
    for value in values:
        validate_keep_fraction(value)
    return values


def _seed32(base_seed: int, namespace: str, *indices: int) -> int:
    if int(base_seed) < 0:
        raise ValueError("seed must be non-negative")
    payload = [int(base_seed), str(namespace)] + [int(value) for value in indices]
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def base_selection_seed(seed: int, object_index: int, raw_count: int, npoint: int) -> int:
    return _seed32(seed, "r7c_base_selection", object_index, raw_count, npoint)


def operator_seed(
    seed: int, object_index: int, replicate: int, perturbation: str
) -> int:
    if perturbation not in PERTURBATIONS:
        raise ValueError("unknown R7C perturbation: %s" % perturbation)
    return _seed32(
        seed,
        "r7c_operator",
        object_index,
        replicate,
        PERTURBATIONS.index(perturbation),
    )


def input_map_seed(
    seed: int,
    object_index: int,
    replicate: int,
    perturbation: str,
    fraction: float,
) -> int:
    if perturbation not in PERTURBATIONS:
        raise ValueError("unknown R7C perturbation: %s" % perturbation)
    return _seed32(
        seed,
        "r7c_support_preserving_input_map",
        object_index,
        replicate,
        PERTURBATIONS.index(perturbation),
        int(round(float(fraction) * 100)),
    )


def frozen_base_indices(
    raw_count: int, npoint: int, seed: int, object_index: int
) -> Tuple[np.ndarray, str, int]:
    """Freeze the base with unique sampling, padding only short raw clouds.

    Raw clouds with at least ``npoint`` rows are sampled without replacement.
    Short raw clouds include every raw ID once, then use replacement only for
    the padding positions. The padding never enlarges the unique-ID universe.
    """
    raw_count = int(raw_count)
    npoint = int(npoint)
    if raw_count <= 0 or npoint <= 0:
        raise ValueError("raw_count and npoint must be positive")
    derived_seed = base_selection_seed(seed, object_index, raw_count, npoint)
    rng = np.random.default_rng(derived_seed)
    if raw_count >= npoint:
        indices = rng.choice(raw_count, size=npoint, replace=False)
        mode = "raw_without_replacement"
    else:
        extras = rng.choice(raw_count, size=npoint - raw_count, replace=True)
        combined = np.concatenate(
            [np.arange(raw_count, dtype=np.int64), extras.astype(np.int64)]
        )
        indices = combined[rng.permutation(npoint)]
        mode = "all_unique_then_replacement_padding"
    return np.asarray(indices, dtype=np.int64), mode, derived_seed


def original_replacement_base_indices(
    raw_count: int,
    npoint: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the submitted PartNormalDataset base RNG stream exactly.

    This mirrors the shared ``RandomState`` loop in
    ``segmentation/eval_shapenetpart_r7_resampling.py:545-566``: draw
    ``npoint`` raw rows with replacement, then consume the per-object
    ``permutation(npoint)`` draw before sampling the next object. The sampling
    operation itself mirrors ``segmentation/dataset.py:154-159``.
    """
    raw_count = int(raw_count)
    npoint = int(npoint)
    if raw_count <= 0 or npoint <= 0:
        raise ValueError("raw_count and npoint must be positive")
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("original replay requires numpy.random.RandomState")
    indices = rng.choice(raw_count, npoint, replace=True)
    legacy_priority = rng.permutation(npoint)
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(legacy_priority, dtype=np.int64),
    )


def support_preserving_input_map(
    retained_size: int, npoint: int, seed: int
) -> np.ndarray:
    """Mirror rebuttal_eval/r7.py:79-87 for SUPPORT_PRESERVING inputs."""
    retained_size = int(retained_size)
    npoint = int(npoint)
    if retained_size <= 0 or npoint <= 0:
        raise ValueError("retained_size and npoint must be positive")
    if retained_size > npoint:
        raise ValueError("R7C retained support cannot exceed the model input size")
    rng = np.random.RandomState(int(seed))
    if retained_size >= npoint:
        return rng.choice(retained_size, npoint, replace=False).astype(
            np.int64, copy=False
        )
    extras = rng.choice(retained_size, npoint - retained_size, replace=True)
    combined = np.concatenate(
        [np.arange(retained_size, dtype=np.int64), extras]
    )
    return combined[rng.permutation(npoint)].astype(np.int64, copy=False)


def _semantic_expected_count(labels: np.ndarray, fraction: float) -> int:
    labels = np.asarray(labels)
    return int(
        sum(
            max(1, int(round(int(np.sum(labels == label)) * float(fraction))))
            for label in np.unique(labels)
        )
    )


def expected_support_count(
    labels: np.ndarray, perturbation: str, fraction: float
) -> int:
    if perturbation == SEMANTIC_KEEP:
        return _semantic_expected_count(labels, fraction)
    if perturbation in (RANDOM_KEEP, STRUCTURED_KEEP):
        return keep_count(int(np.asarray(labels).size), fraction)
    raise ValueError("unknown R7C perturbation: %s" % perturbation)


def nested_operator_supports(
    xyz: np.ndarray,
    labels: np.ndarray,
    perturbation: str,
    fractions: Sequence[float],
    seed: int,
) -> Dict[float, np.ndarray]:
    """Couple published operator marginals with one nested priority.

    Random support uses one uniform no-replacement ordering. Semantic support
    uses one independent no-replacement ordering within each present part and
    the published per-part rounding rule. These couplings preserve the
    published operator distribution at every fraction while making the
    required cross-fraction nesting explicit. Structured support calls the
    exact published anchor operator at each fraction with one shared seed.
    """
    values = _validate_fractions(fractions)
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    if xyz.ndim != 2 or xyz.shape != (labels.size, 3) or labels.size == 0:
        raise ValueError("xyz and labels must have shapes (N, 3) and (N,)")

    result: Dict[float, np.ndarray] = {}
    if perturbation == RANDOM_KEEP:
        rng = np.random.default_rng(seed)
        priority = rng.choice(labels.size, size=labels.size, replace=False)
        for fraction in values:
            result[fraction] = np.sort(
                priority[: keep_count(labels.size, fraction)]
            ).astype(np.int64)
    elif perturbation == STRUCTURED_KEEP:
        for fraction in values:
            result[fraction] = structured_keep_by_anchor(xyz, fraction, seed).astype(
                np.int64
            )
    elif perturbation == SEMANTIC_KEEP:
        rng = np.random.default_rng(seed)
        priorities: Dict[int, np.ndarray] = {}
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            priorities[int(label)] = rng.choice(
                idx, size=idx.size, replace=False
            ).astype(np.int64)
        for fraction in values:
            kept: List[int] = []
            for label in np.unique(labels):
                idx = priorities[int(label)]
                count = max(1, int(round(idx.size * fraction)))
                kept.extend(idx[:count].tolist())
            result[fraction] = np.sort(np.asarray(kept, dtype=np.int64))
    else:
        raise ValueError("unknown R7C perturbation: %s" % perturbation)

    ordered = sorted(values, key=lambda value: (result[value].size, value))
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(result[smaller].tolist()).issubset(
            set(result[larger].tolist())
        ):
            raise AssertionError("operator supports are not nested")
    return result


def unique_raw_universe(
    points: np.ndarray, targets: np.ndarray, original_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted unique raw IDs and their canonical base representatives."""
    points = np.asarray(points)
    targets = np.asarray(targets)
    original_ids = np.asarray(original_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        raise ValueError("points must have non-empty shape (N, channels>=3)")
    if targets.shape != (points.shape[0],) or original_ids.shape != (
        points.shape[0],
    ):
        raise ValueError("targets and original_ids must have shape (N,)")
    unique_ids, first_positions = np.unique(original_ids, return_index=True)
    order = np.argsort(unique_ids, kind="stable")
    unique_ids = unique_ids[order]
    first_positions = first_positions[order].astype(np.int64)
    for raw_id, first_position in zip(unique_ids, first_positions):
        positions = np.flatnonzero(original_ids == raw_id)
        if not np.all(targets[positions] == targets[first_position]):
            raise AssertionError("duplicate raw ID has inconsistent labels")
        if not np.array_equal(
            points[positions], np.repeat(points[first_position][None, :], positions.size, axis=0)
        ):
            raise AssertionError("duplicate raw ID has inconsistent point content")
    return (
        unique_ids.astype(np.int64),
        first_positions,
        np.asarray(points[first_positions, :3], dtype=np.float64),
        np.asarray(targets[first_positions]),
    )


def fraction_suffix(fraction: float) -> str:
    value = float(fraction)
    validate_keep_fraction(value)
    return str(int(round(100 * value)))


def retained_positions_name(
    perturbation: str, fraction: float, replicate: int
) -> str:
    return "retained_positions_%s_%s_rep%d" % (
        perturbation,
        fraction_suffix(fraction),
        int(replicate),
    )


def retained_original_ids_name(
    perturbation: str, fraction: float, replicate: int
) -> str:
    return "retained_original_ids_%s_%s_rep%d" % (
        perturbation,
        fraction_suffix(fraction),
        int(replicate),
    )


def retained_counts_name(
    perturbation: str, fraction: float, replicate: int
) -> str:
    return "retained_counts_%s_%s_rep%d" % (
        perturbation,
        fraction_suffix(fraction),
        int(replicate),
    )


def input_map_name(perturbation: str, fraction: float, replicate: int) -> str:
    return "support_preserving_input_map_%s_%s_rep%d" % (
        perturbation,
        fraction_suffix(fraction),
        int(replicate),
    )


def input_map_seeds_name(
    perturbation: str, fraction: float, replicate: int
) -> str:
    return "input_map_seeds_%s_%s_rep%d" % (
        perturbation,
        fraction_suffix(fraction),
        int(replicate),
    )


def operator_seeds_name(perturbation: str, replicate: int) -> str:
    return "operator_seeds_%s_rep%d" % (perturbation, int(replicate))


def _save_array(
    directory: Path,
    name: str,
    value: np.ndarray,
    files: MutableMapping[str, str],
) -> None:
    path = directory / (name + ".npy")
    np.save(path, value, allow_pickle=False)
    files[path.name] = file_sha256(path)


def _save_json(
    directory: Path,
    filename: str,
    value: object,
    files: MutableMapping[str, str],
) -> None:
    path = directory / filename
    path.write_bytes(canonical_json_bytes(value))
    files[path.name] = file_sha256(path)


def load_plan_array(directory: Path, name: str) -> np.ndarray:
    path = Path(directory) / (name + ".npy")
    if not path.is_file():
        raise FileNotFoundError("missing frozen-plan array: %s" % path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _validate_base_arrays(
    points: np.ndarray,
    targets: np.ndarray,
    original_ids: np.ndarray,
    class_labels: np.ndarray,
    object_keys: Sequence[Mapping[str, object]],
    seed: int,
    base_sampling: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    base_sampling = validate_base_sampling(base_sampling)
    points = np.asarray(points)
    targets = np.asarray(targets)
    original_ids = np.asarray(original_ids)
    class_labels = np.asarray(class_labels)
    if points.ndim != 3 or points.shape[2] < 3:
        raise ValueError("base_points must have shape (objects, npoint, channels>=3)")
    num_objects, npoint = points.shape[:2]
    if targets.shape != (num_objects, npoint):
        raise ValueError("base_targets shape mismatch")
    if original_ids.shape != (num_objects, npoint):
        raise ValueError("base_original_ids shape mismatch")
    if class_labels.shape == (num_objects,):
        class_labels = class_labels[:, None]
    if class_labels.shape != (num_objects, 1):
        raise ValueError("class_labels must have shape (objects, 1)")
    if len(object_keys) != num_objects:
        raise ValueError("object_keys length mismatch")

    replay_rng = (
        np.random.RandomState(int(seed))
        if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT
        else None
    )
    universes = []
    for object_index in range(num_objects):
        key = object_keys[object_index]
        if int(key.get("object_index", -1)) != object_index:
            raise AssertionError("object key index mismatch")
        raw_count = int(key.get("raw_count", 0))
        if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT:
            expected_ids, legacy_priority = (
                original_replacement_base_indices(
                    raw_count,
                    npoint,
                    replay_rng,
                )
            )
            expected_mode = (
                "partnormal_2048_with_replacement_shared_randomstate"
            )
            expected_seed = int(seed)
            if key.get("legacy_priority_sha256") != array_sha256(
                legacy_priority
            ):
                raise AssertionError(
                    "original-replay legacy priority RNG accounting mismatch"
                )
        else:
            expected_ids, expected_mode, expected_seed = frozen_base_indices(
                raw_count, npoint, seed, object_index
            )
        actual_ids = np.asarray(original_ids[object_index], dtype=np.int64)
        if not np.array_equal(actual_ids, expected_ids):
            raise AssertionError(
                "frozen base original IDs do not match the declared selection seed"
            )
        if key.get("base_selection_mode") != expected_mode:
            raise AssertionError("base selection mode mismatch")
        if int(key.get("base_selection_seed", -1)) != expected_seed:
            raise AssertionError("base selection seed mismatch")
        if actual_ids.min() < 0 or actual_ids.max() >= raw_count:
            raise AssertionError("base original ID is outside raw-file bounds")
        class_label = int(np.asarray(class_labels[object_index]).reshape(-1)[0])
        if int(key.get("class_label", -1)) != class_label:
            raise AssertionError("frozen category class label mismatch")
        valid_parts = np.asarray(key.get("valid_part_ids", []), dtype=np.int64)
        if valid_parts.ndim != 1 or valid_parts.size == 0:
            raise AssertionError("object key has no valid category parts")
        invalid_targets = np.setdiff1d(
            np.unique(np.asarray(targets[object_index], dtype=np.int64)),
            valid_parts,
        )
        if invalid_targets.size:
            raise AssertionError(
                "base target label is invalid for frozen category binding"
            )
        universe = unique_raw_universe(
            points[object_index],
            targets[object_index],
            actual_ids,
        )
        if int(key.get("base_unique_raw_id_count", -1)) != universe[0].size:
            raise AssertionError("base unique raw-ID count record mismatch")
        expected_unique = (
            np.unique(expected_ids).size
            if base_sampling == BASE_SAMPLING_ORIGINAL_REPLACEMENT
            else min(raw_count, npoint)
        )
        if universe[0].size != expected_unique:
            raise AssertionError("frozen base unique raw-ID count is not fail-closed")
        universes.append(universe)
    return universes


def materialize_frozen_plan(
    directory: Path,
    points: np.ndarray,
    targets: np.ndarray,
    original_ids: np.ndarray,
    class_labels: np.ndarray,
    object_keys: Sequence[Mapping[str, object]],
    dataset_manifest: Mapping[str, object],
    seed: int,
    replicates: int,
    fractions: Sequence[float] = FRACTIONS,
    base_sampling: str = BASE_SAMPLING_UNIQUE_RAW_ID,
) -> Dict[str, object]:
    """Freeze the complete standalone R7C plan under a new schema tag."""
    base_sampling = validate_base_sampling(base_sampling)
    values = _validate_fractions(fractions)
    if int(replicates) != 2:
        raise ValueError("R7C is locked to exactly two perturbation replicates")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise FileExistsError("R7C frozen plan already exists: %s" % directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "refusing to overwrite non-empty R7C plan directory: %s" % directory
        )

    points = np.asarray(points)
    targets = np.asarray(targets)
    original_ids = np.asarray(original_ids)
    class_labels = np.asarray(class_labels)
    if class_labels.ndim == 1:
        class_labels = class_labels[:, None]
    universes = _validate_base_arrays(
        points,
        targets,
        original_ids,
        class_labels,
        object_keys,
        seed,
        base_sampling,
    )
    num_objects, npoint = points.shape[:2]
    if dataset_manifest.get("content_sha256") is None:
        raise ValueError("dataset manifest is missing content_sha256")

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=directory.name + ".materializing.", dir=directory.parent)
    )
    files: Dict[str, str] = {}
    support_digests: Dict[str, List[str]] = {
        perturbation: [] for perturbation in PERTURBATIONS
    }
    condition_count_ranges: Dict[str, Dict[str, int]] = {}
    try:
        _save_array(temporary, "base_points", points, files)
        _save_array(temporary, "base_targets", targets, files)
        _save_array(temporary, "base_original_ids", original_ids, files)
        _save_array(temporary, "class_labels", class_labels, files)
        _save_array(
            temporary,
            "base_unique_raw_id_counts",
            np.asarray([universe[0].size for universe in universes], dtype=np.int32),
            files,
        )
        _save_json(temporary, "object_keys.json", list(object_keys), files)
        _save_json(temporary, "dataset_manifest.json", dict(dataset_manifest), files)

        for replicate in range(int(replicates)):
            for perturbation in PERTURBATIONS:
                seeds = np.asarray(
                    [
                        operator_seed(seed, object_index, replicate, perturbation)
                        for object_index in range(num_objects)
                    ],
                    dtype=np.uint32,
                )
                _save_array(
                    temporary,
                    operator_seeds_name(perturbation, replicate),
                    seeds,
                    files,
                )
                supports_by_object: List[Dict[float, np.ndarray]] = []
                for object_index, universe in enumerate(universes):
                    _, first_positions, xyz, labels = universe
                    support_positions = nested_operator_supports(
                        xyz,
                        labels,
                        perturbation,
                        values,
                        int(seeds[object_index]),
                    )
                    supports_by_object.append(
                        {
                            fraction: first_positions[support_positions[fraction]].astype(
                                np.int64
                            )
                            for fraction in values
                        }
                    )

                replicate_digest = hashlib.sha256()
                for fraction in values:
                    rows = [
                        np.asarray(support[fraction], dtype=np.int64)
                        for support in supports_by_object
                    ]
                    counts = np.asarray([row.size for row in rows], dtype=np.int32)
                    max_count = int(counts.max())
                    positions = np.full(
                        (num_objects, max_count), -1, dtype=np.int32
                    )
                    retained_ids = np.full_like(positions, -1)
                    maps = np.empty((num_objects, npoint), dtype=np.int32)
                    map_seeds = np.empty(num_objects, dtype=np.uint32)
                    for object_index, row in enumerate(rows):
                        count = int(row.size)
                        positions[object_index, :count] = row.astype(np.int32)
                        retained_ids[object_index, :count] = np.asarray(
                            original_ids[object_index, row], dtype=np.int32
                        )
                        derived_map_seed = input_map_seed(
                            seed,
                            object_index,
                            replicate,
                            perturbation,
                            fraction,
                        )
                        map_seeds[object_index] = derived_map_seed
                        maps[object_index] = support_preserving_input_map(
                            count, npoint, derived_map_seed
                        ).astype(np.int32)
                        replicate_digest.update(
                            np.asarray([object_index, count], dtype=np.int64).tobytes()
                        )
                        replicate_digest.update(
                            np.ascontiguousarray(
                                retained_ids[object_index, :count]
                            ).tobytes()
                        )
                    _save_array(
                        temporary,
                        retained_positions_name(
                            perturbation, fraction, replicate
                        ),
                        positions,
                        files,
                    )
                    _save_array(
                        temporary,
                        retained_original_ids_name(
                            perturbation, fraction, replicate
                        ),
                        retained_ids,
                        files,
                    )
                    _save_array(
                        temporary,
                        retained_counts_name(
                            perturbation, fraction, replicate
                        ),
                        counts,
                        files,
                    )
                    _save_array(
                        temporary,
                        input_map_name(perturbation, fraction, replicate),
                        maps,
                        files,
                    )
                    _save_array(
                        temporary,
                        input_map_seeds_name(
                            perturbation, fraction, replicate
                        ),
                        map_seeds,
                        files,
                    )
                    condition_key = "%s_%s_rep%d" % (
                        perturbation,
                        fraction_suffix(fraction),
                        replicate,
                    )
                    condition_count_ranges[condition_key] = {
                        "min_unique_raw_ids": int(counts.min()),
                        "max_unique_raw_ids": int(counts.max()),
                    }
                support_digests[perturbation].append(replicate_digest.hexdigest())

        short_objects = [
            {
                "object_index": int(key["object_index"]),
                "relative_path": str(key["relative_path"]),
                "raw_count": int(key["raw_count"]),
                "base_unique_raw_id_count": int(key["base_unique_raw_id_count"]),
                "base_selection_mode": str(key["base_selection_mode"]),
            }
            for key in object_keys
            if int(key["raw_count"]) < int(npoint)
        ]
        manifest: Dict[str, object] = {
            "schema": frozen_plan_schema_for_base_sampling(base_sampling),
            **(
                {"base_sampling": base_sampling}
                if base_sampling
                == BASE_SAMPLING_ORIGINAL_REPLACEMENT
                else {}
            ),
            "seed": int(seed),
            "npoint": int(npoint),
            "num_objects": int(num_objects),
            "replicates": int(replicates),
            "fractions": list(values),
            "perturbations": list(PERTURBATIONS),
            "dataset_content_sha256": str(dataset_manifest["content_sha256"]),
            "base_selection_rule": (
                (
                    "sample npoint raw rows with replacement from one shared "
                    "numpy.random.RandomState(seed), then consume "
                    "permutation(npoint) before the next object, mirroring "
                    "segmentation/eval_shapenetpart_r7_resampling.py:545-566"
                )
                if base_sampling
                == BASE_SAMPLING_ORIGINAL_REPLACEMENT
                else (
                    "sample raw rows without replacement when raw_count >= "
                    "npoint; otherwise include every unique raw ID once and "
                    "replacement-pad only the fixed-size frozen base"
                )
            ),
            "identity_space": "unique raw row IDs within each evaluated object file",
            "input_map": "support_preserving fixed-npoint map from rebuttal_eval/r7.py:79-87",
            "operator_sources": {
                RANDOM_KEEP: "WhatScoresMiss/src/bds/support.py:21-25",
                STRUCTURED_KEEP: "WhatScoresMiss/src/bds/support.py:58-69",
                SEMANTIC_KEEP: "WhatScoresMiss/src/bds/support.py:72-84",
                "dispatch": (
                    "WhatScoresMiss/reproduce/support/"
                    "apply_object_support.py:13-22"
                ),
            },
            "nesting_rule": (
                "one per-object per-replicate operator priority couples fractions; "
                "each fraction preserves the published operator marginal"
            ),
            "semantic_count_rule": (
                "independent per-present-part round(part_count*fraction), minimum one"
            ),
            "random_structured_count_rule": (
                "round(unique_raw_id_count*fraction), minimum one"
            ),
            "short_raw_objects": short_objects,
            "condition_count_ranges": condition_count_ranges,
            "replicate_support_sha256": support_digests,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        verify_frozen_plan(
            temporary,
            expected_seed=seed,
            expected_replicates=replicates,
            expected_dataset_content_sha256=str(
                dataset_manifest["content_sha256"]
            ),
            expected_npoint=npoint,
            expected_base_sampling=base_sampling,
        )
        if directory.exists():
            directory.rmdir()
        os.replace(str(temporary), str(directory))
        return verify_frozen_plan(
            directory,
            expected_seed=seed,
            expected_replicates=replicates,
            expected_dataset_content_sha256=str(
                dataset_manifest["content_sha256"]
            ),
            expected_npoint=npoint,
            expected_base_sampling=base_sampling,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_manifest(directory: Path) -> Dict[str, object]:
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("missing R7C frozen-plan manifest: %s" % path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base_sampling = validate_base_sampling(
        manifest.get("base_sampling", BASE_SAMPLING_UNIQUE_RAW_ID)
    )
    if manifest.get("schema") != frozen_plan_schema_for_base_sampling(
        base_sampling
    ):
        raise AssertionError("unexpected R7C frozen-plan schema")
    stored = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if stored != canonical_json_sha256(unhashed):
        raise AssertionError("R7C frozen-plan manifest hash mismatch")
    declared = set(manifest.get("files", {}))
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != declared:
        raise AssertionError(
            "R7C frozen-plan file set mismatch: missing=%s unexpected=%s"
            % (sorted(declared - actual), sorted(actual - declared))
        )
    for filename, expected_hash in manifest["files"].items():
        if file_sha256(directory / filename) != expected_hash:
            raise AssertionError(
                "R7C frozen-plan artifact hash mismatch: %s" % filename
            )
    return manifest


def _assert_semantic_invariants(
    labels: np.ndarray, support_indices: np.ndarray, fraction: float
) -> None:
    labels = np.asarray(labels)
    support_indices = np.asarray(support_indices, dtype=np.int64)
    kept_labels = labels[support_indices]
    for label in np.unique(labels):
        expected = max(1, int(round(int(np.sum(labels == label)) * fraction)))
        actual = int(np.sum(kept_labels == label))
        if actual != expected:
            raise AssertionError(
                "semantic per-part count or minimum-one invariant failed"
            )


def verify_frozen_plan(
    directory: Path,
    expected_seed: Optional[int] = None,
    expected_replicates: Optional[int] = None,
    expected_dataset_content_sha256: Optional[str] = None,
    expected_npoint: Optional[int] = None,
    expected_base_sampling: Optional[str] = None,
) -> Dict[str, object]:
    """Hash and semantically verify every loaded R7C plan artifact.

    Verification rechecks bounds, unique raw-ID counts, label and ID
    consistency, category binding, nested supports, operator reproduction,
    support-preserving input coverage, and all derived seeds.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    base_sampling = validate_base_sampling(
        manifest.get("base_sampling", BASE_SAMPLING_UNIQUE_RAW_ID)
    )
    values = _validate_fractions(manifest.get("fractions", []))
    if tuple(manifest.get("perturbations", [])) != PERTURBATIONS:
        raise AssertionError("R7C perturbation list mismatch")
    seed = int(manifest["seed"])
    replicates = int(manifest["replicates"])
    npoint = int(manifest["npoint"])
    num_objects = int(manifest["num_objects"])
    if replicates != 2:
        raise AssertionError("R7C plan is not locked to two replicates")
    if expected_seed is not None and seed != int(expected_seed):
        raise AssertionError("R7C plan seed does not match the requested seed")
    if expected_replicates is not None and replicates != int(expected_replicates):
        raise AssertionError("R7C plan replicate count mismatch")
    if expected_npoint is not None and npoint != int(expected_npoint):
        raise AssertionError("R7C plan npoint mismatch")
    if (
        expected_base_sampling is not None
        and base_sampling != validate_base_sampling(expected_base_sampling)
    ):
        raise AssertionError("R7C plan base sampling mismatch")
    if (
        expected_dataset_content_sha256 is not None
        and manifest.get("dataset_content_sha256")
        != expected_dataset_content_sha256
    ):
        raise AssertionError("R7C plan dataset content hash mismatch")

    points = load_plan_array(directory, "base_points")
    targets = load_plan_array(directory, "base_targets")
    original_ids = load_plan_array(directory, "base_original_ids")
    class_labels = load_plan_array(directory, "class_labels")
    recorded_unique_counts = load_plan_array(
        directory, "base_unique_raw_id_counts"
    )
    object_keys = json.loads(
        (directory / "object_keys.json").read_text(encoding="utf-8")
    )
    dataset_manifest = json.loads(
        (directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("content_sha256") != manifest.get(
        "dataset_content_sha256"
    ):
        raise AssertionError("stored dataset manifest content hash mismatch")
    universes = _validate_base_arrays(
        points,
        targets,
        original_ids,
        class_labels,
        object_keys,
        seed,
        base_sampling,
    )
    if points.shape[:2] != (num_objects, npoint):
        raise AssertionError("R7C base array shape differs from manifest")
    expected_unique_counts = np.asarray(
        [universe[0].size for universe in universes], dtype=np.int64
    )
    if not np.array_equal(
        np.asarray(recorded_unique_counts, dtype=np.int64),
        expected_unique_counts,
    ):
        raise AssertionError("base unique raw-ID count array mismatch")
    expected_short = [
        int(key["object_index"])
        for key in object_keys
        if int(key["raw_count"]) < npoint
    ]
    recorded_short = [
        int(row["object_index"]) for row in manifest.get("short_raw_objects", [])
    ]
    if expected_short != recorded_short:
        raise AssertionError("short raw-object count records mismatch")

    computed_support_digests: Dict[str, List[str]] = {
        perturbation: [] for perturbation in PERTURBATIONS
    }
    for replicate in range(replicates):
        for perturbation in PERTURBATIONS:
            stored_operator_seeds = np.asarray(
                load_plan_array(
                    directory, operator_seeds_name(perturbation, replicate)
                ),
                dtype=np.uint32,
            )
            expected_operator_seeds = np.asarray(
                [
                    operator_seed(seed, object_index, replicate, perturbation)
                    for object_index in range(num_objects)
                ],
                dtype=np.uint32,
            )
            if stored_operator_seeds.shape != (num_objects,) or not np.array_equal(
                stored_operator_seeds, expected_operator_seeds
            ):
                raise AssertionError("operator seed array mismatch")

            expected_supports: List[Dict[float, np.ndarray]] = []
            canonical_position_sets = []
            for object_index, universe in enumerate(universes):
                _, first_positions, xyz, labels = universe
                local_supports = nested_operator_supports(
                    xyz,
                    labels,
                    perturbation,
                    values,
                    int(stored_operator_seeds[object_index]),
                )
                expected_supports.append(
                    {
                        fraction: first_positions[local_supports[fraction]].astype(
                            np.int64
                        )
                        for fraction in values
                    }
                )
                canonical_position_sets.append(set(first_positions.tolist()))

            loaded: Dict[float, Dict[str, np.ndarray]] = {}
            replicate_digest = hashlib.sha256()
            for fraction in values:
                positions = np.asarray(
                    load_plan_array(
                        directory,
                        retained_positions_name(
                            perturbation, fraction, replicate
                        ),
                    )
                )
                retained_ids = np.asarray(
                    load_plan_array(
                        directory,
                        retained_original_ids_name(
                            perturbation, fraction, replicate
                        ),
                    )
                )
                counts = np.asarray(
                    load_plan_array(
                        directory,
                        retained_counts_name(
                            perturbation, fraction, replicate
                        ),
                    ),
                    dtype=np.int64,
                )
                maps = np.asarray(
                    load_plan_array(
                        directory,
                        input_map_name(perturbation, fraction, replicate),
                    )
                )
                map_seeds = np.asarray(
                    load_plan_array(
                        directory,
                        input_map_seeds_name(
                            perturbation, fraction, replicate
                        ),
                    ),
                    dtype=np.uint32,
                )
                if counts.shape != (num_objects,):
                    raise AssertionError("retained count array shape mismatch")
                if positions.ndim != 2 or positions.shape[0] != num_objects:
                    raise AssertionError("retained position array shape mismatch")
                if retained_ids.shape != positions.shape:
                    raise AssertionError("retained ID array shape mismatch")
                if maps.shape != (num_objects, npoint):
                    raise AssertionError("input map array shape mismatch")
                if map_seeds.shape != (num_objects,):
                    raise AssertionError("input map seed array shape mismatch")
                expected_max_count = max(
                    expected_support_count(
                        universe[3], perturbation, fraction
                    )
                    for universe in universes
                )
                if positions.shape[1] != expected_max_count:
                    raise AssertionError("retained padded width count mismatch")

                for object_index, universe in enumerate(universes):
                    unique_ids, first_positions, _, labels = universe
                    count = int(counts[object_index])
                    expected_count = expected_support_count(
                        labels, perturbation, fraction
                    )
                    if count != expected_count:
                        raise AssertionError(
                            "retained unique-raw-ID count mismatch"
                        )
                    if count <= 0 or count > positions.shape[1]:
                        raise AssertionError("retained count is outside padded bounds")
                    active_positions = np.asarray(
                        positions[object_index, :count], dtype=np.int64
                    )
                    stored_active_ids = np.asarray(
                        retained_ids[object_index, :count]
                    )
                    active_ids = np.asarray(stored_active_ids, dtype=np.int64)
                    if np.any(positions[object_index, count:] != -1) or np.any(
                        retained_ids[object_index, count:] != -1
                    ):
                        raise AssertionError("retained padded tail is not sentinel-filled")
                    if active_positions.min() < 0 or active_positions.max() >= npoint:
                        raise AssertionError("retained position is out of bounds")
                    if np.unique(active_positions).size != count:
                        raise AssertionError("retained positions are not unique")
                    if not set(active_positions.tolist()).issubset(
                        canonical_position_sets[object_index]
                    ):
                        raise AssertionError(
                            "retained position is not a unique raw-ID representative"
                        )
                    expected_ids = np.asarray(
                        original_ids[object_index, active_positions], dtype=np.int64
                    )
                    if not np.array_equal(active_ids, expected_ids):
                        raise AssertionError(
                            "retained label/ID consistency check failed"
                        )
                    if np.unique(active_ids).size != count:
                        raise AssertionError(
                            "duplicated raw IDs were double-counted in support"
                        )
                    if not set(active_ids.tolist()).issubset(set(unique_ids.tolist())):
                        raise AssertionError("retained raw ID is outside the base universe")
                    position_to_local = {
                        int(position): local_index
                        for local_index, position in enumerate(first_positions)
                    }
                    local_indices = np.asarray(
                        [
                            position_to_local[int(position)]
                            for position in active_positions
                        ],
                        dtype=np.int64,
                    )
                    if perturbation == SEMANTIC_KEEP:
                        _assert_semantic_invariants(
                            labels, local_indices, fraction
                        )

                    expected_map_seed = input_map_seed(
                        seed,
                        object_index,
                        replicate,
                        perturbation,
                        fraction,
                    )
                    if int(map_seeds[object_index]) != expected_map_seed:
                        raise AssertionError("support-preserving input map seed mismatch")
                    active_map = np.asarray(maps[object_index], dtype=np.int64)
                    if active_map.min() < 0 or active_map.max() >= count:
                        raise AssertionError(
                            "support-preserving input map is out of bounds"
                        )
                    coverage = np.bincount(active_map, minlength=count)
                    if coverage.shape != (count,) or np.any(coverage < 1):
                        raise AssertionError(
                            "support-preserving input map coverage failed"
                        )
                    expected_map = support_preserving_input_map(
                        count, npoint, expected_map_seed
                    )
                    if not np.array_equal(active_map, expected_map):
                        raise AssertionError(
                            "support-preserving input map reproduction failed"
                        )
                    replicate_digest.update(
                        np.asarray([object_index, count], dtype=np.int64).tobytes()
                    )
                    replicate_digest.update(
                        np.ascontiguousarray(stored_active_ids).tobytes()
                    )
                loaded[fraction] = {
                    "positions": positions,
                    "counts": counts,
                }

            for object_index in range(num_objects):
                object_order = sorted(
                    values,
                    key=lambda fraction: (
                        int(loaded[fraction]["counts"][object_index]),
                        fraction,
                    ),
                )
                for smaller, larger in zip(object_order, object_order[1:]):
                    small_count = int(
                        loaded[smaller]["counts"][object_index]
                    )
                    large_count = int(
                        loaded[larger]["counts"][object_index]
                    )
                    small = set(
                        np.asarray(
                            loaded[smaller]["positions"][
                                object_index, :small_count
                            ],
                            dtype=np.int64,
                        ).tolist()
                    )
                    large = set(
                        np.asarray(
                            loaded[larger]["positions"][
                                object_index, :large_count
                            ],
                            dtype=np.int64,
                        ).tolist()
                    )
                    if not small.issubset(large):
                        raise AssertionError(
                            "retained support nesting invariant failed"
                        )

            for fraction in values:
                for object_index in range(num_objects):
                    count = int(loaded[fraction]["counts"][object_index])
                    actual = np.asarray(
                        loaded[fraction]["positions"][
                            object_index, :count
                        ],
                        dtype=np.int64,
                    )
                    if not np.array_equal(
                        actual, expected_supports[object_index][fraction]
                    ):
                        raise AssertionError(
                            "stored support does not reproduce the declared operator"
                        )
            computed_support_digests[perturbation].append(
                replicate_digest.hexdigest()
            )

    if manifest.get("replicate_support_sha256") != computed_support_digests:
        raise AssertionError("replicate support digest mismatch")
    return manifest


def result_payload_sha256(result: Mapping[str, object]) -> str:
    payload = dict(result)
    payload.pop("canonical_payload_sha256", None)
    payload.pop("derived_artifacts", None)
    return canonical_json_sha256(payload)


def stamp_result_payload(result: Mapping[str, object]) -> Dict[str, object]:
    stamped = dict(result)
    stamped["canonical_payload_sha256"] = result_payload_sha256(stamped)
    return stamped


def result_base_sampling(result: Mapping[str, object]) -> str:
    protocol = result.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise TypeError("R7C result protocol is not a mapping")
    return validate_base_sampling(
        protocol.get("base_sampling", BASE_SAMPLING_UNIQUE_RAW_ID)
    )


def verify_result_payload(result: Mapping[str, object]) -> None:
    base_sampling = result_base_sampling(result)
    if result.get("schema") != result_schema_for_base_sampling(
        base_sampling
    ):
        raise AssertionError("unexpected R7C result schema")
    stored = result.get("canonical_payload_sha256")
    if stored != result_payload_sha256(result):
        raise AssertionError("R7C canonical result payload hash mismatch")


def write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    Path(path).write_bytes(canonical_json_bytes(value))


def _condition_name(perturbation: str, fraction: float) -> str:
    suffix = fraction_suffix(fraction)
    if perturbation == SEMANTIC_KEEP:
        return "part_keep%s_per_part" % suffix
    return "%s%s" % (perturbation, suffix)


def long_rows_from_result(
    result: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Derive the paper-compatible long curve rows from canonical JSON."""
    verify_result_payload(result)
    artifact_paths = result.get("artifact_paths", {})
    source_result = str(artifact_paths.get("output_json", ""))
    if not source_result:
        raise ValueError("canonical result is missing output_json artifact path")
    protocol = result.get("protocol", {})
    base_sampling = result_base_sampling(result)
    model = str(result.get("model", ""))
    expected_model = model_label_for_base_sampling(base_sampling)
    if model != expected_model:
        raise AssertionError("unexpected corrected model label")
    shared = {
        "model": model,
        "selection_protocol": str(protocol["selection_protocol"]),
        "task": "ShapeNetPart",
        "metric_scope": "unique_retained_raw_ids",
        "logit_aggregation": "mean_by_original_index",
        "source_result": source_result,
    }
    clean = result["clean"]
    clean_value = float(clean["instance_miou_percent"])
    rows: List[Dict[str, object]] = [
        {
            **shared,
            "perturbation": "clean",
            "keep_fraction": 1.0,
            "condition": "clean",
            "instance_miou_percent": clean_value,
            "clean_subset_instance_miou_percent": clean_value,
            "damage_pp": 0.0,
        }
    ]
    cells = {
        (str(cell["perturbation"]), float(cell["keep_fraction"])): cell
        for cell in result["cells"]
    }
    if len(cells) != len(PERTURBATIONS) * len(FRACTIONS):
        raise AssertionError("canonical result has missing or duplicate cells")
    for perturbation in PERTURBATIONS:
        for fraction in FRACTIONS:
            cell = cells[(perturbation, fraction)]
            aggregate = cell["replicate_aggregate"]
            value = float(aggregate["instance_miou_percent_mean"])
            clean_subset = float(
                aggregate["clean_subset_instance_miou_percent_mean"]
            )
            damage = float(aggregate["damage_pp_mean"])
            if not np.isclose(
                damage,
                clean_subset - value,
                rtol=0.0,
                atol=1e-10,
            ):
                raise AssertionError("aggregate damage is not mechanically paired")
            rows.append(
                {
                    **shared,
                    "perturbation": perturbation,
                    "keep_fraction": fraction,
                    "condition": _condition_name(perturbation, fraction),
                    "instance_miou_percent": value,
                    "clean_subset_instance_miou_percent": clean_subset,
                    "damage_pp": damage,
                }
            )
    return rows


def write_long_csv_from_json(json_path: Path, csv_path: Path) -> List[Dict[str, object]]:
    result = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = long_rows_from_result(result)
    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LONG_CSV_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def derive_wide_row_from_long_csv(long_csv: Path) -> Dict[str, str]:
    """Derive the R=0.5 paper row, rounding only at the wide-table boundary."""
    with Path(long_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0].keys()) != LONG_CSV_COLUMNS:
        raise AssertionError("long CSV columns do not match the declared schema")
    models = {row["model"] for row in rows}
    if len(models) != 1 or next(iter(models)) not in {
        MODEL_LABEL,
        ORIGINAL_REPLACEMENT_MODEL_LABEL,
    }:
        raise AssertionError("long CSV must contain exactly the corrected model")
    model = next(iter(models))

    def one(perturbation: str, fraction: float) -> float:
        matches = [
            row
            for row in rows
            if row["perturbation"] == perturbation
            and float(row["keep_fraction"]) == float(fraction)
        ]
        if len(matches) != 1:
            raise AssertionError(
                "long CSV does not have exactly one required wide-row cell"
            )
        return float(matches[0]["instance_miou_percent"])

    return {
        "model": model,
        "clean_inst_miou": format(one("clean", 1.0), ".2f"),
        "random50_inst_miou": format(one(RANDOM_KEEP, 0.5), ".2f"),
        "structured50_inst_miou": format(one(STRUCTURED_KEEP, 0.5), ".2f"),
        "semantic50_inst_miou": format(one(SEMANTIC_KEEP, 0.5), ".2f"),
    }


def write_wide_csv_from_long(long_csv: Path, wide_csv: Path) -> Dict[str, str]:
    row = derive_wide_row_from_long_csv(long_csv)
    with Path(wide_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(WIDE_CSV_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerow(row)
    return row
