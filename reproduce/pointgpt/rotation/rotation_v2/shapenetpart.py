"""Frozen official-style ShapeNetPart sampling and metrics for v2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rotation_v2.frozen_bank import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    logical_array_sha256,
)


SAMPLE_PLAN_SCHEMA = "rotation_matrix_v2_shapenetpart_sample_plan_v1"
DATASET_MANIFEST_SCHEMA = "rotation_matrix_v2_shapenetpart_dataset_manifest_v1"
NPOINT = 2048
NUM_CATEGORIES = 16
NUM_PARTS = 50
OFFICIAL_COMMIT = "7445a68068d707da6d959b66773bfef84c4e3e32"
OFFICIAL_DATASET_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    + OFFICIAL_COMMIT
    + "/segmentation/dataset.py#L66-L164"
)
OFFICIAL_EVALUATOR_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    + OFFICIAL_COMMIT
    + "/segmentation/main.py#L217-L253"
)
LOCAL_EVALUATOR_MIRROR = "segmentation/main.py:302-344"

CATEGORY_NAMES = (
    "Airplane",
    "Bag",
    "Cap",
    "Car",
    "Chair",
    "Earphone",
    "Guitar",
    "Knife",
    "Lamp",
    "Laptop",
    "Motorbike",
    "Mug",
    "Pistol",
    "Rocket",
    "Skateboard",
    "Table",
)
CATEGORY_SYNSETS = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243",
}
SEG_CLASSES = {
    "Airplane": (0, 1, 2, 3),
    "Bag": (4, 5),
    "Cap": (6, 7),
    "Car": (8, 9, 10, 11),
    "Chair": (12, 13, 14, 15),
    "Earphone": (16, 17, 18),
    "Guitar": (19, 20, 21),
    "Knife": (22, 23),
    "Lamp": (24, 25, 26, 27),
    "Laptop": (28, 29),
    "Motorbike": (30, 31, 32, 33, 34, 35),
    "Mug": (36, 37),
    "Pistol": (38, 39, 40),
    "Rocket": (41, 42, 43),
    "Skateboard": (44, 45, 46),
    "Table": (47, 48, 49),
}
PLAN_ARTIFACTS = (
    "points.npy",
    "targets.npy",
    "sampled_indices.npy",
    "class_labels.npy",
    "raw_counts.npy",
    "example_ids.json",
    "shape_records.json",
    "category_part_mapping.json",
    "dataset_manifest.json",
)


@dataclass(frozen=True)
class ShapeRecord:
    example_id: str
    category: str
    class_index: int
    synset: str
    shape_id: str
    relative_path: str
    path: Path


@dataclass(frozen=True)
class FrozenShapeNetPartPlan:
    directory: Path
    manifest: Mapping[str, object]
    example_ids: tuple[str, ...]
    shape_records: tuple[Mapping[str, object], ...]
    points: np.ndarray
    targets: np.ndarray
    sampled_indices: np.ndarray
    class_labels: np.ndarray
    raw_counts: np.ndarray


def category_part_mapping() -> dict[str, object]:
    return {
        "category_order": list(CATEGORY_NAMES),
        "category_to_class_index": {
            category: index for index, category in enumerate(CATEGORY_NAMES)
        },
        "category_to_synset": dict(CATEGORY_SYNSETS),
        "category_to_valid_parts": {
            category: list(SEG_CLASSES[category]) for category in CATEGORY_NAMES
        },
        "num_categories": NUM_CATEGORIES,
        "num_parts": NUM_PARTS,
    }


def normalize_xyz_official(points: np.ndarray) -> np.ndarray:
    """Mirror Point-MAE ``pc_normalize`` on float32 XYZ before sampling."""
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 3:
        raise ValueError("raw XYZ must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("raw XYZ contains a non-finite value")
    centroid = np.mean(value, axis=0)
    centered = value - centroid
    radius = np.max(np.sqrt(np.sum(centered**2, axis=1)))
    if not np.isfinite(radius) or float(radius) <= 0.0:
        raise ValueError("ShapeNetPart shape has zero normalization radius")
    normalized = np.asarray(centered / radius, dtype=np.float32)
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("normalized XYZ contains a non-finite value")
    return normalized


def official_sample_indices(
    raw_counts: Sequence[int],
    *,
    seed: int,
    npoint: int = NPOINT,
) -> np.ndarray:
    """Replay one single-process sequence of official replacement sampling."""
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("sampling seed must be a non-negative integer")
    counts = np.asarray(raw_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("raw_counts must be a non-empty positive vector")
    if int(npoint) != NPOINT:
        raise ValueError("v2 ShapeNetPart sampling is fixed to 2048 points")
    rng = np.random.RandomState(int(seed))
    result = np.empty((counts.size, NPOINT), dtype=np.int64)
    for index, count in enumerate(counts):
        result[index] = rng.choice(int(count), NPOINT, replace=True)
    return result


def _safe_dataset_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("dataset file resolves outside its root") from error
    if not candidate.is_file():
        raise FileNotFoundError("dataset source is not a file: %s" % candidate)
    return candidate


def _parse_category_file(root: Path) -> tuple[dict[str, str], Path]:
    path = _safe_dataset_file(root, "synsetoffset2category.txt")
    parsed: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = line.strip().split()
        if len(fields) != 2:
            raise ValueError("invalid category file line %d" % line_number)
        category, synset = fields
        if category in parsed:
            raise ValueError("duplicate ShapeNetPart category")
        parsed[category] = synset
    if tuple(parsed) != CATEGORY_NAMES or parsed != CATEGORY_SYNSETS:
        raise ValueError("ShapeNetPart category order or synset mapping changed")
    return parsed, path


def enumerate_official_test_shapes(
    dataset_root: Path,
) -> tuple[tuple[ShapeRecord, ...], dict[str, object]]:
    """Mirror the official category-order and sorted-filename test enumeration."""
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("ShapeNetPart root is not a directory")
    categories, category_path = _parse_category_file(root)
    split_path = _safe_dataset_file(
        root,
        "train_test_split/shuffled_test_file_list.json",
    )
    try:
        split_entries = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("ShapeNetPart test split JSON is invalid") from error
    if not isinstance(split_entries, list) or not split_entries:
        raise ValueError("ShapeNetPart test split is empty")
    test_ids: set[str] = set()
    for entry in split_entries:
        if not isinstance(entry, str):
            raise TypeError("ShapeNetPart test split entry is not a string")
        fields = entry.split("/")
        if len(fields) < 3 or not fields[1] or not fields[2]:
            raise ValueError("ShapeNetPart test split entry has unknown layout")
        # Category-qualified ID: the official test list contains one uuid that
        # appears under two synsets (702cebffa33a19f019f079d1b712f46f in both
        # 04379243 and 03001627), so bare uuids are NOT unique across categories.
        test_ids.add("%s/%s" % (fields[1], fields[2]))
    if len(test_ids) != len(split_entries):
        raise ValueError("ShapeNetPart test split contains duplicate shape IDs")

    records: list[ShapeRecord] = []
    matched_ids: set[str] = set()
    for class_index, category in enumerate(categories):
        synset = categories[category]
        category_directory = (root / synset).resolve(strict=True)
        try:
            category_directory.relative_to(root)
        except ValueError as error:
            raise ValueError("category directory resolves outside dataset root") from error
        if not category_directory.is_dir():
            raise FileNotFoundError("missing category directory: %s" % synset)
        for filename in sorted(os.listdir(category_directory)):
            shape_id = filename[:-4]
            if "%s/%s" % (synset, shape_id) not in test_ids:
                continue
            path = _safe_dataset_file(root, "%s/%s" % (synset, filename))
            example_id = "%s/%s" % (synset, shape_id)
            records.append(
                ShapeRecord(
                    example_id=example_id,
                    category=category,
                    class_index=class_index,
                    synset=synset,
                    shape_id=shape_id,
                    relative_path=path.relative_to(root).as_posix(),
                    path=path,
                )
            )
            matched_ids.add("%s/%s" % (synset, shape_id))
    missing_ids = sorted(test_ids - matched_ids)
    if missing_ids:
        raise FileNotFoundError(
            "test split references absent shape files: %s" % missing_ids[:10]
        )
    if len(records) != len(test_ids):
        raise ValueError("test shape IDs are not unique across category directories")

    source_files: list[dict[str, object]] = []
    for path in (category_path, split_path):
        source_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "role": "metadata",
            }
        )
    for record in records:
        source_files.append(
            {
                "path": record.relative_path,
                "bytes": int(record.path.stat().st_size),
                "sha256": file_sha256(record.path),
                "role": "test_shape",
            }
        )
    dataset_manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "official_dataset_reference": OFFICIAL_DATASET_URL,
        "split": "test",
        "normal_channel": False,
        "npoint": NPOINT,
        "num_shapes": len(records),
        "enumeration": (
            "category_file_insertion_order_then_sorted_os_listdir_filtered_"
            "by_shuffled_test_shape_id"
        ),
        "source_files": source_files,
        "source_file_set_sha256": canonical_json_sha256(source_files),
        "test_split_entry_sha256": canonical_json_sha256(split_entries),
        "category_part_mapping": category_part_mapping(),
        "category_part_mapping_sha256": canonical_json_sha256(
            category_part_mapping()
        ),
        "example_ids": [record.example_id for record in records],
    }
    dataset_manifest["manifest_sha256"] = canonical_json_sha256(dataset_manifest)
    return tuple(records), dataset_manifest


def _array_artifact(path: Path, value: np.ndarray) -> dict[str, object]:
    return {
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha256(path),
        "logical_sha256": logical_array_sha256(value),
        "dtype": np.asarray(value).dtype.str,
        "shape": [int(size) for size in np.asarray(value).shape],
    }


def _json_artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "logical_sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "canonical-json-utf8",
    }


def _shape_record_json(records: Sequence[ShapeRecord]) -> list[dict[str, object]]:
    return [
        {
            "example_id": record.example_id,
            "category": record.category,
            "class_index": record.class_index,
            "synset": record.synset,
            "shape_id": record.shape_id,
            "relative_path": record.relative_path,
        }
        for record in records
    ]


def freeze_shapenetpart_sample_plan(
    dataset_root: Path,
    directory: Path,
    *,
    seed: int,
) -> FrozenShapeNetPartPlan:
    """Create the one shared official-style 2048-point test sample."""
    root = Path(dataset_root).resolve(strict=True)
    target = Path(directory).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite sample plan: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    records, dataset_manifest = enumerate_official_test_shapes(root)

    raw_points: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    raw_counts = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        before_hash = file_sha256(record.path)
        data = np.loadtxt(record.path).astype(np.float32)
        if file_sha256(record.path) != before_hash:
            raise RuntimeError("dataset shape changed while being read")
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
            raise ValueError("ShapeNetPart source has invalid shape")
        xyz = normalize_xyz_official(data[:, 0:3])
        target_values = data[:, -1]
        if not np.all(np.equal(target_values, np.floor(target_values))):
            raise ValueError("ShapeNetPart target contains a non-integer label")
        target_labels = target_values.astype(np.int64)
        valid_parts = np.asarray(SEG_CLASSES[record.category], dtype=np.int64)
        if not np.all(np.isin(target_labels, valid_parts)):
            raise ValueError("ShapeNetPart target is invalid for its category")
        raw_points.append(xyz)
        raw_targets.append(target_labels)
        raw_counts[index] = xyz.shape[0]
    sampled_indices = official_sample_indices(raw_counts, seed=seed)
    points = np.empty((len(records), NPOINT, 3), dtype=np.float32)
    targets = np.empty((len(records), NPOINT), dtype=np.int64)
    for index in range(len(records)):
        points[index] = raw_points[index][sampled_indices[index]]
        targets[index] = raw_targets[index][sampled_indices[index]]
    class_labels = np.asarray(
        [record.class_index for record in records],
        dtype=np.int64,
    )
    example_ids = [record.example_id for record in records]
    shape_records = _shape_record_json(records)
    mapping = category_part_mapping()

    staging = Path(
        tempfile.mkdtemp(prefix=target.name + ".partial.", dir=target.parent)
    )
    try:
        arrays = {
            "points.npy": points,
            "targets.npy": targets,
            "sampled_indices.npy": sampled_indices,
            "class_labels.npy": class_labels,
            "raw_counts.npy": raw_counts,
        }
        for filename, value in arrays.items():
            np.save(staging / filename, value, allow_pickle=False)
        json_values = {
            "example_ids.json": example_ids,
            "shape_records.json": shape_records,
            "category_part_mapping.json": mapping,
            "dataset_manifest.json": dataset_manifest,
        }
        for filename, value in json_values.items():
            (staging / filename).write_bytes(canonical_json_bytes(value))
        artifacts = {
            filename: _array_artifact(staging / filename, value)
            for filename, value in arrays.items()
        }
        artifacts.update(
            {
                filename: _json_artifact(staging / filename)
                for filename in json_values
            }
        )
        source_path = Path(__file__).resolve()
        manifest: dict[str, object] = {
            "schema": SAMPLE_PLAN_SCHEMA,
            "dataset_root_at_freeze": str(root),
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "num_shapes": len(records),
            "npoint": NPOINT,
            "normal_channel": False,
            "sampling": {
                "operation": "numpy.random.RandomState.choice",
                "replace": True,
                "single_process_official_test_order": True,
                "generated_once": True,
                "seed": int(seed),
            },
            "shared_scope": "all_rotations_and_all_models_in_task",
            "artifacts": artifacts,
            "generator_source": {
                "path": "rotation_v2/shapenetpart.py",
                "sha256": file_sha256(source_path),
            },
            "official_dataset_reference": OFFICIAL_DATASET_URL,
            "official_evaluator_reference": OFFICIAL_EVALUATOR_URL,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_shapenetpart_sample_plan(target, dataset_root=root)


def _verify_manifest_self_hash(manifest: Mapping[str, object]) -> None:
    recorded = manifest.get("manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256", None)
    if recorded != canonical_json_sha256(without_hash):
        raise ValueError("sample plan manifest self-hash mismatch")


def load_shapenetpart_sample_plan(
    directory: Path,
    *,
    dataset_root: Path,
    expected_manifest_sha256: str | None = None,
) -> FrozenShapeNetPartPlan:
    """Verify every sample artifact and every current dataset source byte."""
    plan_root = Path(directory).resolve(strict=True)
    if not plan_root.is_dir():
        raise NotADirectoryError("sample plan path is not a directory")
    expected_files = set(PLAN_ARTIFACTS) | {"manifest.json"}
    actual_files = {
        path.name for path in plan_root.iterdir() if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("sample plan file set mismatch")
    if any(path.is_dir() or path.is_symlink() for path in plan_root.iterdir()):
        raise ValueError("sample plan contains a directory or symlink")
    manifest_raw = (plan_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SAMPLE_PLAN_SCHEMA:
        raise ValueError("sample plan manifest schema mismatch")
    if canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("sample plan manifest is not canonical JSON")
    _verify_manifest_self_hash(manifest)
    if (
        expected_manifest_sha256 is not None
        and manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("sample plan manifest differs from preregistration")
    if (
        manifest.get("npoint") != NPOINT
        or manifest.get("normal_channel") is not False
    ):
        raise ValueError("sample plan input schema differs from contract")
    generator_source = manifest.get("generator_source")
    if (
        not isinstance(generator_source, Mapping)
        or generator_source.get("path") != "rotation_v2/shapenetpart.py"
        or generator_source.get("sha256") != file_sha256(Path(__file__).resolve())
    ):
        raise ValueError("sample plan generator source hash changed")

    arrays: dict[str, np.ndarray] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(PLAN_ARTIFACTS):
        raise ValueError("sample plan artifact manifest mismatch")
    for filename in PLAN_ARTIFACTS:
        path = plan_root / filename
        record = artifacts[filename]
        if not isinstance(record, Mapping):
            raise ValueError("sample plan artifact record is invalid")
        if file_sha256(path) != record.get("file_sha256"):
            raise ValueError("sample plan artifact bytes changed: %s" % filename)
        if filename.endswith(".npy"):
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if logical_array_sha256(value) != record.get("logical_sha256"):
                raise ValueError("sample plan array content changed: %s" % filename)
            if value.dtype.str != record.get("dtype") or list(value.shape) != record.get(
                "shape"
            ):
                raise ValueError("sample plan array schema changed: %s" % filename)
            arrays[filename] = value
        else:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
            if raw != canonical_json_bytes(value):
                raise ValueError("sample plan JSON is not canonical: %s" % filename)

    current_records, current_dataset_manifest = enumerate_official_test_shapes(
        dataset_root
    )
    stored_dataset_raw = (plan_root / "dataset_manifest.json").read_bytes()
    if stored_dataset_raw != canonical_json_bytes(current_dataset_manifest):
        raise ValueError("ShapeNetPart source files or test metadata changed")
    if (
        current_dataset_manifest["manifest_sha256"]
        != manifest.get("dataset_manifest_sha256")
    ):
        raise ValueError("dataset manifest hash differs from sample plan")

    example_ids_value = json.loads(
        (plan_root / "example_ids.json").read_text(encoding="utf-8")
    )
    shape_records_value = json.loads(
        (plan_root / "shape_records.json").read_text(encoding="utf-8")
    )
    mapping_value = json.loads(
        (plan_root / "category_part_mapping.json").read_text(encoding="utf-8")
    )
    expected_record_json = _shape_record_json(current_records)
    if example_ids_value != [record.example_id for record in current_records]:
        raise ValueError("sample plan example IDs differ from official test order")
    if shape_records_value != expected_record_json:
        raise ValueError("sample plan shape records changed")
    if mapping_value != category_part_mapping():
        raise ValueError("sample plan category/part mapping changed")

    num_shapes = len(current_records)
    expected_shapes = {
        "points.npy": (num_shapes, NPOINT, 3),
        "targets.npy": (num_shapes, NPOINT),
        "sampled_indices.npy": (num_shapes, NPOINT),
        "class_labels.npy": (num_shapes,),
        "raw_counts.npy": (num_shapes,),
    }
    for filename, shape in expected_shapes.items():
        if arrays[filename].shape != shape:
            raise ValueError("sample plan array has unexpected shape: %s" % filename)
    if arrays["points.npy"].dtype != np.float32:
        raise TypeError("sampled points must use float32")
    for filename in (
        "targets.npy",
        "sampled_indices.npy",
        "class_labels.npy",
        "raw_counts.npy",
    ):
        if arrays[filename].dtype != np.int64:
            raise TypeError("sample plan integer array must use int64")
    if not np.all(np.isfinite(arrays["points.npy"])):
        raise FloatingPointError("sampled points contain a non-finite value")
    if np.any(arrays["sampled_indices.npy"] < 0) or np.any(
        arrays["sampled_indices.npy"] >= arrays["raw_counts.npy"][:, None]
    ):
        raise ValueError("sampled raw index is out of range")
    replay = official_sample_indices(
        arrays["raw_counts.npy"],
        seed=int(manifest["sampling"]["seed"]),
    )
    if not np.array_equal(replay, arrays["sampled_indices.npy"]):
        raise ValueError("sample indices do not replay from the declared seed")
    for value in arrays.values():
        value.flags.writeable = False
    return FrozenShapeNetPartPlan(
        directory=plan_root,
        manifest=manifest,
        example_ids=tuple(str(value) for value in example_ids_value),
        shape_records=tuple(shape_records_value),
        points=arrays["points.npy"],
        targets=arrays["targets.npy"],
        sampled_indices=arrays["sampled_indices.npy"],
        class_labels=arrays["class_labels.npy"],
        raw_counts=arrays["raw_counts.npy"],
    )


def category_restricted_predictions(
    logits: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Mirror official argmax restricted to each shape's valid part set."""
    values = np.asarray(logits)
    labels = np.asarray(class_labels, dtype=np.int64)
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[-1] != NUM_PARTS
        or labels.shape != (values.shape[0],)
    ):
        raise ValueError("logits or category labels have invalid shape")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("segmentation logits contain a non-finite value")
    predictions = np.empty(values.shape[:2], dtype=np.int64)
    for shape_index, class_index in enumerate(labels):
        if int(class_index) < 0 or int(class_index) >= NUM_CATEGORIES:
            raise ValueError("category label is outside [0, 15]")
        category = CATEGORY_NAMES[int(class_index)]
        parts = np.asarray(SEG_CLASSES[category], dtype=np.int64)
        predictions[shape_index] = parts[
            np.argmax(values[shape_index][:, parts], axis=1)
        ]
    return predictions


def official_shape_instance_ious(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Mirror official per-part absent-both=1 and per-shape part mean."""
    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    labels = np.asarray(class_labels, dtype=np.int64)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predictions and targets must share shape (S, N)")
    if labels.shape != (predicted.shape[0],):
        raise ValueError("category labels must have shape (S,)")
    result = np.empty(predicted.shape[0], dtype=np.float64)
    for shape_index, class_index in enumerate(labels):
        if int(class_index) < 0 or int(class_index) >= NUM_CATEGORIES:
            raise ValueError("category label is outside [0, 15]")
        category = CATEGORY_NAMES[int(class_index)]
        valid_parts = SEG_CLASSES[category]
        if not np.all(np.isin(target[shape_index], valid_parts)):
            raise ValueError("target contains a category-invalid part")
        if not np.all(np.isin(predicted[shape_index], valid_parts)):
            raise ValueError("prediction contains a category-invalid part")
        part_ious: list[float] = []
        for part in valid_parts:
            pred_mask = predicted[shape_index] == part
            target_mask = target[shape_index] == part
            if not np.any(pred_mask) and not np.any(target_mask):
                part_ious.append(1.0)
            else:
                intersection = np.sum(pred_mask & target_mask)
                union = np.sum(pred_mask | target_mask)
                part_ious.append(float(intersection / union))
        result[shape_index] = float(np.mean(part_ious))
    return result


def segmentation_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_labels: np.ndarray,
) -> dict[str, object]:
    """Return official instance/class mIoU and complete per-shape values."""
    predictions = category_restricted_predictions(logits, class_labels)
    result = segmentation_metrics_from_predictions(
        predictions,
        targets,
        class_labels,
    )
    result["prediction_sha256"] = logical_array_sha256(predictions)
    return result


def segmentation_metrics_from_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_labels: np.ndarray,
) -> dict[str, object]:
    """Compute official metrics from already category-restricted predictions."""
    predictions = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    labels = np.asarray(class_labels, dtype=np.int64)
    shape_ious = official_shape_instance_ious(predictions, target, labels)
    per_category: dict[str, float] = {}
    for class_index, category in enumerate(CATEGORY_NAMES):
        mask = labels == class_index
        if np.any(mask):
            per_category[category] = float(np.mean(shape_ious[mask]))
    confusion = np.zeros((NUM_PARTS, NUM_PARTS), dtype=np.int64)
    np.add.at(confusion, (target.reshape(-1), predictions.reshape(-1)), 1)
    return {
        "accuracy": float(np.mean(predictions == target)),
        "instance_miou": float(np.mean(shape_ious)),
        "class_avg_iou": float(np.mean(list(per_category.values()))),
        "per_category_iou": per_category,
        "per_shape_instance_iou": shape_ious.tolist(),
        "confusion_50x50": confusion.tolist(),
        "metric_contract": {
            "primary": "category_valid_per_shape_instance_miou",
            "absent_prediction_and_target_part_iou": 1.0,
            "official_evaluator": OFFICIAL_EVALUATOR_URL,
            "local_mirror": LOCAL_EVALUATOR_MIRROR,
        },
    }


def paired_shape_bootstrap_ci(
    candidate_shape_values: np.ndarray,
    clean_shape_values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    scale: float = 100.0,
) -> dict[str, object]:
    """Paired shape-level percentile CI for score and clean-relative delta."""
    candidate = np.asarray(candidate_shape_values, dtype=np.float64)
    clean = np.asarray(clean_shape_values, dtype=np.float64)
    if (
        candidate.ndim != 1
        or candidate.size == 0
        or candidate.shape != clean.shape
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(clean))
    ):
        raise ValueError("bootstrap inputs must be paired finite shape vectors")
    if isinstance(replicates, bool) or int(replicates) < 2:
        raise ValueError("bootstrap replicates must be at least 2")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    score_draws = np.empty(int(replicates), dtype=np.float64)
    delta_draws = np.empty(int(replicates), dtype=np.float64)
    offset = 0
    while offset < int(replicates):
        count = min(256, int(replicates) - offset)
        indices = rng.integers(
            0,
            candidate.size,
            size=(count, candidate.size),
            endpoint=False,
        )
        sampled_candidate = candidate[indices].mean(axis=1)
        sampled_clean = clean[indices].mean(axis=1)
        score_draws[offset : offset + count] = sampled_candidate
        delta_draws[offset : offset + count] = sampled_candidate - sampled_clean
        offset += count
    lower, upper = (2.5, 97.5)
    return {
        "unit": "shape",
        "method": "paired_percentile_cluster_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "scale": float(scale),
        "score": {
            "estimate": float(candidate.mean() * scale),
            "ci95": [
                float(np.percentile(score_draws, lower) * scale),
                float(np.percentile(score_draws, upper) * scale),
            ],
        },
        "clean_relative_delta": {
            "estimate": float((candidate - clean).mean() * scale),
            "ci95": [
                float(np.percentile(delta_draws, lower) * scale),
                float(np.percentile(delta_draws, upper) * scale),
            ],
        },
    }
