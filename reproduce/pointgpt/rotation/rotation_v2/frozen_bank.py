"""Frozen, order-independent rotation bank for rotation_matrix_v2.

The stored convention is an active rotation matrix ``R`` with row-vector
application ``x_rotated = x @ R.T``. Quaternions use ``[w, x, y, z]``.
Identity is a separate control condition. It is not a member of the yaw
aggregate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "rotation_matrix_v2_frozen_bank_v1"
YAW_DEGREES = (45, 90, 135, 180, 225, 270, 315)
SO3_REPEATS = 8
FLOAT_DTYPE = np.dtype("<f8")
ORTHONORMAL_ATOL = 2.0e-12
DETERMINANT_ATOL = 2.0e-12
QUATERNION_ATOL = 2.0e-12
ARTIFACT_FILENAMES = (
    "matrices.npy",
    "quaternions.npy",
    "example_ids.json",
    "condition_order.json",
)
SOURCE_FILENAMES = (
    "rotation_v2/frozen_bank.py",
    "rotation_v2/generate_frozen_bank.py",
)


def canonical_json_bytes(value: object) -> bytes:
    """Return the one accepted canonical JSON encoding."""
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


def _framed_update(
    digest: "hashlib._Hash",
    label: str,
    payload: bytes | memoryview,
) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(16, "big"))
    digest.update(payload)


def _array_header_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    return canonical_json_bytes(
        {
            "dtype": array.dtype.str,
            "shape": [int(size) for size in array.shape],
        }
    )


def logical_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    _framed_update(digest, "header", _array_header_bytes(array))
    _framed_update(digest, "data", memoryview(array).cast("B"))
    return digest.hexdigest()


def _combined_content_sha256(
    matrices: np.ndarray,
    quaternions: np.ndarray,
    example_id_bytes: bytes,
    condition_bytes: bytes,
) -> str:
    matrices_c = np.ascontiguousarray(matrices)
    quaternions_c = np.ascontiguousarray(quaternions)
    digest = hashlib.sha256()
    _framed_update(digest, "matrices_header", _array_header_bytes(matrices_c))
    _framed_update(
        digest,
        "matrices_data",
        memoryview(matrices_c).cast("B"),
    )
    _framed_update(
        digest,
        "quaternions_header",
        _array_header_bytes(quaternions_c),
    )
    _framed_update(
        digest,
        "quaternions_data",
        memoryview(quaternions_c).cast("B"),
    )
    _framed_update(digest, "example_ids", example_id_bytes)
    _framed_update(digest, "condition_order", condition_bytes)
    return digest.hexdigest()


def canonicalize_example_ids(example_ids: Iterable[str | int]) -> tuple[str, ...]:
    """Normalize IDs to unique strings and sort by UTF-8 bytes."""
    normalized: list[str] = []
    for index, value in enumerate(example_ids):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(
                "example ID at position %d must be a string or integer" % index
            )
        text = str(value)
        if not text:
            raise ValueError("example IDs must be non-empty")
        if "\x00" in text or "\n" in text or "\r" in text:
            raise ValueError("example IDs cannot contain NUL or line breaks")
        normalized.append(text)
    if not normalized:
        raise ValueError("at least one example ID is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("example IDs must be unique after string normalization")
    return tuple(sorted(normalized, key=lambda item: item.encode("utf-8")))


def read_example_id_list(path: Path) -> tuple[str, ...]:
    """Read an explicit JSON list or a one-ID-per-line UTF-8 text file."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("example ID list must be UTF-8") from error
    if source.suffix.lower() == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise TypeError("JSON example ID file must contain a top-level list")
        return canonicalize_example_ids(parsed)
    values: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value:
            raise ValueError("blank example ID at line %d is not allowed" % line_number)
        values.append(value)
    return canonicalize_example_ids(values)


def condition_order() -> tuple[dict[str, object], ...]:
    conditions: list[dict[str, object]] = [
        {
            "index": 0,
            "name": "identity_control",
            "family": "identity",
            "degrees": 0,
            "included_in_yaw_aggregate": False,
        }
    ]
    for degrees in YAW_DEGREES:
        conditions.append(
            {
                "index": len(conditions),
                "name": "yaw_%03d" % degrees,
                "family": "yaw",
                "degrees": degrees,
                "included_in_yaw_aggregate": True,
            }
        )
    for repeat in range(1, SO3_REPEATS + 1):
        conditions.append(
            {
                "index": len(conditions),
                "name": "so3_%02d" % repeat,
                "family": "so3",
                "repeat": repeat,
                "included_in_yaw_aggregate": False,
            }
        )
    return tuple(conditions)


def _canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=FLOAT_DTYPE)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must have four finite components")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("zero quaternion is invalid")
    value = value / norm
    for component in value:
        if component > 0.0:
            break
        if component < 0.0:
            value = -value
            break
    return np.asarray(value, dtype=FLOAT_DTYPE)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert canonical ``[w, x, y, z]`` to an active rotation matrix."""
    w, x, y, z = _canonicalize_quaternion(quaternion)
    matrix = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=FLOAT_DTYPE,
    )
    return matrix


def yaw_quaternion(degrees: int) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    return _canonicalize_quaternion(
        np.asarray(
            [np.cos(radians / 2.0), 0.0, 0.0, np.sin(radians / 2.0)],
            dtype=FLOAT_DTYPE,
        )
    )


def derive_so3_seed(
    global_seed: int,
    example_id: str,
    repeat: int,
) -> int:
    """Derive a 128-bit seed without depending on iteration order."""
    payload = {
        "schema": SCHEMA,
        "namespace": "haar_so3_per_example_repeat",
        "global_seed": int(global_seed),
        "example_id": str(example_id),
        "repeat": int(repeat),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def sample_haar_quaternion(
    global_seed: int,
    example_id: str,
    repeat: int,
) -> np.ndarray:
    """Sample Haar SO(3) with a per-ID NumPy PCG64 stream."""
    if repeat < 1 or repeat > SO3_REPEATS:
        raise ValueError("SO(3) repeat must be in [1, 8]")
    rng = np.random.Generator(
        np.random.PCG64(derive_so3_seed(global_seed, example_id, repeat))
    )
    u1, u2, u3 = rng.random(3)
    quaternion_xyzw = np.asarray(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        dtype=FLOAT_DTYPE,
    )
    return _canonicalize_quaternion(quaternion_xyzw[[3, 0, 1, 2]])


def assert_rotation_matrices(
    matrices: np.ndarray,
    *,
    orthonormal_atol: float = ORTHONORMAL_ATOL,
    determinant_atol: float = DETERMINANT_ATOL,
) -> dict[str, float]:
    """Fail unless every trailing 3x3 matrix lies in SO(3)."""
    value = np.asarray(matrices)
    if value.ndim < 2 or value.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in shape (3, 3)")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("rotation matrices must have floating dtype")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("rotation matrices contain a non-finite value")
    flattened = value.reshape(-1, 3, 3).astype(np.float64, copy=False)
    identity = np.eye(3, dtype=np.float64)
    gram = np.matmul(
        np.transpose(flattened, (0, 2, 1)),
        flattened,
    )
    orthonormal_error = float(np.max(np.abs(gram - identity)))
    determinants = np.linalg.det(flattened)
    determinant_error = float(np.max(np.abs(determinants - 1.0)))
    if orthonormal_error > float(orthonormal_atol):
        raise AssertionError(
            "rotation orthonormality error %.3e exceeds %.3e"
            % (orthonormal_error, orthonormal_atol)
        )
    if determinant_error > float(determinant_atol):
        raise AssertionError(
            "rotation determinant error %.3e exceeds %.3e"
            % (determinant_error, determinant_atol)
        )
    return {
        "maximum_orthonormality_error": orthonormal_error,
        "maximum_determinant_error": determinant_error,
        "minimum_determinant": float(np.min(determinants)),
        "maximum_determinant": float(np.max(determinants)),
    }


def _validate_bank_geometry(
    example_ids: Sequence[str],
    conditions: Sequence[Mapping[str, object]],
    matrices: np.ndarray,
    quaternions: np.ndarray,
    global_seed: int,
) -> dict[str, float]:
    expected_shape = (len(example_ids), len(conditions))
    if matrices.shape != expected_shape + (3, 3):
        raise AssertionError("matrix bank has an unexpected shape")
    if quaternions.shape != expected_shape + (4,):
        raise AssertionError("quaternion bank has an unexpected shape")
    if matrices.dtype != FLOAT_DTYPE:
        raise TypeError("matrix bank must use little-endian float64")
    if quaternions.dtype != FLOAT_DTYPE:
        raise TypeError("quaternion bank must use little-endian float64")
    if not np.all(np.isfinite(quaternions)):
        raise FloatingPointError("quaternion bank contains a non-finite value")
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1.0))
    )
    if quaternion_norm_error > QUATERNION_ATOL:
        raise AssertionError("quaternion normalization check failed")
    geometry = assert_rotation_matrices(matrices)
    rebuilt = np.empty_like(matrices)
    for example_index in range(len(example_ids)):
        for condition_index in range(len(conditions)):
            rebuilt[example_index, condition_index] = quaternion_to_matrix(
                quaternions[example_index, condition_index]
            )
    quaternion_matrix_error = float(np.max(np.abs(rebuilt - matrices)))
    if quaternion_matrix_error > QUATERNION_ATOL:
        raise AssertionError("quaternion and matrix banks disagree")

    expected_conditions = condition_order()
    if list(conditions) != list(expected_conditions):
        raise AssertionError("condition order differs from the v2 contract")
    identity = np.eye(3, dtype=FLOAT_DTYPE)
    identity_quaternion = np.asarray(
        [1.0, 0.0, 0.0, 0.0],
        dtype=FLOAT_DTYPE,
    )
    if not np.array_equal(
        matrices[:, 0], np.broadcast_to(identity, matrices[:, 0].shape)
    ):
        raise AssertionError("identity control matrix is not exact")
    if not np.array_equal(
        quaternions[:, 0],
        np.broadcast_to(identity_quaternion, quaternions[:, 0].shape),
    ):
        raise AssertionError("identity control quaternion is not exact")
    for offset, degrees in enumerate(YAW_DEGREES, start=1):
        expected_quaternion = yaw_quaternion(degrees)
        expected_matrix = quaternion_to_matrix(expected_quaternion)
        if not np.allclose(
            matrices[:, offset],
            expected_matrix,
            rtol=0.0,
            atol=QUATERNION_ATOL,
        ):
            raise AssertionError("yaw matrix does not match its declared angle")
        if not np.allclose(
            quaternions[:, offset],
            expected_quaternion,
            rtol=0.0,
            atol=QUATERNION_ATOL,
        ):
            raise AssertionError("yaw quaternion does not match its declared angle")
    so3_offset = 1 + len(YAW_DEGREES)
    seed_replay_error = 0.0
    for example_index, example_id in enumerate(example_ids):
        for repeat in range(1, SO3_REPEATS + 1):
            condition_index = so3_offset + repeat - 1
            expected_quaternion = sample_haar_quaternion(
                global_seed,
                example_id,
                repeat,
            )
            error = float(
                np.max(
                    np.abs(
                        quaternions[example_index, condition_index]
                        - expected_quaternion
                    )
                )
            )
            seed_replay_error = max(seed_replay_error, error)
    if seed_replay_error > QUATERNION_ATOL:
        raise AssertionError("SO(3) bank does not replay from its declared seed")
    geometry["maximum_quaternion_norm_error"] = quaternion_norm_error
    geometry["maximum_quaternion_matrix_error"] = quaternion_matrix_error
    geometry["maximum_seed_replay_error"] = seed_replay_error
    return geometry


def _default_generator_sources() -> dict[str, Path]:
    directory = Path(__file__).resolve().parent
    return {
        "rotation_v2/frozen_bank.py": directory / "frozen_bank.py",
        "rotation_v2/generate_frozen_bank.py": (directory / "generate_frozen_bank.py"),
    }


def _source_hashes(
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    paths = (
        _default_generator_sources()
        if source_paths is None
        else {str(key): Path(value) for key, value in source_paths.items()}
    )
    if tuple(sorted(paths)) != tuple(sorted(SOURCE_FILENAMES)):
        raise ValueError("generator source set does not match the v2 contract")
    result: dict[str, str] = {}
    for name in sorted(paths):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError("missing generator source: %s" % path)
        result[name] = file_sha256(path)
    return result


def _materialize_arrays(
    example_ids: Sequence[str],
    global_seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    conditions = condition_order()
    matrices = np.empty(
        (len(example_ids), len(conditions), 3, 3),
        dtype=FLOAT_DTYPE,
    )
    quaternions = np.empty(
        (len(example_ids), len(conditions), 4),
        dtype=FLOAT_DTYPE,
    )
    identity_quaternion = np.asarray(
        [1.0, 0.0, 0.0, 0.0],
        dtype=FLOAT_DTYPE,
    )
    quaternions[:, 0] = identity_quaternion
    matrices[:, 0] = np.eye(3, dtype=FLOAT_DTYPE)
    for offset, degrees in enumerate(YAW_DEGREES, start=1):
        quaternion = yaw_quaternion(degrees)
        quaternions[:, offset] = quaternion
        matrices[:, offset] = quaternion_to_matrix(quaternion)
    so3_offset = 1 + len(YAW_DEGREES)
    for example_index, example_id in enumerate(example_ids):
        for repeat in range(1, SO3_REPEATS + 1):
            condition_index = so3_offset + repeat - 1
            quaternion = sample_haar_quaternion(
                global_seed,
                example_id,
                repeat,
            )
            quaternions[example_index, condition_index] = quaternion
            matrices[example_index, condition_index] = quaternion_to_matrix(quaternion)
    _validate_bank_geometry(
        example_ids,
        conditions,
        matrices,
        quaternions,
        global_seed,
    )
    return matrices, quaternions, conditions


def _artifact_record(path: Path, logical_sha256: str) -> dict[str, object]:
    return {
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha256(path),
        "logical_sha256": logical_sha256,
    }


def _write_bank_directory(
    directory: Path,
    example_ids: Sequence[str],
    global_seed: int,
    source_paths: Mapping[str, Path] | None,
) -> dict[str, object]:
    matrices, quaternions, conditions = _materialize_arrays(
        example_ids,
        global_seed,
    )
    matrix_path = directory / "matrices.npy"
    quaternion_path = directory / "quaternions.npy"
    id_path = directory / "example_ids.json"
    condition_path = directory / "condition_order.json"
    np.save(matrix_path, matrices, allow_pickle=False)
    np.save(quaternion_path, quaternions, allow_pickle=False)
    example_id_bytes = canonical_json_bytes(list(example_ids))
    condition_bytes = canonical_json_bytes(list(conditions))
    id_path.write_bytes(example_id_bytes)
    condition_path.write_bytes(condition_bytes)

    source_hashes = _source_hashes(source_paths)
    artifacts = {
        "matrices.npy": {
            **_artifact_record(
                matrix_path,
                logical_array_sha256(matrices),
            ),
            "dtype": matrices.dtype.str,
            "shape": [int(size) for size in matrices.shape],
        },
        "quaternions.npy": {
            **_artifact_record(
                quaternion_path,
                logical_array_sha256(quaternions),
            ),
            "dtype": quaternions.dtype.str,
            "shape": [int(size) for size in quaternions.shape],
        },
        "example_ids.json": {
            **_artifact_record(
                id_path,
                hashlib.sha256(example_id_bytes).hexdigest(),
            ),
            "encoding": "canonical-json-utf8",
        },
        "condition_order.json": {
            **_artifact_record(
                condition_path,
                hashlib.sha256(condition_bytes).hexdigest(),
            ),
            "encoding": "canonical-json-utf8",
        },
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "global_seed": int(global_seed),
        "num_examples": len(example_ids),
        "num_conditions": len(conditions),
        "id_order": "unique_string_ids_sorted_by_utf8_bytes",
        "condition_contract": {
            "identity_control": "stored_separately_and_excluded_from_yaw",
            "yaw_degrees": list(YAW_DEGREES),
            "so3_repeats": SO3_REPEATS,
        },
        "seed_derivation": {
            "hash": "sha256",
            "digest_prefix_bits": 128,
            "rng": "numpy.random.PCG64",
            "scope": "global_seed_example_id_repeat",
            "order_independent": True,
        },
        "matrix_convention": {
            "action": "active",
            "row_vector_application": "x_rotated = x @ R.T",
            "quaternion_order": ["w", "x", "y", "z"],
            "dtype": FLOAT_DTYPE.str,
        },
        "artifacts": artifacts,
        "content_sha256": _combined_content_sha256(
            matrices,
            quaternions,
            example_id_bytes,
            condition_bytes,
        ),
        "generator_sources": source_hashes,
        "generator_source_sha256": canonical_json_sha256(source_hashes),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


@dataclass(frozen=True)
class FrozenRotationBank:
    """Verified in-memory view of a frozen bank."""

    directory: Path
    manifest: Mapping[str, object]
    example_ids: tuple[str, ...]
    conditions: tuple[Mapping[str, object], ...]
    matrices: np.ndarray
    quaternions: np.ndarray

    def example_index(self, example_id: str | int) -> int:
        key = str(example_id)
        try:
            return self.example_ids.index(key)
        except ValueError as error:
            raise KeyError(
                "example ID is absent from the frozen bank: %s" % key
            ) from error

    def condition_index(self, condition_name: str) -> int:
        matches = [
            index
            for index, condition in enumerate(self.conditions)
            if condition["name"] == condition_name
        ]
        if len(matches) != 1:
            raise KeyError("condition is absent or ambiguous: %s" % condition_name)
        return matches[0]

    def matrix(
        self,
        example_id: str | int,
        condition_name: str,
    ) -> np.ndarray:
        return self.matrices[
            self.example_index(example_id),
            self.condition_index(condition_name),
        ]

    def quaternion(
        self,
        example_id: str | int,
        condition_name: str,
    ) -> np.ndarray:
        return self.quaternions[
            self.example_index(example_id),
            self.condition_index(condition_name),
        ]


def _load_canonical_json(path: Path) -> object:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise AssertionError("%s is not canonical JSON" % path.name)
    return value


def load_frozen_bank(
    directory: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> FrozenRotationBank:
    """Load only after byte hashes, semantic hashes, and SO(3) checks pass."""
    bank_dir = Path(directory)
    if not bank_dir.is_dir():
        raise FileNotFoundError("frozen bank directory is missing")
    expected_files = set(ARTIFACT_FILENAMES) | {"manifest.json"}
    entries = list(bank_dir.iterdir())
    actual_files = {path.name for path in entries}
    if actual_files != expected_files:
        raise AssertionError(
            "frozen bank file set mismatch: missing=%s unexpected=%s"
            % (
                sorted(expected_files - actual_files),
                sorted(actual_files - expected_files),
            )
        )
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise AssertionError("frozen bank entries must be regular non-symlink files")
    manifest_value = _load_canonical_json(bank_dir / "manifest.json")
    if not isinstance(manifest_value, dict):
        raise TypeError("frozen bank manifest must be a mapping")
    manifest = manifest_value
    stored_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(stored_manifest_hash, str):
        raise TypeError("manifest_sha256 is missing or invalid")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256")
    if canonical_json_sha256(unhashed_manifest) != stored_manifest_hash:
        raise AssertionError("frozen bank manifest hash mismatch")
    if (
        expected_manifest_sha256 is not None
        and stored_manifest_hash != expected_manifest_sha256
    ):
        raise AssertionError("frozen bank differs from the preregistered manifest")
    if manifest.get("schema") != SCHEMA:
        raise AssertionError("unknown frozen bank schema")
    global_seed_value = manifest.get("global_seed")
    if isinstance(global_seed_value, bool) or not isinstance(global_seed_value, int):
        raise TypeError("manifest global seed must be an integer")
    if global_seed_value < 0 or global_seed_value >= 2**64:
        raise ValueError("manifest global seed is outside uint64")
    if manifest.get("id_order") != "unique_string_ids_sorted_by_utf8_bytes":
        raise AssertionError("manifest ID order contract mismatch")
    if manifest.get("condition_contract") != {
        "identity_control": "stored_separately_and_excluded_from_yaw",
        "yaw_degrees": list(YAW_DEGREES),
        "so3_repeats": SO3_REPEATS,
    }:
        raise AssertionError("manifest condition contract mismatch")
    if manifest.get("seed_derivation") != {
        "hash": "sha256",
        "digest_prefix_bits": 128,
        "rng": "numpy.random.PCG64",
        "scope": "global_seed_example_id_repeat",
        "order_independent": True,
    }:
        raise AssertionError("manifest seed derivation contract mismatch")
    if manifest.get("matrix_convention") != {
        "action": "active",
        "row_vector_application": "x_rotated = x @ R.T",
        "quaternion_order": ["w", "x", "y", "z"],
        "dtype": FLOAT_DTYPE.str,
    }:
        raise AssertionError("manifest matrix convention mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_FILENAMES):
        raise AssertionError("frozen bank artifact manifest is incomplete")
    for filename in ARTIFACT_FILENAMES:
        record = artifacts[filename]
        if not isinstance(record, dict):
            raise TypeError("artifact record must be a mapping")
        path = bank_dir / filename
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise AssertionError("artifact byte count mismatch: %s" % filename)
        if record.get("file_sha256") != file_sha256(path):
            raise AssertionError("artifact file hash mismatch: %s" % filename)

    example_id_value = _load_canonical_json(bank_dir / "example_ids.json")
    condition_value = _load_canonical_json(bank_dir / "condition_order.json")
    if not isinstance(example_id_value, list):
        raise TypeError("example ID artifact must be a JSON list")
    if not isinstance(condition_value, list):
        raise TypeError("condition artifact must be a JSON list")
    example_ids = canonicalize_example_ids(example_id_value)
    if list(example_ids) != example_id_value:
        raise AssertionError("stored example IDs are not in canonical order")
    conditions = tuple(condition_value)
    matrices = np.load(bank_dir / "matrices.npy", allow_pickle=False)
    quaternions = np.load(bank_dir / "quaternions.npy", allow_pickle=False)
    if not isinstance(matrices, np.ndarray) or not isinstance(quaternions, np.ndarray):
        raise TypeError("frozen matrix artifacts must be NumPy arrays")

    matrix_record = artifacts["matrices.npy"]
    quaternion_record = artifacts["quaternions.npy"]
    if matrix_record.get("dtype") != matrices.dtype.str or matrix_record.get(
        "shape"
    ) != [int(size) for size in matrices.shape]:
        raise AssertionError("matrix artifact dtype/shape record mismatch")
    if quaternion_record.get("dtype") != quaternions.dtype.str or quaternion_record.get(
        "shape"
    ) != [int(size) for size in quaternions.shape]:
        raise AssertionError("quaternion artifact dtype/shape record mismatch")
    if artifacts["example_ids.json"].get("encoding") != "canonical-json-utf8":
        raise AssertionError("example ID artifact encoding mismatch")
    if artifacts["condition_order.json"].get("encoding") != "canonical-json-utf8":
        raise AssertionError("condition artifact encoding mismatch")
    if matrix_record.get("logical_sha256") != logical_array_sha256(matrices):
        raise AssertionError("matrix logical hash mismatch")
    if quaternion_record.get("logical_sha256") != logical_array_sha256(quaternions):
        raise AssertionError("quaternion logical hash mismatch")
    id_bytes = canonical_json_bytes(list(example_ids))
    condition_bytes = canonical_json_bytes(list(conditions))
    if (
        artifacts["example_ids.json"].get("logical_sha256")
        != hashlib.sha256(id_bytes).hexdigest()
    ):
        raise AssertionError("example ID logical hash mismatch")
    if (
        artifacts["condition_order.json"].get("logical_sha256")
        != hashlib.sha256(condition_bytes).hexdigest()
    ):
        raise AssertionError("condition order logical hash mismatch")
    if manifest.get("content_sha256") != _combined_content_sha256(
        matrices,
        quaternions,
        id_bytes,
        condition_bytes,
    ):
        raise AssertionError("combined frozen bank content hash mismatch")
    if int(manifest.get("num_examples", -1)) != len(example_ids):
        raise AssertionError("manifest example count mismatch")
    if int(manifest.get("num_conditions", -1)) != len(conditions):
        raise AssertionError("manifest condition count mismatch")
    _validate_bank_geometry(
        example_ids,
        conditions,
        matrices,
        quaternions,
        global_seed_value,
    )

    source_hashes = manifest.get("generator_sources")
    if not isinstance(source_hashes, dict):
        raise TypeError("generator source hash mapping is missing")
    if manifest.get("generator_source_sha256") != canonical_json_sha256(source_hashes):
        raise AssertionError("combined generator source hash mismatch")
    current_source_hashes = _source_hashes()
    if source_hashes != current_source_hashes:
        raise AssertionError("generator source changed after bank creation")

    matrices = np.ascontiguousarray(matrices)
    quaternions = np.ascontiguousarray(quaternions)
    matrices.setflags(write=False)
    quaternions.setflags(write=False)
    return FrozenRotationBank(
        directory=bank_dir.resolve(),
        manifest=manifest,
        example_ids=example_ids,
        conditions=conditions,
        matrices=matrices,
        quaternions=quaternions,
    )


def generate_frozen_bank(
    directory: Path,
    example_ids: Iterable[str | int],
    *,
    global_seed: int,
    source_paths: Mapping[str, Path] | None = None,
) -> FrozenRotationBank:
    """Create a new bank atomically and refuse every overwrite."""
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global seed must be an integer")
    if global_seed < 0 or global_seed >= 2**64:
        raise ValueError("global seed must be in [0, 2**64)")
    canonical_ids = canonicalize_example_ids(example_ids)
    output_dir = Path(directory)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite frozen bank directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=".%s.tmp." % output_dir.name,
            dir=output_dir.parent,
        )
    )
    try:
        manifest = _write_bank_directory(
            temporary_dir,
            canonical_ids,
            global_seed,
            source_paths,
        )
        load_frozen_bank(
            temporary_dir,
            expected_manifest_sha256=str(manifest["manifest_sha256"]),
        )
        os.replace(temporary_dir, output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return load_frozen_bank(
        output_dir,
        expected_manifest_sha256=str(manifest["manifest_sha256"]),
    )
