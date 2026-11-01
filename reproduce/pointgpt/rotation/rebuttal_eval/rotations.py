"""Explicit, row-vector rotation primitives for the guarded E-Rot evaluator."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALLOWED_POST_ROTATION_PROCESSING = ("none", "center_isotropic")


def axis_angle_matrix(axis: str, angle_degrees: float) -> np.ndarray:
    """Return an explicit SO(3) matrix for row-vector points: ``points @ R``."""
    axis = str(axis).lower()
    if axis not in AXIS_TO_INDEX:
        raise ValueError("axis must be x, y, or z")
    theta = np.deg2rad(float(angle_degrees))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    # These are transposes of the usual column-vector active rotations.
    if axis == "x":
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    elif axis == "y":
        matrix = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    else:
        matrix = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    assert_rotation_matrix(matrix)
    return matrix


def quaternion_to_row_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion_xyzw.shape[-1] != 4:
        raise ValueError("quaternion must end in (x, y, z, w)")
    norm = np.linalg.norm(quaternion_xyzw, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero quaternion")
    x, y, z, w = np.moveaxis(quaternion_xyzw / norm, -1, 0)
    # Column-vector active rotation, transposed at return for row-vector points.
    column = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternion_xyzw.shape[:-1] + (3, 3))
    return np.swapaxes(column, -1, -2)


def sample_haar_so3(
    rng: np.random.RandomState, shape: Sequence[int] = ()
) -> Tuple[np.ndarray, np.ndarray]:
    """Shoemake sampler: uniform unit quaternions, hence Haar-uniform SO(3)."""
    shape = tuple(int(x) for x in shape)
    u1 = rng.uniform(size=shape)
    u2 = rng.uniform(size=shape)
    u3 = rng.uniform(size=shape)
    quaternion = np.stack(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        axis=-1,
    )
    matrices = quaternion_to_row_matrix(quaternion)
    return matrices, quaternion


def assert_rotation_matrix(matrix: np.ndarray, atol: float = 1e-10) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in 3x3")
    eye = np.eye(3, dtype=np.float64)
    gram = np.matmul(np.swapaxes(matrix, -1, -2), matrix)
    if not np.allclose(gram, eye, atol=atol, rtol=0.0):
        raise AssertionError("matrix is not orthonormal")
    determinant = np.linalg.det(matrix)
    if not np.allclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise AssertionError("matrix determinant is not +1")


def validate_post_rotation_processing(mode: str) -> None:
    if mode not in ALLOWED_POST_ROTATION_PROCESSING:
        raise AssertionError(
            "post-rotation PCA/upright re-canonicalization is forbidden; "
            "only none or center_isotropic is allowed"
        )


def apply_rotation(
    points: np.ndarray,
    matrix: Optional[np.ndarray],
    normal_slice: Optional[Tuple[int, int]] = None,
    post_rotation_processing: str = "none",
) -> np.ndarray:
    """Rotate selected points/features; ``matrix=None`` is the default-OFF path."""
    validate_post_rotation_processing(post_rotation_processing)
    points = np.asarray(points)
    if matrix is None:
        return points.copy()
    matrix = np.asarray(matrix, dtype=np.float64)
    assert_rotation_matrix(matrix, atol=1e-7)
    if points.shape[-1] < 3:
        raise ValueError("points need at least XYZ channels")
    out = points.copy()
    out[..., :3] = np.matmul(points[..., :3], matrix).astype(points.dtype, copy=False)
    if normal_slice is not None:
        start, stop = normal_slice
        if stop - start != 3 or stop > points.shape[-1]:
            raise ValueError("normal_slice must select exactly three available channels")
        out[..., start:stop] = np.matmul(points[..., start:stop], matrix).astype(
            points.dtype, copy=False
        )
    if post_rotation_processing == "center_isotropic":
        xyz = out[..., :3]
        centroid = xyz.mean(axis=-2, keepdims=True)
        centered = xyz - centroid
        scale = np.linalg.norm(centered, axis=-1).max(axis=-1, keepdims=True)
        scale = np.maximum(scale, np.finfo(out.dtype).eps)
        out[..., :3] = centered / np.expand_dims(scale, axis=-1)
    return out


def pairwise_distance_error(before: np.ndarray, after: np.ndarray) -> float:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3:
        raise ValueError("before and after must both have shape (N, 3)")
    before_dist = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=-1)
    after_dist = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=-1)
    return float(np.max(np.abs(before_dist - after_dist)))


def haar_uniformity_diagnostics(
    matrices: np.ndarray, atol_mean: float = 0.03, atol_second: float = 0.03
) -> Dict[str, object]:
    """Basic deterministic statistical check: R e_z is uniform on S^2."""
    matrices = np.asarray(matrices, dtype=np.float64)
    assert_rotation_matrix(matrices, atol=1e-7)
    north = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    directions = np.matmul(north, matrices)
    mean = directions.mean(axis=0)
    second = directions.T @ directions / directions.shape[0]
    expected_second = np.eye(3) / 3.0
    passed = bool(
        np.max(np.abs(mean)) <= float(atol_mean)
        and np.max(np.abs(second - expected_second)) <= float(atol_second)
    )
    return {
        "passed": passed,
        "direction_mean": mean.tolist(),
        "direction_second_moment": second.tolist(),
        "max_abs_mean": float(np.max(np.abs(mean))),
        "max_abs_second_moment_error": float(np.max(np.abs(second - expected_second))),
        "n": int(directions.shape[0]),
    }

