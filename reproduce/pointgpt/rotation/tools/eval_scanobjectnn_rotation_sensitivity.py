#!/usr/bin/env python
"""Guarded object-level E-Rot evaluator for PointGPT ScanObjectNN variants.

All input points/FPS choices and all rotation matrices are frozen before any
variant is loaded. The evaluator contains no voting, scale/translation test
augmentation, PCA, or upright re-canonicalization.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal_eval.determinism import (  # noqa: E402
    configure_deterministic_model,
    configure_deterministic_runtime,
    deterministic_model_forward,
)
from rebuttal_eval.r7 import canonical_json_sha256, file_sha256, paired_summary  # noqa: E402
from rebuttal_eval.rotations import (  # noqa: E402
    apply_rotation,
    assert_rotation_matrix,
    axis_angle_matrix,
    haar_uniformity_diagnostics,
    pairwise_distance_error,
    sample_haar_so3,
    validate_post_rotation_processing,
)
COORDINATE_CONVENTION = {
    "upright_axis": "z",
    "rationale": (
        "ScanObjectNN is cut from gravity-aligned SceneNN/ScanNet scenes; its "
        "supplement limits horizontal perturbations to the gravity axis. "
        "ScanNet's released world meshes are +Z-up, and this repository's "
        "ScanObjectNN loader passes x,y,z through without an axis permutation. "
        "Therefore yaw is explicitly about z, not the y-axis used by "
        "datasets/data_transforms.py::PointcloudRotate."
    ),
    "sources": [
        "https://hkust-vgd.github.io/scanobjectnn/assets/iccv19_supp.pdf",
        "https://www.scan-net.org/ScanNet/",
        "datasets/ScanObjectNNDataset.py",
    ],
}


def parse_degrees(value: str) -> List[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("angle list cannot be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Guarded ScanObjectNN object rotation sensitivity")
    parser.add_argument(
        "--variant",
        action="append",
        nargs=3,
        metavar=("NAME", "CONFIG", "CKPT"),
        required=True,
        help="Repeat for official and every relaxed objbg checkpoint.",
    )
    parser.add_argument("--official_name", default="official")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upright_axis", default="z", choices=["x", "y", "z"])
    parser.add_argument(
        "--allow_non_z_upright",
        action="store_true",
        default=False,
        help="Explicit override only; recorded as a coordinate-convention risk.",
    )
    parser.add_argument(
        "--axis_override_evidence",
        default="",
        help="Required provenance note when overriding the audited z-up convention.",
    )
    parser.add_argument("--yaw_degrees", type=parse_degrees, default=parse_degrees("0,15,30,45,60,90,120,150,180"))
    parser.add_argument("--tilt_axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--tilt_degrees", type=parse_degrees, default=parse_degrees("0,15,30,45,60,90"))
    parser.add_argument("--so3_repeats", type=int, default=4)
    parser.add_argument("--point_plan_dir", required=True)
    parser.add_argument("--matrix_plan_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--post_rotation_processing",
        default="none",
        choices=["none"],
        help="Locked to none: no post-rotation canonicalization is permitted.",
    )
    parser.add_argument("--pairwise_check_points", type=int, default=128)
    parser.add_argument(
        "--allow_single_variant",
        action="store_true",
        default=False,
        help="Smoke-test escape hatch; full rebuttal evaluation must not use it.",
    )
    return parser.parse_args()


def _load_dataset_base_view(base) -> Dict[str, object]:
    if isinstance(base, (str, Path)):
        import yaml

        with Path(base).open("r") as handle:
            base = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(base, Mapping):
        raise TypeError("dataset _base_ must resolve to a mapping")
    return copy.deepcopy(dict(base))


def dataset_config_view(node: Mapping) -> Dict[str, object]:
    """Resolve the builder-visible dataset config without mutating ``node``.

    PointGPT's dataset builder consumes ``node._base_`` as the dataset config
    and ``node.others`` as default/override arguments.  This view mirrors that
    read path only for signatures and config validation; the real node retains
    ``_base_`` for ``tools.builder.dataset_builder``.
    """
    if "_base_" not in node:
        raise KeyError("dataset node must retain _base_ for PointGPT's builder")
    view = _load_dataset_base_view(node["_base_"])
    others = node.get("others", {})
    if not isinstance(others, Mapping):
        raise TypeError("dataset others must be a mapping")
    for key, value in others.items():
        view[key] = copy.deepcopy(value)
    return view


def _builder_root_from_node_override(node: Mapping) -> Optional[str]:
    """Translate legacy config-prep's node-level ROOT to the builder's ROOT."""
    if "ROOT" not in node:
        return None
    override = Path(str(node["ROOT"]))
    base_root = Path(str(_load_dataset_base_view(node["_base_"])["ROOT"]))
    if override.name == "ScanObjectNN":
        parts = base_root.parts
        if "ScanObjectNN" in parts:
            marker = parts.index("ScanObjectNN")
            override = override.joinpath(*parts[marker + 1 :])
    return str(override)


def load_config(
    path: str,
    batch_size: int,
    split: str,
    config_loader: Optional[Callable[[str], object]] = None,
):
    if config_loader is None:
        from utils.config import cfg_from_yaml_file

        config_loader = cfg_from_yaml_file
    config = config_loader(path)
    test_node = config.dataset.test
    if "_base_" not in test_node:
        raise KeyError("dataset.test must retain _base_ for PointGPT's builder")
    test_node.others.bs = batch_size
    test_node.others.subset = split
    root_override = _builder_root_from_node_override(test_node)
    if root_override is not None:
        test_node.others.ROOT = root_override
    return config


def config_dataset_signature(config) -> Dict[str, object]:
    dataset = dataset_config_view(config.dataset.test)
    return {
        "npoints": int(config.npoints),
        "dataset_name": str(dataset["NAME"]),
        "root": str(Path(dataset["ROOT"]).resolve()),
        "subset": str(dataset["subset"]),
    }


def build_loader(config, num_workers: int):
    from tools import builder

    args = SimpleNamespace(distributed=False, num_workers=num_workers)
    _, loader = builder.dataset_builder(args, config.dataset.test)
    return loader


def build_model(config, checkpoint: str) -> torch.nn.Module:
    from tools import builder

    model = builder.model_builder(config.model)
    builder.load_model(model, checkpoint, logger=None)
    model = model.cuda()
    configure_deterministic_model(model)
    return model


def save_plan_array(plan_dir: Path, name: str, value: np.ndarray, files: Dict[str, str]) -> None:
    path = plan_dir / (name + ".npy")
    np.save(path, value, allow_pickle=False)
    files[path.name] = file_sha256(path)


def verify_plan_files(plan_dir: Path, manifest: Mapping[str, object]) -> None:
    for filename, expected_hash in manifest["files"].items():
        actual_hash = file_sha256(plan_dir / filename)
        if actual_hash != expected_hash:
            raise AssertionError("frozen plan hash mismatch: %s" % filename)
    stored_hash = manifest["manifest_sha256"]
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256")
    if stored_hash != canonical_json_sha256(unhashed):
        raise AssertionError("frozen plan manifest hash mismatch")


def freeze_points_once(args: argparse.Namespace, config) -> Dict[str, object]:
    from utils import misc

    plan_dir = Path(args.point_plan_dir)
    manifest_path = plan_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_plan_files(plan_dir, manifest)
        expected_fields = {
            "seed": int(args.seed),
            "npoints": int(config.npoints),
            "dataset": config_dataset_signature(config),
            "max_objects_request": int(args.max_objects),
        }
        for field, expected in expected_fields.items():
            if manifest.get(field) != expected:
                raise AssertionError(
                    "frozen point plan %s=%r does not match requested %r"
                    % (field, manifest.get(field), expected)
                )
        return manifest
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty point_plan_dir")
    plan_dir.mkdir(parents=True, exist_ok=True)
    loader = build_loader(config, args.num_workers)
    all_points, all_labels = [], []
    seen = 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        for _, _, data in tqdm(loader, desc="freeze exact ScanObjectNN FPS points once"):
            raw_points = data[0].cuda()
            selected = misc.fps(raw_points, int(config.npoints)).cpu().numpy()
            labels = data[1].view(-1).cpu().numpy().astype(np.int64)
            if args.max_objects > 0:
                remaining = args.max_objects - seen
                if remaining <= 0:
                    break
                selected = selected[:remaining]
                labels = labels[:remaining]
            all_points.append(selected.astype(np.float32, copy=False))
            all_labels.append(labels)
            seen += labels.size
            if args.max_objects > 0 and seen >= args.max_objects:
                break
    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    files: Dict[str, str] = {}
    save_plan_array(plan_dir, "selected_points", points, files)
    save_plan_array(plan_dir, "labels", labels, files)
    manifest = {
        "schema": "pointgpt_erot_frozen_points_v1",
        "selection": "one FPS pass before loading/comparing model variants",
        "seed": int(args.seed),
        "num_objects": int(labels.size),
        "npoints": int(config.npoints),
        "max_objects_request": int(args.max_objects),
        "channels": int(points.shape[-1]),
        "dataset": config_dataset_signature(config),
        "random_test_scale_translation": False,
        "voting": False,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    verify_plan_files(plan_dir, manifest)
    return manifest


def condition_name(prefix: str, degrees: float) -> str:
    return "%s_%sdeg" % (prefix, ("%g" % degrees).replace("-", "m").replace(".", "p"))


def freeze_matrix_bank(args: argparse.Namespace, num_objects: int) -> Dict[str, object]:
    plan_dir = Path(args.matrix_plan_dir)
    manifest_path = plan_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_plan_files(plan_dir, manifest)
        expected_fields = {
            "seed": int(args.seed),
            "num_objects": int(num_objects),
            "upright_axis": args.upright_axis,
            "tilt_axis": args.tilt_axis,
            "yaw_degrees": args.yaw_degrees,
            "tilt_degrees": args.tilt_degrees,
            "so3_repeats": int(args.so3_repeats),
        }
        for field, expected in expected_fields.items():
            if manifest.get(field) != expected:
                raise AssertionError(
                    "frozen matrix plan %s=%r does not match requested %r"
                    % (field, manifest.get(field), expected)
                )
        return manifest
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty matrix_plan_dir")
    plan_dir.mkdir(parents=True, exist_ok=True)

    names = ["clean"]
    kinds = ["clean"]
    values = [0.0]
    matrix_rows = [np.repeat(np.eye(3, dtype=np.float64)[None], num_objects, axis=0)]
    for degrees in args.yaw_degrees:
        names.append(condition_name("yaw", degrees))
        kinds.append("yaw")
        values.append(float(degrees))
        matrix_rows.append(
            np.repeat(
                axis_angle_matrix(args.upright_axis, degrees)[None],
                num_objects,
                axis=0,
            )
        )
    for degrees in args.tilt_degrees:
        names.append(condition_name("tilt_%s" % args.tilt_axis, degrees))
        kinds.append("tilt")
        values.append(float(degrees))
        matrix_rows.append(
            np.repeat(axis_angle_matrix(args.tilt_axis, degrees)[None], num_objects, axis=0)
        )
    rng = np.random.RandomState(args.seed + 73_921)
    so3_matrices, so3_quaternions = sample_haar_so3(
        rng, shape=(args.so3_repeats, num_objects)
    )
    for repeat in range(args.so3_repeats):
        names.append("so3_haar_%02d" % repeat)
        kinds.append("so3_haar")
        values.append(None)
        matrix_rows.append(so3_matrices[repeat])
    matrices = np.stack(matrix_rows)
    assert_rotation_matrix(matrices, atol=1e-7)
    files: Dict[str, str] = {}
    save_plan_array(plan_dir, "matrices", matrices, files)
    save_plan_array(plan_dir, "so3_quaternions", so3_quaternions, files)
    conditions_path = plan_dir / "conditions.json"
    conditions_path.write_text(
        json.dumps(
            [
                {"index": index, "name": name, "kind": kind, "degrees": value}
                for index, (name, kind, value) in enumerate(zip(names, kinds, values))
            ],
            indent=2,
        )
    )
    files[conditions_path.name] = file_sha256(conditions_path)
    haar_check = haar_uniformity_diagnostics(
        sample_haar_so3(np.random.RandomState(args.seed + 991), shape=(20000,))[0]
    )
    if not haar_check["passed"]:
        raise AssertionError("Haar SO(3) sampler basic uniformity check failed")
    manifest = {
        "schema": "pointgpt_erot_matrix_bank_v1",
        "seed": int(args.seed),
        "num_objects": int(num_objects),
        "upright_axis": args.upright_axis,
        "tilt_axis": args.tilt_axis,
        "yaw_degrees": args.yaw_degrees,
        "tilt_degrees": args.tilt_degrees,
        "so3_repeats": int(args.so3_repeats),
        "sampler": "Shoemake uniform-unit-quaternion Haar SO(3), not Euler angles",
        "haar_uniformity_check": haar_check,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    verify_plan_files(plan_dir, manifest)
    return manifest


def load_plan_array(plan_dir: Path, name: str) -> np.ndarray:
    return np.load(plan_dir / (name + ".npy"), mmap_mode="r", allow_pickle=False)


def normal_slice_for(points: np.ndarray):
    if points.shape[-1] == 3:
        return None
    if points.shape[-1] >= 6:
        return (3, 6)
    raise AssertionError("unexpected point feature layout; cannot rotate normals consistently")


def evaluate_variant(
    args: argparse.Namespace,
    name: str,
    config,
    config_path: str,
    checkpoint: str,
    points: np.ndarray,
    labels: np.ndarray,
    matrices: np.ndarray,
    conditions: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    model = build_model(config, checkpoint)
    predictions = {
        condition["name"]: np.empty(labels.shape[0], dtype=np.int64)
        for condition in conditions
    }
    pairwise_max_error = {condition["name"]: 0.0 for condition in conditions}
    zero_replay_name = condition_name("yaw", 0.0)
    if zero_replay_name not in predictions:
        raise AssertionError("yaw sweep must include a 0-degree clean replay")
    normal_slice = normal_slice_for(points)
    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, labels.shape[0], args.batch_size),
                desc="E-Rot %s" % name,
            ):
                stop = min(labels.shape[0], start + args.batch_size)
                clean_logits = None
                clean_points = None
                # The seed varies by frozen batch, but is deliberately identical
                # for clean and every rotated condition in that batch.
                forward_seed = (int(args.seed) + int(start)) % (2**63 - 1)
                for condition_index, condition in enumerate(conditions):
                    rotated_rows = []
                    for row, object_index in enumerate(range(start, stop)):
                        matrix = np.asarray(matrices[condition_index, object_index])
                        rotated = apply_rotation(
                            np.asarray(points[object_index]),
                            matrix,
                            normal_slice=normal_slice,
                            post_rotation_processing=args.post_rotation_processing,
                        )
                        rotated_rows.append(rotated)
                        if row == 0:
                            check_n = min(args.pairwise_check_points, rotated.shape[0])
                            error = pairwise_distance_error(
                                np.asarray(points[object_index, :check_n, :3]),
                                rotated[:check_n, :3],
                            )
                            if error > 1e-5:
                                raise AssertionError(
                                    "rotation failed pairwise-distance preservation"
                                )
                            pairwise_max_error[condition["name"]] = max(
                                pairwise_max_error[condition["name"]], error
                            )
                    rotated_np = np.stack(rotated_rows)
                    if condition["name"] == "clean":
                        clean_points = rotated_np.copy()
                        if not np.array_equal(clean_points, np.asarray(points[start:stop])):
                            raise AssertionError(
                                "E-Rot clean input is not the exact frozen point batch"
                            )
                    elif condition["name"] == zero_replay_name:
                        if clean_points is None or not np.array_equal(
                            rotated_np, clean_points
                        ):
                            raise AssertionError(
                                "0-degree rotation is not identity on the clean frozen points"
                            )
                    logits, _ = deterministic_model_forward(
                        model,
                        torch.from_numpy(rotated_np[..., :3]).float().cuda(),
                        seed=forward_seed,
                        compute_recon=False,
                    )
                    if condition["name"] == "clean":
                        clean_logits = logits.detach().clone()
                    elif condition["name"] == zero_replay_name:
                        if clean_logits is None or not torch.equal(logits, clean_logits):
                            raise AssertionError(
                                "0-degree replay differs from clean; stochastic/canonicalization guard failed"
                            )
                    predictions[condition["name"]][start:stop] = (
                        logits.argmax(dim=-1).view(-1).cpu().numpy()
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    clean_pred = predictions["clean"]
    rows = []
    for condition in conditions:
        pred = predictions[condition["name"]]
        correct = pred == labels
        clean_correct = clean_pred == labels
        rows.append(
            {
                "condition": condition,
                "accuracy": float(correct.mean()),
                "accuracy_percent": float(100.0 * correct.mean()),
                "prediction_consistency_with_clean": float((pred == clean_pred).mean()),
                "drop_from_clean_pp": float(100.0 * (clean_correct.mean() - correct.mean())),
                "per_example_correct": correct.astype(np.int8).tolist(),
                "per_example_consistent": (pred == clean_pred).astype(np.int8).tolist(),
                "pairwise_distance_max_abs_error": pairwise_max_error[condition["name"]],
            }
        )
    return {
        "name": name,
        "config": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(Path(config_path)),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(Path(checkpoint)),
        "rows": rows,
        "predictions": {key: value.tolist() for key, value in predictions.items()},
    }


def paired_variant_drop_interactions(
    variants: Sequence[Mapping[str, object]], official_name: str
) -> List[Dict[str, object]]:
    by_name = {variant["name"]: variant for variant in variants}
    official = by_name[official_name]
    official_rows = {row["condition"]["name"]: row for row in official["rows"]}
    output = []
    for variant in variants:
        if variant["name"] == official_name:
            continue
        variant_rows = {row["condition"]["name"]: row for row in variant["rows"]}
        for condition_name_value, official_row in official_rows.items():
            relaxed_row = variant_rows[condition_name_value]
            official_correct = np.asarray(official_row["per_example_correct"], dtype=np.int8)
            relaxed_correct = np.asarray(relaxed_row["per_example_correct"], dtype=np.int8)
            official_clean = np.asarray(
                official_rows["clean"]["per_example_correct"], dtype=np.int8
            )
            relaxed_clean = np.asarray(
                variant_rows["clean"]["per_example_correct"], dtype=np.int8
            )
            paired = (official_clean - official_correct) - (
                relaxed_clean - relaxed_correct
            )
            summary = paired_summary(paired.tolist())
            output.append(
                {
                    "official": official_name,
                    "relaxation": variant["name"],
                    "condition": condition_name_value,
                    "estimand": (
                        "(official clean-correct minus rotated-correct) minus "
                        "(relaxation clean-correct minus rotated-correct)"
                    ),
                    "paired_difference_in_rotation_drop_pp": {
                        key: (100.0 * value if key != "n" else value)
                        for key, value in summary.items()
                    },
                }
            )
    return output


def main() -> None:
    args = parse_args()
    # Configure before freeze_points_once performs the first CUDA operation.
    configure_deterministic_runtime()
    validate_post_rotation_processing(args.post_rotation_processing)
    if args.upright_axis != "z" and not args.allow_non_z_upright:
        raise AssertionError(
            "ScanObjectNN coordinate audit selects z-up; pass --allow_non_z_upright "
            "only with external evidence and accept the recorded risk"
        )
    if args.upright_axis != "z" and not args.axis_override_evidence.strip():
        raise AssertionError(
            "non-z upright override requires --axis_override_evidence provenance"
        )
    if args.tilt_axis == args.upright_axis:
        raise ValueError("fixed tilt axis must be horizontal, not the upright/yaw axis")
    if args.so3_repeats < 1:
        raise ValueError("--so3_repeats must be positive")
    if args.pairwise_check_points < 2:
        raise ValueError("--pairwise_check_points must be at least 2")
    variant_names = [item[0] for item in args.variant]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("variant names must be unique")
    if args.official_name not in variant_names:
        raise ValueError("official variant is missing")
    if len(args.variant) < 2 and not args.allow_single_variant:
        raise ValueError("full E-Rot requires official plus at least one relaxed objbg variant")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    configs = {
        name: load_config(config_path, args.batch_size, args.split)
        for name, config_path, _ in args.variant
    }
    signatures = {
        name: config_dataset_signature(config) for name, config in configs.items()
    }
    first_signature = next(iter(signatures.values()))
    if any(signature != first_signature for signature in signatures.values()):
        raise AssertionError("all variants must use the exact same dataset/split/npoints")

    point_manifest = freeze_points_once(args, configs[args.official_name])
    points = load_plan_array(Path(args.point_plan_dir), "selected_points")
    labels = load_plan_array(Path(args.point_plan_dir), "labels")
    matrix_manifest = freeze_matrix_bank(args, labels.shape[0])
    matrices = load_plan_array(Path(args.matrix_plan_dir), "matrices")
    conditions = json.loads((Path(args.matrix_plan_dir) / "conditions.json").read_text())
    if matrices.shape[:2] != (len(conditions), labels.shape[0]):
        raise AssertionError("matrix bank shape does not match conditions/examples")

    variant_results = []
    for name, config_path, checkpoint in args.variant:
        variant_results.append(
            evaluate_variant(
                args,
                name,
                configs[name],
                config_path,
                checkpoint,
                points,
                labels,
                matrices,
                conditions,
            )
        )

    summary = {
        "schema": "pointgpt_scanobjectnn_erot_v1",
        "coordinate_convention": {
            **COORDINATE_CONVENTION,
            "selected_upright_axis": args.upright_axis,
            "override_used": bool(args.upright_axis != "z"),
            "override_evidence": args.axis_override_evidence or None,
        },
        "anti_cheat": {
            "exact_input_points_selected_once": True,
            "point_plan_manifest_sha256": point_manifest["manifest_sha256"],
            "identical_matrices_all_variants": True,
            "matrix_plan_manifest_sha256": matrix_manifest["manifest_sha256"],
            "random_scale_translation": False,
            "voting": False,
            "model_eval_mode": True,
            "dropout_disabled": True,
            "deterministic_algorithms": True,
            "rng_isolated_per_forward": True,
            "same_forward_seed_clean_and_all_angles": True,
            "post_rotation_processing": args.post_rotation_processing,
            "pca_or_upright_recanonicalization": False,
            "normals_rotated_when_present": True,
            "zero_degree_clean_replay_asserted": True,
            "pairwise_distance_check": (
                "all matrices are asserted orthonormal with det=+1; additionally "
                "all pairwise distances among %d points are checked for the first "
                "example of every batch and condition" % args.pairwise_check_points
            ),
            "haar_sampler": matrix_manifest["sampler"],
            "haar_uniformity_check": matrix_manifest["haar_uniformity_check"],
        },
        "dataset_signatures": signatures,
        "variants": variant_results,
        "load_bearing_paired_difference_in_rotation_drop": paired_variant_drop_interactions(
            variant_results, args.official_name
        ),
        "interpretation_guard": (
            "A common rotation drop is a shared limitation. Only the paired "
            "official-minus-relaxation difference bears on the variant reading; "
            "neither outcome justifies scope exclusion."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
