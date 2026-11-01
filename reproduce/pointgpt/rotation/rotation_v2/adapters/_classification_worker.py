#!/usr/bin/env python3
"""Private JSON-line worker for one official classification snapshot."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal_eval.determinism import configure_deterministic_runtime  # noqa: E402
from rotation_v2.adapters.segmentation_common import (  # noqa: E402
    assert_all_eval,
    state_version_signature,
)
from rotation_v2.adapters.strict_loading import (  # noqa: E402
    StrictLoadError,
    file_sha256,
    load_model_checkpoint_strict,
    state_dict_sha256,
)


def _write_protocol(stream: object, value: Mapping[str, object]) -> None:
    stream.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _dependency_record(module_name: str) -> dict[str, object] | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    source = getattr(module, "__file__", None)
    if source is None:
        return {"module": module_name, "file": None, "sha256": None}
    path = Path(source).resolve()
    return {
        "module": module_name,
        "file": str(path),
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_rpc_input(path_text: object, rpc_root: Path) -> Path:
    if not isinstance(path_text, str):
        raise TypeError("RPC input path is not a string")
    path = Path(path_text).resolve(strict=True)
    if path.parent != rpc_root or not path.is_file() or path.is_symlink():
        raise ValueError("RPC input must be a regular file directly under rpc root")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("pointmae", "pcpmae"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--rpc-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.dont_write_bytecode = True
    protocol_stream = sys.stdout
    snapshot = args.snapshot.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    rpc_root = args.rpc_root.resolve(strict=True)
    try:
        config_path.relative_to(snapshot)
    except ValueError as error:
        raise ValueError("config must be inside the verified snapshot") from error
    if not rpc_root.is_dir():
        raise NotADirectoryError("RPC root is absent")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    configure_deterministic_runtime()
    original_cwd = Path.cwd()
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from tools import builder
            from utils.config import cfg_from_yaml_file

            config = cfg_from_yaml_file(str(config_path))
            if not hasattr(config, "model"):
                raise ValueError("official config has no model section")
            model = builder.model_builder(config.model)
    finally:
        os.chdir(original_cwd)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("official builder did not return a torch module")
    load_record = load_model_checkpoint_strict(
        model,
        checkpoint_path,
        allowed_container_keys=("model", "base_model"),
        allowed_prefix_rewrites=("module.",),
    )
    model = model.to(device)
    model.eval()
    assert_all_eval(model, "%s classification worker" % args.family)
    baseline_hash = state_dict_sha256(model)
    if baseline_hash != load_record.loaded_state_sha256:
        raise StrictLoadError("device transfer changed classification state bytes")
    baseline_versions = state_version_signature(model)
    model_dtype = next(model.parameters()).dtype
    dependency_roots = (
        "chamfer",
        "knn_cuda",
        "pointnet2_ops",
        "extensions.chamfer_dist",
    )
    dependency_names = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == root or name.startswith(root + ".")
                for root in dependency_roots
            )
        )
    )
    dependencies = [
        record
        for record in (_dependency_record(name) for name in dependency_names)
        if record is not None
    ]
    ready_metadata = {
        "worker_schema": "rotation_matrix_v2_official_cls_worker_v1",
        "family": args.family,
        "strict_load": load_record.to_dict(),
        "device": str(device),
        "model_dtype": str(model_dtype),
        "eval_mode_all_modules": True,
        "parameter_hash_pre_evaluation": baseline_hash,
        "official_builder": "tools.builder.model_builder(config.model)",
        "official_forward": "model(points_B_N_3)",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "timm": _package_version("timm"),
            "easydict": _package_version("easydict"),
        },
        "runtime_dependency_files": dependencies,
    }
    _write_protocol(
        protocol_stream,
        {"status": "READY", "metadata": ready_metadata},
    )

    forward_calls = 0
    for line in sys.stdin:
        request_id: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise TypeError("RPC request is not a mapping")
            request_id = request.get("request_id")
            command = request.get("command")
            if command == "forward":
                input_path = _safe_rpc_input(request.get("input_path"), rpc_root)
                seed = request.get("seed")
                if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                    raise ValueError("forward seed must be a non-negative integer")
                points_numpy = np.load(input_path, allow_pickle=False)
                if (
                    not isinstance(points_numpy, np.ndarray)
                    or points_numpy.ndim != 3
                    or points_numpy.shape[-1] != 3
                    or points_numpy.dtype != np.float32
                    or not np.all(np.isfinite(points_numpy))
                ):
                    raise ValueError("worker input must be finite float32 (B, N, 3)")
                assert_all_eval(model, "%s classification worker" % args.family)
                if state_version_signature(model) != baseline_versions:
                    raise StrictLoadError("classification state version changed")
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                points = torch.from_numpy(
                    np.array(points_numpy, dtype=np.float32, copy=True)
                ).to(device)
                if points.dtype != model_dtype:
                    points = points.to(dtype=model_dtype)
                with contextlib.redirect_stdout(sys.stderr):
                    with torch.inference_mode():
                        logits = model(points)
                if (
                    not torch.is_tensor(logits)
                    or logits.ndim != 2
                    or logits.shape[0] != points.shape[0]
                    or logits.shape[1] < 2
                    or not bool(torch.isfinite(logits).all())
                ):
                    raise StrictLoadError(
                        "official classification logits must have shape (B, C)"
                    )
                assert_all_eval(model, "%s classification worker" % args.family)
                if state_version_signature(model) != baseline_versions:
                    raise StrictLoadError(
                        "classification state identity/version changed during forward"
                    )
                forward_calls += 1
                logits_numpy = logits.detach().cpu().to(torch.float32).numpy()
                _write_protocol(
                    protocol_stream,
                    {
                        "status": "OK",
                        "request_id": request_id,
                        "dtype": "float32",
                        "shape": list(logits_numpy.shape),
                        "logits": logits_numpy.tolist(),
                        "forward_calls": forward_calls,
                    },
                )
            elif command == "metadata":
                assert_all_eval(model, "%s classification worker" % args.family)
                if state_version_signature(model) != baseline_versions:
                    raise StrictLoadError("classification state version changed")
                current_hash = state_dict_sha256(model)
                if current_hash != baseline_hash:
                    raise StrictLoadError("classification parameter hash changed")
                _write_protocol(
                    protocol_stream,
                    {
                        "status": "OK",
                        "request_id": request_id,
                        "parameter_hash_post_evaluation": current_hash,
                        "parameter_hash_invariant": True,
                        "forward_calls": forward_calls,
                    },
                )
            elif command == "close":
                _write_protocol(
                    protocol_stream,
                    {
                        "status": "CLOSED",
                        "request_id": request_id,
                        "forward_calls": forward_calls,
                    },
                )
                return 0
            else:
                raise ValueError("unknown RPC command")
        except Exception as error:
            _write_protocol(
                protocol_stream,
                {
                    "status": "ERROR",
                    "request_id": request_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
