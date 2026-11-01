"""R7C-style strict checkpoint helpers shared by v2 adapters."""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


class StrictLoadError(RuntimeError):
    """Raised when a checkpoint cannot be proven to match the target model."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _framed_update(
    digest: "hashlib._Hash",
    label: str,
    payload: bytes,
) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(16, "big"))
    digest.update(payload)


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash every parameter and persistent buffer by key, dtype, shape, bytes."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise StrictLoadError("model state contains a non-tensor value")
        if value.layout != torch.strided or value.device.type == "meta":
            raise StrictLoadError("model state tensor layout cannot be hashed")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "shape": [int(size) for size in tensor.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _framed_update(digest, key, header)
        _framed_update(digest, key + ":bytes", raw)
    return digest.hexdigest()


def load_torch_checkpoint_cpu(path: Path) -> tuple[object, str, str]:
    """Load on CPU with weights_only, falling back only for old Torch APIs."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint file does not exist: %s" % path)
    before_hash = file_sha256(checkpoint_path)
    if "weights_only" in inspect.signature(torch.load).parameters:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            load_mode = "weights_only=True"
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
                dict.fromkeys(
                    [scalar_constructor, np.dtype, *numpy_scalar_types]
                )
            )
            with safe_globals(allowlist):
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            load_mode = "weights_only=True+numpy_scalar_safe_globals"
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_mode = "legacy_torch_without_weights_only_argument"
    after_hash = file_sha256(checkpoint_path)
    if before_hash != after_hash:
        raise StrictLoadError("checkpoint changed while it was being loaded")
    return checkpoint, load_mode, before_hash


def extract_state_dict(
    checkpoint: object,
    *,
    allowed_container_keys: Sequence[str],
) -> tuple[Mapping[str, torch.Tensor], str, list[str]]:
    """Select exactly one allowlisted state container and reject ambiguity."""
    if not isinstance(checkpoint, Mapping):
        raise StrictLoadError("checkpoint container is not a mapping")
    top_level_keys = sorted(str(key) for key in checkpoint)
    candidates = [
        key
        for key in allowed_container_keys
        if key in checkpoint and checkpoint[key] is not None
    ]
    if len(candidates) != 1:
        raise StrictLoadError(
            "checkpoint must contain exactly one state container from %s; "
            "found %s" % (tuple(allowed_container_keys), candidates)
        )
    container_key = candidates[0]
    state = checkpoint[container_key]
    if not isinstance(state, Mapping) or not state:
        raise StrictLoadError("checkpoint state container is empty or invalid")
    if any(not isinstance(key, str) for key in state):
        raise StrictLoadError("checkpoint state keys must all be strings")
    non_tensors = sorted(
        key for key, value in state.items() if not torch.is_tensor(value)
    )
    if non_tensors:
        raise StrictLoadError(
            "checkpoint state contains non-tensors: %s" % non_tensors[:10]
        )
    return state, container_key, top_level_keys


@dataclass(frozen=True)
class StrictLoadRecord:
    checkpoint_sha256: str
    torch_load_mode: str
    state_dict_source: str
    top_level_keys: tuple[str, ...]
    checkpoint_key_count: int
    model_key_count: int
    source_key_list_sha256: str
    model_key_list_sha256: str
    applied_prefix_rewrites: Mapping[str, int]
    allowed_prefix_rewrites: tuple[str, ...]
    shape_mismatches: tuple[Mapping[str, object], ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    strict: bool
    loaded_state_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _key_list_sha256(keys: Sequence[str]) -> str:
    payload = (
        json.dumps(
            list(keys),
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def map_state_dict_keys(
    source_state: Mapping[str, torch.Tensor],
    target_keys: Sequence[str],
    *,
    allowed_prefix_rewrites: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Map keys using only unique, explicitly allowlisted prefix stripping."""
    prefixes = tuple(str(prefix) for prefix in allowed_prefix_rewrites)
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("allowed prefix rewrites must be unique")
    if any(not prefix or not prefix.endswith(".") for prefix in prefixes):
        raise ValueError("each allowed prefix rewrite must be non-empty and end in '.'")
    target_set = set(target_keys)
    mapped: dict[str, torch.Tensor] = {}
    rewrite_counts = {"none": 0, **{prefix: 0 for prefix in prefixes}}
    for source_key, value in source_state.items():
        candidates: list[tuple[str, str]] = []
        if source_key in target_set:
            candidates.append(("none", source_key))
        for prefix in prefixes:
            if source_key.startswith(prefix):
                candidate = source_key[len(prefix) :]
                if candidate in target_set:
                    candidates.append((prefix, candidate))
        if len(candidates) != 1:
            raise StrictLoadError(
                "checkpoint key has no unique allowlisted target: %s" % source_key
            )
        rewrite, target_key = candidates[0]
        if target_key in mapped:
            raise StrictLoadError(
                "checkpoint key rewrite collision at target: %s" % target_key
            )
        mapped[target_key] = value
        rewrite_counts[rewrite] += 1
    return mapped, rewrite_counts


def strict_load_state_dict(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    allowed_prefix_rewrites: Sequence[str] = ("module.",),
    checkpoint_sha256: str = "in-memory",
    torch_load_mode: str = "in-memory",
    state_dict_source: str = "in-memory",
    top_level_keys: Sequence[str] = (),
) -> StrictLoadRecord:
    """Require full key-set and shape equality before calling strict=True.

    This follows the local R7C gate at
    ``segmentation/eval_shapenetpart_r7c_correction.py:475-513``.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    target_state = model.state_dict()
    model_keys = tuple(sorted(target_state))
    source_keys = tuple(sorted(source_state))
    mapped_state, rewrite_counts = map_state_dict_keys(
        source_state,
        model_keys,
        allowed_prefix_rewrites=allowed_prefix_rewrites,
    )
    mapped_keys = set(mapped_state)
    model_key_set = set(model_keys)
    missing_keys = tuple(sorted(model_key_set - mapped_keys))
    unexpected_keys = tuple(sorted(mapped_keys - model_key_set))
    shared_keys = sorted(model_key_set & mapped_keys)
    shape_mismatches = tuple(
        {
            "key": key,
            "model_shape": [int(size) for size in target_state[key].shape],
            "checkpoint_shape": [int(size) for size in mapped_state[key].shape],
        }
        for key in shared_keys
        if tuple(target_state[key].shape) != tuple(mapped_state[key].shape)
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        raise StrictLoadError(
            "strict checkpoint compatibility failed: missing=%d "
            "unexpected=%d shape_mismatches=%d"
            % (
                len(missing_keys),
                len(unexpected_keys),
                len(shape_mismatches),
            )
        )
    incompatible = model.load_state_dict(mapped_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise StrictLoadError("torch strict loader returned incompatible keys")
    model.eval()
    if any(module.training for module in model.modules()):
        raise StrictLoadError("eval() did not reach every model module")
    loaded_hash = state_dict_sha256(model)
    return StrictLoadRecord(
        checkpoint_sha256=checkpoint_sha256,
        torch_load_mode=torch_load_mode,
        state_dict_source=state_dict_source,
        top_level_keys=tuple(top_level_keys),
        checkpoint_key_count=len(source_keys),
        model_key_count=len(model_keys),
        source_key_list_sha256=_key_list_sha256(source_keys),
        model_key_list_sha256=_key_list_sha256(model_keys),
        applied_prefix_rewrites=rewrite_counts,
        allowed_prefix_rewrites=tuple(allowed_prefix_rewrites),
        shape_mismatches=shape_mismatches,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        strict=True,
        loaded_state_sha256=loaded_hash,
    )


def load_model_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    allowed_container_keys: Sequence[str] = ("model", "base_model"),
    allowed_prefix_rewrites: Sequence[str] = ("module.",),
) -> StrictLoadRecord:
    """Read one existing PointGPT checkpoint and load it with exact coverage."""
    checkpoint, load_mode, checkpoint_hash = load_torch_checkpoint_cpu(checkpoint_path)
    state, source, top_level_keys = extract_state_dict(
        checkpoint,
        allowed_container_keys=allowed_container_keys,
    )
    return strict_load_state_dict(
        model,
        state,
        allowed_prefix_rewrites=allowed_prefix_rewrites,
        checkpoint_sha256=checkpoint_hash,
        torch_load_mode=load_mode,
        state_dict_source=source,
        top_level_keys=top_level_keys,
    )
