"""Shared isolation and invariance helpers for segmentation adapters."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Iterable

import torch

from .strict_loading import StrictLoadError


_IMPORT_LOCK = threading.RLock()


def load_module_with_sys_path_isolation(
    module_path: Path,
    *,
    import_roots: Iterable[Path],
    collision_roots: Iterable[str],
    unique_name: str,
) -> ModuleType:
    """Execute one source file while restoring all temporary global imports."""
    source = Path(module_path).resolve(strict=True)
    roots = [str(Path(path).resolve(strict=True)) for path in import_roots]
    collisions = tuple(sorted(set(str(name) for name in collision_roots)))
    if not unique_name or "." in unique_name:
        raise ValueError("unique module name must be a non-empty top-level name")

    def is_collision(name: str) -> bool:
        return any(name == root or name.startswith(root + ".") for root in collisions)

    with _IMPORT_LOCK:
        original_path = list(sys.path)
        original_dont_write_bytecode = sys.dont_write_bytecode
        saved_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if is_collision(name) or name == unique_name
        }
        for name in saved_modules:
            sys.modules.pop(name, None)
        for root in reversed(roots):
            if root in sys.path:
                sys.path.remove(root)
            sys.path.insert(0, root)
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location(unique_name, source)
            if spec is None or spec.loader is None:
                raise ImportError("could not create source import specification")
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
            for name in tuple(sys.modules):
                if is_collision(name) or name == unique_name:
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)
            sys.dont_write_bytecode = original_dont_write_bytecode
    return module


def assert_all_eval(model: torch.nn.Module, family: str) -> None:
    if any(module.training for module in model.modules()):
        raise StrictLoadError("%s adapter found a module outside eval mode" % family)


def state_version_signature(
    model: torch.nn.Module,
) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            entries.append(
                (
                    kind,
                    name,
                    id(value),
                    int(value._version),
                    str(value.dtype),
                    tuple(int(size) for size in value.shape),
                    str(value.device),
                )
            )
    return tuple(entries)


def validate_segmentation_inputs(
    points: torch.Tensor,
    class_labels: torch.Tensor,
    *,
    model_dtype: torch.dtype,
    family: str,
) -> torch.Tensor:
    if not torch.is_tensor(points) or not torch.is_tensor(class_labels):
        raise TypeError("points and class_labels must be torch tensors")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("%s points must have shape (B, N, 3)" % family)
    if not torch.is_floating_point(points) or points.dtype != model_dtype:
        raise TypeError("%s point dtype must match the model dtype" % family)
    if not bool(torch.isfinite(points).all()):
        raise FloatingPointError("%s points contain a non-finite value" % family)
    labels = class_labels
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("%s class labels must have shape (B,) or (B, 1)" % family)
    if labels.dtype != torch.long:
        raise TypeError("%s class labels must use torch.long" % family)
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= 16
    ):
        raise ValueError("%s class labels must be in [0, 15]" % family)
    return labels
