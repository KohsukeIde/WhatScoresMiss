"""Deterministic model-forward helpers for guarded evaluation harnesses."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


def configure_deterministic_runtime() -> None:
    """Require deterministic kernels for every guarded model forward."""
    # This must be set before the first CUDA operation. It makes CUDA >= 10.2
    # GEMMs deterministic when torch's deterministic-algorithm guard is active.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_deterministic_model(model: torch.nn.Module) -> torch.nn.Module:
    """Put a model, including every Dropout/DropPath child, in eval mode."""
    configure_deterministic_runtime()
    model.eval()
    training_modules = [
        module.__class__.__name__ for module in model.modules() if module.training
    ]
    if training_modules:
        raise AssertionError(
            "evaluation model still has training-mode modules: %s"
            % ", ".join(training_modules[:5])
        )
    return model


@contextmanager
def isolated_forward_rng(seed: int, uses_cuda: bool) -> Iterator[None]:
    """Replay Python, NumPy, CPU, and CUDA RNGs without advancing outer state."""
    seed = int(seed)
    numpy_seed = seed % (2**32)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if uses_cuda else []
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(numpy_seed)
            torch.manual_seed(seed)
            if uses_cuda:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def deterministic_model_forward(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    seed: int,
    **forward_kwargs,
):
    """Run one RNG-isolated deterministic model forward."""
    configure_deterministic_model(model)
    with isolated_forward_rng(seed, uses_cuda=model_input.is_cuda):
        return model(model_input, **forward_kwargs)
