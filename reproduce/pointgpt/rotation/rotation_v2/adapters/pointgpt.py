"""PointGPT adapter using the repository builder and a strict v2 loader."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .base import FrozenSystemAdapter
from .strict_loading import (
    StrictLoadError,
    file_sha256,
    load_model_checkpoint_strict,
    state_dict_sha256,
)


class PointGPTAdapter(FrozenSystemAdapter):
    """Wrap the existing PointGPT builder without its broad key replacement.

    Model construction mirrors ``tools/builder.py:40-42``. Checkpoint
    container names mirror ``tools/builder.py:166-183``, but key rewriting is
    restricted to the declared prefix allowlist and exact key plus shape
    equality. Inference mirrors ``models/PointGPT.py:914-978`` with
    ``compute_recon=False``.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str | torch.device = "cpu",
        allowed_prefix_rewrites: Sequence[str] = ("module.",),
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.allowed_prefix_rewrites = tuple(allowed_prefix_rewrites)
        self._model: torch.nn.Module | None = None
        self._loaded_state_hash: str | None = None
        self._loaded_state_versions: tuple[tuple[object, ...], ...] | None = None
        self._metadata: dict[str, object] | None = None
        self._forward_calls = 0
        self._pre_evaluation_hash_checked = False

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError("load_frozen_system() must be called first")
        return self._model

    def _assert_eval(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise StrictLoadError("PointGPT adapter found a module outside eval mode")

    def _assert_parameter_hash_invariant(self) -> str:
        if self._loaded_state_hash is None:
            raise RuntimeError("frozen parameter baseline is unavailable")
        current_hash = state_dict_sha256(self.model)
        if current_hash != self._loaded_state_hash:
            raise StrictLoadError(
                "PointGPT parameter or buffer hash changed after strict loading"
            )
        return current_hash

    def _state_version_signature(self) -> tuple[tuple[object, ...], ...]:
        entries: list[tuple[object, ...]] = []
        for kind, values in (
            ("parameter", self.model.named_parameters()),
            ("buffer", self.model.named_buffers()),
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

    def _assert_state_versions_invariant(self) -> None:
        if self._loaded_state_versions is None:
            raise RuntimeError("frozen state version baseline is unavailable")
        if self._state_version_signature() != self._loaded_state_versions:
            raise StrictLoadError(
                "PointGPT parameter or buffer identity/version changed during "
                "evaluation"
            )

    def load_frozen_system(self) -> "PointGPTAdapter":
        if self._model is not None:
            self._assert_eval()
            self._assert_parameter_hash_invariant()
            return self
        if not self.config_path.is_file():
            raise FileNotFoundError("PointGPT config file does not exist")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("PointGPT checkpoint file does not exist")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")

        config_hash_before = file_sha256(self.config_path)
        from tools import builder
        from utils.config import cfg_from_yaml_file

        config = cfg_from_yaml_file(str(self.config_path))
        model = builder.model_builder(config.model)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("PointGPT builder did not return a torch module")
        load_record = load_model_checkpoint_strict(
            model,
            self.checkpoint_path,
            allowed_container_keys=("model", "base_model"),
            allowed_prefix_rewrites=self.allowed_prefix_rewrites,
        )
        if file_sha256(self.config_path) != config_hash_before:
            raise StrictLoadError("PointGPT config changed during model construction")
        model = model.to(self.device)
        model.eval()
        if any(module.training for module in model.modules()):
            raise StrictLoadError("eval() did not reach every PointGPT module")
        self._model = model
        moved_hash = state_dict_sha256(model)
        if moved_hash != load_record.loaded_state_sha256:
            raise StrictLoadError("moving PointGPT to its device changed state bytes")
        self._loaded_state_hash = moved_hash
        self._loaded_state_versions = self._state_version_signature()

        repo_root = Path(__file__).resolve().parents[2]
        source_files = {
            "rotation_v2/adapters/base.py": (
                repo_root / "rotation_v2" / "adapters" / "base.py"
            ),
            "rotation_v2/adapters/pointgpt.py": (
                repo_root / "rotation_v2" / "adapters" / "pointgpt.py"
            ),
            "rotation_v2/adapters/strict_loading.py": (
                repo_root / "rotation_v2" / "adapters" / "strict_loading.py"
            ),
            "tools/builder.py": repo_root / "tools" / "builder.py",
            "models/GPT.py": repo_root / "models" / "GPT.py",
            "models/PointGPT.py": repo_root / "models" / "PointGPT.py",
            "models/build.py": repo_root / "models" / "build.py",
            "models/z_order.py": repo_root / "models" / "z_order.py",
            "utils/config.py": repo_root / "utils" / "config.py",
            "utils/registry.py": repo_root / "utils" / "registry.py",
        }
        self._metadata = {
            "adapter_schema": "rotation_matrix_v2_pointgpt_adapter_v1",
            "model_family": "PointGPT",
            "complete_system_loaded": True,
            "config_path": str(self.config_path.resolve()),
            "config_sha256": config_hash_before,
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": load_record.checkpoint_sha256,
            "device": str(self.device),
            "consumed_features": ["xyz"],
            "input_schema": "B,N,3 float XYZ",
            "forward_contract": "model(points, compute_recon=False)[0]",
            "eval_mode_all_modules": True,
            "parameter_hash_pre_evaluation": moved_hash,
            "parameter_guard": (
                "full_sha256_before_first_forward_and_in_metadata_after_final_"
                "forward; tensor_identity_and_version_checked_each_forward"
            ),
            "strict_load": load_record.to_dict(),
            "source_sha256": {
                name: file_sha256(path) for name, path in sorted(source_files.items())
            },
        }
        return self

    def forward_logits(self, points: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("load_frozen_system() must be called first")
        if not torch.is_tensor(points):
            raise TypeError("points must be a torch tensor")
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("PointGPT points must have shape (B, N, 3)")
        if not torch.is_floating_point(points):
            raise TypeError("PointGPT points must have floating dtype")
        if not bool(torch.isfinite(points).all()):
            raise FloatingPointError("PointGPT points contain a non-finite value")
        model_dtype = next(self.model.parameters()).dtype
        if points.dtype != model_dtype:
            raise TypeError(
                "point dtype %s does not match model dtype %s"
                % (points.dtype, model_dtype)
            )
        self._assert_eval()
        self._assert_state_versions_invariant()
        if not self._pre_evaluation_hash_checked:
            self._assert_parameter_hash_invariant()
            self._pre_evaluation_hash_checked = True
        model_points = points.to(self.device, non_blocking=False)
        with torch.inference_mode():
            output = self.model(model_points, compute_recon=False)
        if not isinstance(output, tuple) or len(output) != 2:
            raise StrictLoadError(
                "PointGPT forward did not return the expected two-item tuple"
            )
        logits = output[0]
        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise StrictLoadError("PointGPT logits must have shape (B, C)")
        if logits.shape[0] != points.shape[0]:
            raise StrictLoadError("PointGPT logits changed the batch dimension")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("PointGPT logits contain a non-finite value")
        self._assert_eval()
        self._assert_state_versions_invariant()
        self._forward_calls += 1
        return logits

    def metadata(self) -> Mapping[str, object]:
        if self._metadata is None:
            raise RuntimeError("load_frozen_system() must be called first")
        self._assert_eval()
        self._assert_state_versions_invariant()
        current_hash = self._assert_parameter_hash_invariant()
        result = copy.deepcopy(self._metadata)
        result["parameter_hash_post_evaluation"] = current_hash
        result["parameter_hash_invariant"] = current_hash == self._loaded_state_hash
        result["forward_calls"] = self._forward_calls
        result["post_evaluation_hash_after_forward_calls"] = self._forward_calls
        return result
