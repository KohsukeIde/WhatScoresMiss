"""Strict adapter for the repository's frozen PointGPT-S part segmenter."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from .base import FrozenSegmentationAdapter
from .segmentation_common import (
    assert_all_eval,
    load_module_with_sys_path_isolation,
    state_version_signature,
    validate_segmentation_inputs,
)
from .strict_loading import (
    StrictLoadError,
    file_sha256,
    load_model_checkpoint_strict,
    state_dict_sha256,
)


class PointGPTSegAdapter(FrozenSegmentationAdapter):
    """Mirror the fixed R7C PointGPT-S model with exact checkpoint loading."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str | torch.device = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self._model: torch.nn.Module | None = None
        self._loaded_hash: str | None = None
        self._loaded_versions: tuple[tuple[object, ...], ...] | None = None
        self._metadata: dict[str, object] | None = None
        self._forward_calls = 0
        self._pre_forward_hash_checked = False

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError("load_frozen_system() must be called first")
        return self._model

    def _assert_invariant(self, *, full_hash: bool) -> str | None:
        assert_all_eval(self.model, "PointGPT segmentation")
        if self._loaded_versions is None:
            raise RuntimeError("frozen state version baseline is unavailable")
        if state_version_signature(self.model) != self._loaded_versions:
            raise StrictLoadError(
                "PointGPT segmentation parameter or buffer identity/version changed"
            )
        if not full_hash:
            return None
        if self._loaded_hash is None:
            raise RuntimeError("frozen state hash baseline is unavailable")
        current = state_dict_sha256(self.model)
        if current != self._loaded_hash:
            raise StrictLoadError(
                "PointGPT segmentation parameter or buffer hash changed"
            )
        return current

    @staticmethod
    def _enable_official_inplace_relu(module: torch.nn.Module) -> None:
        if "ReLU" in module.__class__.__name__:
            module.inplace = True

    def load_frozen_system(self) -> "PointGPTSegAdapter":
        if self._model is not None:
            self._assert_invariant(full_hash=True)
            return self
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("PointGPT segmentation checkpoint does not exist")
        if self.device.type != "cuda":
            raise RuntimeError(
                "the frozen PointGPT segmentation model requires CUDA because "
                "its bound constructor instantiates CUDA Chamfer modules"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")

        repo_root = Path(__file__).resolve().parents[2]
        segmentation_path = repo_root / "segmentation"
        models_path = segmentation_path / "models"
        source_path = models_path / "pt.py"
        module = load_module_with_sys_path_isolation(
            source_path,
            import_roots=(models_path, segmentation_path, repo_root),
            collision_roots=(
                "logger",
                "pointnet2_utils",
                "gpt2_seg",
                "z_order",
                "extensions",
            ),
            unique_name="rotation_v2_host_pointgpt_seg_pt",
        )
        factory = getattr(module, "get_model", None)
        if not isinstance(factory, type) or not issubclass(factory, torch.nn.Module):
            raise TypeError("PointGPT segmentation get_model class is unavailable")
        cuda_index = self.device.index
        device_context = (
            torch.cuda.device(cuda_index)
            if cuda_index is not None
            else torch.cuda.device(torch.cuda.current_device())
        )
        with device_context:
            model = factory(
                50,
                trans_dim=384,
                depth=12,
                drop_path_rate=0.1,
                num_heads=6,
                decoder_depth=4,
                group_size=32,
                num_group=128,
                prop_dim=1024,
                label_dim1=512,
                label_dim2=256,
                encoder_dims=384,
                group_mode="fps_knn",
                group_radius=0.22,
                group_voxel_grid=6,
            )
        model.apply(self._enable_official_inplace_relu)
        load_record = load_model_checkpoint_strict(
            model,
            self.checkpoint_path,
            allowed_container_keys=("model_state_dict",),
            allowed_prefix_rewrites=(),
        )
        model = model.to(self.device)
        model.eval()
        assert_all_eval(model, "PointGPT segmentation")
        moved_hash = state_dict_sha256(model)
        if moved_hash != load_record.loaded_state_sha256:
            raise StrictLoadError("device transfer changed PointGPT state bytes")
        self._model = model
        self._loaded_hash = moved_hash
        self._loaded_versions = state_version_signature(model)

        source_files = {
            "rotation_v2/adapters/base.py": (
                repo_root / "rotation_v2" / "adapters" / "base.py"
            ),
            "rotation_v2/adapters/pointgpt_seg.py": Path(__file__),
            "rotation_v2/adapters/segmentation_common.py": (
                Path(__file__).with_name("segmentation_common.py")
            ),
            "rotation_v2/adapters/strict_loading.py": (
                Path(__file__).with_name("strict_loading.py")
            ),
            "segmentation/eval_shapenetpart_r7c_correction.py": (
                segmentation_path / "eval_shapenetpart_r7c_correction.py"
            ),
            "segmentation/main.py": segmentation_path / "main.py",
            "segmentation/models/pt.py": source_path,
            "segmentation/models/gpt2_seg.py": models_path / "gpt2_seg.py",
            "segmentation/models/pointnet2_utils.py": (
                models_path / "pointnet2_utils.py"
            ),
        }
        self._metadata = {
            "adapter_schema": "rotation_matrix_v2_pointgpt_seg_adapter_v1",
            "model_family": "PointGPT-S",
            "task": "ShapeNetPart_part_segmentation",
            "complete_system_loaded": True,
            "architecture": {
                "trans_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "decoder_depth": 4,
                "group_size": 32,
                "num_group": 128,
                "prop_dim": 1024,
                "label_dim1": 512,
                "label_dim2": 256,
                "encoder_dims": 384,
                "group_mode": "fps_knn",
                "group_radius": 0.22,
                "group_voxel_grid": 6,
            },
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": load_record.checkpoint_sha256,
            "strict_load": load_record.to_dict(),
            "device": str(self.device),
            "consumed_features": ["xyz", "category_one_hot_16"],
            "input_schema": "points B,N,3 float; class_labels B int64",
            "output_schema": "B,N,50 log probabilities",
            "normal_channel": False,
            "eval_mode_all_modules": True,
            "parameter_hash_pre_evaluation": moved_hash,
            "parameter_guard": (
                "full_sha256_before_first_forward_and_after_final_forward; "
                "tensor_identity_and_version_checked_each_forward"
            ),
            "source_sha256": {
                name: file_sha256(path) for name, path in sorted(source_files.items())
            },
        }
        return self

    def forward_logits(
        self,
        points: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("load_frozen_system() must be called first")
        model_dtype = next(self.model.parameters()).dtype
        labels = validate_segmentation_inputs(
            points,
            class_labels,
            model_dtype=model_dtype,
            family="PointGPT",
        )
        self._assert_invariant(full_hash=False)
        if not self._pre_forward_hash_checked:
            self._assert_invariant(full_hash=True)
            self._pre_forward_hash_checked = True
        model_points = points.to(self.device, non_blocking=False).transpose(1, 2)
        model_labels = labels.to(self.device, non_blocking=False)
        one_hot = F.one_hot(model_labels, num_classes=16).to(dtype=model_dtype)
        with torch.inference_mode():
            logits = self.model(model_points, one_hot)
        if (
            not torch.is_tensor(logits)
            or logits.ndim != 3
            or logits.shape != (points.shape[0], points.shape[1], 50)
        ):
            raise StrictLoadError("PointGPT logits must have shape (B, N, 50)")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("PointGPT logits contain a non-finite value")
        self._assert_invariant(full_hash=False)
        self._forward_calls += 1
        return logits

    def metadata(self) -> Mapping[str, object]:
        if self._metadata is None:
            raise RuntimeError("load_frozen_system() must be called first")
        current = self._assert_invariant(full_hash=True)
        result = copy.deepcopy(self._metadata)
        result["parameter_hash_post_evaluation"] = current
        result["parameter_hash_invariant"] = current == self._loaded_hash
        result["forward_calls"] = self._forward_calls
        return result
