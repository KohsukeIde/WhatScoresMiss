"""Strict official Point-MAE ShapeNetPart segmentation adapter."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from rotation_v2.audit_pointmae_partseg_ckpt import (
    FULL_STATUS,
    audit_checkpoint,
    official_expected_state_shapes,
)
from rotation_v2.snapshot_manifest import verify_snapshot

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


OFFICIAL_MODEL_URL = (
    "https://github.com/Pang-Yatian/Point-MAE/blob/"
    "7445a68068d707da6d959b66773bfef84c4e3e32/"
    "segmentation/models/pt.py#L1-L270"
)


def _assert_exact_shapes(
    model: torch.nn.Module,
    expected: Mapping[str, tuple[int, ...]],
    *,
    description: str,
) -> None:
    actual = model.state_dict()
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(actual)
        if tuple(actual[key].shape) != tuple(expected[key])
    )
    if missing or unexpected or mismatched:
        raise StrictLoadError(
            "%s template mismatch: missing=%d unexpected=%d shapes=%d"
            % (description, len(missing), len(unexpected), len(mismatched))
        )


class PointMAESegAdapter(FrozenSegmentationAdapter):
    """Run the pinned official model through an isolated in-process import.

    The public pinned class has 192 active state tensors. The official release
    checkpoint has six additional ``conv_fuse`` tensors from the segmentation
    model lineage. The adapter first proves the 192-key public class template,
    adds that exact unused compatibility module, then strictly loads all 198
    released tensors without any key rewrite.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        snapshots_root: Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.snapshots_root = (
            Path(snapshots_root) if snapshots_root is not None else None
        )
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
        assert_all_eval(self.model, "Point-MAE segmentation")
        if self._loaded_versions is None:
            raise RuntimeError("frozen state version baseline is unavailable")
        if state_version_signature(self.model) != self._loaded_versions:
            raise StrictLoadError(
                "Point-MAE parameter or buffer identity/version changed"
            )
        if not full_hash:
            return None
        if self._loaded_hash is None:
            raise RuntimeError("frozen state hash baseline is unavailable")
        current = state_dict_sha256(self.model)
        if current != self._loaded_hash:
            raise StrictLoadError("Point-MAE parameter or buffer hash changed")
        return current

    def load_frozen_system(self) -> "PointMAESegAdapter":
        if self._model is not None:
            self._assert_invariant(full_hash=True)
            return self
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Point-MAE checkpoint file does not exist")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")

        snapshot = verify_snapshot(
            "pointmae",
            snapshots_root=self.snapshots_root,
        )
        snapshot_path = Path(str(snapshot["path"]))
        segmentation_path = snapshot_path / "segmentation"
        models_path = segmentation_path / "models"
        source_path = models_path / "pt.py"
        module = load_module_with_sys_path_isolation(
            source_path,
            import_roots=(models_path, segmentation_path, snapshot_path),
            collision_roots=("logger", "pointnet2_utils"),
            unique_name="rotation_v2_official_pointmae_seg_pt",
        )
        factory = getattr(module, "get_model", None)
        if not isinstance(factory, type) or not issubclass(factory, torch.nn.Module):
            raise TypeError("official Point-MAE get_model class is unavailable")
        model = factory(50)

        backbone, head = official_expected_state_shapes()
        released_expected = {**backbone, **head}
        public_expected = {
            key: shape
            for key, shape in released_expected.items()
            if not key.startswith("conv_fuse.")
        }
        _assert_exact_shapes(
            model,
            public_expected,
            description="pinned public Point-MAE segmentation model",
        )
        model.conv_fuse = torch.nn.Sequential(
            torch.nn.Conv1d(1152, 1024, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )
        _assert_exact_shapes(
            model,
            released_expected,
            description="released Point-MAE segmentation model",
        )

        audit = audit_checkpoint(self.checkpoint_path)
        if (
            audit.get("verdict", {}).get("status") != FULL_STATUS
            or audit.get("verdict", {}).get("full_finetuned_head") is not True
        ):
            raise StrictLoadError(
                "Point-MAE checkpoint failed the exact released-template audit"
            )
        load_record = load_model_checkpoint_strict(
            model,
            self.checkpoint_path,
            allowed_container_keys=("model_state_dict",),
            allowed_prefix_rewrites=(),
        )
        model = model.to(self.device)
        model.eval()
        assert_all_eval(model, "Point-MAE segmentation")
        moved_hash = state_dict_sha256(model)
        if moved_hash != load_record.loaded_state_sha256:
            raise StrictLoadError("device transfer changed Point-MAE state bytes")
        self._model = model
        self._loaded_hash = moved_hash
        self._loaded_versions = state_version_signature(model)

        repo_root = Path(__file__).resolve().parents[2]
        source_files = {
            "rotation_v2/adapters/base.py": (
                repo_root / "rotation_v2" / "adapters" / "base.py"
            ),
            "rotation_v2/adapters/pointmae_seg.py": Path(__file__),
            "rotation_v2/adapters/segmentation_common.py": (
                Path(__file__).with_name("segmentation_common.py")
            ),
            "rotation_v2/adapters/strict_loading.py": (
                Path(__file__).with_name("strict_loading.py")
            ),
            "rotation_v2/audit_pointmae_partseg_ckpt.py": (
                repo_root / "rotation_v2" / "audit_pointmae_partseg_ckpt.py"
            ),
            "snapshot/segmentation/models/pt.py": source_path,
            "snapshot/segmentation/models/pointnet2_utils.py": (
                models_path / "pointnet2_utils.py"
            ),
        }
        self._metadata = {
            "adapter_schema": "rotation_matrix_v2_pointmae_seg_adapter_v1",
            "model_family": "Point-MAE",
            "task": "ShapeNetPart_part_segmentation",
            "complete_system_loaded": True,
            "official_model_reference": OFFICIAL_MODEL_URL,
            "snapshot": snapshot,
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": load_record.checkpoint_sha256,
            "checkpoint_release_metadata": audit["release_container_metadata"],
            "released_template": {
                "state_key_count": 198,
                "public_active_key_count": 192,
                "compatibility_conv_fuse_key_count": 6,
                "compatibility_conv_fuse_consumed_by_pinned_forward": False,
            },
            "import_isolation": {
                "mode": "in_process_sys_path_and_sys_modules_restoration",
                "subprocess_required": False,
                "collision_roots": ["logger", "pointnet2_utils"],
                "reason": (
                    "all collision-prone imports are restored immediately "
                    "after executing the pinned source"
                ),
            },
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
            family="Point-MAE",
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
            raise StrictLoadError("Point-MAE logits must have shape (B, N, 50)")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Point-MAE logits contain a non-finite value")
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
