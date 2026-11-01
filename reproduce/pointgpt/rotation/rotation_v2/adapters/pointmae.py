"""Pinned official Point-MAE classification adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from .official_cls_subprocess import OfficialClassificationSubprocessAdapter


class PointMAEAdapter(OfficialClassificationSubprocessAdapter):
    """Build and run one official Point-MAE classifier in its own process."""

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        snapshots_root: Path | None = None,
        device: str | torch.device = "cuda",
        forward_seed: int = 0,
        ready_timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            family="Point-MAE",
            snapshot_name="pointmae",
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            snapshots_root=snapshots_root,
            device=device,
            forward_seed=forward_seed,
            ready_timeout_seconds=ready_timeout_seconds,
        )
