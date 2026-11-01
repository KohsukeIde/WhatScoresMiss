"""Binding adapter contract for all rotation_matrix_v2 model systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch


class FrozenSystemAdapter(ABC):
    """Thin interface around one complete frozen evaluation system.

    Implementations must fail closed on checkpoint container, key, or shape
    mismatch. They must set every module to eval mode, declare consumed input
    features in metadata, and prove parameter-state hash invariance across
    forwards.
    """

    @abstractmethod
    def load_frozen_system(self) -> "FrozenSystemAdapter":
        """Build and strictly load the complete model plus readout."""
        raise NotImplementedError

    @abstractmethod
    def forward_logits(self, points: torch.Tensor) -> torch.Tensor:
        """Return only logits for already frozen, model-ready points."""
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Mapping[str, object]:
        """Run the post-evaluation hash gate and return frozen provenance."""
        raise NotImplementedError


class FrozenSegmentationAdapter(ABC):
    """Frozen ShapeNetPart system with an explicit category-label input."""

    @abstractmethod
    def load_frozen_system(self) -> "FrozenSegmentationAdapter":
        """Build and strictly load the complete 50-part model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Return the loaded module for clean-gate inspection."""
        raise NotImplementedError

    @abstractmethod
    def forward_logits(
        self,
        points: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits with shape ``(B, N, 50)``."""
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Mapping[str, object]:
        """Return provenance after enforcing post-forward state invariance."""
        raise NotImplementedError
