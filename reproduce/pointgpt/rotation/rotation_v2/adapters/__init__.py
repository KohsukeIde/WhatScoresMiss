"""Model adapters for the rotation_matrix_v2 evaluation systems."""

from .base import FrozenSegmentationAdapter, FrozenSystemAdapter
from .pcpmae import PCPMAEAdapter
from .pointgpt import PointGPTAdapter
from .pointgpt_seg import PointGPTSegAdapter
from .pointmae import PointMAEAdapter
from .pointmae_seg import PointMAESegAdapter

__all__ = [
    "FrozenSegmentationAdapter",
    "FrozenSystemAdapter",
    "PCPMAEAdapter",
    "PointGPTAdapter",
    "PointGPTSegAdapter",
    "PointMAEAdapter",
    "PointMAESegAdapter",
]
