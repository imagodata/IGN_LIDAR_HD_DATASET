"""
Formatters for multi-architecture deep learning support.

This module provides formatters to convert IGN LiDAR HD patches
into architecture-specific formats for deep learning.

Supported architectures:
- PointNet++ (Set Abstraction)
- Octree-CNN / OctFormer (Hierarchical)
- Point Transformer / PCT (Attention-based)
- Sparse Convolutions (Voxel-based)
- Hybrid Models (Combinations)
"""

from .multi_arch_formatter import MultiArchitectureFormatter
from .hybrid_formatter import HybridFormatter
from .base_formatter import BaseFormatter
from .ptv3_formatter import Ptv3Formatter
from .ptv3_writer import (
    Ptv3DatasetWriter,
    HashSplitAssigner,
    assign_splits_by_tile,
    assign_splits_by_patch,
)

__all__ = [
    'MultiArchitectureFormatter',
    'HybridFormatter',
    'BaseFormatter',
    'Ptv3Formatter',
    'Ptv3DatasetWriter',
    'HashSplitAssigner',
    'assign_splits_by_tile',
    'assign_splits_by_patch',
]

__version__ = '3.1.0'
