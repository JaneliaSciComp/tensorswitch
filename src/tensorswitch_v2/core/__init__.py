"""
TensorSwitch Phase 5 - Core Processing Layer

This module contains the core processing components for data conversion
and distributed processing coordination.

Architecture Layer: 3 (Processing)
- Format-agnostic conversion logic
- LSF multi-job mode support
- Dask single-job mode support
- Chunk iteration and coordination
- Parallel downsampling from s0
- Batch processing with LSF job arrays

Public API:
    - DistributedConverter: Format-agnostic converter with distributed support
    - Downsampler: Single-level downsampling from s0 with cumulative factors
    - downsample_level: Convenience function for single-level downsampling
    - calculate_cumulative_factors: Calculate cumulative factors for a target level
    - BatchConverter: Batch processing for multiple files
"""

from .converter import DistributedConverter
from .downsampler import Downsampler, downsample_level, calculate_cumulative_factors
from .pyramid import PyramidPlanner, create_pyramid_parallel
from .batch import BatchConverter, detect_input_mode, discover_files

__all__ = [
    'DistributedConverter',
    'Downsampler',
    'downsample_level',
    'calculate_cumulative_factors',
    'PyramidPlanner',
    'create_pyramid_parallel',
    'BatchConverter',
    'detect_input_mode',
    'discover_files',
]
