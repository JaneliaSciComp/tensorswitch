"""
TensorSwitch Phase 5 - Core Processing Layer

This module contains the core processing components for data conversion
and distributed processing coordination.

Architecture Layer: 3 (Processing)
- Format-agnostic conversion logic
- LSF multi-job mode support
- Dask single-job mode support
- Chunk iteration and coordination

Public API:
    - DistributedConverter: Format-agnostic converter with distributed support
"""

from .converter import DistributedConverter

__all__ = ['DistributedConverter']
