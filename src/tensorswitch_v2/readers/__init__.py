"""
TensorSwitch Phase 5 - Reader Layer (Foundation)

This module contains all format-specific readers that convert various
scientific imaging formats into TensorStore arrays.

Architecture Layer: 1 (Foundation)
- Independent, no dependencies on other TensorSwitch layers
- Each reader converts its format → TensorStore spec/array
- Can be used standalone without wrapper

Tier Strategy:
- Tier 1 (Native TensorStore): N5, Zarr2/3, Precomputed - Maximum performance
- Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5 - Reuse existing code
- Tier 3 (BIOIO Adapter): CZI, LIF, + 20 more formats - Broad compatibility

Public API:
    - BaseReader: Abstract base class for all readers
    - N5Reader: Tier 1 - Native TensorStore N5 reader
    - PrecomputedReader: Tier 1 - Neuroglancer Precomputed reader
"""

from .base import BaseReader
from .n5 import N5Reader
from .precomputed import PrecomputedReader

__all__ = ['BaseReader', 'N5Reader', 'PrecomputedReader']
