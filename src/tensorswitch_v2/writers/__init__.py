"""
TensorSwitch Phase 5 - Writer Layer (Output Processing)

This module contains all format-specific writers that convert TensorStore
arrays into various output formats (Zarr3, Zarr2, N5, etc.).

Architecture Layer: 3 (Processing - Depends on Readers/Wrapper)
- Format-specific output implementations
- Accept TensorStore arrays from any source
- No knowledge of input format (reader-agnostic)

Writer Strategy:
- Writers ONLY query TensorStore array properties
- Writers NEVER ask about input format
- Complete decoupling from readers

Output Formats:
- Zarr3: Primary output format with sharding support
- Zarr2: Legacy Zarr format for compatibility
- N5: Rechunking and metadata conversion

Public API:
    - BaseWriter: Abstract base class for all writers
    - Zarr3Writer: Zarr v3 with sharding and OME-NGFF v0.5
    - Zarr2Writer: Zarr v2 with OME-NGFF v0.4 (legacy)
    - N5Writer: N5 format for Java tools
"""

from .base import BaseWriter
from .zarr3 import Zarr3Writer
from .zarr2 import Zarr2Writer
from .n5 import N5Writer

__all__ = [
    'BaseWriter',
    'Zarr3Writer',
    'Zarr2Writer',
    'N5Writer',
]
