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
"""

from .base import BaseWriter

__all__ = ['BaseWriter']
