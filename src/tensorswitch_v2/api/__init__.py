"""
TensorSwitch Phase 5 - Public API Layer

This module provides the user-facing API for TensorSwitch v2.

Architecture Layer: 2 (Wrapper/API)
- TensorSwitchDataset: Unified interface for all formats
- Readers: Static factory for creating readers
- Writers: Static factory for creating writers

Public Classes:
    - TensorSwitchDataset: Format-agnostic data wrapper
    - Readers: Static factory for reader selection
    - Writers: Static factory for writer selection

Example Usage:
    >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers, Writers
    >>>
    >>> # Option 1: Auto-detect format
    >>> reader = Readers.auto_detect("/path/to/data.tif")
    >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)
    >>>
    >>> # Option 2: Explicit reader
    >>> reader = Readers.n5("/path/to/data.n5")
    >>> dataset = TensorSwitchDataset("/path/to/data.n5", reader=reader)
    >>>
    >>> # Get TensorStore array
    >>> ts_array = dataset.get_tensorstore_array(mode='open')
    >>>
    >>> # Create writer
    >>> writer = Writers.zarr3("/path/to/output.zarr")
"""

from .dataset import TensorSwitchDataset
from .readers import Readers
from .writers import Writers

__all__ = ['TensorSwitchDataset', 'Readers', 'Writers']
