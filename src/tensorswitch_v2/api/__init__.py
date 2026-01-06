"""
TensorSwitch Phase 5 - Public API Layer

This module provides the user-facing API for TensorSwitch v2.

Architecture Layer: 2 (Wrapper/API)
- TensorSwitchDataset: Unified interface for all formats
- Readers: Static factory for creating readers (Day 4-5)
- Writers: Static factory for creating writers (Day 4-5)

Public Classes:
    - TensorSwitchDataset: Format-agnostic data wrapper
"""

from .dataset import TensorSwitchDataset

# Readers and Writers factories will be added in Day 4-5
# from .readers import Readers
# from .writers import Writers

__all__ = ['TensorSwitchDataset']
