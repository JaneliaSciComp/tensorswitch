"""
TensorSwitch v2 - Unified Intermediate Format Architecture

Phase 5 implementation with hybrid three-tier reader strategy and
TensorStore as the unified intermediate format.

Architecture Overview:
    Layer 1: Readers (Foundation) - Convert formats → TensorStore
    Layer 2: Wrapper (TensorSwitchDataset) - Auto-detect & orchestrate
    Layer 3: Converter (DistributedConverter) - LSF/Dask processing
    Layer 4: User Interface (CLI/GUI) - User-facing

Hybrid Reader Strategy:
    Tier 1 (Native TensorStore): N5, Zarr2/3, Precomputed - ⚡ Maximum performance
    Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5 - ✅ Production ready
    Tier 3 (BIOIO Adapter): CZI, LIF, + 20 more - 📦 Broad compatibility

Usage Example:
    >>> from tensorswitch_v2.api import TensorSwitchDataset
    >>> dataset = TensorSwitchDataset("/path/to/data.tif")
    >>> ts_array = dataset.get_tensorstore_array()
    >>> metadata = dataset.get_ome_ngff_metadata()

Development Status:
    Branch: unified-architecture
    Phase: 5.1 - Design & Foundation (Week 1)
    Target: March 21, 2026 (10 weeks)

For detailed architecture, see: phase5UnifiedIntermediateFormat.md
"""

__version__ = '2.0.0-alpha'
__author__ = 'Diyi Chen'

# Public API will be exported here as components are built
# Example (when complete):
# from .api import TensorSwitchDataset, Readers, Writers
# from .core import DistributedConverter

__all__ = []
