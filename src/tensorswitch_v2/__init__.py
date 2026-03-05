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
    Tier 1 (Native TensorStore): N5, Zarr2/3, Precomputed - Maximum performance
    Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5, CZI - Production ready
    Tier 3 (BIOIO Adapter): LIF, + 20 more - Broad compatibility (Python plugins)
    Tier 4 (Bio-Formats): 150+ formats - Maximum compatibility (Java-backed)

Usage Example:
    >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers, Writers
    >>> dataset = TensorSwitchDataset("/path/to/data.tif")
    >>> ts_array = dataset.get_tensorstore_array()
    >>> metadata = dataset.get_ome_ngff_metadata()

CLI Usage:
    $ python -m tensorswitch_v2 -i input.tif -o output.zarr
    $ python -m tensorswitch_v2 --auto_multiscale -i dataset.zarr/s0 -o dataset.zarr
    $ tensorswitch-v2 -i input.tif -o output.zarr  # After pip install

For detailed documentation, see: src/tensorswitch_v2/README.md
"""

__version__ = '2.0.0'
__author__ = 'Diyi Chen'

# Public API exports
from .api import TensorSwitchDataset, Readers, Writers

__all__ = [
    '__version__',
    'TensorSwitchDataset',
    'Readers',
    'Writers',
]
