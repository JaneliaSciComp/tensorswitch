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
- Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5, CZI - Reuse existing code
- Tier 3 (BIOIO Adapter): LIF, + 20 more formats - Broad compatibility (Python plugins)
- Tier 4 (Bio-Formats): 150+ formats - Maximum compatibility (Java-backed)

Public API:
    Tier 1 (Native TensorStore):
    - N5Reader: Native TensorStore N5 reader
    - Zarr3Reader: Native TensorStore Zarr v3 reader
    - Zarr2Reader: Native TensorStore Zarr v2 reader
    - PrecomputedReader: Neuroglancer Precomputed reader

    Tier 2 (Custom Optimized):
    - TiffReader: TIFF reader using load_tiff_stack()
    - ND2Reader: Nikon ND2 reader using load_nd2_stack()
    - IMSReader: Imaris IMS reader using load_ims_stack()
    - HDF5Reader: Generic HDF5 reader
    - CZIReader: Zeiss CZI reader using load_czi_stack() (multi-view support)

    Tier 3 (BIOIO Adapter):
    - BIOIOReader: BIOIO adapter for 20+ formats (LIF, etc.) - Python plugins

    Tier 4 (Bio-Formats):
    - BioFormatsReader: Bio-Formats Java backend for 150+ formats

    Base:
    - BaseReader: Abstract base class for all readers
"""

from .base import BaseReader, DaskReader

# Tier 1: Native TensorStore
from .n5 import N5Reader
from .zarr import Zarr3Reader, Zarr2Reader
from .precomputed import PrecomputedReader

# Tier 2: Custom Optimized
from .tiff import TiffReader
from .nd2 import ND2Reader
from .ims import IMSReader
from .hdf5 import HDF5Reader
from .czi import CZIReader

# Tier 3: BIOIO Adapter (Python plugins)
from .bioio_adapter import BIOIOReader

# Tier 4: Bio-Formats (Java-backed, 150+ formats)
from .bioformats import BioFormatsReader

__all__ = [
    # Base
    'BaseReader', 'DaskReader',
    # Tier 1
    'N5Reader', 'Zarr3Reader', 'Zarr2Reader', 'PrecomputedReader',
    # Tier 2
    'TiffReader', 'ND2Reader', 'IMSReader', 'HDF5Reader', 'CZIReader',
    # Tier 3
    'BIOIOReader',
    # Tier 4
    'BioFormatsReader',
]
