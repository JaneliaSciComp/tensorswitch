"""
Readers factory class with static methods for creating format-specific readers.

Provides both auto-detection and explicit reader selection via static methods.
"""

from typing import Optional
from ..readers.base import BaseReader


class Readers:
    """
    Static factory class for creating format-specific readers.

    Provides two usage patterns:
    1. Auto-detection: Intelligent tier selection based on file extension
    2. Explicit selection: Direct reader creation via static methods

    Architecture Layer: 2 (API/Factory)
    - Orchestrates reader selection
    - Implements hybrid three-tier strategy
    - Provides clean API for users

    Hybrid Three-Tier Strategy:
        Tier 1 (Native TensorStore): N5, Zarr2/3, Precomputed
            → Maximum performance, zero conversion overhead
        Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5
            → Reuse existing code, minimal overhead
        Tier 3 (BIOIO Adapter): CZI, LIF, + 20 more formats
            → Broad compatibility via BIOIO ecosystem

    Example (Auto-detection):
        >>> from tensorswitch_v2.api import Readers, TensorSwitchDataset
        >>> reader = Readers.auto_detect("/path/to/data.tif")
        >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)

    Example (Explicit reader):
        >>> reader = Readers.n5("/path/to/data.n5")
        >>> dataset = TensorSwitchDataset("/path/to/data.n5", reader=reader)

    Example (Force BIOIO for testing):
        >>> reader = Readers.bioio("/path/to/data.tif")
        >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)
    """

    @staticmethod
    def auto_detect(path: str) -> BaseReader:
        """
        Automatically detect format and select optimal reader tier.

        Implements intelligent tier selection:
        1. Check for Tier 1 formats (native TensorStore) → fastest
        2. Check for Tier 2 formats (custom optimized) → production-ready
        3. Fallback to Tier 3 (BIOIO adapter) → broad compatibility

        Args:
            path: Path to input data (local, HTTP, GCS, S3)

        Returns:
            BaseReader: Appropriate reader instance for the format

        Raises:
            ValueError: If format cannot be determined
            NotImplementedError: If reader not yet implemented (Week 1-2)

        Example:
            >>> reader = Readers.auto_detect("/data.zarr")
            >>> # Returns Zarr3Reader (Tier 1 - native TensorStore)

            >>> reader = Readers.auto_detect("/data.tif")
            >>> # Returns TiffReader (Tier 2 - custom optimized)

            >>> reader = Readers.auto_detect("/data.czi")
            >>> # Returns BIOIOReader (Tier 3 - BIOIO adapter)

        Tier Selection Logic:
            Tier 1 (Maximum Performance):
            - .n5 → N5Reader
            - .zarr (Zarr3) → Zarr3Reader
            - .zarr (Zarr2) → Zarr2Reader
            - precomputed:// → PrecomputedReader

            Tier 2 (Production Formats):
            - .tif, .tiff → TiffReader
            - .nd2 → ND2Reader
            - .ims → IMSReader
            - .h5, .hdf5 → HDF5Reader

            Tier 3 (Broad Compatibility):
            - .czi, .lif, .sldy, .dv, etc. → BIOIOReader
            - Unknown formats → BIOIOReader (fallback)

        Notes:
            - Extensions checked in order of priority (Tier 1 → 2 → 3)
            - Case-insensitive extension matching
            - For .zarr, checks for Zarr3 vs Zarr2 version
            - User can override with explicit reader if needed
        """
        path_lower = path.lower()

        # Tier 1: Native TensorStore (maximum performance)
        if path_lower.endswith('.n5'):
            return Readers.n5(path)
        elif path_lower.endswith('.zarr'):
            # Distinguish Zarr3 vs Zarr2
            if _is_zarr3(path):
                return Readers.zarr3(path)
            else:
                return Readers.zarr2(path)
        elif 'precomputed://' in path_lower or path_lower.endswith('.precomputed'):
            return Readers.precomputed(path)

        # Tier 2: Custom Optimized (production formats)
        elif path_lower.endswith(('.tif', '.tiff')):
            return Readers.tiff(path)
        elif path_lower.endswith('.nd2'):
            return Readers.nd2(path)
        elif path_lower.endswith('.ims'):
            return Readers.ims(path)
        elif path_lower.endswith(('.h5', '.hdf5')):
            return Readers.hdf5(path)

        # Tier 3: BIOIO Adapter (broad compatibility)
        else:
            return Readers.bioio(path)

    # ========================================================================
    # Tier 1: Native TensorStore Readers (Week 3-4)
    # ========================================================================

    @staticmethod
    def n5(path: str) -> BaseReader:
        """
        Create N5 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to N5 dataset

        Returns:
            N5Reader instance

        Example:
            >>> reader = Readers.n5("/data.n5")

        Implementation Status:
            🚧 Week 3-4 (Phase 5.2 - Tier 1 Readers)
        """
        raise NotImplementedError(
            "N5Reader not yet implemented. "
            "Will be added in Week 3-4 (Phase 5.2 - Tier 1 Readers). "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def zarr3(path: str) -> BaseReader:
        """
        Create Zarr3 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to Zarr3 dataset

        Returns:
            Zarr3Reader instance

        Example:
            >>> reader = Readers.zarr3("/data.zarr")

        Implementation Status:
            🚧 Week 3-4 (Phase 5.2 - Tier 1 Readers)
        """
        raise NotImplementedError(
            "Zarr3Reader not yet implemented. "
            "Will be added in Week 3-4 (Phase 5.2 - Tier 1 Readers). "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def zarr2(path: str) -> BaseReader:
        """
        Create Zarr2 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to Zarr2 dataset

        Returns:
            Zarr2Reader instance

        Example:
            >>> reader = Readers.zarr2("/data.zarr")

        Implementation Status:
            🚧 Week 3-4 (Phase 5.2 - Tier 1 Readers)
        """
        raise NotImplementedError(
            "Zarr2Reader not yet implemented. "
            "Will be added in Week 3-4 (Phase 5.2 - Tier 1 Readers). "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def precomputed(path: str) -> BaseReader:
        """
        Create Neuroglancer Precomputed reader (Tier 1 - Native TensorStore).

        Args:
            path: Path or URL to Precomputed dataset

        Returns:
            PrecomputedReader instance

        Example:
            >>> reader = Readers.precomputed("precomputed://gs://bucket/data")

        Implementation Status:
            🚧 Week 3-4 (Phase 5.2 - Tier 1 Readers)
        """
        raise NotImplementedError(
            "PrecomputedReader not yet implemented. "
            "Will be added in Week 3-4 (Phase 5.2 - Tier 1 Readers). "
            "See PLAN_phase5.md for timeline."
        )

    # ========================================================================
    # Tier 2: Custom Optimized Readers (Week 5-6)
    # ========================================================================

    @staticmethod
    def tiff(path: str) -> BaseReader:
        """
        Create TIFF reader (Tier 2 - Custom Optimized).

        Reuses existing load_tiff_stack() from utils.py.

        Args:
            path: Path to TIFF file or directory

        Returns:
            TiffReader instance

        Example:
            >>> reader = Readers.tiff("/data.tif")

        Implementation Status:
            🚧 Week 5-6 (Phase 5.3 - Tier 2 Readers)
        """
        raise NotImplementedError(
            "TiffReader not yet implemented. "
            "Will be added in Week 5-6 (Phase 5.3 - Tier 2 Readers). "
            "Will reuse existing load_tiff_stack() from utils.py. "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def nd2(path: str) -> BaseReader:
        """
        Create ND2 reader (Tier 2 - Custom Optimized).

        Reuses existing load_nd2_stack() from utils.py.

        Args:
            path: Path to ND2 file

        Returns:
            ND2Reader instance

        Example:
            >>> reader = Readers.nd2("/data.nd2")

        Implementation Status:
            🚧 Week 5-6 (Phase 5.3 - Tier 2 Readers)
        """
        raise NotImplementedError(
            "ND2Reader not yet implemented. "
            "Will be added in Week 5-6 (Phase 5.3 - Tier 2 Readers). "
            "Will reuse existing load_nd2_stack() from utils.py. "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def ims(path: str) -> BaseReader:
        """
        Create IMS reader (Tier 2 - Custom Optimized).

        Reuses existing load_ims_stack() from utils.py.

        Args:
            path: Path to IMS file

        Returns:
            IMSReader instance

        Example:
            >>> reader = Readers.ims("/data.ims")

        Implementation Status:
            🚧 Week 5-6 (Phase 5.3 - Tier 2 Readers)
        """
        raise NotImplementedError(
            "IMSReader not yet implemented. "
            "Will be added in Week 5-6 (Phase 5.3 - Tier 2 Readers). "
            "Will reuse existing load_ims_stack() from utils.py. "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def hdf5(path: str) -> BaseReader:
        """
        Create HDF5 reader (Tier 2 - Custom Optimized).

        Args:
            path: Path to HDF5 file

        Returns:
            HDF5Reader instance

        Example:
            >>> reader = Readers.hdf5("/data.h5")

        Implementation Status:
            🚧 Week 5-6 (Phase 5.3 - Tier 2 Readers)
        """
        raise NotImplementedError(
            "HDF5Reader not yet implemented. "
            "Will be added in Week 5-6 (Phase 5.3 - Tier 2 Readers). "
            "See PLAN_phase5.md for timeline."
        )

    # ========================================================================
    # Tier 3: BIOIO Adapter (Week 7)
    # ========================================================================

    @staticmethod
    def bioio(path: str, format: Optional[str] = None) -> BaseReader:
        """
        Create BIOIO adapter reader (Tier 3 - Broad Compatibility).

        Strategic investment: One adapter unlocks 20+ formats.
        Supports: CZI, LIF, SLDY, DV, and all other BIOIO formats.

        Args:
            path: Path to input file
            format: Optional format hint for BIOIO (auto-detects if None)

        Returns:
            BIOIOReader instance

        Example:
            >>> # Auto-detect format
            >>> reader = Readers.bioio("/data.czi")

            >>> # Explicit format
            >>> reader = Readers.bioio("/data.czi", format="czi")

        Supported Formats:
            CZI, LIF, SLDY, DV, OME-TIFF, and 20+ more via BIOIO plugins.
            See https://github.com/bioio-devs/bioio for full list.

        Implementation Status:
            🚧 Week 7 (Phase 5.4 - BIOIO Adapter)
            Strategic unlock: 1 adapter (~200 LOC) = 20+ formats
        """
        raise NotImplementedError(
            "BIOIOReader not yet implemented. "
            "Will be added in Week 7 (Phase 5.4 - BIOIO Adapter). "
            "This is the strategic unlock: one adapter, 20+ formats. "
            "See PLAN_phase5.md for timeline."
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _is_zarr3(path: str) -> bool:
    """
    Check if Zarr dataset is Zarr3 vs Zarr2.

    Zarr3 has a zarr.json file, Zarr2 has .zarray/.zgroup files.

    Args:
        path: Path to Zarr dataset

    Returns:
        bool: True if Zarr3, False if Zarr2

    Implementation:
        🚧 Placeholder - will be implemented in Week 3-4
        Currently returns False (assume Zarr2 by default)
    """
    # Placeholder: Will check for zarr.json (Zarr3) vs .zarray (Zarr2)
    # For now, assume Zarr2 by default
    import os
    if os.path.exists(os.path.join(path, 'zarr.json')):
        return True
    return False
