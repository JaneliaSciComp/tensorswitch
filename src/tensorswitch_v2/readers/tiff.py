"""
TIFF reader implementation wrapping existing load_tiff_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
Auto-detects dimension names from TIFF metadata (axes string).
"""

from typing import Dict, List, Optional
import tifffile
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_tiff_stack, extract_tiff_ome_metadata
from .base import DaskReader


class TiffReader(DaskReader):
    """
    Reader for TIFF format using existing load_tiff_stack function.

    Wraps the proven load_tiff_stack() from tensorswitch/utils.py which
    returns a Dask array. Then wraps that Dask array in TensorStore's
    'array' driver for compatibility with the unified architecture.

    Tier: 2 (Custom Optimized - Production Ready)
    - Reuses existing optimized code
    - Minimal conversion overhead (Dask → TensorStore)
    - Production-tested implementation
    - Supports single files and directories

    Features:
    - Multi-page TIFF support
    - Directory of TIFFs (Z-stack)
    - OME-TIFF metadata extraction
    - ImageJ metadata support

    Example:
        >>> from tensorswitch_v2.readers import TiffReader
        >>> reader = TiffReader("/path/to/data.tif")
        >>> spec = reader.get_tensorstore_spec()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers
        >>> reader = Readers.tiff("/path/to/stack/")
        >>> dataset = TensorSwitchDataset("/path/to/stack/", reader=reader)
    """

    def __init__(self, path: str):
        """
        Initialize TIFF reader.

        Args:
            path: Path to TIFF file or directory containing TIFFs

        Example:
            >>> # Single TIFF file
            >>> reader = TiffReader("/data/image.tif")
            >>>
            >>> # Directory of TIFFs (Z-stack)
            >>> reader = TiffReader("/data/stack/")
        """
        super().__init__(path)
        self._dask_array = None
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the TIFF data and extract dimension names."""
        if self._dask_array is not None:
            return

        import os

        # Load dask array
        self._dask_array = load_tiff_stack(self.path)

        # Extract actual dimension names from TIFF file
        try:
            if os.path.isfile(self.path):
                with tifffile.TiffFile(self.path) as tif:
                    if tif.series:
                        # axes is a string like 'ZYX', 'CZYX', 'TZCYX'
                        axes_str = tif.series[0].axes
                        self._dimension_names = [c.lower() for c in axes_str]
        except Exception as e:
            print(f"Warning: Could not extract TIFF dimension names: {e}")
            self._dimension_names = None

    def _get_dimension_names(self):
        """Return dimension names from TIFF metadata or infer from shape."""
        return self._dimension_names or self._infer_dimension_names(self._dask_array.shape)

    def get_metadata(self) -> Dict:
        """
        Return TIFF metadata using existing extract_tiff_ome_metadata function.

        Returns:
            dict: TIFF metadata including shape, dtype, OME-XML and voxel sizes

        Example:
            >>> reader = TiffReader("/data.tif")
            >>> metadata = reader.get_metadata()

        Notes:
            - Reuses extract_tiff_ome_metadata from utils.py
            - Cached after first read
            - Returns (ome_xml, voxel_sizes) tuple, converted to dict
        """
        if self._metadata_cache is None:
            # Ensure dask array is loaded to get shape/dtype
            if self._dask_array is None:
                self._dask_array = load_tiff_stack(self.path)

            try:
                ome_xml, voxel_sizes = extract_tiff_ome_metadata(self.path)
                self._metadata_cache = {
                    'shape': tuple(self._dask_array.shape),
                    'dtype': str(self._dask_array.dtype),
                    'ome_xml': ome_xml,
                    'voxel_size_x': voxel_sizes.get('x', 1.0) if voxel_sizes else 1.0,
                    'voxel_size_y': voxel_sizes.get('y', 1.0) if voxel_sizes else 1.0,
                    'voxel_size_z': voxel_sizes.get('z', 1.0) if voxel_sizes else 1.0,
                }
            except Exception as e:
                print(f"Warning: Failed to extract TIFF metadata: {e}")
                self._metadata_cache = {
                    'shape': tuple(self._dask_array.shape),
                    'dtype': str(self._dask_array.dtype),
                }

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from TIFF metadata.

        Extracts voxel sizes from OME-XML or ImageJ metadata if available.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = TiffReader("/data.tif")
            >>> voxel_sizes = reader.get_voxel_sizes()

        Notes:
            - Returns 1.0 for each dimension if metadata unavailable
            - Checks OME-XML first, then ImageJ metadata
            - Values are converted to nanometers from source units
        """
        metadata = self.get_metadata()

        # Try to extract from metadata
        if 'voxel_size_x' in metadata:
            return {
                'x': metadata.get('voxel_size_x', 1.0),
                'y': metadata.get('voxel_size_y', 1.0),
                'z': metadata.get('voxel_size_z', 1.0)
            }

        # Default
        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def __repr__(self) -> str:
        """String representation of TIFF reader."""
        return f"TiffReader(path='{self.path}')"
