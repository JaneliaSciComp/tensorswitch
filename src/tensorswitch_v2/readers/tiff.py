"""
TIFF reader implementation wrapping existing load_tiff_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
"""

from typing import Dict
from tensorswitch.utils import load_tiff_stack, extract_tiff_metadata
from .base import BaseReader


class TiffReader(BaseReader):
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

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping Dask array from load_tiff_stack.

        Reuses existing load_tiff_stack() function which returns a Dask array,
        then wraps it in TensorStore's 'array' driver.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping Dask array

        Example:
            >>> reader = TiffReader("/data.tif")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['driver'])
            'array'

        Notes:
            - Tier 2 approach: Dask array → TensorStore 'array' driver
            - Minimal overhead (one Dask layer)
            - Proven production code (load_tiff_stack already optimized)
        """
        if self._dask_array is None:
            # Reuse existing load_tiff_stack from utils.py
            self._dask_array = load_tiff_stack(self.path)

        # Wrap Dask array in TensorStore 'array' driver
        spec = {
            'driver': 'array',
            'array': self._dask_array,
            'schema': {
                'dtype': str(self._dask_array.dtype),
                'shape': list(self._dask_array.shape),
                'dimension_names': self._infer_dimension_names(self._dask_array.shape)
            }
        }

        return spec

    def get_metadata(self) -> Dict:
        """
        Return TIFF metadata using existing extract_tiff_metadata function.

        Returns:
            dict: TIFF metadata including OME-XML if available

        Example:
            >>> reader = TiffReader("/data.tif")
            >>> metadata = reader.get_metadata()

        Notes:
            - Reuses extract_tiff_metadata from utils.py
            - Cached after first read
        """
        if self._metadata_cache is None:
            try:
                self._metadata_cache = extract_tiff_metadata(self.path)
            except Exception as e:
                print(f"Warning: Failed to extract TIFF metadata: {e}")
                self._metadata_cache = {}

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from TIFF metadata.

        Extracts voxel sizes from OME-XML or ImageJ metadata if available.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in micrometers

        Example:
            >>> reader = TiffReader("/data.tif")
            >>> voxel_sizes = reader.get_voxel_sizes()

        Notes:
            - Returns 1.0 for each dimension if metadata unavailable
            - Checks OME-XML first, then ImageJ metadata
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

    def _infer_dimension_names(self, shape):
        """Infer dimension names from array shape."""
        ndim = len(shape)
        if ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(ndim)]

    def __repr__(self) -> str:
        """String representation of TIFF reader."""
        return f"TiffReader(path='{self.path}')"
