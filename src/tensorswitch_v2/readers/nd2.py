"""
ND2 reader implementation wrapping existing load_nd2_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
Auto-detects dimension names, chunk shape, and memory order from source.
"""

from typing import Dict, List, Optional
import nd2
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_nd2_stack, extract_nd2_ome_metadata
from .base import BaseReader


class ND2Reader(BaseReader):
    """
    Reader for Nikon ND2 format using existing load_nd2_stack function.

    Wraps the proven load_nd2_stack() from tensorswitch/utils.py which
    returns a Dask array. Then wraps that Dask array in TensorStore's
    'array' driver for compatibility with the unified architecture.

    Tier: 2 (Custom Optimized - Production Ready)
    - Reuses existing optimized code
    - Minimal conversion overhead (Dask -> TensorStore)
    - Production-tested implementation
    - Supports multi-channel and time-lapse data

    Features:
    - Native ND2 metadata extraction via nd2 library
    - OME-XML metadata support
    - Physical pixel sizes extraction
    - Multi-channel support

    Example:
        >>> from tensorswitch_v2.readers import ND2Reader
        >>> reader = ND2Reader("/path/to/data.nd2")
        >>> spec = reader.get_tensorstore_spec()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers
        >>> dataset = TensorSwitchDataset("/path/to/data.nd2")
        >>> ts_array = dataset.get_tensorstore_array()
    """

    def __init__(self, path: str):
        """
        Initialize ND2 reader.

        Args:
            path: Path to ND2 file

        Example:
            >>> reader = ND2Reader("/data/experiment.nd2")
        """
        super().__init__(path)
        self._dask_array = None
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the ND2 data and extract dimension names."""
        if self._dask_array is not None:
            return

        # Load dask array
        self._dask_array = load_nd2_stack(self.path)

        # Extract actual dimension names from ND2 file
        try:
            with nd2.ND2File(self.path) as f:
                # f.sizes is a dict like {'Z': 498, 'Y': 2000, 'X': 2000}
                # Keys give us the actual dimension names in order
                self._dimension_names = [dim.lower() for dim in f.sizes.keys()]
        except Exception as e:
            print(f"Warning: Could not extract ND2 dimension names: {e}")
            self._dimension_names = None

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping Dask array from load_nd2_stack.

        Reuses existing load_nd2_stack() function which returns a Dask array,
        then wraps it in TensorStore's 'array' driver.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping Dask array

        Example:
            >>> reader = ND2Reader("/data.nd2")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['driver'])
            'array'
            >>> print(spec['schema']['dimension_names'])  # Auto-detected
            ['z', 'y', 'x']

        Notes:
            - Tier 2 approach: Dask array -> TensorStore 'array' driver
            - Minimal overhead (one Dask layer)
            - Dimension names auto-detected from ND2 file metadata
        """
        self._load()

        # Use auto-detected dimension names, fall back to inference
        dimension_names = self._dimension_names or self._infer_dimension_names(self._dask_array.shape)

        # Wrap Dask array in TensorStore 'array' driver
        spec = {
            'driver': 'array',
            'array': self._dask_array,
            'schema': {
                'dtype': str(self._dask_array.dtype),
                'shape': list(self._dask_array.shape),
                'dimension_names': dimension_names
            }
        }

        return spec

    def get_metadata(self) -> Dict:
        """
        Return ND2 metadata using existing extract_nd2_ome_metadata function.

        Returns:
            dict: ND2 metadata including OME-XML and voxel sizes

        Example:
            >>> reader = ND2Reader("/data.nd2")
            >>> metadata = reader.get_metadata()

        Notes:
            - Reuses extract_nd2_ome_metadata from utils.py
            - Cached after first read
            - Returns OME-XML and voxel sizes
        """
        if self._metadata_cache is None:
            try:
                ome_xml, voxel_sizes = extract_nd2_ome_metadata(self.path)
                self._metadata_cache = {
                    'ome_xml': ome_xml,
                    'voxel_size_x': voxel_sizes.get('x', 1.0) if voxel_sizes else 1.0,
                    'voxel_size_y': voxel_sizes.get('y', 1.0) if voxel_sizes else 1.0,
                    'voxel_size_z': voxel_sizes.get('z', 1.0) if voxel_sizes else 1.0,
                }
            except Exception as e:
                print(f"Warning: Failed to extract ND2 metadata: {e}")
                self._metadata_cache = {}

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from ND2 metadata.

        Extracts voxel sizes from OME metadata via nd2 library.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in micrometers

        Example:
            >>> reader = ND2Reader("/data.nd2")
            >>> voxel_sizes = reader.get_voxel_sizes()

        Notes:
            - Returns 1.0 for each dimension if metadata unavailable
            - ND2 files typically have good voxel size metadata
        """
        metadata = self.get_metadata()

        return {
            'x': metadata.get('voxel_size_x', 1.0),
            'y': metadata.get('voxel_size_y', 1.0),
            'z': metadata.get('voxel_size_z', 1.0)
        }

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
        """String representation of ND2 reader."""
        return f"ND2Reader(path='{self.path}')"
