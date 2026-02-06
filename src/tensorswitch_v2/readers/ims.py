"""
IMS (Imaris) reader implementation wrapping existing load_ims_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
Auto-detects dimension names from IMS HDF5 structure.
"""

from typing import Dict, List, Optional
import h5py
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_ims_stack, extract_ims_metadata
from .base import BaseReader


class IMSReader(BaseReader):
    """
    Reader for Imaris IMS format using existing load_ims_stack function.

    Wraps the proven load_ims_stack() from tensorswitch/utils.py which
    returns a Dask array. Then wraps that Dask array in TensorStore's
    'array' driver for compatibility with the unified architecture.

    Tier: 2 (Custom Optimized - Production Ready)
    - Reuses existing optimized code
    - Minimal conversion overhead (Dask -> TensorStore)
    - Production-tested implementation
    - Supports multi-channel and multi-resolution data

    Features:
    - HDF5-based IMS format support
    - Multi-resolution pyramid access
    - Physical pixel sizes extraction
    - Channel metadata support

    Example:
        >>> from tensorswitch_v2.readers import IMSReader
        >>> reader = IMSReader("/path/to/data.ims")
        >>> spec = reader.get_tensorstore_spec()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers
        >>> dataset = TensorSwitchDataset("/path/to/data.ims")
        >>> ts_array = dataset.get_tensorstore_array()
    """

    def __init__(self, path: str, resolution_level: int = 0):
        """
        Initialize IMS reader.

        Args:
            path: Path to IMS file
            resolution_level: Which resolution level to read (0 = highest resolution)

        Example:
            >>> reader = IMSReader("/data/experiment.ims")
            >>> reader = IMSReader("/data/experiment.ims", resolution_level=1)
        """
        super().__init__(path)
        self._resolution_level = resolution_level
        self._dask_array = None
        self._h5_file = None  # Keep h5 file open for lazy loading
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the IMS data and extract dimension names."""
        if self._dask_array is not None:
            return

        # Load dask array
        self._dask_array, self._h5_file = load_ims_stack(self.path)

        # Extract dimension info from IMS HDF5 structure
        # IMS files use consistent ordering: T (if >1), C (if >1), Z, Y, X
        try:
            with h5py.File(self.path, 'r') as f:
                # Count dimensions from the HDF5 structure
                dataset_path = f'DataSet/ResolutionLevel {self._resolution_level}'
                if dataset_path in f:
                    ds_group = f[dataset_path]
                    # Count time points
                    time_points = len([k for k in ds_group.keys() if k.startswith('TimePoint')])
                    # Count channels (check first time point)
                    if 'TimePoint 0' in ds_group:
                        channels = len([k for k in ds_group['TimePoint 0'].keys() if k.startswith('Channel')])
                    else:
                        channels = 1

                    # Build dimension names based on actual structure
                    ndim = len(self._dask_array.shape)
                    if ndim == 5:
                        self._dimension_names = ['t', 'c', 'z', 'y', 'x']
                    elif ndim == 4:
                        if time_points > 1:
                            self._dimension_names = ['t', 'z', 'y', 'x']
                        else:
                            self._dimension_names = ['c', 'z', 'y', 'x']
                    elif ndim == 3:
                        self._dimension_names = ['z', 'y', 'x']
                    else:
                        self._dimension_names = None
        except Exception as e:
            print(f"Warning: Could not extract IMS dimension names: {e}")
            self._dimension_names = None

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping Dask array from load_ims_stack.

        Reuses existing load_ims_stack() function which returns a Dask array,
        then wraps it in TensorStore's 'array' driver.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping Dask array

        Example:
            >>> reader = IMSReader("/data.ims")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['driver'])
            'array'
            >>> print(spec['schema']['dimension_names'])  # Auto-detected
            ['z', 'y', 'x']

        Notes:
            - Tier 2 approach: Dask array -> TensorStore 'array' driver
            - Minimal overhead (one Dask layer)
            - Dimension names auto-detected from IMS HDF5 structure
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
        Return IMS metadata using existing extract_ims_metadata function.

        Returns:
            dict: IMS metadata including voxel sizes and channel info

        Example:
            >>> reader = IMSReader("/data.ims")
            >>> metadata = reader.get_metadata()

        Notes:
            - Reuses extract_ims_metadata from utils.py
            - Cached after first read
            - IMS stores rich metadata in HDF5 attributes
        """
        if self._metadata_cache is None:
            try:
                metadata = extract_ims_metadata(self.path)
                if isinstance(metadata, tuple):
                    # Returns (metadata_dict, voxel_sizes) where voxel_sizes
                    # is a list [x, y, z] or dict {'x':, 'y':, 'z':}
                    raw_metadata, voxel_sizes = metadata
                    if isinstance(voxel_sizes, (list, tuple)) and len(voxel_sizes) >= 3:
                        vx, vy, vz = voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]
                    elif isinstance(voxel_sizes, dict):
                        vx = voxel_sizes.get('x', 1.0)
                        vy = voxel_sizes.get('y', 1.0)
                        vz = voxel_sizes.get('z', 1.0)
                    else:
                        vx, vy, vz = 1.0, 1.0, 1.0
                    self._metadata_cache = {
                        'raw_metadata': raw_metadata,
                        'voxel_size_x': vx if vx else 1.0,
                        'voxel_size_y': vy if vy else 1.0,
                        'voxel_size_z': vz if vz else 1.0,
                    }
                elif isinstance(metadata, dict):
                    self._metadata_cache = metadata
                    # Try to extract voxel sizes from metadata dict
                    if 'voxel_size_x' not in self._metadata_cache:
                        self._metadata_cache['voxel_size_x'] = 1.0
                        self._metadata_cache['voxel_size_y'] = 1.0
                        self._metadata_cache['voxel_size_z'] = 1.0
                else:
                    self._metadata_cache = {}
            except Exception as e:
                print(f"Warning: Failed to extract IMS metadata: {e}")
                self._metadata_cache = {}

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from IMS metadata.

        Extracts voxel sizes from IMS HDF5 attributes.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in micrometers

        Example:
            >>> reader = IMSReader("/data.ims")
            >>> voxel_sizes = reader.get_voxel_sizes()

        Notes:
            - Returns 1.0 for each dimension if metadata unavailable
            - IMS files typically store voxel sizes in ImageInfo attributes
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
        """String representation of IMS reader."""
        return f"IMSReader(path='{self.path}', resolution_level={self._resolution_level})"
