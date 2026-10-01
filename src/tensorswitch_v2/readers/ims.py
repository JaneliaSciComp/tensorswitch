"""
IMS (Imaris) reader implementation wrapping existing load_ims_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
Dimension names are determined by load_ims_stack() using cross-referencing
of HDF5 TimePoint/Channel structure with the resulting array shape.
"""

from typing import Dict, List, Optional
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_ims_stack, extract_ims_metadata
from .base import DaskReader


class IMSReader(DaskReader):
    """
    Reader for Imaris IMS format using existing load_ims_stack function.

    Wraps the proven load_ims_stack() which returns a Dask array with
    all TimePoints and Channels loaded. DaskReader base class wraps
    that via ts.virtual_chunked for a uniform TensorStore API.

    Tier: 2 (Custom Optimized - Production Ready)

    Example:
        >>> from tensorswitch_v2.readers import IMSReader
        >>> reader = IMSReader("/path/to/data.ims")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str, resolution_level: int = 0):
        super().__init__(path)
        self._resolution_level = resolution_level
        self._h5_file = None  # Keep h5 file open for lazy loading
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the IMS data and extract dimension names."""
        if self._dask_array is not None:
            return

        # load_ims_stack reads all TimePoints and Channels, returns
        # authoritative dimension names from HDF5 structure cross-referencing
        self._dask_array, self._h5_file, self._dimension_names = load_ims_stack(self.path)

    def get_metadata(self) -> Dict:
        """Return IMS metadata using existing extract_ims_metadata function."""
        if self._metadata_cache is None:
            try:
                metadata = extract_ims_metadata(self.path)
                if isinstance(metadata, tuple):
                    raw_metadata, voxel_sizes = metadata
                    if isinstance(voxel_sizes, (list, tuple)) and len(voxel_sizes) >= 3:
                        stated = dict(zip('xyz', voxel_sizes[:3]))
                    elif isinstance(voxel_sizes, dict):
                        stated = {a: voxel_sizes.get(a) for a in 'xyz'}
                    else:
                        stated = {}
                    self._metadata_cache = {
                        'raw_metadata': raw_metadata,
                        'voxel_sizes_stated': {a: v for a, v in stated.items() if v},
                    }
                elif isinstance(metadata, dict):
                    self._metadata_cache = metadata
                else:
                    self._metadata_cache = {}
            except Exception as e:
                print(f"Warning: Failed to extract IMS metadata: {e}")
                self._metadata_cache = {}

        return self._metadata_cache

    def _read_voxel_sizes(self) -> Optional[Dict[str, Optional[float]]]:
        """Voxel sizes stated by the IMS DataSetInfo, in nanometers."""
        return self.get_metadata().get('voxel_sizes_stated') or None

    def __repr__(self) -> str:
        return f"IMSReader(path='{self.path}', resolution_level={self._resolution_level})"
