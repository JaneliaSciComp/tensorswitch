"""
ND2 reader implementation wrapping existing load_nd2_stack function.

Tier 2 reader - reuses proven production code with minimal overhead.
Auto-detects dimension names, chunk shape, and memory order from source.
"""

from typing import Dict, List, Optional
import nd2
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_nd2_stack, extract_nd2_ome_metadata
from .base import DaskReader


class ND2Reader(DaskReader):
    """
    Reader for Nikon ND2 format using existing load_nd2_stack function.

    Wraps the proven load_nd2_stack() which returns a Dask array.
    DaskReader base class wraps that via ts.virtual_chunked for a
    uniform TensorStore API.

    Tier: 2 (Custom Optimized - Production Ready)

    Example:
        >>> from tensorswitch_v2.readers import ND2Reader
        >>> reader = ND2Reader("/path/to/data.nd2")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str):
        super().__init__(path)
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
                self._dimension_names = [dim.lower() for dim in f.sizes.keys()]
        except Exception as e:
            print(f"Warning: Could not extract ND2 dimension names: {e}")
            self._dimension_names = None

    def get_metadata(self) -> Dict:
        """Return ND2 metadata using existing extract_nd2_ome_metadata function."""
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
        """Return voxel dimensions from ND2 metadata in nanometers."""
        metadata = self.get_metadata()
        return {
            'x': metadata.get('voxel_size_x', 1.0),
            'y': metadata.get('voxel_size_y', 1.0),
            'z': metadata.get('voxel_size_z', 1.0)
        }

    def __repr__(self) -> str:
        return f"ND2Reader(path='{self.path}')"
