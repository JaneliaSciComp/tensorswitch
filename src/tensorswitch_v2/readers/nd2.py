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
                    'voxel_sizes_stated': dict(voxel_sizes) if voxel_sizes else {},
                }
            except Exception as e:
                print(f"Warning: Failed to extract ND2 metadata: {e}")
                self._metadata_cache = {}

        return self._metadata_cache

    def _read_voxel_sizes(self) -> Optional[Dict[str, Optional[float]]]:
        """Voxel sizes stated by the ND2 OME metadata, in nanometers."""
        return self.get_metadata().get('voxel_sizes_stated') or None

    def __repr__(self) -> str:
        return f"ND2Reader(path='{self.path}')"
