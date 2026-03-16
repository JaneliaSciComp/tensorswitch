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
    returns a Dask array. DaskReader base class wraps that via
    ts.virtual_chunked for a uniform TensorStore API.

    Tier: 2 (Custom Optimized - Production Ready)
    - Reuses existing optimized code
    - Supports single files and directories (Z-stack via dask_image.imread)
    - OME-TIFF metadata extraction

    Example:
        >>> from tensorswitch_v2.readers import TiffReader
        >>> reader = TiffReader("/path/to/data.tif")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the TIFF data and extract dimension names."""
        if self._dask_array is not None:
            return

        import os

        # Load dask array
        self._dask_array = load_tiff_stack(self.path)

        # Extract actual dimension names from TIFF file.
        # tifffile uses 'Q' for unrecognized dimensions; only accept well-defined axes.
        _KNOWN_TIFF_AXES = frozenset('ZYXTCSIzyxtcsi')
        try:
            # For directories, read metadata from the first sorted TIFF file
            metadata_file = self.path
            if os.path.isdir(self.path):
                from ..utils.format_loaders import _find_tiff_files
                metadata_file = _find_tiff_files(self.path)[0]

            if os.path.isfile(metadata_file):
                with tifffile.TiffFile(metadata_file) as tif:
                    if tif.series:
                        axes_str = tif.series[0].axes
                        if all(c in _KNOWN_TIFF_AXES for c in axes_str):
                            dims = [c.lower() for c in axes_str]
                            # For Z-stack directories, prepend 'z' for stacking dimension
                            if os.path.isdir(self.path) and 'z' not in dims:
                                dims = ['z'] + dims
                            self._dimension_names = dims
        except Exception as e:
            print(f"Warning: Could not extract TIFF dimension names: {e}")
            self._dimension_names = None

    def _get_dimension_names(self) -> List[str]:
        """Return dimension names, auto-detected from TIFF axes or inferred from shape."""
        self._load()
        if self._dimension_names:
            return self._dimension_names
        # Fallback: infer from shape
        ndim = len(self._dask_array.shape)
        if ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(ndim)]

    def get_metadata(self) -> Dict:
        """Return TIFF metadata using existing extract_tiff_ome_metadata function."""
        if self._metadata_cache is None:
            # Ensure dask array is loaded to get shape/dtype
            self._load()

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
        """Return voxel dimensions from TIFF metadata in nanometers."""
        metadata = self.get_metadata()

        if 'voxel_size_x' in metadata:
            return {
                'x': metadata.get('voxel_size_x', 1.0),
                'y': metadata.get('voxel_size_y', 1.0),
                'z': metadata.get('voxel_size_z', 1.0)
            }

        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def __repr__(self) -> str:
        return f"TiffReader(path='{self.path}')"
