"""
CZI reader implementation wrapping existing load_czi_stack function.

Tier 2 reader - reuses proven production code with multi-view support.
Uses pylibCZIrw directly (not bioio-czi) to avoid scene name parsing bugs.
"""

from typing import Dict, Optional, List
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_czi_stack, extract_czi_metadata
from .base import DaskReader


class CZIReader(DaskReader):
    """
    Reader for Zeiss CZI format using existing load_czi_stack function.

    Uses pylibCZIrw directly. Supports multi-view CZI files (V dimension)
    that bioio-czi cannot handle due to scene name parsing bugs.
    DaskReader base class wraps the dask array via ts.virtual_chunked.

    Tier: 2 (Custom Optimized - Production Ready)

    Dimension handling:
    - Single view, single channel: 3D ZYX
    - Single view, multi channel: 4D CZYX
    - Multi view: 5D VCZYX

    Example:
        >>> from tensorswitch_v2.readers import CZIReader
        >>> reader = CZIReader("/path/to/data.czi")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str, view_index: Optional[int] = None):
        super().__init__(path)
        self._view_index = view_index
        self._axes_order = None
        self._metadata_cache = None

    def _load(self):
        """Lazy-load the CZI data."""
        if self._dask_array is not None:
            return
        self._dask_array, _, self._axes_order = load_czi_stack(
            self.path, view_index=self._view_index
        )

    def _get_dimension_names(self) -> List[str]:
        """Return dimension names from CZI axes order."""
        self._load()
        return self.axes_order

    def get_metadata(self) -> Dict:
        """Return CZI metadata using existing extract_czi_metadata function."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._load()

        try:
            raw_xml, voxel_sizes = extract_czi_metadata(self.path)
            self._metadata_cache = {
                'raw_xml': raw_xml,
                'axes_order': self.axes_order,
                'voxel_size_x': voxel_sizes.get('x', 1.0) if voxel_sizes else 1.0,
                'voxel_size_y': voxel_sizes.get('y', 1.0) if voxel_sizes else 1.0,
                'voxel_size_z': voxel_sizes.get('z', 1.0) if voxel_sizes else 1.0,
                'shape': tuple(self._dask_array.shape),
                'dtype': str(self._dask_array.dtype),
            }
        except Exception as e:
            print(f"Warning: Failed to extract CZI metadata: {e}")
            self._metadata_cache = {
                'axes_order': self.axes_order,
                'shape': tuple(self._dask_array.shape),
                'dtype': str(self._dask_array.dtype),
            }

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """Return voxel dimensions from CZI metadata in nanometers."""
        metadata = self.get_metadata()
        return {
            'x': metadata.get('voxel_size_x', 1.0),
            'y': metadata.get('voxel_size_y', 1.0),
            'z': metadata.get('voxel_size_z', 1.0),
        }

    def get_ome_metadata(self) -> Dict:
        """Return OME-NGFF metadata using CZI axes order."""
        self._load()
        voxel_sizes = self.get_voxel_sizes()

        axes = []
        scale = []
        for axis_name in self._axes_order:
            if axis_name in ['z', 'y', 'x']:
                axes.append({'name': axis_name, 'type': 'space', 'unit': 'micrometer'})
                scale.append(voxel_sizes.get(axis_name, 1.0))
            elif axis_name == 'c':
                axes.append({'name': 'c', 'type': 'channel'})
                scale.append(1.0)
            elif axis_name == 't':
                axes.append({'name': 't', 'type': 'time', 'unit': 'second'})
                scale.append(1.0)
            elif axis_name == 'v':
                axes.append({'name': 't', 'type': 'time'})
                scale.append(1.0)
            else:
                axes.append({'name': axis_name})
                scale.append(1.0)

        return {
            'multiscales': [{
                'axes': axes,
                'datasets': [{
                    'path': 's0',
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': scale
                    }]
                }],
                'name': f'Data from CZIReader'
            }]
        }

    @property
    def axes_order(self) -> List[str]:
        """Get the CZI axes order (e.g., ['t', 'c', 'z', 'y', 'x'])."""
        self._load()
        return ['t' if ax == 'v' else ax for ax in self._axes_order]

    def __repr__(self) -> str:
        view_str = f", view_index={self._view_index}" if self._view_index is not None else ""
        return f"CZIReader(path='{self.path}'{view_str})"
