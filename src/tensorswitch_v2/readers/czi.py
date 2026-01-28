"""
CZI reader implementation wrapping existing load_czi_stack function.

Tier 2 reader - reuses proven production code with multi-view support.
Uses pylibCZIrw directly (not bioio-czi) to avoid scene name parsing bugs.
"""

from typing import Dict, Optional, List
from tensorswitch.utils import load_czi_stack, extract_czi_metadata
from .base import BaseReader


class CZIReader(BaseReader):
    """
    Reader for Zeiss CZI format using existing load_czi_stack function.

    Wraps the proven load_czi_stack() from tensorswitch/utils.py which
    uses pylibCZIrw directly. Supports multi-view CZI files (V dimension)
    that bioio-czi cannot handle due to scene name parsing bugs.

    Tier: 2 (Custom Optimized - Production Ready)
    - Reuses existing optimized code (pylibCZIrw)
    - Multi-view support (V dimension → 5D VCZYX arrays)
    - Metadata extraction via CZI XML
    - No dependency on bioio/bioio-czi

    Dimension handling:
    - Single view, single channel: 3D ZYX
    - Single view, multi channel: 4D CZYX
    - Multi view: 5D VCZYX

    Example:
        >>> from tensorswitch_v2.readers import CZIReader
        >>> reader = CZIReader("/path/to/data.czi")
        >>> spec = reader.get_tensorstore_spec()
        >>> print(spec['schema']['shape'])
        [36, 2, 2153, 1920, 1920]  # VCZYX for multi-view

    Example (single view):
        >>> reader = CZIReader("/path/to/data.czi", view_index=0)
        >>> spec = reader.get_tensorstore_spec()
        >>> print(spec['schema']['shape'])
        [2, 2153, 1920, 1920]  # CZYX
    """

    def __init__(self, path: str, view_index: Optional[int] = None):
        """
        Initialize CZI reader.

        Args:
            path: Path to CZI file
            view_index: Optional specific view to load. If None and multiple
                       views exist, loads all views as 5D VCZYX array.
        """
        super().__init__(path)
        self._view_index = view_index
        self._dask_array = None
        self._axes_order = None
        self._metadata_cache = None

    def _load(self):
        """Lazy-load the CZI data."""
        if self._dask_array is not None:
            return
        self._dask_array, _, self._axes_order = load_czi_stack(
            self.path, view_index=self._view_index
        )

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping Dask array from load_czi_stack.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping Dask array
        """
        self._load()

        spec = {
            'driver': 'array',
            'array': self._dask_array,
            'schema': {
                'dtype': str(self._dask_array.dtype),
                'shape': list(self._dask_array.shape),
                'dimension_names': self._axes_order or self._infer_dimension_names(self._dask_array.shape)
            }
        }
        return spec

    def get_metadata(self) -> Dict:
        """
        Return CZI metadata using existing extract_czi_metadata function.

        Returns:
            dict: CZI metadata including raw XML, voxel sizes, and axes order
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._load()

        try:
            raw_xml, voxel_sizes = extract_czi_metadata(self.path)
            self._metadata_cache = {
                'raw_xml': raw_xml,
                'axes_order': self._axes_order,
                'voxel_size_x': voxel_sizes.get('x', 1.0) if voxel_sizes else 1.0,
                'voxel_size_y': voxel_sizes.get('y', 1.0) if voxel_sizes else 1.0,
                'voxel_size_z': voxel_sizes.get('z', 1.0) if voxel_sizes else 1.0,
                'shape': tuple(self._dask_array.shape),
                'dtype': str(self._dask_array.dtype),
            }
        except Exception as e:
            print(f"Warning: Failed to extract CZI metadata: {e}")
            self._metadata_cache = {
                'axes_order': self._axes_order,
                'shape': tuple(self._dask_array.shape),
                'dtype': str(self._dask_array.dtype),
            }

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from CZI metadata.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in micrometers
        """
        metadata = self.get_metadata()
        return {
            'x': metadata.get('voxel_size_x', 1.0),
            'y': metadata.get('voxel_size_y', 1.0),
            'z': metadata.get('voxel_size_z', 1.0),
        }

    def get_ome_metadata(self) -> Dict:
        """
        Return OME-NGFF metadata using CZI axes order.

        Overrides base implementation to use the axes order from
        load_czi_stack() (e.g., ['v', 'c', 'z', 'y', 'x'] for multi-view).
        """
        self._load()
        voxel_sizes = self.get_voxel_sizes()

        # Build axes from CZI axes_order
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
                axes.append({'name': 'v', 'type': 'space'})
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
        """Get the CZI axes order (e.g., ['v', 'c', 'z', 'y', 'x'])."""
        self._load()
        return self._axes_order

    def _infer_dimension_names(self, shape):
        """Infer dimension names from array shape."""
        ndim = len(shape)
        if ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['v', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(ndim)]

    def __repr__(self) -> str:
        view_str = f", view_index={self._view_index}" if self._view_index is not None else ""
        return f"CZIReader(path='{self.path}'{view_str})"
