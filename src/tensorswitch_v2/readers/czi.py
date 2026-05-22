"""
CZI reader implementation wrapping existing load_czi_stack function.

Tier 2 reader - reuses proven production code with multi-view support.
Uses pylibCZIrw directly (not bioio-czi) to avoid scene name parsing bugs.
"""

from typing import Dict, Optional, List
import numpy as np
# Import utility functions from v2 utils (independent from v1)
from ..utils import load_czi_stack, extract_czi_metadata
from ..utils.format_loaders import _get_czidoc
from .base import DaskReader

# Maps our axis labels back to the CZI plane-dict keys used by pylibCZIrw.
_AXIS_TO_CZI_KEY = {'t': 'V', 'c': 'C', 'z': 'Z'}


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

    def _read_fn(self, domain, array, read_params):
        """Read a CZI plane directly from the cached czidoc, bypassing dask.

        The base-class path goes through the 107K-task dask graph on every
        call, adding 100-500 ms of graph-traversal overhead per plane even
        though only one task needs to run.  This override calls czidoc.read()
        directly, keeping the same LRU plane cache for cross-shard reuse.

        Locking strategy:
          _chunk_cache_lock  — guards the in-memory plane cache (fast, no IO)
          czidoc read_lock   — serialises czidoc.read() calls (not thread-safe)
        """
        self._load()
        axes = self._get_dimension_names()   # e.g. ['t','c','z','y','x']
        ndim = domain.ndim
        origin = [int(domain.origin[i]) for i in range(ndim)]
        shape  = [int(domain.shape[i])  for i in range(ndim)]

        # Build the CZI plane dict and locate y/x dimensions
        plane_dict = {}
        y_idx = x_idx = None
        for i, ax in enumerate(axes):
            if ax in _AXIS_TO_CZI_KEY:
                plane_dict[_AXIS_TO_CZI_KEY[ax]] = origin[i]
            elif ax == 'y':
                y_idx = i
            elif ax == 'x':
                x_idx = i

        y_start, x_start = origin[y_idx], origin[x_idx]
        y_size,  x_size  = shape[y_idx],  shape[x_idx]

        # Cache key: all non-spatial coordinates (identifies the full plane)
        plane_key = tuple(origin[i] for i, ax in enumerate(axes)
                          if ax not in ('y', 'x'))

        # Slice index for writing into the output array (all leading dims = 0)
        ns_idx = (0,) * (ndim - 2)

        # --- cache hit ---
        with self._chunk_cache_lock:
            if plane_key in self._chunk_cache:
                plane = self._chunk_cache[plane_key]
                self._chunk_cache_order.remove(plane_key)
                self._chunk_cache_order.append(plane_key)
                array[ns_idx] = plane[y_start:y_start + y_size,
                                      x_start:x_start + x_size]
                return

        # --- cache miss: read from CZI (outside cache lock) ---
        czidoc, read_lock = _get_czidoc(self.path)
        with read_lock:
            result = czidoc.read(plane=plane_dict)

        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        result = np.ascontiguousarray(result)

        # Store full plane in LRU cache
        with self._chunk_cache_lock:
            plane_bytes = result.nbytes
            while (self._chunk_cache_bytes + plane_bytes > self._CHUNK_CACHE_MAX_BYTES
                   and self._chunk_cache_order):
                evict_key = self._chunk_cache_order.pop(0)
                evicted = self._chunk_cache.pop(evict_key, None)
                if evicted is not None:
                    self._chunk_cache_bytes -= evicted.nbytes
            if plane_bytes <= self._CHUNK_CACHE_MAX_BYTES:
                self._chunk_cache[plane_key] = result
                self._chunk_cache_order.append(plane_key)
                self._chunk_cache_bytes += plane_bytes

        array[ns_idx] = result[y_start:y_start + y_size,
                                x_start:x_start + x_size]

    def __repr__(self) -> str:
        view_str = f", view_index={self._view_index}" if self._view_index is not None else ""
        return f"CZIReader(path='{self.path}'{view_str})"
