"""
BIOIO adapter reader for TensorSwitch Phase 5 architecture.

Tier 3 reader - strategic investment unlocking 20+ formats via BIOIO ecosystem.
Converts BIOIO's dask arrays to TensorStore via DaskReader's virtual_chunked.
"""

from typing import Dict, Optional, List, Any
from .base import DaskReader


class BIOIOReader(DaskReader):
    """
    Reader adapter for BIOIO library, supporting 20+ microscopy formats.

    Tier 3 strategic investment: one adapter that unlocks support for CZI,
    LIF, SLDY, DV, and 20+ other formats via the BIOIO plugin ecosystem.
    DaskReader base class wraps the dask array via ts.virtual_chunked.

    How it works:
        1. BIOIO loads the file and returns a dask array (.dask_data)
        2. DaskReader wraps it via ts.virtual_chunked
        3. Metadata extracted from BIOIO's standardized properties

    Example:
        >>> from tensorswitch_v2.readers import BIOIOReader
        >>> reader = BIOIOReader("/path/to/data.czi")
        >>> store = reader.get_tensorstore()
    """

    # BIOIO dimension order (standard for microscopy)
    BIOIO_DIMS = ['T', 'C', 'Z', 'Y', 'X']

    def __init__(
        self,
        path: str,
        scene_index: int = 0,
        channel_index: Optional[int] = None,
        time_index: Optional[int] = None,
        resolution_level: int = 0,
        reader: Optional[Any] = None
    ):
        super().__init__(path)
        self._scene_index = scene_index
        self._channel_index = channel_index
        self._time_index = time_index
        self._resolution_level = resolution_level
        self._explicit_reader = reader

        # Lazy-loaded BIOIO objects
        self._bioimage = None
        self._metadata_cache = None

    def _load_bioimage(self):
        """Lazy-load the BIOIO BioImage object."""
        if self._bioimage is not None:
            return

        try:
            from bioio import BioImage
        except ImportError:
            raise ImportError(
                "BIOIO is not installed. Install it with:\n"
                "  pip install bioio\n\n"
                "For specific format support, install the appropriate plugin:\n"
                "  pip install bioio-czi    # For CZI files\n"
                "  pip install bioio-lif    # For LIF files\n"
                "  pip install bioio-nd2    # For ND2 files\n"
                "  pip install bioio-ome-tiff  # For OME-TIFF files\n\n"
                "See https://github.com/bioio-devs/bioio for all plugins."
            )

        if self._explicit_reader:
            self._bioimage = BioImage(self.path, reader=self._explicit_reader)
        else:
            self._bioimage = BioImage(self.path)

        if self._scene_index > 0 and len(self._bioimage.scenes) > self._scene_index:
            self._bioimage.set_scene(self._scene_index)

        if self._resolution_level > 0:
            try:
                available_levels = self._bioimage.resolution_levels
                if self._resolution_level < len(available_levels):
                    self._bioimage.set_resolution_level(self._resolution_level)
                else:
                    print(f"Warning: Resolution level {self._resolution_level} not available. "
                          f"Available: {available_levels}. Using level 0.")
            except (AttributeError, NotImplementedError):
                if self._resolution_level > 0:
                    print(f"Warning: Format does not support multi-resolution. Using level 0.")

    def _load(self):
        """Lazy-load BIOIO data into self._dask_array."""
        if self._dask_array is not None:
            return

        self._load_bioimage()

        # Get the full dask array (TCZYX order)
        dask_data = self._bioimage.dask_data

        # Handle dimension selection if requested
        if self._time_index is not None or self._channel_index is not None:
            slices = []
            dims_to_keep = []

            # T dimension
            if self._time_index is not None:
                slices.append(self._time_index)
            else:
                slices.append(slice(None))
                if dask_data.shape[0] > 1:
                    dims_to_keep.append('T')

            # C dimension
            if self._channel_index is not None:
                slices.append(self._channel_index)
            else:
                slices.append(slice(None))
                if dask_data.shape[1] > 1:
                    dims_to_keep.append('C')

            # Z, Y, X dimensions (always keep)
            slices.extend([slice(None), slice(None), slice(None)])
            dims_to_keep.extend(['Z', 'Y', 'X'])

            dask_data = dask_data[tuple(slices)]

        # Squeeze singleton dimensions (T=1, C=1)
        shape = dask_data.shape
        squeeze_axes = []

        if len(shape) > 0 and shape[0] == 1 and self._time_index is None:
            squeeze_axes.append(0)

        if len(shape) > 1 and shape[1] == 1 and self._channel_index is None:
            squeeze_axes.append(1)

        if squeeze_axes:
            dask_data = dask_data.squeeze(axis=tuple(squeeze_axes))

        self._dask_array = dask_data

    def _get_dimension_names(self) -> List[str]:
        """Infer dimension names from squeezed array shape."""
        self._load()
        ndim = len(self._dask_array.shape)

        if ndim == 2:
            return ['y', 'x']
        elif ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            # Could be CZYX or TZYX - check metadata
            metadata = self.get_metadata()
            original_shape = metadata.get('shape', ())

            if len(original_shape) >= 5:
                t_size, c_size = original_shape[0], original_shape[1]
                if c_size > 1 and t_size == 1:
                    return ['c', 'z', 'y', 'x']
                elif t_size > 1 and c_size == 1:
                    return ['t', 'z', 'y', 'x']

            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(ndim)]

    def get_metadata(self) -> Dict:
        """Return BIOIO metadata in standardized format."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._load_bioimage()

        metadata = {
            'dims': self._bioimage.dims.order if hasattr(self._bioimage.dims, 'order') else str(self._bioimage.dims),
            'shape': self._bioimage.shape,
            'dtype': str(self._bioimage.dtype),
            'scenes': list(self._bioimage.scenes) if hasattr(self._bioimage, 'scenes') else [],
            'current_scene': self._bioimage.current_scene if hasattr(self._bioimage, 'current_scene') else None,
        }

        try:
            metadata['channel_names'] = list(self._bioimage.channel_names) if self._bioimage.channel_names else []
        except Exception:
            metadata['channel_names'] = []

        try:
            pps = self._bioimage.physical_pixel_sizes
            metadata['physical_pixel_sizes'] = {
                'X': pps.X if pps.X else None,
                'Y': pps.Y if pps.Y else None,
                'Z': pps.Z if pps.Z else None,
            }
        except Exception:
            metadata['physical_pixel_sizes'] = {'X': None, 'Y': None, 'Z': None}

        try:
            if hasattr(self._bioimage, 'ome_metadata') and self._bioimage.ome_metadata:
                metadata['ome_metadata'] = str(self._bioimage.ome_metadata)
        except Exception:
            pass

        try:
            if hasattr(self._bioimage, 'metadata'):
                metadata['has_raw_metadata'] = True
        except Exception:
            metadata['has_raw_metadata'] = False

        self._metadata_cache = metadata
        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """Return voxel dimensions from BIOIO in nanometers."""
        self._load_bioimage()

        try:
            pps = self._bioimage.physical_pixel_sizes
            return {
                'x': pps.X * 1000.0 if pps.X else 1.0,
                'y': pps.Y * 1000.0 if pps.Y else 1.0,
                'z': pps.Z * 1000.0 if pps.Z else 1.0,
            }
        except Exception:
            return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def supports_remote(self) -> bool:
        """Check if BIOIO supports remote access for this format."""
        if 's3://' in self.path or 'gs://' in self.path or 'http' in self.path:
            if '.zarr' in self.path.lower():
                return True
        return False

    @property
    def scenes(self) -> List[str]:
        """Get list of available scenes for multi-scene files."""
        self._load_bioimage()
        try:
            return list(self._bioimage.scenes)
        except Exception:
            return []

    @property
    def channel_names(self) -> List[str]:
        """Get list of channel names."""
        metadata = self.get_metadata()
        return metadata.get('channel_names', [])

    @property
    def resolution_levels(self) -> List[int]:
        """Get list of available resolution levels (pyramid levels)."""
        self._load_bioimage()
        try:
            return list(self._bioimage.resolution_levels)
        except (AttributeError, NotImplementedError):
            return [0]

    @property
    def current_resolution_level(self) -> int:
        """Get the currently selected resolution level."""
        self._load_bioimage()
        try:
            return self._bioimage.current_resolution_level
        except (AttributeError, NotImplementedError):
            return 0

    def set_resolution_level(self, level: int) -> None:
        """Set the resolution level for subsequent reads."""
        self._load_bioimage()
        available = self.resolution_levels
        if level not in available:
            raise ValueError(f"Resolution level {level} not available. "
                           f"Available levels: {available}")

        try:
            self._bioimage.set_resolution_level(level)
            self._resolution_level = level
            # Clear cached data to force reload at new resolution
            self._dask_array = None
            self._ts_store_cache = None
            self._metadata_cache = None
        except (AttributeError, NotImplementedError):
            if level > 0:
                raise ValueError(f"Format does not support multi-resolution. "
                               f"Only level 0 is available.")

    def get_resolution_level_shape(self, level: int) -> tuple:
        """Get the shape at a specific resolution level without loading data."""
        self._load_bioimage()
        try:
            current = self.current_resolution_level
            self._bioimage.set_resolution_level(level)
            shape = self._bioimage.shape
            self._bioimage.set_resolution_level(current)
            return shape
        except (AttributeError, NotImplementedError):
            if level == 0:
                return self._bioimage.shape
            raise ValueError(f"Resolution level {level} not available.")

    def __repr__(self) -> str:
        return f"BIOIOReader(path='{self.path}', scene={self._scene_index}, resolution_level={self._resolution_level})"
