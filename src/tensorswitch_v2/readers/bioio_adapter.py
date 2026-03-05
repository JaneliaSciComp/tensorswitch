"""
BIOIO adapter reader for TensorSwitch Phase 5 architecture.

Tier 3 reader - strategic investment unlocking 20+ formats via BIOIO ecosystem.
Converts BIOIO's dask arrays to TensorStore intermediate format.
"""

from typing import Dict, Optional, List, Any
from .base import BaseReader


class BIOIOReader(BaseReader):
    """
    Reader adapter for BIOIO library, supporting 20+ microscopy formats.

    This is the Tier 3 strategic investment: one adapter (~200 LOC) that
    unlocks support for CZI, LIF, SLDY, DV, and 20+ other formats via
    the BIOIO plugin ecosystem.

    Tier: 3 (BIOIO Adapter - Broad Compatibility)
    - Leverages BIOIO's plugin architecture
    - Converts BIOIO dask arrays to TensorStore
    - Automatic format detection via BIOIO
    - Community-maintained format parsers

    How it works:
        1. BIOIO loads the file and returns a dask array (.dask_data)
        2. We wrap the dask array in TensorStore's 'array' driver
        3. Metadata extracted from BIOIO's standardized properties

    Supported Formats (via BIOIO plugins):
        - CZI (Zeiss) - bioio-czi
        - LIF (Leica) - bioio-lif
        - ND2 (Nikon) - bioio-nd2
        - OME-TIFF - bioio-ome-tiff
        - OME-Zarr - bioio-ome-zarr
        - DV (DeltaVision) - bioio-dv
        - SLDY (SlideBook) - bioio-sldy
        - And 15+ more formats

    Performance Note:
        Tier 3 has slightly more overhead than Tier 1/2 due to the
        BIOIO -> Dask -> TensorStore chain. For production-critical
        formats (TIFF, ND2, IMS), prefer Tier 2 readers.

    Example:
        >>> from tensorswitch_v2.readers import BIOIOReader
        >>> reader = BIOIOReader("/path/to/data.czi")
        >>> spec = reader.get_tensorstore_spec()
        >>> metadata = reader.get_metadata()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers
        >>> dataset = TensorSwitchDataset("/path/to/data.lif")
        >>> ts_array = dataset.get_tensorstore_array()

    Dependencies:
        - bioio (core library)
        - Format-specific plugins (e.g., bioio-czi, bioio-lif)

    See Also:
        - https://github.com/bioio-devs/bioio
        - https://bioio-devs.github.io/bioio/
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
        """
        Initialize BIOIO adapter reader.

        Args:
            path: Path to input file (local path or URL)
            scene_index: Which scene to load for multi-scene files (default: 0)
            channel_index: Optional specific channel to extract (None = all channels)
            time_index: Optional specific timepoint to extract (None = all timepoints)
            resolution_level: Resolution level to read (0=full, higher=lower res). Default: 0
            reader: Optional explicit BIOIO reader class to use (auto-detected if None)

        Example:
            >>> # Auto-detect format
            >>> reader = BIOIOReader("/data.czi")

            >>> # Specific scene in multi-scene file
            >>> reader = BIOIOReader("/data.lif", scene_index=2)

            >>> # Extract single channel
            >>> reader = BIOIOReader("/data.czi", channel_index=0)

            >>> # Read lower resolution level (if available in file)
            >>> reader = BIOIOReader("/data.czi", resolution_level=2)

            >>> # Force specific reader (override auto-detection)
            >>> from bioio_czi import Reader as CZIReader
            >>> reader = BIOIOReader("/data.czi", reader=CZIReader)
        """
        super().__init__(path)
        self._scene_index = scene_index
        self._channel_index = channel_index
        self._time_index = time_index
        self._resolution_level = resolution_level
        self._explicit_reader = reader

        # Lazy-loaded BIOIO objects
        self._bioimage = None
        self._dask_array = None
        self._metadata_cache = None

    def _load_bioimage(self):
        """
        Lazy-load the BIOIO BioImage object.

        Defers loading until actually needed to minimize overhead.
        """
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

        # Create BioImage with optional explicit reader
        if self._explicit_reader:
            self._bioimage = BioImage(self.path, reader=self._explicit_reader)
        else:
            self._bioimage = BioImage(self.path)

        # Set scene if multi-scene
        if self._scene_index > 0 and len(self._bioimage.scenes) > self._scene_index:
            self._bioimage.set_scene(self._scene_index)

        # Set resolution level if multi-resolution (pyramids)
        if self._resolution_level > 0:
            try:
                available_levels = self._bioimage.resolution_levels
                if self._resolution_level < len(available_levels):
                    self._bioimage.set_resolution_level(self._resolution_level)
                else:
                    print(f"Warning: Resolution level {self._resolution_level} not available. "
                          f"Available: {available_levels}. Using level 0.")
            except (AttributeError, NotImplementedError):
                # Format doesn't support multi-resolution
                if self._resolution_level > 0:
                    print(f"Warning: Format does not support multi-resolution. Using level 0.")

    def _get_dask_array(self):
        """
        Get the dask array from BIOIO, with optional dimension slicing.

        Returns:
            dask.array.Array: The image data as a lazy dask array
        """
        if self._dask_array is not None:
            return self._dask_array

        self._load_bioimage()

        # Get the full dask array (TCZYX order)
        dask_data = self._bioimage.dask_data

        # Handle dimension selection if requested
        # BIOIO uses TCZYX order
        if self._time_index is not None or self._channel_index is not None:
            # Build slice tuple for TCZYX
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
        # Keep only non-singleton dimensions or explicitly requested
        shape = dask_data.shape
        squeeze_axes = []

        # Check T (index 0) - squeeze if size 1 and not explicitly selected
        if len(shape) > 0 and shape[0] == 1 and self._time_index is None:
            squeeze_axes.append(0)

        # Check C (index 1 after potential T squeeze) - squeeze if size 1
        c_idx = 1 - len([a for a in squeeze_axes if a < 1])
        if len(shape) > 1 and shape[1] == 1 and self._channel_index is None:
            squeeze_axes.append(1)

        if squeeze_axes:
            dask_data = dask_data.squeeze(axis=tuple(squeeze_axes))

        self._dask_array = dask_data
        return self._dask_array

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping BIOIO's dask array.

        Converts BIOIO's dask array to TensorStore's 'array' driver format,
        which serves as the intermediate representation.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping dask array

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['driver'])
            'array'

        Notes:
            - BIOIO returns dask arrays in TCZYX order
            - Singleton T/C dimensions are squeezed
            - Final order is typically ZYX (3D) or CZYX (4D) or TCZYX (5D)
        """
        dask_array = self._get_dask_array()

        # Infer dimension names based on final shape
        dimension_names = self._infer_dimension_names(dask_array.shape)

        # Wrap dask array in TensorStore 'array' driver
        spec = {
            'driver': 'array',
            'array': dask_array,
            'schema': {
                'dtype': str(dask_array.dtype),
                'shape': list(dask_array.shape),
                'dimension_names': dimension_names
            }
        }

        return spec

    def get_metadata(self) -> Dict:
        """
        Return BIOIO metadata in standardized format.

        Extracts metadata from BIOIO's various properties and returns
        a combined dictionary with format-specific information.

        Returns:
            dict: Metadata including:
                - dims: Dimension order string (e.g., 'ZYX', 'CZYX')
                - shape: Original shape from BIOIO (TCZYX)
                - channel_names: List of channel names if available
                - physical_pixel_sizes: Voxel sizes object
                - ome_types: OME metadata if available
                - scenes: List of scene names for multi-scene files

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> metadata = reader.get_metadata()
            >>> print(metadata['channel_names'])
            ['DAPI', 'GFP', 'RFP']
        """
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

        # Channel names
        try:
            metadata['channel_names'] = list(self._bioimage.channel_names) if self._bioimage.channel_names else []
        except Exception:
            metadata['channel_names'] = []

        # Physical pixel sizes
        try:
            pps = self._bioimage.physical_pixel_sizes
            metadata['physical_pixel_sizes'] = {
                'X': pps.X if pps.X else None,
                'Y': pps.Y if pps.Y else None,
                'Z': pps.Z if pps.Z else None,
            }
        except Exception:
            metadata['physical_pixel_sizes'] = {'X': None, 'Y': None, 'Z': None}

        # Try to get OME metadata if available
        try:
            if hasattr(self._bioimage, 'ome_metadata') and self._bioimage.ome_metadata:
                metadata['ome_metadata'] = str(self._bioimage.ome_metadata)
        except Exception:
            pass

        # Format-specific metadata
        try:
            if hasattr(self._bioimage, 'metadata'):
                # Store raw metadata (may be large, so store reference)
                metadata['has_raw_metadata'] = True
        except Exception:
            metadata['has_raw_metadata'] = False

        self._metadata_cache = metadata
        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from BIOIO's physical_pixel_sizes.

        BIOIO provides standardized access to voxel sizes across all formats.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> voxel_sizes = reader.get_voxel_sizes()
            >>> print(voxel_sizes)
            {'x': 116.0, 'y': 116.0, 'z': 500.0}  # nanometers

        Notes:
            - Returns 1.0 for dimensions where size is not available
            - BIOIO returns sizes in micrometers, converted to nanometers here
        """
        self._load_bioimage()

        try:
            pps = self._bioimage.physical_pixel_sizes
            # BIOIO returns micrometers, convert to nanometers (×1000)
            return {
                'x': pps.X * 1000.0 if pps.X else 1.0,
                'y': pps.Y * 1000.0 if pps.Y else 1.0,
                'z': pps.Z * 1000.0 if pps.Z else 1.0,
            }
        except Exception:
            return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def _infer_dimension_names(self, shape) -> List[str]:
        """
        Infer dimension names from array shape.

        BIOIO uses TCZYX order, but after squeezing singleton dims,
        we need to infer what dimensions remain.

        Args:
            shape: Tuple of dimension sizes

        Returns:
            list: Dimension names (e.g., ['z', 'y', 'x'])
        """
        ndim = len(shape)

        # Common patterns based on dimensionality
        if ndim == 2:
            return ['y', 'x']
        elif ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            # Could be CZYX or TZYX - check metadata
            metadata = self.get_metadata()
            original_shape = metadata.get('shape', ())

            # If original had multiple channels but single timepoint
            if len(original_shape) >= 5:
                t_size, c_size = original_shape[0], original_shape[1]
                if c_size > 1 and t_size == 1:
                    return ['c', 'z', 'y', 'x']
                elif t_size > 1 and c_size == 1:
                    return ['t', 'z', 'y', 'x']

            # Default to CZYX for 4D
            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            # Fallback for unusual dimensions
            return [f'dim_{i}' for i in range(ndim)]

    def supports_remote(self) -> bool:
        """
        Check if BIOIO supports remote access for this format.

        Returns:
            bool: True if remote URLs are supported

        Notes:
            Remote support depends on the underlying BIOIO plugin.
            Most plugins support local files only.
        """
        # BIOIO plugins generally don't support remote
        # Some exceptions exist (ome-zarr)
        if 's3://' in self.path or 'gs://' in self.path or 'http' in self.path:
            # Check if it's a format known to support remote
            if '.zarr' in self.path.lower():
                return True
        return False

    @property
    def scenes(self) -> List[str]:
        """
        Get list of available scenes for multi-scene files.

        Returns:
            list: Scene names/identifiers

        Example:
            >>> reader = BIOIOReader("/multi_scene.lif")
            >>> print(reader.scenes)
            ['Scene_1', 'Scene_2', 'Scene_3']
        """
        self._load_bioimage()
        try:
            return list(self._bioimage.scenes)
        except Exception:
            return []

    @property
    def channel_names(self) -> List[str]:
        """
        Get list of channel names.

        Returns:
            list: Channel names if available

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> print(reader.channel_names)
            ['DAPI', 'GFP']
        """
        metadata = self.get_metadata()
        return metadata.get('channel_names', [])

    @property
    def resolution_levels(self) -> List[int]:
        """
        Get list of available resolution levels (pyramid levels).

        For formats with built-in pyramids (CZI, LIF, OME-TIFF), this returns
        the available resolution levels. Level 0 is full resolution.

        Returns:
            list: Available resolution level indices (e.g., [0, 1, 2, 3])

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> print(reader.resolution_levels)
            [0, 1, 2, 3]  # 4 pyramid levels available
        """
        self._load_bioimage()
        try:
            return list(self._bioimage.resolution_levels)
        except (AttributeError, NotImplementedError):
            return [0]  # Only full resolution available

    @property
    def current_resolution_level(self) -> int:
        """
        Get the currently selected resolution level.

        Returns:
            int: Current resolution level index

        Example:
            >>> reader = BIOIOReader("/data.czi", resolution_level=2)
            >>> print(reader.current_resolution_level)
            2
        """
        self._load_bioimage()
        try:
            return self._bioimage.current_resolution_level
        except (AttributeError, NotImplementedError):
            return 0

    def set_resolution_level(self, level: int) -> None:
        """
        Set the resolution level for subsequent reads.

        This allows switching between pyramid levels after initialization.
        Level 0 is full resolution, higher levels are lower resolution.

        Args:
            level: Resolution level index (0 = full resolution)

        Raises:
            ValueError: If level is not available in the file

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> print(reader.resolution_levels)
            [0, 1, 2, 3]
            >>> reader.set_resolution_level(2)  # Switch to level 2
            >>> spec = reader.get_tensorstore_spec()  # Now returns level 2 data
        """
        self._load_bioimage()
        available = self.resolution_levels
        if level not in available:
            raise ValueError(f"Resolution level {level} not available. "
                           f"Available levels: {available}")

        try:
            self._bioimage.set_resolution_level(level)
            self._resolution_level = level
            # Clear cached dask array to force reload at new resolution
            self._dask_array = None
            self._metadata_cache = None
        except (AttributeError, NotImplementedError):
            if level > 0:
                raise ValueError(f"Format does not support multi-resolution. "
                               f"Only level 0 is available.")

    def get_resolution_level_shape(self, level: int) -> tuple:
        """
        Get the shape at a specific resolution level without loading data.

        Args:
            level: Resolution level index

        Returns:
            tuple: Shape at the specified resolution level

        Example:
            >>> reader = BIOIOReader("/data.czi")
            >>> print(reader.get_resolution_level_shape(0))
            (100, 2048, 2048)  # Full resolution
            >>> print(reader.get_resolution_level_shape(2))
            (100, 512, 512)    # 4x downsampled in XY
        """
        self._load_bioimage()
        try:
            # Temporarily switch level, get shape, switch back
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
        """String representation of BIOIO reader."""
        return f"BIOIOReader(path='{self.path}', scene={self._scene_index}, resolution_level={self._resolution_level})"
