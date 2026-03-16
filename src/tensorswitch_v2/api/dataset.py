"""
TensorSwitchDataset wrapper class - unified interface for all formats.

Inspired by BIOIO's BioImage class but using TensorStore as the
intermediate format instead of Dask.
"""

from typing import Dict, Optional, Tuple
import tensorstore as ts
from ..readers.base import BaseReader
from ..utils import get_dtype_name


class TensorSwitchDataset:
    """
    Unified interface for scientific imaging data.

    TensorSwitchDataset provides a format-agnostic wrapper that auto-detects
    input formats and converts them to TensorStore arrays. Inspired by BIOIO's
    BioImage class but centered on TensorStore as the unified intermediate format.

    Architecture Layer: 2 (Wrapper - Depends on Readers)
    - Auto-detects format and selects appropriate reader
    - Returns TensorStore arrays (virtual, open, or in-memory)
    - Provides standardized metadata access
    - Format-agnostic interface

    Usage Modes:
        1. Auto-detection: Automatically picks optimal reader tier
        2. Explicit reader: User specifies reader directly
        3. Virtual mode: Zero-copy lazy specs (default)
        4. Open mode: Opened TensorStore handles
        5. Copy mode: In-memory arrays

    Example (Auto-detection):
        >>> dataset = TensorSwitchDataset("/path/to/data.tif")
        >>> ts_array = dataset.get_tensorstore_array()  # Virtual spec
        >>> metadata = dataset.get_ome_ngff_metadata()
        >>> print(f"Shape: {dataset.shape}, Dtype: {dataset.dtype}")

    Example (Explicit reader):
        >>> from tensorswitch_v2.readers import TiffReader
        >>> reader = TiffReader("/path/to/data.tif")
        >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)

    Example (Different modes):
        >>> # Virtual mode (lazy, zero-copy)
        >>> ts_spec = dataset.get_tensorstore_array(mode='virtual')
        >>>
        >>> # Open mode (opened handle)
        >>> ts_array = dataset.get_tensorstore_array(mode='open')
        >>>
        >>> # Copy mode (in-memory)
        >>> ts_array = dataset.get_tensorstore_array(mode='copy')

    Design Principles:
        - Simple API for users (hide complexity)
        - Format detection abstracted away
        - Consistent interface across all formats
        - Lazy evaluation by default (virtual mode)
    """

    def __init__(self, path: str, reader: Optional[BaseReader] = None):
        """
        Initialize dataset from path.

        Args:
            path: Path to input data (local file, HTTP URL, GCS/S3 URI)
            reader: Optional explicit reader instance. If None, will use
                   auto-detection via Readers.auto_detect() when Readers
                   factory is implemented (Day 4-5).

        Raises:
            ValueError: If auto-detection is attempted before Readers factory exists
            FileNotFoundError: If path doesn't exist (for local files)
            RuntimeError: If reader fails to initialize

        Example:
            >>> # Auto-detection (when Readers factory is implemented)
            >>> dataset = TensorSwitchDataset("/data.tif")
            >>>
            >>> # Explicit reader
            >>> from tensorswitch_v2.readers import N5Reader
            >>> reader = N5Reader("/data.n5")
            >>> dataset = TensorSwitchDataset("/data.n5", reader=reader)
        """
        self.path = path
        self._reader = reader
        self._ts_spec_cache = None
        self._ts_array_cache = {}  # Cache for different modes

        # If no reader provided, will need auto-detection
        # (placeholder - will be implemented when Readers factory exists)
        if self._reader is None:
            raise ValueError(
                "Auto-detection not yet implemented. Please provide explicit reader. "
                "Readers factory will be added in Day 4-5 (see PLAN_phase5.md)."
            )

    def get_tensorstore_array(self, mode: str = "open") -> ts.TensorStore:
        """
        Return TensorStore array handle.

        Two modes available:
        - 'open': Returns opened TensorStore handle (default)
        - 'copy': Returns in-memory TensorStore array

        Args:
            mode: Access mode ('open' or 'copy')

        Returns:
            ts.TensorStore: Opened TensorStore array

        Example (open mode - default):
            >>> ts_array = dataset.get_tensorstore_array()
            >>> chunk_data = ts_array[0:10, :, :].read().result()

        Example (copy mode):
            >>> ts_array = dataset.get_tensorstore_array(mode='copy')
            >>> # Data loaded into memory
        """
        if mode in self._ts_array_cache:
            return self._ts_array_cache[mode]

        if mode == "open":
            ts_array = self.get_tensorstore()
            self._ts_array_cache[mode] = ts_array
            return ts_array

        elif mode == "copy":
            ts_array = self.get_tensorstore()
            data = ts_array.read().result()

            labels = list(ts_array.domain.labels) if ts_array.domain.labels else []
            mem_spec = {
                'driver': 'array',
                'array': data,
                'schema': {
                    'dtype': str(data.dtype),
                    'shape': list(data.shape),
                    'dimension_names': labels
                }
            }
            mem_array = ts.open(mem_spec).result()
            self._ts_array_cache[mode] = mem_array
            return mem_array

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'open' or 'copy'")

    def get_tensorstore(self) -> ts.TensorStore:
        """
        Return opened TensorStore array.

        Delegates to the reader's get_tensorstore() method.
        Result is cached for efficiency.

        Returns:
            ts.TensorStore: Opened TensorStore with .shape, .dtype, .domain.labels

        Example:
            >>> store = dataset.get_tensorstore()
            >>> print(store.shape)
            >>> print(store.dtype)
        """
        if self._ts_spec_cache is None:
            self._ts_spec_cache = self._reader.get_tensorstore()
        return self._ts_spec_cache

    def get_ome_ngff_metadata(self, version: str = "0.5") -> Dict:
        """
        Return OME-NGFF compatible metadata.

        Delegates to reader's get_ome_metadata() method.
        Supports OME-NGFF v0.4 and v0.5.

        Args:
            version: OME-NGFF version ('0.4' or '0.5')

        Returns:
            dict: OME-NGFF metadata structure with keys:
                - 'multiscales': Multi-resolution metadata
                - 'axes': Dimension definitions
                - 'coordinateTransformations': Voxel size scales

        Example:
            >>> metadata = dataset.get_ome_ngff_metadata()
            >>> axes = metadata['multiscales'][0]['axes']
            >>> print([ax['name'] for ax in axes])  # ['z', 'y', 'x']
        """
        ome_metadata = self._reader.get_ome_metadata()

        # Could add version conversion logic here if needed
        if version not in ["0.4", "0.5"]:
            raise ValueError(f"Unsupported OME-NGFF version: {version}")

        return ome_metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return physical pixel/voxel dimensions in micrometers.

        Delegates to reader's get_voxel_sizes() method.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z'
                  (and 't' if time dimension exists)

        Example:
            >>> voxel_sizes = dataset.get_voxel_sizes()
            >>> print(f"Z: {voxel_sizes['z']} µm")
            >>> print(f"Anisotropic: {voxel_sizes['z'] != voxel_sizes['x']}")
        """
        return self._reader.get_voxel_sizes()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return array shape.

        Example:
            >>> dataset.shape
            (100, 1024, 1024)
        """
        store = self.get_tensorstore()
        return tuple(store.shape)

    @property
    def dtype(self) -> str:
        """
        Return array data type.

        Example:
            >>> dataset.dtype
            'uint16'
        """
        store = self.get_tensorstore()
        return get_dtype_name(store.dtype)

    @property
    def ndim(self) -> int:
        """
        Return number of dimensions.

        Example:
            >>> dataset.ndim
            3
        """
        return len(self.shape)

    @property
    def is_remote(self) -> bool:
        """
        Check if data source is remote (HTTP, GCS, S3).

        Example:
            >>> dataset = TensorSwitchDataset("gs://bucket/data.zarr", reader=...)
            >>> dataset.is_remote
            True
        """
        return self._reader.supports_remote() and any(
            self.path.startswith(prefix)
            for prefix in ['http://', 'https://', 'gs://', 's3://']
        )

    @property
    def reader(self) -> BaseReader:
        """
        Return the underlying reader instance.

        Example:
            >>> print(dataset.reader)
            TiffReader(path='/data.tif')
        """
        return self._reader

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"TensorSwitchDataset(path='{self.path}', "
            f"reader={self._reader.__class__.__name__}, "
            f"shape={self.shape}, dtype='{self.dtype}')"
        )
