"""
Base writer class for TensorSwitch Phase 5 architecture.

All format-specific writers inherit from BaseWriter and must implement
the abstract methods to write TensorStore arrays to their target format.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import tensorstore as ts


class BaseWriter(ABC):
    """
    Abstract base class for all format writers.

    Writers are part of Layer 3 in the Phase 5 architecture.
    Each writer converts TensorStore arrays into a specific output format
    (Zarr3, Zarr2, N5, etc.) without any knowledge of the input format.

    Architecture Layer: 3 (Processing - Depends on Readers/Wrapper)
    - Format-specific output implementation
    - Accepts TensorStore arrays from any source
    - No knowledge of input format (reader-agnostic)

    Design Principles (Critical - Per Phase 5.txt):
    - Writers ONLY query TensorStore array properties (shape, dtype, etc.)
    - Writers NEVER ask about input format ("is this TIFF?", "is this N5?")
    - Writers are format-agnostic processors
    - Complete decoupling from readers

    Example Usage:
        >>> from tensorswitch_v2.writers import Zarr3Writer
        >>> from tensorswitch_v2.api import TensorSwitchDataset
        >>>
        >>> dataset = TensorSwitchDataset("/path/to/input.tif")
        >>> ts_array = dataset.get_tensorstore_array()
        >>>
        >>> writer = Zarr3Writer("/path/to/output.zarr")
        >>> writer.create_output_spec(ts_array, chunk_shape=(32, 256, 256))
        >>> writer.write_metadata(dataset.get_ome_ngff_metadata())

    Subclass Requirements:
        Must implement all @abstractmethod methods.
        Optionally override helper methods for custom behavior.
    """

    def __init__(self, output_path: str):
        """
        Initialize writer with output path.

        Args:
            output_path: Path to output location (local file path, GCS/S3 URI)
        """
        self.output_path = output_path
        self._output_spec = None
        self._output_store = None

    @abstractmethod
    def create_output_spec(
        self,
        input_array: ts.TensorStore,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for output.

        Queries ONLY the TensorStore array properties (shape, dtype, etc.)
        to build the output specification. No knowledge of input format.

        Args:
            input_array: TensorStore array from any source (TIFF, N5, etc.)
            chunk_shape: Optional chunk dimensions (auto-calculated if None)
            shard_shape: Optional shard dimensions (for sharded formats)
            **kwargs: Format-specific options (compression, etc.)

        Returns:
            dict: TensorStore spec for output with keys:
                - 'driver': Output format driver ('zarr3', 'zarr2', 'n5')
                - 'kvstore': Output key-value store specification
                - 'schema': Output schema (dtype, shape, chunks, etc.)
                - 'create': True (creating new output)
                - 'delete_existing': Usually True for fresh conversion

        Example (Zarr3 with sharding):
            {
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': '/output.zarr'},
                'create': True,
                'delete_existing': True,
                'schema': {
                    'dtype': 'uint16',
                    'shape': [100, 1024, 1024],
                    'dimension_names': ['z', 'y', 'x'],
                    'chunk_layout': {
                        'grid_origin': [0, 0, 0],
                        'inner_order': [2, 1, 0],
                        'read_chunk': {'shape': [1, 1024, 1024]},
                        'write_chunk': {'shape': [1, 1024, 1024]}
                    },
                    'codec': {
                        'driver': 'zarr3',
                        'codecs': [
                            {'name': 'sharding_indexed', ...},
                            {'name': 'blosc', ...}
                        ]
                    }
                }
            }

        Notes:
            - Query input_array.shape, input_array.dtype, input_array.schema
            - DO NOT query about source format
            - Use helper methods get_default_chunk_shape() if needed
            - Respect user-provided chunk_shape and shard_shape
        """
        pass

    @abstractmethod
    def write_chunk(
        self,
        output_store: ts.TensorStore,
        chunk_domain: Tuple[slice, ...],
        input_array: ts.TensorStore
    ) -> None:
        """
        Write a single chunk from input to output.

        This is called by DistributedConverter for each chunk in parallel
        (LSF multi-job or Dask single-job mode).

        Args:
            output_store: Opened TensorStore output handle
            chunk_domain: Slice tuple defining chunk region (e.g., (slice(0, 32), slice(0, 256), ...))
            input_array: TensorStore input array (format-agnostic)

        Returns:
            None

        Example:
            chunk_domain = (slice(0, 32), slice(0, 256), slice(0, 256))
            output_store[chunk_domain] = input_array[chunk_domain].read().result()

        Notes:
            - Use TensorStore transactions for atomic writes
            - Handle potential read errors from input_array
            - Don't ask about input format, just read the data
            - Compression/encoding handled by output spec, not here
        """
        pass

    @abstractmethod
    def write_metadata(
        self,
        ome_metadata: Dict,
        voxel_sizes: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Write OME-NGFF metadata to output.

        Serializes OME-NGFF metadata to format-specific location
        (e.g., .zattrs for Zarr, attributes.json for N5).

        Args:
            ome_metadata: OME-NGFF compliant metadata dict
            voxel_sizes: Optional voxel dimensions (x, y, z in micrometers)

        Returns:
            None

        Example (Zarr3):
            Writes to: /output.zarr/.zattrs
            Content: JSON-serialized OME-NGFF metadata

        Example (N5):
            Writes to: /output.n5/attributes.json
            Content: JSON-serialized metadata (adapted to N5 conventions)

        Notes:
            - OME-NGFF metadata is format-agnostic input
            - Writer adapts to format-specific conventions
            - May need to convert coordinate transformations
            - Preserve multiscales structure
        """
        pass

    def supports_sharding(self) -> bool:
        """
        Check if this writer supports sharding.

        Returns:
            bool: True if format supports sharding (e.g., Zarr3), False otherwise

        Example:
            >>> writer = Zarr3Writer("/output.zarr")
            >>> writer.supports_sharding()
            True

            >>> writer = Zarr2Writer("/output.zarr")
            >>> writer.supports_sharding()
            False

        Notes:
            - Zarr3: Supports sharding via sharding_indexed codec
            - Zarr2: No native sharding support
            - N5: No sharding support (uses blocks)
        """
        return False

    def supports_remote(self) -> bool:
        """
        Check if this writer supports remote output (GCS, S3, HTTP).

        Returns:
            bool: True if writer can handle remote kvstores, False otherwise

        Example:
            >>> writer = Zarr3Writer("gs://bucket/output.zarr")
            >>> writer.supports_remote()
            True

        Notes:
            - Zarr3, Zarr2, N5 all support remote via TensorStore kvstore
            - HTTP typically read-only, GCS/S3 read-write
        """
        return True  # Most TensorStore formats support remote

    def get_default_chunk_shape(
        self,
        array_shape: Tuple[int, ...],
        dtype_size: int = 2
    ) -> Tuple[int, ...]:
        """
        Calculate optimal chunk shape for given array.

        Helper method to determine chunk dimensions when user doesn't specify.
        Targets ~16-64 MB chunks for good I/O performance.

        Args:
            array_shape: Full array dimensions (e.g., (100, 1024, 1024))
            dtype_size: Bytes per element (e.g., 2 for uint16)

        Returns:
            tuple: Recommended chunk shape

        Example:
            >>> writer.get_default_chunk_shape((100, 2048, 2048), dtype_size=2)
            (1, 1024, 1024)  # ~2 MB chunks for uint16

        Notes:
            - Default implementation provides basic heuristic
            - Override for format-specific optimization
            - Consider anisotropic dimensions (Z often smaller chunks)
        """
        # Target 32 MB chunks
        target_chunk_bytes = 32 * 1024 * 1024
        target_chunk_elements = target_chunk_bytes // dtype_size

        # Simple heuristic: Full XY plane, minimal Z
        if len(array_shape) == 3:
            z, y, x = array_shape
            xy_elements = y * x
            if xy_elements > 0:
                z_chunk = max(1, min(z, target_chunk_elements // xy_elements))
                return (z_chunk, y, x)

        # Fallback: divide each dimension equally
        ndim = len(array_shape)
        elements_per_dim = int(target_chunk_elements ** (1.0 / ndim))
        return tuple(min(dim, elements_per_dim) for dim in array_shape)

    def get_default_shard_shape(
        self,
        chunk_shape: Tuple[int, ...],
        array_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """
        Calculate optimal shard shape for sharded formats.

        Only applicable if supports_sharding() returns True.
        Shards contain multiple chunks for improved small-file performance.

        Args:
            chunk_shape: Chunk dimensions (read/write chunk)
            array_shape: Full array dimensions

        Returns:
            tuple or None: Recommended shard shape, or None if not applicable

        Example (Zarr3 sharding):
            >>> chunk_shape = (1, 256, 256)
            >>> array_shape = (100, 2048, 2048)
            >>> writer.get_default_shard_shape(chunk_shape, array_shape)
            (32, 1024, 1024)  # Shard contains 32×4×4 = 512 chunks

        Notes:
            - Target ~256-1024 chunks per shard
            - Balance between file count and chunk access overhead
            - Override for format-specific optimization
        """
        if not self.supports_sharding():
            return None

        # Target ~512 chunks per shard
        target_chunks_per_shard = 512

        # Calculate shard as multiple of chunks
        chunks_per_dim = int(target_chunks_per_shard ** (1.0 / len(chunk_shape)))
        shard_shape = []

        for i, (chunk_dim, array_dim) in enumerate(zip(chunk_shape, array_shape)):
            shard_dim = chunk_dim * chunks_per_dim
            shard_dim = min(shard_dim, array_dim)  # Don't exceed array size
            shard_shape.append(shard_dim)

        return tuple(shard_shape)

    def __repr__(self) -> str:
        """String representation of writer."""
        return f"{self.__class__.__name__}(output_path='{self.output_path}')"
