"""
N5 writer for TensorSwitch Phase 5 architecture.

Implements N5 format output for Java tool compatibility (BigDataViewer, etc.).
"""

import os
import json
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import tensorstore as ts

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

from .base import BaseWriter

# Import utility functions from v2 utils (independent from v1)
from ..utils import (
    n5_store_spec,
    get_tensorstore_context,
    get_kvstore_spec,
)


class N5Writer(BaseWriter):
    """
    N5 writer for Java tool compatibility.

    Used for compatibility with Java-based tools:
    - BigDataViewer
    - Fiji/ImageJ plugins
    - N5-based pipelines

    Note: N5 does NOT support sharding.

    Example:
        >>> from tensorswitch_v2.writers import N5Writer
        >>> writer = N5Writer("/output.n5")
        >>> spec = writer.create_output_spec(
        ...     shape=(100, 1024, 1024),
        ...     dtype="uint16",
        ...     chunk_shape=(64, 256, 256)
        ... )
        >>> writer.open_store(spec)
        >>> # ... write chunks ...
        >>> writer.write_metadata(ome_metadata, voxel_sizes)
    """

    def __init__(
        self,
        output_path: str,
        compression: str = "gzip",
        compression_level: int = 5,
        dataset_path: str = "s0"
    ):
        """
        Initialize N5 writer.

        Args:
            output_path: Path to output N5 dataset
            compression: Compression codec ("gzip", "raw", "blosc")
            compression_level: Compression level (1-9, default 5)
            dataset_path: Dataset path within N5 (default "s0" for Janelia convention)
        """
        super().__init__(output_path)
        self.compression = compression
        self.compression_level = compression_level
        self.dataset_path = dataset_path
        self._store = None
        self._spec = None

    def supports_sharding(self) -> bool:
        """N5 does NOT support sharding."""
        return False

    def create_output_spec(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        use_fortran_order: bool = True,  # N5 uses F-order by default
        axes_order: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for N5 output.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Chunk dimensions (N5 calls these "blockSize")
            shard_shape: Ignored (N5 doesn't support sharding)
            use_fortran_order: Use Fortran (column-major) order (N5 default)
            axes_order: Axis names (e.g., ["z", "y", "x"])
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for N5 output
        """
        if shard_shape is not None:
            print("Warning: shard_shape ignored - N5 doesn't support sharding")

        # Convert to list
        shape_list = list(shape)

        # Default chunk shape if not provided
        if chunk_shape is None:
            chunk_shape = self.get_default_chunk_shape(shape)
        chunks_list = list(chunk_shape)

        # Ensure chunks match shape dimensionality
        if len(chunks_list) < len(shape_list):
            extra_dims = len(shape_list) - len(chunks_list)
            chunks_list = [1] * extra_dims + chunks_list

        # Compute full path to dataset
        full_path = os.path.join(self.output_path, self.dataset_path)

        # Build N5 spec
        spec = {
            'driver': 'n5',
            'kvstore': get_kvstore_spec(full_path),
            'metadata': {
                'dataType': self._normalize_dtype(dtype),
                'dimensions': shape_list,
                'blockSize': chunks_list,
                'compression': self._get_compression_spec(),
            }
        }

        # Add context for concurrency control
        spec['context'] = get_tensorstore_context()

        self._spec = spec
        self._shape = tuple(shape_list)
        self._dtype = dtype
        self._axes_order = axes_order
        return spec

    def _normalize_dtype(self, dtype: str) -> str:
        """Convert dtype to N5 format."""
        dtype_map = {
            'uint8': 'uint8',
            'uint16': 'uint16',
            'uint32': 'uint32',
            'uint64': 'uint64',
            'int8': 'int8',
            'int16': 'int16',
            'int32': 'int32',
            'int64': 'int64',
            'float32': 'float32',
            'float64': 'float64',
        }
        return dtype_map.get(dtype, dtype)

    def _get_compression_spec(self) -> Dict:
        """Get N5 compression specification."""
        if self.compression == "gzip":
            return {
                "type": "gzip",
                "level": self.compression_level
            }
        elif self.compression == "raw":
            return {"type": "raw"}
        elif self.compression == "blosc":
            return {
                "type": "blosc",
                "cname": "lz4",
                "clevel": self.compression_level,
                "shuffle": 1
            }
        else:
            return {
                "type": self.compression,
                "level": self.compression_level
            }

    def open_store(
        self,
        spec: Optional[Dict] = None,
        create: bool = True,
        delete_existing: bool = False
    ) -> ts.TensorStore:
        """
        Open TensorStore with the given spec.

        Args:
            spec: TensorStore spec (uses self._spec if None)
            create: Create if doesn't exist
            delete_existing: Delete existing data

        Returns:
            ts.TensorStore: Opened TensorStore handle
        """
        if spec is None:
            spec = self._spec

        if spec is None:
            raise ValueError("No spec available. Call create_output_spec first.")

        # TensorStore open modes:
        # - delete_existing=True: Delete and recreate (first job)
        # - delete_existing=False with create=True: Create or open existing (subsequent jobs)
        # Note: TensorStore doesn't allow open=True with delete_existing=True
        if delete_existing:
            self._store = ts.open(
                spec,
                create=create,
                delete_existing=delete_existing
            ).result()
        else:
            # For LSF multi-job mode: need both create=True and open=True
            # to handle both first creation and subsequent appends
            self._store = ts.open(
                spec,
                create=create,
                open=True  # Allow opening existing store for append mode
            ).result()

        return self._store

    def write_chunk(
        self,
        chunk_domain: Any,
        data: np.ndarray,
        output_store: Optional[ts.TensorStore] = None
    ) -> None:
        """
        Write a single chunk using per-chunk transaction pattern.

        Args:
            chunk_domain: TensorStore IndexDomain or slice tuple
            data: Numpy array data to write
            output_store: TensorStore handle (uses self._store if None)
        """
        store = output_store or self._store
        if store is None:
            raise ValueError("No store available. Call open_store first.")

        # Use per-chunk transaction (Mark's fix for memory safety)
        with ts.Transaction() as txn:
            store[chunk_domain].with_transaction(txn).write(data).result()

    def write_metadata(
        self,
        ome_metadata: Optional[Dict] = None,
        voxel_sizes: Optional[Dict[str, float]] = None,
        image_name: str = "image",
        array_shape: Optional[Tuple[int, ...]] = None,
        axes_order: Optional[List[str]] = None,
        ome_xml: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Write N5 metadata to attributes.json files.

        Args:
            ome_metadata: Pre-built metadata dict
            voxel_sizes: Voxel dimensions dict {"x": nm, "y": nm, "z": nm} in nanometers
            image_name: Image name for metadata
            array_shape: Array shape
            axes_order: Axis names (e.g., ["z", "y", "x"])
            ome_xml: Ignored (N5 doesn't use OME-XML, included for API compatibility)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        if array_shape is None:
            array_shape = getattr(self, '_shape', None)
            if array_shape is None and self._store is not None:
                array_shape = tuple(self._store.shape)

        if axes_order is None:
            axes_order = getattr(self, '_axes_order', None)

        # Write root attributes.json
        root_attrs = self._build_root_attributes(
            image_name=image_name,
            voxel_sizes=voxel_sizes,
            axes_order=axes_order
        )
        root_attrs_path = os.path.join(self.output_path, "attributes.json")
        os.makedirs(self.output_path, exist_ok=True)
        with open(root_attrs_path, 'w') as f:
            json.dump(root_attrs, f, indent=2)

    def _build_root_attributes(
        self,
        image_name: str,
        voxel_sizes: Optional[Dict[str, float]],
        axes_order: Optional[List[str]]
    ) -> Dict:
        """Build N5 root attributes.json content."""
        attrs = {
            "n5": "2.5.0",
        }

        # Add pixel resolution if available
        if voxel_sizes:
            # N5 uses pixelResolution in XYZ order (reversed from ZYX)
            # Voxel sizes are in nanometers (internal standard)
            resolution = [
                voxel_sizes.get('x', 1.0),
                voxel_sizes.get('y', 1.0),
                voxel_sizes.get('z', 1.0)
            ]
            attrs["pixelResolution"] = {
                "dimensions": resolution,
                "unit": "nm"
            }

        # Add axes information if available
        if axes_order:
            attrs["axes"] = axes_order

        return attrs

    def get_chunk_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the write chunk shape from the opened store."""
        if self._store is None:
            return None
        return tuple(self._store.chunk_layout.write_chunk.shape)

    def get_store(self) -> Optional[ts.TensorStore]:
        """Get the opened TensorStore handle."""
        return self._store

    def __repr__(self) -> str:
        return (
            f"N5Writer(output_path='{self.output_path}', "
            f"compression='{self.compression}')"
        )
