"""
Zarr3 writer for TensorSwitch Phase 5 architecture.

Implements Zarr v3 format output with optional sharding support and OME-NGFF v0.5 metadata.
"""

import os
import json
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import tensorstore as ts

from .base import BaseWriter

# Import utility functions from existing tensorswitch
from tensorswitch.utils import (
    zarr3_store_spec,
    get_tensorstore_context,
    write_zarr3_group_metadata,
    create_zarr3_ome_metadata,
)


class Zarr3Writer(BaseWriter):
    """
    Zarr3 writer with sharding support and OME-NGFF v0.5 metadata.

    Primary output format for TensorSwitch with modern features:
    - Sharding support (reduces small file count)
    - Flexible codecs (blosc, gzip, zstd)
    - OME-NGFF v0.5 metadata compliance
    - Remote storage support (GCS, S3)

    Example:
        >>> from tensorswitch_v2.writers import Zarr3Writer
        >>> writer = Zarr3Writer("/output.zarr", use_sharding=True)
        >>> spec = writer.create_output_spec(
        ...     shape=(100, 1024, 1024),
        ...     dtype="uint16",
        ...     chunk_shape=(32, 256, 256)
        ... )
        >>> writer.open_store(spec)
        >>> # ... write chunks ...
        >>> writer.write_metadata(ome_metadata, voxel_sizes)
    """

    def __init__(
        self,
        output_path: str,
        use_sharding: bool = True,
        compression: str = "zstd",
        compression_level: int = 5,
        use_ome_structure: bool = True,
        level_path: str = "s0"
    ):
        """
        Initialize Zarr3 writer.

        Args:
            output_path: Path to output Zarr3 dataset
            use_sharding: Enable sharding (recommended for large datasets)
            compression: Compression codec ("zstd", "blosc", "gzip")
            compression_level: Compression level (1-9, default 5)
            use_ome_structure: Use OME-ZARR directory structure (s0/, s1/, etc.)
            level_path: Level subdirectory name (default "s0")
        """
        super().__init__(output_path)
        self.use_sharding = use_sharding
        self.compression = compression
        self.compression_level = compression_level
        self.use_ome_structure = use_ome_structure
        self.level_path = level_path
        self._store = None
        self._spec = None

    def supports_sharding(self) -> bool:
        """Zarr3 supports sharding via sharding_indexed codec."""
        return True

    def create_output_spec(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        use_fortran_order: bool = False,
        axes_order: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for Zarr3 output.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Inner chunk dimensions (for sharding, these are the read chunks)
            shard_shape: Outer shard dimensions (only used if use_sharding=True)
            use_fortran_order: Use Fortran (column-major) order
            axes_order: Axis names (e.g., ["z", "y", "x"])
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for Zarr3 output
        """
        # Convert to list for zarr3_store_spec
        shape_list = list(shape)

        # Use provided shapes or defaults
        custom_chunk_shape = list(chunk_shape) if chunk_shape else None
        custom_shard_shape = list(shard_shape) if shard_shape else None

        # Create spec using existing utility function
        spec = zarr3_store_spec(
            path=self.output_path,
            shape=shape_list,
            dtype=dtype,
            use_shard=self.use_sharding,
            level_path=self.level_path,
            use_ome_structure=self.use_ome_structure,
            custom_shard_shape=custom_shard_shape,
            custom_chunk_shape=custom_chunk_shape,
            use_v2_encoding=False,
            use_fortran_order=use_fortran_order,
            axes_order=axes_order
        )

        # Add context for concurrency control
        spec['context'] = get_tensorstore_context()

        self._spec = spec
        return spec

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

        # Note: TensorStore doesn't allow open=True with delete_existing=True
        # When creating new store, use create=True only; for opening existing use open=True
        if delete_existing:
            self._store = ts.open(
                spec,
                create=create,
                delete_existing=delete_existing
            ).result()
        else:
            self._store = ts.open(
                spec,
                create=create,
                open=not create  # Only open if not creating new
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
        axes_order: Optional[List[str]] = None
    ) -> None:
        """
        Write OME-NGFF v0.5 metadata to zarr.json.

        Args:
            ome_metadata: Pre-built OME metadata dict (if None, creates from voxel_sizes)
            voxel_sizes: Voxel dimensions dict {"x": um, "y": um, "z": um}
            image_name: Image name for metadata
            array_shape: Array shape (required if ome_metadata is None)
            axes_order: Axis names (e.g., ["z", "y", "x"])
        """
        if ome_metadata is None:
            # Build metadata from voxel sizes
            if array_shape is None:
                if self._store is not None:
                    array_shape = tuple(self._store.shape)
                else:
                    raise ValueError("array_shape required when ome_metadata not provided")

            ome_metadata = create_zarr3_ome_metadata(
                ome_xml=None,
                array_shape=array_shape,
                image_name=image_name,
                pixel_sizes=voxel_sizes,
                axes_order=axes_order
            )

        # Write to zarr.json at root
        write_zarr3_group_metadata(self.output_path, ome_metadata)

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
            f"Zarr3Writer(output_path='{self.output_path}', "
            f"use_sharding={self.use_sharding}, "
            f"compression='{self.compression}')"
        )
