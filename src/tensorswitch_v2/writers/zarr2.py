"""
Zarr2 writer for TensorSwitch Phase 5 architecture.

Implements Zarr v2 format output for legacy compatibility with OME-NGFF v0.4 metadata.
"""

import os
import json
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import tensorstore as ts

from .base import BaseWriter

# Import utility functions from existing tensorswitch
from tensorswitch.utils import (
    zarr2_store_spec,
    get_tensorstore_context,
)


class Zarr2Writer(BaseWriter):
    """
    Zarr2 writer for legacy compatibility with OME-NGFF v0.4 metadata.

    Used for backward compatibility with older tools that don't support Zarr3:
    - Neuroglancer (some versions)
    - napari (older versions)
    - Legacy pipelines

    Note: Zarr2 does NOT support sharding. For large datasets, use Zarr3Writer.

    Example:
        >>> from tensorswitch_v2.writers import Zarr2Writer
        >>> writer = Zarr2Writer("/output.zarr")
        >>> spec = writer.create_output_spec(
        ...     shape=(100, 1024, 1024),
        ...     dtype="uint8",
        ...     chunk_shape=(64, 256, 256)
        ... )
        >>> writer.open_store(spec)
        >>> # ... write chunks ...
        >>> writer.write_metadata(ome_metadata, voxel_sizes)
    """

    def __init__(
        self,
        output_path: str,
        compression: str = "zstd",
        compression_level: int = 5,
        level_path: str = "s0"
    ):
        """
        Initialize Zarr2 writer.

        Args:
            output_path: Path to output Zarr2 dataset
            compression: Compression codec ("zstd", "blosc", "gzip")
            compression_level: Compression level (1-9, default 5)
            level_path: Level subdirectory name (default "s0")
        """
        super().__init__(output_path)
        self.compression = compression
        self.compression_level = compression_level
        self.level_path = level_path
        self._store = None
        self._spec = None

    def supports_sharding(self) -> bool:
        """Zarr2 does NOT support sharding."""
        return False

    def create_output_spec(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        use_fortran_order: bool = False,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for Zarr2 output.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Chunk dimensions
            shard_shape: Ignored (Zarr2 doesn't support sharding)
            use_fortran_order: Use Fortran (column-major) order
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for Zarr2 output
        """
        if shard_shape is not None:
            print("Warning: shard_shape ignored - Zarr2 doesn't support sharding")

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

        # Compute array path (with level subdirectory)
        array_path = os.path.join(self.output_path, self.level_path)

        # Create spec using existing utility function
        spec = zarr2_store_spec(
            zarr_level_path=array_path,
            shape=shape_list,
            chunks=chunks_list,
            use_fortran_order=use_fortran_order
        )

        # Update dtype (zarr2_store_spec hardcodes "|u1")
        spec['metadata']['dtype'] = self._normalize_dtype(dtype)

        # Update compressor
        spec['metadata']['compressor'] = {
            'id': self.compression,
            'level': self.compression_level
        }

        # Add context for concurrency control
        spec['context'] = get_tensorstore_context()

        self._spec = spec
        return spec

    def _normalize_dtype(self, dtype: str) -> str:
        """Convert dtype to Zarr2 format."""
        dtype_map = {
            'uint8': '|u1',
            'uint16': '<u2',
            'uint32': '<u4',
            'uint64': '<u8',
            'int8': '|i1',
            'int16': '<i2',
            'int32': '<i4',
            'int64': '<i8',
            'float32': '<f4',
            'float64': '<f8',
        }
        return dtype_map.get(dtype, dtype)

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
        axes_order: Optional[List[str]] = None
    ) -> None:
        """
        Write OME-NGFF v0.4 metadata to .zattrs files.

        Args:
            ome_metadata: Pre-built OME metadata dict
            voxel_sizes: Voxel dimensions dict {"x": um, "y": um, "z": um}
            image_name: Image name for metadata
            array_shape: Array shape (required if ome_metadata is None)
            axes_order: Axis names (e.g., ["z", "y", "x"])
        """
        if array_shape is None and self._store is not None:
            array_shape = tuple(self._store.shape)

        # Build default OME-NGFF v0.4 metadata if not provided
        if ome_metadata is None:
            ome_metadata = self._build_zarr2_ome_metadata(
                array_shape=array_shape,
                voxel_sizes=voxel_sizes,
                image_name=image_name,
                axes_order=axes_order
            )

        # Write root .zgroup
        zgroup_path = os.path.join(self.output_path, ".zgroup")
        with open(zgroup_path, 'w') as f:
            json.dump({"zarr_format": 2}, f, indent=2)

        # Write root .zattrs with OME metadata
        zattrs_path = os.path.join(self.output_path, ".zattrs")
        with open(zattrs_path, 'w') as f:
            json.dump(ome_metadata, f, indent=2)

    def _build_zarr2_ome_metadata(
        self,
        array_shape: Tuple[int, ...],
        voxel_sizes: Optional[Dict[str, float]],
        image_name: str,
        axes_order: Optional[List[str]]
    ) -> Dict:
        """Build OME-NGFF v0.4 metadata structure."""
        # Determine axes
        if axes_order and len(axes_order) == len(array_shape):
            axes = axes_order
        elif len(array_shape) == 3:
            axes = ["z", "y", "x"] if array_shape[0] > 10 else ["c", "y", "x"]
        elif len(array_shape) == 4:
            axes = ["c", "z", "y", "x"]
        elif len(array_shape) == 5:
            axes = ["t", "c", "z", "y", "x"]
        else:
            axes = [f"dim_{i}" for i in range(len(array_shape))]

        # Build scale factors from voxel sizes
        scale_factors = []
        for axis in axes:
            if voxel_sizes and axis in voxel_sizes:
                scale_factors.append(voxel_sizes[axis])
            else:
                scale_factors.append(1.0)

        return {
            "multiscales": [{
                "version": "0.4",
                "name": image_name,
                "axes": [{"name": a, "type": "space" if a in "xyz" else "channel" if a == "c" else "time"} for a in axes],
                "datasets": [{
                    "path": self.level_path,
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": scale_factors
                    }]
                }]
            }]
        }

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
            f"Zarr2Writer(output_path='{self.output_path}', "
            f"compression='{self.compression}')"
        )
