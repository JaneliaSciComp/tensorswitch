"""
Zarr3 writer for TensorSwitch Phase 5 architecture.

Implements Zarr v3 format output with optional sharding support and OME-NGFF v0.5 metadata.
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
    - Automatic 5D TCZYX expansion for viewer compatibility

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
        level_path: str = "s0",
        include_omero: bool = False
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
            include_omero: Extract and include structured omero channel metadata
        """
        super().__init__(output_path)
        self.use_sharding = use_sharding
        self.compression = compression
        self.compression_level = compression_level
        self.use_ome_structure = use_ome_structure
        self.level_path = level_path
        self.include_omero = include_omero
        self._store = None
        self._spec = None
        # 5D expansion tracking
        self._original_shape = None
        self._original_axes = None
        self._expanded_axes = None

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
        Create TensorStore spec for Zarr3 output with automatic 5D TCZYX expansion.

        All data is expanded to 5D TCZYX format for OME-NGFF viewer compatibility.
        Singleton dimensions are added for missing T, C, Z axes.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Inner chunk dimensions (for sharding, these are the read chunks)
            shard_shape: Outer shard dimensions (only used if use_sharding=True)
            use_fortran_order: Use Fortran (column-major) order
            axes_order: Axis names (e.g., ["z", "y", "x"])
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for Zarr3 output (5D TCZYX)
        """
        # Store original shape and axes for domain conversion
        self._original_shape = tuple(shape)
        self._original_axes = axes_order

        # Expand to 5D TCZYX
        shape_list = list(shape)
        shape_list, expanded_axes, expanded_chunks = self._expand_to_5d(
            shape_list, axes_order, chunk_shape
        )
        self._expanded_axes = expanded_axes
        print(f"Zarr3 5D expansion: {self._original_shape} -> {shape_list}")
        print(f"  Axes: {axes_order} -> {expanded_axes}")

        # Use expanded chunks, or let zarr3_store_spec calculate if None
        custom_chunk_shape = expanded_chunks if expanded_chunks else None

        # Expand shard shape if provided
        custom_shard_shape = None
        if shard_shape:
            custom_shard_shape = self._expand_shard_shape(list(shard_shape), axes_order)

        # Create spec using existing utility function with 5D shape/axes
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
            axes_order=expanded_axes
        )

        # Add context for concurrency control
        spec['context'] = get_tensorstore_context()

        self._spec = spec
        return spec

    def _expand_to_5d(
        self,
        shape: List[int],
        axes_order: Optional[List[str]],
        chunk_shape: Optional[Tuple[int, ...]]
    ) -> Tuple[List[int], List[str], List[int]]:
        """
        Expand shape and axes to 5D TCZYX format for OME-NGFF viewer compatibility.

        Adds singleton dimensions for missing T, C, Z axes.

        Args:
            shape: Original array shape
            axes_order: Original axis names
            chunk_shape: Original chunk shape (optional)

        Returns:
            (expanded_shape, expanded_axes, expanded_chunks)
        """
        # Target: always 5D TCZYX
        TARGET_AXES = ['t', 'c', 'z', 'y', 'x']

        # Infer axes if not provided
        if axes_order is None or len(axes_order) != len(shape):
            axes_order = self._infer_axes(shape)

        # Normalize to lowercase
        axes_lower = [a.lower() for a in axes_order]

        # Build expanded shape and chunks
        expanded_shape = []
        expanded_chunks = []

        for target_axis in TARGET_AXES:
            if target_axis in axes_lower:
                # Axis exists - use its shape/chunk
                idx = axes_lower.index(target_axis)
                expanded_shape.append(shape[idx])
                if chunk_shape and idx < len(chunk_shape):
                    expanded_chunks.append(chunk_shape[idx])
                elif target_axis in ['t', 'c']:
                    expanded_chunks.append(1)  # Non-spatial: chunk=1
                elif target_axis == 'z':
                    expanded_chunks.append(min(shape[idx], 1024))
                else:  # y, x
                    expanded_chunks.append(min(shape[idx], 1024))
            else:
                # Axis doesn't exist - add singleton dimension
                expanded_shape.append(1)
                expanded_chunks.append(1)

        return expanded_shape, TARGET_AXES, expanded_chunks

    def _expand_shard_shape(
        self,
        shard_shape: List[int],
        axes_order: Optional[List[str]]
    ) -> List[int]:
        """Expand shard shape to 5D, adding 1s for missing axes."""
        TARGET_AXES = ['t', 'c', 'z', 'y', 'x']

        if axes_order is None or len(axes_order) != len(shard_shape):
            axes_order = self._infer_axes(shard_shape)

        axes_lower = [a.lower() for a in axes_order]
        expanded_shard = []

        for target_axis in TARGET_AXES:
            if target_axis in axes_lower:
                idx = axes_lower.index(target_axis)
                expanded_shard.append(shard_shape[idx])
            else:
                expanded_shard.append(1)

        return expanded_shard

    def _infer_axes(self, shape: List[int]) -> List[str]:
        """Infer axis names from shape."""
        if len(shape) == 2:
            return ['y', 'x']
        elif len(shape) == 3:
            if shape[0] <= 10:
                return ['c', 'y', 'x']
            else:
                return ['z', 'y', 'x']
        elif len(shape) == 4:
            if shape[0] <= 10:
                return ['c', 'z', 'y', 'x']
            else:
                return ['t', 'z', 'y', 'x']
        elif len(shape) == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(len(shape))]

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

        Automatically expands data to 5D for writing.
        The chunk_domain should be for the ORIGINAL shape (from input reading).

        Args:
            chunk_domain: TensorStore IndexDomain or slice tuple (in input coordinates)
            data: Input array data to write (native shape, e.g., 3D)
            output_store: TensorStore handle (uses self._store if None)
        """
        store = output_store or self._store
        if store is None:
            raise ValueError("No store available. Call open_store first.")

        # Handle 5D expansion: expand domain and data to 5D
        if self._original_shape is not None:
            output_domain, expanded_data = self._expand_chunk_to_5d(chunk_domain, data)
        else:
            output_domain = chunk_domain
            expanded_data = data

        # Use per-chunk transaction (Mark's fix for memory safety)
        with ts.Transaction() as txn:
            store[output_domain].with_transaction(txn).write(expanded_data).result()

    def get_input_domain_from_output(self, output_domain: Any) -> Any:
        """
        Convert a 5D output domain back to input coordinates for reading.

        When 5D expansion is enabled, chunk domains are generated based on the 5D
        output shape. This method converts them back to input coordinates for
        reading from the native (e.g., 3D) input array.

        Args:
            output_domain: TensorStore IndexDomain or slice tuple in 5D

        Returns:
            Domain in input coordinates (e.g., 3D for CYX data)
        """
        if self._original_shape is None:
            return output_domain

        original_axes = self._original_axes or self._infer_axes(list(self._original_shape))
        original_axes_lower = [a.lower() for a in original_axes]
        target_axes = ['t', 'c', 'z', 'y', 'x']

        # Extract slices for original axes from the 5D domain
        input_slices = []
        for orig_axis in original_axes_lower:
            target_idx = target_axes.index(orig_axis)
            if hasattr(output_domain, 'origin'):
                # TensorStore IndexDomain
                start = int(output_domain.origin[target_idx])
                stop = start + int(output_domain.shape[target_idx])
                input_slices.append(slice(start, stop))
            else:
                # Already a tuple of slices
                input_slices.append(output_domain[target_idx])

        return tuple(input_slices)

    def _expand_chunk_to_5d(self, chunk_domain: Any, data: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        Expand chunk domain and data from native shape to 5D TCZYX.

        Args:
            chunk_domain: Domain for input data (in input coordinates, e.g., 3D)
            data: Native shape numpy array

        Returns:
            (expanded_domain, expanded_data) both in 5D
        """
        # Get the mapping from original axes to 5D axes
        original_axes = self._original_axes or self._infer_axes(list(self._original_shape))
        original_axes_lower = [a.lower() for a in original_axes]
        target_axes = ['t', 'c', 'z', 'y', 'x']

        # Build the expanded domain slices
        expanded_slices = []
        expand_dims_positions = []  # Track where to add new dimensions to data

        for i, target_axis in enumerate(target_axes):
            if target_axis in original_axes_lower:
                # This axis exists in original - use its slice from chunk_domain
                orig_idx = original_axes_lower.index(target_axis)
                if hasattr(chunk_domain, 'origin'):
                    # TensorStore IndexDomain
                    start = int(chunk_domain.origin[orig_idx])
                    stop = start + int(chunk_domain.shape[orig_idx])
                    expanded_slices.append(slice(start, stop))
                else:
                    # Already a tuple of slices
                    expanded_slices.append(chunk_domain[orig_idx])
            else:
                # This axis doesn't exist in original - singleton dimension
                expanded_slices.append(slice(0, 1))
                expand_dims_positions.append(i)

        # Reshape data to add singleton dimensions
        expanded_data = data
        for pos in expand_dims_positions:
            expanded_data = np.expand_dims(expanded_data, axis=pos)

        return tuple(expanded_slices), expanded_data

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
        Write OME-NGFF v0.5 metadata to zarr.json.

        Always uses 5D TCZYX axes to match the expanded output format.

        Args:
            ome_metadata: Pre-built OME metadata dict (ignored - always rebuilt for 5D)
            voxel_sizes: Voxel dimensions dict {"x": um, "y": um, "z": um}
            image_name: Image name for metadata
            array_shape: Array shape (ignored - uses 5D expanded shape)
            axes_order: Axis names (ignored - always uses 5D TCZYX)
            ome_xml: Raw OME-XML string from source data (stored in attributes.ome_xml)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # ALWAYS use 5D expanded shape and axes for metadata
        # This ensures metadata matches the actual 5D output format
        if self._store is not None:
            array_shape = tuple(self._store.shape)
        elif self._original_shape is not None:
            # Calculate 5D shape from original
            shape_list, _, _ = self._expand_to_5d(
                list(self._original_shape), self._original_axes, None
            )
            array_shape = tuple(shape_list)
        else:
            raise ValueError("array_shape required when store not available")

        # Always use 5D axes
        axes_order = self._expanded_axes or ['t', 'c', 'z', 'y', 'x']
        print(f"Zarr3 metadata: using 5D expanded axes: {axes_order}")

        # Always rebuild metadata for 5D format
        full_metadata = create_zarr3_ome_metadata(
            ome_xml=ome_xml,
            array_shape=array_shape,
            image_name=image_name,
            pixel_sizes=voxel_sizes,
            axes_order=axes_order,
            include_omero=self.include_omero
        )

        # Write to zarr.json at root
        write_zarr3_group_metadata(self.output_path, full_metadata)

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
