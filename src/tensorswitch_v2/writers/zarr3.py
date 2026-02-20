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
        include_omero: bool = False,
        use_nested_structure: bool = True,
        data_type: str = "image",
        image_key: str = "raw",
        label_key: str = "segmentation"
    ):
        """
        Initialize Zarr3 writer.

        Args:
            output_path: Path to output Zarr3 dataset
            use_sharding: Enable sharding (recommended for large datasets)
            compression: Compression codec ("zstd", "blosc", "gzip")
            compression_level: Compression level (1-9, default 5)
            use_ome_structure: Use OME-ZARR directory structure (s0/, s1/, etc.)
            level_path: Level subdirectory name (default "s0" for Janelia convention)
            include_omero: Extract and include structured omero channel metadata
            use_nested_structure: Use OME-NGFF nested structure (raw/, labels/)
            data_type: 'image' or 'labels' - determines output subdirectory
            image_key: Name for image group (default: "raw")
            label_key: Name for label image (default: "segmentation")
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
        # Axis reordering (for precomputed XYZC -> CXYZ)
        self._transpose_order = None
        # OME-NGFF nested structure
        self.use_nested_structure = use_nested_structure
        self.data_type = data_type
        self.image_key = image_key
        self.label_key = label_key
        self._ome_structure = None
        if use_nested_structure:
            from ..utils.ome_structure import OMEStructure, OMEStructureConfig
            config = OMEStructureConfig(
                image_key=image_key,
                label_name=label_key
            )
            self._ome_structure = OMEStructure(output_path, config)

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
        expand_to_5d: bool = False,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for Zarr3 output.

        By default (expand_to_5d=False), preserves source dimensionality and axis order
        per OME-NGFF RFC-3. Use expand_to_5d=True for legacy 5D TCZYX expansion.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Inner chunk dimensions (for sharding, these are the read chunks)
            shard_shape: Outer shard dimensions (only used if use_sharding=True)
            use_fortran_order: Use Fortran (column-major) order
            axes_order: Axis names (e.g., ["z", "y", "x"])
            expand_to_5d: If True, expand to 5D TCZYX (legacy behavior)
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for Zarr3 output
        """
        # Store original shape and axes for domain conversion
        self._original_shape = tuple(shape)
        self._original_axes = axes_order
        self._expand_to_5d = expand_to_5d

        if expand_to_5d:
            # Legacy behavior: Expand to 5D TCZYX
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
        else:
            # NEW DEFAULT: Preserve source layout (per OME-NGFF RFC-3)
            # But reorder axes so non-spatial (t, c) come before spatial (x, y, z)
            shape_list = list(shape)
            source_axes = axes_order if axes_order else self._infer_axes(shape_list)

            # Reorder axes for OME-NGFF compliance (non-spatial before spatial)
            reordered_axes, reordered_shape, reordered_chunk, reordered_shard, transpose_order = \
                self._reorder_axes_for_ome(source_axes, shape_list, chunk_shape, shard_shape)

            self._expanded_axes = reordered_axes
            self._transpose_order = transpose_order

            if transpose_order:
                print(f"Zarr3 reordering axes: {source_axes} -> {reordered_axes}")
                print(f"  Shape: {shape_list} -> {reordered_shape}")
            else:
                print(f"Zarr3 preserving layout: shape={shape_list}, axes={source_axes}")

            shape_list = reordered_shape
            expanded_axes = reordered_axes

            # Use reordered chunks/shards
            custom_chunk_shape = reordered_chunk
            custom_shard_shape = reordered_shard

        # Determine the actual data path based on nested structure settings
        if self.use_nested_structure and self._ome_structure:
            if self.data_type == 'labels':
                data_path = self._ome_structure.get_label_data_path()
            else:
                data_path = self._ome_structure.get_image_data_path()
            print(f"Zarr3 nested structure: data_type={self.data_type}, path={data_path}")
        else:
            data_path = self.output_path

        # Create spec using existing utility function with 5D shape/axes
        # Build compression codec from writer settings
        compression_codec = {
            'name': self.compression,
            'configuration': {'level': self.compression_level}
        }

        spec = zarr3_store_spec(
            path=data_path,
            shape=shape_list,
            dtype=dtype,
            use_shard=self.use_sharding,
            level_path=self.level_path,
            use_ome_structure=self.use_ome_structure,
            custom_shard_shape=custom_shard_shape,
            custom_chunk_shape=custom_chunk_shape,
            use_v2_encoding=False,
            use_fortran_order=use_fortran_order,
            axes_order=expanded_axes,
            compression=compression_codec
        )

        # Add context for concurrency control
        spec['context'] = get_tensorstore_context()

        self._spec = spec
        self._data_path = data_path  # Store for metadata writing
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

    def _reorder_axes_for_ome(
        self,
        axes: List[str],
        shape: List[int],
        chunk_shape: Optional[Tuple[int, ...]],
        shard_shape: Optional[Tuple[int, ...]]
    ) -> Tuple[List[str], List[int], Optional[List[int]], Optional[List[int]], Optional[Tuple[int, ...]]]:
        """
        Reorder axes so non-spatial dimensions come before spatial dimensions.

        For precomputed format which stores as XYZC, this converts to CXYZ.
        Preserves the relative order within spatial and non-spatial groups.

        Args:
            axes: Original axis names (e.g., ['x', 'y', 'z', 'channel'])
            shape: Original shape
            chunk_shape: Original chunk shape
            shard_shape: Original shard shape

        Returns:
            (reordered_axes, reordered_shape, reordered_chunk, reordered_shard, transpose_order)
        """
        # Identify spatial and non-spatial axes
        SPATIAL_AXES = {'x', 'y', 'z'}

        axes_lower = [a.lower() for a in axes]

        # Separate into non-spatial (first) and spatial (second), preserving order
        non_spatial_indices = []
        spatial_indices = []

        for i, axis in enumerate(axes_lower):
            if axis in SPATIAL_AXES:
                spatial_indices.append(i)
            else:
                non_spatial_indices.append(i)

        # New order: non-spatial first, then spatial
        new_order = non_spatial_indices + spatial_indices

        # Check if reordering is needed
        if new_order == list(range(len(axes))):
            # Already in correct order - but still need to pad chunks/shards if needed
            padded_chunk = None
            if chunk_shape:
                padded_chunk = list(chunk_shape)
                while len(padded_chunk) < len(axes):
                    padded_chunk.append(1)
            padded_shard = None
            if shard_shape:
                padded_shard = list(shard_shape)
                while len(padded_shard) < len(axes):
                    padded_shard.append(1)
            return axes, shape, padded_chunk, padded_shard, None

        # Reorder axes
        reordered_axes = [axes[i] for i in new_order]
        reordered_shape = [shape[i] for i in new_order]

        # Reorder chunk shape if provided (pad with 1s if needed for non-spatial dims)
        reordered_chunk = None
        if chunk_shape:
            # Pad chunk_shape if it's shorter than axes (e.g., 3D chunks for 4D data)
            padded_chunk = list(chunk_shape)
            while len(padded_chunk) < len(axes):
                padded_chunk.append(1)  # Non-spatial dims get chunk=1
            reordered_chunk = [padded_chunk[i] for i in new_order]

        # Reorder shard shape if provided (pad with 1s if needed)
        reordered_shard = None
        if shard_shape:
            padded_shard = list(shard_shape)
            while len(padded_shard) < len(axes):
                padded_shard.append(1)  # Non-spatial dims get shard=1
            reordered_shard = [padded_shard[i] for i in new_order]

        # Store transpose order for data transformation
        transpose_order = tuple(new_order)

        return reordered_axes, reordered_shape, reordered_chunk, reordered_shard, transpose_order

    def _reorder_domain(self, domain: Any, transpose_order: Tuple[int, ...]) -> Any:
        """
        Reorder a chunk domain according to the transpose order.

        Args:
            domain: TensorStore IndexDomain or tuple of slices
            transpose_order: New axis order (e.g., (3, 0, 1, 2) for XYZC -> CXYZ)

        Returns:
            Reordered domain
        """
        if hasattr(domain, 'origin'):
            # TensorStore IndexDomain - extract slices, reorder, return tuple
            slices = []
            for i in range(len(domain.origin)):
                start = int(domain.origin[i])
                stop = start + int(domain.shape[i])
                slices.append(slice(start, stop))
            return tuple(slices[i] for i in transpose_order)
        elif isinstance(domain, tuple):
            # Already a tuple of slices
            return tuple(domain[i] for i in transpose_order)
        else:
            # Unknown format, return as-is
            return domain

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

        # Handle 5D expansion only if enabled (legacy behavior)
        if getattr(self, '_expand_to_5d', False) and self._original_shape is not None:
            output_domain, expanded_data = self._expand_chunk_to_5d(chunk_domain, data)
        elif self._transpose_order is not None:
            # Reorder axes (e.g., XYZC -> CXYZ for precomputed)
            # IMPORTANT: np.transpose creates a view with different strides but same memory
            # TensorStore needs contiguous memory, so we must copy to rearrange the data
            expanded_data = np.ascontiguousarray(np.transpose(data, self._transpose_order))
            # Reorder the domain to match
            output_domain = self._reorder_domain(chunk_domain, self._transpose_order)
        else:
            # Write directly without modification
            output_domain = chunk_domain
            expanded_data = data

        # Use per-chunk transaction (Mark's fix for memory safety)
        with ts.Transaction() as txn:
            store[output_domain].with_transaction(txn).write(expanded_data).result()

    def get_input_domain_from_output(self, output_domain: Any) -> Any:
        """
        Convert output domain back to input coordinates for reading.

        When expand_to_5d=True, chunk domains are generated based on the 5D
        output shape. This method converts them back to input coordinates for
        reading from the native (e.g., 3D) input array.

        When axis reordering is active (e.g., XYZC -> CXYZ), output domain
        needs to be reordered back to input order for reading.

        When expand_to_5d=False and no reordering, domains are returned as-is.

        Args:
            output_domain: TensorStore IndexDomain or slice tuple

        Returns:
            Domain in input coordinates
        """
        # Handle axis reordering case (e.g., XYZC -> CXYZ)
        if self._transpose_order is not None:
            # Compute inverse transpose to go from output -> input
            inverse_order = [0] * len(self._transpose_order)
            for i, j in enumerate(self._transpose_order):
                inverse_order[j] = i
            return self._reorder_domain(output_domain, tuple(inverse_order))

        # If not expanding to 5D, input and output have same shape
        if not getattr(self, '_expand_to_5d', False):
            return output_domain

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
            chunk_domain: Domain for input data (in input coordinates, e.g., 3D or 4D XYZC)
            data: Native shape numpy array

        Returns:
            (expanded_domain, expanded_data) both in 5D TCZYX order
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

        # Step 1: Add singleton dimensions for missing axes
        expanded_data = data
        for pos in expand_dims_positions:
            expanded_data = np.expand_dims(expanded_data, axis=pos)

        # Step 2: Transpose to TCZYX order if needed
        # After expand_dims, the axis order is: [new_axes_at_their_positions] + [original_axes_shifted]
        # We need to compute the current axis order and transpose to target order
        current_axes = []
        orig_idx = 0
        for i in range(5):
            if i in expand_dims_positions:
                current_axes.append(target_axes[i])  # This is a newly added axis
            else:
                current_axes.append(original_axes_lower[orig_idx])
                orig_idx += 1

        # Only transpose if current order differs from target
        if current_axes != target_axes:
            transpose_order = [current_axes.index(t) for t in target_axes]
            expanded_data = np.transpose(expanded_data, transpose_order)

        return tuple(expanded_slices), expanded_data

    def write_metadata(
        self,
        ome_metadata: Optional[Dict] = None,
        voxel_sizes: Optional[Dict[str, float]] = None,
        image_name: str = "image",
        array_shape: Optional[Tuple[int, ...]] = None,
        axes_order: Optional[List[str]] = None,
        ome_xml: Optional[str] = None,
        is_label: bool = False,
        **kwargs
    ) -> None:
        """
        Write OME-NGFF v0.5 metadata to zarr.json.

        Always uses 5D TCZYX axes to match the expanded output format.

        Args:
            ome_metadata: Pre-built OME metadata dict (ignored - always rebuilt for 5D)
            voxel_sizes: Voxel dimensions dict {"x": nm, "y": nm, "z": nm} in nanometers
            image_name: Image name for metadata
            array_shape: Array shape (ignored - uses 5D expanded shape)
            axes_order: Axis names (ignored - always uses 5D TCZYX)
            ome_xml: Raw OME-XML string from source data (stored in attributes.ome_xml)
            is_label: If True, add OME-NGFF image-label metadata for segmentation data
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
        print(f"Zarr3 metadata: using axes: {axes_order}")

        # Handle nested structure metadata
        if self.use_nested_structure and self._ome_structure:
            self._write_nested_metadata(
                array_shape=array_shape,
                axes_order=axes_order,
                voxel_sizes=voxel_sizes,
                image_name=image_name,
                ome_xml=ome_xml,
                is_label=is_label
            )
        else:
            # Legacy: write single metadata file at root
            full_metadata = create_zarr3_ome_metadata(
                ome_xml=ome_xml,
                array_shape=array_shape,
                image_name=image_name,
                pixel_sizes=voxel_sizes,
                axes_order=axes_order,
                include_omero=self.include_omero,
                is_label=is_label
            )
            write_zarr3_group_metadata(self.output_path, full_metadata)

    def _write_nested_metadata(
        self,
        array_shape: Tuple[int, ...],
        axes_order: List[str],
        voxel_sizes: Optional[Dict[str, float]],
        image_name: str,
        ome_xml: Optional[str],
        is_label: bool
    ) -> None:
        """
        Write metadata for OME-NGFF nested structure.

        Creates metadata at all required levels:
        - Data group (raw/zarr.json or labels/segmentation/zarr.json)
        - Labels container (labels/zarr.json) if writing labels
        - Root (zarr.json)
        """
        from ..utils.metadata_utils import generate_default_label_colors

        # Build axes list for metadata
        axes = []
        for axis_name in axes_order:
            axis_lower = axis_name.lower()
            if axis_lower in ['x', 'y', 'z']:
                axes.append({'name': axis_lower, 'type': 'space', 'unit': 'nanometer'})
            elif axis_lower in ['c', 'channel']:
                axes.append({'name': 'c', 'type': 'channel'})
            elif axis_lower in ['t', 'v']:
                axes.append({'name': 't', 'type': 'time', 'unit': 'millisecond'})
            else:
                axes.append({'name': axis_lower, 'type': 'space'})

        # Build scale factors from voxel sizes
        scale_factors = []
        for axis_name in axes_order:
            axis_lower = axis_name.lower()
            if voxel_sizes and axis_lower in voxel_sizes:
                scale_factors.append(voxel_sizes[axis_lower])
            elif voxel_sizes and axis_lower in ['c', 'channel', 't', 'v']:
                scale_factors.append(1.0)
            else:
                scale_factors.append(1.0)

        # Build datasets list (just s0 for now, downsampling adds more)
        datasets = [{
            'path': self.level_path,
            'coordinateTransformations': [{'type': 'scale', 'scale': scale_factors}]
        }]

        multiscales = {'axes': axes, 'datasets': datasets}

        if self.data_type == 'labels' or is_label:
            # Writing labels
            label_colors = generate_default_label_colors(256)

            # Write label image metadata (includes image-label)
            self._ome_structure.write_label_image_metadata(
                multiscales=multiscales,
                name=image_name,
                colors=label_colors,
                source_image_path=f"../../{self.image_key}"  # Reference to image
            )

            # Write labels container metadata
            self._ome_structure.write_labels_container_metadata()

            # Write root metadata (labels only, no image multiscales)
            self._ome_structure.write_root_metadata(
                image_multiscales=None,
                has_labels=True,
                image_name=image_name
            )

            print(f"Wrote nested labels metadata to {self.output_path}")
        else:
            # Writing image
            self._ome_structure.write_image_metadata(
                multiscales=multiscales,
                name=image_name
            )

            # Write root metadata (image only)
            self._ome_structure.write_root_metadata(
                image_multiscales=multiscales,
                has_labels=False,
                image_name=image_name
            )

            print(f"Wrote nested image metadata to {self.output_path}")

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
