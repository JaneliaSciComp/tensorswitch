"""
Zarr2 writer for TensorSwitch Phase 5 architecture.

Implements Zarr v2 format output for legacy compatibility with OME-NGFF v0.4 metadata.
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
    zarr2_store_spec,
    get_tensorstore_context,
    extract_omero_channels,
    generate_default_label_colors,
)


class Zarr2Writer(BaseWriter):
    """
    Zarr2 writer for legacy compatibility with OME-NGFF v0.4 metadata.

    Used for backward compatibility with older tools that don't support Zarr3:
    - Neuroglancer (some versions)
    - napari (older versions)
    - Legacy pipelines

    Note: Zarr2 does NOT support sharding. For large datasets, use Zarr3Writer.

    All data is automatically expanded to 5D TCZYX format for OME-NGFF viewer
    compatibility. Singleton dimensions are added for missing T, C, Z axes.

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
        level_path: str = "s0",
        include_omero: bool = False,
        use_nested_structure: bool = False,
        data_type: str = 'image',
        image_key: str = 'raw',
        label_key: str = 'segmentation'
    ):
        """
        Initialize Zarr2 writer.

        Args:
            output_path: Path to output Zarr2 dataset
            compression: Compression codec ("zstd", "blosc", "gzip")
            compression_level: Compression level (1-9, default 5)
            level_path: Level subdirectory name (default "s0" for Janelia convention)
            include_omero: Extract and include structured omero channel metadata
            use_nested_structure: Use OME-NGFF nested directory structure (raw/, labels/)
            data_type: Type of data being written ('image' or 'labels')
            image_key: Name for image group in nested structure
            label_key: Name for label image in nested structure
        """
        super().__init__(output_path)
        self.compression = compression
        self.compression_level = compression_level
        self.level_path = level_path
        self.include_omero = include_omero
        self._store = None
        self._spec = None
        self._axes_order = None  # Original axes from source
        self._original_shape = None  # Store original shape before expansion
        self._expanded_axes = None  # Store expanded 5D axes
        # Axis reordering (for precomputed XYZC -> CXYZ)
        self._transpose_order = None
        # OME-NGFF nested structure support
        self.use_nested_structure = use_nested_structure
        self.data_type = data_type
        self.image_key = image_key
        self.label_key = label_key
        self._ome_structure = None
        if use_nested_structure:
            from ..utils.ome_structure import OMEStructureZarr2, OMEStructureZarr2Config
            config = OMEStructureZarr2Config(
                image_key=image_key,
                label_name=label_key
            )
            self._ome_structure = OMEStructureZarr2(output_path, config)

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
        axes_order: Optional[List[str]] = None,
        expand_to_5d: bool = False,
        **kwargs
    ) -> Dict:
        """
        Create TensorStore spec for Zarr2 output.

        By default (expand_to_5d=False), preserves source dimensionality and axis order
        per OME-NGFF RFC-3. Use expand_to_5d=True for legacy 5D TCZYX expansion.

        Args:
            shape: Array shape (e.g., (100, 1024, 1024))
            dtype: Data type string (e.g., "uint16", "uint8")
            chunk_shape: Chunk dimensions (auto-calculated if None)
            shard_shape: Ignored (Zarr2 doesn't support sharding)
            use_fortran_order: Use Fortran (column-major) order
            axes_order: Axis names (e.g., ['c', 'y', 'x'])
            expand_to_5d: If True, expand to 5D TCZYX (legacy behavior)
            **kwargs: Additional options

        Returns:
            dict: TensorStore spec for Zarr2 output
        """
        if shard_shape is not None:
            print("Warning: shard_shape ignored - Zarr2 doesn't support sharding")

        # Store original axes_order for domain conversion during writing
        self._axes_order = axes_order
        self._original_shape = tuple(shape)
        self._expand_to_5d = expand_to_5d

        if expand_to_5d:
            # Legacy behavior: expand to 5D TCZYX
            shape_list = list(shape)
            shape_list, expanded_axes, chunks_list = self._expand_to_5d(
                shape_list, axes_order, chunk_shape
            )
            self._expanded_axes = expanded_axes
            print(f"Zarr2 5D expansion: {self._original_shape} -> {shape_list}")
            print(f"  Axes: {axes_order} -> {expanded_axes}")
        else:
            # NEW DEFAULT: Preserve source layout
            # But reorder axes so non-spatial (t, c) come before spatial (x, y, z)
            shape_list = list(shape)
            source_axes = axes_order if axes_order else self._infer_axes(shape_list)

            # Reorder axes for OME-NGFF compliance (non-spatial before spatial)
            reordered_axes, reordered_shape, reordered_chunk, _, transpose_order = \
                self._reorder_axes_for_ome(source_axes, shape_list, chunk_shape, None)

            self._expanded_axes = reordered_axes
            self._transpose_order = transpose_order

            if transpose_order:
                print(f"Zarr2 reordering axes: {source_axes} -> {reordered_axes}")
                print(f"  Shape: {shape_list} -> {reordered_shape}")
            else:
                print(f"Zarr2 preserving layout: shape={shape_list}, axes={source_axes}")

            shape_list = reordered_shape
            expanded_axes = reordered_axes
            chunks_list = reordered_chunk

        # Determine the actual data path based on nested structure settings
        if self.use_nested_structure and self._ome_structure:
            if self.data_type == 'labels':
                data_path = self._ome_structure.get_label_data_path()
            else:
                data_path = self._ome_structure.get_image_data_path()
            print(f"Zarr2 nested structure: data_type={self.data_type}, path={data_path}")
        else:
            data_path = self.output_path

        # Compute array path (with level subdirectory)
        array_path = os.path.join(data_path, self.level_path)

        # Create spec using utility function with smart chunking
        spec = zarr2_store_spec(
            zarr_level_path=array_path,
            shape=shape_list,
            chunks=chunks_list,  # None = auto-calculate based on axes
            use_fortran_order=use_fortran_order,
            axes_order=self._expanded_axes,
            dtype=self._normalize_dtype(dtype),
            compressor=self._get_compressor_config()
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

        Uses small fixed chunks for Zarr2 (no sharding support):
        - Z: max 32 (WebKnossos/Neuroglancer compatible, like v1)
        - Y/X: max 256 (reasonable file size without sharding)

        This is critical because Zarr2 has NO sharding - each chunk = 1 file.
        Large chunks cause huge file sizes and slow random access in viewers.

        Args:
            shape: Original array shape
            axes_order: Original axis names
            chunk_shape: Original chunk shape (optional)

        Returns:
            (expanded_shape, expanded_axes, expanded_chunks)
        """
        # Target: always 5D TCZYX
        TARGET_AXES = ['t', 'c', 'z', 'y', 'x']

        # Zarr2-specific chunk limits (no sharding = small chunks needed)
        # Matches v1's WebKnossos-compatible chunking strategy
        DEFAULT_ZARR2_CHUNK_Z = 32    # Z: small for random Z-slice access
        DEFAULT_ZARR2_CHUNK_YX = 256  # Y/X: reasonable tile size

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
                    # Small Z chunks for random access (Zarr2 has no sharding)
                    expanded_chunks.append(min(shape[idx], DEFAULT_ZARR2_CHUNK_Z))
                else:  # y, x
                    # Reasonable tile size without sharding
                    expanded_chunks.append(min(shape[idx], DEFAULT_ZARR2_CHUNK_YX))
            else:
                # Axis doesn't exist - add singleton dimension
                expanded_shape.append(1)
                expanded_chunks.append(1)

        return expanded_shape, TARGET_AXES, expanded_chunks

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
            shard_shape: Original shard shape (ignored for Zarr2)

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
            # Already in correct order - but still need to pad chunks if needed
            padded_chunk = None
            if chunk_shape:
                padded_chunk = list(chunk_shape)
                while len(padded_chunk) < len(axes):
                    padded_chunk.append(1)
            return axes, shape, padded_chunk, None, None

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

        # Store transpose order for data transformation
        transpose_order = tuple(new_order)

        return reordered_axes, reordered_shape, reordered_chunk, None, transpose_order

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

    def _get_compressor_config(self) -> Dict:
        """
        Build compressor configuration based on compression type.

        Different compressors have different parameter names:
        - zstd: uses 'level'
        - blosc: uses 'clevel', 'cname', 'shuffle', 'blocksize'
        - gzip: uses 'level'

        Returns:
            dict: Compressor configuration for TensorStore
        """
        if self.compression == 'blosc':
            # Blosc uses different parameter names
            # Default to lz4 codec with shuffle enabled (standard for OME-NGFF)
            return {
                'id': 'blosc',
                'cname': 'lz4',
                'clevel': self.compression_level,
                'shuffle': 1,
                'blocksize': 0
            }
        else:
            # zstd, gzip, and others use 'level'
            return {
                'id': self.compression,
                'level': self.compression_level
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

        Automatically expands data to 5D TCZYX for writing.
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

        original_axes = self._axes_order or self._infer_axes(list(self._original_shape))
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
        original_axes = self._axes_order or self._infer_axes(list(self._original_shape))
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
        Write OME-NGFF v0.4 metadata to .zattrs files.

        Always uses 5D TCZYX axes to match the expanded output format.

        Args:
            ome_metadata: Pre-built OME metadata dict (ignored - always rebuilt for 5D)
            voxel_sizes: Voxel dimensions dict {"x": nm, "y": nm, "z": nm} in nanometers
            image_name: Image name for metadata
            array_shape: Array shape (ignored - uses 5D expanded shape)
            axes_order: Axis names (ignored - always uses 5D TCZYX)
            ome_xml: Raw OME-XML string (stored in attributes if provided)
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
                list(self._original_shape), self._axes_order, None
            )
            array_shape = tuple(shape_list)
        else:
            raise ValueError("array_shape required when store not available")

        # Always use 5D axes
        axes_order = self._expanded_axes or ['t', 'c', 'z', 'y', 'x']
        print(f"Zarr2 metadata: using 5D expanded axes: {axes_order}")

        # Handle nested structure metadata (like Zarr3Writer)
        if self.use_nested_structure and self._ome_structure:
            self._write_nested_metadata(
                array_shape=array_shape,
                axes_order=axes_order,
                voxel_sizes=voxel_sizes,
                image_name=image_name,
                ome_xml=ome_xml,
                is_label=is_label
            )
            return

        # Legacy: write single metadata file at output_path
        ome_metadata = self._build_zarr2_ome_metadata(
            array_shape=array_shape,
            voxel_sizes=voxel_sizes,
            image_name=image_name,
            axes_order=axes_order
        )

        # Add image-label metadata for segmentation data
        if is_label:
            label_colors = generate_default_label_colors(256)
            ome_metadata['image-label'] = {
                'version': '0.4',
                'colors': label_colors,
                'source': {
                    'image': '../../raw'  # Standard relative path to source image
                }
            }
            print(f"Zarr2 metadata: added image-label with 256 colors")

        # Add omero channel metadata if requested
        if self.include_omero and ome_xml:
            omero_channels = extract_omero_channels(ome_xml)
            if omero_channels:
                ome_metadata['omero'] = {
                    "channels": omero_channels,
                    "rdefs": {"model": "color"}
                }

        # Add ome_xml to metadata if provided
        if ome_xml:
            ome_metadata['ome_xml'] = ome_xml

        # Write root .zgroup
        zgroup_path = os.path.join(self.output_path, ".zgroup")
        with open(zgroup_path, 'w') as f:
            json.dump({"zarr_format": 2}, f, indent=2)

        # Write root .zattrs with OME metadata
        zattrs_path = os.path.join(self.output_path, ".zattrs")
        with open(zattrs_path, 'w') as f:
            json.dump(ome_metadata, f, indent=2)

        print(f"Zarr2 metadata written to {self.output_path}")

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
        Write metadata for OME-NGFF nested structure (Zarr2 format).

        Creates metadata at all required levels:
        - Data group (raw/.zattrs or labels/segmentation/.zattrs)
        - Labels container (labels/.zattrs) if writing labels
        - Root (.zattrs)
        """
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

        # Build datasets list (just level 0 for now, downsampling adds more)
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
                source_image_path=f"../../{self.image_key}"
            )

            # Write labels container metadata
            self._ome_structure.write_labels_container_metadata()

            # Write root metadata (labels only)
            self._ome_structure.write_root_metadata(
                image_multiscales=None,
                has_labels=True,
                image_name=image_name,
                ome_xml=ome_xml
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
                image_name=image_name,
                ome_xml=ome_xml
            )

            print(f"Wrote nested image metadata to {self.output_path}")

    def _build_zarr2_ome_metadata(
        self,
        array_shape: Tuple[int, ...],
        voxel_sizes: Optional[Dict[str, float]],
        image_name: str,
        axes_order: Optional[List[str]]
    ) -> Dict:
        """Build OME-NGFF v0.4 metadata structure with proper axis detection.

        Same logic as Zarr3 for consistent dimension handling.
        """
        # Determine axes - same logic as zarr3_store_spec
        if axes_order and len(axes_order) == len(array_shape):
            axes = list(axes_order)
            print(f"Zarr2 metadata: using axes from source: {axes}")
        else:
            # Infer from shape - same logic as zarr3
            if len(array_shape) == 2:
                axes = ["y", "x"]
            elif len(array_shape) == 3:
                # For 3D, assume channels if first dimension is small, otherwise Z
                if array_shape[0] <= 10:
                    axes = ["c", "y", "x"]
                else:
                    axes = ["z", "y", "x"]
            elif len(array_shape) == 4:
                # CZYX or TZYX - check first dim
                if array_shape[0] <= 10:
                    axes = ["c", "z", "y", "x"]
                else:
                    axes = ["t", "z", "y", "x"]
            elif len(array_shape) == 5:
                axes = ["t", "c", "z", "y", "x"]
            else:
                axes = [f"dim_{i}" for i in range(len(array_shape))]
            print(f"Zarr2 metadata: inferred axes from shape {array_shape}: {axes}")

        def get_axis_type(axis_name: str) -> str:
            """Get OME-NGFF axis type from axis name."""
            axis_lower = axis_name.lower()
            if axis_lower in ['x', 'y', 'z']:
                return "space"
            elif axis_lower in ['c', 'channel']:
                return "channel"
            elif axis_lower in ['t', 'v']:
                # 'v' (view) treated as time so viewers show it as a slider
                return "time"
            else:
                return "space"  # Default to space for unknown axes

        def normalize_axis_name(axis_name: str) -> str:
            """Normalize axis name to OME-NGFF standard."""
            axis_lower = axis_name.lower()
            if axis_lower == 'channel':
                return 'c'
            elif axis_lower == 'v':
                return 't'
            return axis_lower

        def get_axis_unit(axis_name: str) -> Optional[str]:
            """Get unit for axis if applicable."""
            axis_lower = axis_name.lower()
            if axis_lower in ['x', 'y', 'z']:
                return "nanometer"
            elif axis_lower == 't':
                return "second"
            return None

        # Build axes metadata with proper types and units
        axes_metadata = []
        for axis in axes:
            normalized_name = normalize_axis_name(axis)
            axis_info = {
                "name": normalized_name,
                "type": get_axis_type(axis)
            }
            unit = get_axis_unit(axis)
            if unit:
                axis_info["unit"] = unit
            axes_metadata.append(axis_info)

        # Build scale factors from voxel sizes
        scale_factors = []
        for axis in axes:
            if voxel_sizes and axis in voxel_sizes:
                scale_factors.append(float(voxel_sizes[axis]))
            elif voxel_sizes and axis.lower() in voxel_sizes:
                scale_factors.append(float(voxel_sizes[axis.lower()]))
            else:
                scale_factors.append(1.0)

        return {
            "multiscales": [{
                "version": "0.4",
                "name": image_name,
                "axes": axes_metadata,
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
