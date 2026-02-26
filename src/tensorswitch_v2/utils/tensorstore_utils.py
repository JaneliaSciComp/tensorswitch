# Copied from tensorswitch/utils.py for v2 independence
"""
TensorStore utility functions for tensorswitch_v2.

This module contains core TensorStore-related utilities including:
- Context configuration
- Store specifications (zarr3, zarr2, n5, downsample)
- Input format detection
- Source data order detection
"""

import tensorstore as ts
import numpy as np
import os
from urllib.parse import urlparse, unquote


def get_tensorstore_context(num_cores=None):
    """
    Create TensorStore context with concurrency limits based on LSF allocation.

    This prevents TensorStore from using all available cores on the node,
    which can cause jobs to exceed their LSF memory/CPU limits.

    Args:
        num_cores: Number of cores to limit concurrency to.
                   If None, reads from LSB_DJOB_NUMPROC environment variable.
                   Defaults to 1 if env var not set.

    Returns:
        dict: TensorStore context configuration with data_copy_concurrency
              and file_io_concurrency limits set to num_cores
    """
    if num_cores is None:
        num_cores = int(os.getenv("LSB_DJOB_NUMPROC", 1))

    context = {
        "data_copy_concurrency": {"limit": num_cores},
        "file_io_concurrency": {"limit": num_cores}
    }

    return context


def get_kvstore_spec(path):
    """
    Get kvstore specification for TensorStore, supporting HTTP, GCS, S3, and local file paths.

    Args:
        path: File path, HTTP/HTTPS URL, GCS URL (gs://), or S3 URL (s3://)

    Returns:
        dict: kvstore spec with appropriate driver (http, gcs, s3, or file)
    """
    if path.startswith("http://") or path.startswith("https://"):
        # HTTP/HTTPS URL (includes S3 HTTP endpoints like s3prfs.int.janelia.org)
        parsed = urlparse(path)
        return {
            'driver': 'http',
            'base_url': f"{parsed.scheme}://{parsed.netloc}",
            'path': unquote(parsed.path)
        }
    elif path.startswith("gs://"):
        # Google Cloud Storage URL
        path_without_scheme = path[5:]  # Remove 'gs://'
        parts = path_without_scheme.split('/', 1)
        bucket = parts[0]
        object_path = parts[1] if len(parts) > 1 else ''
        return {
            'driver': 'gcs',
            'bucket': bucket,
            'path': object_path
        }
    elif path.startswith("s3://"):
        # AWS S3 URL (protocol format)
        path_without_scheme = path[4:]  # Remove 's3://'
        parts = path_without_scheme.split('/', 1)
        bucket = parts[0]
        object_path = parts[1] if len(parts) > 1 else ''
        return {
            'driver': 's3',
            'bucket': bucket,
            'path': object_path
        }
    else:
        # Local file path
        return {
            'driver': 'file',
            'path': path
        }


def get_input_driver(input_path):
    """
    Detect whether input is TIFF, N5, Zarr2, Zarr3, ND2, IMS, or Neuroglancer Precomputed.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"""
        Could not detect N5/Zarr version or dataset format at: {input_path}.
        {input_path} does not exist.
        """)

    # Check if input is a single file
    if os.path.isfile(input_path):
        lower_path = input_path.lower()
        if lower_path.endswith(".nd2"):
            return "nd2"
        if lower_path.endswith(".ims"):
            return "ims"
        if lower_path.endswith((".tiff", ".tif")):
            return "tiff"
        if lower_path.endswith(".czi"):
            return "czi"

    # For directories, check for format indicator files
    n5_path = os.path.join(input_path, "attributes.json")
    zarr2_path = os.path.join(input_path, ".zarray")
    zarr3_path = os.path.join(input_path, "zarr.json")
    precomputed_path = os.path.join(input_path, "info")

    if os.path.exists(n5_path):
        return "n5"
    elif os.path.exists(zarr2_path):
        return "zarr"
    elif os.path.exists(zarr3_path):
        return "zarr3"
    elif os.path.exists(precomputed_path):
        return "precomputed"
    else:
        # Check for .tiff or .tif files in directory
        tiff_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".tiff", ".tif"))
        ]
        if tiff_files:
            return "tiff"

        # Check for .nd2 files in directory
        nd2_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(".nd2")
        ]
        if nd2_files:
            return "nd2"

        raise ValueError(f"""
        Could not detect N5/Zarr version or dataset format at: {input_path}.
        {n5_path} does not exist.
        {zarr2_path} does not exist.
        {zarr3_path} does not exist.
        {precomputed_path} does not exist.
        And no TIFF, ND2, or IMS files found in folder.
        """)


def get_zarr_store_spec(path):
    """
    Get TensorStore spec for Zarr/N5 stores, supporting local paths and remote URLs.

    Args:
        path: Path to dataset (local path, HTTP/HTTPS URL, gs:// URL, or s3:// URL) or dict spec

    Returns:
        dict: TensorStore spec with appropriate driver and kvstore
    """
    if isinstance(path, dict):
        return path

    # If remote path (HTTP, GCS, S3), assume N5 driver directly
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://") or
                                   path.startswith("gs://") or path.startswith("s3://")):
        return {
            'driver': 'n5',
            'kvstore': get_kvstore_spec(path)
        }

    input_driver = get_input_driver(path)

    if input_driver == "tiff":
        raise ValueError(f"Cannot build TensorStore spec for TIFF folder: {path}")

    zarr_store_spec = {
        'driver': input_driver,
        'kvstore': {'driver': 'file', 'path': path}
    }
    return zarr_store_spec


def zarr3_store_spec(path, shape, dtype, use_shard=True, level_path="s0", use_ome_structure=True,
                     custom_shard_shape=None, custom_chunk_shape=None, use_v2_encoding=False,
                     use_fortran_order=False, axes_order=None, compression=None):
    """
    Create Zarr3 store specification with smart chunking based on axes.

    Non-spatial axes (c, t, v) get chunk size 1 for per-channel/per-timepoint access.
    Spatial axes (z, y, x) get default 256 inner chunk, 1024 shard.

    Args:
        level_path: Level subdirectory name (default "s0" for Janelia convention)
        compression: Optional compression codec dict from source metadata.
                    Format: {'name': 'zstd', 'configuration': {'level': N}}
                    If None, defaults to zstd level 5 (sharded) or level 1 (non-sharded).
    """
    NON_SPATIAL_AXES = ['c', 't', 'v', 'channel']

    # Default compression codecs
    DEFAULT_SHARDED_COMPRESSION = {'name': 'zstd', 'configuration': {'level': 5}}
    DEFAULT_NONSHARDED_COMPRESSION = {'name': 'zstd', 'configuration': {'level': 1}}

    def build_default_shape(shape, axes_order, spatial_size):
        """Build default chunk/shard shape respecting axis types."""
        if axes_order is None:
            return [spatial_size] * len(shape)

        result = []
        for i, axis in enumerate(axes_order):
            if axis.lower() in NON_SPATIAL_AXES:
                result.append(1)
            else:
                result.append(spatial_size)
        return result

    if use_shard:
        # Use custom chunk shape if provided, otherwise build based on axes
        if custom_chunk_shape is not None:
            inner_chunk_shape = custom_chunk_shape
        else:
            inner_chunk_shape = build_default_shape(shape, axes_order, 256)

        # Adjust inner chunk shape for different array dimensions
        if len(shape) == 3 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = inner_chunk_shape
        elif len(shape) == 3 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1] + inner_chunk_shape
        elif len(shape) == 4 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = [1] + inner_chunk_shape
        elif len(shape) == 4 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1, 1] + inner_chunk_shape
        elif len(shape) == 5 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = [1, 1] + inner_chunk_shape
        elif len(shape) == 5 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1, 1, 1] + inner_chunk_shape
        else:
            adjusted_inner_chunk = inner_chunk_shape

        # Build inner codecs for sharding
        # NOTE: When using F-order with sharding, transpose must be INSIDE the inner codecs
        # (before bytes/compression), not at the array level. This is a TensorStore/Zarr3 requirement.
        inner_codecs = []
        if use_fortran_order:
            transpose_order = list(range(len(shape) - 1, -1, -1))
            inner_codecs.append({'name': 'transpose', 'configuration': {'order': transpose_order}})

        # Use passed compression or default for sharded
        compression_codec = compression if compression else DEFAULT_SHARDED_COMPRESSION
        inner_codecs.extend([
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            compression_codec
        ])

        # Array-level codecs (transpose is inside sharding's inner codecs when using F-order)
        codecs = []

        # Add sharding codec
        codecs.append({
            'name': 'sharding_indexed',
            'configuration': {
                'chunk_shape': adjusted_inner_chunk,
                'codecs': inner_codecs,
                'index_codecs': [
                    {'name': 'bytes', 'configuration': {'endian': 'little'}},
                    {'name': 'crc32c'}
                ],
                'index_location': 'end'
            }
        })

        # Use custom shard shape if provided, otherwise default
        if custom_shard_shape is not None:
            if len(shape) == 3 and len(custom_shard_shape) == 3:
                chunk_shape = custom_shard_shape
            elif len(shape) == 3 and len(custom_shard_shape) == 2:
                chunk_shape = [1] + custom_shard_shape
            elif len(shape) == 4 and len(custom_shard_shape) == 3:
                chunk_shape = [1] + custom_shard_shape
            elif len(shape) == 4 and len(custom_shard_shape) == 2:
                chunk_shape = [1, 1] + custom_shard_shape
            elif len(shape) == 5 and len(custom_shard_shape) == 3:
                chunk_shape = [1, 1] + custom_shard_shape
            elif len(shape) == 5 and len(custom_shard_shape) == 2:
                chunk_shape = [1, 1, 1] + custom_shard_shape
            else:
                chunk_shape = custom_shard_shape
        else:
            chunk_shape = build_default_shape(shape, axes_order, 1024)
    else:
        # Non-sharded codecs
        codecs = []
        if use_fortran_order:
            transpose_order = list(range(len(shape) - 1, -1, -1))
            codecs.append({'name': 'transpose', 'configuration': {'order': transpose_order}})

        # Use passed compression or default for non-sharded
        compression_codec = compression if compression else DEFAULT_NONSHARDED_COMPRESSION
        codecs.extend([
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            compression_codec
        ])
        # Use custom_chunk_shape if provided
        if custom_chunk_shape is not None:
            if len(custom_chunk_shape) < len(shape):
                extra_dims = len(shape) - len(custom_chunk_shape)
                chunk_shape = [1] * extra_dims + list(custom_chunk_shape)
            else:
                chunk_shape = list(custom_chunk_shape)
            print(f"Using custom chunk shape for non-sharded: {chunk_shape}")
        else:
            # Default chunk shapes
            if len(shape) == 5:
                chunk_shape = [1, 1, 64, 64, 64]
            elif len(shape) == 4:
                chunk_shape = [1, 64, 64, 64]
            elif len(shape) == 3:
                chunk_shape = [64, 64, 64]
            elif len(shape) == 2:
                chunk_shape = [64, 64]
            elif len(shape) == 1:
                chunk_shape = [64]

    # Create path based on whether OME-ZARR structure is needed
    if use_ome_structure:
        array_path = os.path.join(path, level_path)
    else:
        array_path = path

    # Determine dimension names (normalize to OME-NGFF standard)
    def normalize_axis(ax):
        ax_lower = ax.lower()
        if ax_lower == 'channel':
            return 'c'
        elif ax_lower == 'v':
            return 't'
        return ax_lower

    if axes_order is not None and len(axes_order) == len(shape):
        dimension_names = [normalize_axis(ax) for ax in axes_order]
        print(f"Using dimension names from source metadata: {dimension_names}")
    else:
        if len(shape) == 3:
            if shape[0] <= 10:
                dimension_names = ["c", "y", "x"]
            else:
                dimension_names = ["z", "y", "x"]
        elif len(shape) == 4:
            dimension_names = ["c", "z", "y", "x"]
        elif len(shape) == 5:
            dimension_names = ["t", "c", "z", "y", "x"]
        else:
            dimension_names = [f"dim_{i}" for i in range(len(shape))]
        print(f"Inferred dimension names from shape: {dimension_names}")

    return {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': array_path},
        'metadata': {
            'shape': shape,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': chunk_shape}},
            'chunk_key_encoding': {'name': 'v2', 'configuration': {'separator': '/'}} if use_v2_encoding else {'name': 'default'},
            'data_type': dtype,
            'node_type': 'array',
            'codecs': codecs,
            'dimension_names': dimension_names
        }
    }


def downsample_spec(base_spec, array_shape=None, dimension_names=None, custom_factors=None, method="mode"):
    """
    Create downsample spec with appropriate factors based on dimension names.

    Args:
        base_spec: Base zarr store spec
        array_shape: Shape of the array
        dimension_names: List of dimension names (e.g., ['c', 'z', 'y', 'x'])
        custom_factors: Optional custom downsampling factors
        method: Downsampling method ("mean", "mode", "median", "stride", "min", "max")
    """
    # If custom factors provided, use them directly
    if custom_factors is not None:
        downsample_factors = custom_factors
    else:
        downsample_factors = [2] * len(array_shape) if array_shape is not None else [2, 2, 2]

        if dimension_names and array_shape:
            downsample_factors = []
            for dim_name in dimension_names:
                if dim_name in ['c', 't']:
                    downsample_factors.append(1)
                else:
                    downsample_factors.append(2)

    return {
        'driver': 'downsample',
        'base': get_zarr_store_spec(base_spec),
        'downsample_factors': downsample_factors,
        'downsample_method': method
    }


def detect_source_order(source_data):
    """
    Detect source data order (C-order vs F-order) from TensorStore object or numpy/dask array.

    Args:
        source_data: TensorStore object OR numpy/dask array

    Returns:
        dict: {
            'is_fortran_order': bool,
            'inner_order': list or None,
            'suggested_axes': list,
            'description': str
        }
    """
    import dask.array as da

    shape = source_data.shape
    ndim = len(shape)

    has_tensorstore_attrs = hasattr(source_data, 'chunk_layout')

    if has_tensorstore_attrs:
        inner_order = list(source_data.chunk_layout.inner_order)
        expected_fortran_order = list(range(ndim - 1, -1, -1))
        expected_c_order = list(range(ndim))
        is_fortran_order = (inner_order == expected_fortran_order)
        is_c_order = (inner_order == expected_c_order)
    else:
        inner_order = None
        if isinstance(source_data, da.Array):
            try:
                first_chunk = source_data.blocks[tuple([0] * ndim)].compute()
                is_fortran_order = first_chunk.flags.f_contiguous and not first_chunk.flags.c_contiguous
                is_c_order = first_chunk.flags.c_contiguous and not first_chunk.flags.f_contiguous
            except:
                is_fortran_order = False
                is_c_order = True
        elif isinstance(source_data, np.ndarray):
            is_fortran_order = source_data.flags.f_contiguous and not source_data.flags.c_contiguous
            is_c_order = source_data.flags.c_contiguous and not source_data.flags.f_contiguous
        else:
            is_fortran_order = False
            is_c_order = True

    # Determine suggested axes labels
    if is_fortran_order:
        if ndim == 3:
            suggested_axes = ['x', 'y', 'z']
        elif ndim == 4:
            suggested_axes = ['x', 'y', 'z', 'c']
        elif ndim == 5:
            suggested_axes = ['x', 'y', 'z', 'c', 't']
        else:
            suggested_axes = [f'dim_{i}' for i in range(ndim)]
        description = f"F-order (Fortran/column-major): dimensions are {suggested_axes}"
    elif is_c_order:
        if ndim == 3:
            suggested_axes = ['z', 'y', 'x']
        elif ndim == 4:
            suggested_axes = ['c', 'z', 'y', 'x']
        elif ndim == 5:
            suggested_axes = ['t', 'c', 'z', 'y', 'x']
        else:
            suggested_axes = [f'dim_{i}' for i in range(ndim)]
        description = f"C-order (C/row-major): dimensions are {suggested_axes}"
    else:
        suggested_axes = [f'dim_{i}' for i in range(ndim)]
        description = f"Custom order (inner_order={inner_order})"
        is_fortran_order = False

    return {
        'is_fortran_order': is_fortran_order,
        'inner_order': inner_order,
        'suggested_axes': suggested_axes,
        'description': description
    }


def n5_store_spec(n5_level_path):
    """
    Create N5 store specification supporting HTTP, GCS, S3, and local file paths.
    """
    return {
        'driver': 'n5',
        'kvstore': get_kvstore_spec(n5_level_path)
    }


def zarr2_store_spec(zarr_level_path, shape, chunks=None, use_fortran_order=False, axes_order=None, dtype="|u1", compressor=None):
    """
    Create Zarr2 store specification with smart chunking based on axes.

    Non-spatial axes (c, t, v) get chunk size 1 for per-channel/per-timepoint access.
    Spatial axes (z, y, x) get default 64 chunk size (matching Zarr3 non-sharded).

    Args:
        zarr_level_path: Path to zarr level
        shape: Array shape
        chunks: Chunk shape (if None, auto-calculated based on axes_order)
        use_fortran_order: If True, use F-order (column-major); if False, use C-order (row-major)
        axes_order: List of axis names (e.g., ['c', 'y', 'x']) for smart chunking
        dtype: Data type string (e.g., '<u2' for uint16)
        compressor: Compressor dict (default: zstd level 5)
    """
    # Non-spatial axes that should have chunk size 1 for efficient per-slice access
    NON_SPATIAL_AXES = ['c', 't', 'v', 'channel']
    # Default spatial chunk size for Zarr2 (no sharding, so each chunk = one file)
    # Matches Zarr3 non-sharded default. 64^3 × 2 bytes = 512 KB per chunk for uint16.
    DEFAULT_SPATIAL_CHUNK = 64

    def build_smart_chunks(shape, axes_order):
        """Build chunk shape respecting axis types - same logic as zarr3."""
        if axes_order is None or len(axes_order) != len(shape):
            # Fallback: infer axes from shape
            axes_order = infer_axes_from_shape(shape)

        result = []
        for i, axis in enumerate(axes_order):
            if axis.lower() in NON_SPATIAL_AXES:
                result.append(1)  # Non-spatial: 1 for per-channel access
            else:
                # Spatial: use 64 or array size if smaller
                result.append(min(DEFAULT_SPATIAL_CHUNK, shape[i]))
        return result

    def infer_axes_from_shape(shape):
        """Infer axis names from shape - same logic as zarr3."""
        if len(shape) == 2:
            return ["y", "x"]
        elif len(shape) == 3:
            # For 3D, assume channels if first dimension is small, otherwise Z
            if shape[0] <= 10:
                return ["c", "y", "x"]
            else:
                return ["z", "y", "x"]
        elif len(shape) == 4:
            # CZYX or TZYX - check first dim
            if shape[0] <= 10:
                return ["c", "z", "y", "x"]
            else:
                return ["t", "z", "y", "x"]
        elif len(shape) == 5:
            return ["t", "c", "z", "y", "x"]
        else:
            return [f"dim_{i}" for i in range(len(shape))]

    # Auto-calculate chunks if not provided
    if chunks is None:
        chunks = build_smart_chunks(shape, axes_order)
        print(f"Zarr2 smart chunking: axes={axes_order or infer_axes_from_shape(shape)}, chunks={chunks}")

    # Default compressor
    if compressor is None:
        compressor = {'id': 'zstd', 'level': 5}

    return {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': zarr_level_path},
        'metadata': {
            'shape': shape,
            'chunks': chunks,
            'dtype': dtype,
            'compressor': compressor,
            'order': 'F' if use_fortran_order else 'C',  # Preserve source order
            'dimension_separator': "/",
            'fill_value': 0  # Required for Neuroglancer compatibility (null causes display issues)
        }
    }
