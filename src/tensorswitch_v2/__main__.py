"""
CLI entry point for tensorswitch_v2 conversion and downsampling.

Usage:
    # Single file conversion (s0)
    python -m tensorswitch_v2 -i input.tif -o output.zarr
    python -m tensorswitch_v2 -i input.n5 -o output.zarr --chunk_shape 32,256,256
    python -m tensorswitch_v2 -i input.tif -o output.zarr --submit -P scicompsoft

    # Batch conversion (directory of files)
    python -m tensorswitch_v2 -i /path/to/tiff_dir/ -o /path/to/output_dir/ --pattern "*.tif"
    python -m tensorswitch_v2 -i /path/to/tiff_dir/ -o /path/to/output_dir/ --submit -P project

    # Check batch status
    python -m tensorswitch_v2 --status -i /path/to/tiff_dir/ -o /path/to/output_dir/

    # Downsampling (single level, parallel from s0)
    python -m tensorswitch_v2 --downsample -i /path/to/s0 -o /path/to/dataset.zarr \\
        --target_level 2 --cumulative_factors 1,4,4

    # Auto-multiscale (full pyramid, all levels in parallel)
    python -m tensorswitch_v2 --auto_multiscale -i /path/to/s0 -o /path/to/dataset.zarr \\
        --submit -P scicompsoft
"""

import os
import sys
import subprocess
import math
import argparse
from typing import Optional, Tuple

import numpy as np

os.umask(0o0002)  # Team permissions: rwxrwxr-x

__version__ = "2.0.0-beta"


def resolve_downsample_method(method: str, input_path: str) -> str:
    """
    Resolve 'auto' downsample method to actual method based on input path.

    Args:
        method: Downsample method ('auto', 'mean', 'mode', etc.)
        input_path: Path to input data for heuristic detection

    Returns:
        str: Resolved method ('mean' or 'mode' if input was 'auto')
    """
    if method != 'auto':
        return method

    # Use filename heuristics to detect label/segmentation data
    label_keywords = ['label', 'mask', 'seg', 'annotation', 'roi', 'binary', 'instance']

    # Check multiple levels of the path (handles /data/labels/dataset.zarr/s0)
    # Normalize and split path into components
    path_parts = input_path.lower().replace('\\', '/').split('/')
    # Check last 4 components (covers most directory structures)
    check_parts = ' '.join(path_parts[-4:]) if len(path_parts) >= 4 else ' '.join(path_parts)

    for keyword in label_keywords:
        if keyword in check_parts:
            return 'mode'

    # Default to 'mean' for intensity images
    return 'mean'


def find_base_level(input_path: str, verbose: bool = False) -> tuple:
    """
    Find the base resolution level (s0) from a zarr path.

    Accepts flexible input:
    - /data/dataset.zarr/s0  → returns (s0_path, root_path)
    - /data/dataset.zarr     → auto-detects s0 from metadata or common patterns
    - /data/dataset.zarr/0   → returns as-is if valid

    Detection order:
    1. If path ends with known level pattern (s0, s1, 0, 1, etc.) → use as-is
    2. Check OME-NGFF metadata for multiscales[0].datasets[0].path
    3. Fallback to common subdirectories: s0, 0

    Args:
        input_path: Path to zarr dataset (root or specific level)
        verbose: Print detection info

    Returns:
        tuple: (s0_path, root_path)

    Raises:
        ValueError: If base level cannot be found
    """
    import json
    import re

    input_path = input_path.rstrip('/\\')

    # Pattern for resolution level directories
    level_pattern = re.compile(r'^(s?\d+)$')

    # Check if path already ends with a level indicator
    basename = os.path.basename(input_path)
    if level_pattern.match(basename):
        # Already pointing to a level (s0, s1, 0, 1, etc.)
        s0_path = input_path
        root_path = os.path.dirname(input_path)
        if verbose:
            print(f"Input is level path: {basename}")
        return s0_path, root_path

    # Input is root path - need to find base level
    root_path = input_path

    # Strategy 1: Check OME-NGFF metadata
    # Try Zarr v3 metadata first
    zarr_json_path = os.path.join(root_path, 'zarr.json')
    zattrs_path = os.path.join(root_path, '.zattrs')

    base_level_path = None

    if os.path.exists(zarr_json_path):
        try:
            with open(zarr_json_path, 'r') as f:
                metadata = json.load(f)
            # Zarr v3: check attributes.ome.multiscales
            attrs = metadata.get('attributes', {})
            ome_attrs = attrs.get('ome', attrs)  # ome key or root level
            multiscales = ome_attrs.get('multiscales', [])
            if multiscales and multiscales[0].get('datasets'):
                base_level_path = multiscales[0]['datasets'][0].get('path')
                if verbose:
                    print(f"Found base level from zarr.json metadata: {base_level_path}")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    if not base_level_path and os.path.exists(zattrs_path):
        try:
            with open(zattrs_path, 'r') as f:
                metadata = json.load(f)
            # Zarr v2: check multiscales directly in .zattrs
            multiscales = metadata.get('multiscales', [])
            if multiscales and multiscales[0].get('datasets'):
                base_level_path = multiscales[0]['datasets'][0].get('path')
                if verbose:
                    print(f"Found base level from .zattrs metadata: {base_level_path}")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Strategy 2: Fallback to common subdirectory patterns
    if not base_level_path:
        for candidate in ['s0', '0']:
            candidate_path = os.path.join(root_path, candidate)
            if os.path.exists(candidate_path):
                base_level_path = candidate
                if verbose:
                    print(f"Found base level by directory scan: {base_level_path}")
                break

    if not base_level_path:
        raise ValueError(
            f"Could not find base resolution level in {root_path}. "
            f"Expected: OME-NGFF metadata with multiscales, or subdirectory 's0' or '0'. "
            f"Please specify the full path to the base level (e.g., {root_path}/s0)"
        )

    s0_path = os.path.join(root_path, base_level_path)

    if not os.path.exists(s0_path):
        raise ValueError(f"Base level path does not exist: {s0_path}")

    return s0_path, root_path


def parse_args(argv=None):
    """Parse command-line arguments."""
    epilog = """
Examples:
  # Single file conversion (TIFF to Zarr3)
  pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr

  # Convert with custom chunk/shard shapes (BigStitcher compatible)
  pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr \\
      --chunk_shape 32,32,32 --shard_shape 256,1024,1024

  # Submit to LSF cluster
  pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr \\
      --submit -P scicompsoft

  # Batch convert directory of TIFFs
  pixi run python -m tensorswitch_v2 -i /path/to/tiffs/ -o /path/to/output/ \\
      --pattern "*.tif" --submit -P scicompsoft --max_concurrent 100

  # Check batch conversion status
  pixi run python -m tensorswitch_v2 --status -i /path/to/tiffs/ -o /path/to/output/

  # Generate multi-scale pyramid (all levels)
  pixi run python -m tensorswitch_v2 --auto_multiscale \\
      -i /path/to/dataset.zarr/s0 -o /path/to/dataset.zarr --submit -P scicompsoft

Supported input formats:
  TIFF, ND2, IMS, CZI, N5, Zarr v2/v3, Precomputed, HDF5, and 200+ via BIOIO

Supported output formats:
  zarr3 (default, with sharding), zarr2, n5
"""
    parser = argparse.ArgumentParser(
        prog="tensorswitch_v2",
        description="TensorSwitch v2: Convert microscopy data between formats with multi-scale pyramid support.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", "-V", action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version and exit",
    )

    # Input/output arguments (not required for --batch_worker mode)
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input file or directory path. Format is auto-detected from extension. "
             "For batch mode, provide a directory and use --pattern to filter files.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path. For single file: output.zarr, For batch: output directory.",
    )

    # Output format
    parser.add_argument(
        "--output_format", default="zarr3",
        choices=["zarr3", "zarr2", "n5"],
        help="Output format (default: zarr3)",
    )

    # Presets for common use cases
    parser.add_argument(
        "--preset", default=None,
        choices=["webknossos"],
        help="Use preset configuration. 'webknossos': zarr3, chunk 32x32x32, shard 1024x1024x1024, zstd.",
    )

    # Dataset paths
    parser.add_argument(
        "--dataset_path", default="",
        help="Path within input container (e.g., 's0' for N5)",
    )
    parser.add_argument(
        "--level_path", default="s0",
        help="Level subdirectory name in output (default: s0)",
    )

    # Chunk/shard configuration
    parser.add_argument(
        "--chunk_shape", default=None,
        help="Comma-separated chunk shape (e.g., '32,256,256')",
    )
    parser.add_argument(
        "--shard_shape", default=None,
        help="Comma-separated shard shape (Zarr3 only, e.g., '64,512,512')",
    )
    parser.add_argument(
        "--no_sharding", action="store_true",
        help="Disable sharding (Zarr3 only)",
    )

    # Compression
    parser.add_argument(
        "--compression", default="zstd",
        help="Compression codec (default: zstd)",
    )
    parser.add_argument(
        "--compression_level", type=int, default=5,
        help="Compression level (default: 5)",
    )

    # Voxel size override
    parser.add_argument(
        "--voxel_size", default=None,
        help="Override voxel size in nanometers, comma-separated X,Y,Z (e.g., '9,9,12'). "
             "Use when source file lacks embedded voxel size metadata.",
    )

    # Label/segmentation mode
    parser.add_argument(
        "--is-label", "--is_label", action="store_true",
        dest="is_label",
        help="Mark output as label/segmentation data. Adds OME-NGFF image-label metadata. "
             "Auto-detected for uint64/uint32 data types, use this flag to force for other types.",
    )

    # Manual bsub chunk-range mode
    parser.add_argument(
        "--start_idx", type=int, default=None,
        help="Starting chunk index (for manual bsub worker)",
    )
    parser.add_argument(
        "--stop_idx", type=int, default=None,
        help="Ending chunk index (for manual bsub worker)",
    )
    parser.add_argument(
        "--write_metadata", action="store_true",
        help="Force metadata write (for manual bsub last job)",
    )

    # CZI-specific
    parser.add_argument(
        "--view_index", type=int, default=None,
        help="CZI view index (None = all views as 5D VCZYX)",
    )

    # Downsampling mode
    parser.add_argument(
        "--downsample", action="store_true",
        help="Downsample mode: create a single level from s0",
    )
    parser.add_argument(
        "--auto_multiscale", action="store_true",
        help="Auto-multiscale mode: generate full pyramid from s0 (all levels in parallel)",
    )
    parser.add_argument(
        "--target_level", type=int, default=None,
        help="Target level to create in downsample mode (1=s1, 2=s2, etc.)",
    )
    parser.add_argument(
        "--single_level_factor", type=str, default=None,
        help="Downsampling factor for single-level mode (--downsample). "
             "Comma-separated cumulative factor from s0 (e.g., '1,4,4' for s2 with 4x Y,X).",
    )
    parser.add_argument(
        "--cumulative_factors", type=str, default=None,
        help=argparse.SUPPRESS,  # Hidden: backward compatibility alias for --single_level_factor
    )
    parser.add_argument(
        "--use_shard", type=int, default=1,
        help="Use sharding for output (1=yes, 0=no, default: 1)",
    )
    parser.add_argument(
        "--per_level_factors", type=str, default=None,
        help="Custom per-level downsampling factors, semicolon-separated "
             "(e.g., '1,2,2;1,2,2;1,2,2;1,2,2' for 4 levels with Z-skip). "
             "Bypasses auto-calculation from voxel sizes. Used with --auto_multiscale.",
    )
    parser.add_argument(
        "--downsample_method", type=str, default="auto",
        choices=["auto", "mean", "mode", "median", "min", "max", "stride"],
        help="Downsampling method: auto (default, detects from filename), "
             "mean (intensity images), mode (labels/segmentation), "
             "median (noise reduction), stride (fastest), min, max.",
    )

    # Output control
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    # LSF cluster submission
    # NOTE: When adding new conversion arguments above, also add passthrough
    # logic in submit_job() so they are forwarded to the cluster job.
    parser.add_argument(
        "--submit", action="store_true",
        help="Submit as a single LSF bsub job instead of running locally",
    )
    parser.add_argument(
        "--project", "-P", type=str, default=None,
        help="LSF project name (required when --submit is used)",
    )
    parser.add_argument(
        "--memory", type=int, default=None,
        help="Memory in GB for LSF job (default: auto-calculated from source data)",
    )
    parser.add_argument(
        "--wall_time", type=str, default=None,
        help="Wall time for LSF job, format H:MM (default: auto-calculated from source data)",
    )
    parser.add_argument(
        "--cores", type=int, default=None,
        help="Number of cores for LSF job (default: auto-calculated from memory)",
    )
    parser.add_argument(
        "--job_group", type=str, default="/scicompsoft/chend/tensorstore",
        help="LSF job group path (default: /scicompsoft/chend/tensorstore)",
    )
    parser.add_argument(
        "--use_bioio", action="store_true",
        help="Force BIOIO adapter (Tier 3) instead of auto-detected Tier 2 reader",
    )
    parser.add_argument(
        "--use_bioformats", action="store_true",
        help="Force Bio-Formats reader (Tier 3+, Java-backed) for 150+ formats. "
             "Requires: conda install -c conda-forge scyjava && pip install bioio-bioformats",
    )

    # Memory order (mutually exclusive)
    order_group = parser.add_mutually_exclusive_group()
    order_group.add_argument(
        "--force_c_order", action="store_true",
        help="Force C-order (row-major) output, overriding auto-detected source order",
    )
    order_group.add_argument(
        "--force_f_order", action="store_true",
        help="Force F-order (column-major) output, overriding auto-detected source order",
    )

    # Layout control
    parser.add_argument(
        "--expand-to-5d", action="store_true",
        help="Force 5D TCZYX expansion (old behavior). "
             "Default: preserve source dimensionality and axis order per OME-NGFF RFC-3",
    )

    # Batch processing mode
    # NOTE: Batch mode is auto-detected when input is a directory (no file extension)
    parser.add_argument(
        "--pattern", default="*.tif",
        help="File pattern for batch mode (default: *.tif)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search subdirectories in batch mode",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=100,
        help="Max concurrent LSF jobs in batch mode (default: 100)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip files that already have output (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Process all files even if output exists",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check batch conversion status",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be done without actually doing it",
    )
    parser.add_argument(
        "--show_spec", action="store_true",
        help="Show input/output TensorStore specs and conversion summary, then exit without converting",
    )
    parser.add_argument(
        "--omero", action="store_true",
        help="Include structured omero channel metadata (extracted from OME-XML) for visualization tools",
    )

    # Batch worker mode (used by LSF job array)
    parser.add_argument(
        "--batch_worker", action="store_true",
        help=argparse.SUPPRESS,  # Hidden: used by LSF job array workers
    )
    parser.add_argument(
        "--index_file", type=str, default=None,
        help=argparse.SUPPRESS,  # Hidden: index file for batch worker
    )

    return parser.parse_args(argv)


def validate_input_path(path: str, allow_directory: bool = True) -> None:
    """Validate that input path exists and is readable.

    Args:
        path: Input file or directory path
        allow_directory: If True, directories are allowed (batch mode)

    Raises:
        FileNotFoundError: If path does not exist
        PermissionError: If path is not readable
        ValueError: If path is a directory but not allowed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input path does not exist: {path}\n"
            f"Please check the path and try again."
        )

    if os.path.isdir(path):
        if not allow_directory:
            raise ValueError(
                f"Input path is a directory: {path}\n"
                f"For batch mode, use --pattern to specify file pattern.\n"
                f"For single file conversion, provide a file path."
            )
        if not os.access(path, os.R_OK):
            raise PermissionError(
                f"Input directory is not readable: {path}\n"
                f"Check permissions with: ls -la {path}"
            )
    else:
        if not os.access(path, os.R_OK):
            raise PermissionError(
                f"Input file is not readable: {path}\n"
                f"Check permissions with: ls -la {path}"
            )


def validate_output_path(path: str) -> None:
    """Validate that output path's parent directory exists and is writable.

    Args:
        path: Output file or directory path

    Raises:
        FileNotFoundError: If parent directory does not exist
        PermissionError: If parent directory is not writable
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(parent):
        raise FileNotFoundError(
            f"Output parent directory does not exist: {parent}\n"
            f"Create it with: mkdir -p {parent}"
        )

    if not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Output directory is not writable: {parent}\n"
            f"Check permissions with: ls -la {parent}"
        )


def parse_shape(s: str, param_name: str = "shape") -> Tuple[int, ...]:
    """Parse a comma-separated shape string into a tuple of ints.

    Args:
        s: Shape string like '32,256,256'
        param_name: Name of the parameter for error messages

    Returns:
        Tuple of ints, e.g. (32, 256, 256)

    Raises:
        ValueError: If shape string is invalid
    """
    try:
        parts = s.split(",")
        shape = tuple(int(x.strip()) for x in parts)

        # Validate all dimensions are positive
        for i, dim in enumerate(shape):
            if dim <= 0:
                raise ValueError(
                    f"Invalid {param_name}: '{s}'\n"
                    f"Dimension {i} has value {dim}, but all dimensions must be positive.\n"
                    f"Example: --{param_name} 32,256,256"
                )
        return shape
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(
                f"Invalid {param_name}: '{s}'\n"
                f"Expected comma-separated integers (e.g., '32,256,256').\n"
                f"Got non-integer value in: {s}"
            )
        raise


def parse_per_level_factors(factors_str: str):
    """Parse per-level factors string into list of factor lists.

    Args:
        factors_str: Semicolon-separated factors, e.g., '1,2,2;1,2,2;1,2,2;1,2,2'
                     Each level's factors are comma-separated.

    Returns:
        List of lists, e.g., [[1,2,2], [1,2,2], [1,2,2], [1,2,2]]

    Raises:
        ValueError: If format is invalid

    Example:
        >>> parse_per_level_factors('1,2,2;1,2,2;1,2,2')
        [[1, 2, 2], [1, 2, 2], [1, 2, 2]]
    """
    try:
        levels = factors_str.split(';')
        result = []
        for i, level_str in enumerate(levels):
            factors = [int(x.strip()) for x in level_str.split(',')]
            # Validate all factors are positive
            for j, f in enumerate(factors):
                if f <= 0:
                    raise ValueError(
                        f"Invalid factor at level {i+1}, position {j}: {f}\n"
                        f"All factors must be positive integers."
                    )
            result.append(factors)
        return result
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(
                f"Invalid --per_level_factors: '{factors_str}'\n"
                f"Expected semicolon-separated levels with comma-separated factors.\n"
                f"Example: '1,2,2;1,2,2;1,2,2;1,2,2' for 4 levels with Z-skip"
            )
        raise


def create_reader(args):
    """Create a reader from CLI arguments using auto-detection."""
    from .api.readers import Readers

    path = args.input
    path_lower = path.lower()

    # Force Bio-Formats reader if requested (Tier 3+, Java-backed)
    if getattr(args, 'use_bioformats', False):
        return Readers.bioformats(path)

    # Force BIOIO adapter if requested (for testing Tier 3)
    if getattr(args, 'use_bioio', False):
        return Readers.bioio(path)

    # CZI: use dedicated Tier 2 reader with optional view_index
    if path_lower.endswith(".czi"):
        return Readers.czi(path, view_index=args.view_index)

    # If dataset_path is provided, use explicit reader for formats that need it
    if args.dataset_path:
        if path_lower.endswith(".n5"):
            return Readers.n5(path, dataset_path=args.dataset_path)
        elif path_lower.endswith(".zarr"):
            return Readers.auto_detect(path)
        elif path_lower.endswith((".h5", ".hdf5")):
            return Readers.hdf5(path, dataset_path=args.dataset_path)

    return Readers.auto_detect(path)


def create_writer(args):
    """Create a writer from CLI arguments."""
    from .api.writers import Writers

    fmt = args.output_format

    # Determine level_path: use "0" for zarr2 by default, "s0" for others
    level_path = args.level_path
    if level_path == "s0" and fmt == "zarr2":
        level_path = "0"  # Default to numeric paths for OME-NGFF viewer compatibility

    if fmt == "zarr3":
        return Writers.zarr3(
            output_path=args.output,
            use_sharding=not args.no_sharding,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=level_path,
            include_omero=getattr(args, 'omero', False),
        )
    elif fmt == "zarr2":
        return Writers.zarr2(
            output_path=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=level_path,
            include_omero=getattr(args, 'omero', False),
        )
    elif fmt == "n5":
        return Writers.n5(
            output_path=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            dataset_path=level_path,
        )
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def _get_input_metadata(args):
    """Read input shape, dtype, and axes for resource estimation.

    Returns:
        tuple: (shape, dtype_str, axes_order) where axes_order may be None
    """
    reader = create_reader(args)
    spec = reader.get_tensorstore_spec()
    if spec.get('driver') == 'array' and 'array' in spec:
        arr = spec['array']
        # Try to get axes from spec schema
        axes_order = spec.get('schema', {}).get('dimension_names')
        return tuple(arr.shape), str(arr.dtype), axes_order
    else:
        import tensorstore as ts
        from .utils import get_tensorstore_context
        spec['context'] = get_tensorstore_context()
        store = ts.open(spec, read=True).result()
        # Get axes from TensorStore domain labels (e.g., precomputed has ['x','y','z','channel'])
        axes_order = None
        if hasattr(store, 'domain') and hasattr(store.domain, 'labels'):
            labels = store.domain.labels
            if labels and all(labels):
                # Normalize 'channel' to 'c'
                axes_order = ['c' if l.lower() == 'channel' else l.lower() for l in labels]
        # Fallback to spec schema if available
        if not axes_order:
            axes_order = spec.get('schema', {}).get('dimension_names')
        return tuple(store.shape), store.dtype.name, axes_order


def _estimate_shard_info(args, volume_shape, dtype_str, axes_order=None):
    """Estimate shard shape and total shards for resource calculation.

    Args:
        args: Parsed command-line arguments
        volume_shape: Input volume shape (may be 3D, 4D, or 5D)
        dtype_str: Data type string
        axes_order: Axis names (e.g., ['x','y','z','c'] for precomputed)

    Returns:
        (shard_shape, total_shards) tuple. If no sharding, shard_shape is the
        chunk_shape and total_shards is the total number of chunks.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    writer = create_writer(args)

    # Determine chunk shape
    if args.chunk_shape:
        chunk_shape = parse_shape(args.chunk_shape, "chunk_shape")
    else:
        chunk_shape = writer.get_default_chunk_shape(volume_shape, dtype_size=dtype_bytes)

    # Determine shard shape
    use_sharding = args.output_format == "zarr3" and not args.no_sharding
    if use_sharding:
        if args.shard_shape:
            shard_shape = parse_shape(args.shard_shape, "shard_shape")
        else:
            shard_shape = writer.get_default_shard_shape(chunk_shape, volume_shape)
            if shard_shape is None:
                shard_shape = chunk_shape
    else:
        shard_shape = chunk_shape

    # Expand both shapes to 5D for compatible division
    # Use writer's expansion methods if available (Zarr3Writer has _expand_to_5d)
    if hasattr(writer, '_expand_to_5d') and hasattr(writer, '_expand_shard_shape'):
        expanded_shape, _, _ = writer._expand_to_5d(list(volume_shape), axes_order, None)
        expanded_shard = writer._expand_shard_shape(list(shard_shape), axes_order if axes_order else ['z', 'y', 'x'])
        total_shards = int(np.prod(np.ceil(np.array(expanded_shape) / np.array(expanded_shard)).astype(int)))
    else:
        # Fallback: pad shard_shape to match volume_shape dimensions
        if len(shard_shape) < len(volume_shape):
            extra_dims = len(volume_shape) - len(shard_shape)
            shard_shape_padded = [1] * extra_dims + list(shard_shape)
        else:
            shard_shape_padded = shard_shape
        total_shards = int(np.prod(np.ceil(np.array(volume_shape) / np.array(shard_shape_padded)).astype(int)))

    return shard_shape, total_shards


def _calculate_memory(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=False):
    """Calculate memory in GB using v1 formula.

    Formula: base (file loading) + shard buffers + task overhead, then x1.5 safety.

    Args:
        use_bioio: If True, applies 3x multiplier for BioIO's higher memory usage
                   due to Dask caching and intermediate data structures.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

    # Base memory for file loading
    if dataset_size_gb < 10:
        base_mem = 2
    elif dataset_size_gb < 100:
        base_mem = min(dataset_size_gb * 0.02, 10)
    else:
        base_mem = min(dataset_size_gb * 0.005, 20)

    # Shard processing buffers (up to 3 concurrent shards, 2x for read+write+compress)
    concurrent_shards = min(total_shards, 3)
    shard_buffer_mem = shard_size_gb * 2.0 * concurrent_shards

    # Task overhead (conversion)
    task_overhead = 4  # GB for Python, TensorStore, etc.

    # Total with 1.5x safety margin, rounded to nearest 5 GB
    total_mem = base_mem + shard_buffer_mem + task_overhead
    recommended = int(math.ceil(total_mem * 1.5 / 5) * 5)

    # BioIO uses ~3x more memory due to Dask caching and intermediate structures
    if use_bioio:
        recommended = int(recommended * 3)

    return max(5, min(recommended, 500))


def _calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=False):
    """Calculate wall time string (H:MM) using v1 formula.

    Formula: (per-shard time x total_shards + overhead) x 2 safety, rounded to 30 min.

    Args:
        use_bioio: If True, applies 10x multiplier for BioIO's slower Dask-based
                   processing compared to native TensorStore readers.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

    # Per-shard time estimate (empirical)
    if shard_size_gb < 0.1:
        minutes_per_shard = 0.5
    elif shard_size_gb < 1.0:
        minutes_per_shard = 2
    else:
        minutes_per_shard = 3

    base_minutes = minutes_per_shard * total_shards

    # Overhead for file loading
    if dataset_size_gb > 1000:
        overhead = 10
    elif dataset_size_gb > 100:
        overhead = 5
    else:
        overhead = 2

    # 2x safety, round to nearest 30 min
    safe_minutes = int(math.ceil((base_minutes + overhead) * 2 / 30) * 30)

    # BioIO is ~10-50x slower due to Dask overhead, apply 10x multiplier
    if use_bioio:
        safe_minutes = int(safe_minutes * 10)

    # Cap at 96 hours (4 days)
    safe_minutes = max(30, min(safe_minutes, 96 * 60))
    hours = safe_minutes // 60
    minutes = safe_minutes % 60
    return f"{hours}:{minutes:02d}"


def submit_job(args, return_job_id=False):
    """Submit a single LSF bsub job that re-invokes tensorswitch_v2.

    Constructs a bsub command with the same conversion arguments (minus
    --submit and LSF-only flags) so the job runs conversion on a cluster node.

    Args:
        args: Parsed command-line arguments
        return_job_id: If True, return the job ID instead of None

    Returns:
        Job ID string if return_job_id=True and submission succeeded, else None
    """
    if not args.project:
        raise ValueError(
            "Missing required argument: --project/-P\n"
            "When using --submit, you must specify an LSF project for billing.\n"
            "Example: pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr --submit -P scicompsoft\n"
            "Common projects: scicompsoft, liconn, ahrens"
        )

    # Auto-calculate resources from source data when not explicitly provided
    memory_gb = args.memory
    wall_time = args.wall_time

    if memory_gb is None or wall_time is None:
        print("Reading input metadata for resource estimation...")
        volume_shape, dtype_str, axes_order = _get_input_metadata(args)
        shard_shape, total_shards = _estimate_shard_info(args, volume_shape, dtype_str, axes_order)
        # Both BIOIO and BioFormats use Dask and have similar overhead
        use_bioio = getattr(args, 'use_bioio', False) or getattr(args, 'use_bioformats', False)
        print(f"  Shape: {volume_shape}, dtype: {dtype_str}")
        print(f"  Shard shape: {shard_shape}, total shards: {total_shards}")
        if use_bioio:
            mode_name = "Bio-Formats" if getattr(args, 'use_bioformats', False) else "BioIO"
            print(f"  {mode_name} mode: applying 10x wall time and 3x memory multipliers")

        if memory_gb is None:
            memory_gb = _calculate_memory(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=use_bioio)
        if wall_time is None:
            wall_time = _calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=use_bioio)

    # Cores: based on memory but capped at 8 (I/O-bound, more cores don't help much)
    cores = args.cores if args.cores is not None else min(8, max(1, int(math.ceil(memory_gb / 15)) * 2))

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    # Job name: tsv2_{src_ext}_to_{out_format}_{input_stem}
    input_name = os.path.basename(args.input)
    input_stem = os.path.splitext(input_name)[0]
    input_ext = os.path.splitext(input_name)[1].lstrip(".")
    job_name = f"tsv2_{input_ext}_to_{args.output_format}_{input_stem}"
    job_name = job_name.replace(" ", "_")[:128]

    # Log directory next to output
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
    error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

    # Build re-invocation command (conversion args only, no LSF-only args)
    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--input", args.input,
        "--output", args.output,
        "--output_format", args.output_format,
    ]
    if args.dataset_path:
        reinvoke += ["--dataset_path", args.dataset_path]
    if args.level_path != "s0":
        reinvoke += ["--level_path", args.level_path]
    if args.chunk_shape:
        reinvoke += ["--chunk_shape", args.chunk_shape]
    if args.shard_shape:
        reinvoke += ["--shard_shape", args.shard_shape]
    if args.no_sharding:
        reinvoke.append("--no_sharding")
    if args.compression != "zstd":
        reinvoke += ["--compression", args.compression]
    if args.compression_level != 5:
        reinvoke += ["--compression_level", str(args.compression_level)]
    if args.start_idx is not None:
        reinvoke += ["--start_idx", str(args.start_idx)]
    if args.stop_idx is not None:
        reinvoke += ["--stop_idx", str(args.stop_idx)]
    if args.write_metadata:
        reinvoke.append("--write_metadata")
    if args.view_index is not None:
        reinvoke += ["--view_index", str(args.view_index)]
    if args.quiet:
        reinvoke.append("--quiet")
    if getattr(args, 'use_bioio', False):
        reinvoke.append("--use_bioio")
    if getattr(args, 'use_bioformats', False):
        reinvoke.append("--use_bioformats")
    if getattr(args, 'force_c_order', False):
        reinvoke.append("--force_c_order")
    if getattr(args, 'force_f_order', False):
        reinvoke.append("--force_f_order")
    if getattr(args, 'voxel_size', None):
        reinvoke += ["--voxel_size", args.voxel_size]
    if getattr(args, 'is_label', False):
        reinvoke.append("--is-label")
    if getattr(args, 'expand_to_5d', False):
        reinvoke.append("--expand-to-5d")

    # Build bsub command
    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024}]",  # GB to MB for LSF
        "-P", args.project,
        "-g", args.job_group,
        "-o", log_path,
        "-e", error_path,
    ] + reinvoke

    # Print summary and submit
    print("=" * 72)
    print("LSF Single-Job Submission")
    print("=" * 72)
    print(f"  Job name:  {job_name}")
    print(f"  Cores:     {cores}")
    print(f"  Memory:    {memory_gb} GB")
    print(f"  Wall time: {wall_time}")
    print(f"  Project:   {args.project}")
    print(f"  Job group: {args.job_group}")
    print(f"  Log:       {log_path}")
    print(f"  Error:     {error_path}")
    print(f"  Command:   {' '.join(reinvoke)}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if not return_job_id:
            print("Job submitted successfully.")
            print(result.stdout.strip())
        # Extract job ID from bsub output: "Job <12345> is submitted..."
        import re
        match = re.search(r'Job <(\d+)>', result.stdout)
        job_id = match.group(1) if match else None
        if return_job_id:
            return job_id
    else:
        print(f"Job submission failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"bsub failed with exit code {result.returncode}")

    return None


def submit_downsample_job(args, cumulative_factors):
    """Submit a single LSF bsub job for downsampling.

    Constructs a bsub command for single-level downsampling from s0.
    """
    if not args.project:
        raise ValueError(
            "Missing required argument: --project/-P\n"
            "When using --submit, you must specify an LSF project for billing.\n"
            "Example: pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr --submit -P scicompsoft\n"
            "Common projects: scicompsoft, liconn, ahrens"
        )

    memory_gb = args.memory or 32
    wall_time = args.wall_time or "2:00"
    # Cores: based on memory but capped at 8 (I/O-bound, more cores don't help much)
    cores = args.cores or min(8, max(1, int(math.ceil(memory_gb / 15)) * 2))

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    # Job name: tsv2_ds_s{level}_{basename}
    input_name = os.path.basename(os.path.dirname(args.input))  # e.g., dataset.zarr
    job_name = f"tsv2_ds_s{args.target_level}_{input_name}"
    job_name = job_name.replace(" ", "_")[:128]

    # Log directory next to output
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
    error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

    # Build re-invocation command
    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--downsample",
        "--input", args.input,
        "--output", args.output,
        "--target_level", str(args.target_level),
        "--single_level_factor", ",".join(map(str, cumulative_factors)),
        "--use_shard", str(args.use_shard),
    ]
    if args.chunk_shape:
        reinvoke += ["--chunk_shape", args.chunk_shape]
    if args.shard_shape:
        reinvoke += ["--shard_shape", args.shard_shape]
    if args.start_idx is not None:
        reinvoke += ["--start_idx", str(args.start_idx)]
    if args.stop_idx is not None:
        reinvoke += ["--stop_idx", str(args.stop_idx)]
    # Resolve 'auto' to actual method before passing to worker
    resolved_method = resolve_downsample_method(args.downsample_method, args.input)
    if resolved_method != "mean":  # mean is the new default for intensity data
        reinvoke += ["--downsample_method", resolved_method]
    if args.quiet:
        reinvoke.append("--quiet")

    # Build bsub command
    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024}]",  # GB to MB for LSF
        "-P", args.project,
        "-g", args.job_group,
        "-o", log_path,
        "-e", error_path,
    ] + reinvoke

    # Print summary and submit
    print("=" * 72)
    print("LSF Downsample Job Submission")
    print("=" * 72)
    print(f"  Job name:    {job_name}")
    print(f"  Target:      s{args.target_level}")
    print(f"  Factor:      {cumulative_factors} (cumulative from s0)")
    print(f"  Cores:       {cores}")
    print(f"  Memory:      {memory_gb} GB")
    print(f"  Wall time:   {wall_time}")
    print(f"  Project:     {args.project}")
    print(f"  Log:         {log_path}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Job submitted successfully.")
        print(result.stdout.strip())
    else:
        print(f"Job submission failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"bsub failed with exit code {result.returncode}")


def show_conversion_spec(reader, writer, args, chunk_shape, shard_shape):
    """Print input/output specs and conversion summary without converting.

    Args:
        reader: Configured reader instance
        writer: Configured writer instance
        args: Parsed CLI arguments
        chunk_shape: Parsed chunk shape tuple or None
        shard_shape: Parsed shard shape tuple or None
    """
    import json

    print("\n" + "=" * 72)
    print("TENSORSWITCH V2 - CONVERSION SPEC PREVIEW")
    print("=" * 72)

    # Input spec
    print("\n--- INPUT ---")
    print(f"Path: {args.input}")
    print(f"Reader: {type(reader).__name__}")

    metadata = reader.get_metadata()
    print(f"Shape: {metadata.get('shape')}")
    print(f"Dtype: {metadata.get('dtype')}")

    voxel_sizes = reader.get_voxel_sizes()
    if voxel_sizes:
        print(f"Voxel sizes: {voxel_sizes}")

    # Try to get TensorStore spec (may not be available for all readers)
    try:
        input_spec = reader.get_tensorstore_spec()
        print(f"\nTensorStore spec:")
        # Simplify for display
        display_spec = {
            "driver": input_spec.get("driver"),
            "dtype": input_spec.get("dtype"),
        }
        if "kvstore" in input_spec:
            kvstore = input_spec["kvstore"]
            if isinstance(kvstore, dict):
                display_spec["kvstore"] = kvstore.get("driver", kvstore)
        print(json.dumps(display_spec, indent=2, default=str))
    except Exception as e:
        print(f"(TensorStore spec not available: {e})")

    # Output spec
    print("\n--- OUTPUT ---")
    print(f"Path: {args.output}")
    print(f"Format: {args.output_format}")
    print(f"Writer: {type(writer).__name__}")

    input_shape = metadata.get('shape', ())

    # Show 5D expansion for Zarr
    if args.output_format in ['zarr3', 'zarr2']:
        # Zarr writers expand to 5D TCZYX
        ndim = len(input_shape)
        if ndim == 3:
            expanded_shape = (1, 1) + tuple(input_shape)  # Add T, C
            print(f"Shape: {input_shape} → {expanded_shape} (5D TCZYX expansion)")
        elif ndim == 4:
            expanded_shape = (1,) + tuple(input_shape)  # Add T
            print(f"Shape: {input_shape} → {expanded_shape} (5D TCZYX expansion)")
        else:
            print(f"Shape: {input_shape}")
    else:
        print(f"Shape: {input_shape}")

    # Chunk/shard info
    effective_chunk = chunk_shape or (128, 128, 128)
    print(f"Chunk shape: {effective_chunk}")

    if args.output_format == 'zarr3' and shard_shape:
        print(f"Shard shape: {shard_shape}")
    elif args.output_format == 'zarr3':
        print(f"Shard shape: (auto-calculated)")

    print(f"Compression: {args.compression or 'zstd'} (level {args.compression_level or 5})")

    # Conversion summary
    print("\n--- CONVERSION SUMMARY ---")

    # Calculate total chunks
    from .utils import get_total_chunks_from_store
    try:
        # Estimate chunks based on shape and chunk size
        shape = input_shape
        chunks = effective_chunk
        total_chunks = 1
        for i, (s, c) in enumerate(zip(shape, chunks)):
            total_chunks *= (s + c - 1) // c
        print(f"Estimated chunks: {total_chunks:,}")
    except Exception:
        print(f"Estimated chunks: (calculation not available)")

    # Estimate output size
    dtype_sizes = {'uint8': 1, 'uint16': 2, 'uint32': 4, 'float32': 4, 'float64': 8}
    dtype = str(metadata.get('dtype', 'uint16'))
    dtype_size = dtype_sizes.get(dtype, 2)
    raw_size = np.prod(input_shape) * dtype_size
    # Assume ~50% compression ratio for zstd
    estimated_size = raw_size * 0.5

    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    print(f"Raw data size: {format_size(raw_size)}")
    print(f"Estimated output: ~{format_size(estimated_size)} (assuming ~50% compression)")

    print("\n" + "=" * 72)
    print("(No conversion performed - remove --show_spec to convert)")
    print("=" * 72 + "\n")


def main(argv=None):
    """Run single-process conversion, downsampling, or submit LSF job."""
    args = parse_args(argv)

    # Apply preset configurations
    if args.preset == "webknossos":
        # WebKnossos preset: zarr3, chunk 32x32x32, shard 1024x1024x1024
        if args.output_format == "zarr3":  # Only override if using default
            pass  # Already zarr3
        if not args.chunk_shape:
            args.chunk_shape = "32,32,32"
        if not args.shard_shape:
            args.shard_shape = "1024,1024,1024"
        if not args.quiet:
            print("Using WebKnossos preset: chunk=32x32x32, shard=1024x1024x1024, zarr3")

    # Parse optional shapes
    chunk_shape = parse_shape(args.chunk_shape, "chunk_shape") if args.chunk_shape else None
    shard_shape = parse_shape(args.shard_shape, "shard_shape") if args.shard_shape else None
    verbose = not args.quiet
    use_shard = bool(args.use_shard)
    skip_existing = args.skip_existing and not args.no_skip_existing

    # Handle batch worker mode (LSF job array worker)
    if args.batch_worker:
        if not args.index_file:
            raise ValueError("--index_file is required with --batch_worker")

        from .core.batch import read_index_file

        # Get job array index from environment
        job_index = int(os.environ.get('LSB_JOBINDEX', '1'))

        # Read input/output paths from index file
        input_path, output_path = read_index_file(args.index_file, job_index)

        print(f"Batch worker: job index {job_index}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        # Override args with paths from index file
        args.input = input_path
        args.output = output_path

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run single-file conversion (fall through to standard conversion below)
        # Skip batch/status modes
        args.status = False
    else:
        # Validate required arguments for non-batch_worker mode
        if not args.input:
            raise ValueError(
                "Missing required argument: --input/-i\n"
                "Provide the input file or directory path.\n"
                "Example: pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr"
            )

        # Check if this is batch pyramid mode (auto_multiscale + directory of zarrs)
        # In this mode, -o is not required because pyramids are written into each zarr
        is_batch_pyramid_mode = False
        if args.auto_multiscale and args.input:
            input_path = args.input.rstrip('/')
            # Batch mode: directory containing multiple .zarr subdirectories
            # NOT batch mode: path ends with .zarr/.n5, or has s0/.zarray inside (single dataset)
            if os.path.isdir(input_path) and not input_path.endswith('.zarr') and not input_path.endswith('.n5'):
                has_s0 = os.path.exists(os.path.join(input_path, 's0'))
                has_zarray = os.path.exists(os.path.join(input_path, '.zarray'))
                zarr_subdirs = [d for d in os.listdir(input_path) if d.endswith('.zarr') and os.path.isdir(os.path.join(input_path, d))]
                is_batch_pyramid_mode = len(zarr_subdirs) > 0 and not has_s0 and not has_zarray

        if not args.output and not is_batch_pyramid_mode:
            raise ValueError(
                "Missing required argument: --output/-o\n"
                "Provide the output path.\n"
                "Example: pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr"
            )

        # Validate paths exist and are accessible
        validate_input_path(args.input, allow_directory=True)
        if args.output:
            validate_output_path(args.output)

    # Handle status check mode
    if args.status:
        from .core.batch import BatchConverter, detect_input_mode

        batch = BatchConverter(
            input_dir=args.input,
            output_dir=args.output,
            pattern=args.pattern,
            output_format=args.output_format,
            recursive=args.recursive,
        )
        batch.check_status()
        return

    # Detect batch mode (input is directory, not file)
    if not args.batch_worker and not args.downsample and not args.auto_multiscale:
        from .core.batch import detect_input_mode, BatchConverter

        input_mode = detect_input_mode(args.input)

        if input_mode == 'batch_directory':
            # Batch mode: process multiple files
            batch = BatchConverter(
                input_dir=args.input,
                output_dir=args.output,
                pattern=args.pattern,
                output_format=args.output_format,
                recursive=args.recursive,
            )

            # Discover files
            files = batch.discover()
            if not files:
                print(f"No files matching '{args.pattern}' found in {args.input}")
                return

            print(f"Discovered {len(files)} files matching '{args.pattern}'")

            if args.submit:
                # Submit as LSF job array
                if not args.project:
                    raise ValueError("--project is required with --submit")

                result = batch.submit_lsf(
                    project=args.project,
                    chunk_shape=args.chunk_shape,
                    shard_shape=args.shard_shape,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    memory_gb=args.memory or 30,
                    wall_time=args.wall_time or "1:00",
                    cores=args.cores or 2,
                    max_concurrent=args.max_concurrent,
                    job_group=args.job_group,
                    skip_existing=skip_existing,
                    dry_run=args.dry_run,
                )
            else:
                # Run locally (sequential)
                result = batch.run_local(
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    skip_existing=skip_existing,
                    verbose=verbose,
                )

            return

    # Handle auto_multiscale mode (full pyramid generation)
    # This is the main entry point for automatic downsampling - just point at s0 and it does the rest
    if args.auto_multiscale:
        from .core.pyramid import PyramidPlanner, create_pyramid_parallel
        from .utils import update_ome_metadata_if_needed
        import glob as glob_module

        # Parse custom per-level factors if provided
        custom_per_level_factors = None
        if args.per_level_factors:
            custom_per_level_factors = parse_per_level_factors(args.per_level_factors)
            print(f"Using custom per-level factors: {custom_per_level_factors}")

        # Detect batch mode: directory containing multiple .zarr subdirectories
        # NOT batch mode: path ends with .zarr/.n5, or has s0/.zarray inside (single dataset)
        input_path = args.input.rstrip('/')
        is_batch_mode = False
        if os.path.isdir(input_path) and not input_path.endswith('.zarr') and not input_path.endswith('.n5'):
            has_s0 = os.path.exists(os.path.join(input_path, 's0'))
            has_zarray = os.path.exists(os.path.join(input_path, '.zarray'))
            zarr_subdirs = [d for d in os.listdir(input_path) if d.endswith('.zarr') and os.path.isdir(os.path.join(input_path, d))]
            is_batch_mode = len(zarr_subdirs) > 0 and not has_s0 and not has_zarray

        # Batch downsampling mode: process multiple datasets
        if is_batch_mode and args.submit:
            if not args.project:
                raise ValueError("--project is required with --submit")

            # Find all matching datasets
            pattern = args.pattern if args.pattern != "*.tif" else "*.zarr"
            search_pattern = os.path.join(input_path, pattern)

            # Handle patterns like "*.zarr" or "setup*/timepoint0"
            matches = sorted(glob_module.glob(search_pattern))

            if not matches:
                print(f"No datasets found matching pattern: {search_pattern}")
                return

            # For each match, find s0 path using auto-detection
            datasets = []
            for match in matches:
                try:
                    s0_path, _ = find_base_level(match, verbose=False)
                    datasets.append(s0_path)
                except ValueError as e:
                    print(f"Warning: Skipping {match} - {e}")
                    continue

            if not datasets:
                print("No valid datasets with s0 found.")
                return

            print(f"\n{'='*60}")
            print(f"BATCH PYRAMID GENERATION")
            print(f"{'='*60}")
            print(f"Found {len(datasets)} datasets to process")
            print(f"Max concurrent coordinator jobs: {args.max_concurrent}")
            if custom_per_level_factors:
                print(f"Custom per-level factors: {custom_per_level_factors}")
                print(f"Number of levels: {len(custom_per_level_factors)}")
            print()

            # Submit coordinator job for each dataset
            all_coordinator_jobs = []
            skipped_count = 0
            for i, s0_path in enumerate(datasets):
                root_path = os.path.dirname(s0_path)
                dataset_name = os.path.basename(root_path)

                # Skip if s1 already exists (pyramid already generated)
                s1_path = os.path.join(root_path, 's1')
                if skip_existing and os.path.exists(s1_path):
                    skipped_count += 1
                    continue

                print(f"[{i+1-skipped_count}/{len(datasets)-skipped_count}] Submitting pyramid for: {dataset_name}")

                try:
                    result = create_pyramid_parallel(
                        s0_path=s0_path,
                        project=args.project,
                        memory=args.memory,
                        wall_time=args.wall_time,
                        cores=args.cores,
                        use_shard=use_shard,
                        downsample_method=args.downsample_method,
                        dry_run=False,
                        verbose=False,  # Reduce output in batch mode
                        custom_per_level_factors=custom_per_level_factors,
                    )
                    coordinator_jid = result.get('coordinator_job_id')
                    num_levels = result.get('pyramid_plan', {}).get('num_levels', 0)
                    if coordinator_jid:
                        all_coordinator_jobs.append(coordinator_jid)
                        print(f"  → Coordinator job {coordinator_jid} ({num_levels} levels)")
                except Exception as e:
                    print(f"  → Error: {e}")

                # Respect max_concurrent by waiting if we've hit the limit
                if len(all_coordinator_jobs) >= args.max_concurrent and i < len(datasets) - 1:
                    # Wait for oldest coordinator to finish before submitting more
                    oldest_job = all_coordinator_jobs.pop(0)
                    print(f"  (Waiting for job {oldest_job} to complete before continuing...)")
                    os.system(f"bwait -w 'ended({oldest_job})' > /dev/null 2>&1")

            print(f"\n{'='*60}")
            print(f"BATCH SUBMISSION COMPLETE")
            print(f"Submitted {len(all_coordinator_jobs)} coordinator jobs")
            if skipped_count > 0:
                print(f"Skipped {skipped_count} datasets (s1 already exists)")
            print(f"{'='*60}")
            return

        # Single dataset mode (original behavior)
        # Auto-detect s0 path from input (supports root path or explicit s0 path)
        s0_path, root_path = find_base_level(args.input, verbose=verbose)

        if args.submit:
            if not args.project:
                raise ValueError("--project is required with --submit")

            # Use create_pyramid_parallel which handles everything automatically
            # Pass None for memory/wall_time/cores to enable auto-calculation per level
            result = create_pyramid_parallel(
                s0_path=s0_path,
                project=args.project,
                memory=args.memory,  # None = auto-calculate per level
                wall_time=args.wall_time,  # None = auto-calculate per level
                cores=args.cores,  # None = auto-calculate based on memory
                use_shard=use_shard,
                downsample_method=args.downsample_method,
                dry_run=False,
                verbose=verbose,
                custom_per_level_factors=custom_per_level_factors,
            )
            num_levels = result.get('pyramid_plan', {}).get('num_levels', 0)
            job_ids = result.get('job_ids', [])
            if num_levels > 0:
                print(f"\nSubmitted {len(job_ids)} jobs for {num_levels} levels")
                print("Jobs will run in parallel - all levels downsample directly from s0.")
                print("After all jobs complete, root metadata will be updated automatically.")
            else:
                print("\nNo pyramid levels needed - dataset is already at minimum size.")
        else:
            # Local mode: run downsampling directly for each level
            from .core.downsampler import downsample_level

            planner = PyramidPlanner(s0_path)
            plan = planner.calculate_pyramid_plan(custom_per_level_factors=custom_per_level_factors)
            planner.print_pyramid_plan(plan)

            print("\nRunning local pyramid generation...")

            # Pre-create all levels first
            planner.precreate_all_levels(plan, use_shard=use_shard, verbose=verbose)

            # Process each level
            for level_info in plan['levels']:
                level = level_info['level']
                cumulative_factors = level_info['cumulative_factor']

                print(f"\n--- Downsampling s{level} (cumulative factor: {cumulative_factors}) ---")

                downsample_level(
                    s0_path=s0_path,
                    output_path=root_path,
                    target_level=level,
                    cumulative_factors=cumulative_factors,
                    use_shard=use_shard,
                    custom_shard_shape=level_info.get('shard_shape'),
                    custom_chunk_shape=level_info.get('chunk_shape'),
                    verbose=verbose,
                )

            # Update root metadata
            print("\nUpdating root metadata...")
            update_ome_metadata_if_needed(root_path, use_ome_structure=True)

            print(f"\n{'='*60}")
            print(f"AUTO-MULTISCALE COMPLETE: {root_path}")
            print(f"Generated s0 through s{plan['num_levels']}")
            print(f"{'='*60}")
        return

    # Handle downsample mode (single level from s0)
    if args.downsample:
        from .core.downsampler import downsample_level

        if args.target_level is None:
            raise ValueError("--target_level is required with --downsample")

        # Support both --single_level_factor (new) and --cumulative_factors (legacy alias)
        factor_str = args.single_level_factor or args.cumulative_factors
        if factor_str is None:
            raise ValueError("--single_level_factor is required with --downsample")

        cumulative_factors = [int(x) for x in factor_str.split(",")]

        if args.submit:
            # Submit as LSF job
            submit_downsample_job(args, cumulative_factors)
        else:
            # Run locally - resolve 'auto' method first
            resolved_method = resolve_downsample_method(args.downsample_method, args.input)
            downsample_level(
                s0_path=args.input,
                output_path=args.output,
                target_level=args.target_level,
                cumulative_factors=cumulative_factors,
                start_idx=args.start_idx or 0,
                stop_idx=args.stop_idx,
                use_shard=use_shard,
                custom_shard_shape=list(shard_shape) if shard_shape else None,
                custom_chunk_shape=list(chunk_shape) if chunk_shape else None,
                downsample_method=resolved_method,
                verbose=verbose,
            )
        return

    # Standard conversion mode (single level / s0)
    if args.submit:
        submit_job(args)
        return

    reader = create_reader(args)
    writer = create_writer(args)

    # Handle --show_spec: print specs and exit without converting
    if args.show_spec:
        show_conversion_spec(reader, writer, args, chunk_shape, shard_shape)
        return

    from .core.converter import DistributedConverter

    converter = DistributedConverter(reader, writer)

    # Determine force_order from CLI args (None = auto-detect)
    force_order = None
    if args.force_c_order:
        force_order = 'c'
    elif args.force_f_order:
        force_order = 'f'

    # Auto-detect segmentation data based on dtype, or use explicit --is-label flag
    is_label = getattr(args, 'is_label', False)
    if not is_label:
        # Auto-detect based on dtype
        try:
            spec = reader.get_tensorstore_spec()
            if spec.get('driver') == 'array' and 'array' in spec:
                dtype_str = str(spec['array'].dtype)
            else:
                import tensorstore as ts
                from .utils import get_tensorstore_context
                spec['context'] = get_tensorstore_context()
                store = ts.open(spec, read=True).result()
                dtype_str = store.dtype.name

            from .utils.metadata_utils import is_segmentation_dtype
            if is_segmentation_dtype(dtype_str):
                is_label = True
                if verbose:
                    print(f"Auto-detected segmentation data (dtype: {dtype_str}), adding image-label metadata")
        except Exception:
            pass  # If detection fails, default to is_label=False

    # Parse voxel_size override if provided
    voxel_size_override = None
    if args.voxel_size:
        parts = args.voxel_size.split(',')
        if len(parts) == 3:
            voxel_size_override = {
                'x': float(parts[0]),
                'y': float(parts[1]),
                'z': float(parts[2]),
            }
        else:
            raise ValueError(
                f"Invalid --voxel_size: '{args.voxel_size}'\n"
                f"Expected comma-separated X,Y,Z values (e.g., '0.16,0.16,0.4')"
            )

    # Get expand_to_5d flag (default False = preserve source layout)
    expand_to_5d = getattr(args, 'expand_to_5d', False)

    if args.start_idx is not None:
        # Manual chunk-range mode (for bsub workers)
        converter.convert(
            start_idx=args.start_idx,
            stop_idx=args.stop_idx,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            write_metadata=args.write_metadata,
            delete_existing=False,
            verbose=verbose,
            force_order=force_order,
            voxel_size_override=voxel_size_override,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
        )
    else:
        # Full single-process conversion
        converter.convert(
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            write_metadata=True,
            verbose=verbose,
            force_order=force_order,
            voxel_size_override=voxel_size_override,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
        )


if __name__ == "__main__":
    main()
