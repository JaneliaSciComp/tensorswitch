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
import shlex
import math
import argparse
import shutil
from typing import Optional, Tuple

import numpy as np

# Force line-buffered stdout/stderr so progress output appears immediately
# (Python fully buffers stdout when not connected to a terminal, e.g. pixi run, bsub)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

os.umask(0o0002)  # Team permissions: rwxrwxr-x

__version__ = "2.0.0-beta"

from .utils.pyramid_utils import resolve_downsample_method
from .utils import get_dtype_name


def _resolve_conversion_subgroup(args) -> Optional[str]:
    """Return the subgroup the conversion writes to, or None for flat/ambiguous.

    When ``--auto_multiscale`` chains a pyramid after conversion, we must tell
    the pyramid step which subgroup inside the container holds the freshly
    written s0. Otherwise ``find_base_level(<container_root>)`` reads the
    container's root OME metadata and may return a stale subgroup (e.g. the
    ``raw/`` multiscales written by a prior stage), causing the pyramid to
    overwrite the wrong levels or skip writing new ones.

    Args:
        args: Parsed CLI namespace.

    Returns:
        - ``"labels/<label_key>"`` when conversion writes labels.
        - ``"<image_key>"`` when conversion writes images.
        - ``None`` when the subgroup cannot be determined from args alone
          (``--no-nested-structure``, non-nested formats like N5, or
          ``--data-type auto`` without ``--is_label``). The caller then falls
          back to legacy container-root behavior.
    """
    # Only zarr2/zarr3 use nested OME-NGFF structure; N5 is always flat.
    fmt = getattr(args, 'output_format', 'zarr3')
    if fmt not in ('zarr2', 'zarr3'):
        return None
    if not getattr(args, 'use_nested_structure', True):
        return None

    data_type = getattr(args, 'data_type', 'auto')
    label_key = getattr(args, 'label_key', 'segmentation')
    image_key = getattr(args, 'image_key', 'raw')

    if getattr(args, 'is_label', False) or data_type == 'labels':
        return f'labels/{label_key}'
    if data_type == 'image':
        return image_key
    # 'auto' without is_label: converter picks subgroup from dtype at runtime;
    # we cannot know here. Preserve legacy container-root behavior.
    return None


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

    # Strategy 2: Check nested group structures (raw/, labels/segmentation/)
    if not base_level_path:
        for nested_group in ['raw', 'labels/segmentation']:
            nested_path = os.path.join(root_path, nested_group)
            nested_meta = os.path.join(nested_path, 'zarr.json')
            if os.path.exists(nested_meta):
                try:
                    with open(nested_meta, 'r') as f:
                        nested_metadata = json.load(f)
                    attrs = nested_metadata.get('attributes', {})
                    ome_attrs = attrs.get('ome', attrs)
                    multiscales = ome_attrs.get('multiscales', [])
                    if multiscales and multiscales[0].get('datasets'):
                        first_ds = multiscales[0]['datasets'][0].get('path')
                        base_level_path = os.path.join(nested_group, first_ds)
                        if verbose:
                            print(f"Found base level from nested {nested_group}/zarr.json metadata: {base_level_path}")
                        break
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    # Strategy 3: Fallback to common subdirectory patterns
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


def _is_pyramid_only_intent(input_path: Optional[str],
                             output_path: Optional[str]) -> bool:
    """True when ``--auto_multiscale -i X -o Y`` means "build pyramid on X".

    Used to route ``--auto_multiscale`` to the pyramid-only branch even when
    ``-o`` is supplied — the pattern the README documents
    (``-i dataset.zarr/s0 -o dataset.zarr``).

    Heuristic:
      1. Input must be an existing directory.
      2. Input is a "level" (basename ``s0``/``0``/``s1``/... with zarr.json,
         .zarray, or attributes.json inside) — covers zarr3, zarr2, n5.
      3. OR input is a "container" (has ``s0``/``0``/``raw``/
         ``labels/segmentation`` child) — covers the 3 formats plus OME-NGFF
         nested structure.
      4. If neither → False (raw source files, TIFF dirs, …) so the
         convert+pyramid workflow for raw sources is untouched.
      5. If ``-o`` is absent → True (preserves existing pyramid-only behavior).
      6. If ``-o`` is given → True only when ``-o`` is ancestor-or-equal of
         ``-i``. This keeps cross-format conversion
         (``-i old.zarr -o different.zarr --auto_multiscale``) routing to
         the convert+pyramid path.
    """
    import re as _re

    if not input_path:
        return False
    p = input_path.rstrip('/\\')
    if not os.path.isdir(p):
        return False

    level_pat = _re.compile(r'^s?\d+$')
    level_meta_files = ('zarr.json', '.zarray', 'attributes.json')
    container_children = (
        's0', '0', 'raw', os.path.join('labels', 'segmentation'),
    )

    is_level = (
        bool(level_pat.match(os.path.basename(p)))
        and any(os.path.exists(os.path.join(p, f)) for f in level_meta_files)
    )
    is_container = any(
        os.path.exists(os.path.join(p, sub)) for sub in container_children
    )

    if not (is_level or is_container):
        return False

    if not output_path:
        return True

    in_abs = os.path.normpath(os.path.abspath(p))
    out_abs = os.path.normpath(os.path.abspath(output_path.rstrip('/\\')))
    try:
        return os.path.commonpath([in_abs, out_abs]) == out_abs
    except ValueError:
        # Different drives on Windows → not the same container
        return False


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
        choices=["webknossos", "paintera"],
        help="Use preset configuration. "
             "'webknossos': zarr3, chunk 32x32x32, shard 1024x1024x1024, zstd. "
             "'paintera': n5, xyz axis order, gzip, chunk 64x64x64 "
             "(or zarr2 with zyx if --output_format zarr2).",
    )

    # Dataset paths
    parser.add_argument(
        "--dataset_path", default="",
        help="Path within input container (e.g., 's0' for N5)",
    )
    parser.add_argument(
        "--level_path", default="s0",
        help="Level subdirectory name in output (default: s0 for Janelia convention)",
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
    parser.add_argument(
        "--dtype", default=None,
        help="Output dtype override (e.g., uint8, int16, uint16, float32). "
             "Default: preserve source dtype.",
    )

    # Voxel size override
    parser.add_argument(
        "--voxel_size", default=None,
        help="Override voxel size, comma-separated X,Y,Z (e.g., '9,9,12'). "
             "Values are in nanometers by default, or in the unit specified by --voxel_unit. "
             "Use when source file lacks embedded voxel size metadata.",
    )
    parser.add_argument(
        "--voxel_unit", default=None,
        choices=["nanometer", "micrometer", "millimeter"],
        help="Override the spatial unit in OME metadata. "
             "When set, voxel sizes are written as-is in this unit (no nm conversion). "
             "Without --voxel_size, defaults to scale [1,1,1] in the specified unit.",
    )

    # Label/segmentation mode
    parser.add_argument(
        "--is-label", "--is_label", action="store_true",
        dest="is_label",
        help="Mark output as label/segmentation data. Adds OME-NGFF image-label metadata. "
             "Auto-detected for uint64/uint32 data types, use this flag to force for other types.",
    )

    # OME-NGFF nested structure options
    parser.add_argument(
        "--data-type", choices=["image", "labels", "auto"],
        default="auto",
        help="Data type for output structure. 'auto' detects based on dtype/encoding. "
             "'image' writes to raw/ subdirectory, 'labels' writes to labels/segmentation/.",
    )
    parser.add_argument(
        "--use-nested-structure", action="store_true", default=True,
        dest="use_nested_structure",
        help="Use OME-NGFF nested structure (raw/, labels/). Default: True for zarr3.",
    )
    parser.add_argument(
        "--no-nested-structure", action="store_false",
        dest="use_nested_structure",
        help="Disable nested structure (write directly to output path).",
    )
    parser.add_argument(
        "--add-to-existing", action="store_true",
        dest="add_to_existing",
        help="Add data to existing container without destroying it. "
             "Safe write applies to the subgroup (e.g., labels/) not the container root.",
    )
    parser.add_argument(
        "--image-only", action="store_true",
        help="Only convert image data (skip labels) when both are found in source folder.",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help="Only convert labels data (skip image) when both are found in source folder.",
    )
    parser.add_argument(
        "--image-key", type=str, default="raw",
        help="Name for image group in output (default: 'raw').",
    )
    parser.add_argument(
        "--label-key", type=str, default="segmentation",
        help="Name for label image in output (default: 'segmentation').",
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
        "--cumulative_factor_for_metadata", type=str, default=None,
        help=argparse.SUPPRESS,  # Internal: true cumulative factor from s0 for metadata (chained mode)
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

    parser.add_argument(
        "--no-translation", action="store_true",
        help="Disable translation transforms in OME-NGFF multiscale metadata. "
             "By default, translation offsets are included for Neuroglancer compatibility.",
    )

    # Upsampling mode
    parser.add_argument(
        "--upsample", action="store_true",
        help="Upsample mode: resample anisotropic data to isotropic resolution. "
             "Uses scipy.ndimage.zoom (trilinear for images, nearest for labels).",
    )
    parser.add_argument(
        "--target_voxel_size", type=float, default=None,
        help="Target isotropic voxel size in nm for --upsample mode. "
             "Default: auto (use smallest source voxel size = highest resolution axis).",
    )
    parser.add_argument(
        "--upsample_method", type=str, default="auto",
        choices=["auto", "trilinear", "nearest", "cubic"],
        help="Upsampling interpolation method: auto (default, trilinear for images, "
             "nearest for labels), trilinear, nearest, cubic.",
    )

    # Subvolume extraction
    parser.add_argument(
        "--bbox", type=str, default=None,
        help="Bounding box for subvolume extraction: origin_0,origin_1,origin_2,size_0,size_1,size_2 "
             "(in source voxel coordinates, source dimension order). "
             "For Neuroglancer precomputed: x,y,z order. For Zarr/N5: z,y,x order. "
             "Example: --bbox 116316,87591,20800,10240,10240,1024",
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
        "--log_dir", type=str, default=None,
        help="Directory for LSF log files (default: output/ next to the output path)",
    )
    parser.add_argument(
        "--no_ome_meta_export", action="store_true",
        help="Disable writing OME/METADATA.ome.xml (or .czi.xml) file",
    )
    parser.add_argument(
        "--no_ome_xml_attr", action="store_true",
        help="Do not embed OME/CZI XML in zarr.json/.zattrs (keeps standalone XML file)",
    )
    parser.add_argument(
        "--use_bioio", action="store_true",
        help="Force BIOIO adapter (Tier 3) instead of auto-detected Tier 2 reader",
    )
    parser.add_argument(
        "--use_bioformats", action="store_true",
        help="Force Bio-Formats reader (Tier 4, Java-backed) for 150+ formats. "
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

    # Spatial axis reorder
    parser.add_argument(
        "--axes_order", type=str, default=None,
        help="Override output spatial axis order. Accepts any permutation of x,y,z "
             "(e.g., 'xyz', 'zyx', 'xzy'). Default: preserve source order. "
             "Example: ND2 source is ZYX, --axes_order xyz transposes to XYZ.",
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
        "--omero", action="store_true", default=False,
        help="(deprecated, now default) Include omero channel metadata for visualization tools",
    )
    parser.add_argument(
        "--no-omero", action="store_true", dest="no_omero", default=False,
        help="Disable omero channel metadata generation (enabled by default)",
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


def parse_bbox(bbox_str):
    """Parse bbox string into (origin, size) tuples.

    Args:
        bbox_str: Comma-separated string 'origin_0,origin_1,origin_2,size_0,size_1,size_2'
                  Coordinates are in source voxel coordinates, source dimension order.
                  For Neuroglancer precomputed: x,y,z order.
                  For Zarr/N5: z,y,x order.

    Returns:
        Tuple of (origin, size) where each is a 3-tuple of ints.

    Raises:
        ValueError: If format is invalid.
    """
    values = [int(v.strip()) for v in bbox_str.split(',')]
    if len(values) != 6:
        raise ValueError(
            f"--bbox requires 6 comma-separated integers: origin_0,origin_1,origin_2,size_0,size_1,size_2\n"
            f"Got {len(values)} values: {bbox_str}"
        )
    origin = tuple(values[:3])
    size = tuple(values[3:])
    if any(s <= 0 for s in size):
        raise ValueError(f"--bbox size values must be positive, got: {size}")
    return origin, size


def validate_input_path(path: str, allow_directory: bool = True) -> None:
    """Validate that input path exists and is readable.

    Args:
        path: Input file or directory path (local or remote URL)
        allow_directory: If True, directories are allowed (batch mode)

    Raises:
        FileNotFoundError: If path does not exist
        PermissionError: If path is not readable
        ValueError: If path is a directory but not allowed
    """
    from .readers.base import is_remote_path

    # Skip local filesystem checks for remote URLs — reader handles connectivity
    if is_remote_path(path) or path.startswith('precomputed://'):
        return

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


def _tmp_path_for(output_path: str) -> str:
    """Return the temporary path used during conversion.

    Appends '.tmp' to the output path so partial writes are never
    confused with completed conversions.  Example:
        /nrs/scicompsoft/rokicki/output.zarr  →
        /nrs/scicompsoft/rokicki/output.zarr.tmp
    """
    return output_path.rstrip('/\\') + '.tmp'


def _finalize_tmp_path(tmp_path: str, final_path: str, verbose: bool = True) -> None:
    """Rename a completed .tmp output to its final path.

    If the final path already exists (e.g. leftover from a prior run),
    it is removed first.
    """
    if not os.path.exists(tmp_path):
        return
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(tmp_path, final_path)
    if verbose:
        print(f"Renamed {tmp_path} → {final_path}")


def _finalize_add_to_existing(
    final_output: str,
    subgroup_parent: str,
    output_format: str,
    verbose: bool = True,
) -> None:
    """Rename subgroup .tmp → final and fix root metadata after --add-to-existing.

    During --add-to-existing conversion, data is written to e.g.
    ``labels.tmp/`` inside the existing container.  This function:

    1. Removes the old subgroup if it exists (e.g. old ``labels/``).
    2. Renames the ``.tmp`` variant to the final name.
    3. Patches the root metadata so references to ``labels.tmp`` become
       ``labels`` (labels list + multiscale dataset paths).

    Args:
        final_output: Container root path (e.g. ``/data/out.zarr``).
        subgroup_parent: Top-level subgroup name (``"labels"`` or ``"raw"``).
        output_format: ``"zarr3"`` or ``"zarr2"``.
        verbose: Print progress messages.
    """
    import json as _json

    tmp_name = subgroup_parent + '.tmp'
    tmp_path = os.path.join(final_output, tmp_name)
    final_path = os.path.join(final_output, subgroup_parent)

    if not os.path.exists(tmp_path):
        return

    # 1. Remove old subgroup (e.g. old labels/)
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(tmp_path, final_path)
    if verbose:
        print(f"Renamed {tmp_path} → {final_path}")

    # 2. Patch root metadata: labels.tmp → labels
    if output_format == 'zarr3':
        meta_path = os.path.join(final_output, 'zarr.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            ome = meta.get('attributes', {}).get('ome', {})
            # Fix labels list
            labels_list = ome.get('labels', [])
            labels_list = [subgroup_parent if l == tmp_name else l for l in labels_list]
            if labels_list:
                ome['labels'] = labels_list
            # Fix multiscale dataset paths
            for ms in ome.get('multiscales', []):
                for ds in ms.get('datasets', []):
                    path = ds.get('path', '')
                    if path.startswith(tmp_name + '/'):
                        ds['path'] = subgroup_parent + path[len(tmp_name):]
            meta['attributes']['ome'] = ome
            with open(meta_path, 'w') as f:
                _json.dump(meta, f, indent=2)
    elif output_format == 'zarr2':
        meta_path = os.path.join(final_output, '.zattrs')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            labels_list = meta.get('labels', [])
            labels_list = [subgroup_parent if l == tmp_name else l for l in labels_list]
            if labels_list:
                meta['labels'] = labels_list
            for ms in meta.get('multiscales', []):
                for ds in ms.get('datasets', []):
                    path = ds.get('path', '')
                    if path.startswith(tmp_name + '/'):
                        ds['path'] = subgroup_parent + path[len(tmp_name):]
            with open(meta_path, 'w') as f:
                _json.dump(meta, f, indent=2)

    if verbose:
        print(f"Updated root metadata: {tmp_name} → {subgroup_parent}")


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

    # Force Bio-Formats reader if requested (Tier 4, Java-backed)
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


def create_writer(args, data_type: str = 'image'):
    """Create a writer from CLI arguments.

    Args:
        args: Parsed command-line arguments
        data_type: 'image' or 'labels' - determines output path structure
    """
    from .api.writers import Writers

    fmt = args.output_format

    # Use level_path as provided (default is "s0" for Janelia convention)
    # All formats (zarr3, zarr2, n5) now use "s0" as default
    level_path = args.level_path

    # Determine if we should use nested structure (enabled by default for zarr3 and zarr2)
    use_nested = getattr(args, 'use_nested_structure', True) and fmt in ["zarr3", "zarr2"]
    image_key = getattr(args, 'image_key', 'raw')
    label_key = getattr(args, 'label_key', 'segmentation')
    labels_container = getattr(args, '_labels_container_override', 'labels')

    include_omero = not getattr(args, 'no_omero', False)
    if fmt == "zarr3":
        return Writers.zarr3(
            output_path=args.output,
            use_sharding=not args.no_sharding,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=level_path,
            include_omero=include_omero,
            use_nested_structure=use_nested,
            data_type=data_type,
            image_key=image_key,
            label_key=label_key,
            labels_container=labels_container,
        )
    elif fmt == "zarr2":
        return Writers.zarr2(
            output_path=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=level_path,
            include_omero=include_omero,
            use_nested_structure=use_nested,
            data_type=data_type,
            image_key=image_key,
            label_key=label_key,
            labels_container=labels_container,
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


def _warn_if_axis_voxel_mismatch(args, axes_order) -> None:
    """Emit a warning when --voxel_size gives 3 values but source axes have
    fewer than 3 spatial dimensions.

    Example: TIFF reports ``CYX`` (1 non-spatial + 2 spatial), but user passes
    ``--voxel_size 5.99,5.99,5.96`` (3 spatial values). Without this warning
    TS silently drops the Z value and the output scale[0] stays at ``1.0``.
    Users can fix their TIFF save pipeline or pass ``--axes_order zyx`` to
    override.
    """
    if not getattr(args, 'voxel_size', None):
        return
    if not axes_order:
        return
    parts = str(args.voxel_size).split(',')
    if len(parts) != 3:
        return
    spatial = [a for a in axes_order if a in ('x', 'y', 'z')]
    if len(spatial) >= 3:
        return

    non_spatial = [a for a in axes_order if a not in ('x', 'y', 'z')]
    import sys as _sys
    print(
        f"WARNING: --voxel_size gave 3 values (X,Y,Z = {parts[0]}, {parts[1]}, {parts[2]}) "
        f"but source reports only {len(spatial)} spatial axes: {axes_order} "
        f"(non-spatial: {non_spatial}). "
        f"The Z value ({parts[2]}) will NOT be applied to dim-0 — output scale[0] "
        f"will default to 1.0. "
        f"If this is actually a Z-stack mis-labeled as channels/samples, "
        f"pass --axes_order zyx to re-interpret dim-0 as Z, "
        f"or fix the source file's axes tag.",
        file=_sys.stderr,
    )


def _get_input_metadata(args):
    """Read input shape, dtype, and axes for resource estimation.

    Returns:
        tuple: (shape, dtype_str, axes_order) where axes_order may be None
    """
    reader = create_reader(args)
    store = reader.get_tensorstore()

    # Get axes from TensorStore domain labels (works for all tiers now)
    axes_order = None
    if hasattr(store, 'domain') and hasattr(store.domain, 'labels'):
        labels = store.domain.labels
        if labels and all(labels):
            # Normalize 'channel' to 'c'
            axes_order = ['c' if l.lower() == 'channel' else l.lower() for l in labels]

    # Catch CYX/IYX/SYX mis-labeling early, before resource estimation silently
    # drops the Z value from --voxel_size.
    _warn_if_axis_voxel_mismatch(args, axes_order)

    dtype_name = get_dtype_name(store.dtype)
    return tuple(store.shape), dtype_name, axes_order


def _estimate_shard_info(args, volume_shape, dtype_str, axes_order=None):
    """Estimate shard/chunk shape and count for resource calculation.

    Thin CLI wrapper around resource_utils.estimate_shard_info() — parses
    args.shard_shape / args.chunk_shape strings, then delegates to the
    shared implementation.
    """
    from .utils.resource_utils import estimate_shard_info
    shard_shape = parse_shape(args.shard_shape, "shard_shape") if args.shard_shape else None
    chunk_shape = parse_shape(args.chunk_shape, "chunk_shape") if args.chunk_shape else None
    return estimate_shard_info(
        volume_shape, dtype_str,
        output_format=args.output_format,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        axes_order=axes_order,
        no_sharding=args.no_sharding,
    )


# Import resource calculation functions from utils (shared with batch.py)
from .utils.resource_utils import calculate_memory as _calculate_memory
from .utils.resource_utils import calculate_wall_time as _calculate_wall_time
from .utils.resource_utils import calculate_job_resources as _calculate_job_resources
from .utils.resource_utils import is_native_source as _is_native_source


def run_local_pyramid(s0_path, root_path, downsample_method="auto",
                      custom_per_level_factors=None, use_shard=True,
                      include_translation=True, verbose=True):
    """Run local pyramid generation from an existing s0.

    Shared by standalone --auto_multiscale and conversion + --auto_multiscale paths.

    Args:
        s0_path: Path to the base resolution level (e.g., /data/volume.zarr/raw/s0)
        root_path: Parent directory of s0 where pyramid levels are written
        downsample_method: Already-resolved method ("mean", "mode", etc.)
        custom_per_level_factors: Optional list of per-level factor lists
        use_shard: Whether to use sharding for output
        include_translation: Include translation transforms in OME metadata
        verbose: Print progress
    """
    from .core.downsampler import downsample_level
    from .core.pyramid import PyramidPlanner
    from .utils import update_ome_metadata_if_needed
    from .utils.metadata_utils import detect_level_format, get_level_name

    planner = PyramidPlanner(s0_path, include_translation=include_translation, downsample_method=downsample_method)
    plan = planner.calculate_pyramid_plan(custom_per_level_factors=custom_per_level_factors)
    planner.print_pyramid_plan(plan)

    if plan['num_levels'] == 0:
        print("\nNo pyramid levels needed - dataset is already at minimum size.")
        return plan

    print("\nRunning local pyramid generation...")
    planner.precreate_all_levels(plan, use_shard=use_shard, verbose=verbose)

    prefix = detect_level_format(root_path)

    for level_info in plan['levels']:
        level = level_info['level']
        per_level_factor = level_info['per_level_factor']
        cumulative_factors = level_info['cumulative_factor']

        source_level = level - 1
        source_level_name = get_level_name(source_level, prefix)
        source_path = os.path.join(root_path, source_level_name)

        print(f"\n--- Downsampling {source_level_name} -> {get_level_name(level, prefix)} (per-level factor: {per_level_factor}, cumulative: {cumulative_factors}) ---")

        downsample_level(
            s0_path=source_path,
            output_path=root_path,
            target_level=level,
            factors=per_level_factor,
            use_shard=use_shard,
            custom_shard_shape=level_info.get('shard_shape'),
            custom_chunk_shape=level_info.get('chunk_shape'),
            downsample_method=downsample_method,
            verbose=verbose,
            cumulative_factor_for_metadata=cumulative_factors,
        )

    update_ome_metadata_if_needed(root_path, use_ome_structure=True, include_translation=include_translation, downsample_method=downsample_method)

    print(f"\n{'='*60}")
    print(f"AUTO-MULTISCALE COMPLETE: {root_path}")
    print(f"Generated s0 through s{plan['num_levels']}")
    print(f"{'='*60}")

    return plan


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
    needs_auto = memory_gb is None or wall_time is None or args.cores is None

    dataset_size_gb = None
    if needs_auto:
        print("Reading input metadata for resource estimation...")
        volume_shape, dtype_str, axes_order = _get_input_metadata(args)

        # When --bbox is used, estimate resources from the bbox subvolume, not the full source
        if getattr(args, 'bbox', None):
            bbox_origin, bbox_size = parse_bbox(args.bbox)
            # bbox_size has same number of spatial dims; squeeze singleton channel if present
            if len(volume_shape) == len(bbox_size) + 1:
                volume_shape = bbox_size
            else:
                volume_shape = bbox_size[:len(volume_shape)]
            print(f"  Bbox applied for resource estimation: size={bbox_size}")

        shard_shape, total_shards = _estimate_shard_info(args, volume_shape, dtype_str, axes_order)
        # Both BIOIO and BioFormats use Dask and have similar overhead
        use_bioio = getattr(args, 'use_bioio', False) or getattr(args, 'use_bioformats', False)
        dtype_bytes = np.dtype(dtype_str).itemsize
        dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)
        print(f"  Shape: {volume_shape}, dtype: {dtype_str}")
        use_sharding = args.output_format == "zarr3" and not args.no_sharding
        if use_sharding:
            print(f"  Shard shape: {shard_shape}, total shards: {total_shards}")
        else:
            print(f"  Chunk shape: {shard_shape}, total chunks: {total_shards}")
        if use_bioio:
            mode_name = "Bio-Formats" if getattr(args, 'use_bioformats', False) else "BioIO"
            print(f"  {mode_name} mode: applying 10x wall time and 3x memory multipliers")

        # Detect source type: TensorStore-native (fast) vs file-decoded (slower)
        is_native = _is_native_source(args.input)
        if not is_native:
            _input_ext = os.path.splitext(args.input)[1].lower()
            print(f"  Source type: file-decoded ({_input_ext})")

        if memory_gb is None:
            memory_gb = _calculate_memory(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=use_bioio, output_format=args.output_format)

    # Cores first (wall time scales with core count for non-sharded)
    # Non-sharded (Zarr2 / Zarr3 --no_sharding): min 4 cores — many small chunk files
    #   benefit from I/O parallelism + compression threads
    # Zarr3 sharded: based on memory (large shard buffers need more cores)
    # Single-shard: 1 core (no parallelism benefit)
    if args.cores is not None:
        cores = args.cores
    elif use_sharding and total_shards == 1:
        cores = 1
    elif not use_sharding and dataset_size_gb is not None:
        cores = min(8, max(4, int(math.ceil(dataset_size_gb / 25)) * 2))
    else:
        cores = min(8, max(1, int(math.ceil(memory_gb / 15)) * 2))

    # Wall time: pass cores so non-sharded estimates scale with parallelism
    if needs_auto and wall_time is None:
        wall_time = _calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=use_bioio,
                                         output_format=args.output_format, no_sharding=args.no_sharding,
                                         cores=cores, is_native_source=is_native)

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    # Job name: tsv2_{src_ext}_to_{out_format}_{input_stem}
    input_name = os.path.basename(args.input)
    input_stem = os.path.splitext(input_name)[0]
    input_ext = os.path.splitext(input_name)[1].lstrip(".")
    job_name = f"tsv2_{input_ext}_to_{args.output_format}_{input_stem}"
    job_name = job_name.replace(" ", "_")[:128]

    # Log directory: user-specified or default next to output
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = args.log_dir or os.path.join(output_parent, "output")
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
    if getattr(args, 'voxel_unit', None):
        reinvoke += ["--voxel_unit", args.voxel_unit]
    if getattr(args, 'is_label', False):
        reinvoke.append("--is-label")
    if getattr(args, 'expand_to_5d', False):
        reinvoke.append("--expand-to-5d")
    if getattr(args, 'data_type', 'auto') != 'auto':
        reinvoke += ["--data-type", args.data_type]
    if getattr(args, 'label_key', 'segmentation') != 'segmentation':
        reinvoke += ["--label-key", args.label_key]
    if getattr(args, 'image_key', 'raw') != 'raw':
        reinvoke += ["--image-key", args.image_key]
    if getattr(args, 'use_nested_structure', True) is False:
        reinvoke.append("--no-nested-structure")
    if getattr(args, 'bbox', None):
        reinvoke += ["--bbox", args.bbox]
    if getattr(args, 'axes_order', None):
        reinvoke += ["--axes_order", args.axes_order]
    if args.log_dir:
        reinvoke += ["--log_dir", args.log_dir]
    if getattr(args, 'no_ome_meta_export', False):
        reinvoke.append("--no_ome_meta_export")
    if getattr(args, 'no_ome_xml_attr', False):
        reinvoke.append("--no_ome_xml_attr")
    if getattr(args, 'no_omero', False):
        reinvoke.append("--no-omero")
    if getattr(args, 'dtype', None):
        reinvoke += ["--dtype", args.dtype]
    if getattr(args, 'add_to_existing', False):
        reinvoke.append("--add-to-existing")
    # Convert to properly quoted shell command string
    # This handles paths with spaces correctly when bsub creates its wrapper
    reinvoke_str = shlex.join(reinvoke)

    # Build bsub command - use bash -c to run the quoted command
    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024}]",  # GB to MB for LSF
        "-P", args.project,
        "-o", log_path,
        "-e", error_path,
    ]
    if args.job_group:
        command += ["-g", args.job_group]
    command += ["/bin/bash", "-c", reinvoke_str]

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
    print(f"  Command:   {reinvoke_str}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
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


def _submit_dependent_pyramid(args, conversion_job_id: str):
    """Submit a dependent auto_multiscale coordinator job that waits for s0 conversion.

    After the conversion job finishes, this coordinator job runs
    --auto_multiscale on the output zarr, which submits chained
    pyramid level jobs via create_pyramid_parallel().

    Args:
        args: Parsed command-line arguments
        conversion_job_id: LSF job ID of the s0 conversion job to depend on
    """
    # Build reinvoke command for standalone auto_multiscale on the output.
    # Use absolute path since the coordinator runs from a potentially different
    # working directory.
    #
    # CRITICAL: point --input at the SUBGROUP that holds the freshly written s0
    # (e.g. <output>/labels/segmentation), not the container root. Otherwise
    # find_base_level() reads the container's root .zattrs and may return a
    # stale subgroup written by a prior stage, causing the pyramid to
    # overwrite the wrong levels.
    output_abs = os.path.abspath(args.output)
    subgroup = _resolve_conversion_subgroup(args)
    pyramid_input = os.path.join(output_abs, subgroup) if subgroup else output_abs
    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--input", pyramid_input,
        "--auto_multiscale",
        "--submit",
        "-P", args.project,
    ]
    if getattr(args, 'downsample_method', 'auto') != 'auto':
        reinvoke += ["--downsample_method", args.downsample_method]
    if getattr(args, 'per_level_factors', None):
        reinvoke += ["--per_level_factors", args.per_level_factors]
    if getattr(args, 'no_translation', False):
        reinvoke.append("--no_translation")
    if args.log_dir:
        reinvoke += ["--log_dir", args.log_dir]
    if args.memory:
        reinvoke += ["--memory", str(args.memory)]
    if args.wall_time:
        reinvoke += ["--wall_time", args.wall_time]
    if args.cores:
        reinvoke += ["--cores", str(args.cores)]

    reinvoke_str = shlex.join(reinvoke)

    # Coordinator job: small allocation, just needs to run bsub for each level
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = args.log_dir or os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    job_name = f"tsv2_pyramid_coordinator_{os.path.basename(args.output)}"
    job_name = job_name.replace(" ", "_")[:128]

    command = [
        "bsub",
        "-J", job_name,
        "-n", "1",
        "-W", "0:30",
        "-M", "15GB",
        "-R", "rusage[mem=15360]",
        "-P", args.project,
        "-w", f"done({conversion_job_id})",
        "-o", os.path.join(log_dir, f"output__{job_name}_%J.log"),
        "-e", os.path.join(log_dir, f"error__{job_name}_%J.log"),
    ]
    if args.job_group:
        command += ["-g", args.job_group]
    command += ["/bin/bash", "-c", reinvoke_str]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        import re
        match = re.search(r'Job <(\d+)>', result.stdout)
        coordinator_id = match.group(1) if match else "unknown"
        print(f"\nPyramid coordinator job {coordinator_id} submitted (depends on conversion job {conversion_job_id}).")
        print(f"  After conversion completes, pyramid levels will be submitted automatically.")
    else:
        print(f"\nWarning: Pyramid coordinator submission failed: {result.stderr.strip()}")
        print(f"  You can manually run pyramid after conversion finishes:")
        print(f"  python -m tensorswitch_v2 -i {args.output} --auto_multiscale --submit -P {args.project}")


def _submit_upsample_job(args, verbose=True):
    """Submit an LSF bsub job for upsampling anisotropic → isotropic.

    Re-invokes tensorswitch_v2 --upsample (without --submit) on a cluster node.
    Uses calculate_upsample_resources() for proper resource estimation.
    """
    if not args.project:
        raise ValueError(
            "Missing required argument: --project/-P\n"
            "When using --submit, you must specify an LSF project for billing."
        )

    import json as _json
    import tensorstore as _ts
    from .utils.resource_utils import calculate_upsample_resources
    from .utils.tensorstore_utils import get_zarr_store_spec as _get_spec
    from .core.upsampler import Upsampler

    # Use Upsampler to compute zoom factors (single source of truth, no duplication)
    from .core.upsampler import upsample_to_isotropic as _upsample_fn
    temp_up = Upsampler(
        input_path=args.input.rstrip('/'),
        output_path="/tmp/_placeholder_",
        target_voxel_sizes=[],  # will read from metadata
    )
    src_voxels, axes_info = temp_up._get_source_voxel_sizes_and_axes()

    # Determine target voxel size (same logic as upsample_to_isotropic)
    spatial_voxels = []
    if axes_info:
        for i, ax in enumerate(axes_info):
            if ax.get("type") == "space" and i < len(src_voxels):
                spatial_voxels.append(src_voxels[i])
    else:
        spatial_voxels = list(src_voxels[:3])
    target_voxel = args.target_voxel_size if args.target_voxel_size else min(spatial_voxels)

    # Build full target voxel list and compute zoom factors via Upsampler
    full_target = []
    spatial_idx = 0
    for i in range(len(src_voxels)):
        if axes_info and i < len(axes_info) and axes_info[i].get("type") == "space":
            full_target.append(target_voxel)
            spatial_idx += 1
        else:
            full_target.append(src_voxels[i])
    temp_up.target_voxel_sizes = full_target
    zoom_factors = temp_up.compute_zoom_factors()

    # Read source shape/chunks via TensorStore
    _src_spec = _get_spec(args.input.rstrip('/'))
    src = _ts.open(_src_spec, open=True).result()
    src_shape = list(src.shape)
    src_chunks = list(src.chunk_layout.read_chunk.shape)
    dtype_str = src.dtype.numpy_dtype.name

    # Auto-calculate resources using the proper system
    shard_shape_list = [int(x) for x in args.shard_shape.split(',')] if args.shard_shape else None
    auto_mem, auto_wall, auto_cores = calculate_upsample_resources(
        source_shape=src_shape,
        dtype=dtype_str,
        source_chunks=src_chunks,
        zoom_factors=zoom_factors,
        output_format=args.output_format,
        no_sharding=args.no_sharding,
        shard_shape=shard_shape_list,
    )

    # Allow user overrides
    memory_gb = args.memory or auto_mem
    wall_time = args.wall_time or auto_wall
    cores = args.cores or auto_cores

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    # Job name
    input_name = os.path.basename(os.path.dirname(os.path.dirname(args.input)))
    job_name = f"tsv2_upsample_{input_name}"
    job_name = job_name.replace(" ", "_")[:128]

    # Log directory
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = getattr(args, 'log_dir', None) or os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
    error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

    # Build re-invocation command (same args, without --submit and LSF flags)
    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--upsample",
        "--input", args.input,
        "--output", args.output,
    ]
    if args.target_voxel_size is not None:
        reinvoke += ["--target_voxel_size", str(args.target_voxel_size)]
    if args.upsample_method != "auto":
        reinvoke += ["--upsample_method", args.upsample_method]
    if getattr(args, 'is_label', False):
        reinvoke.append("--is-label")
    if args.auto_multiscale:
        reinvoke.append("--auto_multiscale")
    if args.downsample_method != "auto":
        reinvoke += ["--downsample_method", args.downsample_method]
    if args.per_level_factors:
        reinvoke += ["--per_level_factors", args.per_level_factors]
    if args.no_translation:
        reinvoke.append("--no-translation")
    # Output format params
    if args.output_format != "zarr3":
        reinvoke += ["--output_format", args.output_format]
    if args.no_sharding:
        reinvoke.append("--no_sharding")
    if args.shard_shape:
        reinvoke += ["--shard_shape", args.shard_shape]
    if args.compression != "zstd":
        reinvoke += ["--compression", args.compression]
    if args.compression_level != 5:
        reinvoke += ["--compression_level", str(args.compression_level)]

    reinvoke_str = shlex.join(reinvoke)

    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024}]",
        "-P", args.project,
        "-o", log_path,
        "-e", error_path,
    ]
    if args.job_group:
        command += ["-g", args.job_group]
    command += ["/bin/bash", "-c", reinvoke_str]

    print("=" * 72)
    print("LSF Upsample Job Submission")
    print("=" * 72)
    print(f"  Job name:    {job_name}")
    print(f"  Source:      {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Source shape: {src_shape}")
    print(f"  Zoom factors: {[round(f, 4) for f in zoom_factors]}")
    print(f"  Format:      {args.output_format}")
    print(f"  Cores:       {cores}")
    print(f"  Memory:      {memory_gb} GB")
    print(f"  Wall time:   {wall_time}")
    print(f"  Project:     {args.project}")
    print(f"  Log:         {log_path}")
    print(f"  Command:     {reinvoke_str}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Job submitted successfully.")
        print(result.stdout.strip())
    else:
        print(f"Job submission failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"bsub failed: {result.stderr.strip()}")


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

    # Log directory: user-specified or default next to output
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = args.log_dir or os.path.join(output_parent, "output")
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
    if args.log_dir:
        reinvoke += ["--log_dir", args.log_dir]
    if getattr(args, 'no_ome_meta_export', False):
        reinvoke.append("--no_ome_meta_export")
    if getattr(args, 'no_ome_xml_attr', False):
        reinvoke.append("--no_ome_xml_attr")
    if getattr(args, 'cumulative_factor_for_metadata', None):
        reinvoke += ["--cumulative_factor_for_metadata", args.cumulative_factor_for_metadata]

    # Convert to properly quoted shell command string
    # This handles paths with spaces correctly when bsub creates its wrapper
    reinvoke_str = shlex.join(reinvoke)

    # Build bsub command - use bash -c to run the quoted command
    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024}]",  # GB to MB for LSF
        "-P", args.project,
        "-o", log_path,
        "-e", error_path,
    ]
    if args.job_group:
        command += ["-g", args.job_group]
    command += ["/bin/bash", "-c", reinvoke_str]

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

    # Try to get TensorStore info
    try:
        input_store = reader.get_tensorstore()
        print(f"\nTensorStore info:")
        print(f"  shape: {list(input_store.shape)}")
        dtype_name = get_dtype_name(input_store.dtype)
        print(f"  dtype: {dtype_name}")
        labels = list(input_store.domain.labels) if input_store.domain.labels else []
        if labels:
            print(f"  domain labels: {labels}")
    except Exception as e:
        print(f"(TensorStore info not available: {e})")

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
    elif args.preset == "paintera":
        # Paintera preset: n5 (default) with xyz, or zarr2 with zyx
        # If user didn't explicitly set output_format (still default zarr3), use n5
        if args.output_format == "zarr3":
            args.output_format = "n5"
        if not args.chunk_shape:
            args.chunk_shape = "64,64,64"
        if args.compression == "zstd":
            args.compression = "gzip"
        if not args.axes_order:
            args.axes_order = "xyz" if args.output_format == "n5" else "zyx"
        if not args.quiet:
            print(f"Using Paintera preset: output={args.output_format}, "
                  f"axes={args.axes_order}, chunk=64x64x64, compression=gzip")

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
        # This handles both:
        # - Regular directories containing .zarr files
        # - BigStitcher multi-tile .zarr directories containing tile .zarr subdirectories
        is_batch_pyramid_mode = False
        if args.auto_multiscale and args.input:
            input_path = args.input.rstrip('/')
            # Batch mode: directory containing multiple .zarr subdirectories
            # NOT batch mode: has s0/.zarray inside (single multiscale dataset)
            if os.path.isdir(input_path):
                has_s0 = os.path.exists(os.path.join(input_path, 's0'))
                has_zarray = os.path.exists(os.path.join(input_path, '.zarray'))
                zarr_subdirs = [d for d in os.listdir(input_path) if d.endswith('.zarr') and os.path.isdir(os.path.join(input_path, d))]
                is_batch_pyramid_mode = len(zarr_subdirs) > 0 and not has_s0 and not has_zarray

        if not args.output and not is_batch_pyramid_mode and not args.auto_multiscale:
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
        from .core.batch import (
            detect_input_mode,
            BatchConverter,
            convert_discovered_folder,
            submit_discovered_folder_lsf,
        )

        input_mode = detect_input_mode(args.input, output_path=args.output)

        if input_mode == 'discovered_folder':
            # Discovered folder mode: convert image and/or segmentation datasets
            # to a single zarr with nested structure
            image_only = getattr(args, 'image_only', False)
            labels_only = getattr(args, 'labels_only', False)
            image_key = getattr(args, 'image_key', 'raw')
            label_key = getattr(args, 'label_key', 'segmentation')
            use_nested = getattr(args, 'use_nested_structure', True)

            if args.submit:
                if not args.project:
                    raise ValueError("--project is required with --submit")

                result = submit_discovered_folder_lsf(
                    input_dir=args.input,
                    output_path=args.output,
                    project=args.project,
                    image_only=image_only,
                    labels_only=labels_only,
                    image_key=image_key,
                    label_key=label_key,
                    use_nested_structure=use_nested,
                    output_format=args.output_format,
                    chunk_shape=args.chunk_shape,
                    shard_shape=args.shard_shape,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    memory_gb=args.memory,  # None = auto-calculate per dataset
                    wall_time=args.wall_time,  # None = auto-calculate per dataset
                    cores=args.cores,  # None = auto-calculate per dataset
                    job_group=args.job_group,
                    dry_run=args.dry_run,
                )
                if result.get('error'):
                    print(f"\nError: {result['error']}")
                    sys.exit(1)
            else:
                result = convert_discovered_folder(
                    input_dir=args.input,
                    output_path=args.output,
                    image_only=image_only,
                    labels_only=labels_only,
                    image_key=image_key,
                    label_key=label_key,
                    use_nested_structure=use_nested,
                    output_format=args.output_format,
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    verbose=verbose,
                )
                if result.error:
                    print(f"\nError: {result.error}")
                    sys.exit(1)

            return

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

                # Auto-calculate resources from first discovered file when not overridden
                memory_gb = args.memory
                wall_time = args.wall_time
                cores = args.cores

                if not (memory_gb and wall_time and cores):
                    first_file = files[0].input_path
                    print(f"Reading sample file for resource estimation: {os.path.basename(first_file)}")
                    original_input = args.input
                    args.input = first_file
                    try:
                        volume_shape, dtype_str, axes_order = _get_input_metadata(args)
                    finally:
                        args.input = original_input

                    is_native = _is_native_source(first_file)

                    auto_mem, auto_wall, auto_cores = _calculate_job_resources(
                        shape=list(volume_shape), dtype=dtype_str,
                        output_format=args.output_format,
                        chunk_shape_str=args.chunk_shape, shard_shape_str=args.shard_shape,
                        axes_order=axes_order, no_sharding=args.no_sharding,
                        is_native_source=is_native,
                    )
                    memory_gb = memory_gb or auto_mem
                    wall_time = wall_time or auto_wall
                    cores = cores or auto_cores
                    print(f"  Auto-calculated: {memory_gb} GB, {wall_time}, {cores} cores")

                result = batch.submit_lsf(
                    project=args.project,
                    chunk_shape=args.chunk_shape,
                    shard_shape=args.shard_shape,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    memory_gb=memory_gb,
                    wall_time=wall_time,
                    cores=cores,
                    max_concurrent=args.max_concurrent,
                    job_group=args.job_group,
                    skip_existing=skip_existing,
                    dry_run=args.dry_run,
                    voxel_size=args.voxel_size,
                    voxel_unit=args.voxel_unit,
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

    # Handle auto_multiscale mode (standalone pyramid generation on existing zarr).
    # Route to pyramid-only when the user's intent is "build pyramid on existing
    # zarr/n5 level-or-container" — i.e. input is a zarr/n5 level or container
    # and output is absent or an ancestor of input. Raw source files (.tif,
    # .czi, ...) fail this check, so the convert+pyramid workflow
    # (``-i raw.czi -o new.zarr --auto_multiscale``) stays on the conversion
    # path and picks up pyramid via the dependent-coordinator chain.
    if args.auto_multiscale and _is_pyramid_only_intent(args.input, args.output):
        if args.output:
            print(
                f"Note: input '{args.input}' is an existing zarr level/container — "
                f"generating pyramid in-place (not converting)."
            )
        from .core.pyramid import PyramidPlanner, create_pyramid_parallel
        from .utils import update_ome_metadata_if_needed
        import glob as glob_module

        # Parse custom per-level factors if provided
        custom_per_level_factors = None
        if args.per_level_factors:
            custom_per_level_factors = parse_per_level_factors(args.per_level_factors)
            print(f"Using custom per-level factors: {custom_per_level_factors}")

        # Detect batch mode: directory containing multiple .zarr subdirectories
        # Batch mode is triggered when:
        # 1. Input is a directory containing .zarr subdirectories (e.g., dataset.ome.zarr/s0-t0.zarr, s1-t0.zarr, ...)
        # 2. AND the input does NOT have s0/.zarray directly (not a single multiscale dataset)
        # This handles both:
        # - Regular directories containing .zarr files
        # - BigStitcher multi-tile .zarr directories containing tile .zarr subdirectories
        input_path = args.input.rstrip('/')
        is_batch_mode = False
        if os.path.isdir(input_path):
            has_s0 = os.path.exists(os.path.join(input_path, 's0'))
            has_zarray = os.path.exists(os.path.join(input_path, '.zarray'))
            zarr_subdirs = [d for d in os.listdir(input_path) if d.endswith('.zarr') and os.path.isdir(os.path.join(input_path, d))]
            # Batch mode if: has .zarr subdirs AND is NOT a single multiscale dataset (no s0/.zarray)
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
                        include_translation=not args.no_translation,
                        log_dir=getattr(args, 'log_dir', None),
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
        s0_path, _ = find_base_level(args.input, verbose=verbose)
        # root_path must be the direct parent of s0 (e.g., raw/) not the container root
        root_path = os.path.dirname(s0_path)

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
                include_translation=not args.no_translation,
                log_dir=getattr(args, 'log_dir', None),
            )
            num_levels = result.get('pyramid_plan', {}).get('num_levels', 0)
            coordinator_job_id = result.get('coordinator_job_id')
            if num_levels > 0 and coordinator_job_id:
                print(f"\nCoordinator will submit {num_levels} level jobs sequentially (chained mode).")
                print("Each level reads from the previous level: s1←s0, s2←s1, etc.")
                print("After all jobs complete, root metadata will be updated automatically.")
            else:
                print("\nNo pyramid levels needed - dataset is already at minimum size.")
        else:
            # Local mode: run downsampling directly for each level
            resolved_method = resolve_downsample_method(args.downsample_method, s0_path)
            run_local_pyramid(
                s0_path=s0_path,
                root_path=root_path,
                downsample_method=resolved_method,
                custom_per_level_factors=custom_per_level_factors,
                use_shard=use_shard,
                include_translation=not args.no_translation,
                verbose=verbose,
            )
        return

    # Handle upsample mode (anisotropic → isotropic)
    if args.upsample:
        from .core.upsampler import upsample_to_isotropic as _upsample_fn

        if not args.output:
            raise ValueError("--output/-o is required with --upsample")

        if args.submit:
            # Submit upsample job to LSF cluster
            _submit_upsample_job(args, verbose=verbose)
            return

        # Determine s0 output path: if output looks like a container root,
        # mirror the input's subgroup structure (e.g., img/s0)
        input_path = args.input.rstrip('/')
        output_base = args.output.rstrip('/')

        # Auto-detect subgroup from input (e.g., img/s0 → group=img, level=s0)
        input_basename = os.path.basename(input_path)
        input_group = os.path.basename(os.path.dirname(input_path))

        # If output already ends with the level name, use as-is
        # Otherwise, mirror the input structure
        import re
        level_pattern = re.compile(r'^(s?\d+)$')
        if level_pattern.match(os.path.basename(output_base)):
            s0_output = output_base
        elif input_group and level_pattern.match(input_basename):
            s0_output = os.path.join(output_base, input_group, input_basename)
        else:
            s0_output = os.path.join(output_base, "s0")

        # Safe write: write to .tmp, rename on completion
        final_output = args.output.rstrip('/')
        tmp_output = _tmp_path_for(final_output)
        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)
            if verbose:
                print(f"Removed leftover temporary path: {tmp_output}")

        # Remap s0_output into tmp space
        if s0_output.startswith(final_output):
            tmp_s0 = tmp_output + s0_output[len(final_output):]
        else:
            tmp_s0 = s0_output  # Shouldn't happen, but be safe

        if verbose:
            print(f"Writing to temporary path: {tmp_output}")

        stats = _upsample_fn(
            input_path=input_path,
            output_path=tmp_s0,
            target_voxel_size=args.target_voxel_size,
            upsample_method=args.upsample_method,
            is_label=getattr(args, 'is_label', False),
            verbose=verbose,
            output_format=args.output_format,
            no_sharding=args.no_sharding,
            shard_shape=[int(x) for x in args.shard_shape.split(',')] if args.shard_shape else None,
            compression=args.compression,
            compression_level=args.compression_level,
        )

        # Safe write: rename .tmp → final before pyramid so pyramid finds s0
        _finalize_tmp_path(tmp_output, final_output, verbose=verbose)

        # Auto-pyramid after upsampling if requested
        if args.auto_multiscale:
            root_path = os.path.dirname(final_output)
            resolved_method = resolve_downsample_method(args.downsample_method, final_output)

            custom_per_level_factors = None
            if args.per_level_factors:
                custom_per_level_factors = parse_per_level_factors(args.per_level_factors)

            run_local_pyramid(
                s0_path=final_output,
                root_path=root_path,
                downsample_method=resolved_method,
                custom_per_level_factors=custom_per_level_factors,
                include_translation=not args.no_translation,
                verbose=verbose,
            )

        if verbose:
            print(f"\nUpsampling complete.")
            print(f"  Output: {final_output}")
            print(f"  Shape:  {stats['input_shape']} → {stats['output_shape']}")
            print(f"  Method: {stats['upsample_method']}")
            print(f"  Time:   {stats['elapsed_time']:.1f}s")
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

        downsample_factors = [int(x) for x in factor_str.split(",")]

        # In chained mode, the coordinator passes --cumulative_factor_for_metadata
        # with the true cumulative factor from s0 for correct metadata
        cumulative_factor_for_metadata = None
        if args.cumulative_factor_for_metadata:
            cumulative_factor_for_metadata = [int(x) for x in args.cumulative_factor_for_metadata.split(",")]

        if args.submit:
            # Submit as LSF job
            submit_downsample_job(args, downsample_factors)
        else:
            # Run locally - resolve 'auto' method first
            resolved_method = resolve_downsample_method(args.downsample_method, args.input)
            downsample_level(
                s0_path=args.input,
                output_path=args.output,
                target_level=args.target_level,
                factors=downsample_factors,
                start_idx=args.start_idx or 0,
                stop_idx=args.stop_idx,
                use_shard=use_shard,
                custom_shard_shape=list(shard_shape) if shard_shape else None,
                custom_chunk_shape=list(chunk_shape) if chunk_shape else None,
                downsample_method=resolved_method,
                verbose=verbose,
                cumulative_factor_for_metadata=cumulative_factor_for_metadata,
            )
        return

    # Standard conversion mode (single level / s0)
    if args.submit:
        job_id = submit_job(args, return_job_id=True)

        # If --auto_multiscale, submit a dependent pyramid coordinator job
        if args.auto_multiscale and job_id:
            _submit_dependent_pyramid(args, conversion_job_id=job_id)

        return

    # --- Safe write: write to .tmp, rename on completion ---
    add_to_existing = getattr(args, 'add_to_existing', False)
    final_output = args.output

    if add_to_existing:
        # Subgroup-level safe write: write to labels.tmp/ inside existing container
        if not os.path.exists(final_output):
            raise FileNotFoundError(
                f"--add-to-existing: target container does not exist: {final_output}"
            )
        # Validate it's a zarr container
        is_zarr3 = os.path.exists(os.path.join(final_output, 'zarr.json'))
        is_zarr2 = os.path.exists(os.path.join(final_output, '.zgroup'))
        if not is_zarr3 and not is_zarr2:
            raise ValueError(
                f"--add-to-existing: target is not a zarr container: {final_output}"
            )
        subgroup = _resolve_conversion_subgroup(args)
        if not subgroup:
            raise ValueError(
                "--add-to-existing requires --data-type labels or --data-type image "
                "(or --is-label for label data)"
            )
        subgroup_parent = subgroup.split('/')[0]  # "labels" or "raw"
        tmp_subgroup_path = os.path.join(final_output, subgroup_parent + '.tmp')
        # Clean leftover .tmp subgroup from prior failed run
        if os.path.exists(tmp_subgroup_path):
            shutil.rmtree(tmp_subgroup_path)
            if verbose:
                print(f"Removed leftover temporary subgroup: {tmp_subgroup_path}")
        # Warn if target subgroup already exists
        existing_subgroup = os.path.join(final_output, subgroup_parent)
        if os.path.exists(existing_subgroup) and verbose:
            print(f"Warning: existing {subgroup_parent}/ will be replaced on completion")
        # Route writer to .tmp subgroup via labels_container override
        if subgroup_parent == 'labels':
            args._labels_container_override = 'labels.tmp'
        else:
            args._image_key_override = subgroup_parent + '.tmp'
        tmp_output = None  # Signal: no container-level .tmp
        if verbose:
            print(f"Writing to subgroup: {tmp_subgroup_path}")
    else:
        tmp_output = _tmp_path_for(final_output)
        # Clean up leftover .tmp from a prior failed run
        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)
            if verbose:
                print(f"Removed leftover temporary path: {tmp_output}")
        args.output = tmp_output
        if verbose:
            print(f"Writing to temporary path: {tmp_output}")

    reader = create_reader(args)

    # Warn about TIFF CYX/IYX-style axes mislabeling before conversion begins
    # (for submit paths the same check runs inside _get_input_metadata).
    try:
        _axes_for_warning = None
        _store_for_warning = reader.get_tensorstore()
        if hasattr(_store_for_warning, 'domain') and hasattr(_store_for_warning.domain, 'labels'):
            _labels = _store_for_warning.domain.labels
            if _labels and all(_labels):
                _axes_for_warning = [
                    'c' if l.lower() == 'channel' else l.lower() for l in _labels
                ]
        _warn_if_axis_voxel_mismatch(args, _axes_for_warning)
    except Exception:
        pass  # warning is best-effort; never block conversion

    # Determine force_order from CLI args (None = auto-detect)
    force_order = None
    if args.force_c_order:
        force_order = 'c'
    elif args.force_f_order:
        force_order = 'f'

    # Resolve data_type and is_label consistently
    # Priority: explicit --data-type > explicit --is-label > auto-detect
    cli_data_type = getattr(args, 'data_type', 'auto')
    is_label = getattr(args, 'is_label', False)

    if cli_data_type == 'labels':
        # Explicit --data-type labels
        resolved_data_type = 'labels'
        is_label = True
    elif is_label:
        # Explicit --is-label flag
        resolved_data_type = 'labels'
    elif cli_data_type == 'auto':
        # Auto-detect based on dtype
        resolved_data_type = 'image'  # Default
        try:
            store = reader.get_tensorstore()
            dtype_str = get_dtype_name(store.dtype)

            from .utils.metadata_utils import is_segmentation_dtype
            if is_segmentation_dtype(dtype_str):
                is_label = True
                resolved_data_type = 'labels'
                if verbose:
                    print(f"Auto-detected segmentation data (dtype: {dtype_str}), using labels structure")
        except Exception:
            pass  # If detection fails, default to image
    else:
        # Explicit --data-type image
        resolved_data_type = 'image'

    writer = create_writer(args, data_type=resolved_data_type)

    # Handle --show_spec: print specs and exit without converting
    if args.show_spec:
        show_conversion_spec(reader, writer, args, chunk_shape, shard_shape)
        return

    from .core.converter import DistributedConverter

    converter = DistributedConverter(reader, writer)

    # Parse voxel_size override if provided
    voxel_size_override = None
    voxel_unit = getattr(args, 'voxel_unit', None)
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

    # Parse --axes_order override
    axes_order_override = None
    if getattr(args, 'axes_order', None):
        axes_order_override = list(args.axes_order.lower().replace(',', ''))
        valid_spatial = {'x', 'y', 'z'}
        if not all(a in valid_spatial for a in axes_order_override):
            raise ValueError(f"--axes_order must contain only x, y, z, got: {args.axes_order}")
        if len(axes_order_override) != len(set(axes_order_override)):
            raise ValueError(f"--axes_order must not have duplicates, got: {args.axes_order}")

    # Parse bbox for subvolume extraction
    bbox = parse_bbox(args.bbox) if getattr(args, 'bbox', None) else None

    no_ome_meta_export = getattr(args, 'no_ome_meta_export', False)
    no_ome_xml_attr = getattr(args, 'no_ome_xml_attr', False)

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
            voxel_unit=voxel_unit,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
            bbox=bbox,
            axes_order_override=axes_order_override,
            no_ome_meta_export=no_ome_meta_export,
            no_ome_xml_attr=no_ome_xml_attr,
            output_dtype=getattr(args, 'dtype', None),
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
            voxel_unit=voxel_unit,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
            bbox=bbox,
            axes_order_override=axes_order_override,
            no_ome_meta_export=no_ome_meta_export,
            no_ome_xml_attr=no_ome_xml_attr,
            output_dtype=getattr(args, 'dtype', None),
        )

    # Write source provenance metadata when --bbox is used
    if bbox:
        from .utils.metadata_utils import write_source_metadata
        write_source_metadata(
            output_path=args.output,
            source_url=args.input,
            bbox=bbox,
            voxel_sizes=voxel_size_override,
        )

    # Chain pyramid generation after s0 conversion if --auto_multiscale
    if args.auto_multiscale:
        # Resolve the subgroup the converter just wrote to (e.g. 'raw' or
        # 'labels/segmentation'). Without this, find_base_level() on the
        # container root reads stale multiscales from a prior stage and may
        # return the wrong subgroup, causing the pyramid to overwrite the
        # wrong levels.
        pyramid_subgroup = _resolve_conversion_subgroup(args)
        if add_to_existing and pyramid_subgroup:
            # Pyramid runs on .tmp subgroup before rename
            # e.g. labels/segmentation → labels.tmp/segmentation
            parts = pyramid_subgroup.split('/', 1)
            pyramid_subgroup = parts[0] + '.tmp' + ('/' + parts[1] if len(parts) > 1 else '')
            base_level_input = os.path.join(final_output, pyramid_subgroup)
        else:
            base_level_input = (
                os.path.join(args.output, pyramid_subgroup) if pyramid_subgroup else args.output
            )
        s0_path, _ = find_base_level(base_level_input, verbose=verbose)
        # root_path must be the direct parent of s0 (e.g., raw/) not the container root
        root_path = os.path.dirname(s0_path)
        resolved_method = resolve_downsample_method(args.downsample_method, s0_path)

        custom_per_level_factors = None
        if args.per_level_factors:
            custom_per_level_factors = parse_per_level_factors(args.per_level_factors)

        run_local_pyramid(
            s0_path=s0_path,
            root_path=root_path,
            downsample_method=resolved_method,
            custom_per_level_factors=custom_per_level_factors,
            use_shard=use_shard,
            include_translation=not args.no_translation,
            verbose=verbose,
        )

    # --- Safe write: rename .tmp → final path ---
    if add_to_existing:
        _finalize_add_to_existing(
            final_output=final_output,
            subgroup_parent=subgroup_parent,
            output_format=args.output_format,
            verbose=verbose,
        )
    else:
        _finalize_tmp_path(tmp_output, final_output, verbose=verbose)


if __name__ == "__main__":
    main()
