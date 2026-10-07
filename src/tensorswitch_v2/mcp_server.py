"""
TensorSwitch MCP Server — exposes TensorSwitch v2 as tools for Claude and other LLM agents.

convert and submit_job build a command line from their parameters and run the real CLI
parser and code (see mcp_args.py), so they cannot drift from the command line.

Usage:
    # Run directly
    pixi run python -m tensorswitch_v2.mcp_server

    # Add to Claude Code
    claude mcp add --transport stdio tensorswitch -- pixi run python -m tensorswitch_v2.mcp_server
"""

import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import tensorstore as ts

from tensorswitch_v2.readers.base import is_remote_path
from tensorswitch_v2.utils.tensorstore_utils import get_zarr_store_spec
from tensorswitch_v2.utils.output_group import apply_parent_group, wrap_for_project

from mcp.server.fastmcp import FastMCP

# Logging to stderr (required for stdio transport — stdout is JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tensorswitch-mcp")

mcp = FastMCP("tensorswitch")

_PLACEHOLDER_VOXEL_NOTE = "No voxel size found for axis"


def _relevant(notes: list, voxel_size: str) -> list:
    """Drop the 'using placeholder 1.0' note when the caller gave voxel_size (the placeholder is unused)."""
    return [n for n in notes if not (voxel_size and n.startswith(_PLACEHOLDER_VOXEL_NOTE))]


# Messages from libraries that are harmless for a conversion and only distract the caller.
_BENIGN_NOTES = ("Casting invalid PixelsID",)


@contextlib.contextmanager
def _quiet_capture():
    """Silence stdout (it carries the JSON-RPC stream) and collect what the code warns about.

    Yields a list that is filled when the block ends with the distinct UserWarning messages
    and any 'WARNING ...' lines written to stderr, so the tool response can show them.
    """
    import warnings

    notes: list = []
    out, err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        warnings.simplefilter("always")
        yield notes
        found = [str(w.message).strip() for w in caught if issubclass(w.category, UserWarning)]
        found += [line.strip() for line in err.getvalue().splitlines() if line.strip().startswith("WARNING")]
        for message in found:
            if message and message not in notes and not any(b in message for b in _BENIGN_NOTES):
                notes.append(message)


# Size guard: refuse in-process conversion for datasets larger than this
MCP_CONVERT_MAX_GB = 2


# ---------------------------------------------------------------------------
# Tool 1: inspect_dataset
# ---------------------------------------------------------------------------
@mcp.tool()
def inspect_dataset(path: str) -> str:
    """Inspect a microscopy dataset and return its metadata.

    Returns shape, dtype, voxel sizes, axes, format, and OME-NGFF metadata
    for any dataset TensorSwitch can read (Zarr2/3, N5, HDF5, TIFF, ND2,
    IMS, CZI, Neuroglancer precomputed, and 150+ formats via Bio-Formats).

    For OME-Zarr containers (with raw/, labels/ subdirectories), inspects
    all layers and pyramid levels.

    Args:
        path: Path to the dataset (file, directory, or URL).
              Examples: /data/volume.zarr, /data/image.tif, /data/stack.h5
    """
    try:
        path = path.strip()

        # Stdout carries the JSON-RPC stream under stdio transport, so nothing
        # the readers or discovery code print may reach it.
        with contextlib.redirect_stdout(io.StringIO()):
            # For local paths, check if this is an OME-Zarr container
            if not is_remote_path(path):
                zarr_json = os.path.join(path, "zarr.json")
                zattrs = os.path.join(path, ".zattrs")
                if os.path.isfile(zarr_json) or os.path.isfile(zattrs):
                    return _inspect_zarr_container(path)

            # Remote paths or non-container local: inspect as single dataset
            return _inspect_single_dataset(path)
    except Exception as e:
        logger.error(f"inspect_dataset failed: {e}\n{traceback.format_exc()}")
        return f"Error inspecting {path}: {e}"


def _inspect_zarr_container(path: str) -> str:
    """Inspect an OME-Zarr container with potential nested structure."""
    result = {"path": path, "type": "ome-zarr-container"}

    # Read root metadata
    zarr_json_path = os.path.join(path, "zarr.json")
    zattrs_path = os.path.join(path, ".zattrs")

    if os.path.isfile(zarr_json_path):
        with open(zarr_json_path) as f:
            root_meta = json.load(f)
        result["zarr_format"] = root_meta.get("zarr_format", "unknown")
        attrs = root_meta.get("attributes", {})
    elif os.path.isfile(zattrs_path):
        with open(zattrs_path) as f:
            attrs = json.load(f)
        result["zarr_format"] = 2
    else:
        attrs = {}

    # Extract OME metadata
    ome = attrs.get("ome", attrs)  # zarr3 nests under "ome", zarr2 is flat
    if "multiscales" in ome:
        ms = ome["multiscales"][0]
        result["axes"] = [a["name"] for a in ms.get("axes", [])]
        axis_units = [a["unit"] for a in ms.get("axes", []) if a.get("unit")]
        result["unit"] = axis_units[0] if axis_units else "unknown"
        result["name"] = ms.get("name", "unknown")
        result["type"] = ms.get("type", "unknown")
        result["num_levels"] = len(ms.get("datasets", []))

        # Extract scale for each level
        levels = []
        for ds in ms.get("datasets", []):
            level_info = {"path": ds["path"]}
            for ct in ds.get("coordinateTransformations", []):
                if ct["type"] == "scale":
                    level_info["scale"] = ct["scale"]
            levels.append(level_info)
        result["levels"] = levels

    # Check for labels — enrich with full metadata from each label layer
    if "labels" in ome:
        result["labels"] = _inspect_label_layers(path, ome["labels"])

    # Check for LMVD provenance
    if "lmvd" in attrs:
        result["lmvd_provenance"] = attrs["lmvd"]

    # Discover layers using folder_discovery
    try:
        from tensorswitch_v2.utils.folder_discovery import discover_datasets

        discovery = discover_datasets(path)
        layers = []
        for img in discovery.all_images:
            layers.append({
                "name": img.name,
                "type": "image",
                "dtype": img.dtype,
                "shape": img.shape,
                "format": img.source_format,
                "num_scales": img.num_scales,
            })
        for seg in discovery.all_segmentations:
            layers.append({
                "name": seg.name,
                "type": "segmentation",
                "dtype": seg.dtype,
                "shape": seg.shape,
                "format": seg.source_format,
                "num_scales": seg.num_scales,
            })
        if layers:
            result["layers"] = layers
    except Exception as e:
        logger.warning(f"Layer discovery failed: {e}")

    return json.dumps(result, indent=2)


def _inspect_label_layers(container_path: str, label_names: list) -> list:
    """Inspect each label layer under labels/{name}/ and return enriched metadata."""
    labels_dir = os.path.join(container_path, "labels")
    enriched = []

    for name in label_names:
        label_path = os.path.join(labels_dir, name)
        info = {"name": name}

        if not os.path.isdir(label_path):
            enriched.append(info)
            continue

        # Read label root metadata (zarr.json or .zattrs)
        label_zarr_json = os.path.join(label_path, "zarr.json")
        label_zattrs = os.path.join(label_path, ".zattrs")
        label_attrs = {}

        if os.path.isfile(label_zarr_json):
            try:
                with open(label_zarr_json) as f:
                    meta = json.load(f)
                label_attrs = meta.get("attributes", {})
                info["zarr_format"] = 3
            except (json.JSONDecodeError, IOError):
                pass
        elif os.path.isfile(label_zattrs):
            try:
                with open(label_zattrs) as f:
                    label_attrs = json.load(f)
                info["zarr_format"] = 2
            except (json.JSONDecodeError, IOError):
                pass

        # Extract OME multiscales for this label
        label_ome = label_attrs.get("ome", label_attrs)
        if "multiscales" in label_ome:
            ms = label_ome["multiscales"][0]
            info["axes"] = [a["name"] for a in ms.get("axes", [])]
            info["num_levels"] = len(ms.get("datasets", []))

            # Extract scale from first dataset
            datasets = ms.get("datasets", [])
            if datasets:
                for ct in datasets[0].get("coordinateTransformations", []):
                    if ct["type"] == "scale":
                        info["voxel_sizes"] = ct["scale"]

        # Extract image-label metadata (colors, version)
        if "image-label" in label_ome:
            il = label_ome["image-label"]
            info["image_label_version"] = il.get("version", "unknown")
            colors = il.get("colors", [])
            if colors:
                info["num_colors"] = len(colors)

        # Read s0 array metadata for shape and dtype
        from tensorswitch_v2.utils.folder_discovery import (
            _read_zarr3_dataset,
            _read_zarr2_dataset,
        )
        ds = _read_zarr3_dataset(label_path) or _read_zarr2_dataset(label_path)
        if ds:
            info["shape"] = ds.shape
            info["dtype"] = ds.dtype
            info["num_scales"] = ds.num_scales

        enriched.append(info)

    return enriched


def _inspect_single_dataset(path: str) -> str:
    """Inspect a single dataset file (HDF5, TIFF, ND2, etc.)."""
    from tensorswitch_v2.api import Readers, TensorSwitchDataset

    reader = Readers.auto_detect(path)
    ds = TensorSwitchDataset(path, reader=reader)

    result = {
        "path": path,
        "shape": list(ds.shape),
        "dtype": ds.dtype,
        "ndim": ds.ndim,
        "is_remote": ds.is_remote,
    }

    try:
        voxel = ds.get_voxel_sizes()
        result["voxel_sizes"] = voxel
    except Exception:
        pass

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 2: discover_datasets
# ---------------------------------------------------------------------------
@mcp.tool()
def discover_datasets(
    path: str,
    pattern: str = "",
    recursive: bool = False,
) -> str:
    """Scan a directory and list all recognized microscopy datasets.

    Recursively discovers image and segmentation layers in Zarr, N5,
    and Neuroglancer precomputed containers.

    Args:
        path: Directory path to scan.
        pattern: Glob pattern to filter files (e.g., "*.tif", "*.nd2").
                 When specified, searches for matching files instead of
                 only scanning immediate subdirectories for containers.
        recursive: Enable recursive subdirectory scanning. Default: False.
    """
    try:
        from tensorswitch_v2.utils.folder_discovery import (
            discover_datasets as _discover,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result = _discover(path.strip(), verbose=False, pattern=pattern, recursive=recursive)

        output = {"path": path, "images": [], "segmentations": []}

        for img in result.all_images:
            output["images"].append({
                "name": img.name,
                "path": img.path,
                "dtype": img.dtype,
                "shape": img.shape,
                "format": img.source_format,
                "num_scales": img.num_scales,
            })
        for seg in result.all_segmentations:
            output["segmentations"].append({
                "name": seg.name,
                "path": seg.path,
                "dtype": seg.dtype,
                "shape": seg.shape,
                "format": seg.source_format,
                "num_scales": seg.num_scales,
            })

        output["summary"] = (
            f"{len(output['images'])} image(s), "
            f"{len(output['segmentations'])} segmentation(s)"
        )
        return json.dumps(output, indent=2)
    except Exception as e:
        logger.error(f"discover_datasets failed: {e}\n{traceback.format_exc()}")
        return f"Error scanning {path}: {e}"


# ---------------------------------------------------------------------------
# Tool 3: convert
# ---------------------------------------------------------------------------
@mcp.tool()
def convert(
    input_path: str,
    output_path: str,
    output_format: str = "zarr3",
    chunk_shape: str = "",
    shard_shape: str = "",
    no_sharding: bool = False,
    voxel_size: str = "",
    voxel_unit: str = "nanometer",
    is_label: bool = False,
    compression: str = "zstd",
    compression_level: int = 5,
    dataset_path: str = "",
    level_path: str = "s0",
    use_bioio: bool = False,
    use_bioformats: bool = False,
    axes_order: str = "",
    force_order: str = "",
    expand_to_5d: bool = False,
    bbox: str = "",
    view_index: int = -1,
    data_type: str = "auto",
    image_key: str = "raw",
    label_key: str = "segmentation",
    no_ome_meta_export: bool = False,
    no_ome_xml_attr: bool = False,
    preset: str = "",
    auto_multiscale: bool = False,
    downsample_method: str = "auto",
    per_level_factors: str = "",
    omero: bool = True,
    no_translation: bool = False,
    expansion_factor: float = 0.0,
    extra_attributes: str = "",
    output_dtype: str = "",
    add_to_existing: bool = False,
    bbox_axes: str = "",
    squeeze_singleton_axes: bool = False,
    input_axes: str = "",
    relabel_axis: str = "",
    output_offset: str = "",
    target_shape: str = "",
) -> str:
    """Convert a microscopy dataset between formats, in-process (datasets up to 2 GB).

    Runs the same code as the command line (`python -m tensorswitch_v2`), so every
    CLI option and default applies. See list_formats for supported inputs (Zarr2/3,
    N5, Precomputed, TIFF, ND2, IMS, HDF5, CZI, NIfTI, MRC, PNG stacks, BioIO and
    Bio-Formats plugins) and outputs (zarr3 with sharding, zarr2, n5).

    The source must state a voxel size for every spatial axis, or you must pass
    voxel_size (X,Y,Z); otherwise the conversion is refused.

    For datasets larger than 2 GB use submit_job to run on the LSF cluster. Data that
    only exists at a URL (zip members, FTP) must be downloaded first with fetch_dataset.

    Args:
        input_path: Path to source dataset.
        output_path: Path for output (e.g., /data/output.zarr).
        output_format: Output format — "zarr3" (default), "zarr2", or "n5".
        chunk_shape: Comma-separated chunk shape (e.g., "64,64,64"). Auto-calculated if empty.
        shard_shape: Comma-separated shard shape for zarr3 (e.g., "512,512,512"). Auto-calculated if empty.
        no_sharding: Disable sharding for zarr3 output. Default: False (sharding enabled).
        voxel_size: Comma-separated voxel sizes in X,Y,Z order (e.g., "6,6,29"). Required if source lacks metadata.
        voxel_unit: Unit for voxel sizes — "nanometer", "micrometer", or "millimeter".
        is_label: Set True for segmentation/label data (uses mode downsampling, adds label metadata).
        compression: Compression codec — "zstd" (default), "gzip", or "none".
        compression_level: Compression level (1-22 for zstd, 1-9 for gzip).
        dataset_path: Path within container file (e.g., "main" for HDF5 dataset name).
        level_path: Level subdirectory name in output (default: "s0").
        use_bioio: Force BIOIO adapter (Tier 3) instead of auto-detected Tier 2 reader.
        use_bioformats: Force Bio-Formats reader (Tier 4, Java-backed) for 150+ formats.
        axes_order: Override output spatial axis order (e.g., "xyz", "zyx"). Default: preserve source order.
        force_order: Force output memory order — "c" for C-order (row-major),
                     "f" for F-order (column-major), or "" for auto-detection.
        expand_to_5d: Force 5D TCZYX expansion.
        bbox: Bounding box for subvolume extraction: origin then size, one pair per source axis in source order
            ("origin_z,origin_y,origin_x,size_z,size_y,size_x" for ZYX; any number of axes, see bbox_axes).
        view_index: CZI view index (-1 = all views as 5D VCZYX).
        data_type: Data type for output structure — "auto", "image", or "labels".
        image_key: Name for image group in output (default: "raw").
        label_key: Name for label image in output (default: "segmentation").
        no_ome_meta_export: Disable writing OME/METADATA.ome.xml file.
        no_ome_xml_attr: Do not embed OME/CZI XML in zarr.json/.zattrs.
        preset: Preset configuration — "webknossos" (chunk 32x32x32, shard 1024x1024x1024).
                          "paintera" (n5, xyz axis order, gzip, chunk 64x64x64; or zarr2 with zyx).
                          "miaai" (alias "mia_lmvd"; zarr3, chunk 128^3, shard 512^3, zstd-5, C-order).
        auto_multiscale: Generate full multiscale pyramid after conversion. Default: False.
        downsample_method: Downsampling method for pyramid — "auto", "mean", "mode", etc. Used with auto_multiscale.
        per_level_factors: Custom per-level factors, semicolon-separated (e.g., "1,2,2;1,2,2"). Used with auto_multiscale.
        omero: Include structured omero channel metadata for visualization tools (default: True).
        no_translation: Disable translation transforms in OME-NGFF multiscale metadata.
        expansion_factor: Expansion microscopy factor (e.g. 4). With preset "miaai" the outer
                          coordinateTransformations become 1/factor on the spatial axes. 0 = not expansion data.
        extra_attributes: JSON file path, or inline JSON object, of extra attributes to add to the zarr.json of
                          the group this call writes (raw/ or labels/<name>/). "ome" and "_software" cannot be set.
        output_dtype: Output dtype override (e.g., "uint8", "int16", "uint16"). Empty = preserve source dtype.
        add_to_existing: Add data to existing container without destroying it.
            Safe write applies to the subgroup (e.g., labels/) not the container root.
        bbox_axes: Which source axes bbox refers to, comma-separated indices (e.g. "2,3,4" for z,y,x of a 5D t,c,z,y,x source).
        squeeze_singleton_axes: Drop length-1 axes (e.g. t, c) from the output. Needs known axis identity.
        input_axes: Name every source axis, one letter per axis in the reader's order, slowest to fastest
                    (e.g. "zyxc" for a TIFF read as i,y,x,s; "zyxs" keeps samples per pixel as their own axis
                    `s`; "yxz" for z slices stored as samples). Letters t, c, s, z, y, x. Only renames; spatial
                    axis order is unchanged. Use the `axes` of a catalog record.
        relabel_axis: Correct a mis-detected source axis, "OLD=NEW" (e.g. "t=z"); several separated by ";".
        output_offset: Sparse label ingest: voxel position of the label inside the existing container,
            comma-separated (e.g. "0,0,128,64,64"). Use with add_to_existing and data_type="labels".
        target_shape: Shape of the target container for output_offset, comma-separated (read from the container if empty).
    """
    params = dict(locals())
    try:
        import contextlib
        import io

        import numpy as np
        from tensorswitch_v2.__main__ import (
            _apply_preset,
            create_reader,
            parse_args,
            parse_bbox,
            parse_bbox_axes,
            run_conversion,
        )
        from tensorswitch_v2.mcp_args import build_argv
        from tensorswitch_v2.utils import get_dtype_name

        input_path = params["input_path"] = input_path.strip()
        output_path = params["output_path"] = output_path.strip()

        # Same path as the CLI: build argv -> real parser -> preset -> run_conversion.
        parse_errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(parse_errors):
                args = parse_args(build_argv(params))
        except SystemExit:
            detail = [l for l in parse_errors.getvalue().strip().splitlines() if l.strip()]
            return json.dumps({"error": "validation_error",
                               "message": detail[-1] if detail else "invalid arguments"}, indent=2)
        _apply_preset(args)

        # Size guard: refuse large datasets (the bbox volume counts when one is given)
        with _quiet_capture() as notes:
            reader = create_reader(args)
            store = reader.get_tensorstore()
        shape = [int(n) for n in store.shape]
        dtype_str = get_dtype_name(store.dtype)
        elements = int(np.prod(shape))
        if args.bbox:
            _, bbox_size = parse_bbox(args.bbox)
            covered = (list(parse_bbox_axes(args.bbox_axes)) if args.bbox_axes
                       else list(range(len(shape) - len(bbox_size), len(shape))))
            if len(covered) == len(bbox_size) and all(0 <= i < len(shape) for i in covered):
                elements = int(np.prod([n for i, n in enumerate(shape) if i not in covered])
                               * np.prod(bbox_size))
        effective_dtype = args.dtype or dtype_str
        dataset_size_gb = (elements * np.dtype(effective_dtype).itemsize) / (1024**3)

        if dataset_size_gb > MCP_CONVERT_MAX_GB:
            return json.dumps({
                "error": "dataset_too_large",
                "dataset_size_gb": round(dataset_size_gb, 2),
                "threshold_gb": MCP_CONVERT_MAX_GB,
                "shape": shape,
                "dtype": dtype_str,
                "recommendation": (
                    f"Dataset is {dataset_size_gb:.1f} GB, exceeding the "
                    f"{MCP_CONVERT_MAX_GB} GB MCP threshold. Use the submit_job "
                    f"tool to run this as an LSF cluster job, or use the CLI: "
                    f"python -m tensorswitch_v2 -i '{input_path}' -o '{output_path}' "
                    f"--submit -P <project>"
                ),
            }, indent=2)

        # Suppress stdout - MCP uses stdio transport (stdout = JSON-RPC)
        try:
            with _quiet_capture() as run_notes:
                info = run_conversion(args) or {}
            notes.extend(n for n in run_notes if n not in notes)
        except (FileNotFoundError, ValueError) as e:
            if str(e).startswith("--add-to-existing"):
                return json.dumps({"error": str(e)}, indent=2)
            raise

        response = {
            "status": "success",
            "input": input_path,
            "output": output_path,
            "format": args.output_format,
            "dataset_size_gb": round(dataset_size_gb, 2),
            "chunks_processed": info.get("chunks_processed", "unknown"),
            "time_seconds": round(info.get("elapsed_seconds") or 0, 1),
        }
        if args.auto_multiscale:
            plan = info.get("pyramid_plan")
            response["auto_multiscale"] = True
            response["pyramid_s0"] = info.get("pyramid_s0")
            if plan and isinstance(plan, dict):
                response["pyramid_levels"] = plan.get("num_levels", 0)
                response["pyramid_info"] = [
                    {"level": f"s{lv['level']}", "factors": lv["cumulative_factor"], "shape": lv["predicted_shape"]}
                    for lv in plan.get("levels", [])
                ]
        notes = _relevant(notes, args.voxel_size)
        if notes:
            response["warnings"] = notes
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"convert failed: {e}\n{traceback.format_exc()}")
        return f"Error converting {input_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 4: generate_pyramid
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_pyramid(
    s0_path: str,
    downsample_method: str = "auto",
    compression: str = "zstd",
    compression_level: int = 5,
    no_translation: bool = False,
    per_level_factors: str = "",
) -> str:
    """Generate a multiscale pyramid from a base-resolution dataset (s0).

    Automatically calculates the number of levels and per-level downsampling
    factors, handling anisotropic voxel sizes (e.g., downsampling XY first
    for ssTEM data).

    The pyramid is written as sibling directories (s1, s2, ...) next to s0,
    and OME-NGFF multiscale metadata is updated.

    Args:
        s0_path: Path to base-level array (e.g., /data/volume.zarr/raw/s0).
        downsample_method: "auto" (detects from data type), "mean" (intensity),
                          "mode" (labels/segmentation), "median", "stride", "min", "max".
        compression: Compression codec for pyramid levels.
        compression_level: Compression level for pyramid levels.
        no_translation: Disable translation transforms in OME-NGFF multiscale metadata.
        per_level_factors: Custom per-level factors, semicolon-separated (e.g., "1,2,2;1,2,2;2,2,2").
                          Overrides automatic factor calculation.
    """
    try:
        import contextlib
        import io

        from tensorswitch_v2.core.pyramid import PyramidPlanner
        from tensorswitch_v2.utils.pyramid_utils import resolve_downsample_method

        s0_path = s0_path.strip()
        downsample_method = resolve_downsample_method(downsample_method, s0_path)
        planner = PyramidPlanner(
            s0_path,
            include_translation=not no_translation,
            downsample_method=downsample_method,
        )

        custom_factors = None
        if per_level_factors:
            custom_factors = [
                [int(x) for x in level.split(",")]
                for level in per_level_factors.split(";")
            ]

        plan = planner.calculate_pyramid_plan(custom_per_level_factors=custom_factors)

        # Run locally (suppress stdout — MCP uses stdio transport)
        with contextlib.redirect_stdout(io.StringIO()):
            planner.precreate_all_levels(plan, use_shard=True, verbose=False)

            from tensorswitch_v2.core.downsampler import downsample_level
            from tensorswitch_v2.utils.metadata_utils import (
                detect_level_format, get_level_name,
            )

            parent_dir = str(Path(s0_path).parent)
            prefix = detect_level_format(parent_dir)
            levels_created = []

            # Chained downsampling: each level reads from previous level
            for level_info in plan["levels"]:
                level_num = level_info["level"]
                per_level_factor = level_info["per_level_factor"]
                cumulative_factors = level_info["cumulative_factor"]

                # Source is previous level (chained)
                source_level = level_num - 1
                source_path = (
                    s0_path if source_level == 0
                    else os.path.join(parent_dir, get_level_name(source_level, prefix))
                )

                downsample_level(
                    s0_path=source_path,
                    output_path=parent_dir,
                    target_level=level_num,
                    factors=per_level_factor,
                    use_shard=True,
                    custom_shard_shape=level_info.get("shard_shape"),
                    custom_chunk_shape=level_info.get("chunk_shape"),
                    downsample_method=downsample_method,
                    verbose=False,
                    cumulative_factor_for_metadata=cumulative_factors,
                )
                levels_created.append({
                    "level": f"s{level_num}",
                    "factors": cumulative_factors,
                    "shape": level_info["predicted_shape"],
                })

            # Update metadata
            from tensorswitch_v2.utils.metadata_utils import update_ome_metadata_if_needed

            update_ome_metadata_if_needed(
                parent_dir,
                use_ome_structure=True,
                include_translation=not no_translation,
                downsample_method=downsample_method,
            )

        return json.dumps(
            {
                "status": "success",
                "s0_path": s0_path,
                "levels_created": levels_created,
                "total_levels": len(levels_created) + 1,
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"generate_pyramid failed: {e}\n{traceback.format_exc()}")
        return f"Error generating pyramid for {s0_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 5: list_formats
# ---------------------------------------------------------------------------
@mcp.tool()
def list_formats() -> str:
    """List the input and output formats TensorSwitch supports, and how to get data that is only at a URL.

    Input formats are grouped by reader tier (Tier 1 = native TensorStore, fastest;
    Tier 2 = dedicated readers; Tier 3 = BioIO plugins actually installed here;
    Tier 4 = Bio-Formats, optional and Java-backed). Each file format names the
    reader that auto-detection picks from the extension.
    """
    from importlib.metadata import entry_points

    try:
        bioio_plugins = sorted({ep.name for ep in entry_points(group="bioio.readers")})
    except Exception:
        bioio_plugins = []
    formats = {
        "input_formats": {
            "tier_1_native_tensorstore": [
                {"name": "Zarr3", "extensions": [".zarr"], "reader": "Zarr3Reader", "notes": "With sharding support"},
                {"name": "Zarr2", "extensions": [".zarr"], "reader": "Zarr2Reader", "notes": "Legacy format"},
                {"name": "N5", "extensions": [".n5"], "reader": "N5Reader", "notes": "Java/BigDataViewer format"},
                {"name": "Neuroglancer Precomputed", "extensions": [], "reader": "PrecomputedReader",
                 "notes": "Local or remote (GCS/S3/HTTP)"},
            ],
            "tier_2_dedicated_readers": [
                {"name": "TIFF", "extensions": [".tif", ".tiff"], "reader": "TiffReader",
                 "notes": "Single file or a folder of 2D slices; OME-TIFF and ImageJ metadata"},
                {"name": "ND2", "extensions": [".nd2"], "reader": "ND2Reader", "notes": "Nikon NIS-Elements"},
                {"name": "IMS", "extensions": [".ims"], "reader": "IMSReader", "notes": "Imaris (HDF5-based)"},
                {"name": "HDF5", "extensions": [".h5", ".hdf5"], "reader": "HDF5Reader",
                 "notes": "Needs dataset_path; voxel size read from dataset attributes"},
                {"name": "CZI", "extensions": [".czi"], "reader": "CZIReader", "notes": "Zeiss, multi-view"},
                {"name": "NIfTI", "extensions": [".nii", ".nii.gz"], "reader": "NIfTIReader",
                 "notes": "Header voxel size is often wrong or missing; pass voxel_size"},
                {"name": "MRC / CCP4", "extensions": [".mrc", ".mrcs", ".rec", ".ali", ".st", ".map", ".rec.nad"], "reader": "MRCReader",
                 "notes": "Cryo-ET and EM volumes; header voxel size is angstroms and only trusted when it looks calibrated"},
                {"name": "PNG", "extensions": [".png"], "reader": "PngReader",
                 "notes": "A folder or .zip of 2D slices becomes one volume; PNG has no voxel size, pass voxel_size"},
            ],
            "tier_3_bioio": {
                "installed_plugins": bioio_plugins,
                "notes": "Used for formats without a dedicated reader when a plugin is installed",
            },
            "tier_4_bioformats": {
                "notes": "Optional, Java-backed (needs the bioformats extra / scyjava); enable with use_bioformats=True",
            },
        },
        "output_formats": [
            {"name": "Zarr3", "notes": "Default. OME-NGFF v0.5, sharding, zstd compression"},
            {"name": "Zarr2", "notes": "Legacy. OME-NGFF v0.4, for tools that don't support Zarr3"},
            {"name": "N5", "notes": "For Java tools (BigDataViewer, BigStitcher)"},
        ],
        "presets": ["webknossos", "paintera", "miaai", "mia_lmvd"],
        "remote_sources": [
            "gs:// (Google Cloud Storage), s3:// (Amazon S3), https:// (HTTP/HTTPS): zarr, N5 and precomputed are read in place",
            "any other file or a member of a remote zip: download it first with fetch_dataset, then convert it",
        ],
        "voxel_size_rule": (
            "Every spatial axis needs a real voxel size from the file or from the voxel_size parameter; "
            "conversion is refused otherwise instead of defaulting to 1."
        ),
    }
    return json.dumps(formats, indent=2)


# ---------------------------------------------------------------------------
# Tool 6: estimate_resources
# ---------------------------------------------------------------------------
@mcp.tool()
def estimate_resources(
    input_path: str,
    output_format: str = "zarr3",
    chunk_shape: str = "",
    shard_shape: str = "",
    no_sharding: bool = False,
) -> str:
    """Estimate compute resources needed to convert a dataset.

    Returns memory, wall time, and core requirements for LSF cluster
    submission. Use this before submit_job to preview resource allocation.

    Args:
        input_path: Path to source dataset.
        output_format: Output format — "zarr3" (default), "zarr2", or "n5".
        chunk_shape: Comma-separated chunk shape (e.g., "64,64,64"). Auto-calculated if empty.
        shard_shape: Comma-separated shard shape for zarr3 (e.g., "512,512,512"). Auto-calculated if empty.
        no_sharding: Disable sharding for zarr3. Default: False.
    """
    try:
        import contextlib
        import io

        import numpy as np
        from tensorswitch_v2.api import Readers
        from tensorswitch_v2.utils import get_dtype_name
        from tensorswitch_v2.utils.resource_utils import (
            calculate_job_resources,
            estimate_shard_info,
            is_native_source,
        )

        input_path = input_path.strip()

        # Suppress stdout — MCP uses stdio transport
        with contextlib.redirect_stdout(io.StringIO()):
            reader = Readers.auto_detect(input_path)
        store = reader.get_tensorstore()
        shape = tuple(store.shape)
        dtype_str = get_dtype_name(store.dtype)

        # Extract axes_order from TensorStore domain labels
        axes_order = None
        if hasattr(store, 'domain') and hasattr(store.domain, 'labels'):
            labels = store.domain.labels
            if labels and all(labels):
                axes_order = ['c' if l.lower() == 'channel' else l.lower()
                              for l in labels]

        # Detect native source (TensorStore-backed = fast, file-decoded = slow)
        is_native = is_native_source(input_path)

        cs_str = chunk_shape if chunk_shape else None
        ss_str = shard_shape if shard_shape else None

        memory_gb, wall_time, cores = calculate_job_resources(
            shape=list(shape),
            dtype=dtype_str,
            output_format=output_format,
            chunk_shape_str=cs_str,
            shard_shape_str=ss_str,
            axes_order=axes_order,
            no_sharding=no_sharding,
            is_native_source=is_native,
        )

        # Get shard/chunk info
        cs = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None
        ss = tuple(int(x) for x in shard_shape.split(",")) if shard_shape else None
        est_shape, total_units = estimate_shard_info(
            shape, dtype_str, output_format, cs, ss,
            axes_order=axes_order, no_sharding=no_sharding,
        )

        dtype_bytes = np.dtype(dtype_str).itemsize
        dataset_size_gb = (np.prod(shape) * dtype_bytes) / (1024**3)

        use_sharding = output_format == "zarr3" and not no_sharding
        unit_label = "shards" if use_sharding else "chunks"

        return json.dumps({
            "dataset_size_gb": round(dataset_size_gb, 2),
            "shape": list(shape),
            "dtype": dtype_str,
            "memory_gb": memory_gb,
            "wall_time": wall_time,
            "cores": cores,
            f"estimated_{unit_label[:-1]}_shape": list(est_shape),
            f"total_{unit_label}": total_units,
            "source_type": "native" if is_native else "file-decoded",
            "exceeds_mcp_convert_limit": bool(dataset_size_gb > MCP_CONVERT_MAX_GB),
        }, indent=2)
    except Exception as e:
        logger.error(f"estimate_resources failed: {e}\n{traceback.format_exc()}")
        return f"Error estimating resources for {input_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 7: submit_job
# ---------------------------------------------------------------------------
@mcp.tool()
def submit_job(
    input_path: str,
    output_path: str,
    project: str,
    output_format: str = "zarr3",
    chunk_shape: str = "",
    shard_shape: str = "",
    no_sharding: bool = False,
    compression: str = "zstd",
    compression_level: int = 5,
    voxel_size: str = "",
    voxel_unit: str = "nanometer",
    is_label: bool = False,
    data_type: str = "auto",
    image_key: str = "raw",
    label_key: str = "segmentation",
    dataset_path: str = "",
    use_bioio: bool = False,
    use_bioformats: bool = False,
    view_index: int = -1,
    axes_order: str = "",
    force_order: str = "",
    expand_to_5d: bool = False,
    bbox: str = "",
    level_path: str = "s0",
    memory: int = 0,
    wall_time: str = "",
    cores: int = 0,
    job_group: str = "",
    log_dir: str = "",
    no_ome_meta_export: bool = False,
    no_ome_xml_attr: bool = False,
    auto_multiscale: bool = False,
    downsample_method: str = "auto",
    per_level_factors: str = "",
    preset: str = "",
    omero: bool = True,
    no_translation: bool = False,
    expansion_factor: float = 0.0,
    extra_attributes: str = "",
    output_dtype: str = "",
    add_to_existing: bool = False,
    bbox_axes: str = "",
    squeeze_singleton_axes: bool = False,
    input_axes: str = "",
    relabel_axis: str = "",
    output_offset: str = "",
    target_shape: str = "",
) -> str:
    """Submit a conversion job to the LSF cluster (bsub).

    Submits an asynchronous cluster job that runs the full TensorSwitch
    conversion pipeline. Returns the LSF job ID for monitoring with
    check_job_status. Resources (memory, wall time, cores) are auto-calculated
    from the source dataset if not specified.

    Args:
        input_path: Path to source dataset.
        output_path: Path for output (e.g., /data/output.zarr).
        project: LSF project name for billing (required). Ask the user which project to use.
        output_format: Output format — "zarr3" (default), "zarr2", or "n5".
        chunk_shape: Comma-separated chunk shape (e.g., "64,64,64"). Auto-calculated if empty.
        shard_shape: Comma-separated shard shape for zarr3. Auto-calculated if empty.
        no_sharding: Disable sharding for zarr3. Default: False.
        compression: Compression codec — "zstd" (default), "gzip", or "none".
        compression_level: Compression level (1-22 for zstd, 1-9 for gzip).
        voxel_size: Override voxel sizes, comma-separated X,Y,Z (e.g., "9,9,12").
        voxel_unit: Unit for voxel sizes — "nanometer", "micrometer", or "millimeter".
        is_label: Set True for segmentation/label data.
        data_type: Output data type — "auto", "image", or "labels".
        image_key: Name for image group (default: "raw").
        label_key: Name for label group (default: "segmentation").
        dataset_path: Path within container file (e.g., "main" for HDF5).
        use_bioio: Force BIOIO adapter (Tier 3).
        use_bioformats: Force Bio-Formats reader (Tier 4).
        view_index: CZI view index (-1 = all views).
        axes_order: Override output spatial axis order (e.g., "xyz", "zyx").
        force_order: Force output memory order — "c" for C-order, "f" for F-order, or "" for auto.
        expand_to_5d: Force 5D TCZYX expansion.
        bbox: Bounding box for subvolume: "origin_z,origin_y,origin_x,size_z,size_y,size_x".
        level_path: Level subdirectory name (default: "s0").
        memory: Memory in GB (0 = auto-calculate from source data).
        wall_time: Wall time in H:MM format (empty = auto-calculate).
        cores: Number of cores (0 = auto-calculate).
        job_group: LSF job group path.
        log_dir: Directory for LSF log files (default: output/ next to output path).
        no_ome_meta_export: Disable writing OME/METADATA.ome.xml file.
        no_ome_xml_attr: Do not embed OME/CZI XML in zarr.json/.zattrs.
        auto_multiscale: Generate full pyramid after s0 conversion (submits chained jobs).
        downsample_method: Downsampling method for pyramid — "auto", "mean", "mode", etc.
        per_level_factors: Custom per-level factors, semicolon-separated (e.g., "1,2,2;1,2,2").
        preset: Preset configuration — "webknossos" (chunk 32, shard 1024).
                          "paintera" (n5, xyz axis order, gzip, chunk 64x64x64; or zarr2 with zyx).
                          "miaai" (alias "mia_lmvd"; zarr3, chunk 128^3, shard 512^3, zstd-5, C-order).
        omero: Include structured omero channel metadata for visualization tools.
        no_translation: Disable translation transforms in OME-NGFF multiscale metadata.
        expansion_factor: Expansion microscopy factor (e.g. 4). With preset "miaai" the outer
                          coordinateTransformations become 1/factor on the spatial axes. 0 = not expansion data.
        extra_attributes: JSON file path, or inline JSON object, of extra attributes to add to the zarr.json of
                          the group this call writes (raw/ or labels/<name>/). "ome" and "_software" cannot be set.
        output_dtype: Output dtype override (e.g., "uint8", "int16", "uint16"). Empty = preserve source dtype.
        add_to_existing: Add data to existing container without destroying it.
            Safe write applies to the subgroup (e.g., labels/) not the container root.
        bbox_axes: Which source axes --bbox refers to, as comma-separated indices
            (e.g. "2,3,4" for z,y,x of a 5D t,c,z,y,x source). Needed for N-D bbox.
        squeeze_singleton_axes: Drop length-1 axes (e.g. t, c) from the output. Needs known axis identity.
        input_axes: Name every source axis, one letter per axis in the reader's order, slowest to fastest
                    (e.g. "zyxc" for a TIFF read as i,y,x,s; "zyxs" keeps samples per pixel as their own axis
                    `s`; "yxz" for z slices stored as samples). Letters t, c, s, z, y, x. Only renames; spatial
                    axis order is unchanged. Use the `axes` of a catalog record.
        relabel_axis: Correct a mis-detected source axis, "OLD=NEW" (e.g. "t=z"); several separated by ";".
        output_offset: Sparse label ingest: voxel position where the label is placed in the existing
            container, comma-separated (e.g. "0,0,128,64,64"). Use with add_to_existing and data_type="labels".
        target_shape: Shape of the target container for output_offset, comma-separated (read from the container if empty).
    """
    params = dict(locals())
    try:
        import contextlib
        import io

        from tensorswitch_v2.__main__ import (
            _apply_preset,
            _resolve_conversion_subgroup as _cli_subgroup,
            _submit_dependent_pyramid,
            find_base_level,
            parse_args,
        )
        from tensorswitch_v2.__main__ import submit_job as _cli_submit_job
        from tensorswitch_v2.mcp_args import build_argv

        input_path = params["input_path"] = input_path.strip()
        output_path = params["output_path"] = output_path.strip()

        # Validate paths are on shared storage (LSF nodes can't see /tmp)
        for label, p in [("input_path", input_path), ("output_path", output_path)]:
            resolved = os.path.realpath(p)
            if resolved.startswith("/tmp") or resolved.startswith("/var/tmp"):
                return json.dumps({
                    "error": "local_path",
                    "message": (
                        f"{label} '{p}' is on node-local storage (/tmp). "
                        f"LSF cluster jobs run on different nodes and cannot access "
                        f"local /tmp. Use a shared filesystem path (e.g., /groups/, "
                        f"/nrs/, /nearline/)."
                    ),
                }, indent=2)

        # Same path as the CLI: build argv -> real parser -> preset -> CLI submit.
        # Every CLI option, default and validation therefore applies here too.
        parse_errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(parse_errors):
                args = parse_args(build_argv(params) + ["--submit"])
        except SystemExit:
            detail = [l for l in parse_errors.getvalue().strip().splitlines() if l.strip()]
            return json.dumps({"error": "validation_error",
                               "message": detail[-1] if detail else "invalid arguments"}, indent=2)
        _apply_preset(args)

        # auto_multiscale on an existing dataset (same input and output): pyramid only.
        if args.auto_multiscale:
            is_existing_dataset = False
            try:
                find_base_level(input_path)
                is_existing_dataset = True
            except (ValueError, OSError):
                pass
            if is_existing_dataset and os.path.abspath(input_path) == os.path.abspath(output_path):
                return _submit_pyramid_job(
                    input_path, output_path, project,
                    downsample_method=args.downsample_method,
                    per_level_factors=args.per_level_factors or "",
                    memory=args.memory or 0, wall_time=args.wall_time or "", cores=args.cores or 0,
                    log_dir=args.log_dir or "",
                    include_translation=not args.no_translation,
                    subgroup=_cli_subgroup(args),
                )

        # Suppress stdout - MCP uses stdio transport (stdout = JSON-RPC)
        with _quiet_capture() as notes:
            job_id = _cli_submit_job(args, return_job_id=True)
            coordinator_id = (_submit_dependent_pyramid(args, conversion_job_id=str(job_id))
                              if args.auto_multiscale and job_id else None)
        notes = _relevant(notes, args.voxel_size)

        if args.auto_multiscale:
            return json.dumps({
                "status": "submitted",
                "mode": "convert_and_pyramid",
                "conversion_job_id": str(job_id),
                "coordinator_job_id": coordinator_id,
                "input": input_path,
                "output": output_path,
                "format": args.output_format,
                "project": project,
                **({"warnings": notes} if notes else {}),
                "message": (
                    f"Conversion job {job_id} submitted. "
                    + (f"Pyramid coordinator job {coordinator_id} will start after conversion completes. "
                       if coordinator_id else
                       "The pyramid coordinator could not be submitted; run auto_multiscale on the output afterwards. ")
                    + "Use check_job_status to monitor."
                ),
            }, indent=2)

        return json.dumps({
            "status": "submitted",
            "job_id": job_id,
            "input": input_path,
            "output": output_path,
            "format": args.output_format,
            "project": project,
            **({"warnings": notes} if notes else {}),
            "message": f"Job {job_id} submitted. Use check_job_status to monitor.",
        }, indent=2)

    except FileNotFoundError as e:
        err_msg = str(e)
        if "bsub" in err_msg or "No such file or directory: 'bsub'" in err_msg:
            return json.dumps({
                "error": "bsub_not_found",
                "message": "bsub command not found. submit_job requires an LSF cluster environment.",
            }, indent=2)
        return json.dumps({
            "error": "file_not_found",
            "message": err_msg,
        }, indent=2)
    except ValueError as e:
        return json.dumps({"error": "validation_error", "message": str(e)}, indent=2)
    except RuntimeError as e:
        return json.dumps({"error": "submission_failed", "message": str(e)}, indent=2)
    except Exception as e:
        logger.error(f"submit_job failed: {e}\n{traceback.format_exc()}")
        return f"Error submitting job: {e}"


def _submit_pyramid_job(
    input_path: str, output_path: str, project: str,
    downsample_method: str = "auto", per_level_factors: str = "",
    memory: int = 0, wall_time: str = "", cores: int = 0,
    log_dir: str = "",
    include_translation: bool = True,
    subgroup: str = None,
) -> str:
    """Submit pyramid generation as chained LSF jobs."""
    import contextlib
    import io

    from tensorswitch_v2.__main__ import find_base_level
    from tensorswitch_v2.core.pyramid import create_pyramid_parallel

    # Narrow to the specific subgroup if provided (e.g. 'raw' or
    # 'labels/segmentation') so find_base_level targets the right levels.
    effective_input = input_path
    if subgroup:
        candidate = os.path.join(input_path, subgroup)
        if os.path.isdir(candidate):
            effective_input = candidate

    # Determine s0 path — use find_base_level for full detection
    # (handles raw/s0, labels/segmentation/s0, OME-NGFF metadata, N5, etc.)
    try:
        s0_path, _ = find_base_level(effective_input)
    except ValueError:
        return json.dumps({
            "error": "s0_not_found",
            "message": (
                f"Cannot find base resolution level in '{input_path}'. "
                f"auto_multiscale requires an already-converted dataset "
                f"(with raw/s0, labels/segmentation/s0, or s0 subdirectory). "
                f"First submit a conversion job without auto_multiscale, wait for it "
                f"to complete, then submit a second job with auto_multiscale=True "
                f"pointing at the output."
            ),
        }, indent=2)

    # Validate s0 array metadata exists (zarr.json, .zarray, or attributes.json for N5)
    s0_zarr_json = os.path.join(s0_path, "zarr.json")
    s0_zarray = os.path.join(s0_path, ".zarray")
    s0_n5_attrs = os.path.join(s0_path, "attributes.json")
    if not (os.path.isfile(s0_zarr_json) or os.path.isfile(s0_zarray) or os.path.isfile(s0_n5_attrs)):
        return json.dumps({
            "error": "s0_not_found",
            "message": (
                f"Found s0 directory at '{s0_path}' but no array metadata "
                f"(zarr.json, .zarray, or attributes.json). The dataset may be "
                f"incomplete or corrupted."
            ),
        }, indent=2)

    # Parse per_level_factors if provided
    custom_factors = None
    if per_level_factors:
        custom_factors = [
            [int(x) for x in level.split(",")]
            for level in per_level_factors.split(";")
        ]

    with contextlib.redirect_stdout(io.StringIO()):
        result = create_pyramid_parallel(
            s0_path=s0_path,
            project=project,
            memory=memory if memory > 0 else None,
            wall_time=wall_time if wall_time else None,
            cores=cores if cores > 0 else None,
            downsample_method=downsample_method,
            custom_per_level_factors=custom_factors,
            log_dir=log_dir if log_dir else None,
            include_translation=include_translation,
        )

    return json.dumps({
        "status": "submitted",
        "mode": "auto_multiscale",
        "s0_path": s0_path,
        "coordinator_job_id": result.get("coordinator_job_id"),
        "num_levels": len(result.get("pyramid_plan", {}).get("levels", [])),
        "project": project,
        "message": (
            f"Pyramid coordinator job {result.get('coordinator_job_id')} submitted. "
            f"It will submit and chain individual level jobs automatically. "
            f"Use check_job_status to monitor the coordinator."
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 8: upsample_to_isotropic
# ---------------------------------------------------------------------------
@mcp.tool()
def upsample_to_isotropic(
    input_path: str,
    output_path: str,
    target_voxel_size: float = 0,
    upsample_method: str = "auto",
    is_label: bool = False,
    output_format: str = "auto",
    no_sharding: bool = False,
    compression: str = "zstd",
    compression_level: int = 5,
    auto_pyramid: bool = True,
    downsample_method: str = "auto",
    per_level_factors: str = "",
    no_translation: bool = False,
) -> str:
    """Upsample anisotropic data to isotropic resolution.

    Resamples data along anisotropic axes (e.g., Z in ssTEM/FIB-SEM) to
    match the highest-resolution axis, producing isotropic voxels. Uses
    scipy.ndimage.zoom for interpolation, TensorStore for output writes.

    For datasets larger than 2 GB, use the CLI instead.

    Args:
        input_path: Path to source s0 array (e.g., /data/volume.zarr/img/s0).
        output_path: Path for output (e.g., /data/output.zarr/img/s0).
        target_voxel_size: Target isotropic voxel size in nm. 0 = auto
            (use smallest source voxel size, i.e. highest resolution axis).
        upsample_method: Interpolation method — "auto" (trilinear for images,
            nearest for labels), "trilinear", "nearest", or "cubic".
        is_label: Set True for segmentation/label data (forces nearest-neighbor).
        output_format: Output format — "auto" (match source), "zarr2", "zarr3".
        no_sharding: Disable sharding for zarr3 output. Default: False.
        compression: Compression codec — "zstd" (default), "gzip", or "none".
        compression_level: Compression level (1-22 for zstd, 1-9 for gzip).
        auto_pyramid: Generate isotropic multiscale pyramid after upsampling.
        downsample_method: Downsampling method for pyramid — "auto", "mean",
            "mode", etc. Used with auto_pyramid.
        per_level_factors: Custom per-level factors, semicolon-separated
            (e.g., "1,2,2;1,2,2"). Used with auto_pyramid.
        no_translation: Disable translation transforms in OME-NGFF metadata.
    """
    try:
        import contextlib
        import io

        import numpy as np
        from tensorswitch_v2.core.upsampler import (
            upsample_to_isotropic as _upsample,
        )

        input_path = input_path.strip()
        output_path = output_path.strip()

        # Size guard
        src = ts.open(get_zarr_store_spec(input_path), open=True).result()
        shape = tuple(src.shape)
        dtype_str = src.dtype.numpy_dtype.name
        dataset_size_gb = (np.prod(shape) * np.dtype(dtype_str).itemsize) / (1024**3)

        if dataset_size_gb > MCP_CONVERT_MAX_GB:
            return json.dumps({
                "error": "dataset_too_large",
                "dataset_size_gb": round(dataset_size_gb, 2),
                "threshold_gb": MCP_CONVERT_MAX_GB,
                "shape": list(shape),
                "dtype": dtype_str,
                "recommendation": (
                    f"Dataset is {dataset_size_gb:.1f} GB, exceeding the "
                    f"{MCP_CONVERT_MAX_GB} GB MCP threshold. Use the CLI: "
                    f"python -m tensorswitch_v2 --upsample "
                    f"-i '{input_path}' -o '{output_path}'"
                ),
            }, indent=2)

        # Safe write: write to .tmp, rename on completion
        final_output = output_path
        # Determine the container root (parent of the group containing s0)
        # e.g., /data/out.zarr/img/s0 → container is /data/out.zarr
        # We apply .tmp at the container level
        s0_name = os.path.basename(output_path)
        group_path = os.path.dirname(output_path)
        container_path = os.path.dirname(group_path) if group_path else output_path

        tmp_container = container_path.rstrip('/\\') + '.tmp'
        tmp_group = os.path.join(tmp_container, os.path.basename(group_path)) if group_path != output_path else tmp_container
        tmp_s0 = os.path.join(tmp_group, s0_name) if group_path != output_path else tmp_container

        if os.path.exists(tmp_container):
            shutil.rmtree(tmp_container)

        target = target_voxel_size if target_voxel_size > 0 else None

        with contextlib.redirect_stdout(io.StringIO()):
            stats = _upsample(
                input_path=input_path,
                output_path=tmp_s0,
                target_voxel_size=target,
                upsample_method=upsample_method,
                is_label=is_label,
                verbose=False,
                output_format=output_format,
                no_sharding=no_sharding,
                compression=compression,
                compression_level=compression_level,
            )

        response = {
            "status": "success",
            "input": input_path,
            "output": final_output,
            "input_shape": stats["input_shape"],
            "output_shape": stats["output_shape"],
            "zoom_factors": [round(f, 4) for f in stats["zoom_factors"]],
            "upsample_method": stats["upsample_method"],
            "time_seconds": round(stats["elapsed_time"], 1),
        }

        # Auto-pyramid after upsampling
        if auto_pyramid:
            from tensorswitch_v2.__main__ import run_local_pyramid
            from tensorswitch_v2.utils.pyramid_utils import resolve_downsample_method

            root_path = os.path.dirname(tmp_s0)
            resolved_method = resolve_downsample_method(downsample_method, tmp_s0)

            custom_factors = None
            if per_level_factors:
                custom_factors = [
                    [int(x) for x in level.split(",")]
                    for level in per_level_factors.split(";")
                ]

            with contextlib.redirect_stdout(io.StringIO()):
                plan = run_local_pyramid(
                    tmp_s0, root_path,
                    downsample_method=resolved_method,
                    custom_per_level_factors=custom_factors,
                    include_translation=not no_translation,
                    verbose=False,
                )
            response["auto_pyramid"] = True
            response["pyramid_s0"] = tmp_s0
            if plan and isinstance(plan, dict):
                response["pyramid_levels"] = plan.get("num_levels", 0)
                response["pyramid_info"] = [
                    {
                        "level": f"s{lv['level']}",
                        "factors": lv["cumulative_factor"],
                        "shape": lv["predicted_shape"],
                    }
                    for lv in plan.get("levels", [])
                ]

        # Safe write: rename .tmp → final path
        if os.path.exists(tmp_container):
            if os.path.exists(container_path):
                shutil.rmtree(container_path)
            os.rename(tmp_container, container_path)
        response["output"] = final_output

        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"upsample_to_isotropic failed: {e}\n{traceback.format_exc()}")
        return f"Error upsampling {input_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 9: check_job_status
# ---------------------------------------------------------------------------
@mcp.tool()
def check_job_status(job_id: str, follow_dependents: bool = True, log_lines: int = 15) -> str:
    """Report on LSF jobs: status, exit code, resources, log tails, and whether it looks right.

    "DONE" only means the command exited 0. A job that never started the program is DONE
    too, so each job also gets flags (for example "suspicious: reported DONE but used only
    21 MB and 0.1 s of CPU") and the tail of its stdout and stderr logs.

    Jobs that wait for it are included: the pyramid coordinator of a conversion and the
    level jobs after it. `chain.state` is "running" while any of them is queued or running,
    then "failed", "suspicious" or "done". A chain is only finished when the state is no
    longer "running". Verify the output with verify_output once it is.

    Args:
        job_id: LSF job ID (e.g., "12345") or several separated by commas or spaces.
        follow_dependents: Also report the jobs that depend on it (default True).
        log_lines: How many lines of each log to return (default 15).
    """
    import re

    try:
        from tensorswitch_v2.utils.job_status import job_report

        ids = [j for j in re.split(r"[,\s]+", job_id.strip()) if j]
        if not ids:
            return json.dumps({"error": "validation_error", "message": "no job id given"}, indent=2)
        report = job_report(ids, follow_dependents=follow_dependents, log_lines=log_lines)
        jobs = report["jobs"]
        return json.dumps(jobs[0] if len(jobs) == 1 else report, indent=2)
    except FileNotFoundError:
        return json.dumps({
            "error": "bjobs_not_found",
            "message": "bjobs command not found. Requires LSF cluster environment.",
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "timeout", "message": "bjobs timed out after 60 seconds"}, indent=2)
    except Exception as e:
        logger.error(f"check_job_status failed: {e}\n{traceback.format_exc()}")
        return f"Error checking job status: {e}"


# ---------------------------------------------------------------------------
# Tool: fetch_dataset
# ---------------------------------------------------------------------------
@mcp.tool()
def fetch_dataset(spec: str, dest_dir: str, max_gb: float = 2.0, background: bool = False) -> str:
    """Download a file (or one member of a remote zip) so it can be converted.

    Use this for data that only exists at a URL, e.g. a Zenodo or EBI link from a
    dataset catalog record. Then pass the returned path to inspect_dataset/convert.

    A download that is cut off is retried and continues from the bytes already saved
    (when the server supports it). For large files or slow servers use background=True.

    Args:
        spec: A URL (http, https, ftp, s3), or "<zip url>::<path inside zip>" to
              pull one member out of a remote zip without downloading the whole zip.
        dest_dir: Folder to save into (created if needed). It must be visible to
                  whatever will read it next (e.g. a batch job), so avoid /tmp then.
        max_gb: Refuse anything larger than this. Default 2 GB, the in-process
                limit; larger values need background=True.
        background: Download in a detached process instead of waiting. The call
                    returns at once with status "started"; call it again with the same
                    arguments to see progress ("downloading"), the result ("success")
                    or the error ("failed"; the next call tries again and resumes).
    """
    from tensorswitch_v2.utils import fetch as _fetch

    try:
        spec, dest_dir = spec.strip(), dest_dir.strip()
        url, member = _fetch.parse_spec(spec)
        _fetch.check_host(url)
        if background:
            state = _fetch.background_fetch(spec, dest_dir, int(max_gb * 1024 ** 3))
            state["status"] = {"done": "success"}.get(state["state"], state["state"])
            del state["state"]
            return json.dumps(state, indent=2)
        if max_gb > MCP_CONVERT_MAX_GB:
            return json.dumps({
                "error": "too_large_for_mcp",
                "message": (
                    f"max_gb={max_gb} exceeds the {MCP_CONVERT_MAX_GB} GB in-process limit. Call again with "
                    f"background=True (the download continues in a detached process and resumes if interrupted), "
                    f"or run it yourself:"
                ),
                "command": f"python -m tensorswitch_v2.utils.fetch '{spec}' '{dest_dir}' --max-gb {max_gb:g}",
            }, indent=2)
        result = _fetch.fetch(spec, dest_dir, int(max_gb * 1024 ** 3))
        result["status"] = "success"
        if os.path.realpath(result["path"]).startswith(("/tmp", "/var/tmp")):
            result["warning"] = "saved under /tmp, which batch or cluster nodes usually cannot see"
        return json.dumps(result, indent=2)
    except _fetch.FetchError as e:
        return json.dumps({"error": "fetch_refused", "message": str(e)}, indent=2)
    except Exception as e:
        logger.error(f"fetch_dataset failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"error": "fetch_failed", "message": str(e)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: plan_conversion_from_yaml
# ---------------------------------------------------------------------------
@mcp.tool()
def plan_conversion_from_yaml(record: str, output_dir: str, project: str = "", whole_dataset: bool = False) -> str:
    """Plan the conversion of a dataset-catalog record (mia-agentic-search YAML).

    Reads the record and returns the exact fetch_dataset / convert / submit_job
    calls to run, in order, plus warnings for everything the record does not
    settle (missing voxel size, unsupported format, no concrete file URL...).
    Nothing is downloaded or converted: review the plan, then run the steps.

    Only 3D and 3D+t records are planned. By default a plan covers the record's
    sample unit (technical.sample.urls), one raw plus its labels. With
    whole_dataset=True it covers every file of the record's zip download (listed
    over the network), one container per sample with raw and labels paired by
    folder and file name; files without a partner are reported in "unpaired",
    and datasets over 50 GB are skipped and listed.

    Args:
        record: Path to a record .yaml, a GitHub URL of one, or a catalog record id.
        output_dir: Folder for the converted containers. Downloads are staged in
                    <output_dir>/source/ and can be deleted after checking the result.
        project: LSF project, used in the plan when the sample is over 2 GB.
        whole_dataset: Plan every file of the dataset instead of the sample unit.
    """
    from tensorswitch_v2.utils import record_planner

    try:
        loaded = record_planner.load_record(record.strip())
        planner = record_planner.plan_dataset if whole_dataset else record_planner.plan_record
        plan = planner(loaded, output_dir.strip(), project.strip() or None)
        return json.dumps(plan, indent=2)
    except record_planner.RecordError as e:
        return json.dumps({"error": "bad_record", "message": str(e)}, indent=2)
    except Exception as e:
        logger.error(f"plan_conversion_from_yaml failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"error": "plan_failed", "message": str(e)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: verify_output
# ---------------------------------------------------------------------------
@mcp.tool()
def verify_output(
    output_path: str,
    source_path: str = "",
    voxel_size: str = "",
    labels: str = "",
    bbox: str = "",
    bbox_axes: str = "",
    dataset_path: str = "",
    output_dtype: str = "",
    group: str = "",
    image_key: str = "raw",
    samples: int = 5,
    input_axes: str = "",
    label_input_axes: str = "",
) -> str:
    """Check a converted OME-Zarr container, above all against the data it came from.

    Use after convert or after a submit_job chain has finished. Each check is pass, fail
    or unverified (it could not run, with the reason). The overall result is fail if any
    check failed, unverified if none failed but a check could not run, else pass; an
    unverified result is NOT a pass. Nothing is deleted or changed except that a small
    verification.json is written into the container.

    Checks: structure (image and labels present, no .tmp leftovers), pyramid levels
    (all exist, consistent shape and dtype), voxel size in nm vs what you expected,
    data not constant, identity with the source (the whole array when it is 1 GB or
    less, otherwise slices spread through it; catches transposed, shifted or dropped
    data), and the Unix group of the files.

    Args:
        output_path: The converted .zarr container.
        source_path: The file the image was converted from. Without it the identity check is unverified.
        voxel_size: Expected voxel size "X,Y,Z" in nm.
        labels: Label arrays to check, "name=source file;name2=source file2" (HDF5: "name=file.h5::dataset").
        bbox: The bbox used for the conversion, so the right region of the source is compared.
        bbox_axes: The bbox_axes used for the conversion.
        dataset_path: Dataset inside an HDF5 source.
        output_dtype: Set if the conversion cast the values (identity is then reported unverified).
        group: Unix group the files should belong to.
        image_key: Name of the image group (default "raw"); empty if the container holds labels only.
        samples: Slices compared when the array is too big to compare whole.
        input_axes: The input_axes used for the image conversion (e.g. "zyxc"), so source and output axes are
                    matched by name. Axes are matched by name anyway when the source states them.
        label_input_axes: The same for labels, "name=axes;name2=axes2" (e.g. "segmentation=yxz").
    """
    import contextlib
    import io

    from tensorswitch_v2.utils.verify import verify_output as _verify

    try:
        label_map = {}
        for item in filter(None, (t.strip() for t in labels.split(";"))):
            name, _, path = item.partition("=")
            if not path:
                return json.dumps({"error": "validation_error",
                                   "message": f"labels must look like 'name=source file;...', got {item!r}"}, indent=2)
            label_map[name.strip()] = path.strip()
        expected = {k: v for k, v in {
            "voxel_size": voxel_size, "labels": label_map, "bbox": bbox, "bbox_axes": bbox_axes,
            "dataset_path": dataset_path, "output_dtype": output_dtype, "group": group}.items() if v}
        label_axes = {}
        for item in filter(None, (t.strip() for t in label_input_axes.split(";"))):
            name, _, axes = item.partition("=")
            if not axes:
                return json.dumps({"error": "validation_error",
                                   "message": f"label_input_axes must look like 'name=axes;...', got {item!r}"}, indent=2)
            label_axes[name.strip()] = axes.strip()
        if input_axes.strip():
            expected["input_axes"] = input_axes.strip()
        if label_axes:
            expected["label_input_axes"] = label_axes
        expected["image_key"] = image_key.strip()        # empty: the container has labels only
        with contextlib.redirect_stdout(io.StringIO()):
            report = _verify(output_path.strip(), source_path.strip() or None, expected, samples=samples)
        return json.dumps(report, indent=2)
    except Exception as e:
        logger.error(f"verify_output failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"error": "verify_failed", "message": str(e)}, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorSwitch MCP Server")
    parser.add_argument(
        "--transport", default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind HTTP server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for HTTP server (default: 8000)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("Starting TensorSwitch MCP server (stdio)")
        mcp.run(transport="stdio")
    else:
        logger.info(
            f"Starting TensorSwitch MCP server ({args.transport}) "
            f"on {args.host}:{args.port}"
        )
        mcp.run(transport=args.transport, host=args.host, port=args.port)
