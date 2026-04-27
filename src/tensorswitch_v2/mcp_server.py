"""
TensorSwitch MCP Server — exposes TensorSwitch v2 as tools for Claude and other LLM agents.

Usage:
    # Run directly
    pixi run python -m tensorswitch_v2.mcp_server

    # Add to Claude Code
    claude mcp add --transport stdio tensorswitch -- pixi run python -m tensorswitch_v2.mcp_server
"""

import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import tensorstore as ts

from tensorswitch_v2.readers.base import is_remote_path
from tensorswitch_v2.utils.tensorstore_utils import get_zarr_store_spec

from mcp.server.fastmcp import FastMCP

# Logging to stderr (required for stdio transport — stdout is JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tensorswitch-mcp")

mcp = FastMCP("tensorswitch")

# Size guard: refuse in-process conversion for datasets larger than this
MCP_CONVERT_MAX_GB = 2


def _resolve_conversion_subgroup(
    output_format: str,
    data_type: str,
    is_label: bool,
    image_key: str = "raw",
    label_key: str = "segmentation",
):
    """Return the OME-NGFF subgroup the conversion writes to, or None.

    Mirrors __main__._resolve_conversion_subgroup() but takes explicit params
    instead of an argparse Namespace.

    Returns:
        "labels/<label_key>" for label output, "<image_key>" for image output,
        or None when the subgroup cannot be determined (N5, or data_type 'auto'
        without is_label).
    """
    if output_format not in ("zarr2", "zarr3"):
        return None
    if is_label or data_type == "labels":
        return f"labels/{label_key}"
    if data_type == "image":
        return image_key
    return None


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
        result["unit"] = ms.get("axes", [{}])[0].get("unit", "unknown")
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
# Helper: create reader based on parameters
# ---------------------------------------------------------------------------
def _create_reader(input_path: str, dataset_path: str = "",
                   use_bioio: bool = False, use_bioformats: bool = False,
                   view_index: int = -1):
    """Create the appropriate reader based on parameters."""
    from tensorswitch_v2.api import Readers

    if use_bioformats:
        return Readers.bioformats(input_path)
    if use_bioio:
        return Readers.bioio(input_path)

    if dataset_path:
        ext = Path(input_path).suffix.lower()
        if ext in (".h5", ".hdf5", ".hdf", ".he5"):
            return Readers.hdf5(input_path, dataset_path=dataset_path)
        elif ext in (".n5",) or (not is_remote_path(input_path) and os.path.isfile(
            os.path.join(input_path, "attributes.json")
        )):
            return Readers.n5(input_path, dataset_path=dataset_path)

    reader = Readers.auto_detect(input_path)

    # Handle CZI view_index
    if view_index >= 0 and hasattr(reader, 'set_view_index'):
        reader.set_view_index(view_index)

    return reader


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
) -> str:
    """Convert a microscopy dataset between formats.

    Supported inputs: HDF5, TIFF, ND2, IMS, CZI, Zarr2, Zarr3, N5,
    Neuroglancer precomputed, and 150+ formats via Bio-Formats.
    Supported outputs: zarr3 (with sharding), zarr2, n5.

    For datasets larger than 2 GB, use submit_job instead to run on the
    LSF cluster asynchronously.

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
        bbox: Bounding box for subvolume extraction: "origin_z,origin_y,origin_x,size_z,size_y,size_x".
        view_index: CZI view index (-1 = all views as 5D VCZYX).
        data_type: Data type for output structure — "auto", "image", or "labels".
        image_key: Name for image group in output (default: "raw").
        label_key: Name for label image in output (default: "segmentation").
        no_ome_meta_export: Disable writing OME/METADATA.ome.xml file.
        no_ome_xml_attr: Do not embed OME/CZI XML in zarr.json/.zattrs.
        preset: Preset configuration — "webknossos" (chunk 32x32x32, shard 1024x1024x1024).
                          "paintera" (n5, xyz axis order, gzip, chunk 64x64x64; or zarr2 with zyx).
        auto_multiscale: Generate full multiscale pyramid after conversion. Default: False.
        downsample_method: Downsampling method for pyramid — "auto", "mean", "mode", etc. Used with auto_multiscale.
        per_level_factors: Custom per-level factors, semicolon-separated (e.g., "1,2,2;1,2,2"). Used with auto_multiscale.
        omero: Include structured omero channel metadata for visualization tools (default: True).
        no_translation: Disable translation transforms in OME-NGFF multiscale metadata.
    """
    try:
        import contextlib
        import io

        import numpy as np
        from tensorswitch_v2.api import Readers, Writers
        from tensorswitch_v2.core.converter import DistributedConverter
        from tensorswitch_v2.utils import get_dtype_name

        input_path = input_path.strip()
        output_path = output_path.strip()

        # Apply preset
        if preset == "webknossos":
            if not chunk_shape:
                chunk_shape = "32,32,32"
            if not shard_shape:
                shard_shape = "1024,1024,1024"
        elif preset == "paintera":
            if output_format == "zarr3":
                output_format = "n5"
            if not chunk_shape:
                chunk_shape = "64,64,64"
            if compression == "zstd":
                compression = "gzip"
            if not axes_order:
                axes_order = "xyz" if output_format == "n5" else "zyx"

        # Create reader (suppress stdout — MCP uses stdio transport)
        with contextlib.redirect_stdout(io.StringIO()):
            reader = _create_reader(input_path, dataset_path, use_bioio, use_bioformats, view_index)

        # Size guard — refuse large datasets (use bbox size if specified)
        store = reader.get_tensorstore()
        shape = tuple(store.shape)
        dtype_str = get_dtype_name(store.dtype)

        # If bbox is provided, use bbox volume for size check
        if bbox:
            bbox_parts = [int(x) for x in bbox.split(",")]
            if len(bbox_parts) == 6:
                effective_shape = tuple(bbox_parts[3:])  # size_z, size_y, size_x
            else:
                effective_shape = shape
        else:
            effective_shape = shape
        dataset_size_gb = (np.prod(effective_shape) * np.dtype(dtype_str).itemsize) / (1024**3)

        if dataset_size_gb > MCP_CONVERT_MAX_GB:
            return json.dumps({
                "error": "dataset_too_large",
                "dataset_size_gb": round(dataset_size_gb, 2),
                "threshold_gb": MCP_CONVERT_MAX_GB,
                "shape": list(shape),
                "dtype": dtype_str,
                "recommendation": (
                    f"Dataset is {dataset_size_gb:.1f} GB, exceeding the "
                    f"{MCP_CONVERT_MAX_GB} GB MCP threshold. Use the submit_job "
                    f"tool to run this as an LSF cluster job, or use the CLI: "
                    f"python -m tensorswitch_v2 -i '{input_path}' -o '{output_path}' "
                    f"--submit -P <project>"
                ),
            }, indent=2)

        # Resolve data_type
        if data_type == "auto":
            resolved_data_type = "labels" if is_label else "image"
        else:
            resolved_data_type = data_type

        # Safe write: write to .tmp, rename on completion
        final_output = output_path
        tmp_output = output_path.rstrip('/\\') + '.tmp'
        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)
        output_path = tmp_output

        # Create writer
        if output_format == "zarr3":
            writer = Writers.zarr3(
                output_path,
                use_sharding=not no_sharding,
                compression=compression,
                compression_level=compression_level,
                data_type=resolved_data_type,
                level_path=level_path,
                image_key=image_key,
                label_key=label_key,
                include_omero=omero,
            )
        elif output_format == "zarr2":
            writer = Writers.zarr2(
                output_path,
                compression=compression,
                compression_level=compression_level,
                data_type=resolved_data_type,
                level_path=level_path,
                image_key=image_key,
                label_key=label_key,
                include_omero=omero,
            )
        elif output_format == "n5":
            writer = Writers.n5(
                output_path,
                compression=compression,
                compression_level=compression_level,
            )
        else:
            return f"Error: unsupported output format '{output_format}'. Use zarr3, zarr2, or n5."

        # Parse optional shapes
        cs = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None
        ss = tuple(int(x) for x in shard_shape.split(",")) if shard_shape else None

        # Parse voxel size override
        voxel_override = None
        if voxel_size:
            parts = [float(x) for x in voxel_size.split(",")]
            if len(parts) == 3:
                voxel_override = {"x": parts[0], "y": parts[1], "z": parts[2]}

        # Parse bbox
        bbox_parsed = None
        if bbox:
            parts = [int(x) for x in bbox.split(",")]
            if len(parts) == 6:
                bbox_parsed = (tuple(parts[:3]), tuple(parts[3:]))

        # Parse axes_order
        axes_order_list = list(axes_order) if axes_order else None

        # Run conversion (suppress stdout — MCP uses stdio transport)
        converter = DistributedConverter(reader, writer)
        with contextlib.redirect_stdout(io.StringIO()):
            result = converter.convert(
                chunk_shape=cs,
                shard_shape=ss,
                voxel_size_override=voxel_override,
                voxel_unit=voxel_unit if voxel_size else None,
                is_label=is_label,
                expand_to_5d=expand_to_5d,
                bbox=bbox_parsed,
                axes_order_override=axes_order_list,
                force_order=force_order if force_order else None,
                no_ome_meta_export=no_ome_meta_export,
                no_ome_xml_attr=no_ome_xml_attr,
            )

        response = {
            "status": "success",
            "input": input_path,
            "output": output_path,
            "format": output_format,
            "dataset_size_gb": round(dataset_size_gb, 2),
            "chunks_processed": result.get("chunks_processed", "unknown"),
            "time_seconds": round(result.get("elapsed_time", 0), 1),
        }

        # Auto-multiscale: generate pyramid after conversion
        if auto_multiscale:
            from tensorswitch_v2.__main__ import find_base_level, run_local_pyramid
            from tensorswitch_v2.utils.pyramid_utils import resolve_downsample_method

            # Resolve the subgroup the converter just wrote to (e.g. 'raw'
            # or 'labels/segmentation') so find_base_level targets the right
            # levels instead of a stale subgroup from a prior stage.
            subgroup = _resolve_conversion_subgroup(
                output_format, data_type, is_label, image_key, label_key,
            )
            find_target = (
                os.path.join(output_path, subgroup) if subgroup else output_path
            )
            s0_path, _ = find_base_level(find_target)
            root_path = os.path.dirname(s0_path)

            resolved_method = resolve_downsample_method(downsample_method, s0_path)

            custom_factors = None
            if per_level_factors:
                custom_factors = [
                    [int(x) for x in level.split(",")]
                    for level in per_level_factors.split(";")
                ]

            with contextlib.redirect_stdout(io.StringIO()):
                plan = run_local_pyramid(
                    s0_path, root_path,
                    downsample_method=resolved_method,
                    custom_per_level_factors=custom_factors,
                    include_translation=not no_translation,
                    verbose=False,
                )
            response["auto_multiscale"] = True
            response["pyramid_s0"] = s0_path
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
        if os.path.exists(tmp_output):
            if os.path.exists(final_output):
                shutil.rmtree(final_output)
            os.rename(tmp_output, final_output)
        response["output"] = final_output

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
    """List all supported input and output formats for TensorSwitch.

    Returns format names, extensions, and reader tiers (Tier 1 = native
    TensorStore for maximum performance, Tier 4 = Bio-Formats for broadest
    compatibility).
    """
    formats = {
        "input_formats": {
            "tier_1_native_tensorstore": [
                {"name": "Zarr3", "extensions": [".zarr"], "notes": "With sharding support"},
                {"name": "Zarr2", "extensions": [".zarr"], "notes": "Legacy format"},
                {"name": "N5", "extensions": [".n5"], "notes": "Java/BigDataViewer format"},
                {"name": "Neuroglancer Precomputed", "extensions": [], "notes": "Local or remote (GCS/S3/HTTP)"},
            ],
            "tier_2_custom_optimized": [
                {"name": "TIFF", "extensions": [".tif", ".tiff"], "notes": "Single or stack, OME-TIFF supported"},
                {"name": "ND2", "extensions": [".nd2"], "notes": "Nikon NIS-Elements"},
                {"name": "IMS", "extensions": [".ims"], "notes": "Imaris/Bitplane HDF5-based"},
                {"name": "HDF5", "extensions": [".h5", ".hdf5", ".hdf", ".he5"], "notes": "Generic HDF5 containers"},
                {"name": "CZI", "extensions": [".czi"], "notes": "Zeiss multi-view"},
            ],
            "tier_3_bioio": [
                {"name": "BIOIO Adapter", "extensions": ["various"], "notes": "20+ formats via aicsimageio/bioio"},
            ],
            "tier_4_bioformats": [
                {"name": "Bio-Formats", "extensions": ["various"], "notes": "150+ formats via Java Bio-Formats (requires scyjava)"},
            ],
        },
        "output_formats": [
            {"name": "Zarr3", "notes": "Default. OME-NGFF v0.5, sharding, zstd compression"},
            {"name": "Zarr2", "notes": "Legacy. OME-NGFF v0.4, for tools that don't support Zarr3"},
            {"name": "N5", "notes": "For Java tools (BigDataViewer, BigStitcher)"},
        ],
        "remote_sources": [
            "gs:// (Google Cloud Storage)",
            "s3:// (Amazon S3)",
            "https:// (HTTP/HTTPS)",
        ],
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
        )

        input_path = input_path.strip()

        # Suppress stdout — MCP uses stdio transport
        with contextlib.redirect_stdout(io.StringIO()):
            reader = Readers.auto_detect(input_path)
        store = reader.get_tensorstore()
        shape = tuple(store.shape)
        dtype_str = get_dtype_name(store.dtype)

        # Detect native source (TensorStore-backed = fast, file-decoded = slow)
        _NATIVE_EXTENSIONS = {'.zarr', '.n5'}
        _input_ext = os.path.splitext(input_path)[1].lower()
        is_native = _input_ext in _NATIVE_EXTENSIONS or '://' in input_path

        cs_str = chunk_shape if chunk_shape else None
        ss_str = shard_shape if shard_shape else None

        memory_gb, wall_time, cores = calculate_job_resources(
            shape=list(shape),
            dtype=dtype_str,
            output_format=output_format,
            chunk_shape_str=cs_str,
            shard_shape_str=ss_str,
            no_sharding=no_sharding,
            is_native_source=is_native,
        )

        # Get shard/chunk info
        cs = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None
        ss = tuple(int(x) for x in shard_shape.split(",")) if shard_shape else None
        est_shape, total_units = estimate_shard_info(
            shape, dtype_str, output_format, cs, ss, no_sharding=no_sharding,
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
        omero: Include structured omero channel metadata for visualization tools.
        no_translation: Disable translation transforms in OME-NGFF multiscale metadata.
    """
    try:
        import argparse
        import contextlib
        import io

        input_path = input_path.strip()
        output_path = output_path.strip()

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

        # Apply preset
        if preset == "webknossos":
            if not chunk_shape:
                chunk_shape = "32,32,32"
            if not shard_shape:
                shard_shape = "1024,1024,1024"
        elif preset == "paintera":
            if output_format == "zarr3":
                output_format = "n5"
            if not chunk_shape:
                chunk_shape = "64,64,64"
            if compression == "zstd":
                compression = "gzip"
            if not axes_order:
                axes_order = "xyz" if output_format == "n5" else "zyx"

        # Handle auto_multiscale mode
        if auto_multiscale:
            from tensorswitch_v2.__main__ import find_base_level

            # Detect whether input is an existing dataset (pyramid-only)
            # or a raw source file (conversion + dependent pyramid).
            is_existing_dataset = False
            try:
                find_base_level(input_path)
                is_existing_dataset = True
            except (ValueError, OSError):
                pass

            if is_existing_dataset:
                # Pyramid-only: input already has s0
                subgroup = _resolve_conversion_subgroup(
                    output_format, data_type, is_label, image_key, label_key,
                )
                return _submit_pyramid_job(
                    input_path, output_path, project,
                    downsample_method=downsample_method,
                    per_level_factors=per_level_factors,
                    memory=memory, wall_time=wall_time, cores=cores,
                    log_dir=log_dir,
                    include_translation=not no_translation,
                    subgroup=subgroup,
                )
            # else: fall through to conversion submission, then chain pyramid

        # Build argparse.Namespace matching what __main__.submit_job() reads
        # Reference: __main__.py lines 836-1057
        args = argparse.Namespace(
            input=input_path,
            output=output_path,
            output_format=output_format,
            project=project,
            memory=memory if memory > 0 else None,
            wall_time=wall_time if wall_time else None,
            cores=cores if cores > 0 else None,
            dataset_path=dataset_path if dataset_path else "",
            level_path=level_path,
            chunk_shape=chunk_shape if chunk_shape else None,
            shard_shape=shard_shape if shard_shape else None,
            no_sharding=no_sharding,
            compression=compression,
            compression_level=compression_level,
            start_idx=None,
            stop_idx=None,
            write_metadata=False,
            view_index=view_index if view_index >= 0 else None,
            quiet=True,
            use_bioio=use_bioio,
            use_bioformats=use_bioformats,
            force_c_order=(force_order.lower() == "c") if force_order else False,
            force_f_order=(force_order.lower() == "f") if force_order else False,
            voxel_size=voxel_size if voxel_size else None,
            voxel_unit=voxel_unit if voxel_size else None,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
            data_type=data_type,
            label_key=label_key,
            image_key=image_key,
            use_nested_structure=True,
            bbox=bbox if bbox else None,
            axes_order=axes_order if axes_order else None,
            log_dir=log_dir if log_dir else None,
            no_ome_meta_export=no_ome_meta_export,
            no_ome_xml_attr=no_ome_xml_attr,
            job_group=job_group if job_group else None,
            omero=omero,
            no_translation=no_translation,
        )

        # Import and call the CLI submit_job function
        from tensorswitch_v2.__main__ import submit_job as _cli_submit_job

        # Suppress stdout — MCP uses stdio transport (stdout = JSON-RPC)
        with contextlib.redirect_stdout(io.StringIO()):
            job_id = _cli_submit_job(args, return_job_id=True)

        # If auto_multiscale was requested for a raw source file, chain a
        # dependent pyramid coordinator job after the conversion finishes.
        if auto_multiscale:
            coordinator_id = _submit_dependent_pyramid_mcp(
                conversion_job_id=str(job_id),
                output_path=output_path,
                project=project,
                output_format=output_format,
                data_type=data_type,
                is_label=is_label,
                image_key=image_key,
                label_key=label_key,
                downsample_method=downsample_method,
                per_level_factors=per_level_factors,
                no_translation=no_translation,
                log_dir=log_dir,
                memory=memory,
                wall_time=wall_time,
                cores=cores,
                job_group=job_group,
            )
            return json.dumps({
                "status": "submitted",
                "mode": "convert_and_pyramid",
                "conversion_job_id": str(job_id),
                "coordinator_job_id": coordinator_id,
                "input": input_path,
                "output": output_path,
                "format": output_format,
                "project": project,
                "message": (
                    f"Conversion job {job_id} submitted. "
                    f"Pyramid coordinator job {coordinator_id} will start after "
                    f"conversion completes. Use check_job_status to monitor."
                ),
            }, indent=2)

        return json.dumps({
            "status": "submitted",
            "job_id": job_id,
            "input": input_path,
            "output": output_path,
            "format": output_format,
            "project": project,
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


def _submit_dependent_pyramid_mcp(
    conversion_job_id: str,
    output_path: str,
    project: str,
    output_format: str = "zarr3",
    data_type: str = "auto",
    is_label: bool = False,
    image_key: str = "raw",
    label_key: str = "segmentation",
    downsample_method: str = "auto",
    per_level_factors: str = "",
    no_translation: bool = False,
    log_dir: str = "",
    memory: int = 0,
    wall_time: str = "",
    cores: int = 0,
    job_group: str = "",
):
    """Submit a dependent pyramid coordinator that waits for conversion to finish.

    Mirrors __main__._submit_dependent_pyramid() but standalone for MCP.
    Returns the coordinator LSF job ID, or "unknown" on failure.
    """
    import re
    import shlex
    import subprocess

    output_abs = os.path.abspath(output_path)
    subgroup = _resolve_conversion_subgroup(
        output_format, data_type, is_label, image_key, label_key,
    )
    pyramid_input = os.path.join(output_abs, subgroup) if subgroup else output_abs

    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--input", pyramid_input,
        "--auto_multiscale",
        "--submit",
        "-P", project,
    ]
    if downsample_method and downsample_method != "auto":
        reinvoke += ["--downsample_method", downsample_method]
    if per_level_factors:
        reinvoke += ["--per_level_factors", per_level_factors]
    if no_translation:
        reinvoke.append("--no_translation")
    if log_dir:
        reinvoke += ["--log_dir", log_dir]
    if memory and memory > 0:
        reinvoke += ["--memory", str(memory)]
    if wall_time:
        reinvoke += ["--wall_time", wall_time]
    if cores and cores > 0:
        reinvoke += ["--cores", str(cores)]

    reinvoke_str = shlex.join(reinvoke)

    output_parent = os.path.dirname(output_abs)
    effective_log_dir = log_dir or os.path.join(output_parent, "output")
    os.makedirs(effective_log_dir, exist_ok=True)

    job_name = f"tsv2_pyramid_coordinator_{os.path.basename(output_path)}"
    job_name = job_name.replace(" ", "_")[:128]

    command = [
        "bsub",
        "-J", job_name,
        "-n", "1",
        "-W", "0:30",
        "-M", "15GB",
        "-R", "rusage[mem=15360]",
        "-P", project,
        "-w", f"done({conversion_job_id})",
        "-o", os.path.join(effective_log_dir, f"output__{job_name}_%J.log"),
        "-e", os.path.join(effective_log_dir, f"error__{job_name}_%J.log"),
    ]
    if job_group:
        command += ["-g", job_group]
    command += ["/bin/bash", "-c", reinvoke_str]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        match = re.search(r'Job <(\d+)>', result.stdout)
        return match.group(1) if match else "unknown"

    logger.error(f"Pyramid coordinator submission failed: {result.stderr.strip()}")
    return "unknown"


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
def check_job_status(job_id: str) -> str:
    """Check the status of one or more LSF cluster jobs.

    Returns the current status of jobs submitted via submit_job.
    Supports checking multiple jobs by passing comma-separated IDs.

    Args:
        job_id: LSF job ID (e.g., "12345") or comma-separated IDs (e.g., "12345,12346").
    """
    try:
        import subprocess

        job_ids = [j.strip() for j in job_id.split(",") if j.strip()]
        results = []

        for jid in job_ids:
            try:
                result = subprocess.run(
                    ["bjobs", "-noheader", "-o", "stat job_name", jid],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split()
                    status = parts[0] if parts else "UNKNOWN"
                    job_name = parts[1] if len(parts) > 1 else "unknown"
                    results.append({
                        "job_id": jid,
                        "status": status,
                        "job_name": job_name,
                    })
                else:
                    results.append({
                        "job_id": jid,
                        "status": "NOT_FOUND",
                        "message": result.stderr.strip() or "Job not found in LSF",
                    })
            except subprocess.TimeoutExpired:
                results.append({
                    "job_id": jid,
                    "status": "TIMEOUT",
                    "message": "bjobs timed out after 30 seconds",
                })

        if len(results) == 1:
            return json.dumps(results[0], indent=2)
        return json.dumps({"jobs": results}, indent=2)

    except FileNotFoundError:
        return json.dumps({
            "error": "bjobs_not_found",
            "message": "bjobs command not found. Requires LSF cluster environment.",
        }, indent=2)
    except Exception as e:
        logger.error(f"check_job_status failed: {e}\n{traceback.format_exc()}")
        return f"Error checking job status: {e}"


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
