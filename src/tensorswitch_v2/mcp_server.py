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
import sys
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Logging to stderr (required for stdio transport — stdout is JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tensorswitch-mcp")

mcp = FastMCP("tensorswitch")


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

        # Check if this is an OME-Zarr container (has zarr.json or .zattrs at root)
        zarr_json = os.path.join(path, "zarr.json")
        zattrs = os.path.join(path, ".zattrs")

        if os.path.isfile(zarr_json) or os.path.isfile(zattrs):
            return _inspect_zarr_container(path)

        # Otherwise, inspect as a single dataset
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

    # Check for labels
    if "labels" in ome:
        result["labels"] = ome["labels"]

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
def discover_datasets(path: str) -> str:
    """Scan a directory and list all recognized microscopy datasets.

    Recursively discovers image and segmentation layers in Zarr, N5,
    and Neuroglancer precomputed containers.

    Args:
        path: Directory path to scan.
    """
    try:
        from tensorswitch_v2.utils.folder_discovery import (
            discover_datasets as _discover,
        )

        result = _discover(path.strip())

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
    voxel_size: str = "",
    voxel_unit: str = "nanometer",
    is_label: bool = False,
    compression: str = "zstd",
    compression_level: int = 5,
    dataset_path: str = "",
) -> str:
    """Convert a microscopy dataset between formats.

    Supported inputs: HDF5, TIFF, ND2, IMS, CZI, Zarr2, Zarr3, N5,
    Neuroglancer precomputed, and 150+ formats via Bio-Formats.
    Supported outputs: zarr3 (with sharding), zarr2, n5.

    Args:
        input_path: Path to source dataset.
        output_path: Path for output (e.g., /data/output.zarr).
        output_format: Output format — "zarr3" (default), "zarr2", or "n5".
        chunk_shape: Comma-separated chunk shape (e.g., "64,64,64"). Auto-calculated if empty.
        shard_shape: Comma-separated shard shape for zarr3 (e.g., "512,512,512"). Auto-calculated if empty.
        voxel_size: Comma-separated voxel sizes in X,Y,Z order (e.g., "6,6,29"). Required if source lacks metadata.
        voxel_unit: Unit for voxel sizes — "nanometer", "micrometer", or "millimeter".
        is_label: Set True for segmentation/label data (uses mode downsampling, adds label metadata).
        compression: Compression codec — "zstd" (default), "gzip", or "none".
        compression_level: Compression level (1-22 for zstd, 1-9 for gzip).
        dataset_path: Path within container file (e.g., "main" for HDF5 dataset name).
    """
    try:
        from tensorswitch_v2.api import Readers, Writers
        from tensorswitch_v2.core.converter import DistributedConverter

        input_path = input_path.strip()
        output_path = output_path.strip()

        # Create reader
        if dataset_path:
            # For formats with internal paths (HDF5, N5, Zarr)
            ext = Path(input_path).suffix.lower()
            if ext in (".h5", ".hdf5", ".hdf", ".he5"):
                reader = Readers.hdf5(input_path, dataset_path=dataset_path)
            elif ext in (".n5",) or os.path.isfile(
                os.path.join(input_path, "attributes.json")
            ):
                reader = Readers.n5(input_path, dataset_path=dataset_path)
            else:
                reader = Readers.auto_detect(input_path)
        else:
            reader = Readers.auto_detect(input_path)

        # Create writer
        data_type = "labels" if is_label else "image"
        if output_format == "zarr3":
            writer = Writers.zarr3(
                output_path,
                use_sharding=True,
                compression=compression,
                compression_level=compression_level,
                data_type=data_type,
            )
        elif output_format == "zarr2":
            writer = Writers.zarr2(
                output_path,
                compression=compression,
                compression_level=compression_level,
                data_type=data_type,
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

        # Run conversion
        converter = DistributedConverter(reader, writer)
        result = converter.convert(
            chunk_shape=cs,
            shard_shape=ss,
            voxel_size_override=voxel_override,
            voxel_unit=voxel_unit if voxel_size else None,
            is_label=is_label,
        )

        return json.dumps(
            {
                "status": "success",
                "input": input_path,
                "output": output_path,
                "format": output_format,
                "chunks_processed": result.get("chunks_processed", "unknown"),
                "time_seconds": round(result.get("elapsed_time", 0), 1),
            },
            indent=2,
        )
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
    """
    try:
        from tensorswitch_v2.core.pyramid import PyramidPlanner

        s0_path = s0_path.strip()
        planner = PyramidPlanner(
            s0_path,
            include_translation=True,
            downsample_method=downsample_method,
        )

        plan = planner.calculate_pyramid_plan()

        # Run locally
        planner.precreate_all_levels(plan, use_shard=True, verbose=False)

        from tensorswitch_v2.core.downsampler import downsample_level

        parent_dir = str(Path(s0_path).parent)
        levels_created = []

        for level_info in plan["levels"]:
            level_num = level_info["level"]
            cum_factors = level_info["cumulative_factors"]
            output_level_path = os.path.join(parent_dir, f"s{level_num}")

            downsample_level(
                s0_path=s0_path,
                output_path=output_level_path,
                target_level=level_num,
                cumulative_factors=cum_factors,
                downsample_method=downsample_method,
                verbose=False,
            )
            levels_created.append({
                "level": f"s{level_num}",
                "factors": cum_factors,
                "shape": level_info["shape"],
            })

        # Update metadata
        from tensorswitch_v2.utils.metadata_utils import update_ome_metadata_if_needed

        update_ome_metadata_if_needed(
            parent_dir,
            include_translation=True,
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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting TensorSwitch MCP server")
    mcp.run(transport="stdio")
