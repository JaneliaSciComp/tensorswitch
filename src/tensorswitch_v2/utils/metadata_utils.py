# Copied from tensorswitch/utils.py for v2 independence
"""
Metadata utility functions for tensorswitch_v2.

This module contains OME-NGFF metadata operations including:
- Creating and updating OME metadata
- Zarr3 and Zarr2 group metadata
- Pre-creation of zarr3 output structures
- Auto-detection of multiscale levels
"""

import os
import json
import glob
import numpy as np
import re
from typing import Optional, Tuple


# ============================================================================
# Level Format Detection Utilities
# ============================================================================

def detect_level_format(root_path: str) -> str:
    """
    Detect the level naming format used in a zarr dataset.

    Checks whether the dataset uses "s0/s1/s2" (s-prefixed) or "0/1/2" (numeric)
    naming convention for multiscale levels.

    For EXISTING datasets: Returns the format of the existing level 0
    For NEW datasets (no existing levels): Returns "s" (Janelia convention)

    Args:
        root_path: Root path of the zarr dataset

    Returns:
        "s" if using s0/s1/s2 format (s-prefixed) or for new datasets
        "" if using 0/1/2 format (numeric) and level 0 already exists
    """
    # Check for s-prefixed format first (s0)
    if os.path.exists(os.path.join(root_path, 's0')):
        return "s"
    # Check for numeric format (0)
    if os.path.exists(os.path.join(root_path, '0')):
        return ""
    # Default to s-prefix for new datasets (Janelia house style convention)
    return "s"


def get_level_name(level: int, prefix: str = "") -> str:
    """
    Get level directory name with specified prefix.

    Args:
        level: Level number (0, 1, 2, ...)
        prefix: Prefix ("s" for s0/s1 format, "" for numeric 0/1 format)

    Returns:
        Level name like "s0", "s1" or "0", "1"
    """
    return f"{prefix}{level}"


def get_level_path(root_path: str, level: int, prefix: str = None) -> str:
    """
    Get full level path using detected or specified prefix.

    Args:
        root_path: Root zarr path
        level: Level number (0, 1, 2, ...)
        prefix: Override prefix ("s" or ""). If None, auto-detect from existing levels.

    Returns:
        Full path like "/data/dataset.zarr/s1" or "/data/dataset.zarr/1"
    """
    if prefix is None:
        prefix = detect_level_format(root_path)
    return os.path.join(root_path, f"{prefix}{level}")


# ============================================================================
# Compression Extraction Utilities
# ============================================================================

def extract_compression_from_zarr3_metadata(metadata: dict) -> Optional[dict]:
    """
    Extract compression codec from Zarr3 metadata.

    Zarr3 stores compression in the codecs array. For sharded arrays,
    the compression codec is inside sharding_indexed.configuration.codecs.

    Args:
        metadata: Zarr3 metadata dict (contents of zarr.json)

    Returns:
        Compression codec dict like {'name': 'zstd', 'configuration': {'level': 5}}
        or None if not found
    """
    codecs = metadata.get('codecs', [])

    # Check for sharding_indexed (compression is in inner codecs)
    for codec in codecs:
        if codec.get('name') == 'sharding_indexed':
            inner_codecs = codec.get('configuration', {}).get('codecs', [])
            for inner_codec in inner_codecs:
                if inner_codec.get('name') in ['zstd', 'blosc', 'gzip']:
                    return inner_codec

    # Check for direct compression codec (non-sharded)
    for codec in codecs:
        if codec.get('name') in ['zstd', 'blosc', 'gzip']:
            return codec

    return None


def extract_compression_from_zarr2_metadata(zarray: dict) -> Optional[dict]:
    """
    Extract compressor from Zarr2 .zarray metadata.

    Zarr2 stores compression in the 'compressor' field.

    Args:
        zarray: Zarr2 .zarray metadata dict

    Returns:
        Compressor dict like {'id': 'zstd', 'level': 5} or None
    """
    return zarray.get('compressor')


# ============================================================================
# OME-XML and Channel Metadata
# ============================================================================

def extract_omero_channels(ome_xml: str) -> list:
    """
    Extract channel info from OME-XML for omero metadata block.

    Args:
        ome_xml: Raw OME-XML string containing Channel elements

    Returns:
        List of channel dicts with label, color, and window info,
        or None if no channels found or parsing fails.
    """
    if not ome_xml or not isinstance(ome_xml, str):
        return None

    channels = []
    channel_pattern = r'<(?:OME:)?Channel\s+([^>]+)>'

    for channel_match in re.finditer(channel_pattern, ome_xml):
        attrs = channel_match.group(1)

        name_match = re.search(r'Name\s*=\s*"([^"]*)"', attrs)
        if not name_match:
            continue
        name = name_match.group(1)

        color_match = re.search(r'Color\s*=\s*"(-?\d+)"', attrs)
        if not color_match:
            continue
        color_int = int(color_match.group(1))

        # Convert integer color (ARGB) to hex RGB
        if color_int < 0:
            color_int = color_int & 0xFFFFFFFF
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        color_hex = f"{r:02X}{g:02X}{b:02X}"

        channels.append({
            "label": name,
            "color": color_hex,
            "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}
        })

    return channels if channels else None


def create_zarr3_ome_metadata(ome_xml, array_shape, image_name, pixel_sizes=None, axes_order=None, include_omero=False, is_label=False, label_colors=None, level_path="s0"):
    """
    Create OME-ZARR metadata structure for zarr3 format.

    Args:
        ome_xml: OME-XML string to include in metadata (can be None)
        array_shape: Shape of the array
        image_name: Name for the image/dataset
        pixel_sizes: Dict with 'x', 'y', 'z' voxel sizes in nanometers
        axes_order: List of axis names (e.g., ["v", "c", "z", "y", "x"] for CZI multi-view)
        include_omero: If True, extract and add omero channel metadata from ome_xml
        is_label: If True, add image-label metadata for segmentation data
        label_colors: Optional list of color dicts for labels. If None and is_label=True,
                      generates default colors.
        level_path: Path for level 0 dataset (default "s0" for Janelia convention)
    """
    def get_axis_type(axis_name):
        axis_lower = axis_name.lower()
        if axis_lower in ['c', 'channel']:
            return 'channel'
        elif axis_lower in ['t', 'v']:
            return 'time'
        else:
            return 'space'

    def normalize_axis_name(axis_name):
        """Normalize axis name to OME-NGFF standard (c instead of channel, etc.)"""
        axis_lower = axis_name.lower()
        if axis_lower == 'channel':
            return 'c'
        elif axis_lower == 'v':
            return 't'
        return axis_lower

    def get_axis_unit(axis_name):
        if axis_name in ['x', 'y', 'z']:
            return 'nanometer'
        elif axis_name == 't':
            return 'millisecond'
        return None

    ndim = len(array_shape)
    axes = []
    scale_factors = [1.0] * ndim

    if axes_order and len(axes_order) == ndim:
        for axis_name in axes_order:
            output_name = normalize_axis_name(axis_name)
            axis_entry = {"name": output_name, "type": get_axis_type(axis_name)}
            unit = get_axis_unit(axis_name)
            if unit:
                axis_entry["unit"] = unit
            axes.append(axis_entry)
        if pixel_sizes is not None:
            scale_factors = [pixel_sizes.get(axis, 1.0) for axis in axes_order]
    elif ndim == 3:
        if array_shape[0] <= 10:
            axes = [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"}
            ]
            if pixel_sizes:
                scale_factors = [1.0, pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
        else:
            axes = [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"}
            ]
            if pixel_sizes:
                scale_factors = [pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
    elif ndim == 4:
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"}
        ]
        if pixel_sizes:
            scale_factors = [1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
    elif ndim == 5:
        axes = [
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"}
        ]
        if pixel_sizes:
            scale_factors = [1.0, 1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
    else:
        axes = [{"name": f"axis_{i}", "type": "space"} for i in range(ndim)]

    coordinate_transformations = [{
        "type": "scale",
        "scale": scale_factors
    }]

    multiscales = [{
        "axes": axes,
        "datasets": [{
            "path": level_path,
            "coordinateTransformations": coordinate_transformations
        }],
        "name": image_name,
        "type": "image"
    }]

    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": multiscales
            }
        }
    }

    if include_omero and ome_xml:
        omero_channels = extract_omero_channels(ome_xml)
        if omero_channels:
            metadata["attributes"]["ome"]["omero"] = {
                "channels": omero_channels,
                "rdefs": {"model": "color"}
            }

    if ome_xml:
        if isinstance(ome_xml, str):
            metadata["attributes"]["ome_xml"] = ome_xml
        else:
            metadata["attributes"]["ome_xml"] = str(ome_xml)

    # Add image-label metadata for segmentation data
    if is_label:
        if label_colors is None:
            label_colors = generate_default_label_colors(256)  # Default 256 colors
        image_label = create_image_label_metadata(
            label_colors=label_colors,
            source_path=None,  # Standalone label, no source reference
            ngff_version="0.5"
        )
        metadata["attributes"]["ome"]["image-label"] = image_label
        # Remove "type": "image" from multiscales for labels (optional but cleaner)
        if "type" in metadata["attributes"]["ome"]["multiscales"][0]:
            del metadata["attributes"]["ome"]["multiscales"][0]["type"]

    return metadata


def write_zarr3_group_metadata(output_path, metadata):
    """Write zarr3 group-level zarr.json file with OME-ZARR metadata."""
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def auto_detect_max_level(output_path):
    """
    Auto-detect the maximum level by scanning for level directories.

    Supports both s-prefixed (s0, s1, s2) and numeric (0, 1, 2) formats.

    Returns:
        tuple: (max_level, prefix) where prefix is "s" or ""
        None if no levels found
    """
    if not os.path.exists(output_path):
        return None, ""

    # Detect the format first
    prefix = detect_level_format(output_path)

    levels = []
    if prefix == "s":
        # S-prefixed format (s0, s1, s2, ...)
        pattern = os.path.join(output_path, 's*')
        level_dirs = glob.glob(pattern)
        for level_dir in level_dirs:
            dirname = os.path.basename(level_dir)
            if dirname.startswith('s') and dirname[1:].isdigit():
                levels.append(int(dirname[1:]))
    else:
        # Numeric format (0, 1, 2, ...)
        try:
            for item in os.listdir(output_path):
                item_path = os.path.join(output_path, item)
                if item.isdigit() and os.path.isdir(item_path):
                    levels.append(int(item))
        except OSError:
            return None, prefix

    if not levels:
        return None, prefix

    return max(levels), prefix


def update_ome_multiscale_metadata(zarr_path, max_level=4, prefix=None):
    """
    Update OME-ZARR metadata to include all multiscale levels.

    Supports both s-prefixed (s0, s1) and numeric (0, 1) level naming.
    """
    zarr_json_path = os.path.join(zarr_path, "zarr.json")

    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)

    multiscales = metadata["attributes"]["ome"]["multiscales"][0]
    level0_scale_factors = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]

    downsampling_factors = metadata.get("attributes", {}).get("custom", {}).get("downsampling_factors", None)

    # Detect level naming format if not provided
    if prefix is None:
        prefix = detect_level_format(zarr_path)

    level0_name = get_level_name(0, prefix)
    level0_metadata_path = os.path.join(zarr_path, level0_name, "zarr.json")
    with open(level0_metadata_path, 'r') as f:
        level0_meta = json.load(f)
    level0_shape = level0_meta.get('shape')

    if downsampling_factors:
        print("Using downsampling factors from custom metadata (incremental method)")
        use_factors = True
        previous_scale = level0_scale_factors
    else:
        print("No downsampling factors found - using dimension ratio method")
        use_factors = False

    datasets = []

    for level in range(max_level + 1):
        level_name = get_level_name(level, prefix)
        level_metadata_path = os.path.join(zarr_path, level_name, "zarr.json")

        if not os.path.exists(level_metadata_path):
            print(f"Warning: {level_name}/zarr.json not found, skipping level {level}")
            break

        if level == 0:
            # Update path in first dataset to use detected format
            first_dataset = multiscales["datasets"][0].copy()
            first_dataset["path"] = level_name
            datasets.append(first_dataset)
            continue

        with open(level_metadata_path, 'r') as f:
            level_meta = json.load(f)
        level_shape = level_meta.get('shape')

        if use_factors:
            prev_level_name = get_level_name(level - 1, prefix)
            factor_key = f"{prev_level_name}_to_{level_name}"
            level_factors = downsampling_factors.get(factor_key)
            if level_factors:
                level_scale = [prev * factor for prev, factor in zip(previous_scale, level_factors)]
            else:
                print(f"Warning: Missing factors for {factor_key}, falling back to ratio method")
                level_scale = [level0_scale_factors[i] * (level0_shape[i] / level_shape[i]) for i in range(len(level0_shape))]
            previous_scale = level_scale
        else:
            level_scale = [level0_scale_factors[i] * (level0_shape[i] / level_shape[i]) for i in range(len(level0_shape))]

        coordinate_transformations = [{"type": "scale", "scale": level_scale}]

        # Add translation transform for Neuroglancer compatibility
        translation = [0.5 * (scale - level0_scale_factors[i]) for i, scale in enumerate(level_scale)]
        if any(t != 0 for t in translation):
            coordinate_transformations.append({"type": "translation", "translation": translation})

        datasets.append({
            "path": level_name,
            "coordinateTransformations": coordinate_transformations
        })

    multiscales["datasets"] = datasets

    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated OME metadata with {len(datasets)} levels")


def update_ome_multiscale_metadata_zarr2(zarr_path, max_level=4, prefix=None):
    """
    Update OME-ZARR metadata for zarr2 format (.zattrs).

    Supports both s-prefixed (s0, s1) and numeric (0, 1) level naming.
    """
    zattrs_path = os.path.join(zarr_path, ".zattrs")

    with open(zattrs_path, 'r') as f:
        metadata = json.load(f)

    multiscales = metadata.get("multiscales", [{}])[0]
    level0_scale_factors = multiscales.get("datasets", [{}])[0].get("coordinateTransformations", [{}])[0].get("scale", [1.0, 1.0, 1.0])

    # Detect level naming format if not provided
    if prefix is None:
        prefix = detect_level_format(zarr_path)

    level0_name = get_level_name(0, prefix)
    zarray_path = os.path.join(zarr_path, level0_name, ".zarray")
    with open(zarray_path, 'r') as f:
        level0_meta = json.load(f)
    level0_shape = level0_meta.get('shape')

    datasets = []

    for level in range(max_level + 1):
        level_name = get_level_name(level, prefix)
        level_zarray_path = os.path.join(zarr_path, level_name, ".zarray")

        if not os.path.exists(level_zarray_path):
            print(f"Warning: {level_name}/.zarray not found, skipping level {level}")
            break

        if level == 0:
            # Update path in first dataset to use detected format
            first_dataset = multiscales.get("datasets", [{}])[0].copy()
            first_dataset["path"] = level_name
            datasets.append(first_dataset)
            continue

        with open(level_zarray_path, 'r') as f:
            level_meta = json.load(f)
        level_shape = level_meta.get('shape')

        level_scale = [level0_scale_factors[i] * (level0_shape[i] / level_shape[i]) for i in range(len(level0_shape))]

        coordinate_transformations = [{"type": "scale", "scale": level_scale}]

        # Add translation transform for Neuroglancer compatibility (same as Zarr3)
        translation = [0.5 * (scale - level0_scale_factors[i]) for i, scale in enumerate(level_scale)]
        if any(t != 0 for t in translation):
            coordinate_transformations.append({"type": "translation", "translation": translation})

        datasets.append({
            "path": level_name,
            "coordinateTransformations": coordinate_transformations
        })

    multiscales["datasets"] = datasets
    metadata["multiscales"] = [multiscales]

    with open(zattrs_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated Zarr2 OME metadata with {len(datasets)} levels")

    # Create parent group metadata for nested OME-NGFF structure
    _create_zarr2_parent_metadata(zarr_path, metadata)


def _create_zarr2_parent_metadata(zarr_path, child_metadata):
    """
    Create parent .zgroup and .zattrs files for Zarr2 nested OME-NGFF structure.

    When updating metadata at a nested path like 'labels/segmentation', this creates:
    - labels/.zgroup and labels/.zattrs (with labels list)
    - root/.zgroup and root/.zattrs (with multiscales pointing to raw + labels list)

    Args:
        zarr_path: Path to the zarr directory being updated (e.g., .../labels/segmentation)
        child_metadata: The metadata dict of the child (used to extract info)
    """
    import os

    # Get absolute path and check parent structure
    abs_path = os.path.abspath(zarr_path)
    parent_dir = os.path.dirname(abs_path)
    grandparent_dir = os.path.dirname(parent_dir)
    child_name = os.path.basename(abs_path)
    parent_name = os.path.basename(parent_dir)

    # Check if this is a labels path (parent named 'labels')
    if parent_name == 'labels':
        # Create labels container metadata
        labels_zgroup = os.path.join(parent_dir, '.zgroup')
        labels_zattrs = os.path.join(parent_dir, '.zattrs')

        if not os.path.exists(labels_zgroup):
            with open(labels_zgroup, 'w') as f:
                json.dump({"zarr_format": 2}, f, indent=2)
            print(f"Created labels/.zgroup")

        # Create or update labels/.zattrs with labels list
        labels_list = [child_name]
        if os.path.exists(labels_zattrs):
            with open(labels_zattrs, 'r') as f:
                existing = json.load(f)
            existing_labels = existing.get('labels', [])
            if child_name not in existing_labels:
                labels_list = existing_labels + [child_name]
            else:
                labels_list = existing_labels
        with open(labels_zattrs, 'w') as f:
            json.dump({"labels": labels_list}, f, indent=2)
        print(f"Updated labels/.zattrs with labels: {labels_list}")

        # Create root metadata
        root_zgroup = os.path.join(grandparent_dir, '.zgroup')
        root_zattrs = os.path.join(grandparent_dir, '.zattrs')

        if not os.path.exists(root_zgroup):
            with open(root_zgroup, 'w') as f:
                json.dump({"zarr_format": 2}, f, indent=2)
            print(f"Created root .zgroup")

        # Update root .zattrs to include labels list
        if os.path.exists(root_zattrs):
            with open(root_zattrs, 'r') as f:
                root_attrs = json.load(f)
        else:
            root_attrs = {}

        if 'labels' not in root_attrs:
            root_attrs['labels'] = [parent_name]  # "labels"
            with open(root_zattrs, 'w') as f:
                json.dump(root_attrs, f, indent=2)
            print(f"Updated root .zattrs with labels reference")

    # Check if this is an image path (named 'raw' or sibling to 'labels')
    elif child_name == 'raw' or os.path.exists(os.path.join(parent_dir, 'labels')):
        # Create root metadata
        root_zgroup = os.path.join(parent_dir, '.zgroup')
        root_zattrs = os.path.join(parent_dir, '.zattrs')

        if not os.path.exists(root_zgroup):
            with open(root_zgroup, 'w') as f:
                json.dump({"zarr_format": 2}, f, indent=2)
            print(f"Created root .zgroup")

        # Update root .zattrs with image multiscales
        if os.path.exists(root_zattrs):
            with open(root_zattrs, 'r') as f:
                root_attrs = json.load(f)
        else:
            root_attrs = {}

        # If we have multiscales in child, create root multiscales with adjusted paths
        if 'multiscales' in child_metadata:
            ms = child_metadata['multiscales'][0].copy()
            # Adjust dataset paths to include the image directory name
            adjusted_datasets = []
            for ds in ms.get('datasets', []):
                new_ds = ds.copy()
                new_ds['path'] = f"{child_name}/{ds['path']}"
                adjusted_datasets.append(new_ds)
            ms['datasets'] = adjusted_datasets
            root_attrs['multiscales'] = [ms]

        # Check if labels directory exists
        if os.path.exists(os.path.join(parent_dir, 'labels')):
            root_attrs['labels'] = ['labels']

        with open(root_zattrs, 'w') as f:
            json.dump(root_attrs, f, indent=2)
        print(f"Updated root .zattrs")


def _update_parent_zarr3_json(inner_path, parent_path, image_key):
    """
    Update parent zarr.json to reflect inner path's multiscale levels with image_key prefix.

    After pyramid generation updates raw/zarr.json with s0, s1, s2..., this propagates
    those levels to the outer <name>.zarr/zarr.json as raw/s0, raw/s1, raw/s2...

    Args:
        inner_path: Path to the inner group (e.g., <name>.zarr/raw/)
        parent_path: Path to the parent zarr root (e.g., <name>.zarr/)
        image_key: Name of the image group (e.g., 'raw')
    """
    inner_zarr_json = os.path.join(inner_path, 'zarr.json')
    parent_zarr_json = os.path.join(parent_path, 'zarr.json')

    with open(inner_zarr_json, 'r') as f:
        inner_metadata = json.load(f)

    inner_ms_list = inner_metadata.get('attributes', {}).get('ome', {}).get('multiscales', [])
    if not inner_ms_list:
        return

    inner_ms = inner_ms_list[0]

    # Build prefixed dataset paths (e.g., s0 -> raw/s0)
    adjusted_datasets = []
    for ds in inner_ms.get('datasets', []):
        new_ds = ds.copy()
        new_ds['path'] = f"{image_key}/{ds['path']}"
        adjusted_datasets.append(new_ds)

    with open(parent_zarr_json, 'r') as f:
        parent_metadata = json.load(f)

    parent_ome = parent_metadata.get('attributes', {}).get('ome', {})
    parent_ms_list = parent_ome.get('multiscales', [])

    if parent_ms_list:
        parent_ms_list[0]['datasets'] = adjusted_datasets
        parent_ms_list[0]['axes'] = inner_ms.get('axes', parent_ms_list[0].get('axes', []))
    else:
        parent_ome['multiscales'] = [{
            'axes': inner_ms.get('axes', []),
            'datasets': adjusted_datasets,
            'name': inner_ms.get('name', 'image'),
            'type': inner_ms.get('type', 'image')
        }]

    parent_metadata['attributes']['ome'] = parent_ome

    with open(parent_zarr_json, 'w') as f:
        json.dump(parent_metadata, f, indent=2)

    print(f"Updated parent zarr3 metadata with {len(adjusted_datasets)} levels ({image_key}/s0 ... {image_key}/{adjusted_datasets[-1]['path'].split('/')[-1]})")


def _update_parent_zarr2_zattrs(inner_path, parent_path, image_key):
    """
    Update parent .zattrs to reflect inner path's multiscale levels with image_key prefix.

    Args:
        inner_path: Path to the inner group (e.g., <name>.zarr/raw/)
        parent_path: Path to the parent zarr root (e.g., <name>.zarr/)
        image_key: Name of the image group (e.g., 'raw')
    """
    inner_zattrs = os.path.join(inner_path, '.zattrs')
    parent_zattrs = os.path.join(parent_path, '.zattrs')

    with open(inner_zattrs, 'r') as f:
        inner_metadata = json.load(f)

    inner_ms_list = inner_metadata.get('multiscales', [])
    if not inner_ms_list:
        return

    inner_ms = inner_ms_list[0]

    # Build prefixed dataset paths
    adjusted_datasets = []
    for ds in inner_ms.get('datasets', []):
        new_ds = ds.copy()
        new_ds['path'] = f"{image_key}/{ds['path']}"
        adjusted_datasets.append(new_ds)

    with open(parent_zattrs, 'r') as f:
        parent_metadata = json.load(f)

    parent_ms_list = parent_metadata.get('multiscales', [])

    if parent_ms_list:
        parent_ms_list[0]['datasets'] = adjusted_datasets
        parent_ms_list[0]['axes'] = inner_ms.get('axes', parent_ms_list[0].get('axes', []))
    else:
        parent_metadata['multiscales'] = [{
            'axes': inner_ms.get('axes', []),
            'datasets': adjusted_datasets,
            'name': inner_ms.get('name', 'image'),
        }]

    with open(parent_zattrs, 'w') as f:
        json.dump(parent_metadata, f, indent=2)

    print(f"Updated parent zarr2 metadata with {len(adjusted_datasets)} levels")


def update_ome_metadata_if_needed(output_path, use_ome_structure):
    """
    Update OME-Zarr metadata if OME structure is used and multiscale levels exist.
    Supports both zarr2 (.zattrs) and zarr3 (zarr.json) formats.

    Also updates the parent zarr.json/zattrs if this path is a nested image group
    (e.g., raw/ inside <name>.zarr/), propagating all pyramid levels with the
    image_key prefix (e.g., raw/s0, raw/s1, ...).
    """
    if not use_ome_structure:
        return

    max_level, prefix = auto_detect_max_level(output_path)
    if max_level is None:
        print("No multiscale levels detected - skipping metadata update")
        return

    level0_name = get_level_name(0, prefix)
    if max_level == 0:
        print(f"Only {level0_name} level detected - no multiscale metadata update needed")
        return

    max_level_name = get_level_name(max_level, prefix)
    zarr3_metadata = os.path.join(output_path, 'zarr.json')
    zarr2_metadata = os.path.join(output_path, '.zattrs')

    try:
        if os.path.exists(zarr3_metadata):
            print(f"Updating zarr3 OME metadata for {output_path} with levels {level0_name}-{max_level_name}")
            update_ome_multiscale_metadata(output_path, max_level=max_level, prefix=prefix)
            print("OME metadata updated successfully!")

            # Also update parent zarr.json if this is a nested image group (e.g., raw/)
            parent_path = os.path.dirname(output_path)
            image_key = os.path.basename(output_path)
            parent_zarr3 = os.path.join(parent_path, 'zarr.json')
            if os.path.exists(parent_zarr3):
                try:
                    print(f"Updating parent zarr3 metadata at {parent_zarr3}")
                    _update_parent_zarr3_json(output_path, parent_path, image_key)
                except Exception as e:
                    print(f"Warning: Failed to update parent zarr3 metadata: {e}")

        elif os.path.exists(zarr2_metadata):
            print(f"Updating zarr2 OME metadata for {output_path} with levels {level0_name}-{max_level_name}")
            update_ome_multiscale_metadata_zarr2(output_path, max_level=max_level, prefix=prefix)
            print("OME metadata updated successfully!")

            # Also update parent .zattrs if this is a nested image group (e.g., raw/)
            parent_path = os.path.dirname(output_path)
            image_key = os.path.basename(output_path)
            parent_zarr2 = os.path.join(parent_path, '.zattrs')
            if os.path.exists(parent_zarr2):
                try:
                    print(f"Updating parent zarr2 metadata at {parent_zarr2}")
                    _update_parent_zarr2_zattrs(output_path, parent_path, image_key)
                except Exception as e:
                    print(f"Warning: Failed to update parent zarr2 metadata: {e}")

        else:
            print("Warning: No zarr metadata file found (.zattrs or zarr.json) - skipping metadata update")
    except Exception as e:
        print(f"Warning: Failed to update OME metadata: {e}")


def precreate_shard_directories(output_path, level, output_shape, shard_shape, use_ome_structure=True, level_prefix=None):
    """
    Pre-create all shard directory structures to avoid race conditions with parallel workers.

    Args:
        level_prefix: Prefix for level naming ("s" for s0/s1 or "" for 0/1).
                     If None, auto-detects from existing levels or defaults to "s".
    """
    import time

    # Detect or use provided prefix
    if level_prefix is None:
        level_prefix = detect_level_format(output_path)
    level_name = get_level_name(level, level_prefix)

    print("\n" + "="*80)
    print(f"PRE-CREATING SHARD DIRECTORIES FOR {level_name}")
    print("="*80)

    if not isinstance(output_shape, list):
        output_shape = list(output_shape)
    if not isinstance(shard_shape, list):
        shard_shape = list(shard_shape)

    # Adjust shard shape to match array dimensions
    if len(output_shape) == 4 and len(shard_shape) == 3:
        shard_shape = [1] + shard_shape
    elif len(output_shape) == 4 and len(shard_shape) == 2:
        shard_shape = [1, 1] + shard_shape
    elif len(output_shape) == 3 and len(shard_shape) == 2:
        shard_shape = [1] + shard_shape
    elif len(output_shape) == 5 and len(shard_shape) == 3:
        shard_shape = [1, 1] + shard_shape
    elif len(output_shape) == 5 and len(shard_shape) == 2:
        shard_shape = [1, 1, 1] + shard_shape

    num_shards = [
        (output_shape[i] + shard_shape[i] - 1) // shard_shape[i]
        for i in range(len(output_shape))
    ]

    total_dirs = np.prod([num_shards[i] for i in range(min(len(num_shards)-1, 3))])

    print(f"Output shape: {output_shape}")
    print(f"Shard shape: {shard_shape}")
    print(f"Number of shards per dimension: {num_shards}")
    print(f"Total directories to create: {total_dirs}")

    if use_ome_structure:
        base_shard_path = os.path.join(output_path, level_name, "c")
    else:
        base_shard_path = os.path.join(output_path, "c")

    print(f"Base shard path: {base_shard_path}")

    created = 0
    start_time = time.time()

    if len(num_shards) == 4:  # CZYX
        for c in range(num_shards[0]):
            for z in range(num_shards[1]):
                for y in range(num_shards[2]):
                    dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                    os.makedirs(dir_path, exist_ok=True)
                    created += 1
                    if created % 100 == 0:
                        print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")
    elif len(num_shards) == 3:  # ZYX or CYX
        for dim0 in range(num_shards[0]):
            for dim1 in range(num_shards[1]):
                dir_path = os.path.join(base_shard_path, str(dim0), str(dim1))
                os.makedirs(dir_path, exist_ok=True)
                created += 1
                if created % 100 == 0:
                    print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")
    elif len(num_shards) == 5:  # TCZYX
        for t in range(num_shards[0]):
            for c in range(num_shards[1]):
                for z in range(num_shards[2]):
                    for y in range(num_shards[3]):
                        dir_path = os.path.join(base_shard_path, str(t), str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
                        created += 1
                        if created % 100 == 0:
                            print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")

    elapsed = time.time() - start_time
    rate = created / elapsed if elapsed > 0 else 0
    print(f"\nCreated {created} shard directories in {elapsed:.2f} seconds ({rate:.1f} dirs/sec)")
    print("="*80 + "\n")

    return created


def precreate_zarr3_metadata_safely(output_path, level, shape, dtype, use_shard,
                                    shard_shape, chunk_shape, use_ome_structure=True,
                                    use_fortran_order=False, use_v2_encoding=False,
                                    force_precreate=False, axes_order=None, compression=None,
                                    level_prefix=None):
    """
    Pre-create zarr.json metadata to avoid worker race conditions.

    Args:
        compression: Optional compression codec dict from level 0.
                    Format: {'name': 'zstd', 'configuration': {'level': N}}
                    If None, defaults to zstd level 5.
        level_prefix: Prefix for level naming ("s" for s0/s1 or "" for 0/1).
                     If None, auto-detects from existing levels or defaults to "s".
    """
    import tensorstore as ts
    from .tensorstore_utils import zarr3_store_spec

    if not (use_fortran_order or force_precreate):
        return False

    # Detect or use provided prefix
    if level_prefix is None:
        level_prefix = detect_level_format(output_path)
    level_name = get_level_name(level, level_prefix)

    level_path = level_name if use_ome_structure else None

    store_spec = zarr3_store_spec(
        path=output_path,
        shape=shape,
        dtype=dtype,
        use_shard=use_shard,
        level_path=level_path or level_name,
        use_ome_structure=use_ome_structure,
        custom_shard_shape=shard_shape,
        custom_chunk_shape=chunk_shape,
        use_v2_encoding=use_v2_encoding,
        use_fortran_order=use_fortran_order,
        axes_order=axes_order,
        compression=compression
    )

    store = ts.open(store_spec, create=True, delete_existing=True).result()
    store = None

    metadata_path = os.path.join(output_path, level_path or level_name, "zarr.json")
    print(f"Pre-created Zarr3 metadata at: {metadata_path}")
    print(f"  Reason: {'F-order codec' if use_fortran_order else 'Forced pre-creation'}")

    return True


def precreate_zarr3_output(output_path, level, output_shape, shard_shape, chunk_shape,
                           dtype, use_ome_structure=True, use_v2_encoding=False,
                           axes_order=None, check_exists=False, create_metadata=True,
                           compression=None, level_prefix=None):
    """
    UNIFIED function to pre-create both shard directories AND zarr.json metadata.

    Args:
        compression: Optional compression codec dict from level 0.
                    Format: {'name': 'zstd', 'configuration': {'level': N}}
                    If None, defaults to zstd level 5.
        level_prefix: Prefix for level naming ("s" for s0/s1 or "" for 0/1).
                     If None, auto-detects from existing levels or defaults to "s".
    """
    result = {'dirs_created': False, 'metadata_created': False}

    # Detect or use provided prefix
    if level_prefix is None:
        level_prefix = detect_level_format(output_path)
    level_name = get_level_name(level, level_prefix)

    print("\n" + "="*80)
    print(f"PRE-CREATING ZARR3 OUTPUT FOR {level_name}")
    print("="*80)

    if check_exists:
        if use_ome_structure:
            base_check_path = os.path.join(output_path, level_name, "c", "0")
        else:
            base_check_path = os.path.join(output_path, "c", "0")

        if os.path.exists(base_check_path):
            print(f"Shard directories already exist for {level_name}, skipping redundant creation")

            metadata_path = os.path.join(output_path, level_name if use_ome_structure else "", "zarr.json")
            if os.path.exists(metadata_path):
                print(f"Metadata also exists, all pre-creation complete")
                return result
            else:
                print(f"Metadata missing, will create it")

    print(f"\n1. Creating shard directory structure...")
    precreate_shard_directories(
        output_path=output_path,
        level=level,
        output_shape=output_shape,
        shard_shape=shard_shape,
        use_ome_structure=use_ome_structure,
        level_prefix=level_prefix
    )
    result['dirs_created'] = True

    if create_metadata:
        print(f"\n2. Pre-creating zarr.json metadata to prevent race conditions...")
        metadata_created = precreate_zarr3_metadata_safely(
            output_path=output_path,
            level=level,
            shape=output_shape,
            dtype=dtype,
            use_shard=True,
            shard_shape=shard_shape,
            chunk_shape=chunk_shape,
            use_ome_structure=use_ome_structure,
            use_fortran_order=False,
            use_v2_encoding=use_v2_encoding,
            force_precreate=True,
            axes_order=axes_order,
            compression=compression,
            level_prefix=level_prefix
        )
        result['metadata_created'] = metadata_created

        if metadata_created:
            print(f"Metadata pre-creation complete")
        else:
            print(f"Metadata pre-creation was skipped")
    else:
        print(f"\n2. Skipping metadata creation (create_metadata=False)")

    print("="*80 + "\n")

    return result


# ============================================================================
# OME-NGFF Labels Support Functions
# ============================================================================

def generate_default_label_colors(num_labels, alpha=255):
    """
    Generate distinct colors for label values using golden ratio.

    Args:
        num_labels: Number of unique label values (excluding 0/background)
        alpha: Alpha value (0-255, default: 255 for opaque)

    Returns:
        list: List of dicts with label-value and rgba

    Example:
        [
            {"label-value": 1, "rgba": [174, 25, 242, 255]},
            {"label-value": 2, "rgba": [111, 206, 59, 255]},
            ...
        ]
    """
    import colorsys

    colors = []
    for i in range(1, num_labels + 1):
        # Generate distinct hues using golden ratio (0.618...)
        # This distributes colors evenly around the color wheel
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.7
        value = 0.9

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgba = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha]

        colors.append({
            "label-value": i,
            "rgba": rgba
        })

    return colors


def is_segmentation_dtype(dtype_str):
    """
    Check if dtype suggests segmentation/label data.

    Segmentation data typically uses integer types for label IDs:
    - uint64: Most common for large segmentations (connectomics)
    - uint32: Common for smaller segmentations
    - uint16: Smaller label counts
    - int64/int32: Signed variants

    Args:
        dtype_str: Data type string (e.g., "uint64", "uint8")

    Returns:
        bool: True if dtype suggests segmentation data
    """
    segmentation_dtypes = ['uint64', 'int64', 'uint32', 'int32']
    return dtype_str in segmentation_dtypes


def create_image_label_metadata(label_colors=None, source_path=None, ngff_version="0.5", num_default_colors=256):
    """
    Create the image-label metadata object for OME-NGFF labels.

    Args:
        label_colors: Optional list of dicts with label-value and rgba.
                     If None, generates default colors.
        source_path: Relative path to parent image (e.g., "../../" if label is at image.zarr/labels/seg/)
                    If None, no source reference is added.
        ngff_version: OME-NGFF version (default: "0.5")
        num_default_colors: Number of default colors to generate if label_colors is None

    Returns:
        dict: image-label metadata object

    Example output:
        {
            "version": "0.5",
            "colors": [
                {"label-value": 1, "rgba": [174, 25, 242, 255]},
                ...
            ],
            "source": {"image": "../../"}
        }
    """
    image_label = {
        "version": ngff_version
    }

    # Add colors (generate defaults if not provided)
    if label_colors is None:
        label_colors = generate_default_label_colors(num_default_colors)
    image_label["colors"] = label_colors

    # Add source reference if provided
    if source_path:
        image_label["source"] = {"image": source_path}

    return image_label
