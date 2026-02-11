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


def create_zarr3_ome_metadata(ome_xml, array_shape, image_name, pixel_sizes=None, axes_order=None, include_omero=False, is_label=False, label_colors=None):
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
    """
    def get_axis_type(axis_name):
        if axis_name == 'c':
            return 'channel'
        elif axis_name in ['t', 'v']:
            return 'time'
        else:
            return 'space'

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
            output_name = 't' if axis_name == 'v' else axis_name
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
            "path": "s0",
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
    """Auto-detect the maximum level by scanning for s* directories at root level."""
    if not os.path.exists(output_path):
        return None

    pattern = os.path.join(output_path, 's*')
    level_dirs = glob.glob(pattern)

    if not level_dirs:
        return None

    levels = []
    for level_dir in level_dirs:
        dirname = os.path.basename(level_dir)
        if dirname.startswith('s') and dirname[1:].isdigit():
            levels.append(int(dirname[1:]))

    if not levels:
        return None

    return max(levels)


def update_ome_multiscale_metadata(zarr_path, max_level=4):
    """
    Update OME-ZARR metadata to include all multiscale levels s0 through max_level.
    """
    zarr_json_path = os.path.join(zarr_path, "zarr.json")

    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)

    multiscales = metadata["attributes"]["ome"]["multiscales"][0]
    s0_scale_factors = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]

    downsampling_factors = metadata.get("attributes", {}).get("custom", {}).get("downsampling_factors", None)

    s0_metadata_path = os.path.join(zarr_path, "s0", "zarr.json")
    with open(s0_metadata_path, 'r') as f:
        s0_meta = json.load(f)
    s0_shape = s0_meta.get('shape')

    if downsampling_factors:
        print("Using downsampling factors from custom metadata (incremental method)")
        use_factors = True
        previous_scale = s0_scale_factors
    else:
        print("No downsampling factors found - using dimension ratio method")
        use_factors = False

    datasets = []

    for level in range(max_level + 1):
        level_metadata_path = os.path.join(zarr_path, f"s{level}", "zarr.json")

        if not os.path.exists(level_metadata_path):
            print(f"Warning: s{level}/zarr.json not found, skipping level {level}")
            break

        if level == 0:
            datasets.append(multiscales["datasets"][0])
            continue

        with open(level_metadata_path, 'r') as f:
            level_meta = json.load(f)
        level_shape = level_meta.get('shape')

        if use_factors:
            factor_key = f"s{level-1}_to_s{level}"
            level_factors = downsampling_factors.get(factor_key)
            if level_factors:
                level_scale = [prev * factor for prev, factor in zip(previous_scale, level_factors)]
            else:
                print(f"Warning: Missing factors for {factor_key}, falling back to ratio method")
                level_scale = [s0_scale_factors[i] * (s0_shape[i] / level_shape[i]) for i in range(len(s0_shape))]
            previous_scale = level_scale
        else:
            level_scale = [s0_scale_factors[i] * (s0_shape[i] / level_shape[i]) for i in range(len(s0_shape))]

        coordinate_transformations = [{"type": "scale", "scale": level_scale}]

        # Add translation transform for Neuroglancer compatibility
        translation = [0.5 * (scale - s0_scale_factors[i]) for i, scale in enumerate(level_scale)]
        if any(t != 0 for t in translation):
            coordinate_transformations.append({"type": "translation", "translation": translation})

        datasets.append({
            "path": f"s{level}",
            "coordinateTransformations": coordinate_transformations
        })

    multiscales["datasets"] = datasets

    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated OME metadata with {len(datasets)} levels")


def update_ome_multiscale_metadata_zarr2(zarr_path, max_level=4):
    """Update OME-ZARR metadata for zarr2 format (.zattrs)."""
    zattrs_path = os.path.join(zarr_path, ".zattrs")

    with open(zattrs_path, 'r') as f:
        metadata = json.load(f)

    multiscales = metadata.get("multiscales", [{}])[0]
    s0_scale_factors = multiscales.get("datasets", [{}])[0].get("coordinateTransformations", [{}])[0].get("scale", [1.0, 1.0, 1.0])

    zarray_path = os.path.join(zarr_path, "s0", ".zarray")
    with open(zarray_path, 'r') as f:
        s0_meta = json.load(f)
    s0_shape = s0_meta.get('shape')

    datasets = []

    for level in range(max_level + 1):
        level_zarray_path = os.path.join(zarr_path, f"s{level}", ".zarray")

        if not os.path.exists(level_zarray_path):
            print(f"Warning: s{level}/.zarray not found, skipping level {level}")
            break

        if level == 0:
            datasets.append(multiscales.get("datasets", [{}])[0])
            continue

        with open(level_zarray_path, 'r') as f:
            level_meta = json.load(f)
        level_shape = level_meta.get('shape')

        level_scale = [s0_scale_factors[i] * (s0_shape[i] / level_shape[i]) for i in range(len(s0_shape))]

        coordinate_transformations = [{"type": "scale", "scale": level_scale}]

        datasets.append({
            "path": f"s{level}",
            "coordinateTransformations": coordinate_transformations
        })

    multiscales["datasets"] = datasets
    metadata["multiscales"] = [multiscales]

    with open(zattrs_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated Zarr2 OME metadata with {len(datasets)} levels")


def update_ome_metadata_if_needed(output_path, use_ome_structure):
    """
    Update OME-Zarr metadata if OME structure is used and multiscale levels exist.
    Supports both zarr2 (.zattrs) and zarr3 (zarr.json) formats.
    """
    if not use_ome_structure:
        return

    max_level = auto_detect_max_level(output_path)
    if max_level is None:
        print("No multiscale levels detected - skipping metadata update")
        return

    if max_level == 0:
        print("Only s0 level detected - no multiscale metadata update needed")
        return

    zarr3_metadata = os.path.join(output_path, 'zarr.json')
    zarr2_metadata = os.path.join(output_path, '.zattrs')

    try:
        if os.path.exists(zarr3_metadata):
            print(f"Updating zarr3 OME metadata for {output_path} with levels s0-s{max_level}")
            update_ome_multiscale_metadata(output_path, max_level=max_level)
            print("OME metadata updated successfully!")
        elif os.path.exists(zarr2_metadata):
            print(f"Updating zarr2 OME metadata for {output_path} with levels s0-s{max_level}")
            update_ome_multiscale_metadata_zarr2(output_path, max_level=max_level)
            print("OME metadata updated successfully!")
        else:
            print("Warning: No zarr metadata file found (.zattrs or zarr.json) - skipping metadata update")
    except Exception as e:
        print(f"Warning: Failed to update OME metadata: {e}")


def precreate_shard_directories(output_path, level, output_shape, shard_shape, use_ome_structure=True):
    """
    Pre-create all shard directory structures to avoid race conditions with parallel workers.
    """
    import time

    print("\n" + "="*80)
    print(f"PRE-CREATING SHARD DIRECTORIES FOR s{level}")
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
        base_shard_path = os.path.join(output_path, f"s{level}", "c")
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
                                    force_precreate=False, axes_order=None):
    """
    Pre-create zarr.json metadata to avoid worker race conditions.
    """
    import tensorstore as ts
    from .tensorstore_utils import zarr3_store_spec

    if not (use_fortran_order or force_precreate):
        return False

    level_path = f"s{level}" if use_ome_structure else None

    store_spec = zarr3_store_spec(
        path=output_path,
        shape=shape,
        dtype=dtype,
        use_shard=use_shard,
        level_path=level_path or f"s{level}",
        use_ome_structure=use_ome_structure,
        custom_shard_shape=shard_shape,
        custom_chunk_shape=chunk_shape,
        use_v2_encoding=use_v2_encoding,
        use_fortran_order=use_fortran_order,
        axes_order=axes_order
    )

    store = ts.open(store_spec, create=True, delete_existing=True).result()
    store = None

    metadata_path = os.path.join(output_path, level_path or f"s{level}", "zarr.json")
    print(f"Pre-created Zarr3 metadata at: {metadata_path}")
    print(f"  Reason: {'F-order codec' if use_fortran_order else 'Forced pre-creation'}")

    return True


def precreate_zarr3_output(output_path, level, output_shape, shard_shape, chunk_shape,
                           dtype, use_ome_structure=True, use_v2_encoding=False,
                           axes_order=None, check_exists=False, create_metadata=True):
    """
    UNIFIED function to pre-create both shard directories AND zarr.json metadata.
    """
    result = {'dirs_created': False, 'metadata_created': False}

    print("\n" + "="*80)
    print(f"PRE-CREATING ZARR3 OUTPUT FOR s{level}")
    print("="*80)

    if check_exists:
        if use_ome_structure:
            base_check_path = os.path.join(output_path, f"s{level}", "c", "0")
        else:
            base_check_path = os.path.join(output_path, "c", "0")

        if os.path.exists(base_check_path):
            print(f"Shard directories already exist for s{level}, skipping redundant creation")

            metadata_path = os.path.join(output_path, f"s{level}" if use_ome_structure else "", "zarr.json")
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
        use_ome_structure=use_ome_structure
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
            axes_order=axes_order
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
