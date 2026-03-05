# Copied from tensorswitch/utils.py for v2 independence
"""
Pyramid utility functions for tensorswitch_v2.

This module contains pyramid planning functions including:
- Anisotropic downsample factor calculation
- Number of multiscale levels calculation
- Pyramid plan calculation

Credit: Yurii Zubov (Janelia CellMap Team) for the anisotropic downsampling algorithm
Reference: https://github.com/janelia-cellmap/zarrify
"""

import os
import json
import numpy as np
import math


def calculate_anisotropic_downsample_factors(voxel_sizes, axes_names, min_ratio=0.5, max_ratio=2.0, use_anisotropic=True):
    """
    Calculate adaptive downsampling factors based on voxel size aspect ratios.

    Algorithm credit: Yurii Zubov (Janelia CellMap Team)
    Based on: https://github.com/janelia-cellmap/zarrify/blob/main/src/zarrify/utils/volume.py#L91

    Goal: Make voxel sizes more homogeneous with each downsampling iteration.

    Args:
        voxel_sizes: List of voxel sizes (e.g., [1.0, 0.116, 0.116] for c, y, x)
        axes_names: List of axis names (e.g., ['c', 'y', 'x'])
        min_ratio: Minimum ratio threshold (default 0.5)
        max_ratio: Maximum ratio threshold (default 2.0)
        use_anisotropic: If False, always use (1, 2, 2, ...) pattern

    Returns:
        List of downsampling factors (e.g., [1, 2, 2] for c, y, x)
    """
    NON_SPATIAL_AXES = ['c', 't', 'v']

    if not use_anisotropic:
        return [1 if axis in NON_SPATIAL_AXES else 2 for axis in axes_names]

    spatial_data = [(axis, voxel_sizes[i]) for i, axis in enumerate(axes_names) if axis not in NON_SPATIAL_AXES]

    if not spatial_data:
        return [1] * len(axes_names)

    axes, dimensions = zip(*spatial_data)

    if len(dimensions) == 1:
        factors = [2]
    else:
        ratios = []
        for i, dim in enumerate(dimensions):
            dim_ratios = tuple(dim / dimensions[j] for j in range(len(dimensions)) if j != i)
            ratios.append(dim_ratios)

        factors = []
        for (i, dim_ratios) in enumerate(ratios):
            if all(ratio >= max_ratio for ratio in dim_ratios):
                factors = [2] * len(ratios)
                factors[i] = 1
                break
            elif all(ratio <= min_ratio for ratio in dim_ratios):
                factors = [1] * len(ratios)
                factors[i] = 2
                break
            else:
                factors.append(2)

    spatial_factors = {k: v for k, v in zip(axes, factors)}
    return [1 if axis in NON_SPATIAL_AXES else spatial_factors[axis] for axis in axes_names]


def calculate_num_multiscale_levels(shape, axes_names, voxel_sizes, chunk_shape=None, dtype_size=2,
                                     min_array_nbytes=None, min_array_shape=None, shard_shape=None, use_anisotropic=True):
    """
    Calculate how many multiscale levels to generate.

    Stops when either:
    - Rule 1: array_nbytes < min_array_nbytes (total volume too small)
    - Rule 2: all(shape[i] < min_array_shape[i]) (all dimensions below threshold)
    - Rule 3: cumulative magnification drifts >4% from power-of-2 (WebKnossos compatibility)

    Args:
        shape: Array shape (e.g., [3, 1196, 31416, 17635] for C, Z, Y, X)
        axes_names: List of axis names (e.g., ['c', 'z', 'y', 'x'])
        voxel_sizes: Initial voxel sizes (e.g., [1.0, 1.0, 0.116, 0.116])
        chunk_shape: Chunk shape for calculating defaults
        dtype_size: Bytes per element (default 2 for uint16)
        min_array_nbytes: Stop when array size < this (default: chunk_nbytes)
        min_array_shape: Stop when all dims < this (default: chunk_shape)
        shard_shape: (Deprecated, not used)
        use_anisotropic: Use anisotropic downsampling factors

    Returns:
        int: Total number of levels (including s0)
    """
    current_shape = list(shape)
    current_voxel_sizes = list(voxel_sizes)

    if chunk_shape is not None:
        if min_array_nbytes is None:
            chunk_nbytes = 1
            for dim in chunk_shape:
                chunk_nbytes *= dim
            chunk_nbytes *= dtype_size
            min_array_nbytes = chunk_nbytes

        if min_array_shape is None:
            min_array_shape = list(chunk_shape)

    if min_array_nbytes is None:
        min_array_nbytes = 524288  # 512KB default
    if min_array_shape is None:
        min_array_shape = [32] * len(shape)

    level = 0
    NON_SPATIAL_AXES = ['c', 't', 'v']

    while True:
        level += 1

        factors = calculate_anisotropic_downsample_factors(
            current_voxel_sizes,
            axes_names,
            use_anisotropic=use_anisotropic
        )

        new_shape = [max(1, dim // factor) for dim, factor in zip(current_shape, factors)]
        new_voxel_sizes = [voxel * factor for voxel, factor in zip(current_voxel_sizes, factors)]

        # Rule 1: Check array_nbytes
        array_nbytes = dtype_size
        for dim in new_shape:
            array_nbytes *= dim

        if array_nbytes < min_array_nbytes:
            return level - 1

        # Rule 2: Check if ALL dimensions below threshold
        if all(new_shape[i] < min_array_shape[i] for i in range(len(new_shape))):
            return level - 1

        # Rule 3: WebKnossos power-of-2 validation
        max_power_of_2_error = 0.0
        for i in range(len(shape)):
            if axes_names[i] in NON_SPATIAL_AXES:
                continue

            cumulative_mag = shape[i] / new_shape[i]

            if cumulative_mag > 0:
                log2_mag = math.log2(cumulative_mag)
                expected_power = round(log2_mag)
                expected_mag = 2 ** expected_power

                error = abs(cumulative_mag - expected_mag) / expected_mag
                max_power_of_2_error = max(max_power_of_2_error, error)

        if max_power_of_2_error > 0.04:
            print(f"Stopping at level {level-1}: Cumulative mag error {max_power_of_2_error*100:.1f}% exceeds WebKnossos 4% tolerance")
            return level - 1

        current_shape = new_shape
        current_voxel_sizes = new_voxel_sizes

        if level > 20:
            print(f"Warning: Stopped at level {level} (safety limit)")
            return level

    return level


def calculate_pyramid_plan(s0_path, min_array_nbytes=None, min_array_shape=None):
    """
    Pre-calculate entire multiscale pyramid plan before submitting cluster jobs.

    This function mathematically predicts voxel sizes for all levels WITHOUT creating data.
    Enables cluster submission with pre-determined anisotropic downsampling factors.

    Args:
        s0_path: Path to s0 level (either zarr3 s0/ or zarr2 multiscale/s0/)
        min_array_nbytes: Stop when array size < this (default: chunk_nbytes from metadata)
        min_array_shape: Stop when all dims < this (default: chunk_shape from metadata)

    Returns:
        dict with keys:
            'format': 'zarr3' or 'zarr2'
            'shape': s0 shape
            'voxel_sizes': s0 voxel sizes
            'axes_names': dimension names
            'chunk_shape': chunk shape
            'dtype_size': bytes per element
            'num_levels': total number of levels needed
            'pyramid_plan': list of dicts for each level
    """
    zarr3_metadata_path = os.path.join(s0_path, "zarr.json")
    zarr2_metadata_path = os.path.join(s0_path, ".zarray")

    is_zarr3 = os.path.exists(zarr3_metadata_path)
    is_zarr2 = os.path.exists(zarr2_metadata_path)

    if not is_zarr3 and not is_zarr2:
        raise FileNotFoundError(
            f"Cannot find zarr metadata at {s0_path}\n"
            f"Expected either:\n"
            f"  - Zarr3: {zarr3_metadata_path}\n"
            f"  - Zarr2: {zarr2_metadata_path}"
        )

    format_type = "zarr3" if is_zarr3 else "zarr2"
    chunk_shape = None
    dtype_size = 2  # default uint16

    if is_zarr3:
        with open(zarr3_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')
        axes_names = metadata.get('dimension_names')

        shard_shape = metadata.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape')
        inner_chunk_shape = None

        codecs = metadata.get('codecs', [])
        if codecs and codecs[0].get('name') == 'sharding_indexed':
            inner_chunk_shape = codecs[0].get('configuration', {}).get('chunk_shape')

        chunk_shape = inner_chunk_shape if inner_chunk_shape is not None else shard_shape

        dtype_str = metadata.get('data_type', 'uint16')
        dtype = np.dtype(dtype_str)
        dtype_size = dtype.itemsize

        root_path = os.path.dirname(s0_path)
        root_zarr_json = os.path.join(root_path, "zarr.json")
        voxel_sizes = None

        if os.path.exists(root_zarr_json):
            with open(root_zarr_json, 'r') as f:
                root_metadata = json.load(f)
                if 'attributes' in root_metadata and 'ome' in root_metadata['attributes'] and 'multiscales' in root_metadata['attributes']['ome']:
                    multiscales = root_metadata['attributes']['ome']['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break
                elif 'attributes' in root_metadata and 'multiscales' in root_metadata['attributes']:
                    multiscales = root_metadata['attributes']['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break

    else:  # zarr2
        with open(zarr2_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')
        chunk_shape = metadata.get('chunks')

        shard_shape = chunk_shape
        inner_chunk_shape = None

        dtype_str = metadata.get('dtype', '<u2')
        dtype = np.dtype(dtype_str)
        dtype_size = dtype.itemsize

        zattrs_path = os.path.join(s0_path, ".zattrs")
        axes_names = None
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                attrs = json.load(f)
                axes_names = attrs.get('_ARRAY_DIMENSIONS')

        multiscale_path = os.path.dirname(s0_path)
        multiscale_zattrs = os.path.join(multiscale_path, ".zattrs")
        voxel_sizes = None
        if os.path.exists(multiscale_zattrs):
            with open(multiscale_zattrs, 'r') as f:
                attrs = json.load(f)
                if 'multiscales' in attrs and len(attrs['multiscales']) > 0:
                    multiscales = attrs['multiscales'][0]
                    # Extract axes from OME-NGFF multiscales (if not found in level .zattrs)
                    if not axes_names and 'axes' in multiscales:
                        axes_names = [ax.get('name', f'dim_{i}') for i, ax in enumerate(multiscales['axes'])]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break

    # Validate metadata
    if not shape:
        raise ValueError(f"Could not extract shape from {s0_path}")

    if not voxel_sizes:
        voxel_sizes = [1.0] * len(shape)
        print(f"Warning: Could not extract voxel sizes, using default: {voxel_sizes}")

    if not axes_names:
        axes_names = ['z', 'y', 'x'][-len(shape):]
        print(f"Warning: Could not extract axes_names, using default: {axes_names}")

    if not chunk_shape:
        chunk_shape = [32] * len(shape)
        print(f"Warning: Could not extract chunk_shape, using default: {chunk_shape}")

    num_levels = calculate_num_multiscale_levels(
        shape, axes_names, voxel_sizes,
        chunk_shape=chunk_shape,
        dtype_size=dtype_size,
        min_array_nbytes=min_array_nbytes,
        min_array_shape=min_array_shape,
        shard_shape=shard_shape
    )

    pyramid_plan = []
    current_voxel_sizes = voxel_sizes.copy()
    current_shape = shape.copy()

    original_shard_shape = shard_shape.copy() if shard_shape else None
    original_inner_chunk_shape = inner_chunk_shape.copy() if inner_chunk_shape else None

    for level in range(1, num_levels + 1):
        factor = calculate_anisotropic_downsample_factors(current_voxel_sizes, axes_names)

        predicted_voxel_sizes = [v * f for v, f in zip(current_voxel_sizes, factor)]
        predicted_shape = [max(1, s // f) for s, f in zip(current_shape, factor)]

        # For Zarr3 with sharding: use inner_chunk_shape
        # For Zarr2 (no sharding): fall back to chunk_shape from source
        scaled_chunk = original_inner_chunk_shape.copy() if original_inner_chunk_shape else (chunk_shape.copy() if chunk_shape else None)
        scaled_shard = original_shard_shape.copy() if original_shard_shape else None

        pyramid_plan.append({
            "level": level,
            "factor": factor,
            "predicted_voxel_sizes": predicted_voxel_sizes,
            "predicted_shape": predicted_shape,
            "shard_shape": scaled_shard,
            "chunk_shape": scaled_chunk
        })

        current_voxel_sizes = predicted_voxel_sizes
        current_shape = predicted_shape

    return {
        'format': format_type,
        'shape': shape,
        'voxel_sizes': voxel_sizes,
        'axes_names': axes_names,
        'chunk_shape': chunk_shape,
        'shard_shape': shard_shape,
        'inner_chunk_shape': inner_chunk_shape,
        'dtype_size': dtype_size,
        'num_levels': num_levels,
        'pyramid_plan': pyramid_plan
    }


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
