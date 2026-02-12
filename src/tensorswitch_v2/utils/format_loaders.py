# Copied from tensorswitch/utils.py for v2 independence
"""
File format loaders for tensorswitch_v2.

This module contains functions to load various microscopy file formats:
- TIFF (including OME-TIFF and ImageJ TIFF)
- ND2 (Nikon)
- IMS (Imaris)
- CZI (Zeiss)
- Precomputed (Neuroglancer)

All voxel sizes are returned in nanometers (standard unit).
"""

import os
import numpy as np
import dask.array as da


def convert_to_nanometers(value: float, unit: str) -> float:
    """
    Convert a length value to nanometers based on its unit.

    Args:
        value: The numeric value to convert
        unit: The unit string (e.g., 'nm', 'um', 'µm', 'micrometer', 'mm', 'm')

    Returns:
        float: Value converted to nanometers

    Examples:
        >>> convert_to_nanometers(0.116, 'micrometer')
        116.0
        >>> convert_to_nanometers(9.0, 'nm')
        9.0
        >>> convert_to_nanometers(1.0, 'mm')
        1000000.0
    """
    if value is None:
        return 1.0

    unit_lower = unit.lower().strip() if unit else ''

    # Nanometers - no conversion
    if unit_lower in ['nm', 'nanometer', 'nanometers', 'nanometre', 'nanometres']:
        return float(value)
    # Also check for enum-style format (e.g., 'unitslength.nanometer')
    if 'nanometer' in unit_lower or 'nanometre' in unit_lower:
        return float(value)

    # Micrometers - multiply by 1000
    if unit_lower in ['um', 'µm', 'micrometer', 'micrometers', 'micrometre', 'micrometres', 'micron', 'microns']:
        return float(value) * 1000.0
    # Also check for enum-style format (e.g., 'unitslength.micrometer')
    if 'micrometer' in unit_lower or 'micrometre' in unit_lower or 'micron' in unit_lower:
        return float(value) * 1000.0

    # Millimeters - multiply by 1,000,000
    if unit_lower in ['mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres']:
        return float(value) * 1_000_000.0
    if 'millimeter' in unit_lower or 'millimetre' in unit_lower:
        return float(value) * 1_000_000.0

    # Meters - multiply by 1,000,000,000 (but not if already matched above)
    if unit_lower in ['m', 'meter', 'meters', 'metre', 'metres']:
        return float(value) * 1_000_000_000.0

    # Unknown unit - assume micrometers (common default in microscopy)
    if unit_lower:
        print(f"Warning: Unknown unit '{unit}', assuming micrometers")
        return float(value) * 1000.0

    # No unit specified - assume already in nanometers or dimensionless
    return float(value)


def load_tiff_stack(folder_or_file):
    """
    Load TIFF data lazily to avoid memory issues with large files.

    Args:
        folder_or_file: Path to single TIFF file or folder containing TIFF files

    Returns:
        dask.array.Array: Lazy-loaded dask array
    """
    import tifffile
    from dask_image import imread as dask_imread

    if os.path.isfile(folder_or_file):
        # Single TIFF file - use lazy loading
        tiff_store = tifffile.imread(folder_or_file, aszarr=True)
        return da.from_zarr(tiff_store, zarr_format=2)
    else:
        # Multiple TIFF files
        return dask_imread.imread(folder_or_file + "/*.tiff")


def extract_tiff_ome_metadata(tiff_file):
    """
    Extract metadata from TIFF file (OME-TIFF or ImageJ TIFF).

    Args:
        tiff_file: Path to TIFF file

    Returns:
        tuple: (ome_xml, voxel_sizes)
            ome_xml: OME-XML string or None
            voxel_sizes: {'x': float, 'y': float, 'z': float} in nanometers, or None
    """
    import tifffile

    if not os.path.isfile(tiff_file):
        raise ValueError(f"TIFF file does not exist: {tiff_file}")

    try:
        with tifffile.TiffFile(tiff_file) as tif:
            ome_xml = None
            voxel_sizes = None

            # Try to get OME-XML from TIFF tags (OME-TIFF)
            if tif.ome_metadata:
                ome_xml = tif.ome_metadata
                try:
                    from ome_types import from_xml
                    ome = from_xml(ome_xml)
                    if ome.images and ome.images[0].pixels:
                        pixels = ome.images[0].pixels
                        # OME-TIFF typically stores in micrometers, get unit if available
                        x_unit = str(pixels.physical_size_x_unit) if hasattr(pixels, 'physical_size_x_unit') and pixels.physical_size_x_unit else 'micrometer'
                        y_unit = str(pixels.physical_size_y_unit) if hasattr(pixels, 'physical_size_y_unit') and pixels.physical_size_y_unit else 'micrometer'
                        z_unit = str(pixels.physical_size_z_unit) if hasattr(pixels, 'physical_size_z_unit') and pixels.physical_size_z_unit else 'micrometer'

                        voxel_sizes = {
                            'x': convert_to_nanometers(pixels.physical_size_x, x_unit) if pixels.physical_size_x else 1.0,
                            'y': convert_to_nanometers(pixels.physical_size_y, y_unit) if pixels.physical_size_y else 1.0,
                            'z': convert_to_nanometers(pixels.physical_size_z, z_unit) if pixels.physical_size_z else 1.0
                        }
                        return ome_xml, voxel_sizes
                except Exception as e:
                    print(f"Warning: Could not extract voxel sizes from OME-TIFF metadata: {e}")
                return ome_xml, None

            # Fall back to ImageJ TIFF metadata
            if tif.is_imagej and tif.imagej_metadata:
                imagej_metadata = tif.imagej_metadata

                z_spacing = imagej_metadata.get('spacing')
                unit = imagej_metadata.get('unit', 'micron')

                if z_spacing is not None:
                    # Convert to nanometers using helper
                    z_spacing_nm = convert_to_nanometers(z_spacing, unit)

                    # Default XY resolution if not in tags (1.0 nm as placeholder)
                    voxel_sizes = {
                        'x': 1.0,
                        'y': 1.0,
                        'z': z_spacing_nm
                    }

                    return None, voxel_sizes

            return None, None
    except Exception as e:
        print(f"Warning: Error extracting TIFF metadata: {e}")
        return None, None


def load_nd2_stack(nd2_file):
    """
    Load ND2 data as a dask array.

    Args:
        nd2_file: Path to ND2 file

    Returns:
        dask.array.Array: Lazy-loaded dask array
    """
    import nd2

    if not os.path.isfile(nd2_file):
        raise ValueError(f"ND2 file does not exist: {nd2_file}")

    nd2_handle = nd2.ND2File(nd2_file)
    dask_array = nd2_handle.to_dask()
    # Note: Don't close the handle here as dask needs it for lazy loading
    return dask_array


def extract_nd2_ome_metadata(nd2_file):
    """
    Extract OME metadata and voxel sizes from ND2 file.

    Args:
        nd2_file: Path to ND2 file

    Returns:
        tuple: (ome_xml, voxel_sizes)
            ome_xml: OME-XML string or None
            voxel_sizes: {'x': float, 'y': float, 'z': float} in nanometers, or None
    """
    import nd2

    if not os.path.isfile(nd2_file):
        raise ValueError(f"ND2 file does not exist: {nd2_file}")

    with nd2.ND2File(nd2_file) as f:
        ome_xml = f.ome_metadata().to_xml()

        voxel_sizes = None
        if ome_xml:
            try:
                from ome_types import from_xml
                ome = from_xml(ome_xml)
                if ome.images and ome.images[0].pixels:
                    pixels = ome.images[0].pixels
                    # ND2/OME typically stores in micrometers, get unit if available
                    x_unit = str(pixels.physical_size_x_unit) if hasattr(pixels, 'physical_size_x_unit') and pixels.physical_size_x_unit else 'micrometer'
                    y_unit = str(pixels.physical_size_y_unit) if hasattr(pixels, 'physical_size_y_unit') and pixels.physical_size_y_unit else 'micrometer'
                    z_unit = str(pixels.physical_size_z_unit) if hasattr(pixels, 'physical_size_z_unit') and pixels.physical_size_z_unit else 'micrometer'

                    voxel_sizes = {
                        'x': convert_to_nanometers(pixels.physical_size_x, x_unit) if pixels.physical_size_x else 1.0,
                        'y': convert_to_nanometers(pixels.physical_size_y, y_unit) if pixels.physical_size_y else 1.0,
                        'z': convert_to_nanometers(pixels.physical_size_z, z_unit) if pixels.physical_size_z else 1.0
                    }
            except Exception as e:
                print(f"Warning: Could not extract voxel sizes from ND2 OME metadata: {e}")
                voxel_sizes = None

        return ome_xml, voxel_sizes


def load_ims_stack(ims_file):
    """
    Load IMS data as a dask array using h5py, trimmed to actual data bounds.

    Args:
        ims_file: Path to IMS file

    Returns:
        tuple: (dask_array, h5_file_handle)
    """
    import h5py

    if not os.path.isfile(ims_file):
        raise ValueError(f"IMS file does not exist: {ims_file}")

    # Extract metadata first to check for padding
    metadata, _ = extract_ims_metadata(ims_file)
    actual_z_slices = None

    if 'Z' in metadata:
        try:
            actual_z_slices = int(metadata['Z'])
            print(f"Metadata indicates {actual_z_slices} actual Z-slices")
        except (ValueError, TypeError):
            print("Warning: Could not parse Z-dimension from metadata")

    h5_file = h5py.File(ims_file, 'r')

    dataset_group = h5_file['DataSet']
    resolution_group = dataset_group['ResolutionLevel 0']
    timepoint_group = resolution_group['TimePoint 0']

    channel_keys = [key for key in timepoint_group.keys() if key.startswith('Channel')]
    channel_keys.sort()

    if not channel_keys:
        raise ValueError(f"No channels found in IMS file: {ims_file}")

    print(f"Found {len(channel_keys)} channels: {channel_keys}")

    channel_arrays = []
    for channel_key in channel_keys:
        channel_group = timepoint_group[channel_key]
        data_dataset = channel_group['Data']

        print(f"{channel_key} full shape: {data_dataset.shape}, dtype: {data_dataset.dtype}")

        if actual_z_slices is not None and actual_z_slices < data_dataset.shape[0]:
            print(f"PADDING DETECTED: Trimming Z from {data_dataset.shape[0]} to {actual_z_slices} slices")
            trimmed_dataset = data_dataset[:actual_z_slices, :, :]
            channel_array = da.from_array(trimmed_dataset, chunks='auto')
            print(f"{channel_key} trimmed shape: {trimmed_dataset.shape}")
        else:
            if actual_z_slices is None:
                print(f"{channel_key}: No metadata Z-dimension found, using full array")
            else:
                print(f"{channel_key}: No padding detected")
            channel_array = da.from_array(data_dataset, chunks='auto')

        channel_arrays.append(channel_array)

    if len(channel_arrays) > 1:
        stacked_array = da.stack(channel_arrays, axis=0)
        print(f"Stacked array shape (CZYX): {stacked_array.shape}")
    else:
        stacked_array = channel_arrays[0]
        print(f"Single channel array shape (ZYX): {stacked_array.shape}")

    return stacked_array, h5_file


def extract_ims_metadata(ims_file):
    """
    Extract basic metadata from IMS file.

    Args:
        ims_file: Path to IMS file

    Returns:
        tuple: (metadata_dict, voxel_sizes)
            voxel_sizes: [x, y, z] in nanometers, or None
    """
    import h5py

    try:
        with h5py.File(ims_file, 'r') as h5_file:
            metadata = {}
            voxel_sizes = None

            if 'DataSetInfo' in h5_file and 'Image' in h5_file['DataSetInfo']:
                info_group = h5_file['DataSetInfo']['Image']
                for attr_name in info_group.attrs:
                    attr_value = info_group.attrs[attr_name]
                    if isinstance(attr_value, np.ndarray) and attr_value.dtype.char == 'S':
                        metadata[attr_name] = b''.join(attr_value).decode('utf-8', errors='ignore')
                    else:
                        metadata[attr_name] = attr_value

            # Get unit from metadata if available (IMS typically uses micrometers)
            unit = metadata.get('Unit', 'micrometer')
            if isinstance(unit, bytes):
                unit = unit.decode('utf-8', errors='ignore')

            if all(key in metadata for key in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2', 'X', 'Y', 'Z']):
                try:
                    x_size = (float(metadata['ExtMax0']) - float(metadata['ExtMin0'])) / float(metadata['X'])
                    y_size = (float(metadata['ExtMax1']) - float(metadata['ExtMin1'])) / float(metadata['Y'])
                    z_size = (float(metadata['ExtMax2']) - float(metadata['ExtMin2'])) / float(metadata['Z'])
                    # Convert to nanometers
                    voxel_sizes = [
                        convert_to_nanometers(x_size, unit),
                        convert_to_nanometers(y_size, unit),
                        convert_to_nanometers(z_size, unit)
                    ]
                    print(f"Calculated voxel sizes (XYZ) in nm: {voxel_sizes}")
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Could not calculate voxel sizes: {e}")
                    voxel_sizes = None

            return metadata, voxel_sizes
    except Exception as e:
        print(f"Warning: Could not extract metadata from {ims_file}: {e}")
        return {}, None


def _read_czi_plane(czi_file, plane_dict, y_size, x_size):
    """
    Read a single plane from CZI file.

    This function opens and closes the file for each read, which is necessary
    for dask delayed execution where each task runs independently.
    """
    from pylibCZIrw import czi

    with czi.open_czi(czi_file) as czidoc:
        result = czidoc.read(plane=plane_dict)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        return result


def load_czi_stack(czi_file, view_index=None):
    """
    Load CZI data as a dask array using pylibCZIrw.

    Args:
        czi_file: Path to CZI file
        view_index: Optional specific view index to load. If None and multiple views exist,
                   loads all views as a 5D array (VCZYX).

    Returns:
        tuple: (dask_array, None, axes_order)
    """
    from pylibCZIrw import czi
    import dask

    if not os.path.isfile(czi_file):
        raise ValueError(f"CZI file does not exist: {czi_file}")

    with czi.open_czi(czi_file) as czidoc:
        bbox = czidoc.total_bounding_box
        pixel_types = czidoc.pixel_types
        pixel_type = pixel_types.get(0, 'Gray16')

    dtype_map = {
        'Gray8': np.uint8,
        'Gray16': np.uint16,
        'Gray32Float': np.float32,
        'Bgr24': np.uint8,
        'Bgr48': np.uint16,
    }
    dtype = dtype_map.get(pixel_type, np.uint16)

    def get_dim_size(dim_name):
        if dim_name in bbox:
            return bbox[dim_name][1] - bbox[dim_name][0]
        return 1

    v_size = get_dim_size('V')
    c_size = get_dim_size('C')
    z_size = get_dim_size('Z')
    y_size = get_dim_size('Y')
    x_size = get_dim_size('X')

    print(f"CZI dimensions: V={v_size}, C={c_size}, Z={z_size}, Y={y_size}, X={x_size}")
    print(f"CZI dtype: {dtype}")

    if view_index is not None:
        if view_index < 0 or view_index >= v_size:
            raise ValueError(f"View index {view_index} out of range (0-{v_size-1})")
        v_range = [view_index]
        include_v_dim = False
        print(f"Loading single view V={view_index}")
    elif v_size > 1:
        v_range = range(v_size)
        include_v_dim = True
        print(f"Loading all {v_size} views")
    else:
        v_range = [0]
        include_v_dim = False
        print("Single view file")

    if include_v_dim:
        axes_order = ['t', 'c', 'z', 'y', 'x']
        v_chunks = []
        for v in v_range:
            c_chunks = []
            for c in range(c_size):
                z_chunks = []
                for z in range(z_size):
                    plane_dict = {'V': v, 'C': c, 'Z': z}
                    delayed_read = dask.delayed(_read_czi_plane)(czi_file, plane_dict, y_size, x_size)
                    arr = da.from_delayed(delayed_read, shape=(y_size, x_size), dtype=dtype)
                    z_chunks.append(arr)
                c_chunks.append(da.stack(z_chunks, axis=0))
            v_chunks.append(da.stack(c_chunks, axis=0))
        dask_array = da.stack(v_chunks, axis=0)
        print(f"Created 5D dask array (VCZYX): {dask_array.shape}")

    elif c_size > 1:
        axes_order = ['c', 'z', 'y', 'x']
        c_chunks = []
        for c in range(c_size):
            z_chunks = []
            for z in range(z_size):
                v = v_range[0]
                plane_dict = {'V': v, 'C': c, 'Z': z} if v_size > 1 else {'C': c, 'Z': z}
                delayed_read = dask.delayed(_read_czi_plane)(czi_file, plane_dict, y_size, x_size)
                arr = da.from_delayed(delayed_read, shape=(y_size, x_size), dtype=dtype)
                z_chunks.append(arr)
            c_chunks.append(da.stack(z_chunks, axis=0))
        dask_array = da.stack(c_chunks, axis=0)
        print(f"Created 4D dask array (CZYX): {dask_array.shape}")

    else:
        axes_order = ['z', 'y', 'x']
        z_chunks = []
        for z in range(z_size):
            v = v_range[0]
            plane_dict = {'V': v, 'Z': z} if v_size > 1 else {'Z': z}
            delayed_read = dask.delayed(_read_czi_plane)(czi_file, plane_dict, y_size, x_size)
            arr = da.from_delayed(delayed_read, shape=(y_size, x_size), dtype=dtype)
            z_chunks.append(arr)
        dask_array = da.stack(z_chunks, axis=0)
        print(f"Created 3D dask array (ZYX): {dask_array.shape}")

    print(f"Dask array shape: {dask_array.shape}, chunks: {dask_array.chunksize}")
    print(f"CZI axes order: {axes_order}")

    return dask_array, None, axes_order


def extract_czi_metadata(czi_file):
    """
    Extract metadata and voxel sizes from CZI file.

    Args:
        czi_file: Path to CZI file

    Returns:
        tuple: (raw_xml_metadata, voxel_sizes_dict)
            voxel_sizes_dict: {'x': float, 'y': float, 'z': float} in nanometers
    """
    from pylibCZIrw import czi
    import xml.etree.ElementTree as ET

    if not os.path.isfile(czi_file):
        raise ValueError(f"CZI file does not exist: {czi_file}")

    with czi.open_czi(czi_file) as czidoc:
        raw_meta = czidoc.raw_metadata
        voxel_sizes = None

        if raw_meta:
            try:
                root = ET.fromstring(raw_meta)

                scaling = root.find('.//Scaling')
                if scaling is not None:
                    items = scaling.find('Items')
                    if items is not None:
                        voxel_sizes = {'x': 1.0, 'y': 1.0, 'z': 1.0}
                        for dist in items.findall('Distance'):
                            id_attr = dist.get('Id')
                            value = dist.find('Value')
                            if value is not None and id_attr:
                                # CZI stores values in meters, convert to nanometers
                                val_nm = convert_to_nanometers(float(value.text), 'meter')
                                if id_attr.upper() == 'X':
                                    voxel_sizes['x'] = val_nm
                                elif id_attr.upper() == 'Y':
                                    voxel_sizes['y'] = val_nm
                                elif id_attr.upper() == 'Z':
                                    voxel_sizes['z'] = val_nm

                        print(f"Extracted CZI voxel sizes (nm): x={voxel_sizes['x']:.2f}, y={voxel_sizes['y']:.2f}, z={voxel_sizes['z']:.2f}")

            except Exception as e:
                print(f"Warning: Could not parse CZI metadata: {e}")

        return raw_meta, voxel_sizes


def extract_precomputed_metadata(path, scale_index=0):
    """
    Extract metadata and voxel sizes from Neuroglancer Precomputed format.

    Reads the 'info' file from a precomputed dataset and extracts resolution
    information. Handles both local paths and remote URLs.

    Args:
        path: Path to precomputed dataset (local path or URL)
            - Local: /path/to/precomputed_data
            - Remote: precomputed://https://... or gs://... or s3://...
        scale_index: Which resolution level to extract (0 = highest resolution)

    Returns:
        tuple: (info_dict, voxel_sizes_dict)
            info_dict: Full info file content as dict
            voxel_sizes_dict: {'x': float, 'y': float, 'z': float} in nanometers

    Example:
        >>> info, voxel_sizes = extract_precomputed_metadata('/data/precomputed')
        >>> print(voxel_sizes)
        {'x': 9.0, 'y': 9.0, 'z': 12.0}  # nanometers

    Notes:
        - Precomputed stores resolution in nanometers (preserved as-is)
        - Resolution is in XYZ order in the info file
        - Returns (None, None) if metadata cannot be read
    """
    import json
    from urllib.parse import urlparse

    # Strip precomputed:// prefix if present
    clean_path = path
    if clean_path.startswith('precomputed://'):
        clean_path = clean_path[len('precomputed://'):]

    # Determine if local or remote
    parsed = urlparse(clean_path)
    is_remote = parsed.scheme.lower() in ('http', 'https', 's3', 'gs')

    info = None

    if is_remote:
        # Remote path - use HTTP request for http/https
        try:
            if clean_path.startswith(('http://', 'https://')):
                import urllib.request
                info_url = clean_path.rstrip('/') + '/info'
                with urllib.request.urlopen(info_url, timeout=30) as response:
                    info = json.load(response)
            else:
                # GCS or S3 - try using TensorStore's kvstore
                import tensorstore as ts
                from tensorswitch_v2.readers.base import build_kvstore
                kvstore = build_kvstore(clean_path)
                if kvstore['driver'] in ('gcs', 's3'):
                    kvstore['path'] = kvstore['path'].rstrip('/') + '/info'
                spec = {'driver': 'json', 'kvstore': kvstore}
                result = ts.open(spec).result()
                info = result.read().result()
        except Exception as e:
            print(f"Warning: Failed to read remote precomputed info: {e}")
            return None, None
    else:
        # Local path
        info_path = os.path.join(clean_path, 'info')
        if os.path.isfile(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to parse precomputed info file: {e}")
                return None, None
        else:
            print(f"Warning: Precomputed info file not found at {info_path}")
            return None, None

    if info is None:
        return None, None

    # Extract voxel sizes from the specified scale
    voxel_sizes = None
    scales = info.get('scales', [])

    if scales:
        # Validate scale_index
        if scale_index >= len(scales):
            print(f"Warning: scale_index {scale_index} >= number of scales {len(scales)}, using scale 0")
            scale_index = 0

        scale_info = scales[scale_index]
        resolution = scale_info.get('resolution')

        if resolution and len(resolution) >= 3:
            # Precomputed resolution is in nanometers, XYZ order
            # Keep as nanometers (standard unit)
            voxel_sizes = {
                'x': float(resolution[0]),
                'y': float(resolution[1]),
                'z': float(resolution[2])
            }

    return info, voxel_sizes
