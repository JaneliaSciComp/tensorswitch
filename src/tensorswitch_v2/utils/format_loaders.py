# Copied from tensorswitch/utils.py for v2 independence
"""
File format loaders for tensorswitch_v2.

This module contains functions to load various microscopy file formats:
- TIFF (including OME-TIFF and ImageJ TIFF)
- ND2 (Nikon)
- IMS (Imaris)
- CZI (Zeiss)
"""

import os
import numpy as np
import dask.array as da


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
            voxel_sizes: {'x': float, 'y': float, 'z': float} in micrometers, or None
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
                        voxel_sizes = {
                            'x': float(pixels.physical_size_x) if pixels.physical_size_x else 1.0,
                            'y': float(pixels.physical_size_y) if pixels.physical_size_y else 1.0,
                            'z': float(pixels.physical_size_z) if pixels.physical_size_z else 1.0
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
                    unit_lower = unit.lower()
                    if unit_lower in ['micron', 'um', 'µm', 'micrometer']:
                        z_spacing_um = float(z_spacing)
                    elif unit_lower in ['nm', 'nanometer']:
                        z_spacing_um = float(z_spacing) / 1000.0
                    elif unit_lower in ['mm', 'millimeter']:
                        z_spacing_um = float(z_spacing) * 1000.0
                    else:
                        print(f"Warning: Unknown unit '{unit}', assuming micrometers")
                        z_spacing_um = float(z_spacing)

                    # Default XY resolution if not in tags
                    voxel_sizes = {
                        'x': 1.0,
                        'y': 1.0,
                        'z': z_spacing_um
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
            voxel_sizes: {'x': float, 'y': float, 'z': float} in micrometers, or None
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
                    voxel_sizes = {
                        'x': float(pixels.physical_size_x) if pixels.physical_size_x else 1.0,
                        'y': float(pixels.physical_size_y) if pixels.physical_size_y else 1.0,
                        'z': float(pixels.physical_size_z) if pixels.physical_size_z else 1.0
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
            voxel_sizes: [x, y, z] in micrometers, or None
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

            if all(key in metadata for key in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2', 'X', 'Y', 'Z']):
                try:
                    x_size = (float(metadata['ExtMax0']) - float(metadata['ExtMin0'])) / float(metadata['X'])
                    y_size = (float(metadata['ExtMax1']) - float(metadata['ExtMin1'])) / float(metadata['Y'])
                    z_size = (float(metadata['ExtMax2']) - float(metadata['ExtMin2'])) / float(metadata['Z'])
                    voxel_sizes = [x_size, y_size, z_size]
                    print(f"Calculated voxel sizes (XYZ): {voxel_sizes}")
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
            voxel_sizes_dict: {'x': float, 'y': float, 'z': float} in micrometers
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
                                val_um = float(value.text) * 1e6
                                if id_attr.upper() == 'X':
                                    voxel_sizes['x'] = val_um
                                elif id_attr.upper() == 'Y':
                                    voxel_sizes['y'] = val_um
                                elif id_attr.upper() == 'Z':
                                    voxel_sizes['z'] = val_um

                        print(f"Extracted CZI voxel sizes: x={voxel_sizes['x']:.4f}, y={voxel_sizes['y']:.4f}, z={voxel_sizes['z']:.4f} um")

            except Exception as e:
                print(f"Warning: Could not parse CZI metadata: {e}")

        return raw_meta, voxel_sizes
