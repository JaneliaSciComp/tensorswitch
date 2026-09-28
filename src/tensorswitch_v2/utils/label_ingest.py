"""
Sparse label ingestion for OME-NGFF zarr containers.

Provides ingest_label_at_offset() and _read_target_shape_from_container()
used by the --output-offset / --add-to-existing branch in __main__.py.
"""

import os
import json


def _read_target_shape_from_container(container_path: str) -> list:
    """
    Infer target shape from the first non-.tmp sibling label's s0/zarr.json.

    Args:
        container_path: Root path of the target zarr container.

    Returns:
        Shape as a list of ints.

    Raises:
        ValueError: If no usable sibling label is found.
    """
    labels_dir = os.path.join(container_path, 'labels')
    if not os.path.isdir(labels_dir):
        raise ValueError(
            f"No labels/ directory in {container_path}. "
            "Supply --target-shape to specify the full output shape."
        )
    for entry in sorted(os.listdir(labels_dir)):
        if entry.endswith('.tmp') or entry == 'zarr.json':
            continue
        s0_meta = os.path.join(labels_dir, entry, 's0', 'zarr.json')
        if not os.path.exists(s0_meta):
            continue
        with open(s0_meta) as f:
            meta = json.load(f)
        shape = meta.get('shape')
        if shape:
            return list(shape)
    raise ValueError(
        f"No readable sibling label found in {labels_dir}/. "
        "Supply --target-shape to specify the full output shape."
    )


def ingest_label_at_offset(
    source_ts,
    target_container: str,
    label_tmp_key: str,
    offset: list,
    target_shape: list,
    chunk_shape: tuple,
    dtype: str,
    compression: str,
    compression_level: int = 5,
) -> None:
    """
    Write source_ts into a full-size label array at a TCZYX offset.

    Creates labels/<label_tmp_key>/s0 (zarr3, fill_value=0) in target_container,
    writes source data at [offset[0]:offset[0]+src[0], ...], and writes
    OME-NGFF label image group metadata.

    _finalize_add_to_existing() must be called afterwards to rename
    labels/<label_tmp_key>/ to labels/<label_name>/ and update container lists.

    Args:
        source_ts:         TensorStore of the source label array.
        target_container:  Root path of the target OME-NGFF zarr container.
        label_tmp_key:     In-flight label name (e.g. 'proofread_t10.tmp').
        offset:            TCZYX offset list (same ndim as target_shape).
        target_shape:      Full output array shape.
        chunk_shape:       Inner chunk shape (padded to ndim automatically).
        dtype:             Output dtype string (e.g. 'uint32').
        compression:       Codec name: 'zstd', 'gzip', or 'blosc'.
        compression_level: Compression level (default 5).
    """
    import numpy as np
    import tensorstore as ts

    ndim = len(target_shape)
    if len(offset) != ndim:
        raise ValueError(
            f"offset length {len(offset)} must match target ndim {ndim}"
        )

    # Pad chunk_shape to ndim (add leading 1s for T, C, etc.)
    chunk_list = list(chunk_shape)
    if len(chunk_list) < ndim:
        chunk_list = [1] * (ndim - len(chunk_list)) + chunk_list
    else:
        chunk_list = chunk_list[-ndim:]

    # Codec stack — no sharding for s0 ingest; pyramid writer handles s1+
    codec_name = (compression or 'zstd').lower()
    if codec_name in ('gzip', 'gz'):
        compress_codec = {'name': 'gzip', 'configuration': {'level': compression_level}}
    elif codec_name == 'blosc':
        compress_codec = {
            'name': 'blosc',
            'configuration': {
                'cname': 'zstd',
                'clevel': compression_level,
                'shuffle': 'bitshuffle',
            },
        }
    else:
        compress_codec = {'name': 'zstd', 'configuration': {'level': compression_level}}

    codecs = [
        {'name': 'bytes', 'configuration': {'endian': 'little'}},
        compress_codec,
    ]

    # Create target array at labels/<label_tmp_key>/s0
    s0_path = os.path.join(target_container, 'labels', label_tmp_key, 's0')
    os.makedirs(os.path.dirname(s0_path), exist_ok=True)

    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': s0_path},
        'create': True,
        'delete_existing': True,
        'metadata': {
            'shape': list(target_shape),
            'chunk_grid': {
                'name': 'regular',
                'configuration': {'chunk_shape': chunk_list},
            },
            'chunk_key_encoding': {'name': 'default'},
            'fill_value': 0,
            'data_type': dtype,
            'codecs': codecs,
        },
    }
    target_store = ts.open(spec).result()

    # Read source into memory, add leading singleton dims if fewer than ndim
    data = source_ts.read().result()
    while data.ndim < ndim:
        data = data[np.newaxis]

    # Cast to target dtype if needed
    np_dtype = np.dtype(dtype)
    if data.dtype != np_dtype:
        data = data.astype(np_dtype)

    # Write source at offset
    src_shape = data.shape
    slices = tuple(slice(offset[i], offset[i] + src_shape[i]) for i in range(ndim))
    target_store[slices].write(data).result()
    print(f"Ingested {list(src_shape)} → {list(target_shape)} at offset {offset}")

    # Write OME-NGFF label image group metadata
    _write_label_group_metadata(
        target_container=target_container,
        label_tmp_key=label_tmp_key,
        target_shape=target_shape,
    )


def _write_label_group_metadata(
    target_container: str,
    label_tmp_key: str,
    target_shape: list,
) -> None:
    """Write zarr.json for labels/<label_tmp_key>/ (OME-NGFF label image group)."""
    from .ome_structure import OMEStructure, OMEStructureConfig

    ndim = len(target_shape)
    axes = _infer_axes(target_container, ndim)
    scale = _infer_base_scale(target_container, ndim)
    datasets = [{
        'path': 's0',
        'coordinateTransformations': [{'type': 'scale', 'scale': scale}],
    }]

    config = OMEStructureConfig(label_name=label_tmp_key)
    ome = OMEStructure(target_container, config)
    ome.write_label_image_metadata(
        multiscales={'axes': axes, 'datasets': datasets},
        name=label_tmp_key.replace('.tmp', ''),
        source_image_path='../../raw',
        label_name=label_tmp_key,
    )


def _infer_base_scale(container_path: str, ndim: int) -> list:
    """Read the base-level (s0) voxel scale from root zarr.json's own
    multiscales (which mirrors raw's), falling back to 1.0/axis if
    unavailable. Without this, offset-ingested labels get a bogus identity
    scale while sibling raw/other labels carry the real voxel size,
    misaligning the label layer in any viewer that respects declared
    physical scale."""
    root_meta = os.path.join(container_path, 'zarr.json')
    if os.path.exists(root_meta):
        try:
            with open(root_meta) as f:
                meta = json.load(f)
            multiscales = meta.get('attributes', {}).get('ome', {}).get('multiscales', [])
            if multiscales:
                datasets = multiscales[0].get('datasets', [])
                if datasets:
                    for ct in datasets[0].get('coordinateTransformations', []):
                        if ct.get('type') == 'scale' and len(ct.get('scale', [])) == ndim:
                            return list(ct['scale'])
        except Exception:
            pass
    return [1.0] * ndim


def _infer_axes(container_path: str, ndim: int) -> list:
    """Read axes from root zarr.json, sibling label zarr.json, or fall back to defaults."""
    root_meta = os.path.join(container_path, 'zarr.json')
    if os.path.exists(root_meta):
        try:
            with open(root_meta) as f:
                meta = json.load(f)
            multiscales = meta.get('attributes', {}).get('ome', {}).get('multiscales', [])
            if multiscales:
                axes = multiscales[0].get('axes', [])
                if len(axes) == ndim:
                    return axes
        except Exception:
            pass

    labels_dir = os.path.join(container_path, 'labels')
    if os.path.isdir(labels_dir):
        for entry in sorted(os.listdir(labels_dir)):
            if entry.endswith('.tmp') or entry == 'zarr.json':
                continue
            label_meta = os.path.join(labels_dir, entry, 'zarr.json')
            if not os.path.exists(label_meta):
                continue
            try:
                with open(label_meta) as f:
                    meta = json.load(f)
                multiscales = (
                    meta.get('attributes', {}).get('ome', {}).get('multiscales', [{}])
                )
                axes = multiscales[0].get('axes', []) if multiscales else []
                if len(axes) == ndim:
                    return axes
            except Exception:
                pass

    _AXIS_DEFAULTS = {
        5: [
            {'name': 't', 'type': 'time', 'unit': 'millisecond'},
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
        ],
        4: [
            {'name': 'c', 'type': 'channel'},
            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
        ],
        3: [
            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
        ],
    }
    return _AXIS_DEFAULTS.get(
        ndim, [{'name': f'dim{i}', 'type': 'space'} for i in range(ndim)]
    )
