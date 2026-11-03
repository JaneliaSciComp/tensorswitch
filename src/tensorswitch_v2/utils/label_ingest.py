"""
Sparse label ingestion for OME-NGFF zarr containers.

Provides ingest_label_at_offset(), _read_target_shape_from_container(), and the
source-orientation helpers (orient_label_source, check_offset_flags_supported) used by the --output-offset / --add-to-existing branch in __main__.py.
"""

import os
import json


def container_format(container_path: str) -> str:
    """'zarr3' when the container has a root zarr.json, 'zarr2' when it has a .zgroup (and no zarr.json), else 'zarr3'.
    The one place that decides the format of a label added to an existing container: the label follows its container."""
    if os.path.exists(os.path.join(container_path, 'zarr.json')):
        return 'zarr3'
    if os.path.exists(os.path.join(container_path, '.zgroup')):
        return 'zarr2'
    return 'zarr3'


def _read_json(path: str):
    """The parsed JSON file at `path`, or None when it does not exist or cannot be read."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _zarr2_compressor(codec_name: str, level: int) -> dict:
    """The zarr 2 compressor for the codec the zarr 3 path uses, by TensorSwitch's one rule for zarr 2 (zstd is written wrapped in blosc: bare
    zstd frames from TensorStore releases before 0.1.80 carry no content size and numcodecs cannot read them)."""
    from .metadata_utils import normalize_zarr2_compressor_for_tensorstore
    if codec_name == 'blosc':
        return {'id': 'blosc', 'cname': 'zstd', 'clevel': level, 'shuffle': 1, 'blocksize': 0}
    return normalize_zarr2_compressor_for_tensorstore({'id': 'gzip' if codec_name in ('gzip', 'gz') else 'zstd', 'level': level})


def _zarr2_multiscales(container_path: str) -> list:
    """The `multiscales` list of a zarr 2 container: its root .zattrs, else raw/.zattrs, else a sibling label's .zattrs (first one found)."""
    candidates = [os.path.join(container_path, '.zattrs'), os.path.join(container_path, 'raw', '.zattrs')]
    labels_dir = os.path.join(container_path, 'labels')
    if os.path.isdir(labels_dir):
        candidates += [os.path.join(labels_dir, e, '.zattrs') for e in sorted(os.listdir(labels_dir))
                       if not e.endswith('.tmp') and os.path.isdir(os.path.join(labels_dir, e))]
    for path in candidates:
        ms = (_read_json(path) or {}).get('multiscales')
        if ms:
            return ms
    return []


def _read_target_shape_from_container(container_path: str) -> list:
    """
    Infer target shape from the first non-.tmp sibling label's s0/zarr.json (zarr 3) or s0/.zarray (zarr 2).

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
    s0_meta_name = '.zarray' if container_format(container_path) == 'zarr2' else 'zarr.json'
    for entry in sorted(os.listdir(labels_dir)):
        if entry.endswith('.tmp') or entry in ('zarr.json', '.zgroup', '.zattrs'):
            continue
        s0_meta = os.path.join(labels_dir, entry, 's0', s0_meta_name)
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


# Source-side flags the standard converter honors but this path does not (yet). Refused loudly
# instead of being ignored, so a label is never written with a layout the caller did not ask for.
_UNSUPPORTED_WITH_OFFSET = (
    ("bbox", "--bbox"),
    ("bbox_axes", "--bbox_axes"),
    ("squeeze_singleton_axes", "--squeeze_singleton_axes"),
)


def check_offset_flags_supported(args) -> None:
    """Raise if a flag this sparse-ingest path cannot apply was given."""
    used = [flag for attr, flag in _UNSUPPORTED_WITH_OFFSET if getattr(args, attr, None)]
    if used:
        raise ValueError(
            f"{', '.join(used)} cannot be combined with --output-offset: the sparse label path "
            "writes the whole source at the offset and does not crop or squeeze it."
        )


def orient_label_source(source_ts, axes_order=None, input_axes=None, relabel_axis=None):
    """Return source_ts with its spatial axes permuted into `axes_order` (e.g. ['z','y','x']).

    Mirrors what the standard converter does for --axes_order: the source axes are first named
    (from the reader's labels, then --input_axes / --relabel_axis), only spatial axes may move,
    and a non-matching request is an error rather than a silent no-op. Returns source_ts
    unchanged when axes_order is not given.
    """
    if not axes_order:
        return source_ts
    labels = [str(l) for l in source_ts.domain.labels]
    explicit = {}
    for spec in (relabel_axis or []):
        old, new = spec.lower().split('=', 1)
        explicit[old.strip()] = new.strip()
    if all(labels):
        names = ['c' if l.lower() == 'channel' else l.lower() for l in labels]
        if input_axes:
            from ..core.converter import DistributedConverter
            explicit = DistributedConverter._relabel_from_input_axes(input_axes, names, explicit)
        names = [explicit.get(a, a) for a in names]
    elif input_axes and len(input_axes.strip()) == source_ts.rank:
        names = list(input_axes.strip().lower())
    else:
        raise ValueError(
            "--axes_order needs named source axes; this source has none. "
            "Name them with --input_axes (one letter per axis, slowest to fastest)."
        )
    if len(names) != len(set(names)):
        raise ValueError(f"source axis names are not unique: {names}")
    source_spatial = [a for a in names if a in ('x', 'y', 'z')]
    if sorted(axes_order) != sorted(source_spatial):
        raise ValueError(
            f"--axes_order {list(axes_order)} does not match the source spatial axes "
            f"{source_spatial}; fix the axis names with --input_axes or --relabel_axis first."
        )
    spatial_iter = iter(axes_order)
    target = [next(spatial_iter) if a in ('x', 'y', 'z') else a for a in names]
    perm = [names.index(a) for a in target]
    if perm == list(range(len(names))):
        return source_ts
    return source_ts.transpose(perm)


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

    Creates labels/<label_tmp_key>/s0 (fill_value=0; zarr 3 or zarr 2, whichever the container is: see container_format()) in target_container,
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

    # Name the array's axes after the container's own, so the pyramid step and any reader know the layout
    # instead of guessing z,y,x from the shape (wrong for a container stored x,y,z).
    axis_names = [a.get('name') for a in _infer_axes(target_container, ndim)]

    # Create target array at labels/<label_tmp_key>/s0
    s0_path = os.path.join(target_container, 'labels', label_tmp_key, 's0')
    os.makedirs(os.path.dirname(s0_path), exist_ok=True)

    fmt = container_format(target_container)
    if fmt == 'zarr2':
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': s0_path},
            'create': True,
            'delete_existing': True,
            'metadata': {
                'shape': list(target_shape),
                'chunks': chunk_list,
                'dtype': np.dtype(dtype).str,
                'compressor': _zarr2_compressor(codec_name, compression_level),
                'order': 'C',
                'dimension_separator': '/',
                'fill_value': 0,
            },
        }
    else:
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
        if all(isinstance(n, str) and n for n in axis_names):
            spec['metadata']['dimension_names'] = axis_names
    target_store = ts.open(spec).result()
    if fmt == 'zarr2' and all(isinstance(n, str) and n for n in axis_names):
        with open(os.path.join(s0_path, '.zattrs'), 'w') as f:                  # a zarr 2 array names its axes here
            json.dump({'_ARRAY_DIMENSIONS': axis_names}, f, indent=2)

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
    from .ome_structure import OMEStructure, OMEStructureConfig, OMEStructureZarr2, OMEStructureZarr2Config

    ndim = len(target_shape)
    axes = _infer_axes(target_container, ndim)
    scale = _infer_base_scale(target_container, ndim)
    datasets = [{
        'path': 's0',
        'coordinateTransformations': [{'type': 'scale', 'scale': scale}],
    }]

    if container_format(target_container) == 'zarr2':
        labels_dir = os.path.join(target_container, 'labels')
        if not os.path.exists(os.path.join(labels_dir, '.zgroup')):         # the first label of a crop: the labels/ group itself
            with open(os.path.join(labels_dir, '.zgroup'), 'w') as f:
                json.dump({'zarr_format': 2}, f, indent=2)
        OMEStructureZarr2(target_container, OMEStructureZarr2Config(label_name=label_tmp_key)).write_label_image_metadata(
            multiscales={'axes': axes, 'datasets': datasets},
            name=label_tmp_key.replace('.tmp', ''),
            source_image_path='../../raw',
            label_name=label_tmp_key,
        )
        return
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
    if container_format(container_path) == 'zarr2':
        for ms in _zarr2_multiscales(container_path):
            datasets = ms.get('datasets', [])
            if datasets:
                for ct in datasets[0].get('coordinateTransformations', []):
                    if ct.get('type') == 'scale' and len(ct.get('scale', [])) == ndim:
                        return list(ct['scale'])
        return [1.0] * ndim
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
    """Read axes from root zarr.json, sibling label zarr.json (zarr 2: the container's .zattrs, raw's, a sibling label's), or fall back to defaults."""
    if container_format(container_path) == 'zarr2':
        for ms in _zarr2_multiscales(container_path):
            if len(ms.get('axes', [])) == ndim:
                return ms['axes']
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
