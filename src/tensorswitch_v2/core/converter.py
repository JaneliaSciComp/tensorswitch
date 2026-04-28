"""
DistributedConverter for TensorSwitch Phase 5 architecture.

Provides format-agnostic conversion with LSF multi-job and Dask single-job support.
"""

import json
import os
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import tensorstore as ts

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

from ..readers.base import BaseReader
from ..writers.base import BaseWriter

# Import utility functions from v2 utils (independent from v1)
from ..utils import (
    get_chunk_domains,
    get_total_chunks_from_store,
    get_tensorstore_context,
    get_dtype_name,
    detect_source_order,
    update_ome_metadata_if_needed,
)
from ..utils.metadata_utils import NON_SPATIAL_AXES


class DistributedConverter:
    """
    Format-agnostic converter with LSF/Dask distributed processing support.

    All readers now return a ts.TensorStore from get_tensorstore(),
    so the converter uses a single uniform code path for all formats.

    Example (Basic conversion):
        >>> from tensorswitch_v2.api import Readers, Writers
        >>> from tensorswitch_v2.core import DistributedConverter
        >>>
        >>> reader = Readers.tiff("/input.tif")
        >>> writer = Writers.zarr3("/output.zarr")
        >>> converter = DistributedConverter(reader, writer)
        >>> converter.convert()
    """

    def __init__(self, reader: BaseReader, writer: BaseWriter):
        self.reader = reader
        self.writer = writer
        self._input_store = None
        self._output_store = None
        self._total_chunks = None

    def convert(
        self,
        start_idx: int = 0,
        stop_idx: Optional[int] = None,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        write_metadata: bool = True,
        preserve_order: bool = True,
        force_order: Optional[str] = None,
        progress_interval: int = 100,
        verbose: bool = True,
        delete_existing: Optional[bool] = None,
        voxel_size_override: Optional[Dict[str, float]] = None,
        voxel_unit: Optional[str] = None,
        is_label: bool = False,
        expand_to_5d: bool = False,
        bbox: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        axes_order_override: Optional[List[str]] = None,
        no_ome_meta_export: bool = False,
        no_ome_xml_attr: bool = False,
        output_dtype: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert data from reader to writer.

        Args:
            bbox: Optional (origin, size) tuple for subvolume extraction.
                  origin and size are 3-tuples in source dimension order.
                  Spatial dimensions are auto-detected from domain labels.
            no_ome_meta_export: If True, skip writing OME/METADATA.ome.xml file.
            no_ome_xml_attr: If True, skip embedding OME/CZI XML in zarr.json/.zattrs.
        """
        start_time = time.time()

        # 1. Open input store from reader (uniform API for all tiers)
        if verbose:
            print(f"Opening input: {self.reader}")
        self._input_store = self.reader.get_tensorstore()

        # 1b. Apply bounding box if specified (subvolume extraction)
        if bbox is not None:
            origin, size = bbox
            try:
                labels = list(self._input_store.domain.labels)
            except Exception:
                labels = []

            # Identify spatial dimensions (skip channel, time, etc.)
            if labels:
                spatial_dims = [i for i, l in enumerate(labels)
                                if l.lower() not in NON_SPATIAL_AXES]
            else:
                # No labels — assume first min(3, ndim) dims are spatial
                spatial_dims = list(range(min(3, self._input_store.ndim)))

            # Validate bbox against source domain
            domain = self._input_store.domain
            for j, dim_i in enumerate(spatial_dims):
                if j >= len(origin):
                    break
                dim_origin = domain.origin[dim_i]
                dim_end = dim_origin + self._input_store.shape[dim_i]
                bbox_start = origin[j]
                bbox_end = origin[j] + size[j]
                dim_label = labels[dim_i] if labels else f"dim{dim_i}"
                if bbox_start < dim_origin or bbox_end > dim_end:
                    raise ValueError(
                        f"Bbox for {dim_label} [{bbox_start}:{bbox_end}) is outside "
                        f"source domain [{dim_origin}:{dim_end})"
                    )

            # Build indexing tuple
            idx = [slice(None)] * self._input_store.ndim
            for j, dim_i in enumerate(spatial_dims):
                if j < len(origin):
                    idx[dim_i] = slice(origin[j], origin[j] + size[j])

            self._input_store = self._input_store[tuple(idx)]

            # Rebase domain to zero-origin so chunk indexing works correctly.
            # TensorStore slicing preserves original coordinates (e.g., x=[116316,116444)),
            # but the converter and writer expect zero-based chunk indices.
            self._input_store = self._input_store[ts.d[:].translate_to[0]]

            if verbose:
                print(f"  Bbox applied: origin={origin}, size={size}")
                print(f"  Cropped shape: {tuple(self._input_store.shape)}")

        input_shape = tuple(self._input_store.shape)
        input_dtype = get_dtype_name(self._input_store.dtype)
        effective_dtype = output_dtype if output_dtype else input_dtype

        if verbose:
            print(f"  Shape: {input_shape}, dtype: {input_dtype}")
            if output_dtype and output_dtype != input_dtype:
                print(f"  Output dtype: {output_dtype} (casting from {input_dtype})")

        # Validate dtype cast safety: check if target range can hold source values
        if output_dtype and output_dtype != input_dtype:
            out_np = np.dtype(output_dtype)
            in_np = np.dtype(input_dtype)
            if np.can_cast(in_np, out_np, casting='safe'):
                if verbose:
                    print(f"  Dtype cast {input_dtype} -> {output_dtype}: safe (no data loss)")
            elif np.issubdtype(out_np, np.integer):
                # Narrowing cast — sample data to check range
                info = np.iinfo(out_np)
                if verbose:
                    print(f"  Dtype cast {input_dtype} -> {output_dtype}: narrowing "
                          f"(target range [{info.min}, {info.max}]), sampling data to verify...")
                # Sample up to 8 evenly-spaced chunks from the input
                total_voxels = int(np.prod(input_shape))
                n_sample = min(8, max(1, total_voxels // (256 * 256 * 256)))
                sample_indices = np.linspace(0, max(0, input_shape[0] - 1), n_sample, dtype=int)
                src_min, src_max = np.inf, -np.inf
                for si in sample_indices:
                    # Read a thin slab along first axis
                    slab = self._input_store[int(si)].read().result()
                    if slab.size > 0:
                        src_min = min(src_min, float(slab.min()))
                        src_max = max(src_max, float(slab.max()))
                if src_min < info.min or src_max > info.max:
                    raise ValueError(
                        f"Dtype cast {input_dtype} -> {output_dtype} would clip data: "
                        f"source range [{src_min:.2f}, {src_max:.2f}] exceeds target "
                        f"range [{info.min}, {info.max}]. Use a wider dtype or pre-normalize "
                        f"the data. (Sampled {n_sample} slabs along axis 0.)"
                    )
                if verbose:
                    print(f"  Data range [{src_min:.2f}, {src_max:.2f}] fits in "
                          f"{output_dtype} [{info.min}, {info.max}] — safe to cast")

        # 2. Detect source order and axes
        use_fortran_order = False
        axes_order = None

        # Get domain labels from TensorStore (works for all tiers now)
        try:
            domain_labels = list(self._input_store.domain.labels)
            if domain_labels and all(isinstance(l, str) and l for l in domain_labels):
                axes_order = domain_labels
                if verbose:
                    print(f"  Axes from TensorStore domain: {axes_order}")
        except Exception:
            pass

        # Fall back to OME-NGFF axes from reader metadata (zarr2 has no domain labels)
        if not axes_order:
            try:
                metadata = self.reader.get_metadata()
                multiscales = metadata.get('multiscales', [])
                # If no multiscales at array level, try parent dir (OME-NGFF
                # stores multiscales one level above the array, e.g. img/.zattrs
                # for img/s0)
                if not multiscales:
                    parent_zattrs = os.path.join(
                        os.path.dirname(self.reader.path), '.zattrs')
                    if os.path.isfile(parent_zattrs):
                        with open(parent_zattrs, 'r') as f:
                            multiscales = json.load(f).get('multiscales', [])
                if multiscales:
                    ome_axes = multiscales[0].get('axes', [])
                    if ome_axes and len(ome_axes) == len(input_shape):
                        axes_order = [
                            a['name'].lower() if isinstance(a, dict) else a.lower()
                            for a in ome_axes
                        ]
                        if verbose:
                            print(f"  Axes from OME-NGFF metadata: {axes_order}")
            except Exception:
                pass

        # Handle order: force_order overrides auto-detection
        if force_order is not None:
            use_fortran_order = (force_order.lower() == 'f')
            if verbose:
                order_name = "F-order" if use_fortran_order else "C-order"
                print(f"  Forcing {order_name} (--force_{force_order.lower()}_order)")
        elif preserve_order:
            try:
                order_info = detect_source_order(self._input_store)
                use_fortran_order = order_info.get('is_fortran_order', False)
                if axes_order is None:
                    axes_order = order_info.get('suggested_axes', None)
                if verbose:
                    order_name = "F-order" if use_fortran_order else "C-order"
                    print(f"  Auto-detected {order_name} from source")
            except Exception:
                pass  # Default to C-order

        self._use_fortran_order = use_fortran_order

        # 2b. Detect and squeeze singleton channel dimension
        # TensorStore's neuroglancer_precomputed driver adds a 4th channel dimension
        # even when num_channels=1. For truly 3D data, we should squeeze this out.
        self._squeeze_channel = False
        self._squeeze_axis = None

        if axes_order and len(axes_order) == len(input_shape):
            axes_lower = [a.lower() for a in axes_order]
            if 'channel' in axes_lower:
                channel_idx = axes_lower.index('channel')
                if input_shape[channel_idx] == 1:
                    self._squeeze_channel = True
                    self._squeeze_axis = channel_idx

                    squeezed_shape = tuple(s for i, s in enumerate(input_shape) if i != channel_idx)
                    squeezed_axes = [a for i, a in enumerate(axes_order) if i != channel_idx]

                    if verbose:
                        print(f"  Squeezing singleton channel: {input_shape} -> {squeezed_shape}")
                        print(f"  Axes: {axes_order} -> {squeezed_axes}")

                    input_shape = squeezed_shape
                    axes_order = squeezed_axes

        # 2c. Compute spatial axis transpose for --axes_order override
        spatial_transpose = None
        _target_axes_order = None
        if axes_order_override and axes_order:
            source_spatial = [a for a in axes_order if a.lower() in {'x', 'y', 'z'}]
            if sorted(axes_order_override) != sorted(source_spatial):
                raise ValueError(
                    f"--axes_order {axes_order_override} doesn't match "
                    f"source spatial axes {source_spatial}")
            if axes_order_override != source_spatial:
                # Build full target axes (non-spatial stay in place, spatial reordered)
                target_axes = []
                spatial_iter = iter(axes_order_override)
                for a in axes_order:
                    if a.lower() in {'x', 'y', 'z'}:
                        target_axes.append(next(spatial_iter))
                    else:
                        target_axes.append(a)
                perm = [axes_order.index(a) for a in target_axes]
                spatial_transpose = tuple(perm)
                _target_axes_order = target_axes
                if verbose:
                    print(f"  Axes reorder: {axes_order} -> {target_axes} (transpose: {perm})")

        # 3. Get metadata from reader
        # Readers return voxel sizes in nanometers when real metadata exists.
        # When no metadata, _default_voxel_sizes() returns {x:1.0, y:1.0, z:1.0} placeholder.
        source_unit = 'nanometer'
        target_unit = voxel_unit or 'nanometer'

        if voxel_size_override:
            # --voxel_size values are in the target unit (--voxel_unit, or nm if not set)
            voxel_sizes = voxel_size_override
            source_unit = target_unit  # override already in target unit
            if verbose:
                print(f"  Using voxel size override ({source_unit}): {voxel_sizes}")
        else:
            try:
                voxel_sizes = self.reader.get_voxel_sizes()
            except Exception:
                voxel_sizes = None
            if voxel_sizes and all(v == 1.0 for v in voxel_sizes.values()):
                # Default placeholder [1,1,1] — no real voxel metadata in source.
                # Refuse to guess; require user to specify explicitly.
                raise ValueError(
                    "No voxel size metadata found in source file. "
                    "Please provide --voxel_size X,Y,Z (and optionally --voxel_unit) "
                    "so the output metadata is correct. "
                    "Example: --voxel_size 0.108,0.108,0.268 --voxel_unit micrometer"
                )

        # Convert voxel sizes from source unit to target unit if they differ
        if voxel_sizes and source_unit != target_unit:
            to_nm = {'nanometer': 1.0, 'micrometer': 1e3, 'millimeter': 1e6}
            factor = to_nm[source_unit] / to_nm[target_unit]
            voxel_sizes = {k: v * factor for k, v in voxel_sizes.items()}
            if verbose:
                print(f"  Converted voxel sizes: {source_unit} -> {target_unit} (x{factor}): {voxel_sizes}")

        try:
            if voxel_size_override:
                # Suppress voxel-size warnings when user provided explicit override
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ome_metadata = self.reader.get_ome_metadata()
            else:
                ome_metadata = self.reader.get_ome_metadata()
        except Exception:
            ome_metadata = None

        ome_xml = None
        source_format = None
        raw_metadata = {}
        try:
            raw_metadata = self.reader.get_metadata()
            ome_xml = raw_metadata.get('ome_xml') or raw_metadata.get('raw_xml')
            source_format = raw_metadata.get('source_format',
                                             self.reader.__class__.__name__.replace('Reader', '').lower())
        except Exception:
            pass

        # Extract channel metadata for OMERO rendering hints
        channel_info = None
        try:
            from ..utils.metadata_utils import extract_channel_info
            reader_path = getattr(self.reader, 'path', None)
            channel_info = extract_channel_info(raw_metadata, source_format, reader_path)
        except Exception:
            pass

        # 3b. Auto-cap chunk/shard shape for frame-based DaskReader sources.
        # When native chunks have small dims (e.g., Z=1 for ND2 full-frame),
        # cap output chunk/shard to native to prevent cache-thrashing reads.
        from ..readers.base import DaskReader
        if isinstance(self.reader, DaskReader):
            native = getattr(self.reader, '_native_chunk_shape', None)
            if native is not None:
                # If channel was squeezed, remove that dim from native too
                if self._squeeze_channel and self._squeeze_axis is not None:
                    native = [n for i, n in enumerate(native)
                              if i != self._squeeze_axis]

                # Detect frame-based dims: native much smaller than array
                frame_dims = [i for i, (nd, ad)
                              in enumerate(zip(native, input_shape))
                              if nd < ad]

                if frame_dims:
                    from ..utils.tensorstore_utils import (
                        adaptive_spatial_chunk, build_default_shape)
                    if chunk_shape is None:
                        sc = adaptive_spatial_chunk(input_shape, input_dtype)
                        default_chunk = build_default_shape(
                            list(input_shape), axes_order, sc)
                        chunk_shape = tuple(
                            min(d, n) if n < a else d
                            for d, n, a in zip(
                                default_chunk, native, input_shape))
                        if verbose:
                            print(f"  Auto-capped chunk shape for frame-based "
                                  f"source: {chunk_shape}")

                    if shard_shape is None:
                        default_shard = build_default_shape(
                            list(input_shape), axes_order, 1024)
                        shard_shape = tuple(
                            min(d, n) if n < a else d
                            for d, n, a in zip(
                                default_shard, native, input_shape))
                        if verbose:
                            print(f"  Auto-capped shard shape for frame-based "
                                  f"source: {shard_shape}")

        # 4. Apply spatial axis reorder (--axes_order)
        if spatial_transpose:
            input_shape = tuple(input_shape[p] for p in spatial_transpose)
            # chunk/shard shapes may be spatial-only (e.g. 3 values for x,y,z)
            # while spatial_transpose covers all dims (including c, t, etc).
            # Pad them to full dimensionality (1 for non-spatial) before permuting.
            for attr_name in ('chunk_shape', 'shard_shape'):
                shape_val = chunk_shape if attr_name == 'chunk_shape' else shard_shape
                if shape_val and len(shape_val) < len(spatial_transpose):
                    padded = []
                    spatial_iter = iter(shape_val)
                    for a in axes_order:
                        if a.lower() in {'x', 'y', 'z'}:
                            padded.append(next(spatial_iter))
                        else:
                            padded.append(1)
                    if attr_name == 'chunk_shape':
                        chunk_shape = tuple(padded[p] for p in spatial_transpose)
                    else:
                        shard_shape = tuple(padded[p] for p in spatial_transpose)
                elif shape_val:
                    if attr_name == 'chunk_shape':
                        chunk_shape = tuple(shape_val[p] for p in spatial_transpose)
                    else:
                        shard_shape = tuple(shape_val[p] for p in spatial_transpose)
            axes_order = _target_axes_order
            if verbose:
                print(f"  Reordered shape: {input_shape}")

        # 5. Create output spec and open store
        if verbose:
            print(f"Creating output: {self.writer}")

        output_spec = self.writer.create_output_spec(
            shape=input_shape,
            dtype=effective_dtype,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            use_fortran_order=use_fortran_order,
            axes_order=axes_order,
            expand_to_5d=expand_to_5d,
            spatial_transpose=spatial_transpose,
        )
        _delete = delete_existing if delete_existing is not None else (start_idx == 0)
        self._output_store = self.writer.open_store(
            output_spec,
            create=True,
            delete_existing=_delete
        )

        # 5. Get chunk information
        write_chunk_shape = tuple(self._output_store.chunk_layout.write_chunk.shape)
        self._total_chunks = get_total_chunks_from_store(
            self._output_store,
            chunk_shape=write_chunk_shape
        )

        if stop_idx is None:
            stop_idx = self._total_chunks

        if verbose:
            print(f"  Chunk shape: {write_chunk_shape}")
            print(f"  Total chunks: {self._total_chunks}")
            print(f"  Processing: {start_idx} to {stop_idx} ({stop_idx - start_idx} chunks)")

        # 6. Generate chunk domains for assigned range
        linear_indices = range(start_idx, stop_idx)
        chunk_domains = get_chunk_domains(
            write_chunk_shape,
            self._output_store,
            linear_indices_to_process=linear_indices
        )

        # 7. Process chunks — uniform path for all reader tiers
        chunks_processed = 0
        last_report_time = start_time

        # Per-channel min/max tracking for OMERO windows
        axes_lower = [a.lower() for a in axes_order]
        c_axis = axes_lower.index('c') if 'c' in axes_lower else None
        num_channels = input_shape[c_axis] if c_axis is not None else 1
        channel_mins = [np.inf] * num_channels
        channel_maxs = [-np.inf] * num_channels

        for idx, chunk_domain in enumerate(chunk_domains, start=start_idx):
            try:
                read_domain = chunk_domain
                write_domain = chunk_domain
                if hasattr(self.writer, 'get_input_domain_from_output'):
                    read_domain = self.writer.get_input_domain_from_output(chunk_domain)

                # If we squeezed a singleton channel, expand domain back for reading
                if self._squeeze_channel and self._squeeze_axis is not None:
                    if hasattr(read_domain, 'origin'):
                        slices = []
                        for i in range(read_domain.ndim):
                            start_val = int(read_domain.origin[i])
                            stop_val = start_val + int(read_domain.shape[i])
                            slices.append(slice(start_val, stop_val))
                        read_domain = tuple(slices)

                    read_domain = list(read_domain)
                    read_domain.insert(self._squeeze_axis, slice(0, 1))
                    read_domain = tuple(read_domain)

                # Read from input — uniform for all tiers
                read_order = 'F' if self._use_fortran_order else 'C'
                data = self._input_store[read_domain].read(order=read_order).result()

                # Squeeze singleton channel if we detected one
                if self._squeeze_channel and self._squeeze_axis is not None:
                    data = np.squeeze(data, axis=self._squeeze_axis)

                # Cast dtype if requested
                if output_dtype and output_dtype != input_dtype:
                    out_np = np.dtype(output_dtype)
                    if np.issubdtype(data.dtype, np.floating) and np.issubdtype(out_np, np.integer):
                        info = np.iinfo(out_np)
                        data = np.clip(data, info.min, info.max)
                    data = data.astype(out_np)

                # Track per-channel min/max for OMERO display windows
                if c_axis is not None:
                    # Determine which channels this chunk covers
                    if hasattr(write_domain, 'origin'):
                        c_start = int(write_domain.origin[c_axis])
                        c_size = int(write_domain.shape[c_axis])
                    else:
                        c_slice = write_domain[c_axis]
                        c_start = c_slice.start or 0
                        c_size = (c_slice.stop or input_shape[c_axis]) - c_start
                    for ci in range(c_size):
                        gc = c_start + ci
                        if gc < num_channels:
                            ch_data = np.take(data, ci, axis=c_axis)
                            channel_mins[gc] = min(channel_mins[gc], float(ch_data.min()))
                            channel_maxs[gc] = max(channel_maxs[gc], float(ch_data.max()))
                else:
                    channel_mins[0] = min(channel_mins[0], float(data.min()))
                    channel_maxs[0] = max(channel_maxs[0], float(data.max()))

                # Write
                self.writer.write_chunk(write_domain, data, self._output_store)
                chunks_processed += 1

                # Progress reporting
                if verbose and (chunks_processed % progress_interval == 0 or
                               time.time() - last_report_time > 30):
                    elapsed = time.time() - start_time
                    rate = chunks_processed / elapsed if elapsed > 0 else 0
                    remaining = stop_idx - start_idx - chunks_processed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  Progress: {chunks_processed}/{stop_idx - start_idx} "
                          f"({rate:.1f} chunks/s, ETA: {eta/60:.1f}m)")
                    last_report_time = time.time()

            except KeyboardInterrupt:
                print(f"\nInterrupted at chunk {idx}. Exiting...")
                raise
            except Exception as e:
                if verbose:
                    print(f"  Warning: Skipping chunk {idx}: {e}")
                continue

        # Build channel_minmax from accumulated stats (only if we saw real data)
        channel_minmax = None
        if any(m != np.inf for m in channel_mins):
            channel_minmax = [
                (float(channel_mins[i]), float(channel_maxs[i]))
                for i in range(num_channels)
            ]

        # 9. Write metadata
        if write_metadata and (stop_idx >= self._total_chunks or start_idx > 0):
            if verbose:
                print("Writing metadata...")
            try:
                source_path = getattr(self.reader, 'path', None)
                if source_path:
                    image_name = os.path.splitext(os.path.basename(source_path))[0]
                else:
                    image_name = "image"

                self.writer.write_metadata(
                    ome_metadata=ome_metadata,
                    voxel_sizes=voxel_sizes,
                    voxel_unit=target_unit,
                    array_shape=input_shape,
                    axes_order=axes_order,
                    ome_xml=ome_xml,
                    image_name=image_name,
                    is_label=is_label,
                    source_format=source_format,
                    no_ome_meta_export=no_ome_meta_export,
                    no_ome_xml_attr=no_ome_xml_attr,
                    channel_info=channel_info,
                    channel_minmax=channel_minmax,
                    dtype=input_dtype,
                )
                root_path = os.path.dirname(self.writer.output_path)
                if root_path and os.path.basename(self.writer.output_path) == 's0':
                    if verbose:
                        print("Writing root OME-NGFF metadata...")
                    update_ome_metadata_if_needed(root_path, use_ome_structure=True)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Metadata write failed: {e}")

        # 10. Return statistics
        elapsed = time.time() - start_time
        stats = {
            'chunks_processed': chunks_processed,
            'total_chunks': self._total_chunks,
            'start_idx': start_idx,
            'stop_idx': stop_idx,
            'elapsed_seconds': elapsed,
            'chunks_per_second': chunks_processed / elapsed if elapsed > 0 else 0,
            'input_shape': input_shape,
            'output_chunk_shape': write_chunk_shape,
        }

        if verbose:
            print(f"Conversion complete: {chunks_processed} chunks in {elapsed:.1f}s "
                  f"({stats['chunks_per_second']:.1f} chunks/s)")

        return stats

    def get_total_chunks(
        self,
        chunk_shape: Optional[Tuple[int, ...]] = None
    ) -> int:
        """Get total number of chunks for the conversion."""
        if self._total_chunks is not None:
            return self._total_chunks

        # Uniform: get shape from reader's TensorStore
        input_store = self.reader.get_tensorstore()
        input_shape = tuple(input_store.shape)
        input_dtype = get_dtype_name(input_store.dtype)

        output_spec = self.writer.create_output_spec(
            shape=input_shape,
            dtype=input_dtype,
            chunk_shape=chunk_shape
        )

        if chunk_shape is None:
            chunk_shape = self.writer.get_default_chunk_shape(input_shape)

        chunk_counts = np.ceil(np.array(input_shape) / np.array(chunk_shape)).astype(int)
        self._total_chunks = int(np.prod(chunk_counts))

        return self._total_chunks

    def get_chunk_ranges(
        self,
        num_jobs: int,
        chunk_shape: Optional[Tuple[int, ...]] = None
    ) -> List[Tuple[int, int]]:
        """Get chunk ranges for distributing work across LSF jobs."""
        total = self.get_total_chunks(chunk_shape)
        chunks_per_job = total // num_jobs
        remainder = total % num_jobs

        ranges = []
        start = 0
        for i in range(num_jobs):
            extra = 1 if i < remainder else 0
            stop = start + chunks_per_job + extra
            ranges.append((start, stop))
            start = stop

        return ranges

    def __repr__(self) -> str:
        return f"DistributedConverter(reader={self.reader}, writer={self.writer})"
