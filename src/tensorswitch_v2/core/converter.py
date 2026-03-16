"""
DistributedConverter for TensorSwitch Phase 5 architecture.

Provides format-agnostic conversion with LSF multi-job and Dask single-job support.
"""

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
        is_label: bool = False,
        expand_to_5d: bool = False,
    ) -> Dict[str, Any]:
        """Convert data from reader to writer."""
        start_time = time.time()

        # 1. Open input store from reader (uniform API for all tiers)
        if verbose:
            print(f"Opening input: {self.reader}")
        self._input_store = self.reader.get_tensorstore()
        input_shape = tuple(self._input_store.shape)
        input_dtype = get_dtype_name(self._input_store.dtype)

        if verbose:
            print(f"  Shape: {input_shape}, dtype: {input_dtype}")

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

        # 3. Get metadata from reader
        try:
            voxel_sizes = self.reader.get_voxel_sizes()
        except Exception:
            voxel_sizes = None

        if voxel_size_override:
            voxel_sizes = voxel_size_override
            if verbose:
                print(f"  Using voxel size override: {voxel_sizes}")

        try:
            ome_metadata = self.reader.get_ome_metadata()
        except Exception:
            ome_metadata = None

        ome_xml = None
        try:
            raw_metadata = self.reader.get_metadata()
            ome_xml = raw_metadata.get('ome_xml') or raw_metadata.get('raw_xml')
        except Exception:
            pass

        # 4. Create output spec and open store
        if verbose:
            print(f"Creating output: {self.writer}")

        output_spec = self.writer.create_output_spec(
            shape=input_shape,
            dtype=input_dtype,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            use_fortran_order=use_fortran_order,
            axes_order=axes_order,
            expand_to_5d=expand_to_5d
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

            except Exception as e:
                if verbose:
                    print(f"  Warning: Skipping chunk {idx}: {e}")
                continue

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
                    array_shape=input_shape,
                    axes_order=axes_order,
                    ome_xml=ome_xml,
                    image_name=image_name,
                    is_label=is_label
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
