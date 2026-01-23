"""
DistributedConverter for TensorSwitch Phase 5 architecture.

Provides format-agnostic conversion with LSF multi-job and Dask single-job support.
"""

import os
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import tensorstore as ts

from ..readers.base import BaseReader
from ..writers.base import BaseWriter

# Import utility functions from existing tensorswitch
from tensorswitch.utils import (
    get_chunk_domains,
    get_total_chunks_from_store,
    get_tensorstore_context,
    detect_source_order,
)


class DistributedConverter:
    """
    Format-agnostic converter with LSF/Dask distributed processing support.

    Connects readers and writers to perform data conversions. Supports:
    - LSF multi-job mode: Chunks are distributed across LSF jobs
    - Dask single-job mode: Local parallel processing
    - Sequential mode: Single-threaded processing

    Design Principles:
    - Readers provide input TensorStore arrays (format-agnostic)
    - Writers handle output format encoding
    - Converter manages chunk iteration and coordination
    - No knowledge of specific input/output formats

    Example (Basic conversion):
        >>> from tensorswitch_v2.api import Readers, Writers
        >>> from tensorswitch_v2.core import DistributedConverter
        >>>
        >>> reader = Readers.tiff("/input.tif")
        >>> writer = Writers.zarr3("/output.zarr")
        >>> converter = DistributedConverter(reader, writer)
        >>> converter.convert()

    Example (LSF multi-job mode):
        >>> # Job 1: Process chunks 0-99
        >>> converter.convert(start_idx=0, stop_idx=100)
        >>>
        >>> # Job 2: Process chunks 100-199
        >>> converter.convert(start_idx=100, stop_idx=200)

    Example (Custom chunk shape):
        >>> converter.convert(chunk_shape=(32, 256, 256))
    """

    def __init__(self, reader: BaseReader, writer: BaseWriter):
        """
        Initialize converter with reader and writer.

        Args:
            reader: BaseReader instance for input data
            writer: BaseWriter instance for output format
        """
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
        progress_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Convert data from reader to writer.

        Args:
            start_idx: Starting chunk index (for LSF multi-job mode)
            stop_idx: Ending chunk index (None = process all remaining)
            chunk_shape: Override chunk shape (uses writer default if None)
            shard_shape: Override shard shape (Zarr3 only, uses default if None)
            write_metadata: Write OME-NGFF metadata after conversion
            preserve_order: Preserve source data order (F-order vs C-order)
            progress_interval: Report progress every N chunks
            verbose: Print progress messages

        Returns:
            dict: Conversion statistics (chunks_processed, elapsed_time, etc.)

        Example:
            >>> # Convert all chunks
            >>> stats = converter.convert()
            >>> print(f"Processed {stats['chunks_processed']} chunks")

            >>> # LSF job processing subset
            >>> stats = converter.convert(start_idx=0, stop_idx=100)
        """
        start_time = time.time()

        # 1. Open input store from reader
        if verbose:
            print(f"Opening input: {self.reader}")
        input_spec = self.reader.get_tensorstore_spec()
        input_spec['context'] = get_tensorstore_context()
        self._input_store = ts.open(input_spec, read=True).result()

        input_shape = tuple(self._input_store.shape)
        input_dtype = str(self._input_store.dtype)

        if verbose:
            print(f"  Shape: {input_shape}, dtype: {input_dtype}")

        # 2. Detect source order if preserving
        use_fortran_order = False
        axes_order = None
        if preserve_order:
            try:
                order_info = detect_source_order(self._input_store)
                use_fortran_order = order_info.get('is_fortran_order', False)
                axes_order = order_info.get('suggested_axes', None)
                if verbose and use_fortran_order:
                    print(f"  Preserving F-order from source")
            except Exception:
                pass  # Default to C-order

        # 3. Get metadata from reader
        try:
            voxel_sizes = self.reader.get_voxel_sizes()
        except Exception:
            voxel_sizes = None

        try:
            ome_metadata = self.reader.get_ome_metadata()
        except Exception:
            ome_metadata = None

        # 4. Create output spec and open store
        if verbose:
            print(f"Creating output: {self.writer}")

        output_spec = self.writer.create_output_spec(
            shape=input_shape,
            dtype=input_dtype,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            use_fortran_order=use_fortran_order,
            axes_order=axes_order
        )
        self._output_store = self.writer.open_store(
            output_spec,
            create=True,
            delete_existing=(start_idx == 0)  # Only delete on first job
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

        # 7. Process chunks with per-chunk transactions
        tasks = []
        chunks_processed = 0
        last_report_time = start_time

        for idx, chunk_domain in enumerate(chunk_domains, start=start_idx):
            try:
                # Read from input
                data = self._input_store[chunk_domain].read().result()

                # Write to output using per-chunk transaction (Mark's fix)
                with ts.Transaction() as txn:
                    task = self._output_store[chunk_domain].with_transaction(txn).write(data)
                tasks.append(task)
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

        # 8. Wait for all tasks to complete
        for task in tasks:
            task.result()

        # 9. Write metadata (only if this job processes through the end)
        if write_metadata and stop_idx >= self._total_chunks:
            if verbose:
                print("Writing metadata...")
            try:
                self.writer.write_metadata(
                    ome_metadata=ome_metadata,
                    voxel_sizes=voxel_sizes,
                    array_shape=input_shape,
                    axes_order=axes_order
                )
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
        """
        Get total number of chunks for the conversion.

        Useful for LSF job distribution - divide total_chunks by num_jobs
        to get chunk ranges for each job.

        Args:
            chunk_shape: Override chunk shape (uses default if None)

        Returns:
            int: Total number of chunks

        Example:
            >>> total = converter.get_total_chunks()
            >>> chunks_per_job = total // num_jobs
            >>> # Job i processes: start=i*chunks_per_job, stop=(i+1)*chunks_per_job
        """
        if self._total_chunks is not None:
            return self._total_chunks

        # Need to open stores to calculate
        input_spec = self.reader.get_tensorstore_spec()
        input_spec['context'] = get_tensorstore_context()
        input_store = ts.open(input_spec, read=True).result()

        input_shape = tuple(input_store.shape)
        input_dtype = str(input_store.dtype)

        # Create temp output spec to get chunk layout
        output_spec = self.writer.create_output_spec(
            shape=input_shape,
            dtype=input_dtype,
            chunk_shape=chunk_shape
        )

        # Calculate total chunks from shape and chunk_shape
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
        """
        Get chunk ranges for distributing work across LSF jobs.

        Args:
            num_jobs: Number of parallel jobs
            chunk_shape: Override chunk shape (uses default if None)

        Returns:
            List of (start_idx, stop_idx) tuples for each job

        Example:
            >>> ranges = converter.get_chunk_ranges(num_jobs=10)
            >>> for i, (start, stop) in enumerate(ranges):
            ...     print(f"Job {i}: chunks {start} to {stop}")
        """
        total = self.get_total_chunks(chunk_shape)
        chunks_per_job = total // num_jobs
        remainder = total % num_jobs

        ranges = []
        start = 0
        for i in range(num_jobs):
            # Distribute remainder across first jobs
            extra = 1 if i < remainder else 0
            stop = start + chunks_per_job + extra
            ranges.append((start, stop))
            start = stop

        return ranges

    def __repr__(self) -> str:
        return f"DistributedConverter(reader={self.reader}, writer={self.writer})"
