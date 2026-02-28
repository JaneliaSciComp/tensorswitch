"""
Resource calculation utilities for LSF job submission.

Provides functions to estimate memory, wall time, and cores based on
dataset size and output format.

Default chunk/shard sizes must match the actual writers in tensorstore_utils.py:
  - Zarr3 sharded:     shard=1024, inner chunk=256 (spatial)
  - Zarr3 non-sharded: chunk=64 (spatial)
  - Zarr2:             chunk=64 (spatial, DEFAULT_SPATIAL_CHUNK)
Non-spatial axes (c, t, v, channel) always get size 1.
"""

import math
import numpy as np
from typing import Tuple, List, Optional

from .tensorstore_utils import adaptive_spatial_chunk, build_default_shape

_ZARR3_SHARD_SPATIAL = 1024    # tensorstore_utils.py zarr3_store_spec default


def calculate_memory(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=False, output_format='zarr3'):
    """Calculate memory in GB for LSF job.

    For Zarr3 (with sharding): based on shard buffer size (few large shards in memory).
    For Zarr2 (no sharding): based on dataset size (source reader overhead dominates).

    Args:
        volume_shape: Shape of the volume (tuple/list of ints)
        dtype_str: Data type string (e.g., 'uint8', 'uint64')
        shard_shape: Shape of each shard/chunk (tuple/list of ints)
        total_shards: Total number of shards/chunks
        use_bioio: If True, applies 3x multiplier for BioIO's higher memory usage
        output_format: 'zarr3' or 'zarr2'

    Returns:
        int: Recommended memory in GB
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)

    if output_format == 'zarr2':
        # Zarr2: no sharding, many small chunks. Memory driven by source reader
        # overhead (dask chunk caching) rather than output chunk buffers.
        if dataset_size_gb < 10:
            base_mem = 4
        elif dataset_size_gb < 100:
            base_mem = min(dataset_size_gb * 0.1, 15)
        else:
            base_mem = min(dataset_size_gb * 0.1, 30)

        # Source reader buffer (dask caches source chunks which may be large)
        reader_buffer = 2

        # Task overhead
        task_overhead = 4

        total_mem = base_mem + reader_buffer + task_overhead
        recommended = int(math.ceil(total_mem * 1.5 / 5) * 5)
    else:
        # Zarr3: shard-buffer-based formula
        shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

        # Base memory for file loading
        if dataset_size_gb < 10:
            base_mem = 2
        elif dataset_size_gb < 100:
            base_mem = min(dataset_size_gb * 0.02, 10)
        else:
            base_mem = min(dataset_size_gb * 0.005, 20)

        # Shard processing buffers (up to 3 concurrent shards, 2x for read+write+compress)
        concurrent_shards = min(total_shards, 3)
        shard_buffer_mem = shard_size_gb * 2.0 * concurrent_shards

        # Task overhead (conversion)
        task_overhead = 4  # GB for Python, TensorStore, etc.

        # Total with 1.5x safety margin, rounded to nearest 5 GB
        total_mem = base_mem + shard_buffer_mem + task_overhead
        recommended = int(math.ceil(total_mem * 1.5 / 5) * 5)

    # BioIO uses ~3x more memory due to Dask caching and intermediate structures
    if use_bioio:
        recommended = int(recommended * 3)

    return max(5, min(recommended, 500))


def calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=False,
                        output_format='zarr3', no_sharding=False, cores=1):
    """Calculate wall time string (H:MM) for LSF job.

    Three modes:
      - Zarr3 sharded: per-shard time estimate x total shards (few large shards)
      - Zarr2 / Zarr3 non-sharded: max of throughput-based and per-chunk estimates
        (many small chunk files — same model for both since the I/O pattern is identical)
        Wall time scales down with more cores (parallel chunk processing).

    Args:
        volume_shape: Shape of the volume (tuple/list of ints)
        dtype_str: Data type string (e.g., 'uint8', 'uint64')
        shard_shape: Shape of each shard/chunk (tuple/list of ints)
        total_shards: Total number of shards/chunks
        use_bioio: If True, applies 10x multiplier for BioIO's slower processing
        output_format: 'zarr3' or 'zarr2'
        no_sharding: If True, zarr3 uses the per-chunk model instead of per-shard
        cores: Number of cores allocated (used to scale down non-sharded wall time)

    Returns:
        str: Wall time string in H:MM format
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)

    # Non-sharded formats (Zarr2 or Zarr3 --no_sharding) use per-chunk model
    use_chunk_model = (output_format == 'zarr2') or no_sharding

    if use_chunk_model:
        # Estimate 1: Data throughput (dominates for large custom chunks)
        # Empirical: ~10 MB/s for ND2/dask source over network filesystem
        throughput_mb_s = 10
        throughput_minutes = (dataset_size_gb * 1024) / throughput_mb_s / 60

        # Estimate 2: Per-chunk overhead (dominates for many small chunks)
        # Empirical: ~0.7 sec/chunk for 64^3 (512 KB) on network filesystem
        # (9 GB TIFF -> Zarr2: 19,440 chunks in ~216 min = 0.67 sec/chunk)
        chunk_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)
        if chunk_size_gb < 0.001:       # < 1 MB (e.g., 64^3 uint16 = 512 KB)
            sec_per_chunk = 0.5
        elif chunk_size_gb < 0.01:      # < 10 MB
            sec_per_chunk = 1.0
        else:                           # >= 10 MB
            sec_per_chunk = 2.0
        chunk_minutes = total_shards * sec_per_chunk / 60

        base_minutes = max(throughput_minutes, chunk_minutes)

        # Scale down by cores — parallel chunk processing
        # Use 0.7 efficiency factor (I/O contention on network FS limits perfect scaling)
        if cores > 1:
            parallel_factor = cores * 0.7
            base_minutes = base_minutes / parallel_factor

        # Overhead for file loading (ND2/TIFF opening)
        if dataset_size_gb > 1000:
            overhead = 10
        elif dataset_size_gb > 100:
            overhead = 5
        else:
            overhead = 2

        # 3x safety, round to nearest 30 min
        safe_minutes = int(math.ceil((base_minutes + overhead) * 3 / 30) * 30)
    else:
        # Zarr3: per-shard time estimate
        shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

        # Per-shard time estimate (empirical)
        if shard_size_gb < 0.1:
            minutes_per_shard = 0.5
        elif shard_size_gb < 1.0:
            minutes_per_shard = 2
        else:
            minutes_per_shard = 3

        base_minutes = minutes_per_shard * total_shards

        # Scale down by cores — TensorStore processes shards in parallel
        # Higher efficiency (0.85) than non-sharded because shards are large sequential I/O
        if cores > 1:
            parallel_factor = cores * 0.85
            base_minutes = base_minutes / parallel_factor

        # Overhead for file loading
        if dataset_size_gb > 1000:
            overhead = 10
        elif dataset_size_gb > 100:
            overhead = 5
        else:
            overhead = 2

        # 2x safety, round to nearest 30 min
        safe_minutes = int(math.ceil((base_minutes + overhead) * 2 / 30) * 30)

    # BioIO is ~10-50x slower due to Dask overhead, apply 10x multiplier
    if use_bioio:
        safe_minutes = int(safe_minutes * 10)

    # Cap at 96 hours (4 days)
    max_minutes = 96 * 60
    if safe_minutes > max_minutes:
        print(f"  WARNING: Estimated wall time ({safe_minutes // 60}h {safe_minutes % 60}m) exceeds 96h cap.")
        print(f"  Consider using larger chunks (--chunk_shape 128,128,128 or 256,256,256)")
        print(f"  or Zarr3 with sharding (--output_format zarr3) for faster conversion.")
        safe_minutes = max_minutes
    safe_minutes = max(30, safe_minutes)
    hours = safe_minutes // 60
    minutes = safe_minutes % 60
    return f"{hours}:{minutes:02d}"


def estimate_shard_info(
    volume_shape: Tuple[int, ...],
    dtype_str: str,
    output_format: str = "zarr3",
    chunk_shape: Optional[Tuple[int, ...]] = None,
    shard_shape: Optional[Tuple[int, ...]] = None,
    axes_order: Optional[List[str]] = None,
    no_sharding: bool = False,
) -> Tuple[Tuple[int, ...], int]:
    """Estimate shard/chunk shape and count for resource calculation.

    Uses the same defaults as the actual writers in tensorstore_utils.py:
      - Zarr3 sharded:     shard=1024 spatial
      - Zarr3 non-sharded: chunk=64 spatial
      - Zarr2:             chunk=64 spatial
    Non-spatial axes (c, t, v) always get size 1.

    Args:
        volume_shape: Input volume shape
        dtype_str: Data type string
        output_format: 'zarr3' or 'zarr2'
        chunk_shape: Custom chunk shape or None for default
        shard_shape: Custom shard shape or None for default
        axes_order: Axis names (e.g., ['z','y','x']) or None
        no_sharding: If True, zarr3 uses non-sharded chunk defaults

    Returns:
        (shape, total_count) tuple.
        For Zarr3 sharded: shard shape and total shards.
        For Zarr2 / Zarr3 non-sharded: chunk shape and total chunks.
    """
    use_sharding = output_format == "zarr3" and not no_sharding

    if use_sharding:
        # Zarr3 sharded: resource needs driven by shard size (1024 spatial)
        if shard_shape is not None:
            shape = shard_shape
        else:
            shape = build_default_shape(volume_shape, axes_order, _ZARR3_SHARD_SPATIAL)
    else:
        # Zarr2 or Zarr3 non-sharded: adaptive chunk based on dataset size
        if chunk_shape is not None:
            shape = chunk_shape
        else:
            spatial = adaptive_spatial_chunk(volume_shape, dtype_str)
            shape = build_default_shape(volume_shape, axes_order, spatial)

    # Calculate total shards/chunks
    total = 1
    for i, dim in enumerate(volume_shape):
        s = shape[i] if i < len(shape) else shape[-1]
        total *= math.ceil(dim / s)

    return shape, total


def calculate_job_resources(
    shape: List[int],
    dtype: str,
    output_format: str = "zarr3",
    chunk_shape_str: Optional[str] = None,
    shard_shape_str: Optional[str] = None,
    axes_order: Optional[List[str]] = None,
    no_sharding: bool = False,
) -> Tuple[int, str, int]:
    """
    Calculate memory, wall time, and cores based on dataset size.

    Convenience function that combines estimate_shard_info, calculate_memory,
    and calculate_wall_time.

    Args:
        shape: Dataset shape (list of ints, e.g., [7650, 9740, 1590])
        dtype: Data type string (e.g., 'uint8', 'uint64')
        output_format: 'zarr3' or 'zarr2'
        chunk_shape_str: Chunk shape string (e.g., '64,64,64') or None
        shard_shape_str: Shard shape string (e.g., '1024,1024,1024') or None
        axes_order: Axis names (e.g., ['z','y','x']) or None
        no_sharding: If True, zarr3 uses non-sharded defaults

    Returns:
        tuple: (memory_gb, wall_time_str, cores)
    """
    # Parse chunk shape
    chunk_shape = None
    if chunk_shape_str:
        chunk_shape = tuple(int(x) for x in chunk_shape_str.split(','))

    # Parse shard shape
    shard_shape = None
    if shard_shape_str:
        shard_shape = tuple(int(x) for x in shard_shape_str.split(','))

    # Estimate shard/chunk info
    shard_shape, total_shards = estimate_shard_info(
        volume_shape=tuple(shape),
        dtype_str=dtype,
        output_format=output_format,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        axes_order=axes_order,
        no_sharding=no_sharding,
    )

    # Calculate memory first (needed for Zarr3 sharded core formula)
    memory_gb = calculate_memory(shape, dtype, shard_shape, total_shards, output_format=output_format)

    # Cores: capped at 8 (I/O-bound)
    # Non-sharded (Zarr2 / Zarr3 --no_sharding): min 4 cores — many small chunk files
    #   benefit from I/O parallelism + compression threads
    # Zarr3 sharded: based on memory (large shard buffers need more cores)
    dtype_bytes = np.dtype(dtype).itemsize
    dataset_size_gb = (np.prod(shape) * dtype_bytes) / (1024 ** 3)
    use_sharding = output_format == "zarr3" and not no_sharding
    if not use_sharding:
        cores = min(8, max(4, int(math.ceil(dataset_size_gb / 25)) * 2))
    else:
        cores = min(8, max(1, int(math.ceil(memory_gb / 15)) * 2))

    # Wall time: pass cores so non-sharded estimates scale with parallelism
    wall_time = calculate_wall_time(shape, dtype, shard_shape, total_shards, output_format=output_format,
                                    no_sharding=no_sharding, cores=cores)

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    return memory_gb, wall_time, cores
