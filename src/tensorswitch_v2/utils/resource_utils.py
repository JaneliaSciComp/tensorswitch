"""
Resource calculation utilities for LSF job submission.

Provides functions to estimate memory, wall time, and cores based on
dataset size and output format.
"""

import math
import numpy as np
from typing import Tuple, List, Optional


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


def calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards, use_bioio=False, output_format='zarr3'):
    """Calculate wall time string (H:MM) for LSF job.

    For Zarr3 (with sharding): per-shard time estimate x total shards.
    For Zarr2 (no sharding): throughput-based estimate from dataset size.

    Args:
        volume_shape: Shape of the volume (tuple/list of ints)
        dtype_str: Data type string (e.g., 'uint8', 'uint64')
        shard_shape: Shape of each shard/chunk (tuple/list of ints)
        total_shards: Total number of shards/chunks
        use_bioio: If True, applies 10x multiplier for BioIO's slower processing
        output_format: 'zarr3' or 'zarr2'

    Returns:
        str: Wall time string in H:MM format
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)

    if output_format == 'zarr2':
        # Zarr2: many small chunks, estimate from dataset throughput
        # Empirical: ~50-200 MB/s for compressed Zarr2 writes on cluster storage
        throughput_mb_s = 100  # Conservative estimate
        base_minutes = (dataset_size_gb * 1024) / throughput_mb_s / 60

        # Overhead for file loading (ND2/TIFF opening)
        if dataset_size_gb > 1000:
            overhead = 10
        elif dataset_size_gb > 100:
            overhead = 5
        else:
            overhead = 2

        # 3x safety (small chunks have more filesystem overhead than throughput suggests)
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
    safe_minutes = max(30, min(safe_minutes, 96 * 60))
    hours = safe_minutes // 60
    minutes = safe_minutes % 60
    return f"{hours}:{minutes:02d}"


def estimate_shard_info(
    volume_shape: Tuple[int, ...],
    dtype_str: str,
    output_format: str = "zarr3",
    chunk_shape: Optional[Tuple[int, ...]] = None,
    shard_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, ...], int]:
    """Estimate shard shape and total shards for resource calculation.

    Simplified version that doesn't require args object.

    Args:
        volume_shape: Input volume shape
        dtype_str: Data type string
        output_format: 'zarr3' or 'zarr2'
        chunk_shape: Chunk shape tuple or None for default
        shard_shape: Shard shape tuple or None for default

    Returns:
        (shard_shape, total_shards) tuple. If no sharding (zarr2), shard_shape
        equals chunk_shape and total_shards is the total number of chunks.
    """
    # Default chunk shape
    if chunk_shape is None:
        chunk_shape = (32, 32, 32)

    # Determine shard shape based on format
    use_sharding = output_format == "zarr3"
    if use_sharding:
        if shard_shape is None:
            shard_shape = (1024, 1024, 1024)
    else:
        # Zarr2: no sharding, each chunk is a "shard"
        shard_shape = chunk_shape

    # Calculate total shards
    total_shards = 1
    for i, dim in enumerate(volume_shape):
        shard_dim = shard_shape[i] if i < len(shard_shape) else shard_shape[-1]
        total_shards *= math.ceil(dim / shard_dim)

    return shard_shape, total_shards


def calculate_job_resources(
    shape: List[int],
    dtype: str,
    output_format: str = "zarr3",
    chunk_shape_str: Optional[str] = None,
    shard_shape_str: Optional[str] = None,
) -> Tuple[int, str, int]:
    """
    Calculate memory, wall time, and cores based on dataset size.

    Convenience function that combines estimate_shard_info, calculate_memory,
    and calculate_wall_time.

    Args:
        shape: Dataset shape (list of ints, e.g., [7650, 9740, 1590])
        dtype: Data type string (e.g., 'uint8', 'uint64')
        output_format: 'zarr3' or 'zarr2'
        chunk_shape_str: Chunk shape string (e.g., '32,32,32') or None
        shard_shape_str: Shard shape string (e.g., '1024,1024,1024') or None

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

    # Estimate shard info
    shard_shape, total_shards = estimate_shard_info(
        volume_shape=tuple(shape),
        dtype_str=dtype,
        output_format=output_format,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )

    # Calculate memory and wall time
    memory_gb = calculate_memory(shape, dtype, shard_shape, total_shards, output_format=output_format)
    wall_time = calculate_wall_time(shape, dtype, shard_shape, total_shards, output_format=output_format)

    # Cores: I/O bound, more cores don't help much
    cores = min(8, max(2, int(math.ceil(memory_gb / 15)) * 2))

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    return memory_gb, wall_time, cores
