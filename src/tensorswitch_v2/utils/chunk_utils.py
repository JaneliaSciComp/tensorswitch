# Copied from tensorswitch/utils.py for v2 independence
"""
Chunk utility functions for tensorswitch_v2.

This module contains chunk and shard operations including:
- Chunk domain calculation
- Total chunk counting
- Shard indexing
"""

import tensorstore as ts
import numpy as np
import itertools


def get_chunk_linear_indices_in_shard(shard_coord, shard_shape, chunk_shape, chunk_grid):
    """
    Generate linear indices for all chunks within a specific shard.
    Works for N-dimensional data (2D, 3D, 4D, 5D, etc.).

    Args:
        shard_coord: N-D shard coordinate (e.g., [z, y, x] for 3D or [c, z, y, x] for 4D)
        shard_shape: Shape of each shard (e.g., [1024, 1024, 1024])
        chunk_shape: Shape of each chunk (e.g., [32, 32, 32])
        chunk_grid: Total number of chunks in each dimension across entire array

    Returns:
        list: Linear indices of all chunks within this shard that are within data bounds
    """
    # Calculate how many chunks fit in each dimension of the shard
    chunks_per_shard_dim = [
        shard_shape[i] // chunk_shape[i]
        for i in range(len(shard_shape))
    ]

    # Base chunk coordinate for this shard (where this shard starts in chunk space)
    base_chunk_coord = [
        shard_coord[i] * chunks_per_shard_dim[i]
        for i in range(len(shard_coord))
    ]

    # Generate all chunk indices within this shard using N-D iteration
    chunk_indices = []
    for chunk_offset in itertools.product(*[range(dim) for dim in chunks_per_shard_dim]):
        # Calculate absolute chunk coordinate in the array
        chunk_coord = [
            base_chunk_coord[i] + chunk_offset[i]
            for i in range(len(base_chunk_coord))
        ]

        # Skip if chunk is outside data bounds
        if any(chunk_coord[i] >= chunk_grid[i] for i in range(len(chunk_coord))):
            continue

        # Convert N-D chunk coordinate to linear index (row-major order)
        linear_idx = 0
        stride = 1
        for i in range(len(chunk_coord) - 1, -1, -1):
            linear_idx += chunk_coord[i] * stride
            stride *= chunk_grid[i]

        chunk_indices.append(linear_idx)

    return chunk_indices


def get_chunk_domains(chunk_shape, array, linear_indices_to_process=None):
    """
    Generate chunk domains for processing.

    Args:
        chunk_shape: Shape of each chunk
        array: TensorStore array
        linear_indices_to_process: Optional list of specific linear indices to process

    Yields:
        ts.IndexDomain: Domain for each chunk
    """
    first_chunk_domain = ts.IndexDomain(inclusive_min=array.origin, shape=chunk_shape)
    chunk_number = -(np.array(array.shape) // -np.array(chunk_shape))

    if linear_indices_to_process is not None:
        linear_indices_iterator = linear_indices_to_process
    else:
        total_chunks = np.prod(chunk_number)
        linear_indices_iterator = range(total_chunks)

    for linear_idx in linear_indices_iterator:
        idx = np.unravel_index(linear_idx, chunk_number)

        yield first_chunk_domain.translate_by[
            tuple(map(lambda i, s: i * s, idx, chunk_shape))
        ].intersect(array.domain)


def get_total_chunks_from_store(store, chunk_shape=None):
    """
    Retrieve total number of chunks dynamically from a TensorStore store.
    Optionally, specify a chunk_shape; otherwise, it defaults to the store's read_chunk shape.

    Args:
        store: TensorStore store object
        chunk_shape: Optional chunk shape to use instead of store's chunk shape

    Returns:
        int: Total number of chunks
    """
    shape = np.array(store.shape)
    if chunk_shape is None:
        chunk_shape = np.array(store.chunk_layout.read_chunk.shape)
    else:
        chunk_shape = np.array(chunk_shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)
    return np.prod(chunk_counts)
