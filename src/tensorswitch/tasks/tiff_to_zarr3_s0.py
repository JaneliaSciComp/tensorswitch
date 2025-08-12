from dask.cache import Cache
from ..utils import load_tiff_stack, zarr3_store_spec, get_chunk_domains, commit_tasks
import tensorstore as ts
import numpy as np
import psutil
import time
# lazy loading
import tifffile
import dask.array as da
import os

def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None):
    print(f"Loading TIFF stack from: {base_path}", flush=True)

    volume = load_tiff_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    print(f"Original chunk structure from dask: {volume.chunksize}", flush=True)

    # DEBUG
    print(f"Volume dimensions: {len(volume.shape)}D")
    print(f"Volume chunk structure from dask: {volume.chunksize}")
    
    # DEBUG: what a single chunk looks like
    if len(volume.shape) == 4:
        print("4D array detected - likely (C, Z, Y, X)")
        print(f"Channels: {volume.shape[0]}")
        print(f"Z-slices: {volume.shape[1]}")
        print(f"Y (height): {volume.shape[2]}")
        print(f"X (width): {volume.shape[3]}")
        
    # RECHUNK to target size [1, 64, 64, 64]
    if len(volume.shape) == 4:
        target_chunks = (1, 64, 64, 64)
        print(f"Rechunking 4D array to: {target_chunks}")
        volume = volume.rechunk(target_chunks)
    else:
        target_chunks = (64, 64, 64)
        print(f"Rechunking 3D array to: {target_chunks}")
        volume = volume.rechunk(target_chunks)
    
    print(f"After rechunking: {volume.chunksize}")

    # Enable Dask opportunistic cache with 8 GB RAM
    cache = Cache(8 * 1024**3)  # 8 GiB = 8 × 1024³ = 8,589,934,592 bytes
    cache.register()

    # Create or open the output Zarr3 store
    store_spec = zarr3_store_spec(
        path=output_path,
        shape=volume.shape,
        dtype=str(volume.dtype),
        use_shard=use_shard
    )

    store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Prepare chunk domains and filter to assigned range
    chunk_shape = store.chunk_layout.write_chunk.shape
    chunk_domains = get_chunk_domains(chunk_shape, store)
    chunk_domains = chunk_domains[start_idx:stop_idx] if stop_idx is not None else chunk_domains[start_idx:]

    print(f"Processing {len(chunk_domains)} chunks: start={start_idx}, stop={stop_idx}", flush=True)

    tasks = []
    ntasks = 0
    txn = ts.Transaction()

    for domain in chunk_domains:
        # Handle both 3D and 4D arrays dynamically
        if len(volume.shape) == 4:
            slice_data = volume[
                domain.inclusive_min[0]:domain.exclusive_max[0],
                domain.inclusive_min[1]:domain.exclusive_max[1],
                domain.inclusive_min[2]:domain.exclusive_max[2],
                domain.inclusive_min[3]:domain.exclusive_max[3],
            ]
        else:
            slice_data = volume[
                domain.inclusive_min[0]:domain.exclusive_max[0],
                domain.inclusive_min[1]:domain.exclusive_max[1],
                domain.inclusive_min[2]:domain.exclusive_max[2],
            ]
        task = store[domain].with_transaction(txn).write(slice_data.compute())

        """
        task = store[domain].with_transaction(txn).write(
            volume[
                domain.inclusive_min[0]:domain.exclusive_max[0],
                domain.inclusive_min[1]:domain.exclusive_max[1],
                domain.inclusive_min[2]:domain.exclusive_max[2],
            ].compute()
        )
        """

        tasks.append(task)
        ntasks += 1

        txn = commit_tasks(tasks, txn, memory_limit=memory_limit)
    
        if ntasks % 512 == 0:
            #chunk_idx = range(start_idx, stop_idx)[ntasks]
            chunk_idx = range(start_idx, stop_idx)[ntasks] if stop_idx else start_idx + ntasks
            print(f"Queued {ntasks} chunk writes up to {chunk_idx}...", flush=True)
    
    for task in tasks:
        task.result()
    
    txn.commit_sync()
    print(f"Completed writing Zarr3 s0 at: {output_path} [{start_idx}:{stop_idx}]", flush=True)