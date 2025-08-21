from dask.cache import Cache
from ..utils import (load_nd2_stack, zarr3_store_spec, get_chunk_domains, commit_tasks, 
                    get_total_chunks_from_store, extract_nd2_ome_metadata, 
                    convert_ome_to_zarr3_metadata, write_zarr3_group_metadata)
import tensorstore as ts
import numpy as np
import psutil
import time
# lazy loading
import nd2
import dask.array as da
import os

def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True):
    print(f"Loading ND2 file from: {base_path}", flush=True)

    volume = load_nd2_stack(base_path)
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
    elif len(volume.shape) == 3:
        print("3D array detected - likely (Z, Y, X)")
        print(f"Z-slices: {volume.shape[0]}")
        print(f"Y (height): {volume.shape[1]}")
        print(f"X (width): {volume.shape[2]}")

    # Enable Dask cache with 8 GB RAM
    cache = Cache(8 * 1024**3)  # 8 GiB = 8 × 1024³ = 8,589,934,592 bytes
    cache.register()

    # Create or open the output Zarr3 store
    store_spec = zarr3_store_spec(
        path=output_path,
        shape=volume.shape,
        dtype=str(volume.dtype),
        use_shard=use_shard,
        use_ome_structure=use_ome_structure
    )

    store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Prepare chunk domains and filter to assigned range
    chunk_shape = store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(store, chunk_shape=chunk_shape)
    print(f"Total chunks: {total_chunks}")
    linear_indices_to_process = range(start_idx, stop_idx or total_chunks)
    chunk_domains = get_chunk_domains(chunk_shape, store, linear_indices_to_process=linear_indices_to_process)

    print(f"Processing {len(linear_indices_to_process)} chunks: start={start_idx}, stop={stop_idx}", flush=True)

    tasks = []
    ntasks = 0
    txn = ts.Transaction()

    for domain in chunk_domains:
        # Handle both 3D and 4D arrays dynamically
        slices = tuple(slice(min, max) for (min,max) in zip(domain.inclusive_min, domain.exclusive_max))
        slice_data = volume[slices]
        task = store[domain].with_transaction(txn).write(slice_data.compute())

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
    
    # Write OME-Zarr metadata only if using OME structure
    if use_ome_structure:
        print("Writing OME-Zarr metadata...", flush=True)
        try:
            ome_metadata = extract_nd2_ome_metadata(base_path)
            # Extract image name from file path
            import os
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            
            zarr3_metadata = convert_ome_to_zarr3_metadata(ome_metadata, volume.shape, image_name)
            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully", flush=True)
        except Exception as e:
            print(f"Warning: Could not write OME-Zarr metadata: {e}", flush=True)
    else:
        print("Skipping OME-ZARR metadata (plain zarr3 format)", flush=True)
    
    print(f"Completed writing Zarr3 s0 at: {output_path} [{start_idx}:{stop_idx}]", flush=True)