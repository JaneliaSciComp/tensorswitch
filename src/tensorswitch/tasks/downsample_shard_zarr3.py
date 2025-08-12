import tensorstore as ts
import numpy as np
import time
import psutil
from ..utils import get_chunk_domains, create_output_store, commit_tasks, print_processing_info, downsample_spec, zarr3_store_spec, get_input_driver, get_total_chunks_from_store
import os

def process(base_path, output_path, level, start_idx=0, stop_idx=None, downsample=True, use_shard=True, memory_limit=50, **kwargs):

    """Downsample and optionally apply sharding to Zarr3 dataset."""
    if base_path.endswith(f"s{level - 1}") or level == 0:
        zarr_input_path = base_path
    else:
        zarr_input_path = os.path.join(base_path, "multiscale", f"s{level - 1}")

    input_driver = get_input_driver(zarr_input_path)
        
    zarr_store_spec = {
        'driver': input_driver,
        'kvstore': {'driver': 'file', 'path': zarr_input_path}
    }

    downsampled_saved_path = f"{output_path}/multiscale/s{level}"

    os.makedirs(os.path.dirname(downsampled_saved_path), exist_ok=True)

    print(f" Downsample: {downsample}, Shard: {use_shard}, Level: {level}")
    print(f"Reading from: {zarr_input_path}")
    print(f"Writing to: {downsampled_saved_path}")

    zarr_store = ts.open(zarr_store_spec).result()

    if downsample and level > 0:
        downsample_spec_dict = downsample_spec(zarr_store_spec)
        downsample_store = ts.open(downsample_spec_dict).result()
    else:
        downsample_store = zarr_store

    downsampled_saved_spec = zarr3_store_spec(
        downsampled_saved_path,
        downsample_store.shape,
        downsample_store.dtype.name,
        use_shard
    )

    downsampled_saved = create_output_store(downsampled_saved_spec)

    #chunk_shape = downsampled_saved.chunk_layout.write_chunk.shape
    #chunk_domains = get_chunk_domains(chunk_shape, downsampled_saved)
    
    chunk_shape = downsample_store.chunk_layout.read_chunk.shape
    print("Shape of downsample_store:", downsample_store.shape)
    print("Chunk shape used:", chunk_shape)
    # compute chunk domains based on the downsampled input when goes from s0 to s1
    total_chunks = get_total_chunks_from_store(downsample_store, chunk_shape=chunk_shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    tasks = []
    txn = ts.Transaction()
    linear_indices_to_process = range(start_idx, stop_idx)
    for chunk_domain in get_chunk_domains(chunk_shape, downsample_store, linear_indices_to_process=linear_indices_to_process):
        task = downsampled_saved[chunk_domain].with_transaction(txn).write(downsample_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        txn.commit_sync()
    print(f"Downsampling complete for level {level}, chunks {start_idx} to {stop_idx}")
   