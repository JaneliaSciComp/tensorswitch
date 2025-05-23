import tensorstore as ts
import numpy as np
import time
import psutil
from ..utils import get_chunk_domains, create_output_store, commit_tasks, print_processing_info, downsample_spec, zarr3_store_spec, get_input_driver
import os

def process(base_path, output_path, level, start_idx=0, stop_idx=None, downsample=True, use_shard=True, memory_limit=50, **kwargs):

    """Downsample and optionally apply sharding to Zarr3 dataset."""
    '''
    if level == 0:
        zarr_input_path = base_path
    else:
        zarr_input_path = f"{base_path}/multiscale/s{level-1}"
    '''
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

    '''
    zarr_store_spec = {
        'driver': 'zarr' + ('3' if level > 0 else ''),
        'kvstore': {'driver': 'file', 'path': zarr_input_path}
    }
    '''
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

    chunk_shape = downsampled_saved.chunk_layout.write_chunk.shape
    chunk_domains = get_chunk_domains(chunk_shape, downsampled_saved)

    if stop_idx is None:
        stop_idx = len(chunk_domains)

    print_processing_info(level, start_idx, stop_idx, len(chunk_domains))

    tasks = []
    txn = ts.Transaction()
    for chunk_domain in chunk_domains[start_idx:stop_idx]:
        task = downsampled_saved[chunk_domain].with_transaction(txn).write(downsample_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        txn.commit_sync()
    print(f"Downsampling complete for level {level}, chunks {start_idx} to {stop_idx}")
   