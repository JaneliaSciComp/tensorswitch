import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, zarr2_store_spec, create_output_store, commit_tasks, print_processing_info, get_total_chunks_from_store

def convert(base_path, output_path, level, start_idx=0, stop_idx=None, memory_limit=50, **kwargs):
    """Convert N5 to Zarr2 format."""
    #n5_level_path = os.path.join(base_path, f"s{level}")
    #zarr_level_path = os.path.join(output_path, f"s{level}")
    n5_level_path = f"{base_path}"
    zarr_level_path = f"{output_path}"
    os.makedirs(zarr_level_path, exist_ok=True)

    n5_store = ts.open(n5_store_spec(n5_level_path)).result()
    shape, chunks = n5_store.shape, n5_store.chunk_layout.read_chunk.shape

    zarr2_spec = zarr2_store_spec(zarr_level_path, shape, chunks)
    zarr2_store = create_output_store(zarr2_spec)

    total_chunks = get_total_chunks_from_store(zarr2_store, chunk_shape=chunks)
    print(f" Total chunks to write: {total_chunks}")
    print(f" Writing from chunk {start_idx} to {stop_idx}")


    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    tasks = []
    txn = ts.Transaction()
    linear_indices_to_process = range(start_idx, stop_idx)
    for chunk_domain in get_chunk_domains(chunks, zarr2_store, linear_indices_to_process=linear_indices_to_process):
        task = zarr2_store[chunk_domain].with_transaction(txn).write(n5_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        txn.commit_sync()
    print(f"Conversion complete for {n5_level_path} to {zarr_level_path}")