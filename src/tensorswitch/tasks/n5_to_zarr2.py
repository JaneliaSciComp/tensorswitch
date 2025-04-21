import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, zarr2_store_spec, create_output_store, commit_tasks, print_processing_info

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

    chunk_domains = get_chunk_domains(chunks, zarr2_store)
    print(f" Total chunks to write: {len(chunk_domains)}")
    print(f" Writing from chunk {start_idx} to {stop_idx}")


    if stop_idx is None:
        stop_idx = len(chunk_domains)

    print_processing_info(level, start_idx, stop_idx, len(chunk_domains))

    tasks = []
    txn = ts.Transaction()
    for chunk_domain in chunk_domains[start_idx:stop_idx]:
        task = zarr2_store[chunk_domain].with_transaction(txn).write(n5_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        txn.commit_sync()
    print(f"Conversion complete for {n5_level_path} to {zarr_level_path}")