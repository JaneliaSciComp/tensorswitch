import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, zarr2_store_spec, create_output_store, commit_tasks, print_processing_info, get_total_chunks_from_store, get_tensorstore_context, detect_source_order

def convert(base_path, output_path, level, start_idx=0, stop_idx=None, memory_limit=50, **kwargs):
    """Convert N5 to Zarr2 format."""
    #n5_level_path = os.path.join(base_path, f"s{level}")
    #zarr_level_path = os.path.join(output_path, f"s{level}")
    n5_level_path = f"{base_path}"
    zarr_level_path = f"{output_path}"
    os.makedirs(zarr_level_path, exist_ok=True)

    # Add TensorStore context to limit concurrency to LSF allocation
    n5_spec = n5_store_spec(n5_level_path)
    n5_spec['context'] = get_tensorstore_context()
    n5_store = ts.open(n5_spec).result()
    shape, chunks = n5_store.shape, n5_store.chunk_layout.read_chunk.shape

    # Detect source data order
    source_order_info = detect_source_order(n5_store)
    print(f"Source data order: {source_order_info['description']}")
    print(f"  Inner order: {source_order_info['inner_order']}")
    print(f"  Detected axes: {source_order_info['suggested_axes']}")

    # NOTE: TensorStore will preserve order when copying N5 → Zarr2
    if source_order_info['is_fortran_order']:
        print(f"✓ Preserving F-order in Zarr2 output")
    else:
        print(f"✓ Preserving C-order in Zarr2 output")

    zarr2_spec = zarr2_store_spec(zarr_level_path, shape, chunks)
    zarr2_spec['context'] = get_tensorstore_context()
    zarr2_store = create_output_store(zarr2_spec)

    total_chunks = get_total_chunks_from_store(zarr2_store, chunk_shape=chunks)
    print(f" Total chunks to write: {total_chunks}")
    print(f" Writing from chunk {start_idx} to {stop_idx}")


    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    # Process chunks with transaction-per-chunk pattern (Mark's fix)
    tasks = []
    linear_indices_to_process = range(start_idx, stop_idx)
    for chunk_domain in get_chunk_domains(chunks, zarr2_store, linear_indices_to_process=linear_indices_to_process):
        # Create transaction per chunk to prevent loading all data simultaneously
        with ts.Transaction() as txn:
            task = zarr2_store[chunk_domain].with_transaction(txn).write(n5_store[chunk_domain])
        tasks.append(task)

    # Wait for all tasks to complete
    for task in tasks:
        task.result()

    print(f"Conversion complete for {n5_level_path} to {zarr_level_path}")