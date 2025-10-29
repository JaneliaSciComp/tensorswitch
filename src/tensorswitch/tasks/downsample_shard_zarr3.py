import tensorstore as ts
import numpy as np
import time
import psutil
from ..utils import get_chunk_domains, create_output_store, commit_tasks, print_processing_info, downsample_spec, zarr3_store_spec, get_input_driver, get_total_chunks_from_store
import os

def process(base_path, output_path, level, start_idx=0, stop_idx=None, downsample=True, use_shard=True, memory_limit=50, custom_shard_shape=None, custom_chunk_shape=None, **kwargs):

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

    downsampled_saved_path = output_path

    os.makedirs(f"{output_path}/multiscale", exist_ok=True)

    print(f" Downsample: {downsample}, Shard: {use_shard}, Level: {level}")
    print(f"Reading from: {zarr_input_path}")
    print(f"Writing to: {downsampled_saved_path}")

    zarr_store = ts.open(zarr_store_spec).result()

    if downsample and level > 0:
        # Extract dimension_names from the zarr store for proper downsampling
        dimension_names = None
        try:
            # Read dimension_names from zarr.json file directly
            import json
            zarr_json_path = os.path.join(zarr_input_path, 'zarr.json')
            if os.path.exists(zarr_json_path):
                with open(zarr_json_path, 'r') as f:
                    metadata = json.load(f)
                    dimension_names = metadata.get('dimension_names')
                    print(f"Extracted dimension_names from zarr.json: {dimension_names}")
        except Exception as e:
            print(f"Warning: Could not extract dimension_names: {e}")

        if not dimension_names:
            print("Warning: No dimension_names found, defaulting to [2,2,2] downsampling")

        print(f"Downsampling with dimension_names: {dimension_names}")
        downsample_spec_dict = downsample_spec(zarr_store_spec, zarr_store.shape, dimension_names)
        downsample_store = ts.open(downsample_spec_dict).result()
    else:
        downsample_store = zarr_store

    downsampled_saved_spec = zarr3_store_spec(
        downsampled_saved_path,
        downsample_store.shape,
        downsample_store.dtype.name,
        use_shard,
        level_path=f"s{level}",
        use_ome_structure=True,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape
    )

    # Create basic output directory structure
    output_array_path = f"{downsampled_saved_path}/multiscale/s{level}"
    os.makedirs(output_array_path, exist_ok=True)

    downsampled_saved = create_output_store(downsampled_saved_spec)

    # Pre-create shard directory structure to avoid race conditions with parallel workers
    if use_shard and custom_shard_shape:
        print("Pre-creating shard directory structure...")
        shard_shape = custom_shard_shape if isinstance(custom_shard_shape, list) else [int(x) for x in custom_shard_shape.split(',')]
        output_shape = list(downsample_store.shape)

        # Adjust shard shape to match array dimensions
        if len(output_shape) == 4 and len(shard_shape) == 3:
            shard_shape = [1] + shard_shape  # CZYX
        elif len(output_shape) == 3 and len(shard_shape) == 2:
            shard_shape = [1] + shard_shape  # CYX

        # Calculate number of shards in each dimension
        num_shards = [((dim_size + shard_size - 1) // shard_size) for dim_size, shard_size in zip(output_shape, shard_shape)]

        # Create all shard parent directories
        base_shard_path = os.path.join(output_array_path, "c")
        if len(num_shards) == 4:  # CZYX
            for c in range(num_shards[0]):
                for z in range(num_shards[1]):
                    for y in range(num_shards[2]):
                        dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
        elif len(num_shards) == 3:  # CYX or ZYX
            for dim0 in range(num_shards[0]):
                for dim1 in range(num_shards[1]):
                    dir_path = os.path.join(base_shard_path, str(dim0), str(dim1))
                    os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory structure for {np.prod(num_shards[:3] if len(num_shards) >= 3 else num_shards)} shard locations")

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