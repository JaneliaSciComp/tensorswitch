import tensorstore as ts
import os
import time
import psutil
import numpy as np
from ..utils import (get_chunk_domains, n5_store_spec, zarr3_store_spec,
                    create_output_store, commit_tasks, get_total_chunks_from_store,
                    fetch_remote_json)

def convert(base_path, output_path, level=0, start_idx=0, stop_idx=None,
           memory_limit=50, use_shard=True, use_ome_structure=True,
           custom_shard_shape=None, custom_chunk_shape=None,
           use_v2_encoding=True, **kwargs):
    """Convert N5 to Zarr3 with sharding."""

    print(f"N5 to Zarr3 conversion")
    print(f"Input: {base_path}")
    print(f"Output: {output_path}")

    os.umask(0o0002)

    # Get LSF cores for concurrency
    num_cores = int(os.getenv("LSB_DJOB_NUMPROC", 1))
    context = {
        "data_copy_concurrency": {"limit": num_cores},
        "file_io_concurrency": {"limit": num_cores}
    }

    # Open N5 source
    n5_input_spec = n5_store_spec(base_path)
    n5_input_spec['context'] = context
    n5_store = ts.open(n5_input_spec).result()

    shape = n5_store.shape
    dtype = str(n5_store.dtype.numpy_dtype)
    n5_chunk_shape = n5_store.chunk_layout.read_chunk.shape

    print(f"Shape: {shape}, dtype: {dtype}")
    print(f"N5 chunk shape: {n5_chunk_shape}")

    # Read N5 attributes if available
    source_attrs = None
    try:
        if base_path.startswith(("http://", "https://", "gs://", "s3://")):
            attr_url = f"{base_path}/attributes.json"
            source_attrs = fetch_remote_json(attr_url)
        else:
            import json
            attr_path = os.path.join(base_path, "attributes.json")
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    source_attrs = json.load(f)
                if 'pixelResolution' in source_attrs:
                    print(f"Pixel resolution: {source_attrs['pixelResolution']}")
                if 'downsamplingFactors' in source_attrs:
                    print(f"Downsampling factors: {source_attrs['downsamplingFactors']}")
    except Exception as e:
        print(f"Warning: Could not read N5 attributes: {e}")

    # Set WebKnossos defaults
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
    if custom_shard_shape is None:
        custom_shard_shape = [1024, 1024, 1024]

    print(f"Chunk shape: {custom_chunk_shape}")
    print(f"Shard shape: {custom_shard_shape}")

    # Create Zarr3 output
    level_path = f"s{level}" if use_ome_structure else None

    store_spec = zarr3_store_spec(
        path=output_path,
        shape=shape,
        dtype=dtype,
        use_shard=use_shard,
        level_path=level_path or f"s{level}",
        use_ome_structure=use_ome_structure,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape,
        use_v2_encoding=use_v2_encoding
    )
    store_spec['context'] = context

    zarr3_store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Pre-create shard directories
    if use_shard and custom_shard_shape:
        print("Pre-creating shard directories...")

        if use_ome_structure:
            base_check_path = os.path.join(output_path, level_path or f"s{level}")
        else:
            base_check_path = output_path

        os.makedirs(base_check_path, exist_ok=True)

        shard_shape = custom_shard_shape if isinstance(custom_shard_shape, list) else [int(x) for x in custom_shard_shape.split(',')]
        num_shards = [((dim_size + shard_size - 1) // shard_size) for dim_size, shard_size in zip(shape, shard_shape)]

        print(f"Shards per dimension: {num_shards}")

        total_dirs = 0
        start_time = time.time()

        # Zarr3 with default chunk key encoding uses 'c/' prefix regardless of dimensionality
        # With v2 encoding, it uses dimension indices directly
        # For simplicity and WebKnossos compatibility, always use 'c/' prefix for Zarr3
        base_shard_path = os.path.join(base_check_path, "c")

        if len(num_shards) == 3:  # ZYX (or similar 3D)
            for z_idx in range(num_shards[0]):
                for y_idx in range(num_shards[1]):
                    dir_path = os.path.join(base_shard_path, str(z_idx), str(y_idx))
                    os.makedirs(dir_path, exist_ok=True)
                    total_dirs += 1
        elif len(num_shards) == 4:  # CZYX
            for c in range(num_shards[0]):
                for z in range(num_shards[1]):
                    for y in range(num_shards[2]):
                        dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
                        total_dirs += 1

        elapsed = time.time() - start_time
        print(f"Created {total_dirs} directories in {elapsed:.2f}s")

    # Calculate chunks
    chunk_shape = zarr3_store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(zarr3_store, chunk_shape=chunk_shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print(f"Total chunks: {total_chunks}")
    print(f"Processing: {start_idx} to {stop_idx}")

    linear_indices_to_process = range(start_idx, stop_idx)
    chunk_domains = get_chunk_domains(chunk_shape, zarr3_store, linear_indices_to_process=linear_indices_to_process)

    # Process chunks
    print(f"Converting {len(linear_indices_to_process)} chunks...")

    tasks = []
    txn = ts.Transaction()
    processed = 0
    start_time = time.time()
    last_report = start_time

    for idx, chunk_domain in enumerate(chunk_domains, start=start_idx):
        try:
            array = n5_store[chunk_domain].read().result()
            task = zarr3_store[chunk_domain].with_transaction(txn).write(array)
            tasks.append(task)
            processed += 1

            txn = commit_tasks(tasks, txn, memory_limit)

            # Progress every 100 chunks or 30s
            now = time.time()
            if processed % 100 == 0 or (now - last_report) > 30:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(linear_indices_to_process) - processed) / rate if rate > 0 else 0
                mem = psutil.virtual_memory().percent
                print(f"{processed}/{len(linear_indices_to_process)} | {rate:.1f} chunks/s | ETA {eta/60:.1f}m | Mem {mem:.1f}%")
                last_report = now

        except Exception as e:
            print(f"Warning: Skipping chunk {idx}: {e}")
            continue

    if txn.open:
        txn.commit_sync()

    elapsed = time.time() - start_time
    print(f"Complete: {processed} chunks in {elapsed:.1f}s ({processed/elapsed:.1f} chunks/s)")


def process(base_path, output_path, level=0, start_idx=0, stop_idx=None,
           memory_limit=50, use_shard=True, use_ome_structure=True,
           custom_shard_shape=None, custom_chunk_shape=None,
           use_v2_encoding=True, **kwargs):
    """Alias for convert()."""
    return convert(base_path=base_path, output_path=output_path, level=level,
                  start_idx=start_idx, stop_idx=stop_idx, memory_limit=memory_limit,
                  use_shard=use_shard, use_ome_structure=use_ome_structure,
                  custom_shard_shape=custom_shard_shape,
                  custom_chunk_shape=custom_chunk_shape,
                  use_v2_encoding=use_v2_encoding, **kwargs)
