import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, create_output_store, commit_tasks, print_processing_info, fetch_http_json, get_total_chunks_from_store

def convert(base_path, output_path, number, level, start_idx=0, stop_idx=None, memory_limit=50, **kwargs):
    """Convert N5 to N5 format."""
    #n5_level_path = f"{base_path}/setup{number}/s{level}"
    #n5_output_path = f"{output_path}/setup{number}/s{level}"
    n5_level_path = f"{base_path}"
    n5_output_path = f"{output_path}"
    os.makedirs(n5_output_path, exist_ok=True)

    #attr_url = f"{n5_level_path}/attributes.json"
    #attr_data = fetch_http_json(attr_url)

    # Get number of cores from LSF environment and set concurrency limits
    num_cores = int(os.getenv("LSB_DJOB_NUMPROC", 1))
    print(f"Setting tensorstore concurrency limits to {num_cores} cores")
    context = {
        "data_copy_concurrency": {"limit": num_cores},
        "file_io_concurrency": {"limit": num_cores}
    }

    # Open source store with context
    n5_input_spec = n5_store_spec(n5_level_path)
    n5_input_spec['context'] = context
    n5_store = ts.open(n5_input_spec).result()

    #shape, chunk_shape = n5_store.shape, [64, 64, 64]

    # Read from original(HTTP) chunk shape but write in specific output chunk shape
    shape, chunk_shape = n5_store.shape, n5_store.chunk_layout.read_chunk.shape
    output_chunk_shape = [64, 64, 64]

    # Try to read source attributes.json to preserve metadata like downsamplingFactors
    source_attrs = None
    try:
        if n5_level_path.startswith("http://") or n5_level_path.startswith("https://"):
            attr_url = f"{n5_level_path}/attributes.json"
            source_attrs = fetch_http_json(attr_url)
            print(f"✓ Fetched attributes from {attr_url}")
        else:
            import json
            attr_path = os.path.join(n5_level_path, "attributes.json")
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    source_attrs = json.load(f)
                print(f"✓ Read attributes from {attr_path}")
    except Exception as e:
        print(f"⚠ Could not read source attributes.json: {e}")

    # Build output metadata
    output_metadata = {
        'dimensions': shape,
        'blockSize': output_chunk_shape, # Write in different chunk shape from source
        'dataType': "uint16",
        'compression': {
            "type": "zstd",
            "level": 5
        }
    }

    # Preserve or infer downsamplingFactors
    if source_attrs and 'downsamplingFactors' in source_attrs:
        # Use downsamplingFactors from source
        output_metadata['downsamplingFactors'] = source_attrs['downsamplingFactors']
        print(f"✓ Preserving downsamplingFactors from source: {source_attrs['downsamplingFactors']}")
    else:
        # Infer downsamplingFactors from path (e.g., .../s0/, .../s1/, etc.)
        # This is specifically for Keller Lab N5 datasets
        downsampling_map = {
            's0': [1, 1, 1],
            's1': [2, 2, 1],
            's2': [4, 4, 1],
            's3': [8, 8, 2],
            's4': [16, 16, 4]
        }

        # Extract level from path (look for /s0/, /s1/, etc.)
        import re
        level_match = re.search(r'/s(\d)(?:/|$)', n5_level_path)
        if level_match:
            level_str = f"s{level_match.group(1)}"
            if level_str in downsampling_map:
                output_metadata['downsamplingFactors'] = downsampling_map[level_str]
                print(f"✓ Inferred downsamplingFactors for {level_str}: {downsampling_map[level_str]}")
        else:
            print(f"⚠ Could not infer level from path, downsamplingFactors not set")

    n5_output_spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_output_path},
        'metadata': output_metadata,
        'context': context
    }

    n5_output_store = create_output_store(n5_output_spec)
    total_chunks = get_total_chunks_from_store(n5_output_store, chunk_shape=chunk_shape)
    print(f"Generated {total_chunks} chunk domains")
    print(f"Start index: {start_idx}, Stop index: {stop_idx}")

    print("Reading from source N5...")

    sample_chunk_domain_for_print = next(iter(get_chunk_domains(chunk_shape, n5_output_store)))
    sample = n5_store[sample_chunk_domain_for_print].read().result()
    print("Sample shape:", sample.shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    tasks = []
    txn = ts.Transaction()
    linear_indices_to_process = range(start_idx, stop_idx)
    for idx, chunk_domain in enumerate(get_chunk_domains(chunk_shape, n5_output_store, linear_indices_to_process=linear_indices_to_process), start=start_idx):
        try:
            array = n5_store[chunk_domain].read().result()
        except Exception as e:
            print(f"[WARNING] Skipping corrupted chunk index {idx} at {chunk_domain}: {e}")
            continue
    
        task = n5_output_store[chunk_domain].with_transaction(txn).write(array)
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)
        print(f"Writing chunk index {idx}: {chunk_domain}, array shape: {array.shape}")


    if txn.open:
        print("Committing final transaction...")
        txn.commit_sync()
        print("Transaction committed.")
        
    print(f"Conversion complete for {n5_level_path} to {n5_output_path}")



