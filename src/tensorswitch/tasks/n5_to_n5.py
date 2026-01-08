import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, create_output_store, commit_tasks, print_processing_info, fetch_http_json, fetch_remote_json, get_total_chunks_from_store, get_tensorstore_context, detect_source_order

# Set umask for team permissions
os.umask(0o0002)

def convert(base_path, output_path, number, level, start_idx=0, stop_idx=None, memory_limit=50, custom_chunk_shape=None, **kwargs):
    """Convert N5 to N5 format with optional custom chunk shape."""
    #n5_level_path = f"{base_path}/setup{number}/s{level}"
    #n5_output_path = f"{output_path}/setup{number}/s{level}"
    n5_level_path = f"{base_path}"
    n5_output_path = f"{output_path}"
    os.makedirs(n5_output_path, exist_ok=True)

    #attr_url = f"{n5_level_path}/attributes.json"
    #attr_data = fetch_http_json(attr_url)

    # Add TensorStore context to limit concurrency to LSF allocation
    context = get_tensorstore_context()

    # Open source store with context
    n5_input_spec = n5_store_spec(n5_level_path)
    n5_input_spec['context'] = context
    n5_store = ts.open(n5_input_spec).result()

    #shape, chunk_shape = n5_store.shape, [64, 64, 64]

    # Read from original (local/HTTP/GCS/S3) chunk shape but write in specific output chunk shape
    shape, chunk_shape = n5_store.shape, n5_store.chunk_layout.read_chunk.shape
    # Use custom chunk shape if provided, otherwise default to [128, 128, 128]
    output_chunk_shape = custom_chunk_shape if custom_chunk_shape else [128, 128, 128]
    print(f"Output chunk shape: {output_chunk_shape}")

    # Detect source data order
    source_order_info = detect_source_order(n5_store)
    print(f"\nSource data order: {source_order_info['description']}")
    print(f"  Inner order: {source_order_info['inner_order']}")
    print(f"  Detected axes: {source_order_info['suggested_axes']}")
    # NOTE: N5 → N5 conversion preserves order automatically via TensorStore
    if source_order_info['is_fortran_order']:
        print(f"✓ Source is F-order → Output will preserve F-order")
    else:
        print(f"✓ Source is C-order → Output will preserve C-order")

    # Try to read source attributes.json to preserve metadata like downsamplingFactors
    source_attrs = None
    try:
        if n5_level_path.startswith(("http://", "https://", "gs://", "s3://")):
            # Remote URL (HTTP, HTTPS, GCS, S3)
            attr_url = f"{n5_level_path}/attributes.json"
            source_attrs = fetch_remote_json(attr_url)
            print(f"✓ Fetched attributes from {attr_url}")
        else:
            # Local file path
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

    # Process chunks with transaction-per-chunk pattern (Mark's fix)
    tasks = []
    linear_indices_to_process = range(start_idx, stop_idx)
    for idx, chunk_domain in enumerate(get_chunk_domains(chunk_shape, n5_output_store, linear_indices_to_process=linear_indices_to_process), start=start_idx):
        try:
            array = n5_store[chunk_domain].read().result()
        except Exception as e:
            print(f"[WARNING] Skipping corrupted chunk index {idx} at {chunk_domain}: {e}")
            continue

        # Create transaction per chunk to prevent loading all data simultaneously
        with ts.Transaction() as txn:
            task = n5_output_store[chunk_domain].with_transaction(txn).write(array)
        tasks.append(task)
        print(f"Writing chunk index {idx}: {chunk_domain}, array shape: {array.shape}")

    # Wait for all tasks to complete
    for task in tasks:
        task.result()

    print(f"Conversion complete for {n5_level_path} to {n5_output_path}")



