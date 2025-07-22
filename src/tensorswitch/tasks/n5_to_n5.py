import tensorstore as ts
import os
import time
import psutil
from ..utils import get_chunk_domains, n5_store_spec, create_output_store, commit_tasks, print_processing_info, fetch_http_json

def convert(base_path, output_path, number, level, start_idx=0, stop_idx=None, memory_limit=50, **kwargs):
    """Convert N5 to N5 format."""
    #n5_level_path = f"{base_path}/setup{number}/s{level}"
    #n5_output_path = f"{output_path}/setup{number}/s{level}"
    n5_level_path = f"{base_path}"
    n5_output_path = f"{output_path}"
    os.makedirs(n5_output_path, exist_ok=True)

    #attr_url = f"{n5_level_path}/attributes.json"
    #attr_data = fetch_http_json(attr_url)

    n5_store = ts.open(n5_store_spec(n5_level_path)).result()

    shape, chunk_shape = n5_store.shape, [64, 64, 64]

    n5_output_spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_output_path},
        'metadata': {
            'dimensions': shape,
            'blockSize': chunk_shape,
            'dataType': "uint16",
            'compression': {
                "type": "blosc",
                "cname": "zstd",
                "clevel": 2,
                "shuffle": 0
            }
        }
    }

    n5_output_store = create_output_store(n5_output_spec)
    chunk_domains = get_chunk_domains(chunk_shape, n5_output_store)
    print(f"Generated {len(chunk_domains)} chunk domains")
    print(f"Start index: {start_idx}, Stop index: {stop_idx}")
    print("First few chunk domains:", chunk_domains[:3])

    print("Reading from source N5...")
    sample = n5_store[chunk_domains[0]].read().result()
    print("Sample shape:", sample.shape)


    if stop_idx is None:
        stop_idx = len(chunk_domains)

    print_processing_info(level, start_idx, stop_idx, len(chunk_domains))

    tasks = []
    txn = ts.Transaction()
    for chunk_domain in chunk_domains[start_idx:stop_idx]:
        task = n5_output_store[chunk_domain].with_transaction(txn).write(n5_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        print("Committing final transaction...")
        txn.commit_sync()
        print("Transaction committed.")
        
    print(f"Conversion complete for {n5_level_path} to {n5_output_path}")



