import tensorstore as ts
import numpy as np
import requests
import psutil

def get_chunk_domains(chunk_shape, array):
    first_chunk_domain = ts.IndexDomain(inclusive_min=array.origin, shape=chunk_shape)
    chunk_number = -(np.array(array.shape) // -np.array(chunk_shape))
    
    cz, cy, cx = chunk_shape
    zn, yn, xn = chunk_number

    chunk_domains = []
    for zi in range(zn):
        for yi in range(yn):
            for xi in range(xn):
                chunk_domain = first_chunk_domain.translate_by[cz*zi, cy*yi, cx*xi].intersect(array.domain)
                chunk_domains.append(chunk_domain)

    return chunk_domains

def n5_store_spec(n5_level_path):
    return {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_level_path}
    }

def zarr2_store_spec(zarr_level_path, shape, chunks):
    return {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': zarr_level_path},
        'metadata': {
            'shape': shape,
            'chunks': chunks,
            'dtype': "|u1",
            'compressor': {'id': 'zstd', 'level': 5},
            'dimension_separator': "/"
        }
    }


def zarr3_store_spec(path, shape, dtype, use_shard=True):
    if use_shard:
        codecs = [
            {
                'name': 'sharding_indexed',
                'configuration': {
                    'chunk_shape': [32, 32, 32],
                    'codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'zstd', 'configuration': {'level': 5}}
                    ],
                    'index_codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'crc32c'}
                    ],
                    'index_location': 'end'
                }
            }
        ]
        chunk_shape = [1024, 1024, 1024]
    else:
        codecs = [
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {'name': 'zstd', 'configuration': {'level': 1}}
        ]
        chunk_shape = [64, 64, 64]

    return {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': path},
        'metadata': {
            'shape': shape,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': chunk_shape}},
            'chunk_key_encoding': {'name': 'default'},
            'data_type': dtype,
            'node_type': 'array',
            'codecs': codecs
        }
    }

def downsample_spec(base_spec):
    return {
        'driver': 'downsample',
        'base': base_spec,
        'downsample_factors': [2, 2, 2],
        'downsample_method': 'mode'
    }

def create_output_store(spec):
    with ts.Transaction() as txn:
        store = ts.open(spec, create=True, open=True, delete_existing=False).result()
    return store

def commit_tasks(tasks, txn, memory_limit=50):
    if len(tasks) % 100 == 0 or psutil.virtual_memory().percent > memory_limit:
        for t in tasks:
            t.result()
        txn.commit_sync()
        tasks.clear()
        return ts.Transaction()
    return txn

def print_processing_info(level, start_idx, stop_idx, total_chunks):
    print(f"[Level {level}] Processing chunks {start_idx} to {stop_idx} out of {total_chunks}")

def get_shape_and_chunks(store):
    return store.shape, store.chunk_layout.read_chunk.shape

def fetch_http_json(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch JSON from {url}")
    return response.json()


def build_job_name(task, level, volume_idx):
    return f"{task}_s{level}_vol{volume_idx}"

def get_total_chunks(dataset_path):
    """Retrieve total number of chunks dynamically from the dataset."""
    dataset_spec = {
        'driver': 'n5' if dataset_path.endswith(".n5") else 'zarr',
        'kvstore': {'driver': 'file', 'path': dataset_path},
    }
    dataset_store = ts.open(dataset_spec).result()

    shape = np.array(dataset_store.shape)
    chunk_shape = np.array(dataset_store.chunk_layout.read_chunk.shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)
    return np.prod(chunk_counts)
