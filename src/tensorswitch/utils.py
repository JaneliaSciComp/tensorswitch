import tensorstore as ts
import numpy as np
import requests
import psutil
import os
import tifffile
import dask_image
from dask_image import imread as dask_imread
from urllib.parse import urlparse, unquote

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
"""
def n5_store_spec(n5_level_path):
    return {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_level_path}
    }
"""

def n5_store_spec(n5_level_path):
    if n5_level_path.startswith("http"):
        parsed = urlparse(n5_level_path)
        return {
            'driver': 'n5',
            'kvstore': {
                'driver': 'http',
                'base_url': f"{parsed.scheme}://{parsed.netloc}",
                'path': unquote(parsed.path)
            }
        }
    else:
        return {
            'driver': 'n5',
            'kvstore': {
                'driver': 'file',
                'path': n5_level_path
            }
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
        'base': get_zarr_store_spec(base_spec),
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

def get_input_driver(input_path):
    """
    Detect whether input is TIFF, N5, Zarr2, or Zarr3

    Checks if there is a attributes.json, .zarray, zarr.json, or .tiff/.tif files
    """

    if not os.path.exists(input_path):
        raise ValueError(f"""
        ❌ Could not detect N5/Zarr version or dataset format at: {input_path}.
        {input_path} does not exist.
        """)
    
    n5_path = os.path.join(input_path, "attributes.json")
    zarr2_path = os.path.join(input_path, ".zarray")
    zarr3_path = os.path.join(input_path, "zarr.json")

    if os.path.exists(n5_path):
        input_driver = "n5"
    elif os.path.exists(zarr2_path):
        input_driver = "zarr"
    elif os.path.exists(zarr3_path):
        input_driver = "zarr3"
    else:
        # Check for .tiff or .tif files
        tiff_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".tiff", ".tif"))
        ]
        if tiff_files:
            return "tiff"
            
        raise ValueError(f"""
        ❌ Could not detect N5/Zarr version or dataset format at: {input_path}.
        {n5_path} does not exist.
        {zarr2_path} does not exist.
        {zarr3_path} does not exist.
        And no TIFF files found in folder.
        """)
    return input_driver

def get_zarr_store_spec(path):
    if isinstance(path, dict):
        return path
    
    # If HTTP path, assume N5 driver directly
    if isinstance(path, str) and path.startswith("http"):
        return {
            'driver': 'n5',
            'kvstore': {'driver': 'http', 'base_url': path}
        }
        
    input_driver = get_input_driver(path)

    if input_driver == "tiff":
        raise ValueError(f"❌ Cannot build TensorStore spec for TIFF folder: {path}")
        
    zarr_store_spec = {
        'driver': input_driver,
        'kvstore': {'driver': 'file', 'path': path}
    }
    return zarr_store_spec

def get_total_chunks(dataset):
    """Retrieve total number of chunks dynamically from the dataset."""

    if isinstance(dataset, dict):
        dataset_spec = dataset
    elif isinstance(dataset, str):
        dataset_spec = get_zarr_store_spec(dataset)
    else:
        raise RuntimeError("dataset must either be a dict or str")

    #dataset_store = ts.open(dataset_spec, create=True, open=True, delete_existing=False).result()
    dataset_store = ts.open(dataset_spec, open=True).result() # for "HTTP"

    shape = np.array(dataset_store.shape)
    chunk_shape = np.array(dataset_store.chunk_layout.read_chunk.shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)
    return np.prod(chunk_counts)

def load_tiff_stack(folder):
    return dask_imread.imread(folder + "/*.tiff")

def estimate_total_chunks_for_tiff(input_path, chunk_shape=(64, 64, 64)):
    """
    Estimate total number of chunks for a TIFF stack,
    given a default chunk shape.
    """
    print(f"Estimating total chunks for TIFF input at {input_path}...")

    file_list = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith((".tiff", ".tif"))
    ])
    if not file_list:
        raise ValueError("No TIFF files found.")

    # Load the shape of a single slice
    sample = tifffile.imread(file_list[0])
    volume_shape = (len(file_list),) + sample.shape  # (z, y, x)

    chunk_shape = np.array(chunk_shape)
    chunk_counts = np.ceil(np.array(volume_shape) / chunk_shape).astype(int)
    total_chunks = int(np.prod(chunk_counts))
    
    return total_chunks

