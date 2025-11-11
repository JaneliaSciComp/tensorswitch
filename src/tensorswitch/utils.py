import tensorstore as ts
import numpy as np
import requests
import psutil
import os
import tifffile
import dask_image
from dask_image import imread as dask_imread
from urllib.parse import urlparse, unquote
import dask.array as da
import itertools
import nd2
import h5py
import json
import glob
from ome_types import from_xml
import xml.etree.ElementTree as ET
import re

def get_kvstore_spec(path):
    """
    Get kvstore specification for TensorStore, supporting HTTP, GCS, S3, and local file paths.

    Args:
        path: File path, HTTP/HTTPS URL, GCS URL (gs://), or S3 URL (s3://)

    Returns:
        dict: kvstore spec with appropriate driver (http, gcs, s3, or file)

    Examples:
        >>> get_kvstore_spec("http://example.com/data")
        {'driver': 'http', 'base_url': 'http://example.com', 'path': '/data'}

        >>> get_kvstore_spec("https://s3prfs.int.janelia.org/bucket/path")
        {'driver': 'http', 'base_url': 'https://s3prfs.int.janelia.org', 'path': '/bucket/path'}

        >>> get_kvstore_spec("gs://bucket-name/path/to/data")
        {'driver': 'gcs', 'bucket': 'bucket-name', 'path': 'path/to/data'}

        >>> get_kvstore_spec("s3://bucket-name/path/to/data")
        {'driver': 's3', 'bucket': 'bucket-name', 'path': 'path/to/data'}

        >>> get_kvstore_spec("/local/path/to/data")
        {'driver': 'file', 'path': '/local/path/to/data'}
    """
    if path.startswith("http://") or path.startswith("https://"):
        # HTTP/HTTPS URL (includes S3 HTTP endpoints like s3prfs.int.janelia.org)
        parsed = urlparse(path)
        return {
            'driver': 'http',
            'base_url': f"{parsed.scheme}://{parsed.netloc}",
            'path': unquote(parsed.path)
        }
    elif path.startswith("gs://"):
        # Google Cloud Storage URL
        # Format: gs://bucket-name/path/to/object
        path_without_scheme = path[5:]  # Remove 'gs://'
        parts = path_without_scheme.split('/', 1)
        bucket = parts[0]
        object_path = parts[1] if len(parts) > 1 else ''
        return {
            'driver': 'gcs',
            'bucket': bucket,
            'path': object_path
        }
    elif path.startswith("s3://"):
        # AWS S3 URL (protocol format, less common but supported)
        # Format: s3://bucket-name/path/to/object
        path_without_scheme = path[4:]  # Remove 's3://'
        parts = path_without_scheme.split('/', 1)
        bucket = parts[0]
        object_path = parts[1] if len(parts) > 1 else ''
        return {
            'driver': 's3',
            'bucket': bucket,
            'path': object_path
        }
    else:
        # Local file path
        return {
            'driver': 'file',
            'path': path
        }

def get_chunk_domains(chunk_shape, array, linear_indices_to_process=None):
    first_chunk_domain = ts.IndexDomain(inclusive_min=array.origin, shape=chunk_shape)
    chunk_number = -(np.array(array.shape) // -np.array(chunk_shape))
    
    if linear_indices_to_process is not None:
        linear_indices_iterator = linear_indices_to_process
    else:
        total_chunks = np.prod(chunk_number)
        linear_indices_iterator = range(total_chunks)

    for linear_idx in linear_indices_iterator:
        idx = np.unravel_index(linear_idx, chunk_number)

        yield first_chunk_domain.translate_by[
            tuple(map(lambda i, s: i * s, idx, chunk_shape))
        ].intersect(array.domain)

def n5_store_spec(n5_level_path):
    """
    Create N5 store specification supporting HTTP, GCS, S3, and local file paths.

    Args:
        n5_level_path: Path to N5 dataset (local path, HTTP/HTTPS URL, gs:// URL, or s3:// URL)

    Returns:
        dict: N5 store spec for TensorStore

    Examples:
        >>> n5_store_spec("/local/path/to/n5")
        >>> n5_store_spec("http://example.com/n5")
        >>> n5_store_spec("gs://bucket-name/path/to/n5")
        >>> n5_store_spec("s3://bucket-name/path/to/n5")
    """
    return {
        'driver': 'n5',
        'kvstore': get_kvstore_spec(n5_level_path)
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


def zarr3_store_spec(path, shape, dtype, use_shard=True, level_path="s0", use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None, use_v2_encoding=False, use_fortran_order=False):
    if use_shard:
        # Use custom chunk shape if provided, otherwise default
        inner_chunk_shape = custom_chunk_shape if custom_chunk_shape is not None else [32, 32, 32]
        
        # Adjust inner chunk shape for different array dimensions
        if len(shape) == 3 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = inner_chunk_shape
        elif len(shape) == 3 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1] + inner_chunk_shape  # 2D images: CYX with YX chunks -> add Z=1
        elif len(shape) == 4 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = [1] + inner_chunk_shape  # CZYX
        elif len(shape) == 4 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1, 1] + inner_chunk_shape  # 2D images: CZYX with YX chunks
        elif len(shape) == 5 and len(inner_chunk_shape) == 3:
            adjusted_inner_chunk = [1, 1] + inner_chunk_shape  # TCZYX
        elif len(shape) == 5 and len(inner_chunk_shape) == 2:
            adjusted_inner_chunk = [1, 1, 1] + inner_chunk_shape  # 2D images: TCZYX with YX chunks
        else:
            adjusted_inner_chunk = inner_chunk_shape
        
        # Build inner codecs for sharding (with optional transpose for F-order)
        inner_codecs = []
        if use_fortran_order:
            # Use explicit dimension reversal [n-1, ..., 1, 0] as recommended by Zarr specs
            # For 3D: [2, 1, 0], for 4D: [3, 2, 1, 0], etc.
            transpose_order = list(range(len(shape) - 1, -1, -1))
            inner_codecs.append({'name': 'transpose', 'configuration': {'order': transpose_order}})
        inner_codecs.extend([
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {'name': 'zstd', 'configuration': {'level': 5}}
        ])

        codecs = [
            {
                'name': 'sharding_indexed',
                'configuration': {
                    'chunk_shape': adjusted_inner_chunk,
                    'codecs': inner_codecs,
                    'index_codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'crc32c'}
                    ],
                    'index_location': 'end'
                }
            }
        ]
        
        # Use custom shard shape if provided, otherwise default
        if custom_shard_shape is not None:
            if len(shape) == 3 and len(custom_shard_shape) == 3:
                chunk_shape = custom_shard_shape
            elif len(shape) == 3 and len(custom_shard_shape) == 2:
                chunk_shape = [1] + custom_shard_shape  # 2D images: CYX with YX shards -> add Z=1
            elif len(shape) == 4 and len(custom_shard_shape) == 3:
                chunk_shape = [1] + custom_shard_shape  # CZYX
            elif len(shape) == 4 and len(custom_shard_shape) == 2:
                chunk_shape = [1, 1] + custom_shard_shape  # 2D images: CZYX with YX shards
            elif len(shape) == 5 and len(custom_shard_shape) == 3:
                chunk_shape = [1, 1] + custom_shard_shape  # TCZYX
            elif len(shape) == 5 and len(custom_shard_shape) == 2:
                chunk_shape = [1, 1, 1] + custom_shard_shape  # 2D images: TCZYX with YX shards
            else:
                chunk_shape = custom_shard_shape
        else:
            chunk_shape = [1024, 1024, 1024]
    else:
        # Non-sharded codecs (with optional transpose for F-order)
        codecs = []
        if use_fortran_order:
            # Use explicit dimension reversal [n-1, ..., 1, 0] as recommended by Zarr specs
            # For 3D: [2, 1, 0], for 4D: [3, 2, 1, 0], etc.
            transpose_order = list(range(len(shape) - 1, -1, -1))
            codecs.append({'name': 'transpose', 'configuration': {'order': transpose_order}})
        codecs.extend([
            {'name': 'bytes', 'configuration': {'endian': 'little'}},
            {'name': 'zstd', 'configuration': {'level': 1}}
        ])
        # Handle 5D to 1D arrays, assuming order
        # TODO: Make this depend on axes metadata detection
        if len(shape) == 5:
            chunk_shape = [1, 1, 64, 64, 64] # TCZYX
        elif len(shape) == 4:
            chunk_shape = [1, 64, 64, 64] # CZYX
        elif len(shape) == 3:
            chunk_shape = [64, 64, 64] # ZYX
        elif len(shape) == 2:
            chunk_shape = [64, 64] # YX
        elif len(shape) == 1:
            chunk_shape = [64] # X

    # Create path based on whether OME-ZARR structure is needed
    if use_ome_structure:
        # For OME-ZARR: write to level subdirectory (no multiscale folder)
        array_path = os.path.join(path, level_path)
    else:
        # For plain zarr3: write directly to specified path
        array_path = path
    
    # Determine dimension names based on shape - will be updated in calling function if OME metadata available
    if len(shape) == 3:
        # For 3D, assume channels if first dimension is small, otherwise Z
        if shape[0] <= 10:
            dimension_names = ["c", "y", "x"]
        else:
            dimension_names = ["z", "y", "x"]
    elif len(shape) == 4:
        dimension_names = ["c", "z", "y", "x"]
    elif len(shape) == 5:
        dimension_names = ["t", "c", "z", "y", "x"]
    else:
        dimension_names = [f"dim_{i}" for i in range(len(shape))]
    
    return {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': array_path},
        'metadata': {
            'shape': shape,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': chunk_shape}},
            'chunk_key_encoding': {'name': 'v2', 'configuration': {'separator': '/'}} if use_v2_encoding else {'name': 'default'},
            'data_type': dtype,
            'node_type': 'array',
            'codecs': codecs,
            'dimension_names': dimension_names
        }
    }

def downsample_spec(base_spec, array_shape=None, dimension_names=None, custom_factors=None):
    """
    Create downsample spec with appropriate factors based on dimension names.
    For 2D multi-channel images (CYX), only downsample Y and X, not C.

    Args:
        base_spec: Base zarr store spec
        array_shape: Shape of the array
        dimension_names: List of dimension names (e.g., ['c', 'z', 'y', 'x'])
        custom_factors: Optional custom downsampling factors (e.g., [1, 1, 2, 2] for anisotropic data)
                       If provided, overrides default factor calculation
    """
    # If custom factors provided, use them directly
    if custom_factors is not None:
        downsample_factors = custom_factors
    else:
        # Default: downsample all dimensions by 2x
        downsample_factors = [2] * len(array_shape) if array_shape is not None else [2, 2, 2]

        # If we have dimension_names, adjust factors to preserve non-spatial dimensions
        if dimension_names and array_shape:
            downsample_factors = []
            for dim_name in dimension_names:
                if dim_name in ['c', 't']:  # Channel or Time - don't downsample
                    downsample_factors.append(1)
                else:  # Spatial dimensions (x, y, z) - downsample by 2x
                    downsample_factors.append(2)

    return {
        'driver': 'downsample',
        'base': get_zarr_store_spec(base_spec),
        'downsample_factors': downsample_factors,
        'downsample_method': 'mode'
    }

def precreate_zarr3_metadata_safely(output_path, level, shape, dtype, use_shard,
                                    shard_shape, chunk_shape, use_ome_structure=True,
                                    use_fortran_order=False, use_v2_encoding=False,
                                    force_precreate=False):
    """
    Pre-create zarr.json metadata to avoid worker race conditions.

    Should be called before parallel job submission when:
    - use_fortran_order=True (non-default codec that differs from C-order)
    - force_precreate=True (other codec customizations)
    - Any scenario where workers might create conflicting metadata

    Args:
        output_path: Base output path for the dataset
        level: Level number (e.g., 0 for s0, 1 for s1)
        shape: Array shape tuple
        dtype: Data type string (e.g., 'uint8', 'uint64')
        use_shard: Whether sharding is enabled
        shard_shape: Shard dimensions (e.g., [1024, 1024, 1024])
        chunk_shape: Chunk dimensions (e.g., [32, 32, 32])
        use_ome_structure: Whether to use OME-NGFF structure (s0, s1, etc.)
        use_fortran_order: Whether to use Fortran (F) order instead of C order
        use_v2_encoding: Whether to use v2 chunk key encoding
        force_precreate: Force pre-creation even for standard C-order

    Returns:
        bool: True if metadata was created, False if skipped (not needed)
    """
    import tensorstore as ts
    import os

    # Skip pre-creation for standard C-order (no risk of codec mismatch)
    if not (use_fortran_order or force_precreate):
        return False

    level_path = f"s{level}" if use_ome_structure else None

    # Create the store spec
    store_spec = zarr3_store_spec(
        path=output_path,
        shape=shape,
        dtype=dtype,
        use_shard=use_shard,
        level_path=level_path or f"s{level}",
        use_ome_structure=use_ome_structure,
        custom_shard_shape=shard_shape,
        custom_chunk_shape=chunk_shape,
        use_v2_encoding=use_v2_encoding,
        use_fortran_order=use_fortran_order
    )

    # Create metadata (no data written, just zarr.json)
    store = ts.open(store_spec, create=True, delete_existing=True).result()
    store = None  # Close immediately to release resources

    metadata_path = os.path.join(output_path, level_path or f"s{level}", "zarr.json")
    print(f"✓ Pre-created Zarr3 metadata at: {metadata_path}")
    print(f"  Reason: {'F-order codec' if use_fortran_order else 'Forced pre-creation'}")

    return True

def precreate_shard_directories(output_path, level, output_shape, shard_shape, use_ome_structure=True):
    """
    Pre-create all shard directory structures to avoid race conditions with parallel workers.

    This function should be called ONCE before submitting parallel jobs, not by each worker.
    Workers can check if directories exist and skip redundant creation.

    Args:
        output_path: Base zarr dataset path (e.g., /path/to/data.zarr)
        level: Target level number (0 for s0, 1 for s1, etc.)
        output_shape: Expected output array shape (e.g., [23500, 6500, 1000] for ZYX)
        shard_shape: Shard shape matching output dimensions (e.g., [1024, 1024, 1024])
        use_ome_structure: Whether to use OME-NGFF structure with s{level} subdirectories

    Returns:
        int: Number of directories created

    Example:
        # For s1 with shape [23500, 6500, 1000] and shard [1024, 1024, 1024]
        precreate_shard_directories(
            "/data/image.zarr",
            level=1,
            output_shape=[23500, 6500, 1000],
            shard_shape=[1024, 1024, 1024]
        )
    """
    import time

    print("\n" + "="*80)
    print(f"PRE-CREATING SHARD DIRECTORIES FOR s{level}")
    print("="*80)

    # Convert to lists if needed
    if not isinstance(output_shape, list):
        output_shape = list(output_shape)
    if not isinstance(shard_shape, list):
        shard_shape = list(shard_shape)

    # Adjust shard shape to match array dimensions if needed
    if len(output_shape) == 4 and len(shard_shape) == 3:
        shard_shape = [1] + shard_shape  # CZYX
    elif len(output_shape) == 4 and len(shard_shape) == 2:
        shard_shape = [1, 1] + shard_shape  # CZYX with YX shards
    elif len(output_shape) == 3 and len(shard_shape) == 2:
        shard_shape = [1] + shard_shape  # CYX or ZYX
    elif len(output_shape) == 5 and len(shard_shape) == 3:
        shard_shape = [1, 1] + shard_shape  # TCZYX
    elif len(output_shape) == 5 and len(shard_shape) == 2:
        shard_shape = [1, 1, 1] + shard_shape  # TCZYX with YX shards

    # Calculate number of shards in each dimension
    num_shards = [
        (output_shape[i] + shard_shape[i] - 1) // shard_shape[i]
        for i in range(len(output_shape))
    ]

    total_dirs = np.prod([num_shards[i] for i in range(min(len(num_shards)-1, 3))])  # Skip last X dimension

    print(f"Output shape: {output_shape}")
    print(f"Shard shape: {shard_shape}")
    print(f"Number of shards per dimension: {num_shards}")
    print(f"Total directories to create: {total_dirs}")

    # Determine base path
    if use_ome_structure:
        base_shard_path = os.path.join(output_path, f"s{level}", "c")
    else:
        base_shard_path = os.path.join(output_path, "c")

    print(f"Base shard path: {base_shard_path}")

    # Create all shard parent directories
    created = 0
    start_time = time.time()

    if len(num_shards) == 4:  # CZYX
        for c in range(num_shards[0]):
            for z in range(num_shards[1]):
                for y in range(num_shards[2]):
                    dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                    os.makedirs(dir_path, exist_ok=True)
                    created += 1
                    if created % 100 == 0:
                        print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")
    elif len(num_shards) == 3:  # ZYX or CYX
        for dim0 in range(num_shards[0]):
            for dim1 in range(num_shards[1]):
                dir_path = os.path.join(base_shard_path, str(dim0), str(dim1))
                os.makedirs(dir_path, exist_ok=True)
                created += 1
                if created % 100 == 0:
                    print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")
    elif len(num_shards) == 5:  # TCZYX
        for t in range(num_shards[0]):
            for c in range(num_shards[1]):
                for z in range(num_shards[2]):
                    for y in range(num_shards[3]):
                        dir_path = os.path.join(base_shard_path, str(t), str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
                        created += 1
                        if created % 100 == 0:
                            print(f"  Progress: {created}/{total_dirs} ({100*created/total_dirs:.1f}%)")

    elapsed = time.time() - start_time
    rate = created / elapsed if elapsed > 0 else 0
    print(f"\n✓ Created {created} shard directories in {elapsed:.2f} seconds ({rate:.1f} dirs/sec)")
    print("="*80 + "\n")

    return created


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
    """
    Fetch JSON from HTTP/HTTPS URL.

    For GCS URLs (gs://), use TensorStore to read the file instead.

    Args:
        url: HTTP/HTTPS URL to JSON file

    Returns:
        dict: Parsed JSON content
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch JSON from {url}")
    return response.json()


def fetch_remote_json(url):
    """
    Fetch JSON from remote sources (HTTP, HTTPS, or GCS).

    Args:
        url: URL to JSON file (http://, https://, or gs://)

    Returns:
        dict: Parsed JSON content

    Examples:
        >>> fetch_remote_json("http://example.com/data/attributes.json")
        >>> fetch_remote_json("gs://bucket-name/path/to/attributes.json")
    """
    if url.startswith("gs://"):
        # For GCS, use TensorStore to read the file
        import json as json_module
        kvstore_spec = get_kvstore_spec(url)

        try:
            # Open as a simple key-value store and read the JSON
            store = ts.open({
                'driver': 'json',
                'kvstore': kvstore_spec
            }).result()
            return store.read().result()
        except:
            # Fallback: read as raw bytes and parse
            kvstore = ts.KvStore.open(kvstore_spec).result()
            content_bytes = kvstore.read().result().value
            return json_module.loads(content_bytes.decode('utf-8'))
    else:
        # For HTTP/HTTPS, use requests
        return fetch_http_json(url)


def build_job_name(task, level, volume_idx):
    return f"{task}_s{level}_vol{volume_idx}"

def get_input_driver(input_path):
    """
    Detect whether input is TIFF, N5, Zarr2, Zarr3, ND2, IMS, or Neuroglancer Precomputed

    Checks if there is a attributes.json, .zarray, zarr.json, info, .tiff/.tif, .nd2, or .ims files
    """

    if not os.path.exists(input_path):
        raise ValueError(f"""
        Could not detect N5/Zarr version or dataset format at: {input_path}.
        {input_path} does not exist.
        """)
    
    # Check if input is a single nd2 file
    if os.path.isfile(input_path) and input_path.lower().endswith(".nd2"):
        return "nd2"
    
    # Check if input is a single ims file
    if os.path.isfile(input_path) and input_path.lower().endswith(".ims"):
        return "ims"
    
    # Check if input is a single tiff file
    if os.path.isfile(input_path) and input_path.lower().endswith((".tiff", ".tif")):
        return "tiff"
    
    # For directories, check for format indicator files
    n5_path = os.path.join(input_path, "attributes.json")
    zarr2_path = os.path.join(input_path, ".zarray")
    zarr3_path = os.path.join(input_path, "zarr.json")
    precomputed_path = os.path.join(input_path, "info")

    if os.path.exists(n5_path):
        input_driver = "n5"
    elif os.path.exists(zarr2_path):
        input_driver = "zarr"
    elif os.path.exists(zarr3_path):
        input_driver = "zarr3"
    elif os.path.exists(precomputed_path):
        input_driver = "precomputed"
    else:
        # Check for .tiff or .tif files in directory
        tiff_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".tiff", ".tif"))
        ]
        if tiff_files:
            return "tiff"
            
        # Check for .nd2 files in directory
        nd2_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith(".nd2")
        ]
        if nd2_files:
            return "nd2"
            
        raise ValueError(f"""
        Could not detect N5/Zarr version or dataset format at: {input_path}.
        {n5_path} does not exist.
        {zarr2_path} does not exist.
        {zarr3_path} does not exist.
        {precomputed_path} does not exist.
        And no TIFF, ND2, or IMS files found in folder.
        """)
    return input_driver

def get_zarr_store_spec(path):
    """
    Get TensorStore spec for Zarr/N5 stores, supporting local paths and remote URLs.

    Args:
        path: Path to dataset (local path, HTTP/HTTPS URL, gs:// URL, or s3:// URL) or dict spec

    Returns:
        dict: TensorStore spec with appropriate driver and kvstore
    """
    if isinstance(path, dict):
        return path

    # If remote path (HTTP, GCS, S3), assume N5 driver directly
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://") or
                                   path.startswith("gs://") or path.startswith("s3://")):
        return {
            'driver': 'n5',
            'kvstore': get_kvstore_spec(path)
        }

    input_driver = get_input_driver(path)

    if input_driver == "tiff":
        raise ValueError(f"Cannot build TensorStore spec for TIFF folder: {path}")

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

def get_total_chunks_from_store(store, chunk_shape=None):
    """Retrieve total number of chunks dynamically from a TensorStore store.
    Optionally, specify a chunk_shape; otherwise, it defaults to the store's read_chunk shape.
    """
    shape = np.array(store.shape)
    if chunk_shape is None:
        chunk_shape = np.array(store.chunk_layout.read_chunk.shape)
    else:
        chunk_shape = np.array(chunk_shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)
    return np.prod(chunk_counts)

def load_tiff_stack(folder_or_file):
    """Load TIFF data lazily to avoid memory issues with large files."""
    # Use tifffile's lazy loading instead of dask_image
    if os.path.isfile(folder_or_file):
        # Single TIFF file - use lazy loading
        tiff_store = tifffile.imread(folder_or_file, aszarr=True)
        return da.from_zarr(tiff_store, zarr_format=2)
    else:
        # Multiple TIFF files
        return dask_imread.imread(folder_or_file + "/*.tiff")

def load_nd2_stack(nd2_file):
    """Load ND2 data as a dask array."""
    if not os.path.isfile(nd2_file):
        raise ValueError(f"ND2 file does not exist: {nd2_file}")
    
    # Open file and return dask array (file handle will be managed by dask)
    nd2_handle = nd2.ND2File(nd2_file)
    dask_array = nd2_handle.to_dask()
    # Note: Don't close the handle here as dask needs it for lazy loading
    return dask_array


def estimate_total_chunks_for_tiff(input_path, chunk_shape=(64, 64, 64)):
    """
    Estimate total number of chunks for a TIFF stack,
    given a default chunk shape.
    """
    print(f"Estimating total chunks for TIFF input at {input_path}...")

    if os.path.isfile(input_path):
        # Single TIFF file - get shape lazily
        tiff_store = tifffile.imread(input_path, aszarr=True)
        volume = da.from_zarr(tiff_store, zarr_format=2)
        volume_shape = volume.shape
    else:
        # Multiple TIFF files
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

    # Handle 4D vs 3D chunk shapes
    # Match zarr3_store_spec
    if len(volume_shape) == 4:
        chunk_shape = (1, 64, 64, 64)
    else:
        chunk_shape = (64, 64, 64)
        
    chunk_shape = np.array(chunk_shape)
    chunk_counts = np.ceil(np.array(volume_shape) / chunk_shape).astype(int)
    total_chunks = int(np.prod(chunk_counts))
    
    return total_chunks

def extract_nd2_ome_metadata(nd2_file):
    """Extract OME metadata from ND2 file."""
    if not os.path.isfile(nd2_file):
        raise ValueError(f"ND2 file does not exist: {nd2_file}")
    
    with nd2.ND2File(nd2_file) as f:
        ome_xml = f.ome_metadata().to_xml()
        if ome_xml:
            # Return the raw XML string
            return ome_xml
        else:
            return None

def extract_tiff_ome_metadata(tiff_file):
    """
    Extract metadata from TIFF file (OME-TIFF or ImageJ TIFF).

    Supports:
    - OME-TIFF: Extracts OME-XML metadata
    - ImageJ TIFF: Extracts voxel sizes from ImageJ metadata

    Args:
        tiff_file: Path to TIFF file

    Returns:
        tuple: (ome_xml, voxel_sizes)
            ome_xml: OME-XML string or None
            voxel_sizes: {'x': float, 'y': float, 'z': float} in micrometers, or None
    """
    if not os.path.isfile(tiff_file):
        raise ValueError(f"TIFF file does not exist: {tiff_file}")

    try:
        with tifffile.TiffFile(tiff_file) as tif:
            ome_xml = None
            voxel_sizes = None

            # Try to get OME-XML from TIFF tags (OME-TIFF)
            if tif.ome_metadata:
                ome_xml = tif.ome_metadata
                # For OME-TIFF, voxel sizes will be extracted from OME-XML by converter
                return ome_xml, None

            # Fall back to ImageJ TIFF metadata
            if tif.is_imagej and tif.imagej_metadata:
                imagej_metadata = tif.imagej_metadata

                # Extract Z-spacing and unit
                z_spacing = imagej_metadata.get('spacing')
                unit = imagej_metadata.get('unit', 'micron')

                if z_spacing is not None:
                    # Convert unit to micrometers
                    unit_lower = unit.lower()
                    if unit_lower in ['micron', 'um', 'µm', 'micrometer']:
                        z_spacing_um = float(z_spacing)
                    elif unit_lower in ['nm', 'nanometer']:
                        z_spacing_um = float(z_spacing) / 1000.0
                    elif unit_lower in ['mm', 'millimeter']:
                        z_spacing_um = float(z_spacing) * 1000.0
                    else:
                        print(f"Warning: Unknown unit '{unit}', assuming micrometers")
                        z_spacing_um = float(z_spacing)

                    # Try to extract XY resolution from TIFF tags
                    xy_resolution_um = None
                    try:
                        # Get first page to check resolution tags
                        page = tif.pages[0]
                        if hasattr(page, 'tags'):
                            # XResolution and YResolution are typically in pixels per unit
                            # Resolution units: 1 = None, 2 = inch, 3 = centimeter
                            x_res_tag = page.tags.get('XResolution')
                            y_res_tag = page.tags.get('YResolution')
                            res_unit_tag = page.tags.get('ResolutionUnit')

                            if x_res_tag and y_res_tag and res_unit_tag:
                                # Extract values (usually stored as rational numbers)
                                x_res = x_res_tag.value
                                y_res = y_res_tag.value
                                res_unit = res_unit_tag.value

                                # Convert rational to float if needed
                                if isinstance(x_res, tuple):
                                    x_res = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
                                if isinstance(y_res, tuple):
                                    y_res = y_res[0] / y_res[1] if y_res[1] != 0 else y_res[0]

                                # Convert to micrometers
                                if res_unit == 2:  # inch
                                    # pixels/inch → µm/pixel
                                    x_spacing_um = 25400.0 / x_res if x_res != 0 else None
                                    y_spacing_um = 25400.0 / y_res if y_res != 0 else None
                                elif res_unit == 3:  # centimeter
                                    # pixels/cm → µm/pixel
                                    x_spacing_um = 10000.0 / x_res if x_res != 0 else None
                                    y_spacing_um = 10000.0 / y_res if y_res != 0 else None
                                elif res_unit == 1:  # None/undefined - try to infer
                                    # Heuristic: If value is in range 0.1-100, likely pixels/µm
                                    # If > 100, likely pixels/inch or pixels/mm
                                    if 0.1 <= x_res <= 100 and 0.1 <= y_res <= 100:
                                        # Assume pixels/micrometer
                                        x_spacing_um = 1.0 / x_res if x_res != 0 else None
                                        y_spacing_um = 1.0 / y_res if y_res != 0 else None
                                        print(f"ResolutionUnit undefined, assuming pixels/µm: {x_res:.2f} → {x_spacing_um:.4f} µm")
                                    elif x_res > 100 and y_res > 100:
                                        # Assume pixels/inch (common for scanners)
                                        x_spacing_um = 25400.0 / x_res if x_res != 0 else None
                                        y_spacing_um = 25400.0 / y_res if y_res != 0 else None
                                        print(f"ResolutionUnit undefined, assuming pixels/inch: {x_res:.2f} → {x_spacing_um:.4f} µm")
                                    else:
                                        x_spacing_um = None
                                        y_spacing_um = None
                                else:
                                    # Unknown unit, can't convert
                                    x_spacing_um = None
                                    y_spacing_um = None

                                if x_spacing_um and y_spacing_um:
                                    xy_resolution_um = (x_spacing_um + y_spacing_um) / 2.0  # Average XY
                                    print(f"Extracted XY resolution from TIFF tags: {xy_resolution_um:.4f} µm")
                    except Exception as e:
                        print(f"Note: Could not extract XY resolution from TIFF tags: {e}")

                    # If XY resolution not found, use heuristic
                    if xy_resolution_um is None:
                        # Heuristic: If Z-spacing is very different from typical XY (e.g., > 2x),
                        # use 1.0 µm as a reasonable default for XY
                        if z_spacing_um > 2.0:
                            xy_resolution_um = 1.0
                            print(f"XY resolution not found, using default: {xy_resolution_um} µm")
                        else:
                            # Z is close to typical XY resolution, use same value
                            xy_resolution_um = z_spacing_um
                            print(f"XY resolution not found, using Z-spacing as estimate: {xy_resolution_um} µm")

                    voxel_sizes = {
                        'x': xy_resolution_um,
                        'y': xy_resolution_um,
                        'z': z_spacing_um
                    }
                    print(f"Extracted voxel sizes from ImageJ TIFF: x={xy_resolution_um:.4f}, y={xy_resolution_um:.4f}, z={z_spacing_um:.4f} µm")
                    return None, voxel_sizes
                else:
                    print("Warning: ImageJ TIFF found but no 'spacing' field in metadata")

            # Neither OME nor ImageJ metadata found
            return None, None

    except Exception as e:
        print(f"Warning: Could not extract metadata from TIFF: {e}")
        return None, None

def convert_ome_to_zarr3_metadata(ome_metadata, array_shape, image_name=None):
    """Convert OME metadata to zarr3 OME-ZARR format."""
    if ome_metadata is None:
        # Create minimal metadata if no OME available
        return create_zarr3_ome_metadata(None, array_shape, image_name or "image")
    
    try:
        # Try to parse the XML if it's a string
        if isinstance(ome_metadata, str):
            ome_obj = from_xml(ome_metadata)
        else:
            ome_obj = ome_metadata
            
        # Extract image information
        image = ome_obj.images[0] if ome_obj.images else None
        if image is None:
            return create_zarr3_ome_metadata(ome_metadata, array_shape, image_name or "image")
        
        # Get image name
        final_image_name = image_name or (image.name if image.name else "image")
        
        # Extract pixel sizes
        pixel_sizes = {}
        pixels = image.pixels if image and image.pixels else None
        if pixels:
            if pixels.physical_size_x:
                pixel_sizes['x'] = float(pixels.physical_size_x)
            if pixels.physical_size_y:
                pixel_sizes['y'] = float(pixels.physical_size_y)
            if pixels.physical_size_z:
                pixel_sizes['z'] = float(pixels.physical_size_z)
        pixel_sizes = pixel_sizes if pixel_sizes else None
        
        return create_zarr3_ome_metadata(ome_metadata, array_shape, final_image_name, pixel_sizes)
        
    except Exception as e:
        print(f"Warning: Could not parse OME metadata: {e}")
        return create_zarr3_ome_metadata(ome_metadata, array_shape, image_name or "image")


def create_zarr3_ome_metadata(ome_xml, array_shape, image_name, pixel_sizes=None):
    """Create OME-ZARR metadata structure for zarr3 format."""
    
    # Build axes information based on array shape
    axes = []
    if len(array_shape) == 3:
        # Check if this is 2D multi-channel (CYX) or true 3D volume (ZYX)
        # Heuristic: if first dimension is small (<=10), treat as channels
        if array_shape[0] <= 10:
            # 2D multi-channel: CYX
            axes = [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        else:
            # True 3D volume: ZYX
            axes = [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
    elif len(array_shape) == 4:
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]
    elif len(array_shape) == 5:
        axes = [
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]
    else:
        # Fallback for other dimensions
        axes = [{"name": f"axis_{i}", "type": "space"} for i in range(len(array_shape))]

    # Extract pixel sizes from OME XML if available
    scale_factors = [1.0] * len(array_shape)
    if pixel_sizes is not None:
        # Map pixel sizes to the correct axes
        if len(array_shape) == 3:
            if array_shape[0] <= 10:
                # 2D multi-channel: CYX -> scale = [1.0, y, x]
                scale_factors = [1.0, pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
            else:
                # True 3D: ZYX -> scale = [z, y, x]
                scale_factors = [pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
        elif len(array_shape) == 4:  # CZYX
            scale_factors = [1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
        elif len(array_shape) == 5:  # TCZYX
            scale_factors = [1.0, 1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]

    # Create coordinate transformations for s0 level
    coordinate_transformations = [{
        "type": "scale",
        "scale": scale_factors
    }]
    
    # Build multiscales metadata
    multiscales = [{
        "axes": axes,
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": coordinate_transformations
        }],
        "name": image_name,
        "type": "image"
    }]
    
    # Create full metadata structure
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": multiscales
            }
        }
    }
    
    # Add OME XML if available - ensure it's a string at same level as ome (not inside ome)
    if ome_xml:
        if isinstance(ome_xml, str):
            metadata["attributes"]["ome_xml"] = ome_xml
        else:
            # Convert OME object to string if needed
            metadata["attributes"]["ome_xml"] = str(ome_xml)

    return metadata


def write_zarr3_group_metadata(output_path, metadata):
    """Write zarr3 group-level zarr.json file with OME-ZARR metadata."""
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def update_ome_multiscale_metadata(zarr_path, max_level=4):
    """Update OME-ZARR metadata to include all multiscale levels s0 through max_level.

    Uses downsampling factors from custom metadata if available (precise method),
    otherwise falls back to calculating from dimension ratios (backward compatible).
    """
    import numpy as np

    zarr_json_path = os.path.join(zarr_path, "zarr.json")

    # Read existing metadata
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)

    # Get existing multiscales and s0 scale factors
    multiscales = metadata["attributes"]["ome"]["multiscales"][0]
    s0_scale_factors = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]

    # Check if downsampling factors are available in custom metadata (PREFERRED METHOD)
    downsampling_factors = metadata.get("attributes", {}).get("custom", {}).get("downsampling_factors", None)

    if downsampling_factors:
        print("✓ Using downsampling factors from custom metadata (precise method)")
        use_factors = True
        previous_scale = s0_scale_factors
    else:
        print("⚠ No downsampling factors found - using dimension ratio method (may have rounding imprecision)")
        use_factors = False
        # Read s0 shape for fallback method
        s0_metadata_path = os.path.join(zarr_path, "s0", "zarr.json")
        with open(s0_metadata_path, 'r') as f:
            s0_meta = json.load(f)
        s0_shape = s0_meta.get('shape')

    # Build datasets for all levels with multiscale paths
    datasets = []

    for level in range(max_level + 1):
        level_metadata_path = os.path.join(zarr_path, f"s{level}", "zarr.json")

        if not os.path.exists(level_metadata_path):
            print(f"Warning: s{level}/zarr.json not found, skipping level {level}")
            break

        if level == 0:
            # s0 uses original scale factors
            current_scale = s0_scale_factors
        else:
            if use_factors:
                # METHOD 1 (PREFERRED): Use exact downsampling factors from metadata
                level_factors = downsampling_factors.get(f"s{level}")
                if not level_factors:
                    print(f"Warning: No factors for s{level}, falling back to ratio method for remaining levels")
                    use_factors = False
                else:
                    # Multiply parent voxel size by downsampling factors
                    # Example: s2 [0.04, 0.04, 0.05] * factors [2,2,2] = s3 [0.08, 0.08, 0.1]
                    current_scale = [previous_scale[i] * level_factors[i] for i in range(len(previous_scale))]
                    voxel_nm = [s * 1000 for s in current_scale]
                    print(f"  s{level}: factors={level_factors} → voxel_size={voxel_nm} nm")

            if not use_factors:
                # METHOD 2 (FALLBACK): Calculate from dimension ratios (backward compatible)
                with open(level_metadata_path, 'r') as f:
                    level_meta = json.load(f)
                level_shape = level_meta.get('shape')

                # Calculate cumulative factor from s0 to current level
                factors = [s0_shape[i] / level_shape[i] for i in range(len(s0_shape))]
                current_scale = [s0_scale_factors[i] * factors[i] for i in range(len(s0_scale_factors))]
                voxel_nm = [s * 1000 for s in current_scale]
                print(f"  s{level}: ratio_factors={[round(f, 2) for f in factors]} → voxel_size={voxel_nm} nm")

        datasets.append({
            "path": f"s{level}",  # Relative path at root level
            "coordinateTransformations": [{
                "type": "scale",
                "scale": current_scale
            }]
        })

        # Update previous_scale for next iteration (used by factor method)
        if use_factors:
            previous_scale = current_scale

    # Update multiscales datasets
    multiscales["datasets"] = datasets

    # Write back updated metadata
    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Updated OME metadata for {zarr_path} with levels s0-s{max_level}")

def update_ome_multiscale_metadata_zarr2(zarr_path, max_level=4):
    """Update OME-ZARR metadata for zarr2 format (.zattrs files) to include all multiscale levels."""
    zattrs_path = os.path.join(zarr_path, ".zattrs")
    
    if not os.path.exists(zattrs_path):
        print(f"Warning: .zattrs file not found at {zattrs_path}")
        return
    
    # Read existing metadata
    with open(zattrs_path, 'r') as f:
        metadata = json.load(f)
    
    # Get existing multiscales and s0 scale factors
    if "multiscales" not in metadata:
        print("Warning: No multiscales metadata found in .zattrs")
        return
        
    multiscales = metadata["multiscales"][0]
    s0_scale_factors = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]
    
    # Build datasets for all levels with multiscale paths
    datasets = []
    for level in range(max_level + 1):
        scale_factor = 2 ** level  # 1, 2, 4, 8, 16 for levels 0-4
        current_scale = [sf * scale_factor for sf in s0_scale_factors]
        
        datasets.append({
            "path": f"s{level}",  # Relative path at root level
            "coordinateTransformations": [{
                "type": "scale",
                "scale": current_scale
            }]
        })

    # Update multiscales datasets
    multiscales["datasets"] = datasets

    # Write back updated metadata
    with open(zattrs_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Ensure .zgroup file exists (create if missing)
    zgroup_path = os.path.join(zarr_path, '.zgroup')
    if not os.path.exists(zgroup_path):
        zgroup_metadata = {"zarr_format": 2}
        with open(zgroup_path, 'w') as f:
            json.dump(zgroup_metadata, f, indent=4)
        print(f"Created missing .zgroup file at {zgroup_path}")

    print(f"Updated zarr2 OME metadata for {zarr_path} with levels s0-s{max_level}")

def load_ims_stack(ims_file):
    """Load IMS data as a dask array using h5py, trimmed to actual data bounds."""
    if not os.path.isfile(ims_file):
        raise ValueError(f"IMS file does not exist: {ims_file}")

    # Extract metadata first to check for padding
    metadata, _ = extract_ims_metadata(ims_file)
    actual_z_slices = None

    # Get actual Z-dimension from metadata
    if 'Z' in metadata:
        try:
            actual_z_slices = int(metadata['Z'])
            print(f"Metadata indicates {actual_z_slices} actual Z-slices")
        except (ValueError, TypeError):
            print("Warning: Could not parse Z-dimension from metadata")

    # Open IMS file with h5py and navigate to the image data
    h5_file = h5py.File(ims_file, 'r')
    
    # Navigate the IMS structure to find image data
    # Standard structure: DataSet/ResolutionLevel 0/TimePoint 0/Channel X/Data
    dataset_group = h5_file['DataSet']
    resolution_group = dataset_group['ResolutionLevel 0']
    timepoint_group = resolution_group['TimePoint 0']
    
    # Find all channels and combine them
    channel_keys = [key for key in timepoint_group.keys() if key.startswith('Channel')]
    channel_keys.sort()  # Ensure consistent order
    
    if not channel_keys:
        raise ValueError(f"No channels found in IMS file: {ims_file}")
    
    print(f"Found {len(channel_keys)} channels: {channel_keys}")
    
    # Load each channel's data and stack them
    channel_arrays = []
    for channel_key in channel_keys:
        channel_group = timepoint_group[channel_key]
        data_dataset = channel_group['Data']

        print(f"{channel_key} full shape: {data_dataset.shape}, dtype: {data_dataset.dtype}")

        # Check for padding mismatch and trim if necessary
        if actual_z_slices is not None and actual_z_slices < data_dataset.shape[0]:
            print(f"PADDING DETECTED: Trimming Z from {data_dataset.shape[0]} to {actual_z_slices} slices")
            # Create trimmed dask array - only use actual data slices
            trimmed_dataset = data_dataset[:actual_z_slices, :, :]
            channel_array = da.from_array(trimmed_dataset, chunks='auto')
            print(f"{channel_key} trimmed shape: {trimmed_dataset.shape}")
        else:
            # No padding detected or no metadata - use full array
            if actual_z_slices is None:
                print(f"{channel_key}: No metadata Z-dimension found, using full array")
            else:
                print(f"{channel_key}: No padding detected (metadata Z={actual_z_slices} matches array Z={data_dataset.shape[0]})")
            channel_array = da.from_array(data_dataset, chunks='auto')

        channel_arrays.append(channel_array)
    
    # Stack channels along first axis if multiple channels exist
    if len(channel_arrays) > 1:
        # Stack channels to create CZYX format
        stacked_array = da.stack(channel_arrays, axis=0)
        print(f"Stacked array shape (CZYX): {stacked_array.shape}")
    else:
        # Single channel - keep original ZYX format
        stacked_array = channel_arrays[0]
        print(f"Single channel array shape (ZYX): {stacked_array.shape}")
    
    # Return both the dask array and h5py file handle
    # Note: h5py file must remain open for dask lazy loading
    return stacked_array, h5_file

def extract_ims_metadata(ims_file):
    """Extract basic metadata from IMS file."""
    try:
        with h5py.File(ims_file, 'r') as h5_file:
            metadata = {}
            voxel_sizes = None
            
            # Extract metadata from DataSetInfo/Image
            if 'DataSetInfo' in h5_file and 'Image' in h5_file['DataSetInfo']:
                info_group = h5_file['DataSetInfo']['Image']
                # Extract basic attributes and decode byte arrays
                for attr_name in info_group.attrs:
                    attr_value = info_group.attrs[attr_name]
                    # Decode byte arrays to strings if needed
                    if isinstance(attr_value, np.ndarray) and attr_value.dtype.char == 'S':
                        metadata[attr_name] = b''.join(attr_value).decode('utf-8', errors='ignore')
                    else:
                        metadata[attr_name] = attr_value
            
            # Try to extract voxel size from ExtMin/ExtMax and X/Y/Z dimensions
            if all(key in metadata for key in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2', 'X', 'Y', 'Z']):
                try:
                    # Calculate voxel sizes from extent and dimensions
                    x_size = (float(metadata['ExtMax0']) - float(metadata['ExtMin0'])) / float(metadata['X'])
                    y_size = (float(metadata['ExtMax1']) - float(metadata['ExtMin1'])) / float(metadata['Y'])
                    z_size = (float(metadata['ExtMax2']) - float(metadata['ExtMin2'])) / float(metadata['Z'])
                    voxel_sizes = [x_size, y_size, z_size]  # XYZ order
                    print(f"Calculated voxel sizes (XYZ): {voxel_sizes}")
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Could not calculate voxel sizes: {e}")
                    voxel_sizes = None
            
            return metadata, voxel_sizes
    except Exception as e:
        print(f"Warning: Could not extract metadata from {ims_file}: {e}")
        return {}, None

def extract_n5_voxel_sizes(n5_attrs, level=0):
    """
    Extract voxel sizes from N5 attributes.json.

    Args:
        n5_attrs: Parsed N5 attributes.json dictionary
        level: Scale level (0, 1, 2, etc.)

    Returns:
        tuple: (voxel_sizes_um, dataset_name)
            voxel_sizes_um: [x, y, z] in micrometers, or None if not found
            dataset_name: Name from N5 multiscales, or None
    """
    if not n5_attrs or 'multiscales' not in n5_attrs:
        return None, None

    try:
        multiscales = n5_attrs['multiscales'][0]
        dataset_name = multiscales.get('name', None)

        # Find the dataset for this level
        datasets = multiscales.get('datasets', [])
        if level >= len(datasets):
            return None, dataset_name

        transform = datasets[level].get('transform', {})
        voxel_sizes_nm = transform.get('scale', None)

        if voxel_sizes_nm:
            # Convert nm to micrometers
            voxel_sizes_um = [v / 1000.0 for v in voxel_sizes_nm]
            return voxel_sizes_um, dataset_name

    except Exception as e:
        print(f"Warning: Could not extract voxel sizes from N5: {e}")

    return None, None


def convert_ims_to_zarr3_metadata(ims_file, array_shape, voxel_sizes=None):
    """Convert IMS metadata to zarr3 OME-ZARR format."""
    image_name = os.path.splitext(os.path.basename(ims_file))[0]
    
    # Build axes information based on array shape
    axes = []
    scale_factors = [1.0] * len(array_shape)
    
    if len(array_shape) == 3:  # ZYX
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]
        if voxel_sizes is not None and len(voxel_sizes) >= 3:
            # IMS provides XYZ, we need ZYX for zarr
            scale_factors = [voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]]
    elif len(array_shape) == 4:  # CZYX
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]
        if voxel_sizes is not None and len(voxel_sizes) >= 3:
            # IMS provides XYZ, we need CZYX for zarr
            scale_factors = [1.0, voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]]
    
    # Create coordinate transformations
    coordinate_transformations = [{
        "type": "scale",
        "scale": scale_factors
    }]
    
    # Build multiscales metadata
    multiscales = [{
        "axes": axes,
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": coordinate_transformations
        }],
        "name": image_name,
        "type": "image"
    }]
    
    # Create full metadata structure
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": multiscales
            }
        }
    }
    
    return metadata

def auto_detect_max_level(output_path):
    """Auto-detect the maximum level by scanning for s* directories at root level."""
    if not os.path.exists(output_path):
        return None

    pattern = os.path.join(output_path, 's*')
    level_dirs = glob.glob(pattern)
    
    if not level_dirs:
        return None
    
    levels = []
    for level_dir in level_dirs:
        dirname = os.path.basename(level_dir)
        if dirname.startswith('s') and dirname[1:].isdigit():
            levels.append(int(dirname[1:]))
    
    if not levels:
        return None
    
    return max(levels)

def create_zarr2_ome_metadata(ome_xml, array_shape, image_name, pixel_sizes=None):
    """Create OME-ZARR metadata structure for zarr2 format (.zattrs)."""

    # Build axes information based on OME-XML metadata if available
    axes = []

    if ome_xml:
        # Extract dimension information from OME-XML
        try:
            # Parse XML to find DimensionOrder and sizes
            if isinstance(ome_xml, str):
                root = ET.fromstring(ome_xml)
            else:
                root = ome_xml

            # Find Pixels element
            pixels_elem = None
            for elem in root.iter():
                if elem.tag.endswith('Pixels'):
                    pixels_elem = elem
                    break

            if pixels_elem is not None:
                dimension_order = pixels_elem.get('DimensionOrder', '')
                size_x = int(pixels_elem.get('SizeX', 0))
                size_y = int(pixels_elem.get('SizeY', 0))
                size_z = int(pixels_elem.get('SizeZ', 1))
                size_c = int(pixels_elem.get('SizeC', 1))
                size_t = int(pixels_elem.get('SizeT', 1))

                print(f"OME DimensionOrder: {dimension_order}")
                print(f"OME Sizes: X={size_x}, Y={size_y}, Z={size_z}, C={size_c}, T={size_t}")
                print(f"Array shape: {array_shape}")

                # Map dimension order to active dimensions (size > 1)
                # For Python/numpy arrays, OME DimensionOrder should be read backwards
                dim_to_size = {'X': size_x, 'Y': size_y, 'Z': size_z, 'C': size_c, 'T': size_t}
                active_dims = []

                # Read dimension order backwards for Python array layout
                for dim_char in reversed(dimension_order):
                    if dim_to_size.get(dim_char, 1) > 1:
                        active_dims.append(dim_char.lower())

                print(f"Active dimensions (size > 1, in Python array order): {active_dims}")

                # Create axes based on active dimensions
                if len(active_dims) == len(array_shape):
                    for dim_name in active_dims:
                        if dim_name in ['x', 'y', 'z']:
                            axes.append({"name": dim_name, "type": "space", "unit": "micrometer"})
                        elif dim_name == 'c':
                            axes.append({"name": dim_name, "type": "channel"})
                        elif dim_name == 't':
                            axes.append({"name": dim_name, "type": "time", "unit": "millisecond"})
                        else:
                            axes.append({"name": dim_name, "type": "space"})

                    print(f"Created axes from OME metadata: {[ax['name'] for ax in axes]}")
                else:
                    print(f"Warning: Active dimensions {len(active_dims)} don't match array shape {len(array_shape)}, using fallback")
                    axes = []  # Fall back to default logic
            else:
                print("No Pixels element found in OME-XML, using fallback")
                axes = []

        except Exception as e:
            print(f"Warning: Could not parse OME-XML for dimension information: {e}")
            axes = []

    # Fallback to default patterns if OME parsing failed or no OME-XML provided
    if not axes:
        if len(array_shape) == 3:
            # For 3D, check if first dimension is small (likely channels)
            if array_shape[0] <= 10:
                axes = [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ]
            else:
                axes = [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ]
        elif len(array_shape) == 4:
            axes = [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        elif len(array_shape) == 5:
            axes = [
                {"name": "t", "type": "time", "unit": "millisecond"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        else:
            # Fallback for other dimensions
            axes = [{"name": f"axis_{i}", "type": "space"} for i in range(len(array_shape))]

        print(f"Using fallback axes: {[ax['name'] for ax in axes]}")
    
    # Extract pixel sizes from OME XML if available
    scale_factors = [1.0] * len(array_shape)
    if pixel_sizes is not None:
        # Map pixel sizes to the correct axes
        if len(array_shape) == 3:  # ZYX
            scale_factors = [pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
        elif len(array_shape) == 4:  # CZYX
            scale_factors = [1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
        elif len(array_shape) == 5:  # TCZYX
            scale_factors = [1.0, 1.0, pixel_sizes.get('z', 1.0), pixel_sizes.get('y', 1.0), pixel_sizes.get('x', 1.0)]
    
    # Create coordinate transformations for s0 level
    coordinate_transformations = [{
        "type": "scale",
        "scale": scale_factors
    }]
    
    # Build multiscales metadata (zarr2 format)
    multiscales = [{
        "version": "0.4",
        "axes": axes,
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": coordinate_transformations
        }],
        "name": image_name,
        "type": "image"
    }]
    
    # Create zarr2 metadata structure
    metadata = {
        "multiscales": multiscales
    }
    
    # Add OME XML if available - ensure it's a string
    if ome_xml:
        if isinstance(ome_xml, str):
            metadata["ome_xml"] = ome_xml
        else:
            # Convert OME object to string if needed
            metadata["ome_xml"] = str(ome_xml)
    
    return metadata

def write_zarr2_group_metadata(multiscale_path, metadata):
    """Write zarr2 group-level .zattrs and .zgroup files with OME-ZARR metadata."""
    os.makedirs(multiscale_path, exist_ok=True)

    # Write .zattrs file with OME-ZARR metadata
    zattrs_path = os.path.join(multiscale_path, '.zattrs')
    with open(zattrs_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write .zgroup file to define zarr2 group
    zgroup_path = os.path.join(multiscale_path, '.zgroup')
    zgroup_metadata = {"zarr_format": 2}
    with open(zgroup_path, 'w') as f:
        json.dump(zgroup_metadata, f, indent=4)

def update_ome_metadata_if_needed(output_path, use_ome_structure):
    """Update OME-Zarr metadata if OME structure is used and multiscale levels exist.
    
    Supports both zarr2 (.zattrs) and zarr3 (zarr.json) formats.
    """
    if not use_ome_structure:
        return
    
    max_level = auto_detect_max_level(output_path)
    if max_level is None:
        print("No multiscale levels detected - skipping metadata update")
        return
    
    if max_level == 0:
        print("Only s0 level detected - no multiscale metadata update needed")
        return
    
    # Detect zarr format by checking for metadata files at root
    zarr3_metadata = os.path.join(output_path, 'zarr.json')
    zarr2_metadata = os.path.join(output_path, '.zattrs')

    try:
        if os.path.exists(zarr3_metadata):
            print(f"Updating zarr3 OME metadata for {output_path} with levels s0-s{max_level}")
            update_ome_multiscale_metadata(output_path, max_level=max_level)
            print("OME metadata updated successfully!")
        elif os.path.exists(zarr2_metadata):
            print(f"Updating zarr2 OME metadata for {output_path} with levels s0-s{max_level}")
            update_ome_multiscale_metadata_zarr2(output_path, max_level=max_level)
            print("OME metadata updated successfully!")
        else:
            print("Warning: No zarr metadata file found (.zattrs or zarr.json) - skipping metadata update")
    except Exception as e:
        print(f"Warning: Failed to update OME metadata: {e}")


# =============================================================================
# Dual Zarr v2/v3 Metadata Functions
# =============================================================================

def convert_dtype_v3_to_v2(v3_dtype):
    """Convert zarr v3 data type to zarr v2 format."""
    dtype_mapping = {
        'uint8': '|u1',
        'uint16': '<u2',
        'uint32': '<u4',
        'uint64': '<u8',
        'int8': '|i1',
        'int16': '<i2',
        'int32': '<i4',
        'int64': '<i8',
        'float32': '<f4',
        'float64': '<f8'
    }
    return dtype_mapping.get(v3_dtype, v3_dtype)

def extract_compressor_from_v3_codecs(codecs):
    """Extract compressor configuration from zarr v3 codecs for zarr v2 format."""
    for codec in codecs:
        if codec['name'] == 'zstd':
            return {
                'id': 'zstd',
                'level': codec['configuration'].get('level', 1)
            }
        elif codec['name'] == 'gzip':
            return {
                'id': 'gzip',
                'level': codec['configuration'].get('level', 6)
            }
    # Default to zstd level 1 if no compressor found
    return {'id': 'zstd', 'level': 1}

def create_zarr2_group_metadata():
    """Create .zgroup file content for zarr v2 compatibility."""
    return {"zarr_format": 2}

def create_zarr2_array_metadata(shape, chunk_shape, dtype, codecs):
    """Create .zarray file content for zarr v2 compatibility."""
    return {
        "shape": shape,
        "chunks": chunk_shape,
        "fill_value": 0,
        "dtype": convert_dtype_v3_to_v2(dtype),
        "filters": None,
        "dimension_separator": "/",
        "zarr_format": 2,
        "order": "C",
        "compressor": extract_compressor_from_v3_codecs(codecs)
    }

def convert_zarr3_to_zarr2_ome_metadata_root(zarr3_metadata):
    """Convert zarr v3 OME metadata to zarr v2 .zattrs format for ROOT level (points to multiscale/s0)."""
    if 'attributes' not in zarr3_metadata or 'ome' not in zarr3_metadata['attributes']:
        return None

    ome_data = zarr3_metadata['attributes']['ome']
    multiscales = ome_data.get('multiscales', [])

    if not multiscales:
        return None

    # Convert v3 format to v2 format - ROOT points to multiscale/s0
    v2_multiscales = []
    for ms in multiscales:
        v2_ms = {
            "name": ms.get("name", "/"),
            "version": "0.4",
            "axes": ms.get("axes", []),
            "datasets": []
        }

        # ROOT level: prepend "multiscale/" to paths: s0 -> multiscale/s0
        for dataset in ms.get("datasets", []):
            v2_dataset = dataset.copy()
            v2_dataset["path"] = f"multiscale/{dataset['path']}"
            v2_ms["datasets"].append(v2_dataset)

        # Copy type field (required for OME-ZARR spec)
        if "type" in ms:
            v2_ms["type"] = ms["type"]

        v2_multiscales.append(v2_ms)

    zarr2_attrs = {"multiscales": v2_multiscales}

    # Copy over any additional OME metadata (ome_xml is at same level as ome, not inside ome)
    if "ome_xml" in zarr3_metadata['attributes']:
        zarr2_attrs["ome_xml"] = zarr3_metadata['attributes']["ome_xml"]

    return zarr2_attrs

def convert_zarr3_to_zarr2_ome_metadata_multiscale(zarr3_metadata):
    """Convert zarr v3 OME metadata to zarr v2 .zattrs format for MULTISCALE level (points to s0)."""
    if 'attributes' not in zarr3_metadata or 'ome' not in zarr3_metadata['attributes']:
        return None

    ome_data = zarr3_metadata['attributes']['ome']
    multiscales = ome_data.get('multiscales', [])

    if not multiscales:
        return None

    # Convert v3 format to v2 format - MULTISCALE keeps original paths: s0, s1, etc.
    v2_multiscales = []
    for ms in multiscales:
        v2_ms = {
            "name": ms.get("name", "/"),
            "version": "0.4",
            "axes": ms.get("axes", []),
            "datasets": []
        }

        # MULTISCALE level: keep original paths: s0, s1, etc.
        for dataset in ms.get("datasets", []):
            v2_dataset = dataset.copy()
            # Keep original path (s0, s1, etc.)
            v2_ms["datasets"].append(v2_dataset)

        # Copy type field (required for OME-ZARR spec)
        if "type" in ms:
            v2_ms["type"] = ms["type"]

        v2_multiscales.append(v2_ms)

    zarr2_attrs = {"multiscales": v2_multiscales}

    # Copy over any additional OME metadata (ome_xml is at same level as ome, not inside ome)
    if "ome_xml" in zarr3_metadata['attributes']:
        zarr2_attrs["ome_xml"] = zarr3_metadata['attributes']["ome_xml"]

    return zarr2_attrs

def create_zarr3_root_metadata(zarr3_multiscale_metadata):
    """Create zarr v3 root group metadata that points to multiscale/s0."""
    if 'attributes' not in zarr3_multiscale_metadata or 'ome' not in zarr3_multiscale_metadata['attributes']:
        return None

    ome_data = zarr3_multiscale_metadata['attributes']['ome']
    multiscales = ome_data.get('multiscales', [])

    if not multiscales:
        return None

    # Create root zarr v3 metadata
    root_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": ome_data.get("version", "0.5"),
                "multiscales": []
            }
        }
    }

    # Convert multiscales to point to multiscale/s0 paths
    for ms in multiscales:
        root_ms = ms.copy()
        root_datasets = []

        for dataset in ms.get("datasets", []):
            root_dataset = dataset.copy()
            root_dataset["path"] = f"multiscale/{dataset['path']}"
            root_datasets.append(root_dataset)

        root_ms["datasets"] = root_datasets
        root_metadata["attributes"]["ome"]["multiscales"].append(root_ms)

    # Copy over any additional OME metadata (ome_xml is at same level as ome, not inside ome)
    if "ome_xml" in zarr3_multiscale_metadata['attributes']:
        root_metadata["attributes"]["ome_xml"] = zarr3_multiscale_metadata['attributes']["ome_xml"]

    return root_metadata


def write_dual_zarr_metadata(output_path, source_file):
    """
    Write dual zarr v2/v3 metadata for compatibility.
    Creates zarr v2 entry points at root level.
    """
    # TODO: Update this function for new OME-Zarr structure (no multiscale folder)
    # Currently still references old multiscale/ paths - needs refactoring
    zarr3_group_path = os.path.join(output_path, "zarr.json")

    if not os.path.exists(zarr3_group_path):
        print(f"Warning: zarr v3 metadata not found at {zarr3_group_path}")
        return False

    try:
        # Read zarr v3 group metadata
        with open(zarr3_group_path, 'r') as f:
            zarr3_metadata = json.load(f)

        # 1. Create zarr v3 root group metadata
        root_zarr3_metadata = create_zarr3_root_metadata(zarr3_metadata)
        root_zarr3_path = os.path.join(output_path, "zarr.json")
        with open(root_zarr3_path, 'w') as f:
            json.dump(root_zarr3_metadata, f, indent=2)
        print(f"Created zarr v3 root metadata: {root_zarr3_path}")

        # 2. Create zarr v2 root metadata (points to multiscale/s0)
        root_zgroup_path = os.path.join(output_path, ".zgroup")
        with open(root_zgroup_path, 'w') as f:
            json.dump(create_zarr2_group_metadata(), f, indent=2)
        print(f"Created zarr v2 root group: {root_zgroup_path}")

        root_zarr2_attrs = convert_zarr3_to_zarr2_ome_metadata_root(zarr3_metadata)
        if root_zarr2_attrs:
            root_zattrs_path = os.path.join(output_path, ".zattrs")
            with open(root_zattrs_path, 'w') as f:
                json.dump(root_zarr2_attrs, f, indent=2)
            print(f"Created zarr v2 root OME metadata: {root_zattrs_path}")

        # 3. Create zarr v2 multiscale metadata (points to s0)
        multiscale_zgroup_path = os.path.join(multiscale_path, ".zgroup")
        with open(multiscale_zgroup_path, 'w') as f:
            json.dump(create_zarr2_group_metadata(), f, indent=2)
        print(f"Created zarr v2 multiscale group: {multiscale_zgroup_path}")

        multiscale_zarr2_attrs = convert_zarr3_to_zarr2_ome_metadata_multiscale(zarr3_metadata)
        if multiscale_zarr2_attrs:
            multiscale_zattrs_path = os.path.join(multiscale_path, ".zattrs")
            with open(multiscale_zattrs_path, 'w') as f:
                json.dump(multiscale_zarr2_attrs, f, indent=2)
            print(f"Created zarr v2 multiscale OME metadata: {multiscale_zattrs_path}")

        # 4. Create zarr v2 array metadata for each scale level (auto-detect chunk encoding)
        detected_v3_chunks = False  # Track if we have v3 chunk encoding

        for item in os.listdir(multiscale_path):
            if item.startswith('s') and os.path.isdir(os.path.join(multiscale_path, item)):
                v3_array_path = os.path.join(multiscale_path, item, "zarr.json")
                v3_array_dir = os.path.join(multiscale_path, item)

                if os.path.exists(v3_array_path):
                    # Read v3 array metadata
                    with open(v3_array_path, 'r') as f:
                        v3_array_metadata = json.load(f)

                    # Detect chunk encoding
                    chunk_encoding = v3_array_metadata.get("chunk_key_encoding", {}).get("name", "default")
                    is_v2_encoding = (chunk_encoding == "v2")
                    is_v3_encoding = (chunk_encoding == "default")

                    # Track v3 chunk encoding for metadata path correction
                    if is_v3_encoding:
                        detected_v3_chunks = True

                    # Convert to v2 array metadata
                    zarr2_array = create_zarr2_array_metadata(
                        shape=v3_array_metadata["shape"],
                        chunk_shape=v3_array_metadata["chunk_grid"]["configuration"]["chunk_shape"],
                        dtype=v3_array_metadata["data_type"],
                        codecs=v3_array_metadata["codecs"]
                    )

                    # Place .zarray file based on chunk encoding (Mark's approach)
                    if is_v2_encoding:
                        # Approach 1: v2 chunk encoding - .zarray colocated with zarr.json
                        zarray_path = os.path.join(v3_array_dir, ".zarray")
                        approach_name = "v2 chunk encoding (colocated)"
                    elif is_v3_encoding:
                        # Approach 2: v3 chunk encoding - .zarray inside c/ directory
                        c_dir = os.path.join(v3_array_dir, "c")
                        if os.path.exists(c_dir):
                            zarray_path = os.path.join(c_dir, ".zarray")
                            approach_name = "v3 chunk encoding (c/ directory)"
                        else:
                            print(f"Warning: v3 chunk encoding detected but no c/ directory found at {c_dir}")
                            # Fallback to colocated
                            zarray_path = os.path.join(v3_array_dir, ".zarray")
                            approach_name = "v3 chunk encoding (fallback colocated)"
                    else:
                        print(f"Warning: Unknown chunk encoding '{chunk_encoding}' at {v3_array_path}, using colocated placement")
                        zarray_path = os.path.join(v3_array_dir, ".zarray")
                        approach_name = f"unknown encoding '{chunk_encoding}' (colocated)"

                    # Write .zarray file
                    with open(zarray_path, 'w') as f:
                        json.dump(zarr2_array, f, indent=2)
                    print(f"Created zarr v2 array metadata ({approach_name}): {zarray_path}")

        # 5. Fix zarr v2 entry point paths for v3 chunk encoding (Mark's approach)
        if detected_v3_chunks:
            print("Detected v3 chunk encoding - fixing zarr v2 entry point paths...")

            # Fix root .zattrs to point to multiscale/s0/c
            root_zattrs_path = os.path.join(output_path, ".zattrs")
            if os.path.exists(root_zattrs_path):
                with open(root_zattrs_path, 'r') as f:
                    root_attrs = json.load(f)

                # Update paths in multiscales datasets
                for ms in root_attrs.get("multiscales", []):
                    for dataset in ms.get("datasets", []):
                        if dataset["path"].endswith("/s0"):
                            dataset["path"] = dataset["path"] + "/c"

                with open(root_zattrs_path, 'w') as f:
                    json.dump(root_attrs, f, indent=2)
                print(f"Updated root zarr v2 paths for v3 chunks: {root_zattrs_path}")

            # Fix multiscale .zattrs to point to s0/c
            multiscale_zattrs_path = os.path.join(multiscale_path, ".zattrs")
            if os.path.exists(multiscale_zattrs_path):
                with open(multiscale_zattrs_path, 'r') as f:
                    multiscale_attrs = json.load(f)

                # Update paths in multiscales datasets
                for ms in multiscale_attrs.get("multiscales", []):
                    for dataset in ms.get("datasets", []):
                        if dataset["path"] == "s0":
                            dataset["path"] = "s0/c"

                with open(multiscale_zattrs_path, 'w') as f:
                    json.dump(multiscale_attrs, f, indent=2)
                print(f"Updated multiscale zarr v2 paths for v3 chunks: {multiscale_zattrs_path}")

        print(f"Created dual metadata for {output_path}")
        print("Zarr v2 entry points:")
        if detected_v3_chunks:
            print(f"  - Root: {output_path}|zarr2: (points to multiscale/s0/c)")
            print(f"  - Multiscale: {output_path}/multiscale|zarr2: (points to s0/c)")
        else:
            print(f"  - Root: {output_path}|zarr2: (points to multiscale/s0)")
            print(f"  - Multiscale: {output_path}/multiscale|zarr2: (points to s0)")
        return True

    except Exception as e:
        print(f"Error creating dual zarr metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Neuroglancer Precomputed to N5 Conversion Functions
# ============================================================================

def precomputed_store_spec(precomputed_path, scale_key):
    """
    Create TensorStore spec for Neuroglancer Precomputed format.

    Args:
        precomputed_path: Base path to precomputed dataset (local, HTTP, GCS, S3)
        scale_key: Scale identifier from info.json (e.g., "10.0x10.0x25.0")

    Returns:
        dict: TensorStore spec for neuroglancer_precomputed driver

    Examples:
        >>> spec = precomputed_store_spec("gs://bucket/dataset", "10.0x10.0x25.0")
        >>> store = ts.open(spec).result()

        >>> spec = precomputed_store_spec("/local/path/data", "20.0x20.0x25.0")
        >>> store = ts.open(spec).result()
    """
    return {
        'driver': 'neuroglancer_precomputed',
        'kvstore': get_kvstore_spec(precomputed_path),
        'scale_metadata': {'key': scale_key},
        'open': True
    }


def n5_output_spec_with_compression(n5_path, shape, dtype,
                                     chunk_shape=None,
                                     compression_level=3,
                                     downsample_factors=None,
                                     pixel_resolution=None):
    """
    Create N5 output specification with zstd compression.

    Args:
        n5_path: Output N5 path (e.g., /path/to/dataset.n5/ch0tp0/s0)
        shape: Array dimensions [X, Y, Z]
        dtype: Data type string ('uint8', 'uint16', 'uint64', etc.)
        chunk_shape: Block size (default [128, 128, 128])
        compression_level: zstd compression level (default 3)
        downsample_factors: Downsampling relative to s0 (e.g., [2,2,1] for s1)
                           None for s0 (full resolution)
        pixel_resolution: Physical resolution in nm (e.g., [10.0, 10.0, 25.0])
                         Extracted from Precomputed info['scales'][i]['resolution']
                         None to skip adding pixelResolution

    Returns:
        dict: TensorStore N5 spec with proper attributes.json format

    Examples:
        >>> spec = n5_output_spec_with_compression(
        ...     "/path/to/data.n5/ch0tp0/s0",
        ...     [47000, 13000, 1000],
        ...     'uint8',
        ...     pixel_resolution=[10.0, 10.0, 25.0]
        ... )

        >>> spec = n5_output_spec_with_compression(
        ...     "/path/to/data.n5/ch0tp0/s1",
        ...     [23500, 6500, 1000],
        ...     'uint8',
        ...     downsample_factors=[2, 2, 1],
        ...     pixel_resolution=[20.0, 20.0, 25.0]
        ... )
    """
    if chunk_shape is None:
        chunk_shape = [128, 128, 128]

    # Map numpy/tensorstore dtype to N5 data type string
    dtype_map = {
        'uint8': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
        'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64',
        'float32': 'float32', 'float64': 'float64'
    }

    n5_dtype = dtype_map.get(str(dtype), str(dtype))

    metadata = {
        'dimensions': list(shape),
        'dataType': n5_dtype,
        'blockSize': list(chunk_shape),
        'compression': {'type': 'zstd', 'level': compression_level}
    }

    # Add downsamplingFactors for all scales (s0: [1,1,1], s1: [2,2,1], etc.)
    if downsample_factors is not None:
        metadata['downsamplingFactors'] = list(downsample_factors)

    # Add pixelResolution from Precomputed info for n5-viewer compatibility
    if pixel_resolution is not None:
        metadata['pixelResolution'] = {
            'unit': 'nm',
            'dimensions': list(pixel_resolution)
        }

    return {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': n5_path},
        'metadata': metadata,
        'create': True,
        'delete_existing': False
    }


def fetch_precomputed_info(precomputed_path):
    """
    Fetch and parse Neuroglancer Precomputed info JSON.
    Supports GCS (gs://), HTTP/HTTPS, and local file paths.

    Args:
        precomputed_path: Base path to precomputed dataset

    Returns:
        dict: Parsed info metadata containing scales, data_type, etc.

    Examples:
        >>> info = fetch_precomputed_info("gs://liconn-public/ExPID124/image")
        >>> print(info['scales'][0]['resolution'])  # [10.0, 10.0, 25.0]

        >>> info = fetch_precomputed_info("/local/path/to/data")
        >>> print(info['data_type'])  # 'uint8'
    """
    if precomputed_path.startswith('gs://') or precomputed_path.startswith('s3://'):
        # Use TensorStore KvStore for GCS/S3
        kvstore_spec = get_kvstore_spec(precomputed_path)
        kvstore = ts.KvStore.open(kvstore_spec).result()
        info_data = kvstore.read('info').result().value
        return json.loads(info_data)
    elif precomputed_path.startswith('http'):
        # HTTP fetch
        info_url = precomputed_path.rstrip('/') + '/info'
        response = requests.get(info_url)
        response.raise_for_status()
        return response.json()
    else:
        # Local file (may be gzipped)
        info_path = os.path.join(precomputed_path, 'info')

        # Try gzip first, then plain text
        try:
            import gzip
            with gzip.open(info_path, 'rt') as f:
                return json.load(f)
        except:
            with open(info_path, 'r') as f:
                return json.load(f)


def ensure_precomputed_info_decompressed(precomputed_path):
    """
    Ensure the Neuroglancer Precomputed info file is decompressed.
    TensorStore requires uncompressed JSON for local file paths.

    Args:
        precomputed_path: Base path to precomputed dataset (local path only)

    Returns:
        None (decompresses info file in-place if needed)
    """
    if precomputed_path.startswith('gs://') or precomputed_path.startswith('s3://') or precomputed_path.startswith('http'):
        # Remote paths don't need decompression
        return

    info_path = os.path.join(precomputed_path, 'info')

    # Check if file is gzipped
    try:
        import gzip
        with gzip.open(info_path, 'rt') as f:
            info_data = f.read()

        # File is gzipped - decompress it
        print(f"  Decompressing gzipped info file: {info_path}")
        with open(info_path, 'w') as f:
            f.write(info_data)
        print(f"  ✓ Info file decompressed")
    except:
        # File is already plain text or doesn't exist
        pass


def calculate_downsample_factors(scale_idx, precomputed_info):
    """
    Calculate N5-style downsampling factors for a given scale.

    Args:
        scale_idx: Index of scale in info.json (0=highest resolution)
        precomputed_info: Parsed info.json dictionary

    Returns:
        list: Downsampling factors [X, Y, Z] relative to s0
              Returns [1, 1, 1] for s0 (full resolution)

    Examples:
        For ExPID124 image with resolutions:
        - s0: [10, 10, 25] nm → [1, 1, 1] (full resolution)
        - s1: [20, 20, 25] nm → [2, 2, 1]
        - s2: [40, 40, 50] nm → [4, 4, 2]
        - s3: [80, 80, 100] nm → [8, 8, 4]
    """
    if scale_idx == 0:
        return [1, 1, 1]  # Full resolution

    scales = precomputed_info['scales']
    s0_resolution = np.array(scales[0]['resolution'])
    current_resolution = np.array(scales[scale_idx]['resolution'])

    # Calculate integer downsampling factors
    factors = [
        int(round(current_resolution[i] / s0_resolution[i]))
        for i in range(3)
    ]
    return factors


def update_n5_root_attributes(n5_root_path, precomputed_info):
    """
    Create or update root attributes.json with complete metadata including pixelResolution.

    If root attributes.json doesn't exist, creates it from scratch by scanning existing
    scales. If it exists, updates it with pixelResolution. pixelResolution is always
    included in the initial creation, not added later.

    Args:
        n5_root_path: Path to N5 dataset root (e.g., /path/to/dataset.n5)
        precomputed_info: Parsed Precomputed info.json dictionary

    Returns:
        bool: True if successful, False otherwise

    Structure created:
        {
          "n5": "4.0.0",
          "Bigstitcher-Spark": {
            "MultiResolutionInfos": [[...scales...]],
            "pixelResolution": [
              [  // Channel 0
                {"unit": "nm", "dimensions": [10.0, 10.0, 25.0]},    // s0
                {"unit": "nm", "dimensions": [20.0, 20.0, 25.0]},    // s1
                ...
              ]
            ],
            ...other metadata...
          }
        }

    Examples:
        >>> info = fetch_precomputed_info("/path/to/precomputed/image")
        >>> update_n5_root_attributes("/path/to/dataset.n5", info)
        True
    """
    from pathlib import Path
    import glob

    root_attrs_path = Path(n5_root_path) / "attributes.json"
    os.umask(0o0002)

    try:
        # Check if root attributes.json exists
        if root_attrs_path.exists():
            # EXISTING FILE: Load and update
            print(f"  → Root attributes.json exists, updating...")
            with open(root_attrs_path, 'r') as f:
                attrs = json.load(f)

            if "Bigstitcher-Spark" not in attrs:
                print(f"  ✗ Warning: No Bigstitcher-Spark section in root attributes.json")
                return False

            bigstitcher = attrs["Bigstitcher-Spark"]

            # Get MultiResolutionInfos
            if "MultiResolutionInfos" not in bigstitcher:
                print(f"  ✗ Warning: No MultiResolutionInfos found in root attributes.json")
                return False

            multi_res_infos = bigstitcher["MultiResolutionInfos"]

        else:
            # NEW FILE: Create from scratch by scanning N5 structure
            print(f"  → Root attributes.json not found, creating from scratch...")

            # Scan for channel groups (ch0tp0, ch1tp0, etc.)
            channel_dirs = sorted(glob.glob(os.path.join(n5_root_path, "ch*tp*")))
            if not channel_dirs:
                print(f"  ✗ Error: No channel directories (ch*tp*) found in {n5_root_path}")
                return False

            # Build MultiResolutionInfos by scanning scales
            multi_res_infos = []
            first_scale_attrs = None

            for ch_dir in channel_dirs:
                ch_name = os.path.basename(ch_dir)
                scale_dirs = sorted(glob.glob(os.path.join(ch_dir, "s*")))

                channel_scales = []
                for scale_dir in scale_dirs:
                    scale_name = os.path.basename(scale_dir)
                    scale_attrs_path = os.path.join(scale_dir, "attributes.json")

                    if not os.path.exists(scale_attrs_path):
                        continue

                    with open(scale_attrs_path, 'r') as f:
                        scale_attrs = json.load(f)

                    # Save first scale for metadata reference
                    if first_scale_attrs is None:
                        first_scale_attrs = scale_attrs

                    # Extract scale index from name (s0 → 0, s1 → 1, etc.)
                    scale_idx = int(scale_name[1:])

                    # Get downsampling factors
                    downsample = scale_attrs.get('downsamplingFactors', [1, 1, 1])

                    # Calculate relative downsampling (always [1,1,1] for s0, then incremental)
                    if scale_idx == 0:
                        relative_downsample = [1, 1, 1]
                        absolute_downsample = [1, 1, 1]
                    else:
                        relative_downsample = downsample
                        absolute_downsample = downsample

                    channel_scales.append({
                        "relativeDownsampling": relative_downsample,
                        "absoluteDownsampling": absolute_downsample,
                        "blockSize": scale_attrs['blockSize'],
                        "dimensions": scale_attrs['dimensions'],
                        "dataset": f"{ch_name}/{scale_name}",
                        "dataType": scale_attrs['dataType']
                    })

                multi_res_infos.append(channel_scales)

            # Create root attributes from scratch
            if first_scale_attrs is None:
                print(f"  ✗ Error: No scale attributes.json found")
                return False

            # Calculate bounding box from s0 dimensions
            s0_dims = multi_res_infos[0][0]['dimensions']

            # Calculate anisotropy factor from precomputed resolution
            s0_resolution = precomputed_info['scales'][0]['resolution']
            # AnisotropyFactor = Z resolution / XY resolution
            anisotropy = s0_resolution[2] / s0_resolution[0]

            attrs = {
                "n5": "4.0.0",
                "Bigstitcher-Spark": {
                    "InputXML": f"file:{n5_root_path}",
                    "NumTimepoints": 1,
                    "NumChannels": len(multi_res_infos),
                    "Boundingbox_min": [0, 0, 0],
                    "Boundingbox_max": s0_dims,
                    "PreserveAnisotropy": True,
                    "AnisotropyFactor": round(anisotropy, 1),
                    "DataType": first_scale_attrs['dataType'],
                    "BlockSize": first_scale_attrs['blockSize'],
                    "MinIntensity": 0.0,
                    "MaxIntensity": 255.0 if first_scale_attrs['dataType'] == 'uint8' else 65535.0,
                    "FusionFormat": "N5",
                    "MultiResolutionInfos": multi_res_infos
                }
            }

            bigstitcher = attrs["Bigstitcher-Spark"]
            print(f"  ✓ Created new root attributes.json structure")
            print(f"    Channels: {len(multi_res_infos)}")
            print(f"    Scales per channel: {len(multi_res_infos[0]) if multi_res_infos else 0}")
            print(f"    Anisotropy factor: {anisotropy:.1f}")

        # Now add pixelResolution to Bigstitcher-Spark
        # Build pixelResolution array mirroring MultiResolutionInfos structure
        scales = precomputed_info['scales']
        pixel_resolution = []

        for ch_idx, channel_scales in enumerate(multi_res_infos):
            channel_resolutions = []

            for scale_entry in channel_scales:
                # Extract actual scale index from dataset path (e.g., "ch0tp0/s8" → 8)
                dataset_path = scale_entry.get('dataset', '')
                if '/s' in dataset_path:
                    scale_idx = int(dataset_path.split('/s')[-1])
                else:
                    print(f"  ✗ Warning: Could not parse scale index from dataset: {dataset_path}")
                    continue

                if scale_idx >= len(scales):
                    print(f"  ✗ Warning: Scale s{scale_idx} exceeds Precomputed scales")
                    continue

                scale_resolution = scales[scale_idx]['resolution']
                channel_resolutions.append({
                    "unit": "nm",
                    "dimensions": list(scale_resolution)
                })

            pixel_resolution.append(channel_resolutions)

        # Add pixelResolution to Bigstitcher-Spark (created with other metadata, not updated later!)
        bigstitcher["pixelResolution"] = pixel_resolution

        # Save complete attributes with pixelResolution included
        with open(root_attrs_path, 'w') as f:
            json.dump(attrs, f, indent=2)

        print(f"  ✓ Root attributes.json saved with pixelResolution included")
        print(f"    Structure: {len(multi_res_infos)} channel(s) × {len(multi_res_infos[0]) if multi_res_infos else 0} scale(s)")

        return True

    except Exception as e:
        print(f"  ✗ Error creating/updating root attributes: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Anisotropy-Aware Multiscale Generation
# =============================================================================
# Based on the algorithm developed by Yurii Zubov (Janelia CellMap Team)
# Original implementation: https://github.com/janelia-cellmap/zarrify
# Reference: https://github.com/janelia-cellmap/zarrify/blob/main/src/zarrify/utils/volume.py#L91
#
# Credit: Yurii Zubov for the anisotropic downsampling algorithm that makes
# voxel sizes more homogeneous with each iteration.
# =============================================================================

def calculate_anisotropic_downsample_factors(voxel_sizes, axes_names, min_ratio=0.5, max_ratio=2.0, use_anisotropic=True):
    """
    Calculate adaptive downsampling factors based on voxel size aspect ratios.

    Algorithm credit: Yurii Zubov (Janelia CellMap Team)
    Based on: https://github.com/janelia-cellmap/zarrify/blob/main/src/zarrify/utils/volume.py#L91

    Goal: Make voxel sizes more homogeneous with each downsampling iteration.

    Example:
        s0: voxel_sizes = (100, 20, 20) nm for (z, y, x)
        → z is 5× larger than y/x
        → factor = (1, 2, 2) to get s1: (100, 40, 40) nm

        s1: voxel_sizes = (100, 40, 40) nm
        → z is 2.5× larger
        → factor = (1, 2, 2) to get s2: (100, 80, 80) nm

        s2: voxel_sizes = (100, 80, 80) nm
        → z is 1.25× larger (< 2.0 threshold)
        → factor = (2, 2, 2) to get s3: (200, 160, 160) nm

    Args:
        voxel_sizes: List of voxel sizes (e.g., [1.0, 0.116, 0.116] for c, y, x)
        axes_names: List of axis names (e.g., ['c', 'y', 'x'])
        min_ratio: Minimum ratio threshold (default 0.5)
        max_ratio: Maximum ratio threshold (default 2.0)
        use_anisotropic: If False, always use (1, 2, 2, ...) pattern

    Returns:
        List of downsampling factors (e.g., [1, 2, 2] for c, y, x)
    """
    if not use_anisotropic:
        # Simple mode: preserve channels/time, downsample spatial by 2
        return [1 if axis in ['c', 't'] else 2 for axis in axes_names]

    # Get spatial dimensions (skip channel and time)
    spatial_data = [(axis, voxel_sizes[i]) for i, axis in enumerate(axes_names) if axis not in ['c', 't']]

    if not spatial_data:
        return [1] * len(axes_names)

    axes, dimensions = zip(*spatial_data)

    if len(dimensions) == 1:
        # Only one spatial dimension
        factors = [2]
    else:
        # Calculate aspect ratios
        ratios = []
        for i, dim in enumerate(dimensions):
            # Calculate ratios of current dimension to all others
            # Example for 3D: [(z/y, z/x), (y/z, y/x), (x/z, x/y)]
            dim_ratios = tuple(dim / dimensions[j] for j in range(len(dimensions)) if j != i)
            ratios.append(dim_ratios)

        # Determine downsampling factors for each spatial dimension
        factors = []
        for (i, dim_ratios) in enumerate(ratios):
            # If this dimension is >= 2× larger than all others
            if all(ratio >= max_ratio for ratio in dim_ratios):
                # This dimension is large, keep it at 1, downsample others by 2
                factors = [2] * len(ratios)
                factors[i] = 1
                break
            # If this dimension is <= 0.5× smaller than all others
            elif all(ratio <= min_ratio for ratio in dim_ratios):
                # This dimension is small, downsample it by 2, keep others at 1
                factors = [1] * len(ratios)
                factors[i] = 2
                break
            else:
                # Within acceptable range, downsample by 2
                factors.append(2)

    # Map spatial factors back to all axes
    spatial_factors = {k: v for k, v in zip(axes, factors)}
    return [1 if axis in ['c', 't'] else spatial_factors[axis] for axis in axes_names]


def calculate_num_multiscale_levels(shape, axes_names, voxel_sizes, chunk_shape=None, dtype_size=2,
                                     min_array_nbytes=None, min_array_shape=None, shard_shape=None, use_anisotropic=True):
    """
    Calculate how many multiscale levels to generate.

    Stops when either:
    - Rule 1: array_nbytes < min_array_nbytes (total volume too small)
    - Rule 2: all(shape[i] < min_array_shape[i]) (all dimensions below threshold)
    - Rule 3: cumulative magnification drifts >4% from power-of-2 (WebKnossos compatibility)

    Args:
        shape: Array shape (e.g., [3, 1196, 31416, 17635] for C, Z, Y, X)
        axes_names: List of axis names (e.g., ['c', 'z', 'y', 'x'])
        voxel_sizes: Initial voxel sizes (e.g., [1.0, 1.0, 0.116, 0.116])
        chunk_shape: Chunk shape for calculating defaults (e.g., [1, 64, 128, 128])
        dtype_size: Bytes per element (default 2 for uint16)
        min_array_nbytes: Stop when array size < this (default: chunk_nbytes)
        min_array_shape: Stop when all dims < this (default: chunk_shape)
        shard_shape: (Deprecated, not used)
        use_anisotropic: Use anisotropic downsampling factors

    Returns:
        int: Total number of levels (including s0)
    """
    current_shape = list(shape)
    current_voxel_sizes = list(voxel_sizes)

    # Calculate defaults from chunk_shape
    if chunk_shape is not None:
        if min_array_nbytes is None:
            chunk_nbytes = 1
            for dim in chunk_shape:
                chunk_nbytes *= dim
            chunk_nbytes *= dtype_size
            min_array_nbytes = chunk_nbytes

        if min_array_shape is None:
            min_array_shape = list(chunk_shape)

    # Fallbacks if no chunk_shape provided
    if min_array_nbytes is None:
        min_array_nbytes = 524288  # 512KB default
    if min_array_shape is None:
        min_array_shape = [32] * len(shape)  # Default 32 per dimension

    level = 0
    while True:
        level += 1

        # Calculate downsampling factors for next level
        factors = calculate_anisotropic_downsample_factors(
            current_voxel_sizes,
            axes_names,
            use_anisotropic=use_anisotropic
        )

        # Apply downsampling to get next level shape
        new_shape = [max(1, dim // factor) for dim, factor in zip(current_shape, factors)]
        new_voxel_sizes = [voxel * factor for voxel, factor in zip(current_voxel_sizes, factors)]

        # Rule 1: Check array_nbytes
        array_nbytes = dtype_size
        for dim in new_shape:
            array_nbytes *= dim

        if array_nbytes < min_array_nbytes:
            return level - 1  # Stop: total volume too small (don't create this level)

        # Rule 2: Check if ALL dimensions below threshold
        if all(new_shape[i] < min_array_shape[i] for i in range(len(new_shape))):
            return level - 1  # Stop: all dimensions small (don't create this level)

        # Rule 3: WebKnossos power-of-2 validation (November 5, 2025)
        # WebKnossos requires cumulative magnifications to be powers of 2 within ~4% tolerance
        import math
        max_power_of_2_error = 0.0
        for i in range(len(shape)):
            # Skip non-spatial dimensions
            if axes_names[i] in ['c', 't']:
                continue

            cumulative_mag = shape[i] / new_shape[i]

            # Find nearest power of 2 using log2
            # e.g., log2(15) = 3.906 → round = 4 → expected_mag = 16
            if cumulative_mag > 0:
                log2_mag = math.log2(cumulative_mag)
                expected_power = round(log2_mag)  # Round to nearest integer power
                expected_mag = 2 ** expected_power

                # Calculate error percentage
                error = abs(cumulative_mag - expected_mag) / expected_mag
                max_power_of_2_error = max(max_power_of_2_error, error)

        # Stop if cumulative magnification drifts >4% from nearest power of 2
        # This prevents WebKnossos upload failures like "invalid mag: (16, 16, 15)"
        if max_power_of_2_error > 0.04:
            print(f"Stopping at level {level-1}: Cumulative mag error {max_power_of_2_error*100:.1f}% exceeds WebKnossos 4% tolerance")
            return level - 1

        current_shape = new_shape
        current_voxel_sizes = new_voxel_sizes

        # Safety check: don't generate too many levels
        if level > 20:
            print(f"Warning: Stopped at level {level} (safety limit)")
            return level

    return level


def calculate_pyramid_plan(s0_path, min_array_nbytes=None, min_array_shape=None):
    """
    Pre-calculate entire multiscale pyramid plan before submitting cluster jobs.

    This function mathematically predicts voxel sizes for all levels WITHOUT creating data.
    Enables cluster submission with pre-determined anisotropic downsampling factors.

    Args:
        s0_path: Path to s0 level (either zarr3 s0/ or zarr2 multiscale/s0/)
        min_array_nbytes: Stop when array size < this (default: chunk_nbytes from metadata)
        min_array_shape: Stop when all dims < this (default: chunk_shape from metadata)

    Returns:
        dict with keys:
            'format': 'zarr3' or 'zarr2'
            'shape': s0 shape
            'voxel_sizes': s0 voxel sizes
            'axes_names': dimension names
            'chunk_shape': chunk shape
            'dtype_size': bytes per element
            'num_levels': total number of levels needed
            'pyramid_plan': list of dicts for each level:
                [
                    {"level": 1, "factor": [1,2,2], "predicted_voxel_sizes": [100,40,40]},
                    {"level": 2, "factor": [1,2,2], "predicted_voxel_sizes": [100,80,80]},
                    ...
                ]

    Example:
        >>> plan = calculate_pyramid_plan("/path/to/dataset.zarr/s0")
        >>> print(f"Need {plan['num_levels']} levels")
        >>> for level_plan in plan['pyramid_plan']:
        ...     print(f"Level {level_plan['level']}: factor {level_plan['factor']}")

    Credit: Based on Yurii Zubov's anisotropic downsampling algorithm (Janelia CellMap Team)
    Reference: https://github.com/janelia-cellmap/zarrify
    """
    import json
    import numpy as np

    # Detect format: zarr3 (has zarr.json) or zarr2 (has .zarray)
    zarr3_metadata_path = os.path.join(s0_path, "zarr.json")
    zarr2_metadata_path = os.path.join(s0_path, ".zarray")

    is_zarr3 = os.path.exists(zarr3_metadata_path)
    is_zarr2 = os.path.exists(zarr2_metadata_path)

    if not is_zarr3 and not is_zarr2:
        raise FileNotFoundError(
            f"Cannot find zarr metadata at {s0_path}\n"
            f"Expected either:\n"
            f"  - Zarr3: {zarr3_metadata_path}\n"
            f"  - Zarr2: {zarr2_metadata_path}"
        )

    format_type = "zarr3" if is_zarr3 else "zarr2"

    # Read metadata based on format
    chunk_shape = None
    dtype_size = 2  # default uint16

    if is_zarr3:
        # Read s0 shape, dimension names, chunk/shard configuration, dtype
        with open(zarr3_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')
        axes_names = metadata.get('dimension_names')

        # Read shard and chunk shapes
        # shard_shape = outer chunk grid (what we pass as --custom_shard_shape)
        # inner_chunk_shape = inner chunks within shard (what we pass as --custom_chunk_shape)
        shard_shape = metadata.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape')
        inner_chunk_shape = None

        # Check if using sharding codec
        codecs = metadata.get('codecs', [])
        if codecs and codecs[0].get('name') == 'sharding_indexed':
            inner_chunk_shape = codecs[0].get('configuration', {}).get('chunk_shape')

        # Use inner_chunk_shape if available (for sharded zarr3), otherwise use shard_shape
        # This ensures stopping criteria are based on actual chunk size, not shard size
        chunk_shape = inner_chunk_shape if inner_chunk_shape is not None else shard_shape

        # Get dtype size
        dtype_str = metadata.get('data_type', 'uint16')
        dtype = np.dtype(dtype_str)
        dtype_size = dtype.itemsize

        # Extract voxel sizes from root zarr.json
        # s0_path is like "/path/dataset.zarr/s0", root is parent
        root_path = os.path.dirname(s0_path)
        root_zarr_json = os.path.join(root_path, "zarr.json")
        voxel_sizes = None

        if os.path.exists(root_zarr_json):
            with open(root_zarr_json, 'r') as f:
                root_metadata = json.load(f)
                # Look for OME-NGFF multiscales metadata
                # Try OME-NGFF v0.5 format first (attributes.ome.multiscales)
                if 'attributes' in root_metadata and 'ome' in root_metadata['attributes'] and 'multiscales' in root_metadata['attributes']['ome']:
                    multiscales = root_metadata['attributes']['ome']['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        # Get s0 transformations
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break
                # Fallback: try older OME-NGFF format (attributes.multiscales)
                elif 'attributes' in root_metadata and 'multiscales' in root_metadata['attributes']:
                    multiscales = root_metadata['attributes']['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break

    else:  # zarr2
        # Read s0 shape, chunks, dtype
        with open(zarr2_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')
        chunk_shape = metadata.get('chunks')

        # Zarr2 doesn't have sharding, so set to None
        shard_shape = chunk_shape  # Use chunk as shard for consistency
        inner_chunk_shape = None

        # Get dtype size
        dtype_str = metadata.get('dtype', '<u2')  # default uint16
        dtype = np.dtype(dtype_str)
        dtype_size = dtype.itemsize

        # Try to get dimension names from .zattrs
        zattrs_path = os.path.join(s0_path, ".zattrs")
        axes_names = None
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                attrs = json.load(f)
                axes_names = attrs.get('_ARRAY_DIMENSIONS')

        # Extract voxel sizes from multiscale .zattrs
        # s0_path is like "/path/dataset.zarr/multiscale/s0", multiscale is parent
        multiscale_path = os.path.dirname(s0_path)
        multiscale_zattrs = os.path.join(multiscale_path, ".zattrs")
        voxel_sizes = None
        if os.path.exists(multiscale_zattrs):
            with open(multiscale_zattrs, 'r') as f:
                attrs = json.load(f)
                if 'multiscales' in attrs and len(attrs['multiscales']) > 0:
                    multiscales = attrs['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break

    # Validate metadata
    if not shape:
        raise ValueError(f"Could not extract shape from {s0_path}")

    if not voxel_sizes:
        voxel_sizes = [1.0] * len(shape)
        print(f"Warning: Could not extract voxel sizes, using default: {voxel_sizes}")

    if not axes_names:
        axes_names = ['z', 'y', 'x'][-len(shape):]
        print(f"Warning: Could not extract axes_names, using default: {axes_names}")

    if not chunk_shape:
        chunk_shape = [32] * len(shape)
        print(f"Warning: Could not extract chunk_shape, using default: {chunk_shape}")

    # Calculate number of levels needed with new stopping criteria
    num_levels = calculate_num_multiscale_levels(
        shape, axes_names, voxel_sizes,
        chunk_shape=chunk_shape,
        dtype_size=dtype_size,
        min_array_nbytes=min_array_nbytes,
        min_array_shape=min_array_shape,
        shard_shape=shard_shape  # Rule 5: Stop when array < shard
    )

    # Pre-calculate all factors and keep shard/chunk shapes constant
    pyramid_plan = []
    current_voxel_sizes = voxel_sizes.copy()
    current_shape = shape.copy()

    # Keep original shard and chunk shapes constant across all levels (Mark's recommendation)
    original_shard_shape = shard_shape.copy() if shard_shape else None
    original_inner_chunk_shape = inner_chunk_shape.copy() if inner_chunk_shape else None

    for level in range(1, num_levels + 1):
        # Calculate anisotropic factor for this level
        factor = calculate_anisotropic_downsample_factors(current_voxel_sizes, axes_names)

        # Predict next level's voxel sizes (without creating data!)
        predicted_voxel_sizes = [v * f for v, f in zip(current_voxel_sizes, factor)]

        # Predict next level's array shape
        predicted_shape = [max(1, s // f) for s, f in zip(current_shape, factor)]

        # Keep chunk and shard shapes constant across all levels (Mark's recommendation)
        # "The easiest would be to just keep the shard and chunk shape constant the whole time."
        scaled_chunk = original_inner_chunk_shape.copy() if original_inner_chunk_shape else None
        scaled_shard = original_shard_shape.copy() if original_shard_shape else None

        pyramid_plan.append({
            "level": level,
            "factor": factor,
            "predicted_voxel_sizes": predicted_voxel_sizes,
            "predicted_shape": predicted_shape,
            "shard_shape": scaled_shard,
            "chunk_shape": scaled_chunk
        })

        # Update for next iteration (only voxel sizes and shape change, not chunk/shard)
        current_voxel_sizes = predicted_voxel_sizes
        current_shape = predicted_shape

    return {
        'format': format_type,
        'shape': shape,
        'voxel_sizes': voxel_sizes,
        'axes_names': axes_names,
        'chunk_shape': chunk_shape,
        'shard_shape': shard_shape,
        'inner_chunk_shape': inner_chunk_shape,
        'dtype_size': dtype_size,
        'num_levels': num_levels,
        'pyramid_plan': pyramid_plan
    }


def generate_cli_coordinator_script(output_path, pyramid_plan, args):
    """
    Generate coordinator script for cluster auto-multiscale from existing s0.

    Args:
        output_path: Zarr dataset path (contains s0/)
        pyramid_plan: Output from calculate_pyramid_plan()
        args: CLI arguments (cores, wall_time, project, etc.)

    Returns:
        Bash script as string
    """
    import sys

    # Pick downsample task based on format
    format_type = pyramid_plan.get('format', 'zarr3')
    downsample_task = "downsample_shard_zarr3" if format_type == 'zarr3' else "downsample_zarr2"

    # Paths to run tensorswitch
    tensorswitch_dir = os.path.dirname(os.path.dirname(__file__))
    python_path = sys.executable
    submit_cmd = f"cd {tensorswitch_dir} && {python_path} -m tensorswitch"

    # Build command arguments
    # Auto-detect sharding from s0 metadata (if inner_chunk_shape exists, s0 uses sharding)
    use_shard = 1 if pyramid_plan.get('inner_chunk_shape') is not None else 0
    # Use num_volumes=1 to avoid race conditions with linear chunk indexing vs 3D shard boundaries
    base_args = f"--downsample 1 --use_shard {use_shard} --memory_limit {args.memory_limit} --cores {args.cores} --wall_time {args.wall_time} --project {args.project} --num_volumes 1"

    if hasattr(args, 'custom_shard_shape') and args.custom_shard_shape:
        base_args += f" --custom_shard_shape {args.custom_shard_shape}"
    if hasattr(args, 'custom_chunk_shape') and args.custom_chunk_shape:
        base_args += f" --custom_chunk_shape {args.custom_chunk_shape}"

    num_levels = pyramid_plan.get('num_levels', 0)
    levels_info = pyramid_plan.get('pyramid_plan', [])

    script = f"""#!/bin/bash
set -e

echo "Auto-multiscale: {output_path} (s1 to s{num_levels})"

extract_job_ids() {{
    grep -oP 'Job <\\K[0-9]+(?=>)' || true
}}

submit_and_wait() {{
    local level=$1
    local task=$2
    local base_path=$3
    local output_path=$4
    shift 4
    local extra_args="$@"

    output=$({submit_cmd} \\
        --task $task \\
        --base_path "$base_path" \\
        --output_path "$output_path" \\
        --level $level \\
        {base_args} \\
        $extra_args \\
        --submit 2>&1)

    job_ids=$(echo "$output" | extract_job_ids | tr '\\n' ' ')

    if [ -z "$job_ids" ]; then
        echo "ERROR: No job IDs for s$level"
        exit 1
    fi

    echo "Submitting s$level... Jobs: $job_ids"

    wait_condition=""
    for job_id in $job_ids; do
        if [ -z "$wait_condition" ]; then
            wait_condition="done($job_id)"
        else
            wait_condition="$wait_condition && done($job_id)"
        fi
    done

    bwait -w "$wait_condition" 2>&1 || true
    echo "s$level done"
}}

"""

    # Add each level with its anisotropic factor and scaled shard/chunk shapes
    for level_info in levels_info:
        level = level_info['level']
        factor = level_info['factor']
        factor_str = ",".join(map(str, factor))
        prev_level_path = os.path.join(output_path, f"s{level-1}")

        # Build extra args with shard/chunk shapes if available
        extra_args = f"--anisotropic_factors \"{factor_str}\""
        if level_info.get('shard_shape'):
            shard_str = ",".join(map(str, level_info['shard_shape']))
            extra_args += f" --custom_shard_shape \"{shard_str}\""
        if level_info.get('chunk_shape'):
            chunk_str = ",".join(map(str, level_info['chunk_shape']))
            extra_args += f" --custom_chunk_shape \"{chunk_str}\""

        script += f"""submit_and_wait {level} "{downsample_task}" "{prev_level_path}" "{output_path}" {extra_args}
"""

    script += f"""
echo "All levels complete: {output_path}"
"""

    return script


def generate_auto_multiscale(output_path, min_dimension=256, use_shard=True, memory_limit=50,
                             custom_shard_shape=None, custom_chunk_shape=None, is_submit_mode=False):
    """
    Automatically generate multiscale pyramid levels using Yurii Zubov's anisotropic algorithm.

    Args:
        output_path: Path to the zarr dataset (contains s0/)
        min_dimension: Minimum spatial dimension for stopping (default: 256)
        use_shard: Whether to use sharding for zarr3 (default: True)
        memory_limit: Memory limit percentage (default: 50)
        custom_shard_shape: Optional custom shard shape list
        custom_chunk_shape: Optional custom chunk shape list
        is_submit_mode: Whether jobs should be submitted to cluster (True) or run locally (False)

    Credit: Yurii Zubov (Janelia CellMap Team) for the anisotropic downsampling algorithm
    """
    import json

    print("\n" + "="*80)
    print("AUTO-MULTISCALE GENERATION (Yurii Zubov's Anisotropic Algorithm)")
    print("="*80)

    # Detect format: zarr3 (has s0/zarr.json) or zarr2 (has multiscale/s0/.zarray)
    zarr3_metadata_path = os.path.join(output_path, "s0", "zarr.json")
    zarr2_metadata_path = os.path.join(output_path, "multiscale", "s0", ".zarray")

    is_zarr3 = os.path.exists(zarr3_metadata_path)
    is_zarr2 = os.path.exists(zarr2_metadata_path)

    if not is_zarr3 and not is_zarr2:
        print(f"ERROR: Cannot find zarr metadata at {output_path}")
        print("Expected either:")
        print(f"  - Zarr3: {zarr3_metadata_path}")
        print(f"  - Zarr2: {zarr2_metadata_path}")
        return

    format_type = "zarr3" if is_zarr3 else "zarr2"
    print(f"Detected format: {format_type}")

    # Read metadata based on format
    if is_zarr3:
        with open(zarr3_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')
        dimension_names = metadata.get('dimension_names')

        # Extract voxel sizes from root zarr.json
        root_zarr_json = os.path.join(output_path, "zarr.json")
        voxel_sizes = None

        if os.path.exists(root_zarr_json):
            with open(root_zarr_json, 'r') as f:
                root_metadata = json.load(f)
                # Look for OME-NGFF multiscales metadata
                if 'attributes' in root_metadata and 'multiscales' in root_metadata['attributes']:
                    multiscales = root_metadata['attributes']['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        # Get s0 transformations
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break
    else:  # zarr2
        with open(zarr2_metadata_path, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape')

        # Try to get dimension_names from .zattrs
        zattrs_path = os.path.join(output_path, "multiscale", "s0", ".zattrs")
        dimension_names = None
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                attrs = json.load(f)
                dimension_names = attrs.get('_ARRAY_DIMENSIONS')

        # Extract voxel sizes from multiscale .zattrs
        multiscale_zattrs = os.path.join(output_path, "multiscale", ".zattrs")
        voxel_sizes = None
        if os.path.exists(multiscale_zattrs):
            with open(multiscale_zattrs, 'r') as f:
                attrs = json.load(f)
                if 'multiscales' in attrs and len(attrs['multiscales']) > 0:
                    multiscales = attrs['multiscales'][0]
                    if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
                        s0_dataset = multiscales['datasets'][0]
                        if 'coordinateTransformations' in s0_dataset:
                            for transform in s0_dataset['coordinateTransformations']:
                                if transform['type'] == 'scale':
                                    voxel_sizes = transform['scale']
                                    break

    if not voxel_sizes:
        print("Warning: Could not extract voxel sizes from metadata")
        # Use default uniform voxel sizes
        voxel_sizes = [1.0] * len(shape)

    if not dimension_names:
        print("Warning: Could not extract dimension_names from metadata")
        # Guess based on shape
        if len(shape) == 3:
            dimension_names = ['z', 'y', 'x']
        elif len(shape) == 4:
            dimension_names = ['c', 'z', 'y', 'x']
        elif len(shape) == 5:
            dimension_names = ['t', 'c', 'z', 'y', 'x']
        else:
            dimension_names = [f'dim{i}' for i in range(len(shape))]

    print(f"s0 Shape: {shape}")
    print(f"s0 Dimension names: {dimension_names}")
    print(f"s0 Voxel sizes: {voxel_sizes}")

    # Calculate pyramid levels
    num_levels, level_info = calculate_num_multiscale_levels(
        shape=shape,
        axes_names=dimension_names,
        voxel_sizes=voxel_sizes,
        min_dimension=min_dimension,
        use_anisotropic=True
    )

    print(f"\nCalculated {num_levels} levels total (including s0)")
    print("\nPyramid Plan:")
    for info in level_info:
        level = info['level']
        level_shape = info['shape']
        level_voxels = info['voxel_sizes']
        factors = info['factors']

        if level == 0:
            print(f"  s{level}: shape={level_shape}, voxel_sizes={[f'{v:.2f}' for v in level_voxels]}")
        else:
            print(f"  s{level}: shape={level_shape}, voxel_sizes={[f'{v:.2f}' for v in level_voxels]}, factors={factors}")

    if is_submit_mode:
        print("\n" + "="*80)
        print("CLUSTER SUBMISSION MODE - TODO: Coordinator Script")
        print("="*80)
        print("NOTE: Auto-multiscale in cluster mode requires a coordinator script")
        print("      because s0 jobs run asynchronously. For now, please:")
        print("      1. Wait for s0 jobs to complete")
        print("      2. Run auto-multiscale locally to generate remaining levels")
        print("\nTODO: Implement coordinator script that:")
        print("  - Monitors s0 job completion")
        print("  - Reads s0 metadata after completion")
        print("  - Submits s1, s2, s3... jobs sequentially")
        print("="*80 + "\n")
        return

    # Local execution mode - generate all levels sequentially
    print("\n" + "="*80)
    print("LOCAL EXECUTION MODE - Generating levels sequentially")
    print("="*80)

    # Import task modules here to avoid circular imports
    from .tasks import downsample_shard_zarr3, downsample_zarr2

    for info in level_info[1:]:  # Skip s0 (already created)
        level = info['level']
        factors = info['factors']

        print(f"\n--- Generating s{level} with factors {factors} ---")

        # Determine input path (previous level)
        prev_level = level - 1

        if is_zarr3:
            input_path = os.path.join(output_path, f"s{prev_level}")

            # Call zarr3 downsample task
            downsample_shard_zarr3.process(
                base_path=input_path,
                output_path=output_path,
                level=level,
                start_idx=0,
                stop_idx=None,
                downsample=True,
                use_shard=use_shard,
                memory_limit=memory_limit,
                custom_shard_shape=custom_shard_shape,
                custom_chunk_shape=custom_chunk_shape,
                anisotropic_factors=factors
            )
        else:  # zarr2
            input_path = os.path.join(output_path, "multiscale", f"s{prev_level}")

            # Call zarr2 downsample task
            downsample_zarr2.process(
                base_path=input_path,
                output_path=output_path,
                level=level,
                start_idx=0,
                stop_idx=None,
                downsample=True,
                memory_limit=memory_limit,
                custom_chunk_shape=custom_chunk_shape,
                anisotropic_factors=factors
            )

        print(f"✓ Completed s{level}")

    print("\n" + "="*80)
    print(f"AUTO-MULTISCALE COMPLETE - Generated s0 through s{num_levels-1}")
    print("="*80 + "\n")


