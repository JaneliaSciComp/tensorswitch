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
import json
from ome_types import from_xml

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


def zarr3_store_spec(path, shape, dtype, use_shard=True, level_path="s0", use_ome_structure=True):
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
        # For OME-ZARR: use multiscale structure with level subdirectory
        array_path = os.path.join(path, level_path)
    else:
        # For plain zarr3: write directly to specified path
        array_path = path
    
    return {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': array_path},
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
    Detect whether input is TIFF, N5, Zarr2, Zarr3, or ND2

    Checks if there is a attributes.json, .zarray, zarr.json, .tiff/.tif, or .nd2 files
    """

    if not os.path.exists(input_path):
        raise ValueError(f"""
        ❌ Could not detect N5/Zarr version or dataset format at: {input_path}.
        {input_path} does not exist.
        """)
    
    # Check if input is a single nd2 file
    if os.path.isfile(input_path) and input_path.lower().endswith(".nd2"):
        return "nd2"
    
    # Check if input is a single tiff file
    if os.path.isfile(input_path) and input_path.lower().endswith((".tiff", ".tif")):
        return "tiff"
    
    # For directories, check for format indicator files
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
        ❌ Could not detect N5/Zarr version or dataset format at: {input_path}.
        {n5_path} does not exist.
        {zarr2_path} does not exist.
        {zarr3_path} does not exist.
        And no TIFF or ND2 files found in folder.
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
        ome_xml = f.ome_metadata()
        if ome_xml:
            # Return the raw XML string
            return ome_xml
        else:
            return None

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
    
    # Add OME XML if available - ensure it's a string
    if ome_xml:
        if isinstance(ome_xml, str):
            metadata["attributes"]["ome"]["ome_xml"] = ome_xml
        else:
            # Convert OME object to string if needed
            metadata["attributes"]["ome"]["ome_xml"] = str(ome_xml)
    
    return metadata


def write_zarr3_group_metadata(output_path, metadata):
    """Write zarr3 group-level zarr.json file with OME-ZARR metadata."""
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def update_ome_multiscale_metadata(zarr_path, max_level=4):
    """Update OME-ZARR metadata to include all multiscale levels s0 through max_level."""
    zarr_json_path = os.path.join(zarr_path, "zarr.json")
    
    # Read existing metadata
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Get existing multiscales and s0 scale factors
    multiscales = metadata["attributes"]["ome"]["multiscales"][0]
    s0_scale_factors = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]
    
    # Build datasets for all levels with multiscale paths
    datasets = []
    for level in range(max_level + 1):
        scale_factor = 2 ** level  # 1, 2, 4, 8, 16 for levels 0-4
        current_scale = [sf * scale_factor for sf in s0_scale_factors]
        
        datasets.append({
            "path": f"multiscale/s{level}",  # Updated to multiscale path
            "coordinateTransformations": [{
                "type": "scale",
                "scale": current_scale
            }]
        })
    
    # Update multiscales datasets
    multiscales["datasets"] = datasets
    
    # Write back updated metadata
    with open(zarr_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated OME metadata for {zarr_path} with levels multiscale/s0-s{max_level}")


