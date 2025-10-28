import tensorstore as ts
import numpy as np
import time
import psutil
from ..utils import get_chunk_domains, commit_tasks, downsample_spec, zarr2_store_spec, get_input_driver, get_total_chunks_from_store, update_ome_multiscale_metadata_zarr2
import os
import json

def process(base_path, output_path, level, start_idx=0, stop_idx=None, downsample=True, memory_limit=50, custom_chunk_shape=None, **kwargs):
    """Downsample zarr2 dataset."""
    
    # Determine input path
    if base_path.endswith(f"s{level - 1}") or level == 0:
        zarr_input_path = base_path
    else:
        zarr_input_path = os.path.join(base_path, "multiscale", f"s{level - 1}")

    input_driver = get_input_driver(zarr_input_path)
    
    # Handle both zarr2 and zarr3 inputs
    if input_driver == "zarr3":
        zarr_store_spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': zarr_input_path}
        }
    else:  # zarr2
        zarr_store_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': zarr_input_path}
        }

    downsampled_saved_path = output_path

    os.makedirs(f"{output_path}/multiscale", exist_ok=True)

    print(f"Downsample: {downsample}, Level: {level} (zarr2 format)")
    print(f"Reading from: {zarr_input_path} (format: {input_driver})")
    print(f"Writing to: {downsampled_saved_path}")

    zarr_store = ts.open(zarr_store_spec).result()

    # Apply downsampling if requested and level > 0
    if downsample and level > 0:
        # Extract dimension_names for proper downsampling
        dimension_names = None
        try:
            # For Zarr v2, try to read from .zattrs file
            zattrs_path = os.path.join(zarr_input_path, '.zattrs')
            if os.path.exists(zattrs_path):
                with open(zattrs_path, 'r') as f:
                    attrs = json.load(f)
                    dimension_names = attrs.get('_ARRAY_DIMENSIONS')
                    print(f"Extracted dimension_names from .zattrs: {dimension_names}")
        except Exception as e:
            print(f"Warning: Could not extract dimension_names from zarr2: {e}")

        if not dimension_names:
            print("Warning: No dimension_names found for zarr2, defaulting to [2,2,2] downsampling")

        print(f"Downsampling zarr2 with dimension_names: {dimension_names}")
        downsample_spec_dict = downsample_spec(zarr_store_spec, zarr_store.shape, dimension_names)
        downsample_store = ts.open(downsample_spec_dict).result()
    else:
        downsample_store = zarr_store

    # Determine chunk shape for zarr2 output
    original_shape = downsample_store.shape
    
    if custom_chunk_shape:
        if len(custom_chunk_shape) != len(original_shape):
            raise ValueError(f"Custom chunk shape {custom_chunk_shape} doesn't match data dimensions {len(original_shape)}")
        chunk_shape = tuple(custom_chunk_shape)
    else:
        # Default chunk shapes based on dimensions
        if len(original_shape) == 3:
            chunk_shape = (1, min(2304, original_shape[1]), min(2304, original_shape[2]))
        elif len(original_shape) == 4:
            chunk_shape = (1, 1, min(2304, original_shape[2]), min(2304, original_shape[3]))
        elif len(original_shape) == 5:
            chunk_shape = (1, 1, 1, min(2304, original_shape[3]), min(2304, original_shape[4]))
        else:
            # Default to chunking along last 2 dimensions
            chunk_shape = tuple([1] * (len(original_shape) - 2) + 
                              [min(2304, original_shape[-2]), min(2304, original_shape[-1])])

    print(f"Using chunk shape: {chunk_shape}")

    # Create zarr2 output store specification  
    output_level_path = os.path.join(downsampled_saved_path, "multiscale", f"s{level}")
    
    downsampled_saved_spec = zarr2_store_spec(
        output_level_path,
        downsample_store.shape,
        chunk_shape
    )
    
    # Update dtype to match input - convert to zarr format
    dtype = downsample_store.dtype
    if dtype == np.uint16:
        zarr_dtype = "<u2"
    elif dtype == np.uint8:
        zarr_dtype = "|u1"
    elif dtype == np.float32:
        zarr_dtype = "<f4"
    elif dtype == np.float64:
        zarr_dtype = "<f8"
    else:
        zarr_dtype = str(dtype)
    
    downsampled_saved_spec['metadata']['dtype'] = zarr_dtype

    print(f"Creating zarr2 output at: {output_level_path}")
    print(f"Output shape: {downsample_store.shape}")
    print(f"Output dtype: {downsample_store.dtype}")

    # Create output store
    output_store = ts.open(downsampled_saved_spec, create=True, delete_existing=True).result()
    
    # Calculate total chunks
    total_chunks = get_total_chunks_from_store(output_store, chunk_shape)
    
    print(f"Total output chunks: {total_chunks}")
    print(f"Processing chunks {start_idx} to {stop_idx if stop_idx else total_chunks}")

    # Get chunk domains for processing
    linear_indices = range(start_idx, stop_idx if stop_idx else total_chunks)
    chunk_domains = list(get_chunk_domains(chunk_shape, output_store, linear_indices))
    
    # Process chunks in batches
    batch_size = 512
    for i in range(0, len(chunk_domains), batch_size):
        batch_domains = chunk_domains[i:i+batch_size]
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > memory_limit:
            print(f"Memory usage {memory_percent:.1f}% > {memory_limit}%, waiting...")
            time.sleep(1)
            continue
        
        # Create tasks for this batch
        tasks = []
        for domain_slice in batch_domains:
            # Read from downsampled store
            data_slice = downsample_store[domain_slice].read().result()
            # Write to output store
            task = output_store[domain_slice].write(data_slice)
            tasks.append(task)
        
        # Commit batch - wait for all tasks to complete
        for task in tasks:
            task.result()
        
        end_chunk = min(i + batch_size, len(chunk_domains)) + start_idx
        print(f"Processed {len(batch_domains)} chunks up to {end_chunk}...")

    # Update OME multiscale metadata for zarr2
    print("Updating OME-Zarr multiscale metadata...")
    try:
        # Auto-detect maximum level by checking what exists
        max_level = level
        for check_level in range(level + 1, 10):  # Check up to s9
            check_path = os.path.join(output_path, "multiscale", f"s{check_level}")
            if not os.path.exists(check_path):
                break
            max_level = check_level
        
        update_ome_multiscale_metadata_zarr2(output_path, max_level=max_level)
        print(f"OME-Zarr metadata updated for zarr2 format (levels s0-s{max_level})")
        
    except Exception as e:
        print(f"Warning: Could not update OME metadata: {e}")

    print(f"Completed zarr2 downsampling level s{level} at: {output_path} [{start_idx}:{stop_idx if stop_idx else total_chunks}]")