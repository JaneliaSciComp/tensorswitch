from dask.cache import Cache
from ..utils import (load_nd2_stack, zarr2_store_spec, get_chunk_domains, commit_tasks, 
                    get_total_chunks_from_store, extract_nd2_ome_metadata, 
                    update_ome_multiscale_metadata_zarr2, create_zarr2_ome_metadata, 
                    write_zarr2_group_metadata, convert_ome_to_zarr3_metadata)
import tensorstore as ts
import numpy as np
import psutil
import time
# lazy loading
import nd2
import dask.array as da
import os
import json

def update_zarr2_ome_xml_nd2(zarr_path, source_nd2_path):
    """Update .zattrs with OME XML from source ND2 for zarr2 format while preserving existing multiscales metadata"""
    zattrs_path = os.path.join(zarr_path, '.zattrs')
    
    # Read existing metadata (should already exist with multiscales)
    if os.path.exists(zattrs_path):
        with open(zattrs_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"Warning: .zattrs file not found at {zattrs_path}")
        return
    
    # Extract OME XML from source ND2
    ome_xml = extract_nd2_ome_metadata(source_nd2_path)
    
    if ome_xml:
        # Add/update ome_xml while preserving existing multiscales metadata
        metadata['ome_xml'] = ome_xml
        
        # Write back to .zattrs
        with open(zattrs_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Successfully updated ome_xml in {zattrs_path} while preserving multiscales metadata")
    else:
        print("No OME XML found in source ND2")

def process(base_path, output_path, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None):
    print(f"Loading ND2 file from: {base_path}", flush=True)

    volume = load_nd2_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    print(f"Original chunk structure from dask: {volume.chunksize}", flush=True)


    # Determine output shape based on volume dimensions
    if len(volume.shape) == 3:
        print("Volume dimensions: 3D")
        print(f"Volume chunk structure from dask: {volume.chunksize}")
        print("3D array detected - likely (Z, Y, X)")
        print(f"Z-slices: {volume.shape[0]}")
        print(f"Y (height): {volume.shape[1]}")
        print(f"X (width): {volume.shape[2]}")
        
        # Default chunk shape for 3D
        if custom_chunk_shape and len(custom_chunk_shape) == 3:
            chunk_shape = tuple(custom_chunk_shape)
        else:
            chunk_shape = (1, min(2304, volume.shape[1]), min(2304, volume.shape[2]))
            
    elif len(volume.shape) == 4:
        print("Volume dimensions: 4D")
        print(f"Volume chunk structure from dask: {volume.chunksize}")
        print("4D array detected - likely (T, Z, Y, X)")
        print(f"T-frames: {volume.shape[0]}")
        print(f"Z-slices: {volume.shape[1]}")
        print(f"Y (height): {volume.shape[2]}")
        print(f"X (width): {volume.shape[3]}")
        
        # Default chunk shape for 4D  
        if custom_chunk_shape and len(custom_chunk_shape) == 4:
            chunk_shape = tuple(custom_chunk_shape)
        else:
            chunk_shape = (1, 1, min(2304, volume.shape[2]), min(2304, volume.shape[3]))
            
    elif len(volume.shape) == 5:
        print("Volume dimensions: 5D")
        print(f"Volume chunk structure from dask: {volume.chunksize}")
        print("5D array detected - likely (T, C, Z, Y, X)")
        print(f"T-frames: {volume.shape[0]}")
        print(f"C-channels: {volume.shape[1]}")
        print(f"Z-slices: {volume.shape[2]}")
        print(f"Y (height): {volume.shape[3]}")
        print(f"X (width): {volume.shape[4]}")
        
        # Default chunk shape for 5D
        if custom_chunk_shape and len(custom_chunk_shape) == 5:
            chunk_shape = tuple(custom_chunk_shape)
        else:
            chunk_shape = (1, 1, 1, min(2304, volume.shape[3]), min(2304, volume.shape[4]))
    else:
        raise ValueError(f"Unsupported volume dimensions: {len(volume.shape)}D")

    print(f"Using chunk shape: {chunk_shape}")

    # Create zarr2 store specification
    if use_ome_structure:
        zarr_level_path = os.path.join(output_path, "multiscale", "s0")
        multiscale_path = os.path.join(output_path, "multiscale")
    else:
        zarr_level_path = output_path
        multiscale_path = output_path

    # Create zarr2 store (no sharding for zarr2)
    store_spec = zarr2_store_spec(
        zarr_level_path,
        volume.shape,
        chunk_shape
    )
    
    # Update dtype in store spec - convert numpy dtype to zarr format
    if volume.dtype == np.uint16:
        zarr_dtype = "<u2"
    elif volume.dtype == np.uint8:
        zarr_dtype = "|u1"
    elif volume.dtype == np.float32:
        zarr_dtype = "<f4"
    elif volume.dtype == np.float64:
        zarr_dtype = "<f8"
    else:
        zarr_dtype = str(volume.dtype)
    
    store_spec['metadata']['dtype'] = zarr_dtype

    # Create the zarr2 store
    store = ts.open(store_spec, create=True, delete_existing=True).result()
    total_chunks = get_total_chunks_from_store(store, chunk_shape)
    
    print(f"Total chunks: {total_chunks}")
    print(f"Processing {stop_idx - start_idx if stop_idx else total_chunks - start_idx} chunks: start={start_idx}, stop={stop_idx}")

    # Get chunk domains for processing
    linear_indices = range(start_idx, stop_idx if stop_idx else total_chunks)
    chunk_domains = list(get_chunk_domains(chunk_shape, store, linear_indices))
    
    # Process chunks in batches
    batch_size = 512
    for i in range(0, len(chunk_domains), batch_size):
        batch_domains = chunk_domains[i:i+batch_size]
        
        # Get memory usage
        memory_percent = psutil.virtual_memory().percent
        
        # Skip this batch if memory usage is too high
        if memory_percent > memory_limit:
            print(f"Memory usage {memory_percent:.1f}% > {memory_limit}%, waiting...")
            time.sleep(1)
            continue
        
        # Create tasks for this batch
        tasks = []
        for domain_slice in batch_domains:
            # Convert domain to slices for dask array indexing
            slices = tuple(slice(min, max) for (min,max) in zip(domain_slice.inclusive_min, domain_slice.exclusive_max))
            # Read data slice
            data_slice = volume[slices]
            # Create write task
            task = store[domain_slice].write(data_slice)
            tasks.append(task)
        
        # Commit batch - wait for all tasks to complete
        for task in tasks:
            task.result()
        
        end_chunk = min(i + batch_size, len(chunk_domains)) + start_idx
        print(f"Queued {len(batch_domains)} chunk writes up to {end_chunk}...")

    # Write OME-Zarr metadata for zarr2
    if use_ome_structure:
        print("Writing OME-Zarr metadata for zarr2...")
        try:
            # Extract OME metadata from ND2 file
            ome_metadata = extract_nd2_ome_metadata(base_path)
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            
            # Create zarr2 OME-ZARR metadata with both multiscales structure and OME XML
            zarr2_metadata = create_zarr2_ome_metadata(ome_metadata, volume.shape, image_name)
            
            # Write .zattrs to multiscale folder
            write_zarr2_group_metadata(multiscale_path, zarr2_metadata)
            
            print("OME-Zarr metadata written successfully with multiscales structure and OME XML")
            
        except Exception as e:
            print(f"Warning: Could not write OME-Zarr metadata: {e}")
            
            # Fallback: try the old method
            print("Attempting fallback metadata creation...")
            try:
                # Create basic multiscales structure first
                basic_metadata = create_zarr2_ome_metadata(None, volume.shape, image_name)
                write_zarr2_group_metadata(multiscale_path, basic_metadata)
                
                # Then add OME XML
                update_zarr2_ome_xml_nd2(multiscale_path, base_path)
                print("Fallback OME-Zarr metadata creation successful")
            except Exception as e2:
                print(f"Warning: Fallback metadata creation also failed: {e2}")

    print(f"Completed writing Zarr2 s0 at: {output_path} [{start_idx}:{stop_idx}]")