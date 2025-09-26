from dask.cache import Cache
from ..utils import (load_ims_stack, zarr2_store_spec, get_chunk_domains, commit_tasks, 
                    get_total_chunks_from_store, extract_ims_metadata, 
                    update_ome_multiscale_metadata_zarr2, create_zarr2_ome_metadata, 
                    write_zarr2_group_metadata, convert_ims_to_zarr3_metadata)
import tensorstore as ts
import numpy as np
import psutil
import time
import os
import json

def update_zarr2_ome_xml_ims(zarr_path, source_ims_path):
    """Update .zattrs with enhanced metadata from source IMS for zarr2 format"""
    zattrs_path = os.path.join(zarr_path, '.zattrs')
    
    # Create .zattrs if it doesn't exist
    if os.path.exists(zattrs_path):
        with open(zattrs_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Extract enhanced metadata from source IMS
    try:
        ims_metadata, voxel_sizes = extract_ims_metadata(source_ims_path)
        
        if ims_metadata:
            # Add IMS metadata to zarr2 attributes
            metadata['ims_metadata'] = ims_metadata
            
            if voxel_sizes:
                metadata['voxel_sizes'] = voxel_sizes
            
            # Write back to .zattrs
            with open(zattrs_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Ensure .zgroup file exists (create if missing)
            zgroup_path = os.path.join(zarr_path, '.zgroup')
            if not os.path.exists(zgroup_path):
                zgroup_metadata = {"zarr_format": 2}
                with open(zgroup_path, 'w') as f:
                    json.dump(zgroup_metadata, f, indent=4)

            print(f"Successfully updated ims_metadata in {zattrs_path}")
        else:
            print("No IMS metadata found in source")
            
    except ImportError as e:
        print(f"Could not import IMS metadata extraction: {e}")
    except Exception as e:
        print(f"Error extracting IMS metadata: {e}")

def process(base_path, output_path, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True):
    print(f"Loading IMS file from: {base_path}", flush=True)

    volume, h5_file = load_ims_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    

    # Determine output shape and chunk strategy
    print(f"Volume dimensions: {len(volume.shape)}D")
    print(f"Volume shape: {volume.shape}")
    
    # Default chunk shapes based on dimensions
    if len(volume.shape) == 3:
        print("3D array detected - likely (Z, Y, X)")
        print(f"Z-slices: {volume.shape[0]}")
        print(f"Y (height): {volume.shape[1]}")
        print(f"X (width): {volume.shape[2]}")
        chunk_shape = (1, min(2304, volume.shape[1]), min(2304, volume.shape[2]))
        
    elif len(volume.shape) == 4:
        print("4D array detected - likely (T, Z, Y, X)")
        print(f"T-frames: {volume.shape[0]}")
        print(f"Z-slices: {volume.shape[1]}")
        print(f"Y (height): {volume.shape[2]}")
        print(f"X (width): {volume.shape[3]}")
        chunk_shape = (1, 1, min(2304, volume.shape[2]), min(2304, volume.shape[3]))
        
    elif len(volume.shape) == 5:
        print("5D array detected - likely (T, C, Z, Y, X)")
        print(f"T-frames: {volume.shape[0]}")
        print(f"C-channels: {volume.shape[1]}")
        print(f"Z-slices: {volume.shape[2]}")
        print(f"Y (height): {volume.shape[3]}")
        print(f"X (width): {volume.shape[4]}")
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

    # Close h5_file properly
    h5_file.close()

    # Write OME-Zarr metadata for zarr2
    if use_ome_structure:
        print("Writing OME-Zarr metadata for zarr2...")
        try:
            # Extract IMS metadata
            ims_metadata, voxel_sizes = extract_ims_metadata(base_path)
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            
            # Create pixel sizes dict from voxel_sizes if available
            pixel_sizes = None
            if voxel_sizes is not None and len(voxel_sizes) >= 3:
                # IMS provides XYZ, convert to dict
                pixel_sizes = {
                    'x': voxel_sizes[0],
                    'y': voxel_sizes[1], 
                    'z': voxel_sizes[2]
                }
            
            # Create zarr2 OME-ZARR metadata with IMS-specific structure
            zarr2_metadata = create_zarr2_ome_metadata(
                ome_xml=None,  # IMS doesn't typically have OME XML
                array_shape=volume.shape,
                image_name=image_name,
                pixel_sizes=pixel_sizes
            )
            
            # Add IMS-specific metadata if available
            if ims_metadata:
                zarr2_metadata['ims_metadata'] = ims_metadata
                
            if voxel_sizes:
                zarr2_metadata['voxel_sizes'] = voxel_sizes
            
            # Write .zattrs to multiscale folder
            write_zarr2_group_metadata(multiscale_path, zarr2_metadata)
            
            print("OME-Zarr metadata written successfully with multiscales structure and IMS metadata")
            
        except Exception as e:
            print(f"Warning: Could not write OME-Zarr metadata: {e}")
            
            # Fallback: try the old method
            print("Attempting fallback metadata creation...")
            try:
                # Create basic multiscales structure first
                image_name = os.path.splitext(os.path.basename(base_path))[0]
                basic_metadata = create_zarr2_ome_metadata(None, volume.shape, image_name)
                write_zarr2_group_metadata(multiscale_path, basic_metadata)
                
                # Then add IMS metadata
                update_zarr2_ome_xml_ims(multiscale_path, base_path)
                print("Fallback OME-Zarr metadata creation successful")
            except Exception as e2:
                print(f"Warning: Fallback metadata creation also failed: {e2}")

    print(f"Completed writing Zarr2 s0 at: {output_path} [{start_idx}:{stop_idx}]")