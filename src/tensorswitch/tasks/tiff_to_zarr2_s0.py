from dask.cache import Cache
from ..utils import (load_tiff_stack, zarr2_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, update_ome_multiscale_metadata_zarr2,
                    create_zarr2_ome_metadata, write_zarr2_group_metadata,
                    extract_tiff_ome_metadata, detect_anisotropic_voxels,
                    get_tensorstore_context)
import tensorstore as ts
import numpy as np
import psutil
import time
import os
import json

def update_zarr2_ome_xml_tiff(zarr_path, source_tiff_path):
    """Update .zattrs with metadata from source TIFF for zarr2 format"""
    zattrs_path = os.path.join(zarr_path, '.zattrs')
    
    # Create .zattrs if it doesn't exist
    if os.path.exists(zattrs_path):
        with open(zattrs_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Add TIFF source information
    metadata['source_tiff'] = source_tiff_path
    metadata['conversion_type'] = 'tiff_to_zarr2'
    
    # Write back to .zattrs
    with open(zattrs_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Ensure .zgroup file exists (create if missing)
    zgroup_path = os.path.join(zarr_path, '.zgroup')
    if not os.path.exists(zgroup_path):
        zgroup_metadata = {"zarr_format": 2}
        with open(zgroup_path, 'w') as f:
            json.dump(zgroup_metadata, f, indent=4)

    print(f"Successfully updated source metadata in {zattrs_path}")

def process(base_path, output_path, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_chunk_shape=None):
    print(f"Loading TIFF stack from: {base_path}", flush=True)

    volume = load_tiff_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)

    # Extract voxel sizes from TIFF metadata and detect anisotropic voxels
    try:
        ome_xml_early, voxel_sizes_um = extract_tiff_ome_metadata(base_path)
        if voxel_sizes_um:
            print(f"Extracted voxel sizes: x={voxel_sizes_um['x']:.4f}, y={voxel_sizes_um['y']:.4f}, z={voxel_sizes_um['z']:.4f} µm")
            detect_anisotropic_voxels(voxel_sizes_um, volume.shape)
        else:
            print("Note: Could not extract voxel sizes from TIFF metadata")
    except Exception as e:
        print(f"Note: Could not extract voxel sizes for anisotropic detection: {e}")

    # Set WebKnossos-compatible default chunk shape if not specified
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
        print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")

    # Determine output shape and chunk strategy
    print(f"Volume dimensions: {len(volume.shape)}D")
    print(f"Volume shape: {volume.shape}")

    # Convert custom_chunk_shape to tuple and adapt to volume dimensions
    if isinstance(custom_chunk_shape, list):
        chunk_shape_base = tuple(custom_chunk_shape)
    else:
        chunk_shape_base = custom_chunk_shape

    # Adjust chunk shape to match array dimensions
    if len(volume.shape) == 3:
        print("3D array detected - likely (Z, Y, X) or (C, Y, X)")
        print(f"Dim0: {volume.shape[0]}, Y: {volume.shape[1]}, X: {volume.shape[2]}")
        if len(chunk_shape_base) == 3:
            chunk_shape = chunk_shape_base
        elif len(chunk_shape_base) == 2:
            chunk_shape = (1,) + chunk_shape_base  # ZYX or CYX with YX chunks
        else:
            chunk_shape = chunk_shape_base

    elif len(volume.shape) == 4:
        print("4D array detected - likely (C, Z, Y, X) or (T, Z, Y, X)")
        print(f"Dim0: {volume.shape[0]}, Z: {volume.shape[1]}, Y: {volume.shape[2]}, X: {volume.shape[3]}")
        if len(chunk_shape_base) == 3:
            chunk_shape = (1,) + chunk_shape_base  # CZYX/TZYX with ZYX chunks
        elif len(chunk_shape_base) == 2:
            chunk_shape = (1, 1) + chunk_shape_base  # CZYX with YX chunks
        else:
            chunk_shape = chunk_shape_base

    elif len(volume.shape) == 5:
        print("5D array detected - likely (T, C, Z, Y, X)")
        print(f"T: {volume.shape[0]}, C: {volume.shape[1]}, Z: {volume.shape[2]}, Y: {volume.shape[3]}, X: {volume.shape[4]}")
        if len(chunk_shape_base) == 3:
            chunk_shape = (1, 1) + chunk_shape_base  # TCZYX with ZYX chunks
        elif len(chunk_shape_base) == 2:
            chunk_shape = (1, 1, 1) + chunk_shape_base  # TCZYX with YX chunks
        else:
            chunk_shape = chunk_shape_base
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

    # Add TensorStore context to limit concurrency to LSF allocation
    store_spec['context'] = get_tensorstore_context()

    # Create the zarr2 store
    store = ts.open(store_spec, create=True, delete_existing=True).result()
    total_chunks = get_total_chunks_from_store(store, chunk_shape)
    
    print(f"Total chunks: {total_chunks}")
    print(f"Processing {stop_idx - start_idx if stop_idx else total_chunks - start_idx} chunks: start={start_idx}, stop={stop_idx}")

    # Get chunk domains for processing
    linear_indices = range(start_idx, stop_idx if stop_idx else total_chunks)
    chunk_domains = list(get_chunk_domains(chunk_shape, store, linear_indices))

    print(f"Processing {len(chunk_domains)} chunks: start={start_idx}, stop={stop_idx}", flush=True)

    # Process chunks with transaction-per-chunk pattern (Mark's fix)
    tasks = []
    for domain in chunk_domains:
        # Convert domain to slices for dask array indexing
        slices = tuple(slice(min, max) for (min, max) in zip(domain.inclusive_min, domain.exclusive_max))
        slice_data = volume[slices]

        # Create transaction per chunk to prevent loading all data simultaneously
        with ts.Transaction() as txn:
            task = store[domain].with_transaction(txn).write(slice_data.compute(scheduler='synchronous'))

        tasks.append(task)

    # Wait for all tasks to complete
    for task in tasks:
        task.result()

    # Write OME-Zarr metadata for zarr2
    if use_ome_structure:
        print("Writing OME-Zarr metadata for zarr2...")
        try:
            # Try to extract OME metadata from TIFF file if available
            ome_metadata = None
            try:
                ome_metadata = extract_tiff_ome_metadata(base_path)
            except Exception as e:
                print(f"Note: Could not extract OME metadata from TIFF: {e}")
            
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            
            # Create zarr2 OME-ZARR metadata
            zarr2_metadata = create_zarr2_ome_metadata(
                ome_xml=ome_metadata,
                array_shape=volume.shape,
                image_name=image_name
            )
            
            # Add TIFF source information
            zarr2_metadata['source_tiff'] = base_path
            zarr2_metadata['conversion_type'] = 'tiff_to_zarr2'
            
            # Write .zattrs to multiscale folder
            write_zarr2_group_metadata(multiscale_path, zarr2_metadata)
            
            print("OME-Zarr metadata written successfully with multiscales structure and TIFF metadata")
            
        except Exception as e:
            print(f"Warning: Could not write OME-Zarr metadata: {e}")
            
            # Fallback: try the old method
            print("Attempting fallback metadata creation...")
            try:
                # Create basic multiscales structure first
                image_name = os.path.splitext(os.path.basename(base_path))[0]
                basic_metadata = create_zarr2_ome_metadata(None, volume.shape, image_name)
                write_zarr2_group_metadata(multiscale_path, basic_metadata)
                
                # Then add TIFF metadata
                update_zarr2_ome_xml_tiff(multiscale_path, base_path)
                print("Fallback OME-Zarr metadata creation successful")
            except Exception as e2:
                print(f"Warning: Fallback metadata creation also failed: {e2}")

    print(f"Completed writing Zarr2 s0 at: {output_path} [{start_idx}:{stop_idx}]")