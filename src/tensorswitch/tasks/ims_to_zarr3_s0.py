from dask.cache import Cache
from ..utils import (load_ims_stack, zarr3_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, extract_ims_metadata,
                    convert_ims_to_zarr3_metadata, write_zarr3_group_metadata,
                    write_dual_zarr_metadata, detect_anisotropic_voxels)
import tensorstore as ts
import numpy as np
import psutil
import time
import os
import json

def update_zarr_ome_xml_ims(zarr_root_path, source_ims_path):
    """Update zarr.json with enhanced metadata from source IMS (like update_metadata.py --check-ome-xml)"""
    zarr_json_path = os.path.join(zarr_root_path, 'zarr.json')

    if not os.path.exists(zarr_json_path):
        raise ValueError(f"zarr.json not found in {zarr_root_path}")
    
    # Read current metadata
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract enhanced metadata from source IMS
    try:
        ims_metadata, voxel_sizes = extract_ims_metadata(source_ims_path)
        
        if ims_metadata:
            # Add IMS metadata to top level attributes
            metadata['attributes']['ims_metadata'] = ims_metadata
            
            # Update voxel size information if available
            if voxel_sizes and 'ome' in metadata.get('attributes', {}):
                multiscales = metadata['attributes']['ome'].get('multiscales', [])
                if multiscales and 'datasets' in multiscales[0]:
                    datasets = multiscales[0]['datasets']
                    if datasets and 'coordinateTransformations' in datasets[0]:
                        # Update scale with actual voxel sizes
                        transforms = datasets[0]['coordinateTransformations']
                        for transform in transforms:
                            if transform.get('type') == 'scale' and len(voxel_sizes) >= 3:
                                # IMS provides XYZ, convert to ZYX for zarr
                                if len(transform.get('scale', [])) >= 3:
                                    transform['scale'] = [voxel_sizes[2], voxel_sizes[1], voxel_sizes[0]]
            
            # Write back to zarr.json
            with open(zarr_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Successfully added IMS metadata to {zarr_json_path}")
        else:
            print("No enhanced IMS metadata found")
            
    except Exception as e:
        print(f"Could not extract IMS metadata: {e}")

def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None, create_dual_metadata=True, use_v2_encoding=True, use_fortran_order=False):
    print(f"Loading IMS file from: {base_path}", flush=True)

    volume, h5_file = load_ims_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    print(f"Original chunk structure from dask: {volume.chunksize}", flush=True)

    # DEBUG
    print(f"Volume dimensions: {len(volume.shape)}D")
    print(f"Volume chunk structure from dask: {volume.chunksize}")
    
    # DEBUG: what a single chunk looks like
    if len(volume.shape) == 4:
        print("4D array detected - likely (C, Z, Y, X)")
        print(f"Channels: {volume.shape[0]}")
        print(f"Z-slices: {volume.shape[1]}")
        print(f"Y (height): {volume.shape[2]}")
        print(f"X (width): {volume.shape[3]}")
    elif len(volume.shape) == 3:
        print("3D array detected - likely (Z, Y, X)")
        print(f"Z-slices: {volume.shape[0]}")
        print(f"Y (height): {volume.shape[1]}")
        print(f"X (width): {volume.shape[2]}")

    # Enable Dask cache with 8 GB RAM
    cache = Cache(8 * 1024**3)  # 8 GiB = 8 × 1024³ = 8,589,934,592 bytes
    cache.register()

    # Extract voxel sizes from IMS metadata
    ims_metadata, voxel_sizes = extract_ims_metadata(base_path)
    if voxel_sizes and len(voxel_sizes) >= 3:
        voxel_sizes_um = {'x': voxel_sizes[0], 'y': voxel_sizes[1], 'z': voxel_sizes[2]}
        print(f"Extracted voxel sizes: x={voxel_sizes_um['x']:.4f}, y={voxel_sizes_um['y']:.4f}, z={voxel_sizes_um['z']:.4f} µm")
        # Detect anisotropic voxels and warn
        detect_anisotropic_voxels(voxel_sizes_um, volume.shape)
    else:
        print("Note: Could not extract voxel sizes from IMS metadata")

    # Set WebKnossos defaults if not specified
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
        print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")
    if custom_shard_shape is None and use_shard:
        custom_shard_shape = [1024, 1024, 1024]
        print(f"Using WebKnossos default shard shape: {custom_shard_shape}")

    # Create or open the output Zarr3 store
    store_spec = zarr3_store_spec(
        path=output_path,
        shape=volume.shape,
        dtype=str(volume.dtype),
        use_shard=use_shard,
        level_path="s0",
        use_ome_structure=use_ome_structure,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape,
        use_v2_encoding=use_v2_encoding,
        use_fortran_order=use_fortran_order
    )

    store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Prepare chunk domains and filter to assigned range
    chunk_shape = store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(store, chunk_shape=chunk_shape)
    print(f"Total chunks: {total_chunks}")
    linear_indices_to_process = range(start_idx, stop_idx or total_chunks)
    chunk_domains = get_chunk_domains(chunk_shape, store, linear_indices_to_process=linear_indices_to_process)

    print(f"Processing {len(linear_indices_to_process)} chunks: start={start_idx}, stop={stop_idx}", flush=True)

    tasks = []
    ntasks = 0
    txn = ts.Transaction()

    try:
        for domain in chunk_domains:
            # Handle both 3D and 4D arrays dynamically
            slices = tuple(slice(min, max) for (min,max) in zip(domain.inclusive_min, domain.exclusive_max))
            slice_data = volume[slices]
            task = store[domain].with_transaction(txn).write(slice_data.compute())

            tasks.append(task)
            ntasks += 1

            txn = commit_tasks(tasks, txn, memory_limit=memory_limit)
        
            if ntasks % 512 == 0:
                chunk_idx = range(start_idx, stop_idx or total_chunks)[ntasks-1] if ntasks > 0 else start_idx
                print(f"Queued {ntasks} chunk writes up to {chunk_idx}...", flush=True)
        
        for task in tasks:
            task.result()
        
        txn.commit_sync()
        
    finally:
        # Ensure h5py file is properly closed
        if h5_file:
            h5_file.close()
            print("Closed IMS file handle")
    
    # Write OME-Zarr metadata only if using OME structure
    if use_ome_structure:
        print("Writing OME-Zarr metadata...", flush=True)
        try:
            metadata, voxel_sizes = extract_ims_metadata(base_path)
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            
            zarr3_metadata = convert_ims_to_zarr3_metadata(base_path, volume.shape, voxel_sizes)
            # Write zarr.json to root (no multiscale folder)
            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully", flush=True)
            
            # Update metadata with enhanced IMS metadata (like update_metadata.py --check-ome-xml)
            try:
                print("Updating zarr.json with enhanced IMS metadata...", flush=True)
                update_zarr_ome_xml_ims(output_path, base_path)
                print("Enhanced IMS metadata updated successfully", flush=True)
            except Exception as e:
                print(f"Warning: Could not update enhanced IMS metadata: {e}", flush=True)
                
        except Exception as e:
            print(f"Warning: Could not write OME-Zarr metadata: {e}", flush=True)
    else:
        print("Skipping OME-ZARR metadata (plain zarr3 format)", flush=True)

    # Create dual zarr v2/v3 metadata if requested and using OME structure
    if create_dual_metadata and use_ome_structure:
        print("Creating dual metadata...", flush=True)
        try:
            success = write_dual_zarr_metadata(output_path, base_path)
            if success:
                print("Dual metadata created", flush=True)
            else:
                print("Warning: Failed to create dual metadata", flush=True)
        except Exception as e:
            print(f"Warning: Could not create dual metadata: {e}", flush=True)

    print(f"Completed writing Zarr3 s0 at: {output_path} [{start_idx}:{stop_idx}]", flush=True)