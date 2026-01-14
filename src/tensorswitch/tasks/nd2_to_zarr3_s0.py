from dask.cache import Cache
from ..utils import (load_nd2_stack, zarr3_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, extract_nd2_ome_metadata,
                    convert_ome_to_zarr3_metadata, write_zarr3_group_metadata,
                    write_dual_zarr_metadata, detect_anisotropic_voxels,
                    update_zarr_metadata_from_source, precreate_shard_directories_inline,
                    get_tensorstore_context, detect_source_order)
import tensorstore as ts
import numpy as np
import psutil
import time
# lazy loading
import nd2
import dask.array as da
import os
import json

def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None, create_dual_metadata=True, use_v2_encoding=True, use_fortran_order=None):
    print(f"Loading ND2 file from: {base_path}", flush=True)

    volume = load_nd2_stack(base_path)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    print(f"Original chunk structure from dask: {volume.chunksize}", flush=True)

    # Detect source data order (C-order vs F-order)
    source_order_info = detect_source_order(volume)
    print(f"Source data order: {source_order_info['description']}")
    print(f"  Detected axes: {source_order_info['suggested_axes']}")

    # Determine final use_fortran_order based on:
    # 1. User explicit override (if use_fortran_order is not None)
    # 2. Otherwise, preserve source order (auto-detect)
    if use_fortran_order is not None:
        # User explicitly set use_fortran_order → respect it
        if use_fortran_order:
            print(f"✓ Using F-order output (user override)")
        else:
            print(f"✓ Using C-order output (user override)")
    else:
        # No user override → preserve source order (auto-detect)
        use_fortran_order = source_order_info['is_fortran_order']
        if use_fortran_order:
            print(f"✓ Preserving F-order from source (auto-detected)")
        else:
            print(f"✓ Preserving C-order from source (auto-detected)")

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

    # Extract voxel sizes from ND2 metadata
    ome_xml_from_nd2, voxel_sizes_um = extract_nd2_ome_metadata(base_path)
    if voxel_sizes_um:
        print(f"Extracted voxel sizes: x={voxel_sizes_um['x']:.4f}, y={voxel_sizes_um['y']:.4f}, z={voxel_sizes_um['z']:.4f} µm")
        # Detect anisotropic voxels and warn
        detect_anisotropic_voxels(voxel_sizes_um, volume.shape)
    else:
        print("Note: Could not extract voxel sizes from ND2 metadata")
        ome_xml_from_nd2 = None

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

    # Add TensorStore context to limit concurrency to LSF allocation
    store_spec['context'] = get_tensorstore_context()

    store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Pre-create shard directories if using sharded format (safety fallback)
    if use_shard and custom_shard_shape:
        precreate_shard_directories_inline(output_path, volume.shape, custom_shard_shape, use_ome_structure)

    # Prepare chunk domains and filter to assigned range
    chunk_shape = store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(store, chunk_shape=chunk_shape)
    print(f"Total chunks: {total_chunks}")
    linear_indices_to_process = range(start_idx, stop_idx or total_chunks)
    chunk_domains = get_chunk_domains(chunk_shape, store, linear_indices_to_process=linear_indices_to_process)

    print(f"Processing {len(linear_indices_to_process)} chunks: start={start_idx}, stop={stop_idx}", flush=True)

    tasks = []
    ntasks = 0

    for domain in chunk_domains:
        # Handle both 3D and 4D arrays dynamically
        slices = tuple(slice(min, max) for (min,max) in zip(domain.inclusive_min, domain.exclusive_max))
        slice_data = volume[slices]
        # Create transaction per chunk to prevent loading all data simultaneously
        with ts.Transaction() as txn:
            task = store[domain].with_transaction(txn).write(slice_data.compute(scheduler='synchronous'))

        tasks.append(task)
        ntasks += 1

    for task in tasks:
        task.result()
    
    # Write OME-Zarr metadata only if using OME structure
    if use_ome_structure:
        print("Writing OME-Zarr metadata...", flush=True)
        try:
            # Use already extracted metadata (ome_xml_from_nd2, voxel_sizes_um)
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]

            zarr3_metadata = convert_ome_to_zarr3_metadata(ome_xml_from_nd2, volume.shape, image_name)
            # Write zarr.json to root (no multiscale folder)
            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully", flush=True)
            
            # Update metadata with OME XML from source ND2 (like update_metadata.py --check-ome-xml)
            try:
                print("Updating zarr.json with enhanced OME XML metadata...", flush=True)
                update_zarr_metadata_from_source(output_path, base_path, source_type='nd2')
                print("Enhanced OME XML metadata updated successfully", flush=True)
            except Exception as e:
                print(f"Warning: Could not update enhanced OME XML metadata: {e}", flush=True)
                
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