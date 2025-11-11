from dask.cache import Cache
from ..utils import (load_tiff_stack, zarr3_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, extract_tiff_ome_metadata,
                    convert_ome_to_zarr3_metadata, write_zarr3_group_metadata,
                    write_dual_zarr_metadata)
import tensorstore as ts
import numpy as np
import psutil
import time
# lazy loading
import tifffile
import dask.array as da
import os
import json

def update_zarr_ome_xml(multiscale_path, source_tiff_path):
    """Update zarr.json with OME XML from source TIFF (like update_metadata.py --check-ome-xml)"""
    zarr_json_path = os.path.join(multiscale_path, 'zarr.json')
    
    if not os.path.exists(zarr_json_path):
        raise ValueError(f"zarr.json not found in {multiscale_path}")
    
    # Read current metadata
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract OME XML from source TIFF
    ome_xml, _ = extract_tiff_ome_metadata(source_tiff_path)  # Unpack tuple (ome_xml, voxel_sizes)

    if ome_xml:
        # Add ome_xml to metadata at top level (like your update_metadata.py fix)
        metadata['attributes']['ome_xml'] = ome_xml
        
        # Remove old ome_xml under ['ome']['ome_xml'] if it exists
        if (metadata.get('attributes', {}).get('ome', {}).get('ome_xml')):
            metadata['attributes']['ome'].pop('ome_xml', None)
        
        # Write back to zarr.json
        with open(zarr_json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Successfully moved ome_xml to top-level attributes in {zarr_json_path}")
    else:
        print("No OME XML found in source TIFF")

def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None, create_dual_metadata=True, use_v2_encoding=True, use_fortran_order=False):
    print(f"Loading TIFF stack from: {base_path}", flush=True)

    # Set WebKnossos-compatible defaults if not specified
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
        print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")
    if custom_shard_shape is None and use_shard:
        custom_shard_shape = [1024, 1024, 1024]
        print(f"Using WebKnossos default shard shape: {custom_shard_shape}")

    volume = load_tiff_stack(base_path)
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

    # Extract voxel sizes from TIFF metadata (OME-TIFF or ImageJ TIFF)
    ome_xml_from_tiff, voxel_sizes_um = extract_tiff_ome_metadata(base_path)
    if voxel_sizes_um:
        print(f"Extracted voxel sizes: x={voxel_sizes_um['x']:.4f}, y={voxel_sizes_um['y']:.4f}, z={voxel_sizes_um['z']:.4f} µm")

        # Detect anisotropic voxels and warn
        if len(volume.shape) >= 3:
            z_res = voxel_sizes_um.get('z', 1.0)
            xy_res = voxel_sizes_um.get('x', 1.0)
            anisotropy_ratio = z_res / xy_res

            if anisotropy_ratio > 2.0:
                print(f"⚠️  ANISOTROPIC VOXELS DETECTED: {xy_res:.4f}×{voxel_sizes_um.get('y', 1.0):.4f}×{z_res:.4f} µm")
                print(f"   Anisotropy ratio (Z/XY): {anisotropy_ratio:.2f}×")
                print(f"   For first downsampling, consider: --anisotropic_factors 2,2,1 (preserve Z resolution)")
                print(f"   After voxels become ~isotropic, use uniform 2,2,2")
    else:
        print("Note: Could not extract voxel sizes from TIFF metadata")
        ome_xml_from_tiff = None

    # Enable Dask opportunistic cache with 8 GB RAM
    cache = Cache(8 * 1024**3)  # 8 GiB = 8 × 1024³ = 8,589,934,592 bytes
    cache.register()

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

    # Note: Shard directory structure should be pre-created by submit_job() before workers start.
    # This block is kept as a safety fallback but should rarely execute.
    if use_shard and custom_shard_shape:
        # Check if directories already exist (they should have been pre-created)
        if use_ome_structure:
            base_check_path = os.path.join(output_path, "s0", "c", "0")
        else:
            base_check_path = os.path.join(output_path, "c", "0")

        if os.path.exists(base_check_path):
            print("✓ Shard directories already exist (pre-created), skipping redundant creation")
        else:
            print("⚠ Shard directories not found, creating them now (this should be rare)...")
            import numpy as np
            shard_shape = custom_shard_shape if isinstance(custom_shard_shape, list) else [int(x) for x in custom_shard_shape.split(',')]
            output_shape = list(volume.shape)

            # Adjust shard shape to match array dimensions
            if len(output_shape) == 4 and len(shard_shape) == 3:
                shard_shape = [1] + shard_shape  # CZYX
            elif len(output_shape) == 4 and len(shard_shape) == 2:
                shard_shape = [1, 1] + shard_shape  # CZYX with YX shards
            elif len(output_shape) == 3 and len(shard_shape) == 2:
                shard_shape = [1] + shard_shape  # CYX
            elif len(output_shape) == 5 and len(shard_shape) == 3:
                shard_shape = [1, 1] + shard_shape  # TCZYX
            elif len(output_shape) == 5 and len(shard_shape) == 2:
                shard_shape = [1, 1, 1] + shard_shape  # TCZYX with YX shards

            # Calculate number of shards in each dimension
            num_shards = [((dim_size + shard_size - 1) // shard_size) for dim_size, shard_size in zip(output_shape, shard_shape)]

            # Create all shard parent directories
            if use_ome_structure:
                base_shard_path = os.path.join(output_path, "s0", "c")
            else:
                base_shard_path = os.path.join(output_path, "c")

            if len(num_shards) == 4:  # CZYX
                for c in range(num_shards[0]):
                    for z in range(num_shards[1]):
                        for y in range(num_shards[2]):
                            dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                            os.makedirs(dir_path, exist_ok=True)
            elif len(num_shards) == 3:  # CYX or ZYX
                for dim0 in range(num_shards[0]):
                    for dim1 in range(num_shards[1]):
                        dir_path = os.path.join(base_shard_path, str(dim0), str(dim1))
                        os.makedirs(dir_path, exist_ok=True)
            elif len(num_shards) == 5:  # TCZYX
                for t in range(num_shards[0]):
                    for c in range(num_shards[1]):
                        for z in range(num_shards[2]):
                            for y in range(num_shards[3]):
                                dir_path = os.path.join(base_shard_path, str(t), str(c), str(z), str(y))
                                os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory structure for {np.prod(num_shards[:3] if len(num_shards) >= 3 else num_shards)} shard locations")

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

    for domain in chunk_domains:
        # Handle both 3D and 4D arrays dynamically
        slices = tuple(slice(min, max) for (min,max) in zip(domain.inclusive_min, domain.exclusive_max))
        slice_data = volume[slices]
        task = store[domain].with_transaction(txn).write(slice_data.compute())

        tasks.append(task)
        ntasks += 1

        txn = commit_tasks(tasks, txn, memory_limit=memory_limit)
    
        if ntasks % 512 == 0:
            #chunk_idx = range(start_idx, stop_idx)[ntasks]
            chunk_idx = range(start_idx, stop_idx)[ntasks] if stop_idx else start_idx + ntasks
            print(f"Queued {ntasks} chunk writes up to {chunk_idx}...", flush=True)
    
    for task in tasks:
        task.result()
    
    txn.commit_sync()
    
    # Write OME-Zarr metadata only if using OME structure
    if use_ome_structure:
        print("Writing OME-Zarr metadata...", flush=True)
        try:
            # Use already extracted metadata (ome_xml_from_tiff, voxel_sizes_um)
            # Extract image name from file path
            image_name = os.path.splitext(os.path.basename(base_path))[0]

            # Convert OME metadata to zarr3 format
            zarr3_metadata = convert_ome_to_zarr3_metadata(ome_xml_from_tiff, volume.shape, image_name)

            # Update with extracted voxel sizes if available
            if voxel_sizes_um:
                # Update coordinate transformations in metadata
                if 'attributes' in zarr3_metadata and 'ome' in zarr3_metadata['attributes']:
                    multiscales = zarr3_metadata['attributes']['ome'].get('multiscales', [])
                    if multiscales and 'datasets' in multiscales[0]:
                        datasets = multiscales[0]['datasets']
                        if datasets and 'coordinateTransformations' in datasets[0]:
                            transforms = datasets[0]['coordinateTransformations']
                            for transform in transforms:
                                if transform.get('type') == 'scale':
                                    # Update scale with actual voxel sizes (ZYX order for 3D)
                                    if len(volume.shape) == 3:
                                        transform['scale'] = [voxel_sizes_um['z'], voxel_sizes_um['y'], voxel_sizes_um['x']]
                                        print(f"Updated metadata with voxel sizes: {transform['scale']}")
                                    elif len(volume.shape) == 4:  # CZYX
                                        transform['scale'] = [1.0, voxel_sizes_um['z'], voxel_sizes_um['y'], voxel_sizes_um['x']]
                                        print(f"Updated metadata with voxel sizes: {transform['scale']}")

            # Write zarr.json to root (no multiscale folder)
            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully", flush=True)

            # Update metadata with OME XML from source TIFF (like update_metadata.py --check-ome-xml)
            try:
                print("Updating zarr.json with enhanced OME XML metadata...", flush=True)
                update_zarr_ome_xml(output_path, base_path)
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
