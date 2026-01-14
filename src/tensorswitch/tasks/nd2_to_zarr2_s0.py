from dask.cache import Cache
from ..utils import (load_nd2_stack, zarr2_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, extract_nd2_ome_metadata,
                    update_ome_multiscale_metadata_zarr2, create_zarr2_ome_metadata,
                    write_zarr2_group_metadata, convert_ome_to_zarr3_metadata,
                    detect_anisotropic_voxels, get_tensorstore_context, detect_source_order)
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
    
    # Extract OME XML from source ND2 (returns tuple: (ome_xml, voxel_sizes))
    ome_xml, _ = extract_nd2_ome_metadata(source_nd2_path)

    if ome_xml:
        # Add/update ome_xml while preserving existing multiscales metadata
        metadata['ome_xml'] = ome_xml
        
        # Write back to .zattrs
        with open(zattrs_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Ensure .zgroup file exists (create if missing)
        zgroup_path = os.path.join(zarr_path, '.zgroup')
        if not os.path.exists(zgroup_path):
            zgroup_metadata = {"zarr_format": 2}
            with open(zgroup_path, 'w') as f:
                json.dump(zgroup_metadata, f, indent=4)

        print(f"Successfully updated ome_xml in {zattrs_path} while preserving multiscales metadata")
    else:
        print("No OME XML found in source ND2")

def process(base_path, output_path, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None, use_fortran_order=None):
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

    # Extract voxel sizes from ND2 metadata and detect anisotropic voxels
    try:
        ome_xml_early, voxel_sizes_early = extract_nd2_ome_metadata(base_path)
        if voxel_sizes_early:
            voxel_sizes_um = {
                'x': voxel_sizes_early.get('x', 1.0),
                'y': voxel_sizes_early.get('y', 1.0),
                'z': voxel_sizes_early.get('z', 1.0)
            }
            print(f"Extracted voxel sizes: x={voxel_sizes_um['x']:.4f}, y={voxel_sizes_um['y']:.4f}, z={voxel_sizes_um['z']:.4f} µm")
            detect_anisotropic_voxels(voxel_sizes_um, volume.shape)
        else:
            print("Note: Could not extract voxel sizes from ND2 metadata")
    except Exception as e:
        print(f"Note: Could not extract voxel sizes for anisotropic detection: {e}")

    # Set WebKnossos-compatible default chunk shape if not specified
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
        print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")

    # Determine output shape and adapt chunk shape to volume dimensions
    print(f"Volume dimensions: {len(volume.shape)}D")

    # Convert custom_chunk_shape to appropriate dimensions
    chunk_shape_base = tuple(custom_chunk_shape)

    if len(volume.shape) == 3:
        print("3D array detected - likely (Z, Y, X)")
        print(f"Z-slices: {volume.shape[0]}, Y: {volume.shape[1]}, X: {volume.shape[2]}")
        if len(chunk_shape_base) == 3:
            chunk_shape = chunk_shape_base
        elif len(chunk_shape_base) == 2:
            chunk_shape = (1,) + chunk_shape_base  # ZYX with YX chunks
        else:
            chunk_shape = chunk_shape_base

    elif len(volume.shape) == 4:
        print("4D array detected - likely (T, Z, Y, X) or (C, Z, Y, X)")
        print(f"Dim0: {volume.shape[0]}, Z: {volume.shape[1]}, Y: {volume.shape[2]}, X: {volume.shape[3]}")
        if len(chunk_shape_base) == 3:
            chunk_shape = (1,) + chunk_shape_base  # TZYX/CZYX with ZYX chunks
        elif len(chunk_shape_base) == 2:
            chunk_shape = (1, 1) + chunk_shape_base  # TZYX with YX chunks
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
        chunk_shape,
        use_fortran_order=use_fortran_order
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
            # Extract OME metadata from ND2 file (returns tuple: (ome_xml, voxel_sizes))
            ome_metadata, voxel_sizes = extract_nd2_ome_metadata(base_path)
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