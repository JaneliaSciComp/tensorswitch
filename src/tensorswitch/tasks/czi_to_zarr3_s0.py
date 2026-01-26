from dask.cache import Cache
from ..utils import (load_czi_stack, zarr3_store_spec, get_chunk_domains, commit_tasks,
                    get_total_chunks_from_store, extract_czi_metadata,
                    create_zarr3_ome_metadata, write_zarr3_group_metadata,
                    write_dual_zarr_metadata, detect_anisotropic_voxels,
                    update_zarr_metadata_from_source, precreate_shard_directories_inline,
                    get_tensorstore_context, detect_source_order, transform_czi_to_ome_xml)
import tensorstore as ts
import numpy as np
import psutil
import time
import os
import json


def process(base_path, output_path, use_shard=False, memory_limit=50, start_idx=0, stop_idx=None,
            use_ome_structure=True, custom_shard_shape=None, custom_chunk_shape=None,
            create_dual_metadata=True, use_v2_encoding=True, use_fortran_order=None,
            view_index=None):
    """
    Convert CZI file to Zarr3 format (s0 level).

    Args:
        base_path: Path to input CZI file
        output_path: Path to output Zarr3 dataset
        use_shard: Whether to use sharding (default False for this task)
        memory_limit: Memory limit in GB
        start_idx: Starting chunk index
        stop_idx: Ending chunk index
        use_ome_structure: Whether to use OME-NGFF structure with s0 subdirectory
        custom_shard_shape: Custom shard shape (not used when use_shard=False)
        custom_chunk_shape: Custom chunk shape (e.g., [256, 256, 256])
        create_dual_metadata: Whether to create dual v2/v3 metadata
        use_v2_encoding: Whether to use v2 chunk key encoding
        use_fortran_order: Force F-order output (None = auto-detect from source)
        view_index: Optional specific view index to load for multi-view CZI files
    """
    print(f"Loading CZI file from: {base_path}", flush=True)

    # Set default chunk shape for this task (256^3 as specified)
    if custom_chunk_shape is None:
        custom_chunk_shape = [256, 256, 256]
        print(f"Using default chunk shape: {custom_chunk_shape}")

    # Load CZI as dask array (returns axes_order from CZI dimension analysis)
    volume, czi_handle, czi_axes_order = load_czi_stack(base_path, view_index=view_index)
    print(f"Original volume shape: {volume.shape}, dtype: {volume.dtype}", flush=True)
    print(f"Original chunk structure from dask: {volume.chunksize}", flush=True)
    print(f"CZI axes order: {czi_axes_order}")

    # Detect source data order (C-order vs F-order)
    source_order_info = detect_source_order(volume)
    print(f"Source data order: {source_order_info['description']}")

    # Determine final use_fortran_order based on:
    # 1. User explicit override (if use_fortran_order is not None)
    # 2. Otherwise, preserve source order (auto-detect)
    if use_fortran_order is not None:
        if use_fortran_order:
            print(f"Using F-order output (user override)")
        else:
            print(f"Using C-order output (user override)")
    else:
        use_fortran_order = source_order_info['is_fortran_order']
        if use_fortran_order:
            print(f"Preserving F-order from source (auto-detected)")
        else:
            print(f"Preserving C-order from source (auto-detected)")

    # Print dimension info
    print(f"Volume dimensions: {len(volume.shape)}D")
    if len(volume.shape) == 5:
        print("5D array detected - (V, C, Z, Y, X)")
        print(f"Views: {volume.shape[0]}")
        print(f"Channels: {volume.shape[1]}")
        print(f"Z-slices: {volume.shape[2]}")
        print(f"Y (height): {volume.shape[3]}")
        print(f"X (width): {volume.shape[4]}")
    elif len(volume.shape) == 4:
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

    # Adjust chunk shape if needed for higher dimensional data
    if len(volume.shape) > len(custom_chunk_shape):
        # Prepend 1s for extra dimensions (V, C, etc.)
        extra_dims = len(volume.shape) - len(custom_chunk_shape)
        custom_chunk_shape = [1] * extra_dims + custom_chunk_shape
        print(f"Adjusted chunk shape for {len(volume.shape)}D data: {custom_chunk_shape}")

    # Enable Dask cache with 8 GB RAM
    cache = Cache(8 * 1024**3)
    cache.register()

    # Extract voxel sizes from CZI metadata
    czi_metadata, voxel_sizes = extract_czi_metadata(base_path)
    if voxel_sizes:
        print(f"Extracted voxel sizes: x={voxel_sizes['x']:.4f}, y={voxel_sizes['y']:.4f}, z={voxel_sizes['z']:.4f} um")
        # Detect anisotropic voxels and warn
        detect_anisotropic_voxels(voxel_sizes, volume.shape)
    else:
        print("Note: Could not extract voxel sizes from CZI metadata")
        voxel_sizes = {'x': 1.0, 'y': 1.0, 'z': 1.0}

    # Create or open the output Zarr3 store (use CZI axes order for dimension_names)
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
        use_fortran_order=use_fortran_order,
        axes_order=czi_axes_order
    )

    # Add TensorStore context to limit concurrency
    store_spec['context'] = get_tensorstore_context()

    store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Pre-create shard directories if using sharded format
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

    try:
        for domain in chunk_domains:
            # Handle arrays of any dimension dynamically
            slices = tuple(slice(min, max) for (min, max) in zip(domain.inclusive_min, domain.exclusive_max))
            slice_data = volume[slices]

            # Create transaction per chunk to prevent loading all data simultaneously
            with ts.Transaction() as txn:
                task = store[domain].with_transaction(txn).write(slice_data.compute(scheduler='synchronous'))

            tasks.append(task)
            ntasks += 1

        for task in tasks:
            task.result()

    finally:
        # Ensure CZI file handle is properly closed
        if czi_handle:
            try:
                czi_handle.__exit__(None, None, None)
                print("Closed CZI file handle")
            except:
                pass

    # Write OME-Zarr metadata only if using OME structure
    if use_ome_structure:
        print("Writing OME-Zarr metadata...", flush=True)
        try:
            # CZI-specific: Transform CZI XML to OME-XML using XSLT
            ome_xml = None
            if czi_metadata:
                ome_xml = transform_czi_to_ome_xml(czi_metadata)
                if ome_xml:
                    print("Transformed CZI metadata to OME-XML (via XSLT)")

            # Use common function for OME-NGFF metadata structure (same as N5, TIFF, ND2, IMS)
            image_name = os.path.splitext(os.path.basename(base_path))[0]
            zarr3_metadata = create_zarr3_ome_metadata(
                ome_xml=ome_xml,
                array_shape=volume.shape,
                image_name=image_name,
                pixel_sizes=voxel_sizes,
                axes_order=czi_axes_order
            )

            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully", flush=True)

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
