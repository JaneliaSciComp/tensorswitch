import tensorstore as ts
import os
import time
import psutil
import numpy as np
from ..utils import (get_chunk_domains, n5_store_spec, zarr3_store_spec,
                    create_output_store, commit_tasks, get_total_chunks_from_store,
                    fetch_remote_json, create_zarr3_ome_metadata, write_zarr3_group_metadata,
                    write_dual_zarr_metadata, detect_anisotropic_voxels)

def convert(base_path, output_path, level=0, start_idx=0, stop_idx=None,
           memory_limit=50, use_shard=True, use_ome_structure=True,
           custom_shard_shape=None, custom_chunk_shape=None,
           use_v2_encoding=True, create_dual_metadata=False, **kwargs):
    """Convert N5 to Zarr3 with sharding."""

    print(f"N5 to Zarr3 conversion")
    print(f"Input: {base_path}")
    print(f"Output: {output_path}")

    os.umask(0o0002)

    # Get LSF cores for concurrency
    num_cores = int(os.getenv("LSB_DJOB_NUMPROC", 1))
    context = {
        "data_copy_concurrency": {"limit": num_cores},
        "file_io_concurrency": {"limit": num_cores}
    }

    # Open N5 source
    n5_input_spec = n5_store_spec(base_path)
    n5_input_spec['context'] = context
    n5_store = ts.open(n5_input_spec).result()

    shape = n5_store.shape
    dtype = str(n5_store.dtype.numpy_dtype)
    n5_chunk_shape = n5_store.chunk_layout.read_chunk.shape

    print(f"Shape: {shape}, dtype: {dtype}")
    print(f"N5 chunk shape: {n5_chunk_shape}")

    # Read N5 attributes from BOTH level and root directories
    source_attrs = None
    root_attrs = None

    try:
        if base_path.startswith(("http://", "https://", "gs://", "s3://")):
            attr_url = f"{base_path}/attributes.json"
            source_attrs = fetch_remote_json(attr_url)
        else:
            import json

            # Read level attributes (for pixelResolution)
            attr_path = os.path.join(base_path, "attributes.json")
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    source_attrs = json.load(f)
                if 'pixelResolution' in source_attrs:
                    print(f"Pixel resolution: {source_attrs['pixelResolution']}")
                if 'downsamplingFactors' in source_attrs:
                    print(f"Downsampling factors: {source_attrs['downsamplingFactors']}")

            # ALSO read root N5 attributes (for multiscales with axes)
            # base_path format: .../dataset.n5/ch0tp0/s0
            # Need to find: .../dataset.n5/attributes.json
            path_parts = base_path.split(os.sep)
            for i in range(len(path_parts) - 1, -1, -1):
                if path_parts[i].endswith('.n5'):
                    root_n5_path = os.sep.join(path_parts[:i+1])
                    root_attr_path = os.path.join(root_n5_path, "attributes.json")
                    if os.path.exists(root_attr_path):
                        with open(root_attr_path, 'r') as f:
                            root_attrs = json.load(f)
                        print(f"Found root N5 attributes at: {root_attr_path}")
                    break
    except Exception as e:
        print(f"Warning: Could not read N5 attributes: {e}")

    # Extract voxel sizes AND axes from N5 metadata
    voxel_sizes_um = None
    dataset_name = None
    axes_order = None  # NEW: Extract axes from N5

    # Try to extract from multiscales metadata (BigStitcher-Spark style)
    # Look in root_attrs first (for axes), then source_attrs
    attrs_to_check = root_attrs if root_attrs else source_attrs

    if attrs_to_check and 'multiscales' in attrs_to_check:
        try:
            multiscales = attrs_to_check['multiscales'][0]
            dataset_name = multiscales.get('name', 'N5_dataset')

            # Find the dataset for this level
            datasets = multiscales.get('datasets', [])
            if level < len(datasets):
                transform = datasets[level].get('transform', {})
                voxel_sizes_nm = transform.get('scale', None)
                axes_order = transform.get('axes', None)  # NEW: Extract axes

                if voxel_sizes_nm:
                    # Convert nm to micrometers
                    voxel_sizes_um = [v / 1000.0 for v in voxel_sizes_nm]
                    print(f"Voxel sizes (from N5 multiscales): {voxel_sizes_nm} nm = {voxel_sizes_um} µm")

                if axes_order:
                    print(f"Axes order (from N5 multiscales): {axes_order}")
        except Exception as e:
            print(f"Warning: Could not extract metadata from N5 multiscales: {e}")

    # Fallback: Try to extract from pixelResolution attribute
    if voxel_sizes_um is None and source_attrs and 'pixelResolution' in source_attrs:
        try:
            pixel_res = source_attrs['pixelResolution']
            unit = pixel_res.get('unit', 'nm')
            dimensions = pixel_res.get('dimensions', None)

            if dimensions:
                # Convert to micrometers based on unit
                if unit == 'nm' or unit == 'nanometer':
                    voxel_sizes_um = [d / 1000.0 for d in dimensions]
                elif unit == 'um' or unit == 'micrometer' or unit == 'µm':
                    voxel_sizes_um = dimensions
                else:
                    print(f"Warning: Unknown unit '{unit}' in pixelResolution, assuming nm")
                    voxel_sizes_um = [d / 1000.0 for d in dimensions]

                print(f"Voxel sizes (from N5 pixelResolution): {dimensions} {unit} = {voxel_sizes_um} µm")
        except Exception as e:
            print(f"Warning: Could not extract voxel sizes from N5 pixelResolution: {e}")

    # Detect anisotropic voxels and warn (if voxel sizes were found)
    if voxel_sizes_um and len(voxel_sizes_um) >= 3:
        # N5 uses [x, y, z] order, convert to dict for unified function
        voxel_sizes_dict = {'x': voxel_sizes_um[0], 'y': voxel_sizes_um[1], 'z': voxel_sizes_um[2]}
        detect_anisotropic_voxels(voxel_sizes_dict, shape)

    # Set WebKnossos defaults
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
    if custom_shard_shape is None:
        custom_shard_shape = [1024, 1024, 1024]

    print(f"Chunk shape: {custom_chunk_shape}")
    print(f"Shard shape: {custom_shard_shape}")

    # Create Zarr3 output
    level_path = f"s{level}" if use_ome_structure else None

    # Get use_fortran_order from kwargs (defaults to False)
    use_fortran_order = kwargs.get('use_fortran_order', False)

    store_spec = zarr3_store_spec(
        path=output_path,
        shape=shape,
        dtype=dtype,
        use_shard=use_shard,
        level_path=level_path or f"s{level}",
        use_ome_structure=use_ome_structure,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape,
        use_v2_encoding=use_v2_encoding,
        use_fortran_order=use_fortran_order,
        axes_order=axes_order  # NEW: Pass axes from N5
    )
    store_spec['context'] = context

    # SAFETY NET: Check if metadata needs pre-creation (backup for direct calls)
    # This should normally be handled by submit_job() in __main__.py
    if use_fortran_order:
        metadata_path = os.path.join(output_path, level_path or f"s{level}", "zarr.json")
        if not os.path.exists(metadata_path):
            # Metadata wasn't pre-created by main process - create it now
            from tensorswitch.utils import precreate_zarr3_metadata_safely
            print("⚠ Metadata not pre-created - creating now (task-level safety net)")
            precreate_zarr3_metadata_safely(
                output_path=output_path,
                level=level,
                shape=shape,
                dtype=dtype,
                use_shard=use_shard,
                shard_shape=custom_shard_shape,
                chunk_shape=custom_chunk_shape,
                use_ome_structure=use_ome_structure,
                use_fortran_order=use_fortran_order,
                use_v2_encoding=use_v2_encoding,
                axes_order=axes_order
            )

    zarr3_store = ts.open(store_spec, create=True, open=True, delete_existing=False).result()

    # Pre-create shard directories
    if use_shard and custom_shard_shape:
        print("Pre-creating shard directories...")

        if use_ome_structure:
            base_check_path = os.path.join(output_path, level_path or f"s{level}")
        else:
            base_check_path = output_path

        os.makedirs(base_check_path, exist_ok=True)

        shard_shape = custom_shard_shape if isinstance(custom_shard_shape, list) else [int(x) for x in custom_shard_shape.split(',')]
        num_shards = [((dim_size + shard_size - 1) // shard_size) for dim_size, shard_size in zip(shape, shard_shape)]

        print(f"Shards per dimension: {num_shards}")

        total_dirs = 0
        start_time = time.time()

        # Zarr3 with default chunk key encoding uses 'c/' prefix regardless of dimensionality
        # With v2 encoding, it uses dimension indices directly
        # For simplicity and WebKnossos compatibility, always use 'c/' prefix for Zarr3
        base_shard_path = os.path.join(base_check_path, "c")

        if len(num_shards) == 3:  # ZYX (or similar 3D)
            for z_idx in range(num_shards[0]):
                for y_idx in range(num_shards[1]):
                    dir_path = os.path.join(base_shard_path, str(z_idx), str(y_idx))
                    os.makedirs(dir_path, exist_ok=True)
                    total_dirs += 1
        elif len(num_shards) == 4:  # CZYX
            for c in range(num_shards[0]):
                for z in range(num_shards[1]):
                    for y in range(num_shards[2]):
                        dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
                        total_dirs += 1

        elapsed = time.time() - start_time
        print(f"Created {total_dirs} directories in {elapsed:.2f}s")

    # Create root zarr.json with OME-NGFF metadata if using OME structure and this is level 0
    if use_ome_structure and level == 0:
        try:
            print("Creating root zarr.json with OME-NGFF metadata...")

            # Build pixel_sizes dict for create_zarr3_ome_metadata
            pixel_sizes = None
            if voxel_sizes_um and len(voxel_sizes_um) >= 3:
                # N5 uses [x, y, z] order, OME-NGFF also uses [x, y, z] in the scale array
                # But our axes are [z, y, x], so we need to match properly
                pixel_sizes = {
                    'x': voxel_sizes_um[0],
                    'y': voxel_sizes_um[1],
                    'z': voxel_sizes_um[2]
                }
                print(f"Using voxel sizes from N5: x={pixel_sizes['x']}, y={pixel_sizes['y']}, z={pixel_sizes['z']} µm")

            # Create OME metadata structure
            zarr3_metadata = create_zarr3_ome_metadata(
                ome_xml=None,  # N5 doesn't have OME-XML
                array_shape=shape,
                image_name=dataset_name or os.path.basename(output_path),
                pixel_sizes=pixel_sizes,
                axes_order=axes_order  # Pass axes from N5
            )

            # Write root zarr.json
            write_zarr3_group_metadata(output_path, zarr3_metadata)
            print("✓ Root zarr.json created with OME-NGFF metadata")

        except Exception as e:
            print(f"Warning: Could not create root zarr.json: {e}")
            print("  Conversion will continue, but root metadata will be missing")

    # Calculate chunks
    chunk_shape = zarr3_store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(zarr3_store, chunk_shape=chunk_shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print(f"Total chunks: {total_chunks}")
    print(f"Processing: {start_idx} to {stop_idx}")

    linear_indices_to_process = range(start_idx, stop_idx)
    chunk_domains = get_chunk_domains(chunk_shape, zarr3_store, linear_indices_to_process=linear_indices_to_process)

    # Process chunks
    print(f"Converting {len(linear_indices_to_process)} chunks...")

    tasks = []
    txn = ts.Transaction()
    processed = 0
    start_time = time.time()
    last_report = start_time

    for idx, chunk_domain in enumerate(chunk_domains, start=start_idx):
        try:
            array = n5_store[chunk_domain].read().result()
            task = zarr3_store[chunk_domain].with_transaction(txn).write(array)
            tasks.append(task)
            processed += 1

            txn = commit_tasks(tasks, txn, memory_limit)

            # Progress every 100 chunks or 30s
            now = time.time()
            if processed % 100 == 0 or (now - last_report) > 30:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(linear_indices_to_process) - processed) / rate if rate > 0 else 0
                mem = psutil.virtual_memory().percent
                print(f"{processed}/{len(linear_indices_to_process)} | {rate:.1f} chunks/s | ETA {eta/60:.1f}m | Mem {mem:.1f}%")
                last_report = now

        except Exception as e:
            print(f"Warning: Skipping chunk {idx}: {e}")
            continue

    if txn.open:
        txn.commit_sync()

    elapsed = time.time() - start_time
    print(f"Complete: {processed} chunks in {elapsed:.1f}s ({processed/elapsed:.1f} chunks/s)")

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


def process(base_path, output_path, level=0, start_idx=0, stop_idx=None,
           memory_limit=50, use_shard=True, use_ome_structure=True,
           custom_shard_shape=None, custom_chunk_shape=None,
           use_v2_encoding=True, create_dual_metadata=False, **kwargs):
    """Alias for convert()."""
    return convert(base_path=base_path, output_path=output_path, level=level,
                  start_idx=start_idx, stop_idx=stop_idx, memory_limit=memory_limit,
                  use_shard=use_shard, use_ome_structure=use_ome_structure,
                  custom_shard_shape=custom_shard_shape,
                  custom_chunk_shape=custom_chunk_shape,
                  use_v2_encoding=use_v2_encoding, create_dual_metadata=create_dual_metadata,
                  **kwargs)
