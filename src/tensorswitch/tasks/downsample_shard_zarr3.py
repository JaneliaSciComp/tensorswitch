import tensorstore as ts
import numpy as np
import time
import psutil
from ..utils import get_chunk_domains, create_output_store, commit_tasks, print_processing_info, downsample_spec, zarr3_store_spec, get_input_driver, get_total_chunks_from_store, calculate_anisotropic_downsample_factors, get_tensorstore_context
import os

def process(base_path, output_path, level, start_idx=0, stop_idx=None, downsample=True, use_shard=True, memory_limit=50, custom_shard_shape=None, custom_chunk_shape=None, anisotropic_factors=None, shard_coord=None, **kwargs):

    """
    Downsample and optionally apply sharding to Zarr3 dataset.

    Args:
        anisotropic_factors: Optional list of downsampling factors (e.g., [1, 1, 2, 2] for CZYX)
                            If provided, uses these instead of default [1, 2, 2, 2]
        shard_coord: Optional 3D shard coordinate [z, y, x] - alternative to start_idx/stop_idx for sharded arrays
    """
    if base_path.endswith(f"s{level - 1}") or level == 0:
        zarr_input_path = base_path
    else:
        zarr_input_path = os.path.join(base_path, f"s{level - 1}")

    input_driver = get_input_driver(zarr_input_path)

    zarr_store_spec = {
        'driver': input_driver,
        'kvstore': {'driver': 'file', 'path': zarr_input_path},
        'context': get_tensorstore_context()
    }

    downsampled_saved_path = output_path

    print(f" Downsample: {downsample}, Shard: {use_shard}, Level: {level}")
    print(f"Reading from: {zarr_input_path}")
    print(f"Writing to: {downsampled_saved_path}")

    zarr_store = ts.open(zarr_store_spec).result()

    # Set WebKnossos defaults if not specified
    if custom_chunk_shape is None:
        custom_chunk_shape = [32, 32, 32]
        print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")
    if custom_shard_shape is None and use_shard:
        custom_shard_shape = [1024, 1024, 1024]
        print(f"Using WebKnossos default shard shape: {custom_shard_shape}")

    if downsample and level > 0:
        # Extract dimension_names from the zarr store for proper downsampling
        dimension_names = None
        try:
            # Read dimension_names from zarr.json file directly
            import json
            zarr_json_path = os.path.join(zarr_input_path, 'zarr.json')
            if os.path.exists(zarr_json_path):
                with open(zarr_json_path, 'r') as f:
                    metadata = json.load(f)
                    dimension_names = metadata.get('dimension_names')
                    print(f"Extracted dimension_names from zarr.json: {dimension_names}")
        except Exception as e:
            print(f"Warning: Could not extract dimension_names: {e}")

        if not dimension_names:
            print("Warning: No dimension_names found, defaulting to [2,2,2] downsampling")

        # Auto-calculate anisotropic factors if not provided
        if anisotropic_factors:
            print(f"Using user-provided anisotropic downsampling factors: {anisotropic_factors}")
        else:
            # Try to extract voxel sizes from root zarr.json OME metadata
            voxel_sizes = None
            try:
                # Get root zarr path (remove /s{level-1} if present)
                if base_path.endswith(f"s{level - 1}"):
                    root_zarr_path = os.path.dirname(base_path)
                else:
                    root_zarr_path = base_path

                root_zarr_json = os.path.join(root_zarr_path, 'zarr.json')
                if os.path.exists(root_zarr_json):
                    with open(root_zarr_json, 'r') as f:
                        root_metadata = json.load(f)
                        # Extract voxel sizes from OME metadata
                        ome = root_metadata.get('attributes', {}).get('ome', {})
                        multiscales = ome.get('multiscales', [])
                        if multiscales:
                            datasets = multiscales[0].get('datasets', [])
                            # Find the current input level (s{level-1})
                            current_level_path = f"s{level - 1}"
                            for dataset in datasets:
                                if dataset.get('path') == current_level_path:
                                    transforms = dataset.get('coordinateTransformations', [])
                                    for transform in transforms:
                                        if transform.get('type') == 'scale':
                                            voxel_sizes = transform.get('scale')
                                            # Convert from micrometers to nanometers for display
                                            voxel_sizes_nm = [v * 1000 for v in voxel_sizes]
                                            print(f"Extracted voxel sizes for {current_level_path}: {voxel_sizes_nm} nm")
                                            break
                                    break
            except Exception as e:
                print(f"Warning: Could not extract voxel sizes from OME metadata: {e}")

            # Calculate anisotropic factors based on voxel sizes
            if voxel_sizes and dimension_names:
                anisotropic_factors = calculate_anisotropic_downsample_factors(
                    voxel_sizes,
                    dimension_names,
                    min_ratio=0.5,
                    max_ratio=2.0
                )
                voxel_sizes_nm = [v * 1000 for v in voxel_sizes]
                next_voxel_sizes_nm = [voxel_sizes_nm[i] * anisotropic_factors[i] for i in range(len(voxel_sizes_nm))]
                print(f"Auto-calculated anisotropic factors: {anisotropic_factors}")
                print(f"  Current voxel sizes: {voxel_sizes_nm} nm")
                print(f"  Next level voxel sizes: {next_voxel_sizes_nm} nm")
            else:
                print(f"Using default downsampling with dimension_names: {dimension_names}")

        downsample_spec_dict = downsample_spec(zarr_store_spec, zarr_store.shape, dimension_names, custom_factors=anisotropic_factors)
        downsample_spec_dict['context'] = get_tensorstore_context()
        downsample_store = ts.open(downsample_spec_dict).result()
    else:
        downsample_store = zarr_store

    # Get use_fortran_order from kwargs (defaults to False)
    use_fortran_order = kwargs.get('use_fortran_order', False)

    downsampled_saved_spec = zarr3_store_spec(
        downsampled_saved_path,
        downsample_store.shape,
        downsample_store.dtype.name,
        use_shard,
        level_path=f"s{level}",
        use_ome_structure=True,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape,
        use_fortran_order=use_fortran_order,
        axes_order=dimension_names  # Preserve axes from source level
    )

    # Add TensorStore context to limit concurrency to LSF allocation
    downsampled_saved_spec['context'] = get_tensorstore_context()

    # Create basic output directory structure
    output_array_path = f"{downsampled_saved_path}/s{level}"
    os.makedirs(output_array_path, exist_ok=True)

    downsampled_saved = create_output_store(downsampled_saved_spec)

    # If shard_coord is provided, compute start_idx and stop_idx from 3D shard boundaries
    if shard_coord is not None:
        import math

        print(f"Processing shard coordinate: {shard_coord}")

        # Get output shape and chunk shape
        output_shape = downsampled_saved.shape
        chunk_shape = downsampled_saved.chunk_layout.write_chunk.shape

        # Calculate chunk grid dimensions
        chunk_grid = [
            (output_shape[i] + chunk_shape[i] - 1) // chunk_shape[i]
            for i in range(len(output_shape))
        ]

        # Calculate chunks per shard dimension
        chunks_per_shard_dim = [
            custom_shard_shape[i] // chunk_shape[i]
            for i in range(len(custom_shard_shape))
        ]

        # Base chunk coordinate for this shard
        base_chunk_coord = [
            shard_coord[i] * chunks_per_shard_dim[i]
            for i in range(len(shard_coord))
        ]

        # Generate all chunk indices within this shard
        chunk_indices = []
        for dz in range(chunks_per_shard_dim[0]):
            for dy in range(chunks_per_shard_dim[1]):
                for dx in range(chunks_per_shard_dim[2]):
                    chunk_coord = [
                        base_chunk_coord[0] + dz,
                        base_chunk_coord[1] + dy,
                        base_chunk_coord[2] + dx
                    ]

                    # Skip if chunk is outside data bounds
                    if (chunk_coord[0] >= chunk_grid[0] or
                        chunk_coord[1] >= chunk_grid[1] or
                        chunk_coord[2] >= chunk_grid[2]):
                        continue

                    # Convert 3D chunk coordinate to linear index
                    linear_idx = (
                        chunk_coord[0] * chunk_grid[1] * chunk_grid[2] +
                        chunk_coord[1] * chunk_grid[2] +
                        chunk_coord[2]
                    )
                    chunk_indices.append(linear_idx)

        # Override start_idx and stop_idx to process only these specific chunks
        # We'll use start_idx=0 and stop_idx=total_chunks, but filter by chunk_indices in get_chunk_domains
        start_idx = 0
        stop_idx = math.prod(chunk_grid)
        linear_indices_to_process = chunk_indices

        print(f"Shard {shard_coord}: processing {len(chunk_indices)} chunks (indices {min(chunk_indices)} to {max(chunk_indices)})")
    else:
        linear_indices_to_process = None

    # Check if shard directories already exist (should be pre-created by submit_job)
    # This block is kept as a safety fallback but should rarely execute
    if use_shard and custom_shard_shape:
        # Check if directories already exist
        base_check_path = os.path.join(output_array_path, "c", "0")

        if os.path.exists(base_check_path):
            print("✓ Shard directories already exist (pre-created by submit_job), skipping redundant creation")
        else:
            print("⚠ Shard directories not found, creating them now (this should be rare with --submit)...")
            shard_shape = custom_shard_shape if isinstance(custom_shard_shape, list) else [int(x) for x in custom_shard_shape.split(',')]
            output_shape = list(downsample_store.shape)

            # Adjust shard shape to match array dimensions
            if len(output_shape) == 4 and len(shard_shape) == 3:
                shard_shape = [1] + shard_shape  # CZYX
            elif len(output_shape) == 3 and len(shard_shape) == 2:
                shard_shape = [1] + shard_shape  # CYX

            # Calculate number of shards in each dimension
            num_shards = [((dim_size + shard_size - 1) // shard_size) for dim_size, shard_size in zip(output_shape, shard_shape)]

            # Create all shard parent directories
            base_shard_path = os.path.join(output_array_path, "c")
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
            print(f"Created directory structure for {np.prod(num_shards[:3] if len(num_shards) >= 3 else num_shards)} shard locations")

    #chunk_shape = downsampled_saved.chunk_layout.write_chunk.shape
    #chunk_domains = get_chunk_domains(chunk_shape, downsampled_saved)

    # Use OUTPUT chunk shape for processing, not input chunk shape
    # (input chunks are downsampled, but we write to full-sized output chunks)
    chunk_shape = downsampled_saved.chunk_layout.write_chunk.shape
    print("Shape of downsample_store (input):", downsample_store.shape)
    print("Shape of downsampled_saved (output):", downsampled_saved.shape)
    print("Chunk shape used for processing:", chunk_shape)
    # compute chunk domains based on the downsampled input when goes from s0 to s1
    total_chunks = get_total_chunks_from_store(downsampled_saved, chunk_shape=chunk_shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    tasks = []
    txn = ts.Transaction()
    # Use linear_indices_to_process from shard_coord if available, otherwise use range
    if linear_indices_to_process is None:
        linear_indices_to_process = range(start_idx, stop_idx)
    # Use downsampled_saved (output) for chunk domains since we're writing to it
    for chunk_domain in get_chunk_domains(chunk_shape, downsampled_saved, linear_indices_to_process=linear_indices_to_process):
        task = downsampled_saved[chunk_domain].with_transaction(txn).write(downsample_store[chunk_domain])
        tasks.append(task)
        txn = commit_tasks(tasks, txn, memory_limit)

    if txn.open:
        txn.commit_sync()
    print(f"Downsampling complete for level {level}, chunks {start_idx} to {stop_idx}")