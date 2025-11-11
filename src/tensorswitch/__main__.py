import argparse
import subprocess
import os
import time
import tensorstore as ts
import numpy as np
import sys
import re

if not __package__:
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)

from . import tasks
from .utils import (get_total_chunks, downsample_spec, zarr3_store_spec, get_chunk_domains,
                    estimate_total_chunks_for_tiff, get_input_driver, get_total_chunks_from_store,
                    load_tiff_stack, load_nd2_stack, load_ims_stack, update_ome_metadata_if_needed,
                    calculate_num_multiscale_levels, calculate_anisotropic_downsample_factors,
                    generate_auto_multiscale, calculate_pyramid_plan, generate_cli_coordinator_script,
                    precreate_shard_directories as precreate_shard_directories_universal)
from .tasks import (downsample_shard_zarr3, n5_to_n5, n5_to_zarr2, n5_to_zarr3_s0, tiff_to_zarr3_s0,
                    nd2_to_zarr3_s0, ims_to_zarr3_s0, nd2_to_zarr2_s0, ims_to_zarr2_s0,
                    tiff_to_zarr2_s0, downsample_zarr2, precomputed_to_n5)
from .dask_utils import submit_dask_job, submit_dask_wrapper_job

# Set umask to allow group write access
os.umask(0o0002)


def precreate_shard_directories(args, volume_shape, use_v2_encoding=True):
    """
    Pre-create all shard directories before job submission to avoid race conditions.
    This function runs once on the submission node, not on each worker.
    """
    # Only create directories if sharding is enabled
    if not bool(args.use_shard):
        print("Sharding disabled, skipping directory pre-creation")
        return

    if not args.custom_shard_shape:
        print("No custom shard shape specified, skipping directory pre-creation")
        return

    print("\n" + "="*80)
    print("PRE-CREATING SHARD DIRECTORY STRUCTURE")
    print("="*80)

    # Parse shard shape
    shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
    output_shape = list(volume_shape)

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
    num_shards = [((dim_size + shard_size - 1) // shard_size)
                  for dim_size, shard_size in zip(output_shape, shard_shape)]

    print(f"Data shape: {output_shape}")
    print(f"Shard shape: {shard_shape}")
    print(f"Number of shards per dimension: {num_shards}")

    # Determine base path
    use_ome_structure = bool(args.use_ome_structure)
    if use_ome_structure:
        base_shard_path = os.path.join(args.output_path, "s0", "c")
    else:
        base_shard_path = os.path.join(args.output_path, "c")

    # Create all shard parent directories
    total_dirs = 0
    import time
    start_time = time.time()

    if len(num_shards) == 4:  # CZYX
        for c in range(num_shards[0]):
            for z in range(num_shards[1]):
                for y in range(num_shards[2]):
                    dir_path = os.path.join(base_shard_path, str(c), str(z), str(y))
                    os.makedirs(dir_path, exist_ok=True)
                    total_dirs += 1
    elif len(num_shards) == 3:  # CYX or ZYX
        for dim0 in range(num_shards[0]):
            for dim1 in range(num_shards[1]):
                dir_path = os.path.join(base_shard_path, str(dim0), str(dim1))
                os.makedirs(dir_path, exist_ok=True)
                total_dirs += 1
    elif len(num_shards) == 5:  # TCZYX
        for t in range(num_shards[0]):
            for c in range(num_shards[1]):
                for z in range(num_shards[2]):
                    for y in range(num_shards[3]):
                        dir_path = os.path.join(base_shard_path, str(t), str(c), str(z), str(y))
                        os.makedirs(dir_path, exist_ok=True)
                        total_dirs += 1

    elapsed = time.time() - start_time
    print(f"✓ Created {total_dirs} shard directories in {elapsed:.2f} seconds")
    print(f"✓ Base path: {base_shard_path}")
    print("="*80 + "\n")


def get_total_chunks_for_task(args, use_v2_encoding=True):
    """Calculate total chunks for a given task."""
    # Parse custom shard and chunk shapes
    custom_shard_shape = None
    if args.custom_shard_shape:
        custom_shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]

    custom_chunk_shape = None
    if args.custom_chunk_shape:
        custom_chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

    input_driver = "n5" if args.base_path.startswith("http") else get_input_driver(args.base_path)

    if args.task == "n5_to_zarr3_s0":
        # Open N5 to get shape
        from .utils import n5_store_spec
        n5_store = ts.open(n5_store_spec(args.base_path)).result()
        shape = n5_store.shape
        dtype = str(n5_store.dtype.numpy_dtype)

        store_spec = zarr3_store_spec(
            path=args.output_path,
            shape=shape,
            dtype=dtype,
            use_shard=bool(args.use_shard),
            level_path=f"s{args.level}",
            use_ome_structure=bool(args.use_ome_structure),
            custom_shard_shape=custom_shard_shape,
            custom_chunk_shape=custom_chunk_shape,
            use_v2_encoding=use_v2_encoding
        )
        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()

        # For sharded arrays, count by write_chunk (shard shape), not read_chunk (inner chunks)
        # write_chunk = shard size (1024³), read_chunk = inner chunk size (32³)
        total_chunks = get_total_chunks_from_store(temp_store, chunk_shape=temp_store.chunk_layout.write_chunk.shape)

    elif args.task == "tiff_to_zarr3_s0" and input_driver == "tiff":
        volume = load_tiff_stack(args.base_path)

        # Apply WebKnossos defaults if not specified (same as in process() function)
        if custom_chunk_shape is None:
            custom_chunk_shape = [32, 32, 32]
            print(f"Using WebKnossos default chunk shape: {custom_chunk_shape}")
        if custom_shard_shape is None and bool(args.use_shard):
            custom_shard_shape = [1024, 1024, 1024]
            print(f"Using WebKnossos default shard shape: {custom_shard_shape}")

        store_spec = zarr3_store_spec(
            path=args.output_path,
            shape=volume.shape,
            dtype=str(volume.dtype),
            use_shard=bool(args.use_shard),
            use_ome_structure=bool(args.use_ome_structure),
            custom_shard_shape=custom_shard_shape,
            custom_chunk_shape=custom_chunk_shape,
            use_v2_encoding=use_v2_encoding
        )
        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()

        # For sharded arrays, count by write_chunk (shard shape), not read_chunk (inner chunks)
        # This matches the N5 approach and correctly handles WebKnossos defaults
        total_chunks = get_total_chunks_from_store(temp_store, chunk_shape=temp_store.chunk_layout.write_chunk.shape)

    elif args.task == "nd2_to_zarr3_s0":
        volume = load_nd2_stack(args.base_path)
        store_spec = zarr3_store_spec(
            path=args.output_path,
            shape=volume.shape,
            dtype=str(volume.dtype),
            use_shard=bool(args.use_shard),
            use_ome_structure=bool(args.use_ome_structure),
            custom_shard_shape=custom_shard_shape,
            custom_chunk_shape=custom_chunk_shape,
            use_v2_encoding=use_v2_encoding
        )
        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()

        # Handle custom shapes for chunk counting
        if custom_shard_shape:
            if len(volume.shape) == 3 and len(custom_shard_shape) == 2:
                chunk_shape_for_counting = [1] + custom_shard_shape  # 2D images: CYX with YX shards
            elif len(volume.shape) == 4 and len(custom_shard_shape) == 3:
                chunk_shape_for_counting = [1] + custom_shard_shape
            elif len(volume.shape) == 4 and len(custom_shard_shape) == 2:
                chunk_shape_for_counting = [1, 1] + custom_shard_shape  # 2D images: CZYX with YX shards
            elif len(volume.shape) == 5 and len(custom_shard_shape) == 3:
                chunk_shape_for_counting = [1, 1] + custom_shard_shape
            elif len(volume.shape) == 5 and len(custom_shard_shape) == 2:
                chunk_shape_for_counting = [1, 1, 1] + custom_shard_shape  # 2D images: TCZYX with YX shards
            else:
                chunk_shape_for_counting = custom_shard_shape
            total_chunks = get_total_chunks_from_store(temp_store, chunk_shape_for_counting)
        else:
            total_chunks = get_total_chunks_from_store(temp_store)

    elif args.task == "ims_to_zarr3_s0":
        volume, h5_file = load_ims_stack(args.base_path)
        h5_file.close()
        store_spec = zarr3_store_spec(
            path=args.output_path,
            shape=volume.shape,
            dtype=str(volume.dtype),
            use_shard=bool(args.use_shard),
            level_path="s0",
            use_ome_structure=bool(args.use_ome_structure),
            use_v2_encoding=use_v2_encoding
        )
        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()
        total_chunks = get_total_chunks_from_store(temp_store)

    elif args.task == "precomputed_to_n5":
        # Import precomputed-specific utilities
        from .utils import fetch_precomputed_info, precomputed_store_spec

        # Fetch Precomputed metadata
        info = fetch_precomputed_info(args.base_path)
        scale = info['scales'][args.level]

        # Open Precomputed scale to get actual shape
        store = ts.open(precomputed_store_spec(args.base_path, scale['key'])).result()

        # Calculate chunks based on requested chunk size (default 128^3)
        chunk_shape_3d = custom_chunk_shape if custom_chunk_shape else [128, 128, 128]

        # Precomputed is 4D [X, Y, Z, C], adjust chunk shape to match
        if len(store.shape) == 4:
            chunk_shape = chunk_shape_3d + [1]  # Add channel dimension
        else:
            chunk_shape = chunk_shape_3d

        total_chunks = get_total_chunks_from_store(store, chunk_shape=chunk_shape)

    else:
        if args.downsample and args.level > 0:
            prev_level = args.level - 1
            if args.base_path.endswith(f"s{prev_level}"):
                input_path = args.base_path
            else:
                input_path = os.path.join(args.base_path, f"s{prev_level}")

            # Open input store to get input shape
            downsample_store = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": input_path}}).result()
            input_shape = downsample_store.shape

            # Parse anisotropic factors (default to [2,2,2] if not provided)
            if hasattr(args, 'anisotropic_factors') and args.anisotropic_factors:
                aniso_factors = [int(x) for x in args.anisotropic_factors.split(',')]
            else:
                aniso_factors = [2, 2, 2]

            # Calculate OUTPUT shape (input / factors)
            import math
            output_shape = [
                math.ceil(input_shape[i] / aniso_factors[i])
                for i in range(len(input_shape))
            ]

            # Use custom_chunk_shape if provided, otherwise use input's chunk shape
            if custom_chunk_shape:
                chunk_shape = custom_chunk_shape
            else:
                chunk_shape = downsample_store.chunk_layout.read_chunk.shape

            # Calculate total chunks based on OUTPUT shape, not input shape!
            total_chunks = math.prod([
                (output_shape[i] + chunk_shape[i] - 1) // chunk_shape[i]
                for i in range(len(output_shape))
            ])
        else:
            total_chunks = get_total_chunks(args.base_path)

    print(f"Calculated total chunks: {total_chunks} for task: {args.task}")
    return total_chunks

def submit_job(args, use_v2_encoding=True):
    """Handles LSF cluster job submission for different tasks."""
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Use the centralized function to calculate total chunks
    total_chunks = get_total_chunks_for_task(args, use_v2_encoding)

    print(f"The total number of chunks is {total_chunks} with downsample={args.downsample} and level={args.level}")

    # Get volume shape for directory pre-creation
    volume_shape = None
    if args.task == "tiff_to_zarr3_s0":
        volume = load_tiff_stack(args.base_path)
        volume_shape = volume.shape
    elif args.task == "nd2_to_zarr3_s0":
        volume = load_nd2_stack(args.base_path)
        volume_shape = volume.shape
    elif args.task == "ims_to_zarr3_s0":
        volume = load_ims_stack(args.base_path)
        volume_shape = volume.shape
    elif args.task == "downsample_shard_zarr3":
        # For downsampling, get shape from the source zarr and calculate output shape
        store = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file', 'path': args.base_path}}).result()
        source_shape = store.shape

        # Calculate downsampled output shape
        if args.anisotropic_factors:
            factors = [int(x) for x in args.anisotropic_factors.split(',')]
        else:
            # Default 2x downsampling in all spatial dimensions
            factors = [2, 2, 2]

        # Apply downsampling factors to calculate output shape
        if len(source_shape) == 3:  # ZYX
            volume_shape = tuple(s // f for s, f in zip(source_shape, factors))
        elif len(source_shape) == 4:  # CZYX - don't downsample channel dimension
            volume_shape = (source_shape[0],) + tuple(s // f for s, f in zip(source_shape[1:], factors))
        else:
            volume_shape = source_shape  # Fallback

        print(f"Downsample: source shape {source_shape} -> output shape {volume_shape} (factors: {factors})")

    # Pre-create all shard directories BEFORE submitting any jobs
    if volume_shape is not None and args.task in ["tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0"]:
        precreate_shard_directories(args, volume_shape, use_v2_encoding)
    elif volume_shape is not None and args.task == "downsample_shard_zarr3":
        # For downsample tasks, use the universal function from utils
        if bool(args.use_shard) and args.custom_shard_shape:
            shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
            precreate_shard_directories_universal(
                output_path=args.output_path,
                level=args.level,
                output_shape=list(volume_shape),
                shard_shape=shard_shape,
                use_ome_structure=bool(args.use_ome_structure)
            )
    elif args.task == "n5_to_zarr3_s0":
        # For N5 conversion, pre-create metadata and directories if F-order or sharding is enabled
        if bool(args.use_shard) and args.custom_shard_shape:
            from tensorswitch.utils import get_kvstore_spec, precreate_zarr3_metadata_safely

            # Read N5 metadata to get shape and dtype
            print("Reading N5 metadata for pre-creation...")
            n5_kvstore = get_kvstore_spec(args.base_path)
            n5_store = ts.open({
                'driver': 'n5',
                'kvstore': n5_kvstore,
                'recheck_cached_data': False,
                'recheck_cached_metadata': False
            }).result()

            volume_shape = n5_store.shape
            n5_dtype = n5_store.dtype.name
            n5_store = None  # Close

            # Parse shapes
            shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
            chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

            # Pre-create shard directories
            precreate_shard_directories_universal(
                output_path=args.output_path,
                level=args.level,
                output_shape=list(volume_shape),
                shard_shape=shard_shape,
                use_ome_structure=bool(args.use_ome_structure)
            )

            # Pre-create zarr.json metadata (PRIMARY PROTECTION - happens ONCE before workers)
            precreate_zarr3_metadata_safely(
                output_path=args.output_path,
                level=args.level,
                shape=volume_shape,
                dtype=n5_dtype,
                use_shard=True,
                shard_shape=shard_shape,
                chunk_shape=chunk_shape,
                use_ome_structure=bool(args.use_ome_structure),
                use_fortran_order=bool(args.use_fortran_order),
                use_v2_encoding=use_v2_encoding
            )

    # Check if using Dask JobQueue
    if hasattr(args, 'use_dask_jobqueue') and args.use_dask_jobqueue:
        print("Using Dask JobQueue submission")
        return submit_dask_wrapper_job(args, total_chunks)

    # Traditional LSF bsub method
    print("Using traditional LSF bsub submission")

    # Calculate chunks per shard for shard-aligned distribution
    chunks_per_shard = 1  # Default: no sharding
    total_3d_shards = None  # Will be calculated if using 3D sharding

    # Note: n5_to_zarr3_s0 already processes by shard domains (write_chunk), not inner chunks
    # So skip shard-aligned calculation to avoid double-counting (598 shards != 598/32768 shards)
    if args.task != "n5_to_zarr3_s0" and bool(args.use_shard) and args.custom_shard_shape and args.custom_chunk_shape:
        # Parse shard and chunk shapes
        shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
        chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

        # Calculate chunks per shard in each dimension
        chunks_per_shard = 1
        for s, c in zip(shard_shape, chunk_shape):
            chunks_per_shard *= (s // c)

        print(f"Shard-aligned distribution: {chunks_per_shard} chunks per shard")

        # Calculate 3D shard count from output shape (more accurate than linear chunk count)
        if volume_shape:
            import math
            total_3d_shards = math.prod([
                (volume_shape[i] + shard_shape[i] - 1) // shard_shape[i]
                for i in range(len(volume_shape))
            ])
            print(f"3D shard count: {total_3d_shards} (from output shape {volume_shape})")

    # Calculate total shards
    # Use 3D shard count if available (more accurate for spatial distribution)
    # Otherwise fall back to linear chunk-based count
    if total_3d_shards:
        total_shards = total_3d_shards
    else:
        total_shards = (total_chunks + chunks_per_shard - 1) // chunks_per_shard

    # The number of volumes can be at most total_shards
    num_volumes = min(total_shards, args.num_volumes)
    print(f"Distributing {total_chunks:,} chunks across {num_volumes} workers ({total_shards} shards total)")

    # For 3D sharded arrays, distribute by explicit 3D shard coordinates
    # to ensure no overlap (linear distribution doesn't map cleanly to 3D shards)
    if total_3d_shards and volume_shape:
        import math

        # Parse shapes
        shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
        chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

        # Calculate 3D shard grid dimensions
        shard_grid = [
            (volume_shape[i] + shard_shape[i] - 1) // shard_shape[i]
            for i in range(len(volume_shape))
        ]

        # Calculate chunk grid dimensions
        chunk_grid = [
            (volume_shape[i] + chunk_shape[i] - 1) // chunk_shape[i]
            for i in range(len(volume_shape))
        ]

        print(f"Using 3D shard-based distribution: shard grid {shard_grid}, chunk grid {chunk_grid}")

        # Generate all 3D shard coordinates (z, y, x order)
        all_shard_coords = []
        for z in range(shard_grid[0]):
            for y in range(shard_grid[1]):
                for x in range(shard_grid[2]):
                    all_shard_coords.append([z, y, x])

        # Distribute shards among workers
        shards_per_worker = len(all_shard_coords) // num_volumes

        for i in range(num_volumes):
            # Get this worker's assigned 3D shard coordinates
            start_shard_idx = i * shards_per_worker
            if i == num_volumes - 1:
                end_shard_idx = len(all_shard_coords)
            else:
                end_shard_idx = (i + 1) * shards_per_worker

            assigned_shards = all_shard_coords[start_shard_idx:end_shard_idx]

            # Calculate chunk ranges for all assigned shards
            chunk_indices = []
            for shard_coord in assigned_shards:
                # Calculate chunk range within this 3D shard
                # Each shard contains: chunks_per_shard_dim[i] chunks in dimension i
                chunks_per_shard_dim = [
                    shard_shape[j] // chunk_shape[j]
                    for j in range(len(shard_shape))
                ]

                # Base chunk coordinate for this shard
                base_chunk_coord = [
                    shard_coord[j] * chunks_per_shard_dim[j]
                    for j in range(len(shard_coord))
                ]

                # Generate all chunk coordinates within this shard
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

            # Get min/max chunk indices for this worker
            if chunk_indices:
                start_idx = min(chunk_indices)
                stop_idx = max(chunk_indices) + 1
            else:
                start_idx = 0
                stop_idx = 0

            shard_desc = f"{len(assigned_shards)} shard(s): {assigned_shards[0]}" + (f"...{assigned_shards[-1]}" if len(assigned_shards) > 1 else "")
            print(f"vol{i}: {shard_desc}")

            # Extract setup number from base_path if any
            # Build job name with optional prefix
            prefix = f"{args.job_prefix}_" if args.job_prefix else ""
            match = re.search(r"setup(\d+)", args.base_path)
            if match:
                setup_number = match.group(1)
                job_name = f"{prefix}{args.task}_setup{setup_number}_s{args.level}_vol{i}"
            else:
                job_name = f"{prefix}{args.task}_s{args.level}_vol{i}"


            command = [
                "bsub",
                "-J", job_name,
                "-n", args.cores,
                "-W", args.wall_time,
                "-P", args.project,
                "-g", "/scicompsoft/chend/tensorstore",
                "-o", f"{output_dir}/output__{job_name}_%J.log",
                "-e", f"{output_dir}/error__{job_name}_%J.log",
                sys.executable,
                "-m", "tensorswitch",
            ]

            string_args = ["task", "base_path", "output_path", "custom_shard_shape", "custom_chunk_shape", "dual_zarr_approach", "anisotropic_factors"]
            int_args = ["level", "downsample", "use_shard", "use_ome_structure", "memory_limit", "use_fortran_order"]
            boolean_flags = ["use_dask_jobqueue"]

            # Add string and integer arguments
            for arg in string_args + int_args:
                value = getattr(args, arg)
                if value is not None and str(value) != "None":
                    command += ["--"+arg, str(value)]

            # Add boolean flags (only if True)
            for arg in boolean_flags:
                if getattr(args, arg, False):
                    command += ["--"+arg]

            # Pass shard coordinate instead of chunk indices for 3D sharded distribution
            # Submit one job per shard coordinate to ensure all shards are processed
            for shard_idx, shard_coord in enumerate(assigned_shards):
                shard_coord_str = ",".join(map(str, shard_coord))
                shard_command = command + ["--shard_coord", shard_coord_str]

                # Update job name to include shard index
                shard_job_name = f"{job_name}_shard{shard_idx}"
                shard_command[shard_command.index(job_name)] = shard_job_name

                # print(shard_command)  # Uncomment for debugging

                subprocess.run(shard_command)
                print(f"Submitted {shard_job_name}, volume={i}, shard={shard_coord}")

                # Small delay between job submissions to avoid overwhelming the scheduler
                time.sleep(0.05)

    else:
        # Fall back to linear distribution for non-3D or non-sharded cases
        for i in range(num_volumes):
            # Distribute by complete shards
            start_shard = i * (total_shards // num_volumes)
            if i == num_volumes - 1:
                stop_shard = total_shards
            else:
                stop_shard = (i + 1) * (total_shards // num_volumes)

            # Convert shard indices to chunk indices
            start_idx = start_shard * chunks_per_shard
            stop_idx = min(stop_shard * chunks_per_shard, total_chunks)

            print(f"vol{i}: shards {start_shard}–{stop_shard} → chunks {start_idx}–{stop_idx}")

            # Extract setup number from base_path if any
            # Build job name with optional prefix
            prefix = f"{args.job_prefix}_" if args.job_prefix else ""
            match = re.search(r"setup(\d+)", args.base_path)
            if match:
                setup_number = match.group(1)
                job_name = f"{prefix}{args.task}_setup{setup_number}_s{args.level}_vol{i}"
            else:
                job_name = f"{prefix}{args.task}_s{args.level}_vol{i}"

            command = [
                "bsub",
                "-J", job_name,
                "-n", args.cores,
                "-W", args.wall_time,
                "-P", args.project,
                "-g", "/scicompsoft/chend/tensorstore",
                "-o", f"{output_dir}/output__{job_name}_%J.log",
                "-e", f"{output_dir}/error__{job_name}_%J.log",
                sys.executable,
                "-m", "tensorswitch",
            ]

            string_args = ["task", "base_path", "output_path", "custom_shard_shape", "custom_chunk_shape", "dual_zarr_approach", "anisotropic_factors"]
            int_args = ["level", "downsample", "use_shard", "use_ome_structure", "memory_limit", "use_fortran_order"]
            boolean_flags = ["use_dask_jobqueue"]

            # Add string and integer arguments
            for arg in string_args + int_args:
                value = getattr(args, arg)
                if value is not None and str(value) != "None":
                    command += ["--"+arg, str(value)]

            # Add boolean flags (only if True)
            for arg in boolean_flags:
                if getattr(args, arg, False):
                    command += ["--"+arg]

            command += ["--start_idx", str(start_idx)]
            if stop_idx is not None:
                command += ["--stop_idx", str(stop_idx)]

            print(command)

            subprocess.run(command)
            print(f"Submitted {job_name}, volume={i}, chunks={start_idx}-{stop_idx}")

            # Small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Manager")
    parser.add_argument("--task", required=True, choices=["n5_to_zarr2", "n5_to_zarr3_s0", "n5_to_n5", "downsample_shard_zarr3", "tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0", "nd2_to_zarr2_s0", "ims_to_zarr2_s0", "tiff_to_zarr2_s0", "downsample_zarr2", "precomputed_to_n5"])
    parser.add_argument("--base_path", required=True, help="Input dataset path")
    parser.add_argument("--output_path", required=True, help="Output dataset path (only needed for conversions)")
    parser.add_argument("--level", type=int, default=0, help="Levels to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Chunk start index (for local processing)")
    parser.add_argument("--stop_idx", type=int, help="Chunk stop index (for local processing)")
    parser.add_argument("--shard_coord", type=str, help="3D shard coordinate as comma-separated values (e.g., '5,0,0') - alternative to start/stop_idx for sharded arrays")
    parser.add_argument("--num_volumes", type=int, default=8, help="Number of volumes per level (for cluster jobs)")
    parser.add_argument("--downsample", type=int, default=1, choices=[0, 1], help="Enable downsampling (default: 1)")
    parser.add_argument("--use_shard", type=int, default=0, choices=[0, 1], help="Use sharded format (for downsample)")
    parser.add_argument("--use_ome_structure", type=int, default=1, choices=[0, 1], help="Use OME-ZARR multiscale structure (s0 subdirectory)")
    parser.add_argument("--use_fortran_order", type=int, default=0, choices=[0, 1], help="Use Fortran (F) order instead of C order (adds transpose codec)")
    parser.add_argument("--submit", action="store_true", help="Submit to the cluster scheduler")
    parser.add_argument("--memory_limit", type=int, default=50, help="memory limit percentage" )
    parser.add_argument("--project", default="None", help="Project to charge")
    parser.add_argument("--cores", type=str, default="2", help="Number of cores for LSF job (-n flag)")
    parser.add_argument("--wall_time", type=str, default="1:00", help="Wall time for LSF job (-W flag)")
    parser.add_argument("--job_prefix", type=str, default="", help="Prefix for job names (e.g., 'ahrens_', 'lavis_brain_')")
    parser.add_argument("--custom_shard_shape", type=str, help="Custom shard shape as comma-separated values (e.g., '128,576,576')")
    parser.add_argument("--custom_chunk_shape", type=str, help="Custom chunk shape as comma-separated values (e.g., '32,32,32')")
    parser.add_argument("--use_dask_jobqueue", action="store_true", help="Use Dask JobQueue instead of direct LSF submission")
    parser.add_argument("--dual_zarr_approach", type=str, default="none", choices=["v2_chunks", "v3_chunks", "none"], help="Dual zarr v2/v3 compatibility approach: none (default, pure zarr v3), v2_chunks (colocated metadata), or v3_chunks (.zarray in c/ directory)")
    parser.add_argument("--auto_multiscale", action="store_true", help="Automatically generate all multiscale levels until thumbnail-sized (uses Yurii Zubov's anisotropic algorithm)")
    parser.add_argument("--min_array_nbytes", type=int, default=None, help="Stop pyramid when array size < this in bytes (default: chunk_nbytes from metadata)")
    parser.add_argument("--min_array_shape", type=str, default=None, help="Stop pyramid when all dims < this, format: '32,64,128' (default: chunk_shape from metadata)")
    parser.add_argument("--anisotropic_factors", type=str, default=None, help="Anisotropic downsampling factors, format: '1,2,2' (used by auto-multiscale coordinator)")

    args = parser.parse_args()

    # Validate dual format and sharding compatibility
    dual_format_requested = args.dual_zarr_approach in ["v2_chunks", "v3_chunks"]
    if dual_format_requested and bool(args.use_shard):
        print("ERROR: Cannot use dual format with sharding.")
        print("Choose either --dual_zarr_approach v2_chunks or --use_shard 1")
        sys.exit(1)

    # Auto-disable sharding when dual format is requested
    if dual_format_requested:
        args.use_shard = 0
        print(f"INFO: Sharding disabled for dual format ({args.dual_zarr_approach})")

    # Determine chunk encoding and dual metadata creation
    use_v2_encoding = (args.dual_zarr_approach == "v2_chunks")
    create_dual_metadata = (args.dual_zarr_approach != "none")

    # Parse custom shard and chunk shapes
    custom_shard_shape = None
    if args.custom_shard_shape and args.custom_shard_shape != "None":
        custom_shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]

    custom_chunk_shape = None
    if args.custom_chunk_shape and args.custom_chunk_shape != "None":
        custom_chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

    # Parse anisotropic factors if provided
    anisotropic_factors = None
    if args.anisotropic_factors and args.anisotropic_factors != "None":
        anisotropic_factors = [int(x) for x in args.anisotropic_factors.split(',')]

    # Check for auto_multiscale FIRST (before submit check)
    if args.auto_multiscale:
            # Phase 1: Auto-multiscale from existing s0 (downsample tasks only)
            if args.task in ["downsample_shard_zarr3", "downsample_zarr2"]:
                print("\nAuto-multiscale from s0 (cluster mode)")
                print(f"Base path (s0): {args.base_path}")
                print(f"Output path: {args.output_path}")

                # Parse min_array_shape if provided
                min_array_shape = None
                if args.min_array_shape:
                    min_array_shape = [int(x) for x in args.min_array_shape.split(',')]

                # Calculate pyramid plan from s0
                try:
                    pyramid_plan = calculate_pyramid_plan(
                        args.base_path,
                        min_array_nbytes=args.min_array_nbytes,
                        min_array_shape=min_array_shape
                    )
                except Exception as e:
                    print(f"ERROR: Failed to calculate pyramid plan: {e}")
                    return

                print(f"Pyramid plan: {pyramid_plan['num_levels']} levels needed")
                for level_info in pyramid_plan['pyramid_plan']:
                    print(f"  s{level_info['level']}: factor {level_info['factor']}")

                # Generate coordinator script
                script_content = generate_cli_coordinator_script(args.output_path, pyramid_plan, args)

                # Write script to output directory
                output_dir = os.path.dirname(args.output_path) if args.output_path else os.path.expanduser("~/tensorswitch_jobs")
                os.makedirs(output_dir, exist_ok=True)

                timestamp = int(time.time())
                script_path = os.path.join(output_dir, f"coordinator_autoscale_{timestamp}.sh")

                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)

                print(f"\nCoordinator script: {script_path}")

                # Submit coordinator job if --submit is used
                if args.submit:
                    if args.project == "None":
                        raise ValueError(f"Project cannot be None when submitting.")

                    job_name = f"tensorswitch_autoscale"
                    log_file = os.path.join(output_dir, f"coordinator_{timestamp}.log")
                    err_file = os.path.join(output_dir, f"coordinator_{timestamp}.err")

                    cmd = [
                        "bsub",
                        "-J", job_name,
                        "-n", "1",
                        "-W", "24:00",
                        "-P", args.project,
                        "-o", log_file,
                        "-e", err_file,
                        script_path
                    ]

                    print(f"\nSubmitting coordinator: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        print(f"Coordinator submitted")
                        print(f"Monitor: tail -f {log_file}")
                        # Extract job ID
                        import re
                        match = re.search(r'Job <(\d+)>', result.stdout)
                        if match:
                            print(f"Job ID: {match.group(1)}")
                    else:
                        print(f"ERROR: Coordinator submission failed")
                        print(result.stderr)
                else:
                    print("Dry run mode (no --submit). To submit, add --submit --project <project_name>")

                return

            # Phase 2: Not yet implemented (s0 creation tasks)
            elif args.task in ["tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0", "tiff_to_zarr2_s0", "nd2_to_zarr2_s0", "ims_to_zarr2_s0"]:
                print("\nAuto-multiscale from raw input (not yet implemented)")
                print("Current workflow:")
                print("  1. Submit s0 job first (without --auto_multiscale)")
                print("  2. After s0 completes, run:")
                print(f"     python -m tensorswitch --task downsample_shard_zarr3 \\")
                print(f"       --base_path {args.output_path}/s0 \\")
                print(f"       --output_path {args.output_path} \\")
                print(f"       --auto_multiscale --submit --project {args.project}")
                return

    # Regular submission (non-auto_multiscale)
    if args.submit:
        if args.project == "None":
            raise ValueError(f"Project cannot be None when submitting.")
        submit_job(args, use_v2_encoding)

    else:
        if hasattr(args, 'use_dask_jobqueue') and args.use_dask_jobqueue:
            print("Using Dask JobQueue submission")
            total_chunks = get_total_chunks_for_task(args, use_v2_encoding)
            success = submit_dask_wrapper_job(args, total_chunks)
            if not success:
                print("Dask wrapper job submission failed")
                return
            else:
                print("Dask wrapper job submitted successfully")
                return

        if args.task == "n5_to_n5":
            n5_to_n5.convert(args.base_path, args.output_path, args.num_volumes, args.level, args.start_idx, args.stop_idx, args.memory_limit, custom_chunk_shape)
        elif args.task == "n5_to_zarr2":
            n5_to_zarr2.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit)
        elif args.task == "n5_to_zarr3_s0":
            n5_to_zarr3_s0.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit, bool(args.use_shard), bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape, use_v2_encoding, use_fortran_order=bool(args.use_fortran_order))
        elif args.task == "downsample_shard_zarr3":
            # Parse shard_coord if provided
            shard_coord_list = None
            if args.shard_coord:
                shard_coord_list = [int(x) for x in args.shard_coord.split(',')]

            downsample_shard_zarr3.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.downsample), bool(args.use_shard), args.memory_limit, custom_shard_shape, custom_chunk_shape, anisotropic_factors, shard_coord=shard_coord_list, use_fortran_order=bool(args.use_fortran_order))

            # Store downsampling factors in metadata if this is the first worker (shard [0,0,0])
            # This enables precise voxel size calculation in update_ome_metadata_if_needed
            if shard_coord_list is None or shard_coord_list == [0, 0, 0]:
                if anisotropic_factors:
                    try:
                        import json
                        zarr_json_path = os.path.join(args.output_path, "zarr.json")
                        if os.path.exists(zarr_json_path):
                            with open(zarr_json_path, 'r') as f:
                                metadata = json.load(f)

                            # Initialize custom metadata if not present
                            if "custom" not in metadata.get("attributes", {}):
                                metadata["attributes"]["custom"] = {}
                            if "downsampling_factors" not in metadata["attributes"]["custom"]:
                                metadata["attributes"]["custom"]["downsampling_factors"] = {}

                            # Store factors for this level
                            metadata["attributes"]["custom"]["downsampling_factors"][f"s{args.level}"] = anisotropic_factors

                            # Write back
                            with open(zarr_json_path, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            print(f"✓ Stored downsampling factors for s{args.level}: {anisotropic_factors}")
                    except Exception as e:
                        print(f"Warning: Could not store downsampling factors: {e}")
        elif args.task == "tiff_to_zarr3_s0":
            tiff_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape, create_dual_metadata, use_v2_encoding, use_fortran_order=bool(args.use_fortran_order))
        elif args.task == "nd2_to_zarr3_s0":
            nd2_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape, create_dual_metadata, use_v2_encoding)
        elif args.task == "ims_to_zarr3_s0":
            ims_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), create_dual_metadata, use_v2_encoding)
        elif args.task == "nd2_to_zarr2_s0":
            nd2_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape)
        elif args.task == "ims_to_zarr2_s0":
            ims_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure))
        elif args.task == "tiff_to_zarr2_s0":
            tiff_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_chunk_shape)
        elif args.task == "downsample_zarr2":
            downsample_zarr2.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.downsample), args.memory_limit, custom_chunk_shape, anisotropic_factors)
        elif args.task == "precomputed_to_n5":
            precomputed_to_n5.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit, custom_chunk_shape)
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # Auto-multiscale generation (local mode only)
        if args.auto_multiscale:
            # Only trigger for s0 tasks
            if args.task in ["tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0", "tiff_to_zarr2_s0", "nd2_to_zarr2_s0", "ims_to_zarr2_s0"]:
                print("\n" + "="*80)
                print("AUTO-MULTISCALE ENABLED - Starting pyramid generation")
                print("="*80)

                # Call generate_auto_multiscale from utils
                generate_auto_multiscale(
                    output_path=args.output_path,
                    min_dimension=args.min_dimension,
                    use_shard=bool(args.use_shard),
                    memory_limit=args.memory_limit,
                    custom_shard_shape=custom_shard_shape,
                    custom_chunk_shape=custom_chunk_shape,
                    is_submit_mode=False
                )

        # Update OME-Zarr metadata after processing is complete
        # For zarr3 tasks that support use_ome_structure parameter
        if args.task in ["tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0"]:
            update_ome_metadata_if_needed(args.output_path, bool(args.use_ome_structure))
        # For zarr3 downsample tasks that work with existing OME structure
        elif args.task == "downsample_shard_zarr3":
            update_ome_metadata_if_needed(args.output_path, use_ome_structure=True)
        # Note: zarr2 tasks handle their own OME metadata updates internally

if __name__ == "__main__":
    main()
