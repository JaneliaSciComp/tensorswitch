#!/usr/bin/env python3
"""
Dask utilities for TensorSwitch Single-Job Mode.

Single-Job Mode uses Dask's LocalCluster to create multiple workers within
a single LSF job, providing:
- Simple debugging (1 log file instead of N)
- Centralized monitoring (Dask dashboard)
- Automatic load balancing
- Lower cluster noise (1 job submission)
"""

import os
import sys


# ============================================================================
# Single-Job Mode: One LSF Job + Internal LocalCluster
# ============================================================================

def process_with_local_cluster(args):
    """
    Generic LocalCluster-based processing for any task (Single-Job Mode).

    This function runs INSIDE a single LSF job and creates an in-memory
    LocalCluster to parallelize chunk writing across N workers.

    Benefits over multi-job approach:
    - Single LSF job (1 log file instead of 50)
    - No pre-creation needed (Zarr handles race conditions)
    - Simpler debugging (all workers in one job)
    - Reusable for ALL tasks (tiff, nd2, ims, downsample)

    Args:
        args: Argparse namespace with all task parameters

    Returns:
        True if successful, False otherwise
    """
    from dask.distributed import Client, LocalCluster
    import time

    # Auto-detect workers from LSF allocation
    num_workers = args.num_volumes
    if num_workers is None or num_workers == 8:  # 8 is default
        num_workers = int(os.environ.get('LSB_DJOB_NUMPROC',
                                         os.environ.get('LSB_MAX_NUM_PROCESSORS', 4)))

    # Calculate use_v2_encoding and create_dual_metadata from args
    use_v2_encoding = (args.dual_zarr_approach == "v2_chunks")
    create_dual_metadata = (args.dual_zarr_approach != "none")

    print("="*80)
    print(f"SINGLE-JOB MODE: 1 LSF Job + Internal LocalCluster")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Input: {args.base_path}")
    print(f"Output: {args.output_path}")
    print(f"Workers: {num_workers}")
    print(f"Chunk encoding: {'v2' if use_v2_encoding else 'default'}")
    print(f"Dual metadata: {create_dual_metadata}")

    # Calculate total chunks (reuse existing logic)
    from tensorswitch.__main__ import get_total_chunks_for_task
    total_chunks = get_total_chunks_for_task(args, use_v2_encoding=use_v2_encoding)
    print(f"Total chunks: {total_chunks:,}")

    # Start LocalCluster
    print(f"\nStarting LocalCluster...")
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=2,
        memory_limit=f'{args.memory_limit}GB',
        processes=True  # Use processes, not threads
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")

    try:
        # Distribute chunks among workers
        chunks_per_worker = total_chunks // num_workers
        print(f"\nDistributing {total_chunks:,} chunks across {num_workers} workers")
        print(f"~{chunks_per_worker} chunks per worker\n")

        # Create task arguments for each worker
        futures = []
        for worker_id in range(num_workers):
            # Calculate chunk range for this worker
            start_idx = worker_id * chunks_per_worker
            if worker_id == num_workers - 1:
                stop_idx = total_chunks  # Last worker gets remainder
            else:
                stop_idx = (worker_id + 1) * chunks_per_worker

            # Create task arguments (copy args and add chunk range)
            worker_args = {
                'task': args.task,
                'base_path': args.base_path,
                'output_path': args.output_path,
                'start_idx': start_idx,
                'stop_idx': stop_idx,
                'use_shard': args.use_shard,
                'use_ome_structure': args.use_ome_structure,
                'memory_limit': args.memory_limit,
                'custom_shard_shape': args.custom_shard_shape,
                'custom_chunk_shape': args.custom_chunk_shape,
                'use_fortran_order': args.use_fortran_order if hasattr(args, 'use_fortran_order') else False,
                'use_v2_encoding': use_v2_encoding,
                'create_dual_metadata': create_dual_metadata,
                # Downsample-specific parameters
                'level': getattr(args, 'level', None),
                'downsample': getattr(args, 'downsample', True),
                'anisotropic_factors': getattr(args, 'anisotropic_factors', None),
            }

            # Submit to worker
            future = client.submit(run_task_on_worker, worker_args)
            futures.append(future)
            print(f"  Worker {worker_id}: chunks {start_idx}-{stop_idx} ({stop_idx - start_idx} chunks)")

        # Wait for completion
        print(f"\nWaiting for {num_workers} workers to complete...")
        start_time = time.time()
        results = client.gather(futures)
        elapsed = time.time() - start_time

        # Check results
        successful = sum(1 for r in results if r)
        print(f"\nConversion complete!")
        print(f"  Workers succeeded: {successful}/{num_workers}")
        print(f"  Elapsed time: {elapsed:.1f} seconds")
        print(f"  Throughput: {total_chunks/elapsed:.1f} chunks/sec")

        if successful < num_workers:
            print(f"  WARNING: {num_workers - successful} workers failed!")
            return False

        # Write metadata (run on main process, not workers)
        if args.task in ["tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0"]:
            write_metadata_for_task(args)
        elif args.task in ["downsample_shard_zarr3", "downsample_zarr2"]:
            # Update OME-Zarr metadata for downsampling tasks
            print("\nUpdating OME-Zarr metadata after downsampling...")
            from tensorswitch.utils import update_ome_metadata_if_needed
            update_ome_metadata_if_needed(args.output_path, use_ome_structure=True)
            print("Metadata updated successfully!")

        print("\n" + "="*80)
        print("SINGLE-JOB MODE COMPLETE")
        print("="*80)
        return True

    finally:
        print("\nShutting down LocalCluster...")
        client.close()
        cluster.close()


def run_task_on_worker(worker_args):
    """
    Run task.process() on a Dask worker.

    This function is executed on each worker and calls the existing
    task-specific process() function (reuses existing code).

    Args:
        worker_args: Dictionary with task parameters

    Returns:
        True if successful, False otherwise
    """
    import dask

    # CRITICAL FIX: Use synchronous scheduler for Dask operations within workers
    # This prevents conflicts with LocalCluster's distributed scheduler
    # Workers load data using Dask arrays (load_tiff_stack, etc.) which would
    # otherwise try to serialize between schedulers, causing TypeError
    dask.config.set(scheduler='synchronous')

    # Add tensorswitch to path (in case worker doesn't have it)
    tensorswitch_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if tensorswitch_path not in sys.path:
        sys.path.insert(0, tensorswitch_path)

    task = worker_args['task']
    start_idx = worker_args['start_idx']
    stop_idx = worker_args['stop_idx']

    print(f"Worker starting: task={task}, chunks={start_idx}-{stop_idx}, PID={os.getpid()}")

    try:
        # Import the appropriate task function
        if task == "tiff_to_zarr3_s0":
            from tensorswitch.tasks.tiff_to_zarr3_s0 import process
        elif task == "nd2_to_zarr3_s0":
            from tensorswitch.tasks.nd2_to_zarr3_s0 import process
        elif task == "ims_to_zarr3_s0":
            from tensorswitch.tasks.ims_to_zarr3_s0 import process
        elif task == "downsample_shard_zarr3":
            from tensorswitch.tasks.downsample_shard_zarr3 import process
        elif task == "downsample_zarr2":
            from tensorswitch.tasks.downsample_zarr2 import process
        elif task == "n5_to_zarr3_s0":
            from tensorswitch.tasks.n5_to_zarr3_s0 import process
        elif task == "tiff_to_zarr2_s0":
            from tensorswitch.tasks.tiff_to_zarr2_s0 import process
        elif task == "nd2_to_zarr2_s0":
            from tensorswitch.tasks.nd2_to_zarr2_s0 import process
        elif task == "ims_to_zarr2_s0":
            from tensorswitch.tasks.ims_to_zarr2_s0 import process
        else:
            print(f"Unsupported task: {task}")
            return False

        # Parse custom shapes (they come as strings from command line)
        custom_shard_shape = worker_args.get('custom_shard_shape')
        if custom_shard_shape and isinstance(custom_shard_shape, str):
            custom_shard_shape = [int(x) for x in custom_shard_shape.split(',')]

        custom_chunk_shape = worker_args.get('custom_chunk_shape')
        if custom_chunk_shape and isinstance(custom_chunk_shape, str):
            custom_chunk_shape = [int(x) for x in custom_chunk_shape.split(',')]

        anisotropic_factors = worker_args.get('anisotropic_factors')
        if anisotropic_factors and isinstance(anisotropic_factors, str):
            anisotropic_factors = [int(x) for x in anisotropic_factors.split(',')]

        # Call the existing process() function
        # Downsample tasks have different signature (require level parameter)
        if task in ["downsample_shard_zarr3", "downsample_zarr2"]:
            process(
                base_path=worker_args['base_path'],
                output_path=worker_args['output_path'],
                level=worker_args['level'],  # Required for downsample
                start_idx=start_idx,
                stop_idx=stop_idx,
                downsample=worker_args.get('downsample', True),
                use_shard=worker_args['use_shard'],
                memory_limit=worker_args['memory_limit'],
                custom_shard_shape=custom_shard_shape,
                custom_chunk_shape=custom_chunk_shape,
                anisotropic_factors=anisotropic_factors,  # Use parsed version
            )
        else:
            # Conversion tasks (tiff/nd2/ims/n5 to zarr2/zarr3)
            process(
                base_path=worker_args['base_path'],
                output_path=worker_args['output_path'],
                use_shard=worker_args['use_shard'],
                memory_limit=worker_args['memory_limit'],
                start_idx=start_idx,
                stop_idx=stop_idx,
                use_ome_structure=worker_args['use_ome_structure'],
                custom_shard_shape=custom_shard_shape,
                custom_chunk_shape=custom_chunk_shape,
                create_dual_metadata=worker_args.get('create_dual_metadata', False),
                use_v2_encoding=worker_args.get('use_v2_encoding', False),
                use_fortran_order=worker_args.get('use_fortran_order', False)
            )

        print(f"Worker finished successfully: chunks {start_idx}-{stop_idx}")
        return True

    except Exception as e:
        print(f"Worker failed for chunks {start_idx}-{stop_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False


def write_metadata_for_task(args):
    """
    Write OME metadata after conversion completes.

    This runs on the main process (not workers) after all chunks are written.

    Args:
        args: Argparse namespace with task parameters
    """
    print("\nWriting OME-Zarr metadata...")
    try:
        from tensorswitch.utils import (
            extract_tiff_ome_metadata, extract_nd2_ome_metadata,
            extract_ims_metadata, convert_ome_to_zarr3_metadata,
            convert_ims_to_zarr3_metadata, write_zarr3_group_metadata,
            update_zarr_metadata_from_source
        )
        import json

        # NGFF 0.5: Write metadata to root, no multiscale folder
        output_group_path = args.output_path

        # Read array shape from s0/zarr.json (created by workers)
        s0_zarr_json_path = os.path.join(output_group_path, 's0', 'zarr.json')
        if not os.path.exists(s0_zarr_json_path):
            print(f"Warning: s0/zarr.json not found at {s0_zarr_json_path}")
            array_shape = None
        else:
            with open(s0_zarr_json_path, 'r') as f:
                s0_metadata = json.load(f)
                array_shape = tuple(s0_metadata.get('shape', []))
                print(f"Read array shape from s0/zarr.json: {array_shape}")

        if args.task == "tiff_to_zarr3_s0":
            ome_metadata, voxel_sizes = extract_tiff_ome_metadata(args.base_path)
            if ome_metadata:
                image_name = os.path.splitext(os.path.basename(args.base_path))[0]
                zarr3_metadata = convert_ome_to_zarr3_metadata(ome_metadata, array_shape, image_name)

                # Write zarr.json to root (NGFF 0.5 - no multiscale folder)
                write_zarr3_group_metadata(output_group_path, zarr3_metadata)
                print("OME-Zarr metadata written successfully")

                # Update metadata with OME XML from source TIFF (like multi-job method)
                try:
                    print("Updating zarr.json with enhanced OME XML metadata...")
                    update_zarr_metadata_from_source(output_group_path, args.base_path, source_type='tiff')
                    print("Enhanced OME XML metadata updated successfully")
                except Exception as e:
                    print(f"Warning: Could not update enhanced OME XML metadata: {e}")

        elif args.task == "nd2_to_zarr3_s0":
            ome_metadata = extract_nd2_ome_metadata(args.base_path)
            if ome_metadata:
                image_name = os.path.splitext(os.path.basename(args.base_path))[0]
                zarr3_metadata = convert_ome_to_zarr3_metadata(ome_metadata, array_shape, image_name)

                # Write zarr.json to root (NGFF 0.5 - no multiscale folder)
                write_zarr3_group_metadata(output_group_path, zarr3_metadata)
                print("OME-Zarr metadata written successfully")

                # Update metadata with OME XML from source ND2 (like multi-job method)
                try:
                    print("Updating zarr.json with enhanced OME XML metadata...")
                    update_zarr_metadata_from_source(output_group_path, args.base_path, source_type='nd2')
                    print("Enhanced OME XML metadata updated successfully")
                except Exception as e:
                    print(f"Warning: Could not update enhanced OME XML metadata: {e}")

        elif args.task == "ims_to_zarr3_s0":
            metadata, voxel_sizes = extract_ims_metadata(args.base_path)
            zarr3_metadata = convert_ims_to_zarr3_metadata(args.base_path, None, voxel_sizes)

            # Write zarr.json to root (NGFF 0.5 - no multiscale folder)
            write_zarr3_group_metadata(output_group_path, zarr3_metadata)
            print("OME-Zarr metadata written successfully")

            # Update metadata with IMS metadata (like multi-job method)
            try:
                print("Updating zarr.json with enhanced IMS metadata...")
                update_zarr_metadata_from_source(output_group_path, args.base_path, source_type='ims')
                print("Enhanced IMS metadata updated successfully")
            except Exception as e:
                print(f"Warning: Could not update enhanced IMS metadata: {e}")

    except Exception as e:
        print(f"Warning: Could not write OME metadata: {e}")
