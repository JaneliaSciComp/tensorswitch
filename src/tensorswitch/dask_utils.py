#!/usr/bin/env python3
"""
Dask utilities for TensorSwitch LSF cluster integration.
"""

import os
import sys
import re
import logging
import subprocess
import argparse
import tensorstore as ts
from dask.distributed import Client
from dask_jobqueue import LSFCluster
from tensorswitch.utils import (zarr3_store_spec, get_chunk_domains, load_nd2_stack,
                                load_tiff_stack, extract_nd2_ome_metadata, extract_tiff_ome_metadata,
                                extract_ims_metadata, convert_ome_to_zarr3_metadata,
                                convert_ims_to_zarr3_metadata, write_zarr3_group_metadata)

logger = logging.getLogger(__name__)


def create_lsf_cluster(
    num_workers: int = 96,
    cores: int = 4,
    memory: str = "60GB", 
    walltime: str = "1:00",
    project: str = "scicompsoft"
) -> LSFCluster:
    """Create LSFCluster using proven configuration with output files in output/ directory."""
    
    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    cluster = LSFCluster(
        num_workers, 
        project=project, 
        ncpus=cores, 
        cores=cores, 
        memory=memory, 
        walltime=walltime, 
        job_extra_directives=[f"-o {output_dir}/output_%J.txt", f"-e {output_dir}/error_%J.txt"], 
        use_stdin=True
    )
    return cluster


def process_shard_task(shard_task):
    """Process all chunks within a 3D shard on a Dask worker."""
    # Add tensorswitch to path
    sys.path.insert(0, shard_task['tensorswitch_path'])

    # Task parameters
    task_type = shard_task['task']
    base_path = shard_task['base_path']
    output_path = shard_task['output_path']
    shard_coord = shard_task['shard_coord']
    chunk_indices = shard_task['chunk_indices']
    use_shard = shard_task['use_shard']
    use_ome_structure = shard_task['use_ome_structure']

    print(f"Worker processing shard {shard_coord} ({len(chunk_indices)} chunks) for task {task_type}")

    try:
        if task_type == "nd2_to_zarr3_s0":
            volume = load_nd2_stack(base_path)
            store_spec = zarr3_store_spec(
                path=output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=use_shard,
                use_ome_structure=use_ome_structure,
                custom_shard_shape=shard_task.get('custom_shard_shape'),
                custom_chunk_shape=shard_task.get('custom_chunk_shape')
            )
            store = ts.open(store_spec, open=True).result()

        elif task_type == "tiff_to_zarr3_s0":
            volume = load_tiff_stack(base_path)
            store_spec = zarr3_store_spec(
                path=output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=use_shard,
                use_ome_structure=use_ome_structure,
                custom_shard_shape=shard_task.get('custom_shard_shape'),
                custom_chunk_shape=shard_task.get('custom_chunk_shape')
            )
            store = ts.open(store_spec, open=True).result()

        else:
            print(f"Unsupported task type for shard processing: {task_type}")
            return False

        # Process all chunks in this shard
        chunk_shape = store.chunk_layout.write_chunk.shape
        successful_chunks = 0

        for chunk_idx in chunk_indices:
            try:
                chunk_domains = list(get_chunk_domains(chunk_shape, store, linear_indices_to_process=[chunk_idx]))
                if not chunk_domains:
                    continue

                domain = chunk_domains[0]
                slices = tuple(slice(min, max) for (min, max) in zip(domain.inclusive_min, domain.exclusive_max))
                slice_data = volume[slices]

                # Write chunk
                task = store[domain].write(slice_data.compute())
                task.result()
                successful_chunks += 1

            except Exception as e:
                print(f"Error processing chunk {chunk_idx} in shard {shard_coord}: {e}")
                continue

        print(f"Successfully processed shard {shard_coord}: {successful_chunks}/{len(chunk_indices)} chunks")
        return successful_chunks == len(chunk_indices)

    except Exception as e:
        print(f"Error processing shard {shard_coord}: {e}")
        return False


def process_chunk_task(chunk_task):
    """Process a single chunk on a Dask worker."""
    # Add tensorswitch to path
    sys.path.insert(0, chunk_task['tensorswitch_path'])

    # Task parameters
    task_type = chunk_task['task']
    base_path = chunk_task['base_path']
    output_path = chunk_task['output_path']
    chunk_idx = chunk_task['chunk_idx']
    use_shard = chunk_task['use_shard']
    use_ome_structure = chunk_task['use_ome_structure']

    print(f"Worker processing chunk {chunk_idx} for task {task_type}")

    try:
        if task_type == "nd2_to_zarr3_s0":
            # Load data and open store
            volume = load_nd2_stack(base_path)
            store_spec = zarr3_store_spec(
                path=output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=use_shard,
                use_ome_structure=use_ome_structure,
                custom_shard_shape=chunk_task.get('custom_shard_shape'),
                custom_chunk_shape=chunk_task.get('custom_chunk_shape')
            )

            store = ts.open(store_spec, open=True).result()

            # Get specific chunk domain
            chunk_shape = store.chunk_layout.write_chunk.shape
            chunk_domains = list(get_chunk_domains(chunk_shape, store, linear_indices_to_process=[chunk_idx]))

            if not chunk_domains:
                print(f"No chunk domain found for index {chunk_idx}")
                return False

            domain = chunk_domains[0]

            # Process the chunk
            slices = tuple(slice(min, max) for (min, max) in zip(domain.inclusive_min, domain.exclusive_max))
            slice_data = volume[slices]

            # Write chunk
            task = store[domain].write(slice_data.compute())
            task.result()

            print(f"Successfully processed chunk {chunk_idx}")
            return True

        elif task_type == "tiff_to_zarr3_s0":
            # Load data and open store
            volume = load_tiff_stack(base_path)
            store_spec = zarr3_store_spec(
                path=output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=use_shard,
                use_ome_structure=use_ome_structure
            )

            store = ts.open(store_spec, open=True).result()

            # Get specific chunk domain
            chunk_shape = store.chunk_layout.write_chunk.shape
            chunk_domains = list(get_chunk_domains(chunk_shape, store, linear_indices_to_process=[chunk_idx]))

            if not chunk_domains:
                print(f"No chunk domain found for index {chunk_idx}")
                return False

            domain = chunk_domains[0]

            # Process the chunk
            slices = tuple(slice(min, max) for (min, max) in zip(domain.inclusive_min, domain.exclusive_max))
            slice_data = volume[slices]

            # Write chunk
            task = store[domain].write(slice_data.compute())
            task.result()

            print(f"Successfully processed chunk {chunk_idx}")
            return True

        else:
            print(f"Task type {task_type} not implemented for chunk-level processing yet")
            return False

    except Exception as e:
        print(f"Error processing chunk {chunk_idx}: {e}")
        return False

def submit_dask_job(args, total_chunks):
    """Submit TensorSwitch job using Dask task distribution."""

    print("Using Dask JobQueue for task submission")
    print(f"Total chunks: {total_chunks}")
    print(f"Workers: {args.num_volumes}, Cores per worker: {args.cores}")

    cluster = None
    client = None

    try:
        # Create cluster
        cluster = create_lsf_cluster(
            num_workers=args.num_volumes,
            cores=int(args.cores),
            memory="60GB",
            walltime=args.wall_time,
            project=args.project
        )

        client = Client(cluster)
        print(f"Dashboard: {client.dashboard_link}")

        # Parse custom shapes
        custom_shard_shape = None
        if hasattr(args, 'custom_shard_shape') and args.custom_shard_shape:
            custom_shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]

        custom_chunk_shape = None
        if hasattr(args, 'custom_chunk_shape') and args.custom_chunk_shape:
            custom_chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]

        # Initialize output store
        print("Initializing output store...")
        if args.task == "nd2_to_zarr3_s0":

            volume = load_nd2_stack(args.base_path)
            store_spec = zarr3_store_spec(
                path=args.output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=bool(args.use_shard),
                use_ome_structure=bool(args.use_ome_structure),
                custom_shard_shape=custom_shard_shape,
                custom_chunk_shape=custom_chunk_shape
            )

            store = ts.open(store_spec, create=True, delete_existing=True).result()

        elif args.task == "tiff_to_zarr3_s0":

            volume = load_tiff_stack(args.base_path)
            store_spec = zarr3_store_spec(
                path=args.output_path,
                shape=volume.shape,
                dtype=str(volume.dtype),
                use_shard=bool(args.use_shard),
                use_ome_structure=bool(args.use_ome_structure)
            )

            store = ts.open(store_spec, create=True, delete_existing=True).result()

        # Determine if we should use 3D shard-based distribution
        # (for sharded arrays to avoid concurrent writes to same shard)
        use_3d_shard_distribution = (
            bool(args.use_shard) and
            custom_shard_shape is not None and
            custom_chunk_shape is not None
        )

        tensorswitch_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if use_3d_shard_distribution:
            # Get volume shape for 3D shard calculation
            if args.task == "nd2_to_zarr3_s0":
                volume_shape = volume.shape
            elif args.task == "tiff_to_zarr3_s0":
                volume_shape = volume.shape
            else:
                # For downsampling, need to get output shape
                # This is handled in the main code, so we fall back to linear for now
                use_3d_shard_distribution = False
                volume_shape = None

            if use_3d_shard_distribution and volume_shape:
                import math

                # Calculate N-D shard grid
                shard_grid = [
                    (volume_shape[i] + custom_shard_shape[i] - 1) // custom_shard_shape[i]
                    for i in range(len(volume_shape))
                ]

                # Calculate chunk grid
                chunk_grid = [
                    (volume_shape[i] + custom_chunk_shape[i] - 1) // custom_chunk_shape[i]
                    for i in range(len(volume_shape))
                ]

                # Generate all N-D shard coordinates dynamically (supports 2D, 3D, 4D, 5D, etc.)
                # Use itertools.product to iterate over all combinations of shard indices
                import itertools
                all_shard_coords = [
                    list(coord) for coord in itertools.product(*[range(dim) for dim in shard_grid])
                ]

                print(f"Using N-D shard-based distribution: {len(all_shard_coords)} shards (grid: {shard_grid}, {len(shard_grid)}D data)")
                print(f"Creating {len(all_shard_coords)} shard tasks...")

                # Create tasks per shard (not per chunk)
                shard_tasks = []
                for shard_coord in all_shard_coords:
                    # Calculate all chunk indices within this N-D shard
                    chunks_per_shard_dim = [
                        custom_shard_shape[j] // custom_chunk_shape[j]
                        for j in range(len(custom_shard_shape))
                    ]

                    # Base chunk coordinate for this shard
                    base_chunk_coord = [
                        shard_coord[j] * chunks_per_shard_dim[j]
                        for j in range(len(shard_coord))
                    ]

                    # Generate all chunk indices within this shard using N-D iteration
                    chunk_indices = []
                    for chunk_offset in itertools.product(*[range(dim) for dim in chunks_per_shard_dim]):
                        chunk_coord = [
                            base_chunk_coord[i] + chunk_offset[i]
                            for i in range(len(base_chunk_coord))
                        ]

                        # Skip if chunk is outside data bounds
                        if any(chunk_coord[i] >= chunk_grid[i] for i in range(len(chunk_coord))):
                            continue

                        # Convert N-D chunk coordinate to linear index
                        linear_idx = 0
                        stride = 1
                        for i in range(len(chunk_coord) - 1, -1, -1):
                            linear_idx += chunk_coord[i] * stride
                            stride *= chunk_grid[i]
                        chunk_indices.append(linear_idx)

                    shard_task = {
                        'task': args.task,
                        'base_path': args.base_path,
                        'output_path': args.output_path,
                        'shard_coord': shard_coord,
                        'chunk_indices': chunk_indices,  # All chunks in this shard
                        'use_shard': bool(args.use_shard),
                        'use_ome_structure': bool(args.use_ome_structure),
                        'custom_shard_shape': custom_shard_shape,
                        'custom_chunk_shape': custom_chunk_shape,
                        'tensorswitch_path': tensorswitch_path,
                        'use_3d_shard': True
                    }
                    shard_tasks.append(shard_task)

                # Submit shard tasks to workers
                print(f"Submitting {len(shard_tasks)} shard tasks to {args.num_volumes} workers...")
                futures = []
                for shard_task in shard_tasks:
                    future = client.submit(process_shard_task, shard_task)
                    futures.append(future)
            else:
                use_3d_shard_distribution = False

        if not use_3d_shard_distribution:
            # Fall back to linear chunk distribution
            print(f"Creating {total_chunks} chunk tasks (linear distribution)...")
            chunk_tasks = []

            for chunk_idx in range(total_chunks):
                chunk_task = {
                    'task': args.task,
                    'base_path': args.base_path,
                    'output_path': args.output_path,
                    'chunk_idx': chunk_idx,
                    'use_shard': bool(args.use_shard),
                    'use_ome_structure': bool(args.use_ome_structure),
                    'custom_shard_shape': custom_shard_shape,
                    'custom_chunk_shape': custom_chunk_shape,
                    'tensorswitch_path': tensorswitch_path,
                    'use_3d_shard': False
                }
                chunk_tasks.append(chunk_task)

            # Submit chunk tasks to workers
            print(f"Submitting {len(chunk_tasks)} chunk tasks to {args.num_volumes} workers...")
            futures = []
            for chunk_task in chunk_tasks:
                future = client.submit(process_chunk_task, chunk_task)
                futures.append(future)

        print("Waiting for chunk tasks to complete...")
        results = client.gather(futures)

        # Check results
        successful_chunks = sum(results)
        failed_chunks = len(results) - successful_chunks

        print(f"Chunk processing complete: {successful_chunks} successful, {failed_chunks} failed")

        if failed_chunks > 0:
            print(f"WARNING: {failed_chunks} chunks failed to process")
            return False

        # Write metadata after chunk processing
        if args.task in ["nd2_to_zarr3_s0", "tiff_to_zarr3_s0", "ims_to_zarr3_s0"] and bool(args.use_ome_structure):
            print("Writing OME-Zarr metadata...")
            try:
                if bool(args.use_ome_structure):
                    output_group_path = os.path.join(args.output_path, "multiscale")
                else:
                    output_group_path = args.output_path

                if args.task == "nd2_to_zarr3_s0":
                    ome_metadata = extract_nd2_ome_metadata(args.base_path)
                    if ome_metadata:
                        image_name = os.path.splitext(os.path.basename(args.base_path))[0]
                        zarr3_metadata = convert_ome_to_zarr3_metadata(ome_metadata, image_name)
                        write_zarr3_group_metadata(output_group_path, zarr3_metadata)
                        print("OME metadata written successfully")

                        # Update metadata with OME XML from source ND2
                        try:
                            print("Updating zarr.json with enhanced OME XML metadata...")
                            from tensorswitch.tasks.nd2_to_zarr3_s0 import update_zarr_ome_xml_nd2
                            update_zarr_ome_xml_nd2(output_group_path, args.base_path)
                            print("Enhanced OME XML metadata updated successfully")
                        except Exception as e:
                            print(f"Warning: Could not update enhanced OME XML metadata: {e}")

                elif args.task == "tiff_to_zarr3_s0":
                    ome_metadata = extract_tiff_ome_metadata(args.base_path)
                    image_name = os.path.splitext(os.path.basename(args.base_path))[0]
                    zarr3_metadata = convert_ome_to_zarr3_metadata(ome_metadata, image_name)
                    write_zarr3_group_metadata(output_group_path, zarr3_metadata)
                    print("OME metadata written successfully")

                    # Update metadata with OME XML from source TIFF
                    try:
                        print("Updating zarr.json with enhanced OME XML metadata...")
                        from tensorswitch.tasks.tiff_to_zarr3_s0 import update_zarr_ome_xml
                        update_zarr_ome_xml(output_group_path, args.base_path)
                        print("Enhanced OME XML metadata updated successfully")
                    except Exception as e:
                        print(f"Warning: Could not update enhanced OME XML metadata: {e}")

                elif args.task == "ims_to_zarr3_s0":
                    metadata, voxel_sizes = extract_ims_metadata(args.base_path)
                    zarr3_metadata = convert_ims_to_zarr3_metadata(args.base_path, None, voxel_sizes)  # volume.shape not available here
                    write_zarr3_group_metadata(output_group_path, zarr3_metadata)
                    print("OME metadata written successfully")

                    # Update metadata with enhanced IMS metadata
                    try:
                        print("Updating zarr.json with enhanced IMS metadata...")
                        from tensorswitch.tasks.ims_to_zarr3_s0 import update_zarr_ome_xml_ims
                        update_zarr_ome_xml_ims(output_group_path, args.base_path)
                        print("Enhanced IMS metadata updated successfully")
                    except Exception as e:
                        print(f"Warning: Could not update enhanced IMS metadata: {e}")

                else:
                    print("No OME metadata found in source file")
            except Exception as e:
                print(f"Warning: Could not write OME metadata: {e}")

        print("All chunk tasks completed successfully")
        print("=== NORMAL COMPLETION: Dask will now terminate worker jobs ===")
        print("NOTE: LSF 'TERM_OWNER: job killed by owner' messages are EXPECTED")
        print("This indicates normal Dask cleanup, not a failure")
        return True

    except Exception as e:
        print(f"Dask job submission failed: {e}")
        print("=== JOB FAILED: Check error details above ===")
        return False

    finally:
        print("Jobs submitted - cluster will manage LSF execution independently")
        print(f"Logs in: {os.path.abspath('./output')}")
        if client:
            print("=== Shutting down Dask cluster - worker termination is expected ===")
            client.close()
        if cluster:
            cluster.close()



def submit_dask_wrapper_job(args, total_chunks):
    """Submit wrapper LSF job that runs Dask job independently."""

    # Validate project before submission
    if not args.project or args.project == "None" or args.project.strip() == "":
        raise ValueError(f"Invalid project '{args.project}' for Dask JobQueue submission. Please specify a valid LSF project.")

    # Create output directory
    output_dir = os.path.abspath("./output")
    os.makedirs(output_dir, exist_ok=True)

    # Build the Python command that will run inside the wrapper job
    python_cmd = [
        sys.executable, "-c", f'''
import sys
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

import argparse

# Recreate args namespace
class Args:
    def __init__(self):
        self.task = "{args.task}"
        self.base_path = "{args.base_path}"
        self.output_path = "{args.output_path}"
        self.level = {args.level}
        self.num_volumes = {args.num_volumes}
        self.downsample = {args.downsample}
        self.use_shard = {args.use_shard}
        self.use_ome_structure = {args.use_ome_structure}
        self.memory_limit = {args.memory_limit}
        self.project = "{args.project}"
        self.cores = "{args.cores}"
        self.wall_time = "{args.wall_time}"
        self.custom_shard_shape = "{args.custom_shard_shape if args.custom_shard_shape else ''}"
        self.custom_chunk_shape = "{args.custom_chunk_shape if args.custom_chunk_shape else ''}"

args = Args()
total_chunks = {total_chunks}

# Import and run the actual Dask job
from tensorswitch.dask_utils import submit_dask_job
print("=== Dask Wrapper Job Started ===")
print(f"Task: {{args.task}}")
print(f"Input: {{args.base_path}}")
print(f"Output: {{args.output_path}}")
print(f"Total chunks: {{total_chunks}}")

success = submit_dask_job(args, total_chunks)
if success:
    print("=== Dask Wrapper Job Completed Successfully ===")
    print("NOTE: Any subsequent TERM_OWNER job killed by owner messages")
    print("in worker logs are NORMAL and indicate proper Dask cleanup")
else:
    print("=== Dask Wrapper Job Failed ===")
    print("Check error messages above for details")
'''
    ]

    # Build LSF bsub command
    job_name = f"dask_wrapper_{args.task}"
    bsub_cmd = [
        "bsub",
        "-J", job_name,
        "-P", args.project,
        "-n", "2",
        "-W", "1:00",
        "-o", f"{output_dir}/wrapper_{job_name}_%J.out",
        "-e", f"{output_dir}/wrapper_{job_name}_%J.err"
    ]

    # Add the Python command
    bsub_cmd.extend(python_cmd)

    print("=== Submitting Dask Wrapper Job ===")
    print(f"Command: {' '.join(bsub_cmd[:10])}...")
    print(f"Job name: {job_name}")
    print(f"Project: {args.project}")
    print(f"Output files will be in: {output_dir}")

    try:
        result = subprocess.run(bsub_cmd, capture_output=True, text=True, check=True)
        print(f"Wrapper job submitted successfully!")
        print(f"LSF output: {result.stdout.strip()}")

        # Extract job ID from bsub output
        job_id_match = re.search(r'Job <(\d+)>', result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            print(f"Job ID: {job_id}")
            print(f"Monitor with: bjobs {job_id}")
            print(f"Logs: {output_dir}/wrapper_{job_name}_{job_id}.out")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit wrapper job: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error submitting wrapper job: {e}")
        return False

