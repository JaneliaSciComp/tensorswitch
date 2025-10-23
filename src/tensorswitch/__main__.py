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
                    load_tiff_stack, load_nd2_stack, load_ims_stack, update_ome_metadata_if_needed)
from .tasks import (downsample_shard_zarr3, n5_to_n5, n5_to_zarr2, tiff_to_zarr3_s0,
                    nd2_to_zarr3_s0, ims_to_zarr3_s0, nd2_to_zarr2_s0, ims_to_zarr2_s0,
                    tiff_to_zarr2_s0, downsample_zarr2, precomputed_to_n5)
from .dask_utils import submit_dask_job, submit_dask_wrapper_job

# Set umask to allow group write access
os.umask(0o0002)


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

    if args.task == "tiff_to_zarr3_s0" and input_driver == "tiff":
        volume = load_tiff_stack(args.base_path)
        store_spec = zarr3_store_spec(
            path=args.output_path,
            shape=volume.shape,
            dtype=str(volume.dtype),
            use_shard=bool(args.use_shard),
            use_ome_structure=bool(args.use_ome_structure),
            use_v2_encoding=use_v2_encoding
        )
        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()
        total_chunks = get_total_chunks_from_store(temp_store)

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
            if len(volume.shape) == 4 and len(custom_shard_shape) == 3:
                chunk_shape_for_counting = [1] + custom_shard_shape
            elif len(volume.shape) == 5 and len(custom_shard_shape) == 3:
                chunk_shape_for_counting = [1, 1] + custom_shard_shape
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

            downsample_store = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": input_path}}).result()
            chunk_shape = downsample_store.chunk_layout.read_chunk.shape
            total_chunks = get_total_chunks_from_store(downsample_store, chunk_shape=chunk_shape)
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

    # Check if using Dask JobQueue
    if hasattr(args, 'use_dask_jobqueue') and args.use_dask_jobqueue:
        print("Using Dask JobQueue submission")
        return submit_dask_wrapper_job(args, total_chunks)

    # Traditional LSF bsub method
    print("Using traditional LSF bsub submission")
    
    # The number of volumes can be at most total_chunks
    num_volumes = min(total_chunks, args.num_volumes)
    for i in range(num_volumes):
        start_idx = i * (total_chunks // num_volumes)
        if i == num_volumes - 1:
            stop_idx = total_chunks
        else:
            stop_idx = (i + 1) * (total_chunks // num_volumes)
        print(f"vol{i}: {start_idx}–{stop_idx}")

        # Extract setup number from base_path if any
        match = re.search(r"setup(\d+)", args.base_path)
        if match:
            setup_number = match.group(1)
            job_name = f"{args.task}_setup{setup_number}_s{args.level}_vol{i}"
        else:
            job_name = f"{args.task}_s{args.level}_vol{i}"


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

        string_args = ["task", "base_path", "output_path", "custom_shard_shape", "custom_chunk_shape", "dual_zarr_approach"]
        int_args = ["level", "downsample", "use_shard", "use_ome_structure", "memory_limit"]
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
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Manager")
    parser.add_argument("--task", required=True, choices=["n5_to_zarr2", "n5_to_n5", "downsample_shard_zarr3", "tiff_to_zarr3_s0", "nd2_to_zarr3_s0", "ims_to_zarr3_s0", "nd2_to_zarr2_s0", "ims_to_zarr2_s0", "tiff_to_zarr2_s0", "downsample_zarr2", "precomputed_to_n5"])
    parser.add_argument("--base_path", required=True, help="Input dataset path")
    parser.add_argument("--output_path", required=True, help="Output dataset path (only needed for conversions)")
    parser.add_argument("--level", type=int, default=0, help="Levels to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Chunk start index (for local processing)")
    parser.add_argument("--stop_idx", type=int, help="Chunk stop index (for local processing)")
    parser.add_argument("--num_volumes", type=int, default=8, help="Number of volumes per level (for cluster jobs)")
    parser.add_argument("--downsample", type=int, default=0, choices=[0, 1], help="Enable downsampling (default: 1)")
    parser.add_argument("--use_shard", type=int, default=0, choices=[0, 1], help="Use sharded format (for downsample)")
    parser.add_argument("--use_ome_structure", type=int, default=1, choices=[0, 1], help="Use OME-ZARR multiscale structure (s0 subdirectory)")
    parser.add_argument("--submit", action="store_true", help="Submit to the cluster scheduler")
    parser.add_argument("--memory_limit", type=int, default=50, help="memory limit percentage" )
    parser.add_argument("--project", default="None", help="Project to charge")
    parser.add_argument("--cores", type=str, default="2", help="Number of cores for LSF job (-n flag)")
    parser.add_argument("--wall_time", type=str, default="1:00", help="Wall time for LSF job (-W flag)")
    parser.add_argument("--custom_shard_shape", type=str, help="Custom shard shape as comma-separated values (e.g., '128,576,576')")
    parser.add_argument("--custom_chunk_shape", type=str, help="Custom chunk shape as comma-separated values (e.g., '32,32,32')")
    parser.add_argument("--use_dask_jobqueue", action="store_true", help="Use Dask JobQueue instead of direct LSF submission")
    parser.add_argument("--dual_zarr_approach", type=str, default="none", choices=["v2_chunks", "v3_chunks", "none"], help="Dual zarr v2/v3 compatibility approach: none (default, pure zarr v3), v2_chunks (colocated metadata), or v3_chunks (.zarray in c/ directory)")

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
        elif args.task == "downsample_shard_zarr3":
            downsample_shard_zarr3.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.downsample), bool(args.use_shard), args.memory_limit, custom_shard_shape, custom_chunk_shape)
        elif args.task == "tiff_to_zarr3_s0":
            tiff_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), create_dual_metadata, use_v2_encoding)
        elif args.task == "nd2_to_zarr3_s0":
            nd2_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape, create_dual_metadata, use_v2_encoding)
        elif args.task == "ims_to_zarr3_s0":
            ims_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), create_dual_metadata, use_v2_encoding)
        elif args.task == "nd2_to_zarr2_s0":
            nd2_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure), custom_shard_shape, custom_chunk_shape)
        elif args.task == "ims_to_zarr2_s0":
            ims_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure))
        elif args.task == "tiff_to_zarr2_s0":
            tiff_to_zarr2_s0.process(args.base_path, args.output_path, args.memory_limit, args.start_idx, args.stop_idx, bool(args.use_ome_structure))
        elif args.task == "downsample_zarr2":
            downsample_zarr2.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.downsample), args.memory_limit, custom_chunk_shape)
        elif args.task == "precomputed_to_n5":
            precomputed_to_n5.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit, custom_chunk_shape)
        else:
            raise ValueError(f"Unsupported task: {args.task}")
        
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
