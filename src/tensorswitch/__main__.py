import argparse
import subprocess
import os
import time
import tensorstore as ts
import numpy as np
import sys

if not __package__:
    # Make CLI runnable from source tree with
    #    python src/package
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)

from . import tasks
from .utils import get_total_chunks, downsample_spec, zarr3_store_spec, get_chunk_domains, estimate_total_chunks_for_tiff, get_input_driver
from .tasks import downsample_shard_zarr3
from .tasks import n5_to_n5
from .tasks import n5_to_zarr2
from .tasks import tiff_to_zarr3_s0

# Set umask to allow group write access
os.umask(0o0002)

def submit_job(args):
    """Handles LSF cluster job submission for different tasks."""
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    input_driver = get_input_driver(args.base_path)
    
    if args.task == "tiff_to_zarr3_s0" and input_driver == "tiff":
        total_chunks = estimate_total_chunks_for_tiff(args.base_path)
    else:
        if args.downsample and args.level > 0:
            """
            # Figure out the number of downsampled chunks
            downsample_spec_dict = downsample_spec(args.base_path)
            downsample_store = ts.open(downsample_spec_dict).result()
            downsampled_saved_path = f"{args.output_path}/multiscale/s{args.level}"
            # TODO generalize beyond zarr3
            downsampled_saved_spec = zarr3_store_spec(
                downsampled_saved_path,
                downsample_store.shape,
                downsample_store.dtype.name,
                args.use_shard
            )
            total_chunks = get_total_chunks(downsampled_saved_spec)
            """
            # Always get chunks from the true input, not from the (possibly non-existent) output
            # Use s(level-1) as input
            prev_level = args.level - 1
            if args.base_path.endswith(f"s{prev_level}"):
                input_path = args.base_path
            else:
                input_path = os.path.join(args.base_path, "multiscale", f"s{prev_level}")

            # Build the input spec and open it
            downsample_spec_dict = downsample_spec(input_path)
            downsample_store = ts.open(downsample_spec_dict).result()

            # Create output spec as it would be written
            downsampled_saved_path = f"{args.output_path}/multiscale/s{args.level}"
            downsampled_saved_spec = zarr3_store_spec(
                downsampled_saved_path,
                downsample_store.shape,
                downsample_store.dtype.name,
                args.use_shard
            )

            # Count output chunks
            # total_chunks = get_total_chunks(downsampled_saved_spec) 
            chunk_shape = downsample_store.chunk_layout.read_chunk.shape
            total_chunks = len(get_chunk_domains(chunk_shape, downsample_store))

        else:
            total_chunks = get_total_chunks(args.base_path)

    print(f"The total number of chunks is {total_chunks} with downsample={args.downsample} and level={args.level}")
    
    # The number of volumes can be at most total_chunks
    num_volumes = min(total_chunks, args.num_volumes)
    for i in range(num_volumes):
        start_idx = i * (total_chunks // num_volumes)
        # stop_idx = (i + 1) * (total_chunks // num_volumes) if i < num_volumes - 1 else None
        # for evenly split chunks per job with final chunk from last job be the total_chunks - 1 (inclusive), covering the full dataset
        if i == num_volumes - 1:
            stop_idx = total_chunks
        else:
            stop_idx = (i + 1) * (total_chunks // num_volumes)
        print(f"vol{i}: {start_idx}–{stop_idx}")

            
        job_name = f"{args.task}_vol{i}"
        
        command = [
            "bsub",
            "-J", job_name,
            "-n", "25",
            "-W", "24:00",
            "-P", args.project,
            "-g", "/scicompsoft/chend/tensorstore",
            "-o", f"{output_dir}/output__vol{i}_%J.log",
            "-e", f"{output_dir}/error_vol{i}_%J.log",
            sys.executable,
            "-m", "tensorswitch",
        ]

        forwarded_args = [
            "task",
            "base_path",
            "output_path",
            "level",
            "downsample",
            "use_shard",
            "memory_limit"
        ]
        for arg in forwarded_args:
            command += ["--"+arg, str(getattr(args, arg))]

        command += ["--start_idx", str(start_idx)]
        if stop_idx is not None:
            command += ["--stop_idx", str(stop_idx)]
        
        print(command)

        subprocess.run(command)
        print(f"Submitted {job_name}, volume={i}, chunks={start_idx}-{stop_idx}")
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Manager")
    parser.add_argument("--task", required=True, choices=["n5_to_zarr2", "n5_to_n5", "downsample_shard_zarr3", "tiff_to_zarr3_s0"])
    parser.add_argument("--base_path", required=True, help="Input dataset path")
    parser.add_argument("--output_path", required=True, help="Output dataset path (only needed for conversions)")
    parser.add_argument("--level", type=int, default=0, help="Levels to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Chunk start index (for local processing)")
    parser.add_argument("--stop_idx", type=int, help="Chunk stop index (for local processing)")
    parser.add_argument("--num_volumes", type=int, default=8, help="Number of volumes per level (for cluster jobs)")
    parser.add_argument("--downsample", type=int, default=1, choices=[0, 1], help="Enable downsampling (default: 1)")
    parser.add_argument("--use_shard", type=int, default=1, choices=[0, 1], help="Use sharded format (for downsample)")
    parser.add_argument("--submit", action="store_true", help="Submit to the cluster scheduler")
    parser.add_argument("--memory_limit", type=int, default=50, help="memory limit percentage" )
    parser.add_argument("--project", default="None", help="Project to charge")

    args = parser.parse_args()

    if args.submit:
        if args.project == "None":
            raise ValueError(f"Project cannot be None when submitting.")
        submit_job(args)

    else:
        if args.task == "n5_to_n5":
            n5_to_n5.convert(args.base_path, args.output_path, args.num_volumes, args.level, args.start_idx, args.stop_idx, args.memory_limit)
        elif args.task == "n5_to_zarr2":
            n5_to_zarr2.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit)
        elif args.task == "downsample_shard_zarr3":
            downsample_shard_zarr3.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.downsample), bool(args.use_shard), args.memory_limit)
        elif args.task == "tiff_to_zarr3_s0":
            tiff_to_zarr3_s0.process(args.base_path, args.output_path, bool(args.use_shard), args.memory_limit, args.start_idx, args.stop_idx)
        else:
            raise ValueError(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    main()
