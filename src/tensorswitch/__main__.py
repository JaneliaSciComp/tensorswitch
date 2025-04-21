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
from .utils import get_total_chunks
from .tasks import downsample_shard_zarr3
from .tasks import n5_to_n5
from .tasks import n5_to_zarr2

# Set umask to allow group write access
os.umask(0o0002)

def submit_job(args):
    """Handles LSF cluster job submission for different tasks."""
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    total_chunks = get_total_chunks(args.base_path)
    # The number of volumes can be at most total_chunks
    num_volumes = min(total_chunks, args.num_volumes)
    for i in range(num_volumes):
        start_idx = i * (total_chunks // num_volumes)
        stop_idx = (i + 1) * (total_chunks // num_volumes) if i < num_volumes - 1 else None
        job_name = f"{args.task}_vol{i}"
        
        command = [
            "bsub",
            "-J", job_name,
            "-n", "2",
            "-W", "1:00",
            "-P", "scicompsoft",
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
            "use_shard",
            "memory_limit"
        ]
        for arg in forwarded_args:
            command += ["--"+arg, str(getattr(args, arg))]

        command += ["--start_idx", str(start_idx)]
        command += ["--stop_idx", str(stop_idx)]
        
        print(command)

        subprocess.run(command)
        print(f"Submitted {job_name}, volume={i}, chunks={start_idx}-{stop_idx}")
        time.sleep(0.1)

def submit_job_old(task, base_path, output_path, levels, num_volumes, use_shard):
    """Handles LSF cluster job submission for different tasks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    conversion_scripts = {
        "n5_to_n5": "n5_to_n5.py",
        "n5_to_zarr2": "n5_to_zarr2.py",
        "downsample_shard_zarr3": "downsample_shard_zarr3.py"
    }

    if task not in conversion_scripts:
        raise ValueError(f"Unknown task: {task}")

    conversion_script = os.path.join(script_dir, conversion_scripts[task])

    volume_settings = {
        0: 32,  
        1: 16,  
        2: 8,   
        3: 4,   
        4: 2,   
    }

    for level in levels:
        total_chunks = get_total_chunks(f"{base_path}/s{level}")
        volumes_per_level = volume_settings.get(level, num_volumes)

        print(f"Submitting {task} jobs for level {level}, total chunks: {total_chunks}, volumes per level: {volumes_per_level}")

        for i in range(volumes_per_level):
            start_idx = i * (total_chunks // volumes_per_level)
            stop_idx = (i + 1) * (total_chunks // volumes_per_level) if i < volumes_per_level - 1 else None

            job_name = f"{task}_s{level}_vol{i}"

            command = [
                "bsub",
                "-J", job_name,
                "-n", "2",
                "-W", "1:00",
                "-P", "scicompsoft",
                "-g", "/scicompsoft/chend/tensorstore",
                "-o", f"{output_dir}/output_s{level}_vol{i}_%J.log",
                "-e", f"{output_dir}/error_s{level}_vol{i}_%J.log",
                sys.executable,
                "-m", "tensorswitch",
            ]

            forwarded_args = [
                "task",
                "base_path",
                "output_path",
                "level",
                "start_idx",
                "stop_idx",
                "num_volumes",
                "use_shard",
                "memory_limit"
            ]
            for arg in forwarded_args:
                command += ["--"*arg, str(getattr(args, arg))]
            #if task in ["n5_to_n5", "n5_to_zarr2"]:
            #    command += [base_path, output_path, str(level), str(start_idx), str(stop_idx)]
            #elif task == "downsample_shard_zarr3":
            #    command += [base_path, str(level), str(start_idx), str(stop_idx), str(int(use_shard))]

            subprocess.run(command)
            print(f"Submitted {job_name}, level={level}, volume={i}, chunks={start_idx}-{stop_idx}, use_shard={use_shard}")
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Manager")
    parser.add_argument("--task", required=True, choices=["n5_to_zarr2", "n5_to_n5", "downsample_shard_zarr3"])
    parser.add_argument("--base_path", required=True, help="Input dataset path")
    parser.add_argument("--output_path", required=True, help="Output dataset path (only needed for conversions)")
    parser.add_argument("--level", type=int, default=0, help="Levels to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Chunk start index (for local processing)")
    parser.add_argument("--stop_idx", type=int, help="Chunk stop index (for local processing)")
    parser.add_argument("--num_volumes", type=int, default=8, help="Number of volumes per level (for cluster jobs)")
    parser.add_argument("--use_shard", type=int, default=1, choices=[0, 1], help="Use sharded format (for downsample)")
    parser.add_argument("--submit", action="store_true", help="Submit to the cluster scheduler")
    parser.add_argument("--memory_limit", type=int, default=50, help="memory limit percentage" )

    args = parser.parse_args()

    if args.submit:
        submit_job(args)
        """
        submit_job(
            task=args.task,
            base_path=args.base_path,
            output_path=args.output_path or "dummy_output_path",
            levels=args.level,
            num_volumes=args.num_volumes,
            use_shard=bool(args.use_shard)
        )
        """
    else:
        if args.task == "n5_to_n5":
            n5_to_n5.convert(args.base_path, args.output_path, args.num_volumes, args.level, args.start_idx, args.stop_idx, args.memory_limit)
        elif args.task == "n5_to_zarr2":
            n5_to_zarr2.convert(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, args.memory_limit)
        elif args.task == "downsample_shard_zarr3":
            downsample_shard_zarr3.process(args.base_path, args.output_path, args.level, args.start_idx, args.stop_idx, bool(args.use_shard), args.memory_limit)

        else:
            raise ValueError(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    main()
