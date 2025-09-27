#!/usr/bin/env python3
"""
Generic bleaching correction submission script for s0 level.
Usage: python submit_bleaching_correction_general.py <input_path> <output_path> [num_volumes] [project]
"""

import subprocess
import os
import time
import sys
import argparse

# Add path to use tensorswitch utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorstore as ts
import numpy as np

# Set umask to allow group write access
os.umask(0o0002)

def main():
    parser = argparse.ArgumentParser(description="Submit bleaching correction jobs to LSF cluster for s0 level")
    parser.add_argument("input_path", help="Input zarr dataset path")
    parser.add_argument("output_path", help="Output zarr dataset path")
    parser.add_argument("--num_volumes", type=int, default=16, help="Number of job volumes (default: 16)")
    parser.add_argument("--project", default="tavakoli", help="LSF project name (default: tavakoli)")
    parser.add_argument("--cores", type=str, default="2", help="Number of cores per job (default: 2)")
    parser.add_argument("--wall_time", default="2:00", help="Wall time per job (default: 2:00)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    # Path to the bleaching correction script (same directory)
    correction_script = os.path.join(script_dir, "bleaching_correction_task.py")

    print("Input path:", args.input_path)
    print("Output path:", args.output_path)
    print("Correction script:", correction_script)
    print("Project:", args.project)
    print("Number of volumes:", args.num_volumes)

    # Process s0 level - same as successful processing
    level = "s0"

    zarr_level_path = f"{args.input_path}/multiscale/{level}"
    total_chunks = get_total_chunks(zarr_level_path)

    print(f"Submitting jobs for {level}, total volumes: {args.num_volumes}, total chunks: {total_chunks}")

    for i in range(args.num_volumes):
        start_idx = i * (total_chunks // args.num_volumes)
        stop_idx = (i + 1) * (total_chunks // args.num_volumes) if i < args.num_volumes - 1 else total_chunks

        job_name = f"bleach_corr_{level}_vol{i}"

        # LSF submission command
        command = [
            "bsub",
            "-J", job_name,
            "-n", args.cores,
            "-W", args.wall_time,
            "-P", args.project,
            "-g", "/scicompsoft/chend/tensorstore",
            "-o", f"{output_dir}/output_{level}_vol{i}_%J.log",
            "-e", f"{output_dir}/error_{level}_vol{i}_%J.log",
            sys.executable,
            correction_script,
            args.input_path,      # input_path
            args.output_path,     # output_path
            level,                # level (s0)
            "0",                  # use_shard = False for s0
            "50",                 # memory_limit
            str(start_idx),       # start_idx
            str(stop_idx),        # stop_idx
            "1"                   # use_ome_structure = True
        ]

        print(f"Command: {' '.join(command)}")
        subprocess.run(command)
        print(f"Submitted job for {level}, volume {i}, chunks {start_idx} to {stop_idx}")
        time.sleep(0.5)  # Brief pause between submissions

    print("s0 bleaching correction job submission completed!")
    print(f"Monitor jobs with: bjobs -g /scicompsoft/chend/tensorstore")
    print(f"Check logs in: {output_dir}/")
    print(f"Output will be saved to: {args.output_path}")

def get_total_chunks(zarr_level_path):
    """Retrieve total number of chunks dynamically from the Zarr dataset."""
    zarr_store_spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': zarr_level_path},
    }
    zarr_store = ts.open(zarr_store_spec).result()

    shape = np.array(zarr_store.shape)
    chunk_shape = np.array(zarr_store.chunk_layout.read_chunk.shape)
    chunk_counts = np.ceil(shape / chunk_shape).astype(int)

    total_chunks = np.prod(chunk_counts)
    return total_chunks

if __name__ == "__main__":
    main()