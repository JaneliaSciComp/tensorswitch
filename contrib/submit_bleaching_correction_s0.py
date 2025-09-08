#!/usr/bin/env python3

import subprocess
import os
import time
import sys

# Add path to use tensorswitch utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorstore as ts
import numpy as np

# Set umask to allow group write access
os.umask(0o0002)

# Paths
input_path = "/groups/tavakoli/tavakolilab/data_internal/zarr_converted/20250708_DID02_Brain_4%PFA_Atto488_40XW_005.zarr"
output_base_path = "/groups/tavakoli/tavakolilab/data_internal/zarr_converted/z-direction_bleach_corrected/20250708_DID02_Brain_4%PFA_Atto488_40XW_005_bleach_corrected.zarr"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_base_path, exist_ok=True)

# Path to the bleaching correction script (same directory)
correction_script = os.path.join(script_dir, "bleaching_correction_task.py")

print("Input path:", input_path)
print("Output path:", output_base_path) 
print("Correction script:", correction_script)

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

# Process s0 level
level = "s0"
num_volumes = 16  # Split s0 into 16 jobs

zarr_level_path = f"{input_path}/multiscale/{level}"
total_chunks = get_total_chunks(zarr_level_path)

print(f"Submitting jobs for {level}, total volumes: {num_volumes}, total chunks: {total_chunks}")

for i in range(num_volumes):
    start_idx = i * (total_chunks // num_volumes)
    stop_idx = (i + 1) * (total_chunks // num_volumes) if i < num_volumes - 1 else total_chunks
    
    job_name = f"bleach_corr_{level}_vol{i}"
    
    # LSF submission command
    command = [
        "bsub", 
        "-J", job_name,
        "-n", "2",
        "-W", "2:00",  # 2 hour wall time
        "-P", "tavakoli",
        "-g", "/scicompsoft/chend/tensorstore",
        "-o", f"{output_dir}/output_{level}_vol{i}_%J.log",
        "-e", f"{output_dir}/error_{level}_vol{i}_%J.log",
        sys.executable,
        correction_script, 
        input_path,           # input_path
        output_base_path,     # output_path  
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
print(f"Output will be saved to: {output_base_path}")