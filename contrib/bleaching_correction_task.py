#!/usr/bin/env python3
"""
Bleaching correction task for zarr3 datasets using linear fit method.
"""

import sys
import os
import numpy as np
import tensorstore as ts
from dask.cache import Cache

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.tensorswitch.utils import (
    get_chunk_domains,
    commit_tasks,
    get_total_chunks_from_store,
    zarr3_store_spec,
    get_input_driver,
    write_zarr3_group_metadata,
    update_ome_metadata_if_needed,
    auto_detect_max_level
)
import json
import shutil

# Set umask to allow group write access
os.umask(0o0002)

def calculate_linear_correction_factors(input_path, level="s0"):
    """Calculate linear fit bleaching correction factors for a zarr dataset"""
    print(f"Calculating correction factors for {input_path}/multiscale/{level}...")
    
    # Open the dataset
    dataset = ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file', 
            'path': f"{input_path}/multiscale/{level}"},
    }).result()
    
    print(f"Dataset shape: {dataset.shape}")
    z_planes, height, width = dataset.shape
    
    # Calculate mean intensity for each z-plane
    print("Calculating mean intensity per z-plane...")
    z_mean_intensities = []
    
    for z in range(z_planes):
        plane = dataset[z, :, :].read().result()
        mean_intensity = np.mean(plane)
        z_mean_intensities.append(mean_intensity)
        
        if z % 50 == 0 or z == z_planes - 1:
            print(f"  Processed z-plane {z+1}/{z_planes}")
    
    z_mean_intensities = np.array(z_mean_intensities)
    
    # Fit linear model and calculate correction factors
    z_indices = np.arange(len(z_mean_intensities))
    linear_fit = np.polyfit(z_indices, z_mean_intensities, 1)
    slope, intercept = linear_fit
    
    print(f"Linear fit: slope = {slope:.3f}, intercept = {intercept:.1f}")
    
    # Calculate expected intensities from linear fit
    expected_intensities = np.polyval(linear_fit, z_indices)
    
    # Calculate correction factors using linear fit
    reference_intensity = intercept  # Intensity at z=0 from linear fit
    correction_factors_linear = reference_intensity / expected_intensities
    
    print(f"Correction factors range: {correction_factors_linear.min():.3f} - {correction_factors_linear.max():.3f}")
    
    return correction_factors_linear

def copy_multiscale_metadata(input_path, output_path):
    """Copy multiscale zarr.json from input to output and update paths/names"""
    input_zarr_json = None

    # Look for zarr.json in various locations in input
    possible_input_paths = [
        os.path.join(input_path, "zarr.json"),
        os.path.join(input_path, "multiscale", "zarr.json")
    ]

    for path in possible_input_paths:
        if os.path.exists(path):
            input_zarr_json = path
            break

    if not input_zarr_json:
        print(f"Warning: No multiscale zarr.json found in {input_path}")
        return False

    # Copy to output multiscale directory
    output_multiscale_dir = os.path.join(output_path, "multiscale")
    os.makedirs(output_multiscale_dir, exist_ok=True)
    output_zarr_json = os.path.join(output_multiscale_dir, "zarr.json")

    shutil.copy2(input_zarr_json, output_zarr_json)

    # Update the name in the copied zarr.json
    try:
        with open(output_zarr_json, 'r') as f:
            metadata = json.load(f)

        # Update the name to match the output dataset
        output_name = os.path.basename(output_path).replace('.zarr', '')
        if 'attributes' in metadata and 'ome' in metadata['attributes']:
            if 'multiscales' in metadata['attributes']['ome']:
                for multiscale in metadata['attributes']['ome']['multiscales']:
                    multiscale['name'] = output_name

        with open(output_zarr_json, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Copied and updated multiscale metadata from {input_zarr_json} to {output_zarr_json}")
        return True

    except Exception as e:
        print(f"Error updating metadata: {e}")
        return False

def add_dimension_names_to_level(level_path, shape):
    """Add dimension_names to a level's zarr.json file"""
    zarr_json_path = os.path.join(level_path, "zarr.json")

    if not os.path.exists(zarr_json_path):
        print(f"Warning: zarr.json not found at {zarr_json_path}")
        return

    try:
        with open(zarr_json_path, 'r') as f:
            metadata = json.load(f)

        # Determine dimension names based on shape
        if len(shape) == 3:
            if shape[0] <= 10:
                dimension_names = ["c", "y", "x"]
            else:
                dimension_names = ["z", "y", "x"]
        elif len(shape) == 4:
            dimension_names = ["c", "z", "y", "x"]
        elif len(shape) == 5:
            dimension_names = ["t", "c", "z", "y", "x"]
        else:
            dimension_names = [f"dim_{i}" for i in range(len(shape))]

        # Add dimension_names at the same level as shape, node_type, etc.
        metadata['dimension_names'] = dimension_names

        with open(zarr_json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Added dimension_names {dimension_names} to {zarr_json_path}")

    except Exception as e:
        print(f"Error adding dimension_names to {zarr_json_path}: {e}")

def process(input_path, output_path, level="s0", use_shard=True, memory_limit=50, start_idx=0, stop_idx=None, use_ome_structure=True):
    """Apply linear fit bleaching correction to zarr dataset chunks"""
    
    print(f"Processing {input_path} -> {output_path}, level {level}")
    print(f"Chunk range: {start_idx} to {stop_idx}")
    
    # Calculate correction factors
    correction_factors = calculate_linear_correction_factors(input_path, level)
    
    # Open input dataset
    input_spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file', 
            'path': f"{input_path}/multiscale/{level}"},
    }
    input_store = ts.open(input_spec).result()
    
    print(f"Input dataset shape: {input_store.shape}, dtype: {input_store.dtype}")
    
    # Create output dataset with same properties as input
    # Convert numpy dtype to tensorstore format
    dtype_str = input_store.dtype.name  # This gives 'uint16' instead of 'dtype("uint16")'
    
    # Get the input store's chunk structure to match exactly
    input_chunk_shape = input_store.chunk_layout.read_chunk.shape
    print(f"Input chunk shape: {input_chunk_shape}")
    
    # Create output using TensorSwitch zarr3_store_spec utility
    output_path_level = f"{output_path}/multiscale/{level}" if use_ome_structure else output_path
    output_spec = zarr3_store_spec(
        output_path_level,
        input_store.shape,
        input_store.dtype.name,
        use_shard=use_shard,
        level_path=level,
        use_ome_structure=False  # Path already includes level structure
    )

    # Copy structure from input store - create only if it doesn't exist
    output_store = ts.open(
        output_spec,
        create=True,
        open=True,
        delete_existing=False
    ).result()
    
    # Enable Dask cache with 8 GB RAM
    cache = Cache(8 * 1024**3)
    cache.register()
    
    # Get chunk information
    chunk_shape = output_store.chunk_layout.write_chunk.shape
    total_chunks = get_total_chunks_from_store(output_store, chunk_shape=chunk_shape)
    print(f"Total chunks: {total_chunks}")
    print(f"Chunk shape: {chunk_shape}")
    
    # Prepare chunk domains and filter to assigned range
    linear_indices_to_process = range(start_idx, stop_idx or total_chunks)
    chunk_domains = get_chunk_domains(chunk_shape, output_store, linear_indices_to_process=linear_indices_to_process)
    
    print(f"Processing {len(linear_indices_to_process)} chunks: start={start_idx}, stop={stop_idx}", flush=True)
    
    tasks = []
    ntasks = 0
    txn = ts.Transaction()
    
    for domain in chunk_domains:
        # Read entire chunk from input
        slices = tuple(slice(min_val, max_val) for (min_val, max_val) in zip(domain.inclusive_min, domain.exclusive_max))
        chunk_data = input_store[slices].read().result()
        
        # Apply correction factors efficiently across the z-dimension of the chunk
        z_start, z_end = slices[0].start, slices[0].stop
        
        # Get the correction factors for this chunk's z-range
        chunk_correction_factors = correction_factors[z_start:z_end]
        
        # Apply corrections efficiently using broadcasting
        # Reshape correction factors to (z, 1, 1) for broadcasting across (z, y, x) chunk
        correction_factors_reshaped = chunk_correction_factors.reshape(-1, 1, 1)
        
        # Apply correction to entire chunk at once
        corrected_chunk = chunk_data * correction_factors_reshaped
        
        # Convert back to original dtype to avoid casting errors
        corrected_chunk = corrected_chunk.astype(chunk_data.dtype)
        
        # Write corrected chunk
        task = output_store[domain].with_transaction(txn).write(corrected_chunk)
        tasks.append(task)
        ntasks += 1
        
        txn = commit_tasks(tasks, txn, memory_limit=memory_limit)
        
        if ntasks % 512 == 0:
            chunk_idx = start_idx + ntasks
            print(f"Queued {ntasks} chunk writes up to chunk {chunk_idx}...", flush=True)
    
    # Final commit
    for task in tasks:
        task.result()
    txn.commit_sync()
    
    print(f"Completed bleaching correction for level {level}: {output_path} [{start_idx}:{stop_idx}]", flush=True)

    # Handle metadata only for the first chunk (start_idx == 0)
    if start_idx == 0:
        print("Setting up metadata for bleach-corrected dataset...")

        # Copy multiscale metadata from input
        if copy_multiscale_metadata(input_path, output_path):
            print("Multiscale metadata copied successfully")


        print("Metadata setup completed for bleach-corrected dataset")

def main():
    if len(sys.argv) < 4:
        print("Usage: python bleaching_correction_task.py <input_path> <output_path> <level> [use_shard] [memory_limit] [start_idx] [stop_idx] [use_ome_structure]")
        print("Example: python bleaching_correction_task.py /path/to/input.zarr /path/to/output.zarr s0 1 50 0 100 1")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    level = sys.argv[3]
    use_shard = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    memory_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    start_idx = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    stop_idx = int(sys.argv[7]) if len(sys.argv) > 7 else None
    use_ome_structure = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    
    # Create output directory structure
    if use_ome_structure:
        os.makedirs(f"{output_path}/multiscale", exist_ok=True)
    else:
        os.makedirs(output_path, exist_ok=True)
    
    # Apply bleaching correction
    process(input_path, output_path, level, use_shard, memory_limit, start_idx, stop_idx, use_ome_structure)

if __name__ == "__main__":
    main()