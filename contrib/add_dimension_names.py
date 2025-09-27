#!/usr/bin/env python3
"""
Add dimension_names to zarr.json files for all levels in a dataset.
Usage: python add_dimension_names.py <dataset_path>
"""

import sys
import os
import json
import tensorstore as ts

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python add_dimension_names.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Get shape from s0 level
    s0_path = os.path.join(dataset_path, "multiscale", "s0")

    try:
        dataset = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': s0_path}
        }).result()
        shape = dataset.shape
        print(f"Dataset shape: {shape}")

        # Add dimension_names to all levels
        multiscale_path = os.path.join(dataset_path, "multiscale")
        for level in ['s0', 's1', 's2', 's3', 's4']:
            level_path = os.path.join(multiscale_path, level)
            if os.path.exists(level_path):
                add_dimension_names_to_level(level_path, shape)

        print("Successfully added dimension_names to all levels")

    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    main()