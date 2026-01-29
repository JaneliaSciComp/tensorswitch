#!/usr/bin/env python
"""
Test script for BIOIO adapter conversion on cluster.
Forces use of BIOIO adapter (Tier 3) instead of Tier 2 readers.
"""

import sys
import os
import argparse
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from tensorswitch_v2.readers import BIOIOReader
from tensorswitch_v2.writers import Zarr3Writer
from tensorswitch_v2.core.converter import DistributedConverter


def main():
    parser = argparse.ArgumentParser(description="Test BIOIO adapter conversion")
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output zarr path")
    parser.add_argument("--start_idx", type=int, default=0, help="Start chunk index")
    parser.add_argument("--stop_idx", type=int, default=None, help="Stop chunk index")
    parser.add_argument("--delete_existing", action="store_true", help="Delete existing output")
    args = parser.parse_args()

    # Set team permissions
    os.umask(0o0002)

    print("=" * 70)
    print("BIOIO Adapter Conversion Test")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Clean up if requested
    if args.delete_existing and os.path.exists(args.output):
        print(f"Deleting existing output: {args.output}")
        shutil.rmtree(args.output)

    # Create BIOIO reader (forces Tier 3)
    print("\nCreating BIOIO reader...")
    reader = BIOIOReader(args.input)
    spec = reader.get_tensorstore_spec()
    voxels = reader.get_voxel_sizes()

    print(f"  Shape: {spec['schema']['shape']}")
    print(f"  Dtype: {spec['schema']['dtype']}")
    print(f"  Dims: {spec['schema']['dimension_names']}")
    print(f"  Voxels: {voxels}")

    # Create writer
    print("\nCreating Zarr3 writer...")
    writer = Zarr3Writer(args.output)

    # Run conversion
    print("\nRunning conversion...")
    converter = DistributedConverter(reader, writer)

    # Convert all chunks (or specified range)
    stats = converter.convert(
        start_idx=args.start_idx,
        stop_idx=args.stop_idx,  # None = all chunks
        progress_interval=10,
        delete_existing=args.delete_existing
    )

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print(f"  Chunks processed: {stats.get('chunks_processed', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
