#!/usr/bin/env python3
"""
Update N5 segmentation attributes to align with image data coordinate system.

This script fixes segmentation N5 attributes when the segmentation data is missing
the full-resolution scale. It aligns downsamplingFactors with the image data's
coordinate system and adds WebKnossos-compatible multiscales metadata.

Use Case:
- Image data:        s0 at 10×10×25 nm (full resolution)
- Segmentation data: s1 at 20×20×25 nm (missing s0)
- Need to align segmentation with image coordinate system

Usage:
    python update_segmentation_attributes.py \\
        --image_n5 /path/to/image.n5 \\
        --seg_n5 /path/to/seg.n5 \\
        --seg_precomputed /path/to/seg_precomputed

Example:
    python update_segmentation_attributes.py \\
        --image_n5 /groups/tavakoli/.../ExPID124_image.n5 \\
        --seg_n5 /groups/tavakoli/.../ExPID124_seg.n5 \\
        --seg_precomputed /groups/tavakoli/.../ExPID124_segmentation

Author: Claude Code / Diyi Chen
Date: 2025-10-24
"""

import json
import os
import argparse
from pathlib import Path
import glob


def read_json(path):
    """Read and parse JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, data):
    """Write JSON file with proper formatting."""
    os.umask(0o0002)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    # Fix permissions
    os.chmod(path, 0o664)


def update_scale_attributes(seg_n5_path, image_n5_path):
    """
    Update scale-level attributes.json files with correct downsamplingFactors.

    Maps segmentation scales to image scales:
    - seg s1 → image s1 (both 20×20×25 nm)
    - seg s2 → image s2 (both 40×40×50 nm)
    etc.

    Args:
        seg_n5_path: Path to segmentation N5 dataset
        image_n5_path: Path to image N5 dataset (reference)

    Returns:
        int: Number of scales updated
    """
    print("=" * 80)
    print("STEP 1: Updating Scale-Level attributes.json")
    print("=" * 80)

    seg_scales = sorted(glob.glob(os.path.join(seg_n5_path, "ch0tp0", "s*")))

    if not seg_scales:
        print("  ✗ Error: No scales found in segmentation N5")
        return 0

    updated_count = 0

    for seg_scale_dir in seg_scales:
        scale_name = os.path.basename(seg_scale_dir)  # e.g., "s1"
        seg_attrs_path = os.path.join(seg_scale_dir, "attributes.json")

        if not os.path.exists(seg_attrs_path):
            print(f"  ⊘ Skipping {scale_name}: No attributes.json")
            continue

        # Read corresponding image scale attributes
        image_scale_path = os.path.join(image_n5_path, "ch0tp0", scale_name, "attributes.json")
        if not os.path.exists(image_scale_path):
            print(f"  ⊘ Warning: {scale_name} not found in image N5, skipping")
            continue

        seg_attrs = read_json(seg_attrs_path)
        image_attrs = read_json(image_scale_path)

        # Update downsamplingFactors to match image
        old_factors = seg_attrs.get('downsamplingFactors', [1, 1, 1])
        new_factors = image_attrs['downsamplingFactors']

        if old_factors != new_factors:
            seg_attrs['downsamplingFactors'] = new_factors
            write_json(seg_attrs_path, seg_attrs)
            print(f"  ✓ Updated {scale_name}: {old_factors} → {new_factors}")
            updated_count += 1
        else:
            print(f"  ✓ {scale_name}: Already correct {new_factors}")

    print(f"\n  Summary: Updated {updated_count} scale(s)")
    return updated_count


def create_root_attributes(seg_n5_path, image_n5_path, seg_precomputed_path):
    """
    Create complete root attributes.json with:
    - Complete MultiResolutionInfos (all seg scales)
    - Complete pixelResolution array (all seg scales)
    - Correct AnisotropyFactor
    - WebKnossos multiscales field

    Args:
        seg_n5_path: Path to segmentation N5 dataset
        image_n5_path: Path to image N5 dataset (reference for structure)
        seg_precomputed_path: Path to segmentation Precomputed (for resolution info)

    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 80)
    print("STEP 2: Creating Root attributes.json")
    print("=" * 80)

    # Read segmentation precomputed info
    seg_info_path = os.path.join(seg_precomputed_path, "info")
    if not os.path.exists(seg_info_path):
        print(f"  ✗ Error: Segmentation info file not found: {seg_info_path}")
        return False

    seg_info = read_json(seg_info_path)

    # Scan segmentation N5 scales
    seg_scales = sorted(glob.glob(os.path.join(seg_n5_path, "ch0tp0", "s*")))
    if not seg_scales:
        print("  ✗ Error: No scales found in segmentation N5")
        return False

    print(f"  Found {len(seg_scales)} scales in segmentation N5")

    # Build MultiResolutionInfos and pixelResolution
    multi_res_infos = []
    pixel_resolution = []

    for seg_scale_dir in seg_scales:
        scale_name = os.path.basename(seg_scale_dir)  # e.g., "s1"
        scale_attrs_path = os.path.join(seg_scale_dir, "attributes.json")

        if not os.path.exists(scale_attrs_path):
            continue

        scale_attrs = read_json(scale_attrs_path)
        scale_idx = int(scale_name[1:])  # "s1" → 1

        # Get resolution from scale attributes
        resolution = scale_attrs['pixelResolution']['dimensions']

        # Build MultiResolutionInfos entry
        # Use image reference for relative downsampling pattern
        if scale_idx == 1:
            relative_ds = [1, 1, 1]  # First seg scale treated as base
            absolute_ds = scale_attrs['downsamplingFactors']
        else:
            # Calculate relative to previous scale
            prev_scale_path = os.path.join(seg_n5_path, "ch0tp0", f"s{scale_idx-1}", "attributes.json")
            if os.path.exists(prev_scale_path):
                prev_attrs = read_json(prev_scale_path)
                prev_res = prev_attrs['pixelResolution']['dimensions']
                relative_ds = [
                    int(round(resolution[i] / prev_res[i])) if prev_res[i] > 0 else 1
                    for i in range(3)
                ]
            else:
                relative_ds = [2, 2, 2]  # Default
            absolute_ds = scale_attrs['downsamplingFactors']

        multi_res_infos.append({
            "relativeDownsampling": relative_ds,
            "absoluteDownsampling": absolute_ds,
            "blockSize": scale_attrs['blockSize'],
            "dimensions": scale_attrs['dimensions'],
            "dataset": f"ch0tp0/{scale_name}",
            "dataType": scale_attrs['dataType']
        })

        # Build pixelResolution entry
        pixel_resolution.append({
            "unit": scale_attrs['pixelResolution']['unit'],
            "dimensions": resolution
        })

    # Get first scale for metadata
    first_scale_attrs = read_json(os.path.join(seg_scales[0], "attributes.json"))

    # Calculate anisotropy factor from first scale resolution
    first_resolution = first_scale_attrs['pixelResolution']['dimensions']
    anisotropy = first_resolution[2] / first_resolution[0]

    # Get bounding box from first scale
    bbox_max = first_scale_attrs['dimensions']

    # Create root attributes
    root_attrs = {
        "n5": "4.0.0",
        "multiscales": [
            {
                "version": "0.1",
                "name": os.path.basename(seg_n5_path).replace('.n5', ''),
                "datasets": [
                    {
                        "path": entry['dataset'],
                        "transform": {
                            "axes": ["x", "y", "z"],
                            "scale": pixel_resolution[i]['dimensions'],
                            "units": ["nm", "nm", "nm"]
                        }
                    }
                    for i, entry in enumerate(multi_res_infos)
                ]
            }
        ],
        "Bigstitcher-Spark": {
            "InputXML": f"file:{seg_precomputed_path}",
            "NumTimepoints": 1,
            "NumChannels": 1,
            "Boundingbox_min": [0, 0, 0],
            "Boundingbox_max": bbox_max,
            "PreserveAnisotropy": True,
            "AnisotropyFactor": round(anisotropy, 2),
            "DataType": first_scale_attrs['dataType'],
            "BlockSize": first_scale_attrs['blockSize'],
            "MinIntensity": 0.0,
            "MaxIntensity": 255.0 if first_scale_attrs['dataType'] == 'uint8' else 65535.0,
            "FusionFormat": "N5",
            "MultiResolutionInfos": [multi_res_infos],
            "pixelResolution": [pixel_resolution]
        }
    }

    # Write root attributes
    root_attrs_path = os.path.join(seg_n5_path, "attributes.json")
    write_json(root_attrs_path, root_attrs)

    print(f"  ✓ Created root attributes.json")
    print(f"    Scales: {len(multi_res_infos)}")
    print(f"    AnisotropyFactor: {anisotropy:.2f}")
    print(f"    WebKnossos multiscales: Yes")
    print(f"    Bounding box: {bbox_max}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update N5 segmentation attributes to align with image data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--image_n5', required=True, help='Path to image N5 dataset (reference)')
    parser.add_argument('--seg_n5', required=True, help='Path to segmentation N5 dataset (to update)')
    parser.add_argument('--seg_precomputed', required=True, help='Path to segmentation Precomputed source')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.image_n5):
        print(f"Error: Image N5 not found: {args.image_n5}")
        return 1

    if not os.path.exists(args.seg_n5):
        print(f"Error: Segmentation N5 not found: {args.seg_n5}")
        return 1

    if not os.path.exists(args.seg_precomputed):
        print(f"Error: Segmentation Precomputed not found: {args.seg_precomputed}")
        return 1

    print("N5 Segmentation Attributes Updater")
    print("=" * 80)
    print(f"Image N5:      {args.image_n5}")
    print(f"Seg N5:        {args.seg_n5}")
    print(f"Seg Precomp:   {args.seg_precomputed}")
    print()

    # Step 1: Update scale-level attributes
    updated_scales = update_scale_attributes(args.seg_n5, args.image_n5)

    if updated_scales == 0:
        print("\n⚠ Warning: No scales were updated. Attributes may already be correct.")

    # Step 2: Create root attributes
    success = create_root_attributes(args.seg_n5, args.image_n5, args.seg_precomputed)

    if success:
        print("\n" + "=" * 80)
        print("✅ SUCCESS: Segmentation attributes updated!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Verify attributes: cat <seg_n5>/attributes.json")
        print("2. Upload to WebKnossos with URL:")
        print(f"   https://s3prfs.int.janelia.org/tavakoli-data-internal/.../")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ FAILED: Could not update attributes")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
