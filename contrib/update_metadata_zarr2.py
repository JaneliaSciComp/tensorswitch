#!/usr/bin/env python3
"""
Update OME-ZARR multiscale metadata for zarr2 datasets (.zattrs files).

This script fixes zarr2 datasets where .zattrs files only contain OME XML
without proper multiscale metadata structure.

Usage:
    python update_metadata_zarr2.py <zarr_path> [max_level]
    
Arguments:
    zarr_path:  Path to the zarr2 multiscale directory containing s0, s1, etc.
    max_level:  Maximum level to include (default: auto-detect from directory)

Examples:
    python update_metadata_zarr2.py /path/to/zarr2_dataset/ 4
    python update_metadata_zarr2.py /path/to/zarr2_dataset/  # auto-detect levels
"""

import sys
import os
import argparse
import glob
import json

# Add src directory to path for tensorswitch imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from tensorswitch.utils import (create_zarr2_ome_metadata, write_zarr2_group_metadata, 
                               extract_nd2_ome_metadata)

def auto_detect_max_level(zarr_path):
    """Auto-detect the maximum level by scanning for s* directories in multiscale folder."""
    multiscale_path = os.path.join(zarr_path, 'multiscale')
    pattern = os.path.join(multiscale_path, 's*')
    level_dirs = glob.glob(pattern)
    
    if not level_dirs:
        raise ValueError(f"No s* level directories found in {multiscale_path}")
    
    levels = []
    for level_dir in level_dirs:
        dirname = os.path.basename(level_dir)
        if dirname.startswith('s') and dirname[1:].isdigit():
            levels.append(int(dirname[1:]))
    
    if not levels:
        raise ValueError(f"No valid s* level directories found in {zarr_path}")
    
    return max(levels)

def fix_zarr2_metadata(zarr_path, max_level, dry_run=False):
    """Fix zarr2 .zattrs file by creating proper multiscale structure."""
    multiscale_path = os.path.join(zarr_path, 'multiscale')
    zattrs_path = os.path.join(multiscale_path, '.zattrs')
    
    if not os.path.exists(zattrs_path):
        print(f"Error: .zattrs not found in {multiscale_path}")
        return False
    
    # Read current metadata
    with open(zattrs_path, 'r') as f:
        current_metadata = json.load(f)
    
    print(f"Current .zattrs keys: {list(current_metadata.keys())}")
    
    # Check if it already has proper multiscales structure
    if "multiscales" in current_metadata:
        print("Multiscales metadata already exists - checking if OME XML needs updating")
        if "ome_xml" in current_metadata:
            print("OME XML also present - metadata appears correct")
            return True
        else:
            print("OME XML missing - will try to add from ND2 source")
    
    # Extract OME XML if present
    existing_ome_xml = current_metadata.get('ome_xml')
    
    # Get array shape from s0 level
    s0_path = os.path.join(multiscale_path, 's0')
    if not os.path.exists(s0_path):
        print(f"Error: s0 level not found at {s0_path}")
        return False
    
    # Read s0 .zarray to get shape
    zarray_path = os.path.join(s0_path, '.zarray')
    if not os.path.exists(zarray_path):
        print(f"Error: .zarray not found at {zarray_path}")
        return False
    
    with open(zarray_path, 'r') as f:
        zarray_metadata = json.load(f)
    
    array_shape = zarray_metadata['shape']
    print(f"Detected array shape: {array_shape}")
    
    # Get image name from zarr path
    zarr_name = os.path.basename(zarr_path.rstrip('/'))
    if zarr_name.endswith('.zarr'):
        image_name = zarr_name[:-5]  # Remove .zarr extension
    else:
        image_name = zarr_name
    
    if dry_run:
        print(f"[DRY RUN] Would create proper zarr2 multiscale metadata for {image_name}")
        print(f"[DRY RUN] Array shape: {array_shape}, Max level: {max_level}")
        if existing_ome_xml:
            print("[DRY RUN] Would preserve existing OME XML")
        else:
            print("[DRY RUN] Would try to extract OME XML from corresponding ND2 file")
        return True
    
    # Create proper zarr2 metadata structure
    try:
        new_metadata = create_zarr2_ome_metadata(
            ome_xml=existing_ome_xml,
            array_shape=array_shape,
            image_name=image_name
        )
        
        # If no OME XML was preserved, try to find it from ND2 source
        if not existing_ome_xml:
            print("No OME XML found in current metadata, trying to extract from ND2 source...")
            
            # Try to find corresponding ND2 file (similar logic to zarr3 script)
            if zarr_name.endswith('_zarr2'):
                nd2_name = zarr_name[:-6] + '.nd2'  # Remove _zarr2 suffix
            else:
                nd2_name = zarr_name + '.nd2'
            
            # Common ND2 locations to check
            nd2_paths = [
                f"/groups/tavakoli/tavakolilab/data_internal/{nd2_name}",
                f"/groups/tavakoli/tavakolilab/data_internal/nd2_files/{nd2_name}",
                os.path.join(os.path.dirname(zarr_path), nd2_name),
                os.path.join(os.path.dirname(os.path.dirname(zarr_path)), nd2_name)
            ]
            
            nd2_path = None
            for path in nd2_paths:
                if os.path.exists(path):
                    nd2_path = path
                    break
            
            if nd2_path:
                print(f"Found ND2 source: {nd2_path}")
                try:
                    ome_xml = extract_nd2_ome_metadata(nd2_path)
                    if ome_xml:
                        new_metadata['ome_xml'] = ome_xml
                        print("Successfully extracted OME XML from ND2 source")
                    else:
                        print("Warning: No OME XML found in ND2 source")
                except Exception as e:
                    print(f"Warning: Could not extract OME XML from ND2: {e}")
            else:
                print(f"Warning: No corresponding ND2 file found for {zarr_name}")
        
        # Update multiscale levels if max_level > 0
        if max_level > 0:
            multiscales = new_metadata["multiscales"][0]
            datasets = []
            
            # Get s0 scale factors (default to 1.0 if not specified)
            s0_scale = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]
            
            for level in range(max_level + 1):
                scale_factor = 2 ** level  # 1, 2, 4, 8, 16 for levels 0-4
                current_scale = [sf * scale_factor for sf in s0_scale]
                
                datasets.append({
                    "path": f"s{level}",
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": current_scale
                    }]
                })
            
            multiscales["datasets"] = datasets
        
        # Write the corrected metadata
        with open(zattrs_path, 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        print(f"Successfully fixed zarr2 metadata at {zattrs_path}")
        print(f"New .zattrs keys: {list(new_metadata.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error creating zarr2 metadata: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Fix OME-ZARR multiscale metadata for zarr2 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'zarr_path',
        help='Path to the zarr2 dataset directory containing multiscale folder'
    )
    parser.add_argument(
        'max_level',
        nargs='?',
        type=int,
        help='Maximum level to include (default: auto-detect from directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Validate zarr_path exists
    if not os.path.exists(args.zarr_path):
        print(f"Error: Path does not exist: {args.zarr_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.zarr_path):
        print(f"Error: Path is not a directory: {args.zarr_path}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-detect max_level if not provided
    if args.max_level is None:
        try:
            max_level = auto_detect_max_level(args.zarr_path)
            print(f"Auto-detected max level: {max_level}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        max_level = args.max_level

    # Check if multiscale directory exists
    multiscale_path = os.path.join(args.zarr_path, 'multiscale')
    if not os.path.exists(multiscale_path):
        print(f"Error: multiscale directory not found: {multiscale_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if .zattrs exists
    zattrs_path = os.path.join(multiscale_path, '.zattrs')
    if not os.path.exists(zattrs_path):
        print(f"Error: .zattrs not found: {zattrs_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        success = fix_zarr2_metadata(args.zarr_path, max_level, dry_run=args.dry_run)
        
        if success:
            if not args.dry_run:
                print(f"\n✅ Successfully fixed zarr2 metadata!")
                print(f"Updated multiscale levels:")
                for level in range(max_level + 1):
                    scale_factor = 2 ** level
                    print(f"  s{level}: {scale_factor}x scale factor")
        else:
            print("❌ Failed to fix zarr2 metadata", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()