#!/usr/bin/env python3
"""
Update OME-ZARR multiscale metadata for zarr3 datasets.

This script updates the OME-ZARR metadata in zarr.json to include all multiscale
levels from s0 to the specified maximum level, with proper scale factors.

Usage:
    python update_metadata.py <zarr_path> [max_level]
    
Arguments:
    zarr_path:  Path to the zarr3 multiscale directory containing s0, s1, etc.
    max_level:  Maximum level to include (default: auto-detect from directory)

Examples:
    python update_metadata.py /path/to/multiscale/ 4
    python update_metadata.py /path/to/multiscale/  # auto-detect levels
"""

import sys
import os
import argparse
import glob

# Add src directory to path for tensorswitch imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from tensorswitch.utils import update_ome_multiscale_metadata, extract_nd2_ome_metadata

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

def check_and_add_ome_xml(multiscale_path, dry_run=False):
    """Check if ome_xml metadata exists and add it from corresponding ND2 file if missing."""
    import json
    
    zarr_json_path = os.path.join(multiscale_path, 'zarr.json')
    if not os.path.exists(zarr_json_path):
        print(f"Warning: zarr.json not found in {multiscale_path}")
        return False
    
    # Read current metadata
    with open(zarr_json_path, 'r') as f:
        metadata = json.load(f)
    
    # Check if ome_xml already exists
    ome_attrs = metadata.get('attributes', {}).get('ome', {})
    if 'ome_xml' in ome_attrs:
        print(f"ome_xml already exists in {multiscale_path}")
        #return False
    
    # Try to find corresponding ND2 file
    print("multiscale_path:", multiscale_path)
    zarr_path = multiscale_path.removesuffix('multiscale')
    zarr_path = zarr_path.rstrip('/')
    zarr_name = os.path.basename(zarr_path)
    print("zarr_name_after:", zarr_name)
    print(f"{zarr_path=}")
    if zarr_name.endswith('.zarr'):
        nd2_name = zarr_name[:-5] + '.nd2'
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
        print("nd2_path path:", path)
        if os.path.exists(path):
            nd2_path = path
            break
    
    if not nd2_path:
        print(f"Warning: No corresponding ND2 file found for {zarr_name}")
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would extract ome_xml from {nd2_path} and add to {zarr_path}")
        return True
    
    try:
        print(f"Extracting ome_xml metadata from {nd2_path}")
        ome_xml = extract_nd2_ome_metadata(nd2_path)
        
        if ome_xml:
            # Add ome_xml to metadata
            metadata['attributes']['ome_xml'] = ome_xml
            # Remove the old ome_xml under ['ome']['ome_xml']
            if metadata['attributes']['ome'] and metadata['attributes']['ome']['ome_xml']:
                metadata['attributes']['ome'].pop('ome_xml')
            
            # Write back to zarr.json
            with open(zarr_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully added ome_xml metadata to {zarr_path}")
            return True
        else:
            print(f"Warning: No ome_xml metadata found in {nd2_path}")
            return False
            
    except Exception as e:
        print(f"Error extracting ome_xml from {nd2_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Update OME-ZARR multiscale metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'zarr_path',
        help='Path to the zarr3 multiscale directory containing s0, s1, etc.'
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
    parser.add_argument(
        '--check-ome-xml',
        action='store_true',
        default=True,
        help='Check for missing ome_xml metadata and add from corresponding ND2 files (default: True)'
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

    # Define multiscale_path before using it
    multiscale_path = os.path.join(args.zarr_path, 'multiscale')
    print("multiscale_path:", multiscale_path)
    
    # Verify that the specified levels exist
    missing_levels = []
    for level in range(max_level + 1):
        level_path = os.path.join(multiscale_path, f's{level}')
        if not os.path.exists(level_path):
            missing_levels.append(f's{level}')
    
    if missing_levels:
        print(f"Warning: Missing level directories: {', '.join(missing_levels)}")
    
    # Check if zarr.json exists in multiscale folder
    zarr_json_path = os.path.join(multiscale_path, 'zarr.json')
    if not os.path.exists(zarr_json_path):
        print(f"Error: zarr.json not found in {multiscale_path}", file=sys.stderr)
        print("Make sure this is a valid OME-ZARR multiscale directory", file=sys.stderr)
        sys.exit(1)
    
    if args.dry_run:
        print(f"[DRY RUN] Would update OME metadata for {args.zarr_path} with levels s0-s{max_level}")
        if args.check_ome_xml:
            check_and_add_ome_xml(multiscale_path, dry_run=True)
        return
    
    try:
        print(f"Updating OME metadata for {args.zarr_path} with levels s0-s{max_level}")
        update_ome_multiscale_metadata(args.zarr_path, max_level=max_level)
        print("OME metadata updated successfully!")
        
        # Check and add ome_xml if requested
        if args.check_ome_xml:
            print("\nChecking for ome_xml metadata...")
            check_and_add_ome_xml(multiscale_path, dry_run=False)
        
        # Show summary of updated levels
        print(f"\nUpdated multiscale levels:")
        for level in range(max_level + 1):
            scale_factor = 2 ** level
            print(f"  s{level}: {scale_factor}x scale factor")
            
    except Exception as e:
        print(f"Error updating metadata: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()