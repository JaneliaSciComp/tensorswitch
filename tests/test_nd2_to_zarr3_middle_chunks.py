#!/usr/bin/env python3
"""
Test script for ND2 to OME-Zarr conversion.
Processes 10 chunks from the middle of the dataset to test functionality.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from tensorswitch.tasks.nd2_to_zarr3_s0 import process
from tensorswitch.utils import load_nd2_stack, zarr3_store_spec, get_total_chunks_from_store
import tensorstore as ts

def test_nd2_to_zarr_middle_chunks():
    """Test ND2 to OME-Zarr conversion with middle chunks."""
    
    # Set up paths
    nd2_path = '/groups/tavakoli/tavakolilab/data_internal/20250708_DID02_Brain_4%PFA_Atto488_40XW_005.nd2'
    output_dir = os.path.abspath('./nd2_conversion_test')
    output_path = os.path.join(output_dir, 'test_nd2_middle_chunks.zarr')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First, calculate total chunks to find the middle
    print("Calculating total chunks...")
    volume = load_nd2_stack(nd2_path)
    print(f"Volume shape: {volume.shape}")
    
    # Create store spec to get chunk info
    store_spec = zarr3_store_spec(
        path=output_path,
        shape=volume.shape,
        dtype=str(volume.dtype),
        use_shard=False
    )
    
    # Open store to get chunk layout
    temp_store = ts.open(store_spec, create=True, delete_existing=True).result()
    total_chunks = get_total_chunks_from_store(temp_store)
    
    # Close the volume to avoid the warning
    if hasattr(volume, '_meta') and hasattr(volume._meta, 'close'):
        try:
            volume._meta.close()
        except:
            pass
    print(f"Total chunks: {total_chunks}")
    
    # Calculate middle chunk range (10 chunks from the middle)
    middle_start = (total_chunks // 2) - 5
    middle_end = middle_start + 10
    print(f"Processing chunks {middle_start} to {middle_end-1} (middle of dataset)")
    
    # Test conversion with middle chunks
    print("Testing ND2 to OME-Zarr conversion...")
    process(nd2_path, output_path, use_shard=False, start_idx=middle_start, stop_idx=middle_end)
    
    print(f"\nConversion complete! Output saved to: {output_path}")
    print(f"Original ND2 file remains unchanged at: {nd2_path}")
    
    # Verify zarr3 OME-ZARR output exists
    assert os.path.exists(output_path), f"Output directory {output_path} was not created"
    assert os.path.exists(os.path.join(output_path, 'zarr.json')), "Group-level zarr.json not found"
    assert os.path.exists(os.path.join(output_path, 's0')), "s0 array directory not found"
    assert os.path.exists(os.path.join(output_path, 's0', 'zarr.json')), "s0 array zarr.json not found"
    
    # Verify OME-ZARR metadata structure
    import json
    with open(os.path.join(output_path, 'zarr.json'), 'r') as f:
        metadata = json.load(f)
    
    assert metadata.get('zarr_format') == 3, "Not zarr3 format"
    assert metadata.get('node_type') == 'group', "Not a group node"
    assert 'ome' in metadata.get('attributes', {}), "OME metadata missing"
    assert 'multiscales' in metadata['attributes']['ome'], "Multiscales metadata missing"
    
    multiscales = metadata['attributes']['ome']['multiscales'][0]
    assert multiscales.get('version') == '0.5', "Wrong OME-ZARR version"
    assert len(multiscales.get('axes', [])) == 3, "Wrong number of axes for 3D data"
    assert multiscales['datasets'][0]['path'] == 's0', "Wrong dataset path"
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_nd2_to_zarr_middle_chunks()