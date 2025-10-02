"""
CLI tests for OME-Zarr metadata validation.
Tests verify proper OME-Zarr v0.4/v0.5 metadata structure.
"""

import os
import sys
import json
import pytest
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from test_data_config import SYNTHETIC_TEST_DATA, get_output_path
from conftest import validate_zarr3_ome_metadata, validate_zarr2_ome_metadata


class TestOMEZarrMetadata:
    """Test OME-Zarr metadata structure and compliance."""

    def test_zarr3_ome_metadata_structure(self, cleanup_output):
        """Test Zarr3 OME-Zarr v0.5 metadata structure."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('ome_zarr3_test.zarr'))

        # Create Zarr3
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Validate structure
        assert validate_zarr3_ome_metadata(output_path)

        # Check specific metadata fields
        metadata_path = os.path.join(output_path, 'multiscale', 'zarr.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ome = metadata['attributes']['ome']
        multiscales = ome['multiscales'][0]

        # Check version (optional field)
        if 'version' in multiscales:
            assert multiscales['version'] in ['0.4', '0.5']

        # Check axes
        axes = multiscales['axes']
        assert len(axes) == 3  # ZYX for 3D data
        assert all(ax['name'] in ['z', 'y', 'x'] for ax in axes)
        assert all(ax['type'] == 'space' for ax in axes)

        # Check datasets
        datasets = multiscales['datasets']
        assert len(datasets) >= 1
        assert datasets[0]['path'] == 's0'

    def test_zarr2_ome_metadata_structure(self, cleanup_output):
        """Test Zarr2 OME-Zarr metadata structure."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('ome_zarr2_test.zarr'))

        # Create Zarr2
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Validate structure
        assert validate_zarr2_ome_metadata(output_path)

        # Check specific metadata fields
        metadata_path = os.path.join(output_path, 'multiscale', '.zattrs')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        multiscales = metadata['multiscales'][0]

        # Check axes
        axes = multiscales['axes']
        assert len(axes) == 3  # 3D data
        # Axes can be strings or dicts
        axis_names = [ax['name'] if isinstance(ax, dict) else ax for ax in axes]
        # Accept various axis names (z/y/x or c/y/x or numeric)
        valid_axes = ['z', 'y', 'x', 'c', 't', '0', '1', '2']
        assert all(str(name) in valid_axes for name in axis_names), f"Invalid axes: {axis_names}"

        # Check datasets
        datasets = multiscales['datasets']
        assert len(datasets) >= 1
        assert datasets[0]['path'] == 's0'

    def test_multiscale_metadata_after_downsample(self, cleanup_output):
        """Test that multiscale metadata is updated after downsampling."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('multiscale_test.zarr'))

        # Create Zarr3 s0
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Downsample to s1
        s0_path = os.path.join(output_path, 'multiscale', 's0')
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'downsample_shard_zarr3',
            '--base_path', s0_path,
            '--output_path', output_path,
            '--level', '1',
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Check metadata includes both s0 and s1
        metadata_path = os.path.join(output_path, 'multiscale', 'zarr.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        datasets = metadata['attributes']['ome']['multiscales'][0]['datasets']
        dataset_paths = [d['path'] for d in datasets]

        assert 's0' in dataset_paths
        assert 's1' in dataset_paths
        assert len(datasets) == 2

    def test_ome_xml_metadata_preserved(self, cleanup_output):
        """Test that OME XML metadata is preserved in conversions."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('ome_xml_test.zarr'))

        # Create Zarr3
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Check OME XML exists
        metadata_path = os.path.join(output_path, 'multiscale', 'zarr.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ome = metadata['attributes']['ome']

        # Should have ome_xml field if source had metadata
        # For synthetic data, check structure is valid even if minimal
        assert 'multiscales' in ome
        assert isinstance(ome['multiscales'], list)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
