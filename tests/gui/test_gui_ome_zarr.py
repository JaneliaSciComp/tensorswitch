"""
GUI tests for OME-Zarr metadata validation through GUI workflows.
"""

import os
import sys
import json
import pytest
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from test_data_config import SYNTHETIC_TEST_DATA, get_output_path
from conftest import validate_zarr3_ome_metadata, validate_zarr2_ome_metadata


class TestGUIOMEZarr:
    """Test OME-Zarr metadata through GUI workflows."""

    def test_gui_zarr3_ome_metadata(self, cleanup_output):
        """Test Zarr3 OME metadata generation via GUI."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_ome_zarr3.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Validate OME-Zarr structure
        assert validate_zarr3_ome_metadata(output_path)

        # Check metadata details (in multiscale subdirectory)
        multiscale_path = os.path.join(output_path, 'multiscale')
        with open(os.path.join(multiscale_path, 'zarr.json'), 'r') as f:
            metadata = json.load(f)

        assert 'ome' in metadata['attributes']
        assert 'multiscales' in metadata['attributes']['ome']

    def test_gui_zarr2_ome_metadata(self, cleanup_output):
        """Test Zarr2 OME metadata generation via GUI."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_ome_zarr2.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Validate OME-Zarr structure
        assert validate_zarr2_ome_metadata(output_path)

        # Check metadata details (in multiscale subdirectory)
        multiscale_path = os.path.join(output_path, 'multiscale')
        with open(os.path.join(multiscale_path, '.zattrs'), 'r') as f:
            metadata = json.load(f)

        assert 'multiscales' in metadata

    def test_gui_multiscale_metadata_update(self, cleanup_output):
        """Test that GUI properly updates metadata after downsampling."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_multiscale.zarr'))

        # Create s0
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Downsample to s1 (use multiscale/s0 as base_path)
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

        # Check metadata includes both levels (in multiscale subdirectory)
        multiscale_path = os.path.join(output_path, 'multiscale')
        with open(os.path.join(multiscale_path, 'zarr.json'), 'r') as f:
            metadata = json.load(f)

        datasets = metadata['attributes']['ome']['multiscales'][0]['datasets']
        paths = [d['path'] for d in datasets]

        assert 's0' in paths
        assert 's1' in paths

    def test_gui_ome_structure_parameter(self, cleanup_output):
        """Test GUI OME structure parameter setting."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_ome_structure.zarr'))

        # This simulates the use_ome_structure parameter in GUI
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Should have OME structure by default (in multiscale subdirectory)
        multiscale_path = os.path.join(output_path, 'multiscale')
        assert os.path.exists(multiscale_path)
        assert os.path.exists(os.path.join(multiscale_path, 'zarr.json'))
        validate_zarr3_ome_metadata(output_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
