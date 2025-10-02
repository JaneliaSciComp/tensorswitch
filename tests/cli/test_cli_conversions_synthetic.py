"""
CLI tests for all 10 TensorSwitch conversion tasks using synthetic data.
Tests process complete small files for quick validation.
"""

import os
import sys
import pytest
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from test_data_config import SYNTHETIC_TEST_DATA, get_output_path
from conftest import validate_zarr3_ome_metadata, validate_zarr2_ome_metadata


class TestSyntheticDataConversions:
    """Test CLI conversions with small synthetic data files."""

    def test_tiff_to_zarr3_s0(self, cleanup_output):
        """Test TIFF to Zarr3 conversion."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('tiff_to_zarr3_synthetic.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr3_ome_metadata(output_path)

    def test_tiff_to_zarr2_s0(self, cleanup_output):
        """Test TIFF to Zarr2 conversion."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('tiff_to_zarr2_synthetic.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr2_ome_metadata(output_path)

    def test_ims_to_zarr3_s0(self, cleanup_output):
        """Test IMS to Zarr3 conversion."""
        input_path = SYNTHETIC_TEST_DATA['ims']
        output_path = cleanup_output(get_output_path('ims_to_zarr3_synthetic.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'ims_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr3_ome_metadata(output_path)

    def test_ims_to_zarr2_s0(self, cleanup_output):
        """Test IMS to Zarr2 conversion."""
        input_path = SYNTHETIC_TEST_DATA['ims']
        output_path = cleanup_output(get_output_path('ims_to_zarr2_synthetic.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'ims_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr2_ome_metadata(output_path)

    def test_n5_to_zarr2(self, cleanup_output):
        """Test N5 to Zarr2 conversion."""
        input_path = SYNTHETIC_TEST_DATA['n5']
        output_path = cleanup_output(get_output_path('n5_to_zarr2_synthetic.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'n5_to_zarr2',
            '--base_path', input_path,
            '--output_path', output_path,
            '--level', '0',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)

    def test_n5_to_n5(self, cleanup_output):
        """Test N5 to N5 rechunking."""
        input_path = SYNTHETIC_TEST_DATA['n5']
        output_path = cleanup_output(get_output_path('n5_to_n5_synthetic.n5'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'n5_to_n5',
            '--base_path', input_path,
            '--output_path', output_path,
            '--level', '0',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)

    def test_downsample_shard_zarr3(self, cleanup_output):
        """Test Zarr3 downsampling with sharding."""
        # First create a zarr3 file from TIFF
        input_path = SYNTHETIC_TEST_DATA['tif']
        zarr3_path = cleanup_output(get_output_path('zarr3_for_downsample.zarr'))

        # Convert TIFF to Zarr3
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', zarr3_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Zarr3 creation failed: {result.stderr}"

        # Now downsample
        s0_path = os.path.join(zarr3_path, 'multiscale', 's0')
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'downsample_shard_zarr3',
            '--base_path', s0_path,
            '--output_path', zarr3_path,
            '--level', '1',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Downsample failed: {result.stderr}"

        # Verify s1 was created
        s1_path = os.path.join(zarr3_path, 'multiscale', 's1')
        assert os.path.exists(s1_path), "s1 level not created"

    def test_downsample_zarr2(self, cleanup_output):
        """Test Zarr2 downsampling."""
        # First create a zarr2 file from TIFF
        input_path = SYNTHETIC_TEST_DATA['tif']
        zarr2_path = cleanup_output(get_output_path('zarr2_for_downsample.zarr'))

        # Convert TIFF to Zarr2
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', zarr2_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Zarr2 creation failed: {result.stderr}"

        # Now downsample
        s0_path = os.path.join(zarr2_path, 'multiscale', 's0')
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'downsample_zarr2',
            '--base_path', s0_path,
            '--output_path', zarr2_path,
            '--level', '1',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Downsample failed: {result.stderr}"

        # Verify s1 was created
        s1_path = os.path.join(zarr2_path, 'multiscale', 's1')
        assert os.path.exists(s1_path), "s1 level not created"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
