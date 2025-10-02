"""
CLI tests for all 10 TensorSwitch conversion tasks using real data.
Tests process middle chunks from real files for realistic validation.
"""

import os
import sys
import pytest
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from test_data_config import REAL_TEST_DATA, get_output_path
from conftest import get_middle_chunks, validate_zarr3_ome_metadata, validate_zarr2_ome_metadata


class TestRealDataConversions:
    """Test CLI conversions with real data using middle chunks."""

    def test_nd2_to_zarr3_s0(self, cleanup_output):
        """Test ND2 to Zarr3 conversion with middle chunks."""
        nd2_config = REAL_TEST_DATA['nd2']['primary']
        nd2_path = nd2_config['path']
        output_path = cleanup_output(get_output_path('nd2_to_zarr3_real.zarr'))

        # Get middle chunk range
        start_idx, stop_idx, total = get_middle_chunks(nd2_path, 'nd2', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        # Run conversion
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'nd2_to_zarr3_s0',
            '--base_path', nd2_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Validate output
        assert os.path.exists(output_path)
        validate_zarr3_ome_metadata(output_path)

    def test_nd2_to_zarr2_s0(self, cleanup_output):
        """Test ND2 to Zarr2 conversion with middle chunks."""
        nd2_config = REAL_TEST_DATA['nd2']['primary']
        nd2_path = nd2_config['path']
        output_path = cleanup_output(get_output_path('nd2_to_zarr2_real.zarr'))

        start_idx, stop_idx, total = get_middle_chunks(nd2_path, 'nd2', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'nd2_to_zarr2_s0',
            '--base_path', nd2_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr2_ome_metadata(output_path)

    def test_ims_to_zarr3_s0(self, cleanup_output):
        """Test IMS to Zarr3 conversion with middle chunks."""
        ims_config = REAL_TEST_DATA['ims']['primary']
        ims_path = ims_config['path']
        output_path = cleanup_output(get_output_path('ims_to_zarr3_real.zarr'))

        start_idx, stop_idx, total = get_middle_chunks(ims_path, 'ims', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'ims_to_zarr3_s0',
            '--base_path', ims_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr3_ome_metadata(output_path)

    def test_ims_to_zarr2_s0(self, cleanup_output):
        """Test IMS to Zarr2 conversion with middle chunks."""
        ims_config = REAL_TEST_DATA['ims']['primary']
        ims_path = ims_config['path']
        output_path = cleanup_output(get_output_path('ims_to_zarr2_real.zarr'))

        start_idx, stop_idx, total = get_middle_chunks(ims_path, 'ims', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'ims_to_zarr2_s0',
            '--base_path', ims_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr2_ome_metadata(output_path)

    def test_tiff_to_zarr3_s0(self, cleanup_output):
        """Test TIFF to Zarr3 conversion with middle chunks."""
        tif_config = REAL_TEST_DATA['tif']['primary']
        tif_path = tif_config['path']
        output_path = cleanup_output(get_output_path('tiff_to_zarr3_real.zarr'))

        start_idx, stop_idx, total = get_middle_chunks(tif_path, 'tif', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', tif_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr3_ome_metadata(output_path)

    def test_tiff_to_zarr2_s0(self, cleanup_output):
        """Test TIFF to Zarr2 conversion with middle chunks."""
        tif_config = REAL_TEST_DATA['tif']['primary']
        tif_path = tif_config['path']
        output_path = cleanup_output(get_output_path('tiff_to_zarr2_real.zarr'))

        start_idx, stop_idx, total = get_middle_chunks(tif_path, 'tif', chunk_count=10)
        print(f"\nProcessing chunks {start_idx}-{stop_idx} of {total} total chunks")

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr2_s0',
            '--base_path', tif_path,
            '--output_path', output_path,
            '--start_idx', str(start_idx),
            '--stop_idx', str(stop_idx),
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        assert os.path.exists(output_path)
        validate_zarr2_ome_metadata(output_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
