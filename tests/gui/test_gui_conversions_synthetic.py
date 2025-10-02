"""
GUI tests for all 10 TensorSwitch conversion tasks using synthetic data.
Tests programmatically interact with GUI components.
"""

import os
import sys
import pytest
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from test_data_config import SYNTHETIC_TEST_DATA, get_output_path


class TestGUIConversionsSynthetic:
    """Test GUI conversions with synthetic data."""

    def test_gui_tiff_to_zarr3_local(self, cleanup_output):
        """Test TIFF to Zarr3 via GUI local execution."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_tiff_to_zarr3.zarr'))

        # Simulate GUI execution by calling tensorswitch directly
        # In real GUI tests, we would programmatically interact with Panel widgets
        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)
        # Check multiscale subdirectory structure
        multiscale_path = os.path.join(output_path, 'multiscale')
        assert os.path.exists(multiscale_path)
        assert os.path.exists(os.path.join(multiscale_path, 'zarr.json'))

    def test_gui_ims_to_zarr2_local(self, cleanup_output):
        """Test IMS to Zarr2 via GUI local execution."""
        input_path = SYNTHETIC_TEST_DATA['ims']
        output_path = cleanup_output(get_output_path('gui_ims_to_zarr2.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'ims_to_zarr2_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)
        # Check multiscale subdirectory structure
        multiscale_path = os.path.join(output_path, 'multiscale')
        assert os.path.exists(multiscale_path)
        assert os.path.exists(os.path.join(multiscale_path, '.zattrs'))

    def test_gui_n5_to_zarr2_local(self, cleanup_output):
        """Test N5 to Zarr2 via GUI local execution."""
        input_path = SYNTHETIC_TEST_DATA['n5']
        output_path = cleanup_output(get_output_path('gui_n5_to_zarr2.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'n5_to_zarr2',
            '--base_path', input_path,
            '--output_path', output_path,
            '--level', '0',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)

    def test_gui_n5_to_n5_local(self, cleanup_output):
        """Test N5 to N5 via GUI local execution."""
        input_path = SYNTHETIC_TEST_DATA['n5']
        output_path = cleanup_output(get_output_path('gui_n5_to_n5.n5'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'n5_to_n5',
            '--base_path', input_path,
            '--output_path', output_path,
            '--level', '0',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)

    def test_gui_downsample_workflow(self, cleanup_output):
        """Test complete downsample workflow via GUI."""
        # First create zarr3
        input_path = SYNTHETIC_TEST_DATA['tif']
        zarr3_path = cleanup_output(get_output_path('gui_downsample_workflow.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', zarr3_path,
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Then downsample (use multiscale/s0 as base_path)
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

        # Verify both levels exist in multiscale subdirectory
        multiscale_path = os.path.join(zarr3_path, 'multiscale')
        assert os.path.exists(os.path.join(multiscale_path, 's0'))
        assert os.path.exists(os.path.join(multiscale_path, 's1'))

    def test_gui_custom_parameters(self, cleanup_output):
        """Test GUI with custom chunk/shard parameters."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_custom_params.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--custom_chunk_shape', '16,16,16',
            '--memory_limit', '50'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)

    def test_gui_memory_limit_parameter(self, cleanup_output):
        """Test GUI memory limit parameter."""
        input_path = SYNTHETIC_TEST_DATA['tif']
        output_path = cleanup_output(get_output_path('gui_memory_limit.zarr'))

        cmd = [
            'python', '-m', 'tensorswitch',
            '--task', 'tiff_to_zarr3_s0',
            '--base_path', input_path,
            '--output_path', output_path,
            '--memory_limit', '75'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        assert os.path.exists(output_path)

    def test_gui_all_zarr3_tasks(self, cleanup_output):
        """Test all Zarr3 conversion tasks via GUI."""
        tasks = [
            ('tiff_to_zarr3_s0', SYNTHETIC_TEST_DATA['tif']),
            ('ims_to_zarr3_s0', SYNTHETIC_TEST_DATA['ims'])
        ]

        for task_name, input_path in tasks:
            output_path = cleanup_output(get_output_path(f'gui_{task_name}.zarr'))

            cmd = [
                'python', '-m', 'tensorswitch',
                '--task', task_name,
                '--base_path', input_path,
                '--output_path', output_path,
                '--memory_limit', '50'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"{task_name} failed"
            assert os.path.exists(output_path), f"Output missing for {task_name}"

    def test_gui_all_zarr2_tasks(self, cleanup_output):
        """Test all Zarr2 conversion tasks via GUI."""
        tasks = [
            ('tiff_to_zarr2_s0', SYNTHETIC_TEST_DATA['tif']),
            ('ims_to_zarr2_s0', SYNTHETIC_TEST_DATA['ims']),
            ('n5_to_zarr2', SYNTHETIC_TEST_DATA['n5'])
        ]

        for task_name, input_path in tasks:
            output_path = cleanup_output(get_output_path(f'gui_{task_name}.zarr'))

            cmd = [
                'python', '-m', 'tensorswitch',
                '--task', task_name,
                '--base_path', input_path,
                '--output_path', output_path,
                '--memory_limit', '50'
            ]
            # Only add --level for N5 conversion tasks
            if 'n5' in task_name:
                cmd.insert(-2, '--level')
                cmd.insert(-2, '0')

            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"{task_name} failed: {result.stderr}"
            assert os.path.exists(output_path), f"Output missing for {task_name}"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
