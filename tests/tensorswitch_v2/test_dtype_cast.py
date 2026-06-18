"""
Tests for --dtype output dtype casting feature.

Tests that DistributedConverter correctly casts output dtype during conversion,
including range validation, clipping for float→int, and preserving dtype when
no override is given.
"""

import os
import json
import pytest
import numpy as np
import tensorstore as ts

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter

# Synthetic test data has no voxel metadata; provide explicit defaults
_TEST_VOXEL_SIZE = {'x': 1.0, 'y': 1.0, 'z': 1.0}


def _create_zarr2_array(path, data):
    """Create a zarr2 array with the given numpy data using TensorStore."""
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': path + '/'},
        'metadata': {
            'dtype': data.dtype.str,
            'shape': list(data.shape),
            'chunks': list(data.shape),  # single chunk for simplicity
            'compressor': None,
            'order': 'C',
        },
    }
    store = ts.open(spec, create=True, delete_existing=True).result()
    store[...] = data
    return path


def _read_zarr_dtype(path):
    """Read the dtype of a zarr array from its metadata."""
    # Try zarr3 first
    zarr_json = os.path.join(path, 'zarr.json')
    if os.path.exists(zarr_json):
        with open(zarr_json) as f:
            meta = json.load(f)
        return meta.get('data_type', None)
    # Try zarr2
    zarray = os.path.join(path, '.zarray')
    if os.path.exists(zarray):
        with open(zarray) as f:
            meta = json.load(f)
        return np.dtype(meta['dtype']).name
    return None


class TestDtypeCast:
    """Tests for output dtype casting in DistributedConverter."""

    def test_float32_to_int16(self, temp_dir):
        """float32 data within int16 range converts correctly."""
        data = np.array([[[0.0, 100.5, 200.9], [300.0, -100.3, -200.7]]], dtype=np.float32)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)
        converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE,
                          output_dtype="int16")

        # Read output
        out_store = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0') + '/'},
        }, open=True).result()
        result = out_store.read().result()
        assert result.dtype == np.int16
        # float32 values get truncated (not rounded) by astype
        expected = data.astype(np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_float32_to_uint16(self, temp_dir):
        """float32 positive data converts to uint16."""
        data = np.array([[[0.0, 500.5, 65000.0]]], dtype=np.float32)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)
        converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE,
                          output_dtype="uint16")

        out_store = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0') + '/'},
        }, open=True).result()
        result = out_store.read().result()
        assert result.dtype == np.uint16

    def test_float32_to_uint8_clips(self, temp_dir):
        """float32 data outside uint8 range gets clipped, not wrapped."""
        data = np.array([[[300.0, -50.0, 128.0]]], dtype=np.float32)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)
        # Range check samples along axis 0; with shape (1,1,3) and values
        # outside uint8 range, it should raise ValueError
        with pytest.raises(ValueError, match="would clip data"):
            converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE,
                              output_dtype="uint8")

    def test_range_check_raises_on_overflow(self, temp_dir):
        """Upfront range check raises ValueError when source exceeds target range."""
        data = np.array([[[40000.0, 50000.0, 100.0]]], dtype=np.float32)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)

        with pytest.raises(ValueError, match="would clip data"):
            converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE,
                              output_dtype="int16")

    def test_int16_to_float32_widening(self, temp_dir):
        """Widening cast int16 → float32 works without issues."""
        data = np.array([[[100, -200, 32000]]], dtype=np.int16)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)
        converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE,
                          output_dtype="float32")

        out_store = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0') + '/'},
        }, open=True).result()
        result = out_store.read().result()
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, data.astype(np.float32))

    def test_no_cast_preserves_dtype(self, temp_dir):
        """When output_dtype is not set, source dtype is preserved."""
        data = np.array([[[1.5, 2.5, 3.5]]], dtype=np.float32)
        input_path = _create_zarr2_array(os.path.join(temp_dir, "in.zarr"), data)
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.auto_detect(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)
        converter.convert(verbose=False, voxel_size_override=_TEST_VOXEL_SIZE)

        out_store = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0') + '/'},
        }, open=True).result()
        result = out_store.read().result()
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, data)
