"""
Unit tests for tensorswitch_v2 writers.

Tests all writer implementations (Zarr3, Zarr2, N5) for correct behavior.
"""

import os
import json
import pytest
import numpy as np
import tensorstore as ts

from tensorswitch_v2.writers import (
    BaseWriter,
    Zarr3Writer,
    Zarr2Writer,
    N5Writer,
)
from tensorswitch_v2.api import Writers

from .conftest import validate_zarr3_output, validate_zarr2_output, validate_n5_output


class TestZarr3Writer:
    """Tests for Zarr3Writer."""

    def test_init(self, temp_dir):
        """Test Zarr3Writer initialization."""
        output_path = os.path.join(temp_dir, "output.zarr")
        writer = Zarr3Writer(output_path)

        assert writer.output_path == output_path
        assert writer.use_sharding == True  # Default

    def test_supports_sharding(self, temp_dir):
        """Test that Zarr3Writer supports sharding."""
        writer = Zarr3Writer(os.path.join(temp_dir, "out.zarr"))
        assert writer.supports_sharding() == True

    def test_create_output_spec(self, temp_dir):
        """Test creating output spec."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr3Writer(output_path)

        spec = writer.create_output_spec(
            shape=(32, 64, 64),
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )

        assert spec['driver'] == 'zarr3'
        assert 'kvstore' in spec
        assert 'metadata' in spec

    def test_write_and_read(self, temp_dir, sample_3d_array):
        """Test end-to-end write and read."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr3Writer(output_path, use_sharding=False)

        # Create and open store
        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype=str(sample_3d_array.dtype),
            chunk_shape=(16, 32, 32)
        )
        store = writer.open_store(spec, create=True, delete_existing=True)

        # Write data directly
        store[...] = sample_3d_array

        # Verify by reading back
        read_data = store[...].read().result()
        assert np.array_equal(read_data, sample_3d_array)

    def test_write_metadata(self, temp_dir, sample_3d_array):
        """Test metadata writing."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr3Writer(output_path)

        # Create store first
        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )
        writer.open_store(spec, create=True, delete_existing=True)

        # Write metadata
        writer.write_metadata(
            voxel_sizes={'x': 0.5, 'y': 0.5, 'z': 1.0},
            array_shape=sample_3d_array.shape
        )

        # Verify zarr.json exists
        zarr_json_path = os.path.join(output_path, "zarr.json")
        assert os.path.exists(zarr_json_path)

        with open(zarr_json_path) as f:
            metadata = json.load(f)

        assert metadata['zarr_format'] == 3
        assert 'ome' in metadata.get('attributes', {})

    def test_factory_method(self, temp_dir):
        """Test Writers.zarr3() factory method."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Writers.zarr3(output_path)

        assert isinstance(writer, Zarr3Writer)
        assert writer.use_sharding == True


class TestZarr2Writer:
    """Tests for Zarr2Writer."""

    def test_init(self, temp_dir):
        """Test Zarr2Writer initialization."""
        output_path = os.path.join(temp_dir, "output.zarr")
        writer = Zarr2Writer(output_path)

        assert writer.output_path == output_path

    def test_supports_sharding(self, temp_dir):
        """Test that Zarr2Writer does NOT support sharding."""
        writer = Zarr2Writer(os.path.join(temp_dir, "out.zarr"))
        assert writer.supports_sharding() == False

    def test_create_output_spec(self, temp_dir):
        """Test creating output spec."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr2Writer(output_path)

        spec = writer.create_output_spec(
            shape=(32, 64, 64),
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )

        assert spec['driver'] == 'zarr'
        assert 'kvstore' in spec
        assert 'metadata' in spec
        assert spec['metadata']['dtype'] == '|u1'

    def test_write_and_read(self, temp_dir, sample_3d_array):
        """Test end-to-end write and read."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr2Writer(output_path)

        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype=str(sample_3d_array.dtype),
            chunk_shape=(16, 32, 32)
        )
        store = writer.open_store(spec, create=True, delete_existing=True)

        # Write data
        store[...] = sample_3d_array

        # Verify
        read_data = store[...].read().result()
        assert np.array_equal(read_data, sample_3d_array)

    def test_write_metadata(self, temp_dir, sample_3d_array):
        """Test metadata writing."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Zarr2Writer(output_path)

        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )
        writer.open_store(spec, create=True, delete_existing=True)

        writer.write_metadata(
            voxel_sizes={'x': 0.5, 'y': 0.5, 'z': 1.0},
            array_shape=sample_3d_array.shape
        )

        # Verify .zgroup and .zattrs exist
        assert os.path.exists(os.path.join(output_path, ".zgroup"))
        assert os.path.exists(os.path.join(output_path, ".zattrs"))

    def test_factory_method(self, temp_dir):
        """Test Writers.zarr2() factory method."""
        output_path = os.path.join(temp_dir, "out.zarr")
        writer = Writers.zarr2(output_path)

        assert isinstance(writer, Zarr2Writer)


class TestN5Writer:
    """Tests for N5Writer."""

    def test_init(self, temp_dir):
        """Test N5Writer initialization."""
        output_path = os.path.join(temp_dir, "output.n5")
        writer = N5Writer(output_path)

        assert writer.output_path == output_path

    def test_supports_sharding(self, temp_dir):
        """Test that N5Writer does NOT support sharding."""
        writer = N5Writer(os.path.join(temp_dir, "out.n5"))
        assert writer.supports_sharding() == False

    def test_create_output_spec(self, temp_dir):
        """Test creating output spec."""
        output_path = os.path.join(temp_dir, "out.n5")
        writer = N5Writer(output_path)

        spec = writer.create_output_spec(
            shape=(32, 64, 64),
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )

        assert spec['driver'] == 'n5'
        assert 'kvstore' in spec
        assert 'metadata' in spec

    def test_write_and_read(self, temp_dir, sample_3d_array):
        """Test end-to-end write and read."""
        output_path = os.path.join(temp_dir, "out.n5")
        writer = N5Writer(output_path)

        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype=str(sample_3d_array.dtype),
            chunk_shape=(16, 32, 32)
        )
        store = writer.open_store(spec, create=True, delete_existing=True)

        # Write data
        store[...] = sample_3d_array

        # Verify
        read_data = store[...].read().result()
        assert np.array_equal(read_data, sample_3d_array)

    def test_write_metadata(self, temp_dir, sample_3d_array):
        """Test metadata writing."""
        output_path = os.path.join(temp_dir, "out.n5")
        writer = N5Writer(output_path)

        spec = writer.create_output_spec(
            shape=sample_3d_array.shape,
            dtype="uint8",
            chunk_shape=(16, 32, 32)
        )
        writer.open_store(spec, create=True, delete_existing=True)

        writer.write_metadata(
            voxel_sizes={'x': 0.5, 'y': 0.5, 'z': 1.0},
            array_shape=sample_3d_array.shape
        )

        # Verify attributes.json exists
        assert os.path.exists(os.path.join(output_path, "attributes.json"))

    def test_factory_method(self, temp_dir):
        """Test Writers.n5() factory method."""
        output_path = os.path.join(temp_dir, "out.n5")
        writer = Writers.n5(output_path)

        assert isinstance(writer, N5Writer)


class TestWritersFactory:
    """Tests for Writers factory class."""

    def test_zarr3_factory(self, temp_dir):
        """Test Writers.zarr3() factory."""
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))
        assert isinstance(writer, Zarr3Writer)
        assert writer.supports_sharding() == True

    def test_zarr2_factory(self, temp_dir):
        """Test Writers.zarr2() factory."""
        writer = Writers.zarr2(os.path.join(temp_dir, "out.zarr"))
        assert isinstance(writer, Zarr2Writer)
        assert writer.supports_sharding() == False

    def test_n5_factory(self, temp_dir):
        """Test Writers.n5() factory."""
        writer = Writers.n5(os.path.join(temp_dir, "out.n5"))
        assert isinstance(writer, N5Writer)
        assert writer.supports_sharding() == False

    def test_factory_with_options(self, temp_dir):
        """Test factory methods with custom options."""
        writer = Writers.zarr3(
            os.path.join(temp_dir, "out.zarr"),
            use_sharding=False,
            compression="gzip",
            compression_level=3
        )
        assert writer.use_sharding == False
        assert writer.compression == "gzip"
        assert writer.compression_level == 3
