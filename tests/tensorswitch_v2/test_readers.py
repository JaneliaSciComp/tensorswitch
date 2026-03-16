"""
Unit tests for tensorswitch_v2 readers.

Tests all reader implementations (Tier 1, 2, 3) for correct behavior.
"""

import os
import pytest
import numpy as np

from tensorswitch_v2.readers import (
    BaseReader,
    N5Reader,
    Zarr3Reader,
    Zarr2Reader,
    PrecomputedReader,
    TiffReader,
)
from tensorswitch_v2.api import Readers


class TestTiffReader:
    """Tests for TiffReader (Tier 2)."""

    def test_init(self, sample_tiff_path):
        """Test TiffReader initialization."""
        reader = TiffReader(sample_tiff_path)
        assert reader.path == sample_tiff_path

    def test_get_tensorstore(self, sample_tiff_path):
        """Test getting TensorStore from TIFF."""
        import tensorstore as ts
        reader = TiffReader(sample_tiff_path)
        store = reader.get_tensorstore()

        assert isinstance(store, ts.TensorStore)
        assert len(store.shape) >= 3

    def test_get_metadata(self, sample_tiff_path):
        """Test getting metadata from TIFF."""
        reader = TiffReader(sample_tiff_path)
        metadata = reader.get_metadata()

        assert metadata is not None
        assert 'shape' in metadata
        assert metadata['shape'] == (32, 64, 64)

    def test_factory_method(self, sample_tiff_path):
        """Test Readers.tiff() factory method."""
        reader = Readers.tiff(sample_tiff_path)
        assert isinstance(reader, TiffReader)

    def test_auto_detect_tiff(self, sample_tiff_path):
        """Test auto_detect returns TiffReader for .tif files."""
        reader = Readers.auto_detect(sample_tiff_path)
        assert isinstance(reader, TiffReader)


class TestZarr3Reader:
    """Tests for Zarr3Reader (Tier 1)."""

    def test_init(self, sample_zarr3_path):
        """Test Zarr3Reader initialization."""
        reader = Zarr3Reader(sample_zarr3_path)
        assert reader.path == sample_zarr3_path

    def test_get_tensorstore(self, sample_zarr3_path):
        """Test getting TensorStore from Zarr3."""
        import tensorstore as ts
        reader = Zarr3Reader(sample_zarr3_path, dataset_path="s0")
        store = reader.get_tensorstore()

        assert isinstance(store, ts.TensorStore)
        assert store.shape == (32, 64, 64)

    def test_get_metadata(self, sample_zarr3_path):
        """Test getting metadata from Zarr3."""
        reader = Zarr3Reader(sample_zarr3_path, dataset_path="s0")
        metadata = reader.get_metadata()

        assert metadata is not None
        assert 'shape' in metadata
        assert metadata['shape'] == (32, 64, 64)

    def test_factory_method(self, sample_zarr3_path):
        """Test Readers.zarr3() factory method."""
        reader = Readers.zarr3(sample_zarr3_path, dataset_path="s0")
        assert isinstance(reader, Zarr3Reader)


class TestZarr2Reader:
    """Tests for Zarr2Reader (Tier 1)."""

    def test_init(self, sample_zarr2_path):
        """Test Zarr2Reader initialization."""
        reader = Zarr2Reader(sample_zarr2_path)
        assert reader.path == sample_zarr2_path

    def test_get_tensorstore(self, sample_zarr2_path):
        """Test getting TensorStore from Zarr2."""
        import tensorstore as ts
        reader = Zarr2Reader(sample_zarr2_path, dataset_path="s0")
        store = reader.get_tensorstore()

        assert isinstance(store, ts.TensorStore)
        assert store.shape == (32, 64, 64)

    def test_factory_method(self, sample_zarr2_path):
        """Test Readers.zarr2() factory method."""
        reader = Readers.zarr2(sample_zarr2_path, dataset_path="s0")
        assert isinstance(reader, Zarr2Reader)


class TestN5Reader:
    """Tests for N5Reader (Tier 1)."""

    def test_init(self, sample_n5_path):
        """Test N5Reader initialization."""
        reader = N5Reader(sample_n5_path)
        assert reader.path == sample_n5_path

    def test_get_tensorstore(self, sample_n5_path):
        """Test getting TensorStore from N5."""
        import tensorstore as ts
        reader = N5Reader(sample_n5_path, dataset_path="s0")
        store = reader.get_tensorstore()

        assert isinstance(store, ts.TensorStore)
        assert store.shape == (32, 64, 64)

    def test_get_metadata(self, sample_n5_path):
        """Test getting metadata from N5."""
        reader = N5Reader(sample_n5_path, dataset_path="s0")
        metadata = reader.get_metadata()

        assert metadata is not None
        assert 'shape' in metadata
        assert metadata['shape'] == (32, 64, 64)

    def test_factory_method(self, sample_n5_path):
        """Test Readers.n5() factory method."""
        reader = Readers.n5(sample_n5_path, dataset_path="s0")
        assert isinstance(reader, N5Reader)

    def test_auto_detect_n5(self, sample_n5_path):
        """Test auto_detect returns N5Reader for .n5 paths."""
        reader = Readers.auto_detect(sample_n5_path)
        assert isinstance(reader, N5Reader)


class TestReadersFactory:
    """Tests for Readers factory class."""

    def test_auto_detect_tiff(self, sample_tiff_path):
        """Test auto-detection for TIFF files."""
        reader = Readers.auto_detect(sample_tiff_path)
        assert isinstance(reader, TiffReader)

    def test_auto_detect_n5(self, sample_n5_path):
        """Test auto-detection for N5 datasets."""
        reader = Readers.auto_detect(sample_n5_path)
        assert isinstance(reader, N5Reader)

    def test_auto_detect_zarr3(self, sample_zarr3_path):
        """Test auto-detection for Zarr3 datasets."""
        reader = Readers.auto_detect(sample_zarr3_path)
        assert isinstance(reader, (Zarr3Reader, Zarr2Reader))

    def test_explicit_reader_selection(self, sample_tiff_path):
        """Test explicit reader selection via factory."""
        reader = Readers.tiff(sample_tiff_path)
        assert isinstance(reader, TiffReader)


class TestReaderDataAccess:
    """Tests for actual data access through readers."""

    def test_tiff_read_data(self, sample_tiff_path, sample_3d_array):
        """Test reading data through TiffReader.

        TiffReader wraps dask arrays via ts.virtual_chunked, so
        get_tensorstore() returns a real TensorStore that can be read.
        """
        reader = TiffReader(sample_tiff_path)
        store = reader.get_tensorstore()

        data = store.read().result()

        assert data.shape == sample_3d_array.shape
        assert np.array_equal(data, sample_3d_array)

    def test_zarr3_read_data(self, sample_zarr3_path, sample_3d_array):
        """Test reading data through Zarr3Reader."""
        reader = Zarr3Reader(sample_zarr3_path, dataset_path="s0")
        store = reader.get_tensorstore()

        data = store[...].read().result()

        assert data.shape == sample_3d_array.shape
        assert np.array_equal(data, sample_3d_array)

    def test_n5_read_data(self, sample_n5_path, sample_3d_array):
        """Test reading data through N5Reader."""
        reader = N5Reader(sample_n5_path, dataset_path="s0")
        store = reader.get_tensorstore()

        data = store[...].read().result()

        assert data.shape == sample_3d_array.shape
        assert np.array_equal(data, sample_3d_array)
