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
        assert store.spec().to_json()['driver'] == 'zarr3'

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
        assert store.spec().to_json()['driver'] == 'zarr'

    def test_factory_method(self, sample_zarr2_path):
        """Test Readers.zarr2() factory method."""
        reader = Readers.zarr2(sample_zarr2_path, dataset_path="s0")
        assert isinstance(reader, Zarr2Reader)


class TestZarr2ReaderFallback:
    """Zarr2Reader falls back to zarr-python for incompatible compressor metadata."""

    def test_fallback_emits_warning(self, sample_zarr2_incompatible_path):
        """gzip level=-1 triggers fallback with a UserWarning."""
        import warnings
        reader = Zarr2Reader(sample_zarr2_incompatible_path, dataset_path="s0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reader.get_tensorstore()
        assert any(
            issubclass(w.category, UserWarning) and "fallback" in str(w.message).lower()
            for w in caught
        )

    def test_fallback_returns_tensorstore(self, sample_zarr2_incompatible_path):
        """Fallback returns a ts.TensorStore with correct shape."""
        import tensorstore as ts
        import warnings
        reader = Zarr2Reader(sample_zarr2_incompatible_path, dataset_path="s0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = reader.get_tensorstore()
        assert isinstance(store, ts.TensorStore)
        assert list(store.shape) == [32, 64, 64]

    def test_fallback_correct_values(self, sample_zarr2_incompatible_path, sample_3d_array):
        """Fallback returns correct data values."""
        import warnings
        reader = Zarr2Reader(sample_zarr2_incompatible_path, dataset_path="s0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = reader.get_tensorstore()
        result = store[...].read().result()
        np.testing.assert_array_equal(result, sample_3d_array)

    def test_fallback_bigendian_byteswap(self, sample_zarr2_bigendian_path):
        """Fallback correctly byte-swaps big-endian >u2 data to native uint16."""
        import warnings
        path, expected = sample_zarr2_bigendian_path
        reader = Zarr2Reader(path, dataset_path="s0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = reader.get_tensorstore()
        result = store[...].read().result()
        assert result.dtype == np.dtype('uint16')  # native, not big-endian
        np.testing.assert_array_equal(result, expected.astype('uint16'))

    def test_native_driver_unaffected(self, sample_zarr2_path):
        """Normal zarr2 (valid gzip level) still uses TensorStore native driver."""
        reader = Zarr2Reader(sample_zarr2_path, dataset_path="s0")
        store = reader.get_tensorstore()
        assert store.spec().to_json().get('driver') == 'zarr'


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
        assert store.spec().to_json()['driver'] == 'n5'

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


class TestN5ReaderPixelResFallback:
    """Tests for N5Reader fallback for N5 v2.0.0 pixelResolution groups.

    Covers the case where the group-level attributes.json uses
    {"pixelResolution": {...}} without a standard "dimensions" array,
    causing TensorStore's native N5 driver to raise "member is missing".
    N5Reader should auto-discover s0 and open it transparently.
    """

    def test_fallback_emits_warning(self, n5_pixelres_group):
        """N5Reader emits UserWarning when using pixelResolution fallback."""
        import warnings as _warnings
        reader = N5Reader(n5_pixelres_group)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            reader.get_tensorstore()
        assert any(issubclass(x.category, UserWarning) for x in w), \
            "Expected a UserWarning when falling back for pixelResolution group"

    def test_fallback_correct_shape(self, n5_pixelres_group, sample_3d_array):
        """N5Reader fallback opens s0 with the correct array shape."""
        import tensorstore as ts
        reader = N5Reader(n5_pixelres_group)
        store = reader.get_tensorstore()
        assert isinstance(store, ts.TensorStore)
        assert tuple(store.shape) == sample_3d_array.shape

    def test_fallback_correct_values(self, n5_pixelres_group, sample_3d_array):
        """N5Reader fallback reads correct data values from s0."""
        reader = N5Reader(n5_pixelres_group)
        store = reader.get_tensorstore()
        data = store.read().result()
        np.testing.assert_array_equal(data, sample_3d_array)

    def test_standard_n5_unaffected(self, sample_n5_path, sample_3d_array):
        """Standard N5 array (with dataset_path='s0') opens without fallback warning."""
        import warnings as _warnings
        import tensorstore as ts
        reader = N5Reader(sample_n5_path, dataset_path="s0")
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            store = reader.get_tensorstore()
        fallback_warns = [x for x in w if 'pixelResolution' in str(x.message)]
        assert len(fallback_warns) == 0, \
            "Standard N5 array should not trigger pixelResolution fallback warning"
        assert isinstance(store, ts.TensorStore)
        assert tuple(store.shape) == sample_3d_array.shape


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
