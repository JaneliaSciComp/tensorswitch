"""
Tests for voxel-size provenance (``BaseReader.has_voxel_metadata``) and the
HDF5 reader's handling of files with no voxel attributes.

Regression: the HDF5 reader used to default a missing voxel size to 1.0 *um*,
which became 1000 nm. The converter only refused all-1.0 placeholders, so a
file with no voxel metadata was silently written with a 1000 nm voxel size.
"""

import os

import h5py
import numpy as np
import pytest

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core.converter import DistributedConverter
from tensorswitch_v2.readers.hdf5 import HDF5Reader


def _write_h5(path, shape, attrs=None):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("main", data=np.zeros(shape, dtype=np.uint8))
        for key, value in (attrs or {}).items():
            ds.attrs[key] = value
    return path


class TestHDF5VoxelProvenance:
    def test_no_attrs_reports_no_metadata_and_placeholder(self, temp_dir):
        path = _write_h5(os.path.join(temp_dir, "bare.h5"), (4, 8, 8))
        reader = HDF5Reader(path)
        assert reader.has_voxel_metadata() is False
        assert reader.get_voxel_sizes() == {"x": 1.0, "y": 1.0, "z": 1.0}

    def test_full_attrs_are_converted_to_nm(self, temp_dir):
        attrs = {"voxel_size_x": 0.02, "voxel_size_y": 0.02, "voxel_size_z": 0.025}
        path = _write_h5(os.path.join(temp_dir, "full.h5"), (4, 8, 8), attrs)
        reader = HDF5Reader(path)
        assert reader.has_voxel_metadata() is True
        sizes = reader.get_voxel_sizes()
        assert sizes["x"] == pytest.approx(20.0)
        assert sizes["z"] == pytest.approx(25.0)

    def test_partial_attrs_on_3d_are_not_enough(self, temp_dir):
        attrs = {"voxel_size_x": 0.02, "voxel_size_y": 0.02}
        path = _write_h5(os.path.join(temp_dir, "partial.h5"), (4, 8, 8), attrs)
        reader = HDF5Reader(path)
        assert reader.has_voxel_metadata() is False
        # the axis that was found keeps its real value; the missing one stays a placeholder
        sizes = reader.get_voxel_sizes()
        assert sizes["x"] == pytest.approx(20.0)
        assert sizes["z"] == 1.0

    def test_2d_dataset_does_not_need_z(self, temp_dir):
        attrs = {"voxel_size_x": 0.02, "voxel_size_y": 0.02}
        path = _write_h5(os.path.join(temp_dir, "flat.h5"), (8, 8), attrs)
        assert HDF5Reader(path).has_voxel_metadata() is True


class TestConverterUsesProvenance:
    def _convert(self, src, out, **kwargs):
        reader = Readers.hdf5(src, dataset_path="main")
        writer = Writers.zarr3(out)
        return DistributedConverter(reader, writer).convert(**kwargs)

    def test_refuses_file_without_voxel_metadata(self, temp_dir):
        src = _write_h5(os.path.join(temp_dir, "bare.h5"), (4, 8, 8))
        with pytest.raises(ValueError, match="No voxel size metadata"):
            self._convert(src, os.path.join(temp_dir, "out.zarr"))

    def test_override_allows_file_without_voxel_metadata(self, temp_dir):
        src = _write_h5(os.path.join(temp_dir, "bare.h5"), (4, 8, 8))
        self._convert(
            src,
            os.path.join(temp_dir, "out.zarr"),
            voxel_size_override={"x": 20.0, "y": 20.0, "z": 25.0},
            voxel_unit="nanometer",
        )
        assert os.path.exists(os.path.join(temp_dir, "out.zarr"))

    def test_file_with_attrs_converts_without_override(self, temp_dir):
        attrs = {"voxel_size_x": 0.02, "voxel_size_y": 0.02, "voxel_size_z": 0.025}
        src = _write_h5(os.path.join(temp_dir, "full.h5"), (4, 8, 8), attrs)
        self._convert(src, os.path.join(temp_dir, "out.zarr"))
        assert os.path.exists(os.path.join(temp_dir, "out.zarr"))


class TestBaseDefault:
    def test_reader_that_does_not_track_returns_none(self, temp_dir):
        import tifffile

        path = os.path.join(temp_dir, "a.tif")
        tifffile.imwrite(path, np.zeros((3, 8, 8), dtype=np.uint8))
        assert Readers.tiff(path).has_voxel_metadata() is None
