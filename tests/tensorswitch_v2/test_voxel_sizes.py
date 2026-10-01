"""
Tests for the uniform voxel-size contract: BaseReader._read_voxel_sizes(),
the VoxelSizes result object, and how the converter uses it.

Every reader reports the sizes its source states (None for unknown axes); the
base class fills the 1.0 placeholder, tracks which axes are real, and the
converter refuses when a needed axis is a placeholder.
"""

import os
import warnings

import dask.array as da
import numpy as np
import pytest

from tensorswitch_v2.api import Writers
from tensorswitch_v2.core.converter import DistributedConverter
from tensorswitch_v2.readers import DaskReader, VoxelSizes


class FakeReader(DaskReader):
    """Smallest possible reader: a zero array and whatever sizes the test supplies."""

    def __init__(self, shape, stated):
        super().__init__("fake")
        self._shape = shape
        self._stated = stated
        self.reads = 0

    def _load(self):
        if self._dask_array is None:
            self._dask_array = da.zeros(self._shape, dtype="uint8", chunks=self._shape)

    def get_metadata(self):
        return {"shape": self._shape, "dtype": "uint8"}

    def _read_voxel_sizes(self):
        self.reads += 1
        return self._stated


class TestVoxelSizesObject:
    def test_unknown_axes_hold_the_placeholder(self):
        vs = VoxelSizes({"x": 116.0, "y": 116.0, "z": None})
        assert dict(vs) == {"x": 116.0, "y": 116.0, "z": 1.0}
        assert vs.known == {"x", "y"}
        assert vs.missing == ["z"]
        assert vs.is_complete is False

    def test_complete_when_all_required_axes_known(self):
        vs = VoxelSizes({"x": 1.0, "y": 2.0, "z": 3.0})
        assert vs.is_complete and vs.missing == []

    def test_2d_requires_only_x_and_y(self):
        vs = VoxelSizes({"x": 5.0, "y": 5.0}, required=("x", "y"))
        assert vs.is_complete

    def test_none_or_empty_input_is_all_placeholder(self):
        assert VoxelSizes(None).known == frozenset()
        assert dict(VoxelSizes({})) == {"x": 1.0, "y": 1.0, "z": 1.0}

    def test_non_positive_values_count_as_unknown(self):
        assert VoxelSizes({"x": 0.0, "y": -3.0, "z": 4.0}).known == {"z"}

    def test_extra_keys_are_kept(self):
        assert VoxelSizes({"x": 1.0, "y": 1.0, "z": 1.0, "t": 2.5})["t"] == 2.5

    def test_copy_keeps_provenance_and_is_independent(self):
        vs = VoxelSizes({"x": 10.0, "y": 10.0, "z": None})
        clone = vs.copy()
        clone["x"] = 99.0
        assert vs["x"] == 10.0
        assert clone.known == vs.known and clone.missing == ["z"]


class TestBaseContract:
    def test_full_metadata_is_complete(self):
        reader = FakeReader((4, 5, 6), {"x": 10.0, "y": 10.0, "z": 20.0})
        assert reader.has_voxel_metadata() is True
        assert reader.get_voxel_sizes()["z"] == 20.0

    def test_partial_metadata_is_not_complete(self):
        reader = FakeReader((4, 5, 6), {"x": 10.0, "y": 10.0})
        with pytest.warns(UserWarning, match="axis z"):
            sizes = reader.get_voxel_sizes()
        assert sizes["z"] == 1.0 and sizes.missing == ["z"]
        assert reader.has_voxel_metadata() is False

    def test_no_metadata_at_all(self):
        reader = FakeReader((4, 5, 6), None)
        with pytest.warns(UserWarning):
            assert reader.has_voxel_metadata() is False

    def test_2d_array_does_not_need_z(self):
        reader = FakeReader((5, 6), {"x": 10.0, "y": 10.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert reader.has_voxel_metadata() is True

    def test_subclass_is_read_once(self):
        reader = FakeReader((4, 5, 6), {"x": 1.5, "y": 1.5, "z": 1.5})
        reader.get_voxel_sizes()
        reader.get_voxel_sizes()
        reader.has_voxel_metadata()
        assert reader.reads == 1

    def test_returned_sizes_cannot_corrupt_the_cache(self):
        reader = FakeReader((4, 5, 6), {"x": 10.0, "y": 10.0, "z": 10.0})
        reader.get_voxel_sizes()["x"] = 999.0
        assert reader.get_voxel_sizes()["x"] == 10.0

    def test_reader_with_no_hook_fails_loudly(self):
        class Bare(FakeReader):
            _read_voxel_sizes = DaskReader._read_voxel_sizes

        with pytest.raises(NotImplementedError, match="_read_voxel_sizes"):
            Bare((2, 2, 2), None).get_voxel_sizes()


class TestConverterUsesContract:
    def _convert(self, reader, temp_dir, **kwargs):
        out = os.path.join(temp_dir, "out.zarr")
        DistributedConverter(reader, Writers.zarr3(out)).convert(**kwargs)
        return out

    def test_refuses_partial_metadata(self, temp_dir):
        reader = FakeReader((4, 8, 8), {"x": 10.0, "y": 10.0})
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="No voxel size metadata"):
            self._convert(reader, temp_dir)

    def test_accepts_complete_metadata(self, temp_dir):
        reader = FakeReader((4, 8, 8), {"x": 10.0, "y": 10.0, "z": 20.0})
        assert os.path.exists(self._convert(reader, temp_dir))

    def test_override_rescues_partial_metadata(self, temp_dir):
        reader = FakeReader((4, 8, 8), {"x": 10.0, "y": 10.0})
        out = self._convert(
            reader, temp_dir,
            voxel_size_override={"x": 10.0, "y": 10.0, "z": 20.0}, voxel_unit="nanometer",
        )
        assert os.path.exists(out)
