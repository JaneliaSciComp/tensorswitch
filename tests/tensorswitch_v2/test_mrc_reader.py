"""
Tests for the MRC reader (Tier 2, mrcfile-backed).

MRC headers store voxel size in angstroms. Two things are easy to get silently
wrong: the angstrom -> nm conversion, and trusting headers that were never
calibrated (1.0 A default) or that hold micrometers in the angstrom field
(light-sheet SIM datasets such as BioSR).
"""

import os
import warnings

import numpy as np
import pytest

mrcfile = pytest.importorskip("mrcfile")

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core.converter import DistributedConverter
from tensorswitch_v2.readers import MRCReader


def _write_mrc(path, arr, voxel_angstrom=None):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(arr)
        if voxel_angstrom is not None:
            m.voxel_size = voxel_angstrom
    return path


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    # distinct per-axis sizes so a wrong axis order cannot pass
    return rng.integers(0, 60000, (5, 7, 9)).astype(np.uint16)


class TestMRCData:
    def test_axes_shape_and_values(self, temp_dir, volume):
        path = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (13.48, 13.48, 13.48))
        store = MRCReader(path).get_tensorstore()
        assert list(store.domain.labels) == ["z", "y", "x"]
        assert tuple(store.shape) == (5, 7, 9)
        np.testing.assert_array_equal(store.read().result(), volume)

    def test_int8_dtype_is_preserved(self, temp_dir):
        arr = np.arange(-60, 60, dtype=np.int8).reshape(3, 5, 8)
        path = _write_mrc(os.path.join(temp_dir, "i8.mrc"), arr, (10.0, 10.0, 10.0))
        store = MRCReader(path).get_tensorstore()
        assert store.dtype.numpy_dtype == np.int8

    def test_2d_image_has_yx_axes(self, temp_dir):
        arr = np.zeros((6, 8), dtype=np.float32)
        path = _write_mrc(os.path.join(temp_dir, "img.mrc"), arr, (5.0, 5.0, 5.0))
        store = MRCReader(path).get_tensorstore()
        assert list(store.domain.labels) == ["y", "x"]

    def test_complex_data_is_rejected(self, temp_dir):
        path = os.path.join(temp_dir, "c.mrc")
        _write_mrc(path, np.zeros((3, 4, 5), dtype=np.complex64), (5.0, 5.0, 5.0))
        with pytest.raises(ValueError, match="Complex"):
            MRCReader(path).get_tensorstore()


class TestMRCVoxelSize:
    def test_angstrom_header_is_converted_to_nm(self, temp_dir, volume):
        path = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (13.48, 13.48, 20.0))
        reader = MRCReader(path)
        assert reader.has_voxel_metadata() is True
        sizes = reader.get_voxel_sizes()
        assert sizes["x"] == pytest.approx(1.348, rel=1e-4)
        assert sizes["z"] == pytest.approx(2.0, rel=1e-4)

    def test_uncalibrated_default_is_treated_as_missing(self, temp_dir, volume):
        path = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (1.0, 1.0, 1.0))
        assert MRCReader(path).has_voxel_metadata() is False

    def test_micrometers_in_angstrom_field_are_treated_as_missing(self, temp_dir, volume):
        # BioSR LLS-SIM: 0.0926 um x 0.0926 um x 0.3906 um stored as "angstroms"
        path = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (0.0926, 0.0926, 0.3906))
        reader = MRCReader(path)
        assert reader.has_voxel_metadata() is False
        with pytest.warns(UserWarning, match="implausibly small"):
            assert reader.get_voxel_sizes() == {"x": 1.0, "y": 1.0, "z": 1.0}

    def test_2d_image_needs_only_x_and_y(self, temp_dir):
        arr = np.zeros((6, 8), dtype=np.uint8)
        path = _write_mrc(os.path.join(temp_dir, "img.mrc"), arr, (5.0, 5.0, 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = MRCReader(path)
            assert reader.has_voxel_metadata() is True
            assert reader.get_voxel_sizes()["x"] == pytest.approx(0.5)


class TestMRCAutoDetect:
    @pytest.mark.parametrize("name", ["a.mrc", "a.rec", "a.mrcs", "a.ali", "a.st", "A.MRC"])
    def test_extensions_route_to_mrc_reader(self, temp_dir, volume, name):
        path = _write_mrc(os.path.join(temp_dir, name), volume, (10.0, 10.0, 10.0))
        assert isinstance(Readers.auto_detect(path), MRCReader)


class TestMRCConversion:
    def _convert(self, src, out, **kwargs):
        return DistributedConverter(Readers.mrc(src), Writers.zarr3(out)).convert(**kwargs)

    def test_converter_refuses_uncalibrated_file(self, temp_dir, volume):
        src = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="No voxel size metadata"):
            self._convert(src, os.path.join(temp_dir, "out.zarr"))

    def test_converter_accepts_override_for_micrometer_header(self, temp_dir, volume):
        src = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (0.0926, 0.0926, 0.3906))
        out = os.path.join(temp_dir, "out.zarr")
        self._convert(
            src, out,
            voxel_size_override={"x": 92.6, "y": 92.6, "z": 390.6},
            voxel_unit="nanometer",
        )
        assert os.path.exists(out)

    def test_calibrated_file_converts_without_override(self, temp_dir, volume):
        src = _write_mrc(os.path.join(temp_dir, "v.mrc"), volume, (13.48, 13.48, 13.48))
        out = os.path.join(temp_dir, "out.zarr")
        self._convert(src, out)
        assert os.path.exists(out)
