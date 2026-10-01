"""
Per-reader checks of the voxel-size contract for TIFF, ND2, IMS and Zarr.

These readers used to fill any unstated axis with 1.0 silently, so a file with a
calibrated XY but no Z spacing came out with Z = 1.0 nm and was accepted. They
now report only what the file states, and the base class flags the rest.
"""

import os
from unittest import mock

import numpy as np
import pytest
import tifffile

from tensorswitch_v2.readers import IMSReader, ND2Reader, TiffReader


def _tiff(path, **kwargs):
    tifffile.imwrite(path, np.zeros((4, 8, 8), dtype=np.uint8), **kwargs)
    return path


class TestTiffVoxelSizes:
    def test_ome_tiff_with_all_axes(self, temp_dir):
        path = _tiff(
            os.path.join(temp_dir, "full.ome.tif"),
            metadata={"axes": "ZYX", "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.4,
                      "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm", "PhysicalSizeZUnit": "µm"},
        )
        reader = TiffReader(path)
        assert reader.has_voxel_metadata() is True
        sizes = reader.get_voxel_sizes()
        assert sizes["x"] == pytest.approx(100.0)
        assert sizes["z"] == pytest.approx(400.0)

    def test_ome_tiff_missing_z_is_not_silently_one_nm(self, temp_dir):
        path = _tiff(
            os.path.join(temp_dir, "noz.ome.tif"),
            metadata={"axes": "ZYX", "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1,
                      "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"},
        )
        reader = TiffReader(path)
        with pytest.warns(UserWarning, match="axis z"):
            sizes = reader.get_voxel_sizes()
        assert sizes.missing == ["z"]
        assert sizes["x"] == pytest.approx(100.0)
        assert reader.has_voxel_metadata() is False

    def test_imagej_tiff_with_spacing_and_resolution(self, temp_dir):
        path = _tiff(
            os.path.join(temp_dir, "ij.tif"),
            imagej=True, resolution=(10.0, 10.0),
            metadata={"axes": "ZYX", "spacing": 0.5, "unit": "micron"},
        )
        reader = TiffReader(path)
        assert reader.has_voxel_metadata() is True
        sizes = reader.get_voxel_sizes()
        assert sizes["x"] == pytest.approx(100.0)
        assert sizes["z"] == pytest.approx(500.0)

    def test_plain_tiff_has_no_voxel_metadata(self, temp_dir):
        path = _tiff(os.path.join(temp_dir, "plain.tif"))
        assert TiffReader(path).has_voxel_metadata() is False


class TestND2VoxelSizes:
    def _reader(self, stated):
        with mock.patch("tensorswitch_v2.readers.nd2.extract_nd2_ome_metadata", return_value=("<xml/>", stated)):
            reader = ND2Reader("unused.nd2")
            reader.get_metadata()
        return reader

    def test_partial_axes_are_flagged(self):
        reader = self._reader({"x": 160.0, "y": 160.0, "z": None})
        with pytest.warns(UserWarning, match="axis z"):
            sizes = reader.get_voxel_sizes()
        assert sizes["x"] == 160.0 and sizes["z"] == 1.0
        assert reader.has_voxel_metadata() is False

    def test_complete_axes(self):
        reader = self._reader({"x": 160.0, "y": 160.0, "z": 400.0})
        assert reader.has_voxel_metadata() is True

    def test_nothing_stated(self):
        reader = self._reader(None)
        with pytest.warns(UserWarning):
            assert reader.has_voxel_metadata() is False


class TestIMSVoxelSizes:
    def _reader(self, stated):
        with mock.patch("tensorswitch_v2.readers.ims.extract_ims_metadata", return_value=({}, stated)):
            reader = IMSReader("unused.ims")
            reader.get_metadata()
        return reader

    def test_list_form(self):
        reader = self._reader([100.0, 100.0, 300.0])
        assert reader.has_voxel_metadata() is True
        assert reader.get_voxel_sizes()["z"] == 300.0

    def test_zero_entry_counts_as_unstated(self):
        reader = self._reader([100.0, 100.0, 0])
        with pytest.warns(UserWarning, match="axis z"):
            assert reader.has_voxel_metadata() is False

    def test_nothing_stated(self):
        reader = self._reader(None)
        with pytest.warns(UserWarning):
            assert reader.has_voxel_metadata() is False


class TestZarrVoxelSizes:
    """Round trip through TensorSwitch's own writers (Zarr2 has no axis labels)."""

    @pytest.mark.parametrize("fmt", ["zarr3", "zarr2"])
    def test_3d_round_trip(self, temp_dir, fmt):
        from tensorswitch_v2.api import Readers
        from tensorswitch_v2.core.converter import DistributedConverter
        from tensorswitch_v2.api import Writers

        src = os.path.join(temp_dir, "a.tif")
        tifffile.imwrite(src, np.zeros((20, 24, 24), dtype=np.uint8), metadata={"axes": "ZYX"})
        out = os.path.join(temp_dir, "o.zarr")
        writer = Writers.zarr3(out) if fmt == "zarr3" else Writers.zarr2(out)
        DistributedConverter(TiffReader(src), writer).convert(
            voxel_size_override={"x": 8.0, "y": 8.0, "z": 40.0}, voxel_unit="nanometer"
        )
        reader = Readers.auto_detect(os.path.join(out, "raw", "s0"))
        assert reader.has_voxel_metadata() is True
        assert dict(reader.get_voxel_sizes()) == {"x": 8.0, "y": 8.0, "z": 40.0}

    @pytest.mark.parametrize("fmt", ["zarr3", "zarr2"])
    def test_2d_store_does_not_need_z(self, temp_dir, fmt):
        from tensorswitch_v2.api import Readers, Writers
        from tensorswitch_v2.core.converter import DistributedConverter

        src = os.path.join(temp_dir, "a.tif")
        tifffile.imwrite(src, np.zeros((24, 24), dtype=np.uint8), metadata={"axes": "YX"})
        out = os.path.join(temp_dir, "o.zarr")
        writer = Writers.zarr3(out) if fmt == "zarr3" else Writers.zarr2(out)
        DistributedConverter(TiffReader(src), writer).convert(
            voxel_size_override={"x": 8.0, "y": 8.0, "z": 8.0}, voxel_unit="nanometer"
        )
        reader = Readers.auto_detect(os.path.join(out, "raw", "s0"))
        assert reader.has_voxel_metadata() is True

