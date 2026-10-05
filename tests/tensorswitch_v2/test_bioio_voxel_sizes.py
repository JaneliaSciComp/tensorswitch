"""Voxel-size contract for the BioIO-based readers, including the Bio-Formats reader.

BioFormatsReader inherits BIOIOReader._read_voxel_sizes, so a fake BioImage that reports
physical pixel sizes in micrometers exercises the same code Bio-Formats uses. A real
Bio-Formats run is added when bioio-bioformats and Java are installed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tensorswitch_v2.readers.bioformats import BioFormatsReader
from tensorswitch_v2.readers.bioio_adapter import BIOIOReader


def _reader(cls, sizes=None, raises=False):
    reader = cls("/nonexistent/image.ome.tif")
    if raises:
        class Broken:
            @property
            def physical_pixel_sizes(self):
                raise RuntimeError("no metadata")
        reader._bioimage = Broken()
    else:
        reader._bioimage = SimpleNamespace(physical_pixel_sizes=SimpleNamespace(**sizes))
    return reader


@pytest.mark.parametrize("cls", [BIOIOReader, BioFormatsReader])
class TestBioIOVoxelContract:
    @pytest.fixture(autouse=True)
    def _no_plugin_needed(self, monkeypatch):
        monkeypatch.setattr(BioFormatsReader, "_get_bioformats_reader", lambda self: object())

    def test_micrometers_become_nanometers(self, cls):
        reader = _reader(cls, dict(X=0.008, Y=0.008, Z=0.04))
        assert dict(reader.get_voxel_sizes()) == pytest.approx({"x": 8.0, "y": 8.0, "z": 40.0})
        assert reader.has_voxel_metadata() is True

    def test_missing_z_is_reported_as_missing(self, cls):
        reader = _reader(cls, dict(X=0.008, Y=0.008, Z=None))
        sizes = reader.get_voxel_sizes()
        assert sizes.missing == ["z"] and reader.has_voxel_metadata() is False

    def test_nothing_stated(self, cls):
        reader = _reader(cls, dict(X=None, Y=None, Z=None))
        assert reader.has_voxel_metadata() is False

    def test_one_nm_on_every_axis_is_not_treated_as_a_stated_size(self, cls):
        # 0.001 micrometer is 1.0 nm on every axis, the placeholder the readers fill in
        reader = _reader(cls, dict(X=0.001, Y=0.001, Z=0.001))
        assert reader.has_voxel_metadata() is False

    def test_plugin_failure_means_no_voxel_size(self, cls):
        assert _reader(cls, raises=True).has_voxel_metadata() is False


@pytest.mark.skipif(not BioFormatsReader.is_available(), reason="bioio-bioformats and Java not installed")
class TestRealBioFormats:
    def _write(self, path, **scales):
        tifffile = pytest.importorskip("tifffile")
        volume = (np.arange(4 * 16 * 16).reshape(4, 16, 16) % 250).astype(np.uint8)
        tifffile.imwrite(str(path), volume, ome=True, metadata={"axes": "ZYX", **scales})
        return str(path)

    def test_calibrated_ome_tiff(self, tmp_path):
        path = self._write(tmp_path / "cal.ome.tif", PhysicalSizeX=0.008, PhysicalSizeY=0.008, PhysicalSizeZ=0.04,
                           PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm", PhysicalSizeZUnit="µm")
        reader = BioFormatsReader(path)
        assert dict(reader.get_voxel_sizes()) == pytest.approx({"x": 8.0, "y": 8.0, "z": 40.0})

    def test_uncalibrated_ome_tiff(self, tmp_path):
        assert BioFormatsReader(self._write(tmp_path / "bare.ome.tif")).has_voxel_metadata() is False
