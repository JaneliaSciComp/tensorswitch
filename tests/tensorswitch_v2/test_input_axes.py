"""--input_axes names the source axes by position (record `axes`), for sources that do not state them."""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import tensorstore as ts
import tifffile

from tensorswitch_v2.__main__ import main as cli_main

RNG = np.random.default_rng(5)


def convert(src, out, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", src, "-o", out, "--voxel_size", "8,8,40", "--voxel_unit", "nanometer", "--quiet", *extra])


def group(out, name="raw"):
    ms = json.load(open(os.path.join(out, name, "zarr.json")))["attributes"]["ome"]["multiscales"][0]
    return [a["name"] for a in ms["axes"]], ms["datasets"][0]["coordinateTransformations"][0]["scale"], ms["axes"]


def read(out, name="raw"):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(out, name, "s0")}}).result().read().result()


@pytest.fixture
def pages_rgb(temp_dir):
    """Plain multi-page RGB TIFF (pages read as `i`, samples as `s`), like deepfoci's raw."""
    arr = RNG.integers(1, 250, (8, 32, 48, 3), dtype=np.uint8)
    path = os.path.join(temp_dir, "pages.tif")
    tifffile.imwrite(path, arr, photometric="rgb", shaped=False)
    return path, arr


@pytest.fixture
def slices_as_samples(temp_dir):
    """One page with 6 samples per pixel that are really z slices, like deepfoci's masks (yxz)."""
    arr = RNG.integers(0, 5, (32, 48, 6), dtype=np.uint16)
    path = os.path.join(temp_dir, "mask.tif")
    tifffile.imwrite(path, arr, photometric="minisblack", planarconfig="contig", shaped=False)
    return path, arr


def test_without_input_axes_the_reader_names_stay(pages_rgb, temp_dir):
    out = os.path.join(temp_dir, "n.zarr")
    convert(pages_rgb[0], out)
    assert group(out)[0] == ["i", "s", "y", "x"]


def test_pages_become_z_and_samples_become_channels(pages_rgb, temp_dir):
    path, arr = pages_rgb
    out = os.path.join(temp_dir, "a.zarr")
    convert(path, out, "--input_axes", "zyxc")
    names, scale, axes = group(out)
    assert names == ["c", "z", "y", "x"] and scale == [1.0, 40.0, 8.0, 8.0]
    assert axes[0] == {"name": "c", "type": "channel"}
    assert np.array_equal(read(out), np.moveaxis(arr, -1, 0))


def test_samples_can_stay_samples(pages_rgb, temp_dir):
    out = os.path.join(temp_dir, "s.zarr")
    convert(pages_rgb[0], out, "--input_axes", "zyxs")
    names, scale, axes = group(out)
    assert names == ["s", "z", "y", "x"] and scale == [1.0, 40.0, 8.0, 8.0] and axes[0]["type"] == "custom"


def test_slices_stored_as_samples_become_z_in_source_order(slices_as_samples, temp_dir):
    path, arr = slices_as_samples
    out = os.path.join(temp_dir, "z.zarr")
    convert(path, out, "--input_axes", "yxz")
    names, scale, axes = group(out)
    assert names == ["y", "x", "z"] and scale == [8.0, 8.0, 40.0]
    assert all(a["type"] == "space" for a in axes)
    assert np.array_equal(read(out), arr)


@pytest.mark.parametrize("value,message", [("zyx", "names 3 axes but the source has 4"),
                                           ("zyxq", "may only use"), ("zzxc", "duplicate axis")])
def test_bad_values_fail_clearly(pages_rgb, temp_dir, value, message):
    with pytest.raises(ValueError, match=message):
        convert(pages_rgb[0], os.path.join(temp_dir, "bad.zarr"), "--input_axes", value)


def test_conflict_with_relabel_axis_is_refused(pages_rgb, temp_dir):
    with pytest.raises(ValueError, match="disagree"):
        convert(pages_rgb[0], os.path.join(temp_dir, "c.zarr"), "--input_axes", "zyxc", "--relabel_axis", "i=t")


def test_agreeing_relabel_axis_is_fine(pages_rgb, temp_dir):
    out = os.path.join(temp_dir, "ok.zarr")
    convert(pages_rgb[0], out, "--input_axes", "zyxc", "--relabel_axis", "i=z")
    assert group(out)[0] == ["c", "z", "y", "x"]


class TestSubmitPathsSeeTheNames:
    def test_resource_estimate_and_voxel_check_use_the_named_axes(self, pages_rgb):
        from argparse import Namespace
        from tensorswitch_v2.__main__ import _renamed_axes
        args = Namespace(input_axes="zyxc", relabel_axis=None)
        assert _renamed_axes(args, ["i", "y", "x", "s"]) == ["z", "y", "x", "c"]
        assert _renamed_axes(Namespace(input_axes=None, relabel_axis=["i=z"]), ["i", "y", "x", "s"]) == ["z", "y", "x", "s"]

    def _convert_capturing_text(self, src, out, *extra):
        import warnings
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            warnings.simplefilter("always")
            cli_main(["-i", src, "-o", out, "--voxel_size", "8,8,40", "--voxel_unit", "nanometer", *extra])
        return buf.getvalue() + " ".join(str(w.message) for w in caught)

    def test_the_missing_z_warning_appears_without_input_axes(self, slices_as_samples, temp_dir):
        text = self._convert_capturing_text(slices_as_samples[0], os.path.join(temp_dir, "w0.zarr"))
        assert "spatial axes" in text

    def test_no_warning_about_a_missing_z_when_input_axes_names_it(self, slices_as_samples, temp_dir):
        text = self._convert_capturing_text(slices_as_samples[0], os.path.join(temp_dir, "w1.zarr"), "--input_axes", "yxz")
        assert "spatial axes" not in text
