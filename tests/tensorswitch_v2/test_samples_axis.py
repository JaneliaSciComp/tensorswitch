"""The TIFF samples-per-pixel axis `s` is its own non-spatial dimension (like BioIO's S): never a channel,
never downsampled, typed `custom`, and placed before the spatial axes."""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import tensorstore as ts
import tifffile

from tensorswitch_v2.__main__ import main as cli_main
from tensorswitch_v2.utils.metadata_utils import NON_SPATIAL_AXES, get_axis_type
from tensorswitch_v2.utils.pyramid_utils import calculate_anisotropic_downsample_factors


def convert(src, out, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", src, "-o", out, "--voxel_size", "8,8,40", "--voxel_unit", "nanometer", "--quiet", *extra])


def axes_of(group_json):
    return json.load(open(group_json))["attributes"]["ome"]["multiscales"][0]["axes"]


def test_s_is_non_spatial_and_custom():
    assert "s" in NON_SPATIAL_AXES and get_axis_type("s") == "custom" and get_axis_type("S") == "custom"
    assert get_axis_type("c") == "channel" and get_axis_type("z") == "space"


def test_pyramid_factors_leave_s_alone_and_downsample_the_spatial_axes():
    with_s = calculate_anisotropic_downsample_factors([1.0, 40.0, 8.0, 8.0], ["s", "z", "y", "x"])
    with_c = calculate_anisotropic_downsample_factors([1.0, 40.0, 8.0, 8.0], ["c", "z", "y", "x"])
    assert with_s == with_c and with_s[0] == 1 and with_s[2:] == [2, 2]


@pytest.fixture
def rgb_tiff(temp_dir):
    rng = np.random.default_rng(0)
    path = os.path.join(temp_dir, "rgb.tif")
    tifffile.imwrite(path, rng.integers(1, 250, (64, 512, 512, 3), dtype=np.uint8), photometric="rgb",
                     metadata={"axes": "ZYXS"})
    return path


def test_samples_axis_written_first_as_custom_with_unit_scale(rgb_tiff, temp_dir):
    out = os.path.join(temp_dir, "o.zarr")
    convert(rgb_tiff, out)
    axes = axes_of(os.path.join(out, "raw", "zarr.json"))
    assert [a["name"] for a in axes] == ["s", "z", "y", "x"]
    assert axes[0] == {"name": "s", "type": "custom"} and all(a["type"] == "space" for a in axes[1:])
    scale = json.load(open(os.path.join(out, "raw", "zarr.json")))["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [1.0, 40.0, 8.0, 8.0]


def test_pyramid_keeps_all_samples_at_every_level(rgb_tiff, temp_dir):
    out = os.path.join(temp_dir, "p.zarr")
    convert(rgb_tiff, out, "--auto_multiscale")
    levels = sorted(d for d in os.listdir(os.path.join(out, "raw")) if d.startswith("s") and d != "s")
    assert len(levels) >= 2
    for level in levels:
        store = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(out, "raw", level)}}).result()
        assert store.shape[0] == 3, f"{level}: {store.shape}"
    s1 = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(out, "raw", "s1")}}).result()
    s0 = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(out, "raw", "s0")}}).result()
    assert s1.shape[2] < s0.shape[2] and s1.shape[3] < s0.shape[3]
