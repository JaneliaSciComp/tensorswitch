"""list_formats must describe what auto-detection really does."""

import json

import pytest

pytest.importorskip("mcp")

from tensorswitch_v2 import mcp_server as m
from tensorswitch_v2.api import Readers

FORMATS = json.loads(m.list_formats())
FILE_FORMATS = [f for f in FORMATS["input_formats"]["tier_2_dedicated_readers"]]


@pytest.mark.parametrize("fmt", FILE_FORMATS, ids=lambda f: f["name"])
def test_every_extension_routes_to_the_stated_reader(fmt):
    for ext in fmt["extensions"]:
        assert type(Readers.auto_detect(f"/nonexistent/file{ext}")).__name__ == fmt["reader"]


def test_new_formats_are_listed():
    names = {f["name"] for f in FILE_FORMATS}
    assert {"NIfTI", "MRC / CCP4", "PNG", "HDF5", "TIFF"} <= names


def test_every_dedicated_reader_class_is_listed():
    from tensorswitch_v2 import readers

    listed = {f["reader"] for f in FILE_FORMATS}
    dedicated = {"TiffReader", "ND2Reader", "IMSReader", "HDF5Reader", "CZIReader", "NIfTIReader", "MRCReader", "PngReader"}
    assert dedicated <= listed and all(hasattr(readers, name) for name in dedicated)


def test_bioio_plugins_are_the_installed_ones():
    from importlib.metadata import entry_points

    assert FORMATS["input_formats"]["tier_3_bioio"]["installed_plugins"] == \
        sorted({ep.name for ep in entry_points(group="bioio.readers")})


def test_presets_match_the_cli():
    from tensorswitch_v2.__main__ import _apply_preset, parse_args

    for preset in FORMATS["presets"]:
        args = parse_args(["-i", "a.tif", "-o", "o", "--quiet", "--preset", preset])
        _apply_preset(args)
        assert args.preset == preset


def test_output_formats_are_the_ones_the_cli_accepts():
    from tensorswitch_v2.__main__ import parse_args

    for name in ("zarr3", "zarr2", "n5"):
        assert parse_args(["-i", "a", "-o", "b", "--output_format", name]).output_format == name
