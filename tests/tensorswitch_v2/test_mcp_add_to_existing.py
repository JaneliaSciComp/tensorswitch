"""
MCP convert(add_to_existing=True) must behave like the CLI's --add-to-existing:
the new label is written to labels/<name>.tmp, renamed in place, and listed in
the labels metadata, without touching the existing raw image or other labels.

It used to write into a separate labels.tmp container and call the finalizer
without the label name, so every call failed.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("mcp")
tifffile = pytest.importorskip("tifffile")

from tensorswitch_v2 import mcp_server as m
from tensorswitch_v2.__main__ import main as cli_main


def _tif(folder, name, value, dtype=np.uint8, shape=(8, 16, 16)):
    path = os.path.join(folder, name)
    arr = np.full(shape, value, dtype=dtype)
    tifffile.imwrite(path, arr, metadata={"axes": "ZYX"})
    return path


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(root) for f in fs
    )


def _labels_listed(container, fmt):
    meta_path = os.path.join(container, "labels", "zarr.json" if fmt == "zarr3" else ".zattrs")
    meta = json.load(open(meta_path))
    return sorted(meta.get("attributes", {}).get("ome", meta).get("labels", []))


def _mcp(src, out, **kw):
    result = json.loads(m.convert(src, out, voxel_size="1,2,3", **kw))
    assert result.get("status") == "success", result
    return result


def _cli(src, out, fmt, *extra):
    cli_main(["-i", src, "-o", out, "--output_format", fmt, "--voxel_size", "1,2,3", *extra])


@pytest.mark.parametrize("fmt", ["zarr3", "zarr2"])
class TestAddLabel:
    def test_adds_label_and_leaves_raw_and_other_labels_alone(self, temp_dir, fmt):
        raw, lab1, lab2 = _tif(temp_dir, "r.tif", 5), _tif(temp_dir, "l1.tif", 1), _tif(temp_dir, "l2.tif", 2)
        out = os.path.join(temp_dir, "c.zarr")
        _mcp(raw, out, output_format=fmt)
        before = _files(os.path.join(out, "raw"))
        _mcp(lab1, out, output_format=fmt, is_label=True, data_type="labels", label_key="seg1", add_to_existing=True)
        _mcp(lab2, out, output_format=fmt, is_label=True, data_type="labels", label_key="seg2", add_to_existing=True)

        assert _labels_listed(out, fmt) == ["seg1", "seg2"]
        assert sorted(os.listdir(os.path.join(out, "labels"))) == sorted(
            ["seg1", "seg2", "zarr.json" if fmt == "zarr3" else ".zattrs"] + ([".zgroup"] if fmt == "zarr2" else []))
        assert not [f for f in _files(out) if ".tmp" in f], "no .tmp leftovers"
        assert _files(os.path.join(out, "raw")) == before
        for name, value in (("seg1", 1), ("seg2", 2)):
            level = os.path.join(out, "labels", name, "s0")
            assert (np.asarray(__import__("tensorswitch_v2.api", fromlist=["Readers"]).Readers.auto_detect(level)
                               .get_tensorstore().read().result()) == value).all()

    def test_same_label_name_replaces_it(self, temp_dir, fmt):
        raw, a, b = _tif(temp_dir, "r.tif", 5), _tif(temp_dir, "a.tif", 1), _tif(temp_dir, "b.tif", 9)
        out = os.path.join(temp_dir, "c.zarr")
        _mcp(raw, out, output_format=fmt)
        for src in (a, b):
            _mcp(src, out, output_format=fmt, is_label=True, data_type="labels", label_key="seg", add_to_existing=True)
        assert _labels_listed(out, fmt) == ["seg"]
        from tensorswitch_v2.api import Readers
        assert (Readers.auto_detect(os.path.join(out, "labels", "seg", "s0")).get_tensorstore().read().result() == 9).all()

    def test_matches_the_cli_file_for_file(self, temp_dir, fmt):
        raw, lab = _tif(temp_dir, "r.tif", 5), _tif(temp_dir, "l.tif", 3)
        via_mcp, via_cli = os.path.join(temp_dir, "mcp.zarr"), os.path.join(temp_dir, "cli.zarr")
        _mcp(raw, via_mcp, output_format=fmt)
        _cli(raw, via_cli, fmt)
        _mcp(lab, via_mcp, output_format=fmt, is_label=True, data_type="labels", label_key="seg", add_to_existing=True)
        _cli(lab, via_cli, fmt, "--add-to-existing", "--data-type", "labels", "--label-key", "seg")
        assert _files(via_mcp) == _files(via_cli)
        assert _labels_listed(via_mcp, fmt) == _labels_listed(via_cli, fmt) == ["seg"]

    def test_with_auto_multiscale_builds_levels_on_the_new_label(self, temp_dir, fmt):
        big = (64, 512, 512)  # small volumes need no downsampling level
        raw, lab = _tif(temp_dir, "r.tif", 5, shape=big), _tif(temp_dir, "l.tif", 3, shape=big)
        out = os.path.join(temp_dir, "c.zarr")
        _mcp(raw, out, output_format=fmt)
        result = _mcp(lab, out, output_format=fmt, is_label=True, data_type="labels", label_key="seg",
                      add_to_existing=True, auto_multiscale=True)
        assert result.get("pyramid_levels", 0) >= 1
        levels = sorted(d for d in os.listdir(os.path.join(out, "labels", "seg")) if d.startswith("s"))
        assert len(levels) >= 2
        assert not [f for f in _files(out) if ".tmp" in f]


def test_failure_does_not_leave_a_half_written_label_listed(temp_dir):
    raw = _tif(temp_dir, "r.tif", 5)
    out = os.path.join(temp_dir, "c.zarr")
    _mcp(raw, out)
    result = m.convert(os.path.join(temp_dir, "missing.tif"), out, voxel_size="1,2,3", is_label=True,
                       data_type="labels", label_key="seg", add_to_existing=True)
    assert "seg" not in _labels_listed(out, "zarr3") if os.path.exists(os.path.join(out, "labels", "zarr.json")) else True
    assert not os.path.exists(os.path.join(out, "labels", "seg"))
    assert "rror" in result


def test_requires_an_existing_container(temp_dir):
    result = json.loads(m.convert(_tif(temp_dir, "l.tif", 1), os.path.join(temp_dir, "nope.zarr"),
                                  voxel_size="1,2,3", is_label=True, data_type="labels", add_to_existing=True))
    assert "does not exist" in result["error"]
