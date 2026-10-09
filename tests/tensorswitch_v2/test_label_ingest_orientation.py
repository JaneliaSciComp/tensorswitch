"""
--output-offset (sparse label ingest) must honor --axes_order / --input_axes, and must refuse
source-side flags it cannot apply instead of silently ignoring them.

The branch writes the whole source at the offset and bypasses the standard converter, so these
flags used to be dropped silently: a Paintera N5 export stored x,y,z was written untransposed
into a z,y,x container, with the right ids but the wrong layout.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from tensorswitch_v2.api import Readers
from tensorswitch_v2.utils.label_ingest import check_offset_flags_supported, orient_label_source

tifffile = pytest.importorskip("tifffile")
from tensorswitch_v2.__main__ import main as cli_main


def _labeled(arr, names):
    t = ts.array(arr)
    return t[ts.d[:].label[tuple(names)]]


class TestOrientLabelSource:
    def test_transposes_xyz_to_zyx(self):
        a = np.arange(24, dtype=np.uint32).reshape(2, 3, 4)  # x=2, y=3, z=4
        out = orient_label_source(_labeled(a, "xyz"), axes_order=["z", "y", "x"])
        assert out.shape == (4, 3, 2)
        assert np.array_equal(out.read().result(), a.transpose(2, 1, 0))

    def test_already_in_order_is_unchanged(self):
        a = np.zeros((4, 3, 2), np.uint8)
        src = _labeled(a, "zyx")
        assert orient_label_source(src, axes_order=["z", "y", "x"]).shape == (4, 3, 2)

    def test_no_axes_order_is_a_no_op(self):
        src = _labeled(np.zeros((2, 3, 4), np.uint8), "xyz")
        assert orient_label_source(src).shape == (2, 3, 4)

    def test_non_spatial_axis_stays_in_place(self):
        a = np.arange(48, dtype=np.uint8).reshape(2, 2, 3, 4)  # c, x, y, z
        out = orient_label_source(_labeled(a, "cxyz"), axes_order=["z", "y", "x"])
        assert out.shape == (2, 4, 3, 2)
        assert np.array_equal(out.read().result(), a.transpose(0, 3, 2, 1))

    def test_input_axes_names_an_unlabeled_source(self):
        a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out = orient_label_source(ts.array(a), axes_order=["z", "y", "x"], input_axes="xyz")
        assert np.array_equal(out.read().result(), a.transpose(2, 1, 0))

    def test_input_axes_renames_labeled_axes(self):
        a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out = orient_label_source(_labeled(a, "zyx"), axes_order=["z", "y", "x"], input_axes="xyz")
        assert np.array_equal(out.read().result(), a.transpose(2, 1, 0))

    def test_unlabeled_source_without_input_axes_is_an_error(self):
        with pytest.raises(ValueError, match="named source axes"):
            orient_label_source(ts.array(np.zeros((2, 3, 4), np.uint8)), axes_order=["z", "y", "x"])

    def test_axes_order_that_does_not_match_is_an_error(self):
        with pytest.raises(ValueError, match="does not match"):
            orient_label_source(_labeled(np.zeros((2, 3), np.uint8), "xy"), axes_order=["z", "y", "x"])


class TestUnsupportedFlags:
    @pytest.mark.parametrize("attr,flag", [("bbox", "--bbox"), ("bbox_axes", "--bbox_axes"),
                                           ("squeeze_singleton_axes", "--squeeze_singleton_axes")])
    def test_refused(self, attr, flag):
        with pytest.raises(ValueError, match=flag):
            check_offset_flags_supported(SimpleNamespace(**{attr: "1" if attr != "squeeze_singleton_axes" else True}))

    def test_none_given_is_fine(self):
        check_offset_flags_supported(SimpleNamespace(bbox=None, bbox_axes=None, squeeze_singleton_axes=False))


def _n5_export(path, arr_xyz):
    """An N5 array stored x,y,z the way a Paintera export is (dimensions in x,y,z order)."""
    s = ts.open({"driver": "n5", "kvstore": {"driver": "file", "path": path},
                 "metadata": {"dimensions": list(arr_xyz.shape), "blockSize": list(arr_xyz.shape),
                              "dataType": "uint8", "compression": {"type": "raw"}}},
                create=True).result()
    s.write(arr_xyz).result()


class TestEndToEnd:
    def _container(self, temp_dir, zyx_shape):
        raw = os.path.join(temp_dir, "raw.tif")
        tifffile.imwrite(raw, np.full(zyx_shape, 5, np.uint8), metadata={"axes": "ZYX"})
        out = os.path.join(temp_dir, "c.zarr")
        cli_main(["-i", raw, "-o", out, "--output_format", "zarr3", "--voxel_size", "1,2,3"])
        return out

    def test_n5_export_is_written_z_y_x(self, temp_dir):
        x, y, z = 6, 5, 4
        src = np.zeros((x, y, z), np.uint8)
        src[1, 2, 3] = 7
        src[5, 0, 0] = 9
        export = os.path.join(temp_dir, "export.n5")
        _n5_export(export, src)
        out = self._container(temp_dir, (z, y, x))
        cli_main(["-i", export, "-o", out, "--output_format", "zarr3", "--add-to-existing", "--is_label",
                  "--label-key", "seg", "--input_axes", "xyz", "--axes_order", "zyx",
                  "--output-offset", "0", "0", "0", "--target-shape", str(z), str(y), str(x),
                  "--voxel_size", "1,2,3"])
        got = np.asarray(Readers.auto_detect(os.path.join(out, "labels", "seg", "s0")).get_tensorstore().read().result())
        assert got.shape == (z, y, x)
        assert got[3, 2, 1] == 7 and got[0, 0, 5] == 9
        assert np.array_equal(got, src.transpose(2, 1, 0))

    def test_offset_places_the_oriented_block(self, temp_dir):
        x, y, z = 3, 2, 2
        src = np.ones((x, y, z), np.uint8)
        export = os.path.join(temp_dir, "export.n5")
        _n5_export(export, src)
        out = self._container(temp_dir, (6, 5, 7))
        cli_main(["-i", export, "-o", out, "--output_format", "zarr3", "--add-to-existing", "--is_label",
                  "--label-key", "seg", "--input_axes", "xyz", "--axes_order", "zyx",
                  "--output-offset", "1", "2", "3", "--target-shape", "6", "5", "7", "--voxel_size", "1,2,3"])
        got = np.asarray(Readers.auto_detect(os.path.join(out, "labels", "seg", "s0")).get_tensorstore().read().result())
        assert got.shape == (6, 5, 7)
        assert got[1:3, 2:4, 3:6].all() and got.sum() == src.size  # z 2, y 2, x 3 block, nothing else

    def test_bbox_with_offset_is_refused_not_ignored(self, temp_dir):
        src = np.ones((3, 2, 2), np.uint8)
        export = os.path.join(temp_dir, "export.n5")
        _n5_export(export, src)
        out = self._container(temp_dir, (6, 5, 7))
        with pytest.raises(ValueError, match="--bbox"):
            cli_main(["-i", export, "-o", out, "--output_format", "zarr3", "--add-to-existing", "--is_label",
                      "--label-key", "seg", "--bbox", "0,0,0,2,2,2", "--output-offset", "0", "0", "0",
                      "--target-shape", "6", "5", "7", "--voxel_size", "1,2,3"])


class TestThroughMcp:
    """MCP convert runs the same code, so it gets the same orientation and the same refusal."""

    def test_convert_orients_an_n5_export(self, temp_dir):
        import json
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server as m

        x, y, z = 6, 5, 4
        src = np.zeros((x, y, z), np.uint8)
        src[1, 2, 3] = 7
        export = os.path.join(temp_dir, "export.n5")
        _n5_export(export, src)
        raw = os.path.join(temp_dir, "raw.tif")
        tifffile.imwrite(raw, np.full((z, y, x), 5, np.uint8), metadata={"axes": "ZYX"})
        out = os.path.join(temp_dir, "c.zarr")
        assert json.loads(m.convert(raw, out, voxel_size="1,2,3")).get("status") == "success"
        res = json.loads(m.convert(export, out, voxel_size="1,2,3", is_label=True, data_type="labels",
                                   label_key="seg", add_to_existing=True, input_axes="xyz", axes_order="zyx",
                                   output_offset="0,0,0", target_shape=f"{z},{y},{x}"))
        assert res.get("status") == "success", res
        got = np.asarray(Readers.auto_detect(os.path.join(out, "labels", "seg", "s0")).get_tensorstore().read().result())
        assert np.array_equal(got, src.transpose(2, 1, 0))
