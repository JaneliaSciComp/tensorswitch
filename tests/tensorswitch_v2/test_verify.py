"""
verify_output: a correct conversion passes, and each kind of damage is caught.
Outputs come from the real CLI conversion; damage is applied to the written store.
"""

import contextlib
import io
import json
import os
import shutil

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from tensorswitch_v2.__main__ import main as cli_main
from tensorswitch_v2.utils import verify as v

VOX = "8,8,40"


def convert(src, out, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", src, "-o", out, "--voxel_size", VOX, "--quiet", *extra])


def tif(folder, name, array):
    path = os.path.join(folder, name)
    tifffile.imwrite(path, array, metadata={"axes": "ZYX"})
    return path


@pytest.fixture
def data(temp_dir):
    rng = np.random.default_rng(0)
    raw = rng.integers(1, 200, (8, 16, 16), dtype=np.uint8)
    lab = rng.integers(0, 4, (8, 16, 16), dtype=np.uint8)
    return {"raw": raw, "lab": lab, "raw_path": tif(temp_dir, "raw.tif", raw), "lab_path": tif(temp_dir, "lab.tif", lab)}


@pytest.fixture
def container(temp_dir, data):
    out = os.path.join(temp_dir, "c.zarr")
    convert(data["raw_path"], out)
    convert(data["lab_path"], out, "--add-to-existing", "--is_label", "--label-key", "seg")
    return out


def status(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def verify(container, data, **expected):
    expected.setdefault("voxel_size", VOX)
    expected.setdefault("labels", {"seg": data["lab_path"]})
    return v.verify_output(container, data["raw_path"], expected)


class TestGoodConversion:
    def test_passes_and_every_check_ran(self, container, data):
        report = verify(container, data)
        assert report["overall"] == "pass", report["checks"]
        names = {c["name"] for c in report["checks"]}
        assert {"structure", "voxel_size", "levels:raw", "levels:labels/seg", "data:raw",
                "identity:raw", "identity:labels/seg"} <= names

    def test_small_array_is_compared_whole(self, container, data):
        detail = next(c["detail"] for c in verify(container, data)["checks"] if c["name"] == "identity:raw")
        assert "whole array" in detail

    def test_large_array_is_sampled(self, temp_dir, data):
        out = os.path.join(temp_dir, "big.zarr")
        convert(data["raw_path"], out)
        store = v._open(os.path.join(out, "raw", "s0"))
        src = v._open(data["raw_path"])
        result = v.compare_with_source(store, src, samples=4, full_bytes=10)
        assert result["mode"] == "sampled" and result["status"] == "pass"

    def test_report_is_written_into_the_container(self, container, data):
        verify(container, data)
        saved = json.load(open(os.path.join(container, "verification.json")))
        assert saved["overall"] == "pass" and saved["tensorswitch"] and saved["checks"]

    def test_report_is_not_written_when_asked_not_to(self, container, data):
        v.verify_output(container, data["raw_path"], {"voxel_size": VOX}, write_report=False)
        assert not os.path.exists(os.path.join(container, "verification.json"))

    def test_verification_json_does_not_count_as_leftover(self, container, data):
        verify(container, data)
        assert verify(container, data)["overall"] == "pass"


class TestDamageIsCaught:
    def test_one_changed_value(self, container, data):
        wrong = data["raw"].copy()
        wrong[5, 3, 3] ^= 1
        report = v.verify_output(container, tif(os.path.dirname(container), "wrong.tif", wrong), {"voxel_size": VOX})
        assert status(report, "identity:raw") == "fail" and report["overall"] == "fail"
        assert "first differing position" in next(c["detail"] for c in report["checks"] if c["name"] == "identity:raw")

    def test_transposed_axes(self, container, data):
        transposed = np.transpose(data["raw"], (0, 2, 1)).copy()
        report = v.verify_output(container, tif(os.path.dirname(container), "t.tif", transposed), {"voxel_size": VOX})
        assert status(report, "identity:raw") == "fail"

    def test_shifted_slices(self, container, data):
        shifted = np.roll(data["raw"], 1, axis=0)
        report = v.verify_output(container, tif(os.path.dirname(container), "s.tif", shifted), {"voxel_size": VOX})
        assert status(report, "identity:raw") == "fail"

    def test_wrong_voxel_size(self, container, data):
        report = verify(container, data, voxel_size="8,8,41")
        assert status(report, "voxel_size") == "fail" and report["overall"] == "fail"

    def test_missing_label(self, container, data):
        report = verify(container, data, labels={"seg": data["lab_path"], "other": data["lab_path"]})
        assert status(report, "structure") == "fail"

    def test_leftover_tmp(self, container, data):
        os.makedirs(os.path.join(container, "labels", "seg2.tmp"))
        assert status(verify(container, data), "structure") == "fail"

    def test_all_zero_raw(self, temp_dir, data):
        zeros = tif(temp_dir, "z.tif", np.zeros((8, 16, 16), dtype=np.uint8))
        out = os.path.join(temp_dir, "z.zarr")
        convert(zeros, out)
        report = v.verify_output(out, zeros, {"voxel_size": VOX})
        assert status(report, "data:raw") == "fail"

    def test_constant_labels_are_allowed(self, temp_dir, data):
        out = os.path.join(temp_dir, "c2.zarr")
        convert(data["raw_path"], out)
        empty = tif(temp_dir, "e.tif", np.zeros((8, 16, 16), dtype=np.uint8))
        convert(empty, out, "--add-to-existing", "--is_label", "--label-key", "seg")
        report = v.verify_output(out, data["raw_path"], {"voxel_size": VOX, "labels": {"seg": empty}})
        assert status(report, "data:labels/seg") == "pass"

    def test_missing_pyramid_level(self, temp_dir):
        big = tif(temp_dir, "b.tif", np.random.default_rng(1).integers(1, 200, (64, 512, 512), dtype=np.uint8))
        out = os.path.join(temp_dir, "b.zarr")
        convert(big, out, "--auto_multiscale")
        report = v.verify_output(out, big, {"voxel_size": VOX})
        assert report["overall"] == "pass", report["checks"]
        shutil.rmtree(os.path.join(out, "raw", "s1"))
        assert status(v.verify_output(out, big, {"voxel_size": VOX}), "levels:raw") == "fail"

    def test_output_that_does_not_exist(self, temp_dir):
        report = v.verify_output(os.path.join(temp_dir, "nope.zarr"))
        assert report["overall"] == "fail"


class TestUnverifiedIsNeverAPass:
    def test_no_source_given(self, container):
        report = v.verify_output(container, None, {"voxel_size": VOX})
        assert status(report, "identity:raw") == "unverified" and report["overall"] == "unverified"
        assert "identity:raw" in report["unverified"]

    def test_source_deleted_after_conversion(self, container, data):
        os.remove(data["raw_path"])
        report = v.verify_output(container, data["raw_path"], {"voxel_size": VOX})
        assert status(report, "identity:raw") == "unverified" and report["overall"] == "unverified"

    def test_no_expected_voxel_size(self, container, data):
        report = v.verify_output(container, data["raw_path"], {})
        assert status(report, "voxel_size") == "unverified"

    def test_intentionally_cast_values(self, temp_dir, data):
        out = os.path.join(temp_dir, "cast.zarr")
        convert(data["raw_path"], out, "--dtype", "uint16")
        report = v.verify_output(out, data["raw_path"], {"voxel_size": VOX, "output_dtype": "uint16"})
        assert status(report, "identity:raw") == "unverified"

    def test_dtype_changed_without_saying_so_fails(self, temp_dir, data):
        out = os.path.join(temp_dir, "cast2.zarr")
        convert(data["raw_path"], out, "--dtype", "uint16")
        assert status(v.verify_output(out, data["raw_path"], {"voxel_size": VOX}), "identity:raw") == "fail"


class TestMappings:
    def test_bbox_conversion_is_compared_at_the_bbox_position(self, temp_dir, data):
        out = os.path.join(temp_dir, "bbox.zarr")
        convert(data["raw_path"], out, "--bbox", "2,4,4,4,8,8")
        report = v.verify_output(out, data["raw_path"], {"voxel_size": VOX, "bbox": "2,4,4,4,8,8"})
        assert status(report, "identity:raw") == "pass", report["checks"]
        # same output checked against the wrong place must fail
        wrong = v.verify_output(out, data["raw_path"], {"voxel_size": VOX, "bbox": "0,0,0,4,8,8"})
        assert status(wrong, "identity:raw") == "fail"

    def test_nd_bbox_with_axes(self, temp_dir):
        arr = np.random.default_rng(2).integers(1, 200, (2, 3, 8, 16, 16), dtype=np.uint8)
        src = os.path.join(temp_dir, "tczyx.tif")
        tifffile.imwrite(src, arr, metadata={"axes": "TCZYX"})
        out = os.path.join(temp_dir, "nd.zarr")
        convert(src, out, "--bbox", "0,0,2,4,4,1,1,4,8,8", "--bbox_axes", "0,1,2,3,4", "--squeeze_singleton_axes")
        report = v.verify_output(out, src, {"voxel_size": VOX, "bbox": "0,0,2,4,4,1,1,4,8,8", "bbox_axes": "0,1,2,3,4"})
        assert status(report, "identity:raw") == "pass", report["checks"]

    def test_squeezed_singleton_axes(self, temp_dir):
        arr = np.random.default_rng(3).integers(1, 200, (1, 1, 8, 16, 16), dtype=np.uint8)
        src = os.path.join(temp_dir, "sq.tif")
        tifffile.imwrite(src, arr, metadata={"axes": "TCZYX"})
        out = os.path.join(temp_dir, "sq.zarr")
        convert(src, out, "--squeeze_singleton_axes")
        assert status(v.verify_output(out, src, {"voxel_size": VOX}), "identity:raw") == "pass"

    def test_unmappable_shapes_are_unverified_not_failed(self, temp_dir, data):
        out = os.path.join(temp_dir, "ax.zarr")
        convert(data["raw_path"], out, "--axes_order", "xyz")
        report = v.verify_output(out, data["raw_path"], {"voxel_size": VOX})
        assert status(report, "identity:raw") == "unverified"

    def test_hdf5_source_with_dataset_path(self, temp_dir):
        import h5py

        arr = np.random.default_rng(4).integers(1, 200, (8, 16, 16), dtype=np.uint8)
        path = os.path.join(temp_dir, "s.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("vol/raw", data=arr)
        out = os.path.join(temp_dir, "h5.zarr")
        convert(path, out, "--dataset_path", "vol/raw")
        report = v.verify_output(out, path, {"voxel_size": VOX, "dataset_path": "vol/raw"})
        assert status(report, "identity:raw") == "pass", report["checks"]


class TestGroup:
    def test_group_matches(self, container, data):
        import grp

        name = grp.getgrgid(os.stat(container).st_gid).gr_name
        assert status(verify(container, data, group=name), "group") == "pass"

    def test_group_mismatch(self, container, data):
        import grp

        other = next((g for g in os.getgroups() if g != os.stat(container).st_gid), None)
        if other is None:
            pytest.skip("user has no second group")
        report = verify(container, data, group=grp.getgrgid(other).gr_name)
        assert status(report, "group") == "fail"

    def test_unknown_group_is_unverified(self, container, data):
        assert status(verify(container, data, group="no_such_group_xyz"), "group") == "unverified"


class TestMcpTool:
    @pytest.fixture
    def tool(self):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        return lambda *a, **k: json.loads(mcp_server.verify_output(*a, **k))

    def test_good_conversion_passes(self, tool, container, data):
        report = tool(container, data["raw_path"], voxel_size=VOX, labels=f"seg={data['lab_path']}")
        assert report["overall"] == "pass", report["checks"]

    def test_damage_is_reported(self, tool, container, data):
        wrong = tif(os.path.dirname(container), "w.tif", np.roll(data["raw"], 1, axis=0))
        report = tool(container, wrong, voxel_size=VOX)
        assert report["overall"] == "fail" and "identity:raw" in report["failures"]

    def test_no_source_is_unverified_not_pass(self, tool, container):
        assert tool(container, voxel_size=VOX)["overall"] == "unverified"

    def test_bad_labels_argument(self, tool, container):
        assert tool(container, labels="seg")["error"] == "validation_error"

    def test_stdout_stays_clean(self, tool, container, data, capsys):
        tool(container, data["raw_path"], voxel_size=VOX)
        assert capsys.readouterr().out == ""
