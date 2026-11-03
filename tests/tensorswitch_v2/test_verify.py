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


def _open_files(*paths):
    wanted = {os.path.realpath(p) for p in paths}
    fds = os.listdir("/proc/self/fd")
    found = []
    for fd in fds:
        try:
            target = os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            continue
        if target in wanted:
            found.append(target)
    return found


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_source_files_are_not_left_open(container, data):
    verify(container, data)
    assert _open_files(data["raw_path"], data["lab_path"]) == []


def _edit_json(path, change):
    with open(path) as handle:
        doc = json.load(handle)
    change(doc)
    with open(path, "w") as handle:
        json.dump(doc, handle)


class TestMetadataHygiene:
    def test_clean_container_passes_both_checks(self, container, data):
        report = verify(container, data)
        assert status(report, "levels_listed") == "pass" and status(report, "no_zarr2_files") == "pass"

    def test_level_on_disk_but_not_listed_fails(self, container, data):
        shutil.copytree(os.path.join(container, "raw", "s0"), os.path.join(container, "raw", "s1"))
        report = verify(container, data)
        assert status(report, "levels_listed") == "fail"
        detail = next(c["detail"] for c in report["checks"] if c["name"] == "levels_listed")
        assert "raw: levels ['s1'] exist but are not listed" in detail

    def test_stale_root_is_caught_even_when_the_group_is_right(self, container, data):
        shutil.copytree(os.path.join(container, "raw", "s0"), os.path.join(container, "raw", "s1"))
        _edit_json(os.path.join(container, "raw", "zarr.json"), lambda d: d["attributes"]["ome"]["multiscales"][0]
                   ["datasets"].append({"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [80, 146, 146]}]}))
        detail = next(c["detail"] for c in verify(container, data)["checks"] if c["name"] == "levels_listed")
        assert "root zarr.json lists ['s0'] for 'raw' but the folder has ['s0', 's1']" in detail

    def test_listed_level_missing_on_disk_fails(self, container, data):
        _edit_json(os.path.join(container, "raw", "zarr.json"), lambda d: d["attributes"]["ome"]["multiscales"][0]
                   ["datasets"].append({"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [80, 146, 146]}]}))
        detail = next(c["detail"] for c in verify(container, data)["checks"] if c["name"] == "levels_listed")
        assert "listed but missing on disk" in detail

    @pytest.mark.parametrize("name", [".zgroup", ".zattrs"])
    def test_zarr2_file_in_a_zarr3_container_fails(self, container, data, name):
        open(os.path.join(container, "raw", name), "w").write("{}")
        report = verify(container, data)
        assert status(report, "no_zarr2_files") == "fail" and report["overall"] == "fail"


    def test_root_listing_a_label_only_container_is_understood(self, tmp_path):
        out = tmp_path / "labels_only.zarr"
        (out / "labels" / "seg" / "s0").mkdir(parents=True)
        (out / "labels" / "seg" / "s1").mkdir()
        (out / "labels" / "seg" / "zarr.json").write_text(json.dumps({"attributes": {"ome": {"multiscales": [{"datasets": [
            {"path": "s0"}, {"path": "s1"}]}]}}}))
        (out / "zarr.json").write_text(json.dumps({"attributes": {"ome": {"multiscales": [{"datasets": [
            {"path": "labels/seg/s0"}, {"path": "labels/seg/s1"}]}]}}}))
        checks = []
        v._check_listed_levels(checks, str(out), [("labels/seg", str(out / "labels" / "seg"))])
        assert checks[0]["status"] == "pass", checks
        (out / "labels" / "seg" / "s2").mkdir()
        _edit_json(str(out / "labels" / "seg" / "zarr.json"), lambda d: d["attributes"]["ome"]["multiscales"][0]["datasets"].append({"path": "s2"}))
        checks = []
        v._check_listed_levels(checks, str(out), [("labels/seg", str(out / "labels" / "seg"))])
        assert checks[0]["status"] == "fail" and "root zarr.json lists ['s0', 's1'] for 'labels/seg'" in checks[0]["detail"]


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

    def test_a_reordered_output_is_matched_by_axis_name(self, temp_dir, data):
        out = os.path.join(temp_dir, "ax.zarr")
        convert(data["raw_path"], out, "--axes_order", "xyz")
        report = v.verify_output(out, data["raw_path"], {"voxel_size": VOX})
        detail = next(c["detail"] for c in report["checks"] if c["name"] == "identity:raw")
        assert status(report, "identity:raw") == "pass" and "matched by name" in detail

    def test_changed_data_still_fails_when_matched_by_name(self, data):
        import tensorstore as ts
        from tensorswitch_v2.utils.verify import compare_with_source
        src = ts.array(data["raw"])                                   # z,y,x
        good = np.transpose(data["raw"], (2, 1, 0)).copy()            # the same data as x,y,z
        assert compare_with_source(ts.array(good), src, src_names=["z", "y", "x"],
                                   out_names=["x", "y", "z"])["status"] == "pass"
        bad = good.copy()
        bad[3, 4, 5] = bad[3, 4, 5] + 1
        assert compare_with_source(ts.array(bad), src, src_names=["z", "y", "x"],
                                   out_names=["x", "y", "z"])["status"] == "fail"

    def test_names_that_do_not_match_are_unverified_not_guessed(self, data):
        import tensorstore as ts
        from tensorswitch_v2.utils.verify import compare_with_source
        src = ts.array(data["raw"])
        out = ts.array(data["raw"].copy())
        result = compare_with_source(out, src, src_names=["z", "y", "x"], out_names=["a", "b", "c"])
        assert result["status"] == "unverified" and "by name" in result["detail"]

    def test_renamed_and_reordered_source_axes_with_input_axes(self, temp_dir):
        arr = np.random.default_rng(8).integers(1, 250, (6, 16, 20, 3), dtype=np.uint8)
        src = os.path.join(temp_dir, "pages.tif")
        tifffile.imwrite(src, arr, photometric="rgb", shaped=False)         # the reader calls the axes i,y,x,s
        out = os.path.join(temp_dir, "o.zarr")
        convert(src, out, "--input_axes", "zyxc")                           # output is c,z,y,x
        without = v.verify_output(out, src, {"voxel_size": VOX})
        assert status(without, "identity:raw") == "unverified"             # names i,s vs z,c: cannot be matched
        with_axes = v.verify_output(out, src, {"voxel_size": VOX, "input_axes": "zyxc"})
        assert status(with_axes, "identity:raw") == "pass" and with_axes["overall"] == "pass"

    def test_wrong_input_axes_length_is_unverified(self, temp_dir):
        arr = np.random.default_rng(9).integers(1, 250, (6, 16, 20), dtype=np.uint8)
        src = os.path.join(temp_dir, "v.tif")
        tifffile.imwrite(src, arr)
        out = os.path.join(temp_dir, "v.zarr")
        convert(src, out)
        report = v.verify_output(out, src, {"voxel_size": VOX, "input_axes": "zyxc"})
        assert status(report, "identity:raw") == "unverified"

    def test_unmappable_shapes_are_unverified_not_failed(self, temp_dir, data):
        out = os.path.join(temp_dir, "crop.zarr")
        convert(data["raw_path"], out, "--bbox", "0,0,0,4,8,8")
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


class TestLabelSourceDataset:
    def test_label_read_from_a_dataset_inside_an_hdf5_file(self, temp_dir):
        import h5py

        rng = np.random.default_rng(5)
        # more than 10 slices: HDF5 has no axis names and a first dimension of 10 or less is read as a channel
        raw = rng.integers(1, 200, (12, 16, 16), dtype=np.uint8)
        lab = rng.integers(0, 4, (12, 16, 16), dtype=np.uint8)
        path = os.path.join(temp_dir, "both.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("vol/raw", data=raw)
            f.create_dataset("vol/lab", data=lab)
        out = os.path.join(temp_dir, "both.zarr")
        convert(path, out, "--dataset_path", "vol/raw")
        convert(path, out, "--dataset_path", "vol/lab", "--add-to-existing", "--is_label", "--label-key", "seg")
        report = v.verify_output(out, path, {"voxel_size": VOX, "dataset_path": "vol/raw",
                                             "labels": {"seg": f"{path}::vol/lab"}})
        assert report["overall"] == "pass", report["checks"]
        wrong = v.verify_output(out, path, {"voxel_size": VOX, "dataset_path": "vol/raw",
                                            "labels": {"seg": f"{path}::vol/raw"}})
        assert status(wrong, "identity:labels/seg") == "fail"

    def test_a_short_first_dimension_read_as_a_channel_is_caught(self, temp_dir):
        """HDF5 with 8 slices is written with axes c,y,x (no z): the voxel size check reports it."""
        import h5py

        path = os.path.join(temp_dir, "short.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("vol/raw", data=np.random.default_rng(6).integers(1, 200, (8, 16, 16), dtype=np.uint8))
        out = os.path.join(temp_dir, "short.zarr")
        convert(path, out, "--dataset_path", "vol/raw")
        report = v.verify_output(out, path, {"voxel_size": VOX, "dataset_path": "vol/raw"})
        assert status(report, "voxel_size") == "fail" and report["overall"] == "fail"

    def test_labels_only_container(self, temp_dir, data):
        out = os.path.join(temp_dir, "lo.zarr")
        convert(data["lab_path"], out, "--is_label", "--label-key", "seg")
        report = v.verify_output(out, None, {"voxel_size": VOX, "image_key": "", "labels": {"seg": data["lab_path"]}})
        assert status(report, "structure") == "pass" and status(report, "identity:labels/seg") == "pass"


class TestMcpToolAxes:
    def test_tool_passes_input_axes_and_label_axes(self, temp_dir):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server
        arr = np.random.default_rng(10).integers(1, 250, (6, 16, 20, 3), dtype=np.uint8)
        lab = np.random.default_rng(11).integers(0, 4, (16, 20, 6), dtype=np.uint16)
        raw, mask = os.path.join(temp_dir, "r.tif"), os.path.join(temp_dir, "m.tif")
        tifffile.imwrite(raw, arr, photometric="rgb", shaped=False)
        tifffile.imwrite(mask, lab, photometric="minisblack", planarconfig="contig", shaped=False)
        out = os.path.join(temp_dir, "c.zarr")
        convert(raw, out, "--input_axes", "zyxc")
        convert(mask, out, "--add-to-existing", "--is_label", "--label-key", "segmentation", "--input_axes", "yxz")
        report = json.loads(mcp_server.verify_output(out, raw, voxel_size=VOX, labels=f"segmentation={mask}",
                                                     input_axes="zyxc", label_input_axes="segmentation=yxz"))
        assert report["overall"] == "pass", [c for c in report["checks"] if c["status"] != "pass"]
        names = {c["name"]: c["status"] for c in report["checks"]}
        assert names["identity:raw"] == names["identity:labels/segmentation"] == "pass"

    def test_bad_label_axes_text_is_a_validation_error(self, temp_dir):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server
        result = json.loads(mcp_server.verify_output(temp_dir, label_input_axes="segmentation"))
        assert result["error"] == "validation_error"
