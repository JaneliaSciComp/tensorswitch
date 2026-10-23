"""
Paintera hand-off conversion (utils/paintera_convert): geometry by axis name, bbox crop, command, output check, orchestration with a
fake helper, and one run with the real Java helper (skipped when it is not installed).
Axis order is the point: every geometry here has a different scale and translation per axis.
"""

import json
import os
import stat

import numpy as np
import pytest

from tensorswitch_v2.utils import paintera_convert as pc

SCALE = {"z": 9.0537, "y": 9.0172, "x": 9.0001}              # different on every axis
TRANS = {"z": 325.9346, "y": 324.6203, "x": 100.5}


def make_source(tmp_path, axes=("z", "y", "x"), shape_zyx=(24, 30, 36), flat_attrs=True):
    """A zarr3 label array `seg.zarr/segments_x/s0` stored in `axes` order, with OME multiscales on its parent group."""
    import tensorstore as ts
    group = tmp_path / "seg.zarr" / "segments_x"
    arr = group / "s0"
    arr.mkdir(parents=True)
    rng = np.random.default_rng(1)
    zyx = (rng.integers(0, 50, size=shape_zyx) * 1_000_003).astype(np.uint64)
    zyx[10:14, 10:14, 10:14] = 9_007_199_254_740_993          # an id above 2**53 must survive
    stored = np.ascontiguousarray(zyx.transpose([("z", "y", "x").index(a) for a in axes]))
    a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(arr)},
                 "metadata": {"shape": list(stored.shape), "data_type": "uint64",
                              "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8, 8]}},
                              "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd", "configuration": {"level": 3}}]},
                 "create": True}).result()
    a[...].write(stored).result()
    if flat_attrs:                                              # as the LSD sources: raster units, useless for position
        j = json.load(open(arr / "zarr.json"))
        j["attributes"] = {"axis_names": list(axes), "units": ["raster_voxel"] * 3, "voxel_size": [1, 1, 1], "offset": [36, 36, 36], "types": ["space"] * 3}
        json.dump(j, open(arr / "zarr.json", "w"))
    json.dump({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [{
        "axes": [{"name": n, "type": "space", "unit": "nanometer"} for n in axes],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [SCALE[n] for n in axes]}, {"type": "translation", "translation": [TRANS[n] for n in axes]}]}]}]}}},
        open(group / "zarr.json", "w"))
    return str(arr), zyx


def fake_helper(tmp_path, with_structures=True):
    """A stand-in `paintera-convert.sh`: writes the N5 pieces the checks read, from its own arguments."""
    script = tmp_path / "paintera-convert.sh"
    script.write_text("""#!/usr/bin/env python3
import json, os, sys
a = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--") and "=" in x}
out = a["--output-container"]
os.makedirs(out + "/labels/data/s0"); os.makedirs(out + "/labels/data/s1"); os.makedirs(out + "/labels/data/s2")
%s
json.dump({"n5": "4.0.0"}, open(out + "/attributes.json", "w"))
json.dump({"maxId": int(os.environ.get("FAKE_MAXID", "0")) or 0, "labelBlockLookup": {"type": "n5-filesystem-relative"}, "painteraData": {"type": "label"}}, open(out + "/labels/attributes.json", "w"))
res = [float(v) for v in a["--dataset-resolution"].split(",")]; off = [float(v) for v in a["--dataset-offset"].split(",")]
json.dump({"multiScale": True, "resolution": res, "offset": off}, open(out + "/labels/data/attributes.json", "w"))
""" % ('os.makedirs(out + "/labels/unique-labels"); os.makedirs(out + "/labels/label-to-block-mapping")' if with_structures else "pass"))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_geometry_is_read_by_axis_name_for_any_storage_order(tmp_path):
    for axes in (("z", "y", "x"), ("x", "y", "z")):
        arr, _ = make_source(tmp_path / "".join(axes), axes)
        g = pc.read_geometry(arr)
        assert g.axes == list(axes) and g.scale_nm == SCALE and g.translation_nm == TRANS


def test_expected_resolution_and_offset_are_xyz_and_computed_per_axis():
    g = pc.Geometry("/x", ["z", "y", "x"], [10, 10, 10], "uint64", dict(SCALE), dict(TRANS))
    res, off = pc.expected_geometry_xyz(g, {"z": 5, "y": 10, "x": 15})
    assert res == [SCALE["x"], SCALE["y"], SCALE["z"]]
    assert off == pytest.approx([TRANS["x"] + SCALE["x"] * 15, TRANS["y"] + SCALE["y"] * 10, TRANS["z"] + SCALE["z"] * 5])


@pytest.mark.parametrize("axes", [("z", "y", "x"), ("x", "y", "z")])
def test_crop_is_voxel_identical_and_its_flat_block_is_xyz(tmp_path, axes):
    import tensorstore as ts
    arr, zyx = make_source(tmp_path, axes)
    g = pc.read_geometry(arr)
    out = tmp_path / "crop.zarr"
    max_id, nonzero = pc.crop_to_source(g, (3, 5, 7), (10, 12, 14), str(out))
    got = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "segments" / "s0")}}).result().read().result()
    got_zyx = got.transpose([axes.index(a) for a in "zyx"])
    want = zyx[3:13, 5:17, 7:21]
    assert np.array_equal(got_zyx, want) and max_id == int(want.max()) and nonzero == int(np.count_nonzero(want))
    flat = json.load(open(out / "segments" / "s0" / "zarr.json"))["attributes"]
    assert flat["offset"] == pytest.approx([TRANS["x"] + SCALE["x"] * 7, TRANS["y"] + SCALE["y"] * 5, TRANS["z"] + SCALE["z"] * 3])
    assert flat["voxel_size"] == [SCALE["x"], SCALE["y"], SCALE["z"]]


@pytest.mark.parametrize("start,size", [((0, 0, 0), (25, 30, 36)), ((20, 0, 0), (5, 30, 36)), ((-1, 0, 0), (5, 5, 5)), ((0, 0, 0), (0, 5, 5))])
def test_a_bbox_that_does_not_fit_is_refused_never_clipped(tmp_path, start, size):
    arr, _ = make_source(tmp_path)
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.crop_to_source(pc.read_geometry(arr), start, size, str(tmp_path / "c.zarr"))
    assert e.value.code == "bbox_overflow" and not (tmp_path / "c.zarr").exists()


def test_the_command_passes_resolution_and_offset_xyz_and_a_core_cap():
    cmd = pc.build_command("/h/paintera-convert.sh", "/w/source.zarr", "/o/crop.n5", [9.0, 9.1, 9.2], [1.5, 2.5, 3.5])
    assert cmd[:2] == ["/h/paintera-convert.sh", "to-paintera"] and "--type=label" in cmd and "--winner-takes-all-downsampling" not in cmd
    assert "--dataset-resolution=9.0,9.1,9.2" in cmd and "--dataset-offset=1.5,2.5,3.5" in cmd and "--spark-master=local[4]" in cmd
    assert cmd[cmd.index("--scale") + 1:cmd.index("--scale") + 3] == ["2,2,2", "2,2,2"] and "--block-size=32,32,32" in cmd


def _fake_n5(tmp_path, res, off, max_id=7, structures=True):
    n5 = tmp_path / "o.n5"
    (n5 / "labels" / "data").mkdir(parents=True)
    for s in ("s0", "s1", "s2"):
        (n5 / "labels" / "data" / s).mkdir()
    if structures:
        (n5 / "labels" / "unique-labels").mkdir()
        (n5 / "labels" / "label-to-block-mapping").mkdir()
    json.dump({"maxId": max_id, "labelBlockLookup": {}, "painteraData": {"type": "label"}}, open(n5 / "labels" / "attributes.json", "w"))
    json.dump({"resolution": res, "offset": off}, open(n5 / "labels" / "data" / "attributes.json", "w"))
    return str(n5)


def test_the_output_check_catches_a_swapped_axis_even_when_nearly_isotropic(tmp_path):
    res, off = [9.0172, 9.0172, 9.0537], [5482.4768, 6438.3033, 2725.1752]
    good = pc.check_output(_fake_n5(tmp_path, res, off), res, off, 7)
    assert all(c["status"] == "pass" for c in good)
    swapped = pc.check_output(_fake_n5(tmp_path / "s", res, off[::-1]), res, off, 7)       # the #1138 bug: z,y,x written as the offset
    assert [c["name"] for c in swapped if c["status"] == "fail"] == ["offset_xyz"]
    wrong_res = pc.check_output(_fake_n5(tmp_path / "r", res[::-1], off), res, off, 7)
    assert [c["name"] for c in wrong_res if c["status"] == "fail"] == ["resolution_xyz"]


def test_the_output_check_reports_missing_structures_a_wrong_max_id_and_missing_levels(tmp_path):
    res, off = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    checks = {c["name"]: c["status"] for c in pc.check_output(_fake_n5(tmp_path, res, off, max_id=8, structures=False), res, off, 7)}
    assert checks["paintera_structures"] == "fail" and checks["max_id"] == "fail" and checks["levels"] == "pass"
    assert pc.check_output(str(tmp_path / "nope.n5"), res, off)[0]["status"] == "fail"


def test_a_missing_or_old_helper_is_refused(tmp_path, monkeypatch):
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.check_helper(str(tmp_path), str(tmp_path / "nope.sh"))
    assert e.value.code == "helper_missing"
    h = tmp_path / "h"
    (h / "startup-scripts").mkdir(parents=True)
    (h / "target" / "dependency").mkdir(parents=True)
    (h / "startup-scripts" / "paintera-convert.sh").write_text("#!/bin/sh\n")
    (h / "target" / "dependency" / "paintera-conversion-helper-1.2.9.jar").write_text("")
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.check_helper(str(h))
    assert e.value.code == "helper_too_old"
    (h / "target" / "dependency" / "paintera-conversion-helper-1.3.1-SNAPSHOT.jar").write_text("")
    assert pc.check_helper(str(h)).endswith("paintera-convert.sh")
    monkeypatch.setattr(pc.shutil, "which", lambda n: None)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.check_helper(str(h))
    assert e.value.code == "java_missing"


def test_convert_bbox_with_a_fake_helper_checks_cleans_up_and_never_overwrites(tmp_path, monkeypatch):
    arr, zyx = make_source(tmp_path)
    script = fake_helper(tmp_path)
    expect_max = int(zyx[3:13, 5:17, 7:21].max())
    monkeypatch.setenv("FAKE_MAXID", str(expect_max))
    out = tmp_path / "out" / "crop.n5"
    (tmp_path / "out").mkdir()
    r = pc.convert_bbox(arr, (3, 5, 7), (10, 12, 14), str(out), str(tmp_path / "work"), group=None, script=script, helper_dir=str(tmp_path))
    assert r["ok"], r["checks"]
    assert r["offset_xyz"] == pytest.approx([TRANS["x"] + SCALE["x"] * 7, TRANS["y"] + SCALE["y"] * 5, TRANS["z"] + SCALE["z"] * 3])
    assert not (tmp_path / "work" / "source.zarr").exists()                                # the temporary source is gone
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.convert_bbox(arr, (3, 5, 7), (10, 12, 14), str(out), str(tmp_path / "work2"), group=None, script=script, helper_dir=str(tmp_path))
    assert e.value.code == "output_exists"
    with pytest.raises(pc.PainteraConvertError) as e:                                       # nothing is clipped, nothing is written
        pc.convert_bbox(arr, (20, 0, 0), (10, 5, 5), str(tmp_path / "out" / "c2.n5"), str(tmp_path / "work3"), group=None, script=script, helper_dir=str(tmp_path))
    assert e.value.code == "bbox_overflow" and not (tmp_path / "out" / "c2.n5").exists()


def test_a_helper_that_fails_raises_with_its_log(tmp_path):
    arr, _ = make_source(tmp_path)
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/sh\necho boom\nexit 3\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.convert_bbox(arr, (0, 0, 0), (5, 5, 5), str(tmp_path / "o.n5"), str(tmp_path / "w"), group=None, script=str(bad), helper_dir=str(tmp_path))
    assert e.value.code == "helper_failed" and "boom" in open(str(tmp_path / "w" / "paintera_convert.log")).read()


def test_handoff_permissions_give_the_group_write_with_setgid(tmp_path):
    import grp
    n5 = tmp_path / "x.n5"
    (n5 / "labels").mkdir(parents=True)
    (n5 / "labels" / "attributes.json").write_text("{}")
    os.chmod(n5 / "labels" / "attributes.json", 0o600)
    group = grp.getgrgid(os.getgid()).gr_name
    pc.set_handoff_permissions(str(n5), group)
    assert stat.S_IMODE(os.stat(n5).st_mode) == 0o2775 and stat.S_IMODE(os.stat(n5 / "labels").st_mode) == 0o2775
    assert stat.S_IMODE(os.stat(n5 / "labels" / "attributes.json").st_mode) == 0o664


@pytest.mark.skipif(pc.helper_version(pc.DEFAULT_HELPER) is None or not os.path.exists("/bin/java"), reason="the Java helper is not installed")
def test_real_helper_on_an_anisotropic_crop_gives_the_expected_offset_resolution_and_max_id(tmp_path):
    arr, zyx = make_source(tmp_path)
    out = tmp_path / "out.n5"
    r = pc.convert_bbox(arr, (3, 5, 7), (10, 12, 14), str(out), str(tmp_path / "w"), group=None)
    assert r["ok"], r["checks"]
    d = json.load(open(out / "labels" / "data" / "attributes.json"))
    assert d["resolution"] == pytest.approx([SCALE["x"], SCALE["y"], SCALE["z"]]) and d["offset"] == pytest.approx(r["offset_xyz"])
    assert json.load(open(out / "labels" / "attributes.json"))["maxId"] == int(zyx[3:13, 5:17, 7:21].max())


def _add_outer(arr_dir, outer, expansion=None):
    """Turn make_source's group into an ExM-style one: group-level outer scale 1/expansion_factor, inner values in expanded units."""
    gj = os.path.join(os.path.dirname(arr_dir), "zarr.json")
    j = json.load(open(gj))
    if outer is not None:
        j["attributes"]["ome"]["multiscales"][0]["coordinateTransformations"] = [{"type": "scale", "scale": outer}]
    if expansion is not None:
        j["attributes"]["expansion_microscopy"] = {"expansion_factor": expansion}
    json.dump(j, open(gj, "w"))


def test_the_outer_scale_of_an_expansion_source_is_applied_to_scale_and_translation(tmp_path):
    arr, _ = make_source(tmp_path)                      # inner values: SCALE and TRANS per axis
    _add_outer(arr, [0.0625] * 3, expansion=16)
    g = pc.read_geometry(arr)
    assert g.scale_nm == pytest.approx({a: SCALE[a] * 0.0625 for a in "zyx"})
    assert g.translation_nm == pytest.approx({a: TRANS[a] * 0.0625 for a in "zyx"})
    res, off = pc.expected_geometry_xyz(g, {"z": 2, "y": 3, "x": 4})
    assert res == pytest.approx([SCALE["x"] * 0.0625, SCALE["y"] * 0.0625, SCALE["z"] * 0.0625])
    assert off == pytest.approx([(TRANS["x"] + SCALE["x"] * 4) * 0.0625, (TRANS["y"] + SCALE["y"] * 3) * 0.0625, (TRANS["z"] + SCALE["z"] * 2) * 0.0625])


def test_an_anisotropic_outer_scale_follows_the_axis_names(tmp_path):
    arr, _ = make_source(tmp_path, axes=("x", "y", "z"))
    _add_outer(arr, [0.5, 0.25, 0.125], expansion=4)             # the outer scale is in the group's stored axis order (x, y, z here)
    g = pc.read_geometry(arr)
    assert g.scale_nm == pytest.approx({"x": SCALE["x"] * 0.5, "y": SCALE["y"] * 0.25, "z": SCALE["z"] * 0.125})


def test_an_expansion_source_without_an_outer_scale_is_refused_loudly(tmp_path):
    arr, _ = make_source(tmp_path)
    _add_outer(arr, None, expansion=32)
    with pytest.raises(pc.PainteraConvertError) as e:
        pc.read_geometry(arr)
    assert e.value.code == "outer_scale_missing"


def test_a_plain_em_source_without_an_outer_scale_is_unchanged(tmp_path):
    arr, _ = make_source(tmp_path)
    g = pc.read_geometry(arr)
    assert g.scale_nm == SCALE and g.translation_nm == TRANS
    _add_outer(arr, None, expansion=1)                            # an expansion block of 1 is not expansion
    assert pc.read_geometry(arr).scale_nm == SCALE


def test_the_converted_n5_of_an_expansion_source_carries_the_physical_size(tmp_path, monkeypatch):
    arr, zyx = make_source(tmp_path)
    _add_outer(arr, [0.0625] * 3, expansion=16)
    monkeypatch.setenv("FAKE_MAXID", str(int(zyx[3:13, 5:17, 7:21].max())))
    out = tmp_path / "out" / "crop.n5"
    (tmp_path / "out").mkdir()
    r = pc.convert_bbox(arr, (3, 5, 7), (10, 12, 14), str(out), str(tmp_path / "work"), group=None, script=fake_helper(tmp_path), helper_dir=str(tmp_path))
    assert r["ok"], r["checks"]
    d = json.load(open(out / "labels" / "data" / "attributes.json"))
    assert d["resolution"] == pytest.approx([SCALE["x"] * 0.0625, SCALE["y"] * 0.0625, SCALE["z"] * 0.0625])      # not 16x too coarse


def test_the_staged_crop_is_compared_with_its_source_voxel_for_voxel(tmp_path):
    arr, _ = make_source(tmp_path)
    g = pc.read_geometry(arr)
    out = tmp_path / "crop.zarr"
    pc.crop_to_source(g, (3, 5, 7), (10, 12, 14), str(out))
    assert pc.verify_crop(g, (3, 5, 7), (10, 12, 14), str(out)) == (10 * 12 * 14, 0)
    import tensorstore as ts
    dst = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(out / "segments" / "s0")}}).result()
    dst[0, 0, 0].write(123456789).result()                                      # one voxel changed after the crop: the check must see it
    assert pc.verify_crop(g, (3, 5, 7), (10, 12, 14), str(out)) == (10 * 12 * 14, 1)
    assert pc.verify_crop(g, (3, 5, 7), (9, 12, 14), str(out))[1] > 0                # a crop of the wrong shape is a mismatch, not a pass


def test_convert_bbox_reports_the_crop_identity_check(tmp_path, monkeypatch):
    arr, zyx = make_source(tmp_path)
    monkeypatch.setenv("FAKE_MAXID", str(int(zyx[3:13, 5:17, 7:21].max())))
    (tmp_path / "out").mkdir()
    r = pc.convert_bbox(arr, (3, 5, 7), (10, 12, 14), str(tmp_path / "out" / "c.n5"), str(tmp_path / "w"), group=None, script=fake_helper(tmp_path), helper_dir=str(tmp_path))
    assert r["ok"] and r["checks"][0]["name"] == "crop_identity" and r["checks"][0]["status"] == "pass"
