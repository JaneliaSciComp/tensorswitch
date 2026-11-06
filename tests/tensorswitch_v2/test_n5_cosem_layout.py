"""
Tests for the COSEM `transform` layout of an N5 (CellMap's old datasets, e.g. jrc_hela-1_labels.n5).

The array's attributes.json keeps `dimensions` in x, y, z order next to `pixelResolution` and a `transform`
block (`axes` z, y, x with `order: "C"`, `scale`, `translate`, `units`); the group has `axes: [x, y, z]` and the
same transform under `multiscales[0].datasets[0]`. N5Reader reads that once (`cosem_layout`) and gives it to
the TensorStore domain labels, the voxel sizes (by axis name) and `get_translation()`.
"""

import json
import os

import numpy as np
import pytest
import tensorstore as ts

from tensorswitch_v2.readers.n5 import N5Reader, cosem_layout
from tensorswitch_v2.utils.label_ingest import orient_label_source

SHAPE_XYZ = (20, 12, 8)                                   # N5 dimensions: x, y, z


def _transform(scale_zyx=(30.0, 4.0, 5.0), translate_zyx=(15.0, 2.0, 2.5), units="nm", order="C"):
    return {"axes": ["z", "y", "x"], "order": order, "scale": list(scale_zyx), "translate": list(translate_zyx), "units": [units] * 3}


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


def make_cosem_n5(root, data=None, transform=None, pixel_resolution=True, group_axes=True, group_transform=True):
    """A group `root` (axes x, y, z, multiscales with the transform) holding the array `root/s0` of N5 dimensions x, y, z."""
    t = transform if transform is not None else _transform()
    scale_zyx = (t or _transform())["scale"]
    arr = {"dimensions": list(SHAPE_XYZ), "blockSize": [8, 8, 8], "dataType": "uint8", "compression": {"type": "raw"}}
    if pixel_resolution:
        arr["pixelResolution"] = {"dimensions": [scale_zyx[2], scale_zyx[1], scale_zyx[0]], "unit": "nm"}
    if t:
        arr["transform"] = t
    _write_json(f"{root}/s0/attributes.json", arr)
    group = {"n5": "2.0.0"}
    if group_axes:
        group["axes"] = ["x", "y", "z"]
    if group_transform and t:
        group["multiscales"] = [{"datasets": [{"path": "s0", "transform": t}], "name": "labels/x"}]
    _write_json(f"{root}/attributes.json", group)
    store = ts.open({"driver": "n5", "kvstore": {"driver": "file", "path": f"{root}/s0"}, "create": True, "delete_existing": True,
                     "metadata": {"dimensions": list(SHAPE_XYZ), "blockSize": [8, 8, 8], "dataType": "uint8", "compression": {"type": "raw"}}}).result()
    if data is not None:
        store.write(data).result()
    _write_json(f"{root}/s0/attributes.json", arr)          # tensorstore rewrote it: put the layout back
    return root


class TestCosemLayoutParser:
    def test_array_transform_names_the_stored_axes_by_reversing_the_c_order_axes(self):
        meta = {"dimensions": list(SHAPE_XYZ), "transform": _transform()}
        lay = cosem_layout(meta, 3)
        assert lay["axes"] == ["x", "y", "z"]
        assert lay["scale_nm"] == {"x": 5.0, "y": 4.0, "z": 30.0} and lay["translate_nm"] == {"x": 2.5, "y": 2.0, "z": 15.0}

    def test_group_axes_and_multiscales_transform_are_read_when_the_array_has_none(self):
        meta = {"axes": ["x", "y", "z"], "multiscales": [{"datasets": [{"path": "s0", "transform": _transform()}]}]}
        assert cosem_layout(meta, 3)["axes"] == ["x", "y", "z"]

    def test_a_group_level_is_picked_by_its_path(self):
        t1 = _transform(scale_zyx=(60.0, 8.0, 10.0), translate_zyx=(45.0, 6.0, 7.5))
        meta = {"axes": ["x", "y", "z"], "multiscales": [{"datasets": [{"path": "s0", "transform": _transform()}, {"path": "s1", "transform": t1}]}]}
        assert cosem_layout(meta, 3, "s1")["scale_nm"] == {"x": 10.0, "y": 8.0, "z": 60.0} and cosem_layout(meta, 3, "s1")["translate_nm"]["z"] == 45.0
        assert cosem_layout(meta, 3)["scale_nm"]["z"] == 30.0 and cosem_layout(meta, 3, "s9") is None

    def test_units_are_converted_to_nanometers(self):
        meta = {"transform": _transform(scale_zyx=(0.03, 0.004, 0.005), translate_zyx=(0.015, 0.002, 0.0025), units="um")}
        lay = cosem_layout(meta, 3)
        assert lay["scale_nm"]["z"] == pytest.approx(30.0) and lay["translate_nm"]["x"] == pytest.approx(2.5)

    def test_no_transform_or_a_rank_that_does_not_fit_gives_none(self):
        assert cosem_layout({"pixelResolution": {"dimensions": [4, 4, 4], "unit": "nm"}}, 3) is None
        assert cosem_layout({"transform": _transform()}, 4) is None
        assert cosem_layout({"axes": ["x", "y", "q"], "transform": _transform()}, 3) is None


class TestN5Reader:
    def test_the_domain_is_labelled_the_voxel_sizes_are_by_name_and_the_position_is_exposed(self, temp_dir):
        root = make_cosem_n5(os.path.join(temp_dir, "lab.n5"))
        for path in (root + "/s0", root):                     # the array itself, and the group (s0 found by the fallback)
            r = N5Reader(path)
            arr = r.get_tensorstore()
            assert list(arr.domain.labels) == ["x", "y", "z"] and list(arr.shape) == list(SHAPE_XYZ)
            assert dict(r.get_voxel_sizes()) == {"x": 5.0, "y": 4.0, "z": 30.0}
            assert r.get_translation() == {"x": 2.5, "y": 2.0, "z": 15.0}

    def test_the_transform_wins_over_a_pixel_resolution_that_disagrees(self, temp_dir):
        root = make_cosem_n5(os.path.join(temp_dir, "lab.n5"))
        arr = json.load(open(root + "/s0/attributes.json"))
        arr["pixelResolution"] = {"dimensions": [1.0, 1.0, 1.0], "unit": "nm"}
        _write_json(root + "/s0/attributes.json", arr)
        assert dict(N5Reader(root + "/s0").get_voxel_sizes()) == {"x": 5.0, "y": 4.0, "z": 30.0}

    def test_an_n5_without_a_transform_keeps_its_old_behaviour(self, temp_dir):
        root = make_cosem_n5(os.path.join(temp_dir, "plain.n5"), transform={}, group_axes=False, group_transform=False)
        r = N5Reader(root + "/s0")
        arr = r.get_tensorstore()
        assert not any(str(l) for l in arr.domain.labels) and r.get_translation() is None
        assert dict(r.get_voxel_sizes()) == {"x": 5.0, "y": 4.0, "z": 30.0}                 # from pixelResolution, [x, y, z] like the dimensions

    def test_named_axes_let_axes_order_work_without_input_axes(self, temp_dir):
        data = np.arange(np.prod(SHAPE_XYZ), dtype=np.uint8).reshape(SHAPE_XYZ)
        root = make_cosem_n5(os.path.join(temp_dir, "lab.n5"), data=data)
        src = N5Reader(root + "/s0").get_tensorstore()
        out = orient_label_source(src, axes_order=["z", "y", "x"])
        got = np.asarray(out.read().result())
        assert got.shape == (SHAPE_XYZ[2], SHAPE_XYZ[1], SHAPE_XYZ[0]) and np.array_equal(got, data.transpose(2, 1, 0))
