"""A label added with --add-to-existing --output-offset follows its container: a zarr 2 crop gets a zarr 2 label, a zarr 3 crop a zarr 3 one.

The sparse label path used to write zarr 3 whatever the container was (and read zarr.json for the shape, axes and scale), so a label
could not be added to a zarr 2 crop. The layout below is the one of the Betzig tile crops: (t, c, x, y, z), a label with one channel.
"""

import json
import os

import numpy as np
import pytest
import tensorstore as ts

from tensorswitch_v2.__main__ import main as cli_main
from tensorswitch_v2.api import Readers
from tensorswitch_v2.utils.label_ingest import (
    _infer_axes, _infer_base_scale, _read_target_shape_from_container, container_format, ingest_label_at_offset)
from test_label_ingest_orientation import _n5_export

AXES = [{"name": "t", "type": "time", "unit": "millisecond"}, {"name": "c", "type": "channel"},
        {"name": "x", "type": "space", "unit": "nanometer"}, {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "z", "type": "space", "unit": "nanometer"}]
SCALE = [1.0, 1.0, 108.0, 108.0, 200.0]
SHAPE = (4, 1, 6, 5, 3)                                     # t, c, x, y, z


def _json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


def zarr2_crop(root, with_label=False):
    """A tiny zarr 2 crop: root, raw and labels groups, the root multiscales mirroring raw's, optionally one earlier label."""
    c = str(root / "crop.zarr")
    for g in ("", "raw", "labels"):
        _json(os.path.join(c, g, ".zgroup"), {"zarr_format": 2})
    ms = [{"version": "0.4", "axes": AXES, "datasets": [{"path": "raw/s0", "coordinateTransformations": [{"type": "scale", "scale": SCALE}]}]}]
    _json(c + "/.zattrs", {"multiscales": ms})
    _json(c + "/raw/.zattrs", {"multiscales": [{"version": "0.4", "axes": AXES, "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": SCALE}]}]}]})
    _json(c + "/labels/.zattrs", {"labels": ["old"] if with_label else []})
    if with_label:
        _json(c + "/labels/old/.zgroup", {"zarr_format": 2})
        ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": c + "/labels/old/s0"}, "create": True,
                 "metadata": {"shape": list(SHAPE), "chunks": [1, 1, 4, 4, 4], "dtype": "<u2", "compressor": {"id": "zstd", "level": 5}, "fill_value": 0}}).result()
    return c


def zarr3_crop(root):
    c = str(root / "crop3.zarr")
    ms = {"axes": AXES, "datasets": [{"path": "raw/s0", "coordinateTransformations": [{"type": "scale", "scale": SCALE}]}]}
    _json(c + "/zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [ms], "labels": []}}})
    os.makedirs(c + "/labels", exist_ok=True)
    return c


def block():
    return np.arange(1, 1 + 6 * 5 * 3, dtype=np.uint16).reshape(1, 1, 6, 5, 3)


class TestFormatDetection:
    def test_the_container_decides(self, tmp_path):
        assert container_format(zarr2_crop(tmp_path)) == "zarr2"
        assert container_format(zarr3_crop(tmp_path)) == "zarr3"
        assert container_format(str(tmp_path / "nothing.zarr")) == "zarr3"                    # the default is what it always was

    def test_axes_scale_and_shape_come_from_the_zarr2_files(self, tmp_path):
        c = zarr2_crop(tmp_path, with_label=True)
        assert [a["name"] for a in _infer_axes(c, 5)] == ["t", "c", "x", "y", "z"]
        assert _infer_base_scale(c, 5) == SCALE
        assert _read_target_shape_from_container(c) == list(SHAPE)                           # from the sibling label's s0/.zarray
        assert _infer_base_scale(c, 3) == [1.0, 1.0, 1.0]                                     # no scale of that length: the old fallback

    def test_a_zarr2_crop_without_labels_still_gives_axes_from_raw(self, tmp_path):
        c = zarr2_crop(tmp_path)
        os.remove(c + "/.zattrs")                                                              # root without multiscales: raw's are used
        assert [a["name"] for a in _infer_axes(c, 5)] == ["t", "c", "x", "y", "z"] and _infer_base_scale(c, 5) == SCALE


class TestIngestIntoAZarr2Crop:
    def _ingest(self, c, t=2):
        src = block()
        ingest_label_at_offset(source_ts=ts.array(src[0]), target_container=c, label_tmp_key="seg.tmp", offset=[t, 0, 0, 0, 0],
                               target_shape=list(SHAPE), chunk_shape=(4, 4, 4), dtype="uint16", compression="zstd")
        return src

    def test_s0_is_a_zarr2_array_with_the_zarr3_rule_dtype_chunks_and_codec(self, tmp_path):
        c = zarr2_crop(tmp_path)
        self._ingest(c)
        s0 = c + "/labels/seg.tmp/s0"
        arr = json.load(open(s0 + "/.zarray"))
        assert arr["zarr_format"] == 2 and arr["shape"] == list(SHAPE) and arr["dtype"] == "<u2" and arr["chunks"] == [1, 1, 4, 4, 4]
        assert arr["compressor"] == {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1, "blocksize": 0}   # TensorSwitch's one zarr 2 rule for zstd
        assert arr["dimension_separator"] == "/" and arr["fill_value"] == 0
        assert json.load(open(s0 + "/.zattrs")) == {"_ARRAY_DIMENSIONS": ["t", "c", "x", "y", "z"]}
        assert not os.path.exists(s0 + "/zarr.json")

    def test_zarr_python_with_numcodecs_reads_the_label_back(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        c = zarr2_crop(tmp_path)
        src = self._ingest(c, t=2)
        got = np.asarray(zarr.open_array(c + "/labels/seg.tmp/s0", mode="r", zarr_format=2)[:])
        assert np.array_equal(got[2:3], src) and got[[0, 1, 3]].sum() == 0

    def test_the_codecs_follow_the_zarr3_names(self):
        from tensorswitch_v2.utils.label_ingest import _zarr2_compressor
        assert _zarr2_compressor("zstd", 3) == {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}
        assert _zarr2_compressor("gzip", 4) == {"id": "gzip", "level": 4} and _zarr2_compressor("gz", 4) == {"id": "gzip", "level": 4}
        assert _zarr2_compressor("blosc", 5)["id"] == "blosc"

    def test_only_the_target_timepoint_holds_data(self, tmp_path):
        c = zarr2_crop(tmp_path)
        src = self._ingest(c, t=2)
        got = np.asarray(ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": c + "/labels/seg.tmp/s0"}}).result().read().result())
        assert got.shape == SHAPE and np.array_equal(got[2:3, :, :, :, :], src[:, :, :, :, :]) and got[[0, 1, 3]].sum() == 0

    def test_the_label_group_is_written_the_zarr2_way(self, tmp_path):
        c = zarr2_crop(tmp_path)
        self._ingest(c)
        g = c + "/labels/seg.tmp"
        assert json.load(open(g + "/.zgroup")) == {"zarr_format": 2}
        attrs = json.load(open(g + "/.zattrs"))
        ms = attrs["multiscales"][0]
        assert ms["version"] == "0.4" and [a["name"] for a in ms["axes"]] == ["t", "c", "x", "y", "z"]
        assert ms["datasets"] == [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": SCALE}]}]
        assert attrs["image-label"]["source"] == {"image": "../../raw"} and not os.path.exists(g + "/zarr.json")

    def test_the_first_label_of_a_crop_gets_its_labels_group(self, tmp_path):
        c = zarr2_crop(tmp_path)
        os.remove(c + "/labels/.zgroup")
        self._ingest(c)
        assert json.load(open(c + "/labels/.zgroup")) == {"zarr_format": 2}


class TestThroughTheCommandLine:
    def _export(self, temp_dir):
        src = np.zeros((6, 5, 3), np.uint8)                    # x, y, z (the N5 helper writes uint8; --dtype casts it)
        src[1, 2, 1] = 7
        export = os.path.join(temp_dir, "export.n5")
        _n5_export(export, src)
        return export, src

    def _run(self, export, out, *extra):
        cli_main(["-i", export, "-o", out, "--add-to-existing", "--is_label", "--label-key", "seg", "--input_axes", "xyz", "--axes_order", "xyz",
                  "--output-offset", "2", "0", "0", "0", "0", "--target-shape", *[str(n) for n in SHAPE], "--voxel_size", "108,108,200",
                  "--dtype", "uint16", *extra])

    def test_a_zarr2_crop_gets_a_zarr2_label_without_asking_for_it(self, tmp_path):
        export, src = self._export(str(tmp_path))
        c = zarr2_crop(tmp_path, with_label=True)
        self._run(export, c)                                                                     # --output_format left at its zarr3 default
        s0 = c + "/labels/seg/s0"
        assert os.path.exists(s0 + "/.zarray") and not os.path.exists(s0 + "/zarr.json")
        got = np.asarray(ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": s0}}).result().read().result())
        assert got.shape == SHAPE and got[2, 0, 1, 2, 1] == 7 and got.sum() == 7
        assert json.load(open(c + "/labels/.zattrs"))["labels"] == ["old", "seg"]                # registered the zarr 2 way, the earlier label kept
        assert not os.path.exists(c + "/labels/seg.tmp") and not os.path.exists(c + "/labels/zarr.json")

    def test_the_result_is_readable_by_the_tensorswitch_reader(self, tmp_path):
        export, src = self._export(str(tmp_path))
        c = zarr2_crop(tmp_path)
        self._run(export, c)
        got = np.asarray(Readers.auto_detect(c + "/labels/seg/s0").get_tensorstore().read().result())
        assert got.shape == SHAPE and got.sum() == 7

    def test_a_zarr3_crop_still_gets_a_zarr3_label(self, tmp_path):
        export, src = self._export(str(tmp_path))
        c = zarr3_crop(tmp_path)
        self._run(export, c, "--output_format", "zarr3")
        assert os.path.exists(c + "/labels/seg/s0/zarr.json") and not os.path.exists(c + "/labels/seg/s0/.zarray")
        assert json.load(open(c + "/zarr.json"))["attributes"]["ome"]["labels"] == ["seg"]

    def test_zarr2_cannot_be_forced_into_a_zarr3_crop(self, tmp_path):
        export, src = self._export(str(tmp_path))
        c = zarr3_crop(tmp_path)
        with pytest.raises(ValueError, match="zarr3 container"):
            self._run(export, c, "--output_format", "zarr2")
        assert not os.path.exists(c + "/labels/seg") and not os.path.exists(c + "/labels/seg.tmp")


class TestThePyramidOfAZarr2Label:
    def test_the_pyramid_step_builds_zarr2_levels_next_to_s0(self, tmp_path):
        shape = (3, 1, 300, 260, 140)
        c = zarr2_crop(tmp_path)
        src = np.zeros((300, 260, 140), np.uint8)                                       # x, y, z
        src[10:200, 5:100, 3:60] = 5
        export = str(tmp_path / "export.n5")
        _n5_export(export, src)
        cli_main(["-i", export, "-o", c, "--add-to-existing", "--is_label", "--label-key", "seg", "--input_axes", "xyz", "--axes_order", "xyz",
                  "--output-offset", "1", "0", "0", "0", "0", "--target-shape", *[str(n) for n in shape], "--voxel_size", "108,108,200",
                  "--dtype", "uint16", "--preset", "miaai"])
        cli_main(["--auto_multiscale", "--is_label", "-i", c + "/labels/seg/s0", "-o", c + "/labels/seg", "--preset", "miaai", "--output_format", "zarr2"])
        s1 = json.load(open(c + "/labels/seg/s1/.zarray"))
        assert s1["zarr_format"] == 2 and s1["shape"] == [3, 1, 150, 130, 70] and s1["dtype"] == "<u2"
        ms = json.load(open(c + "/labels/seg/.zattrs"))["multiscales"][0]
        assert [d["path"] for d in ms["datasets"]] == ["s0", "s1"] and json.load(open(c + "/labels/.zattrs"))["labels"] == ["seg"]
        got = np.asarray(ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": c + "/labels/seg/s1"}}).result().read().result())
        assert got[1].max() == 5 and got[[0, 2]].sum() == 0                              # the mode of the block: the label survives, other timepoints stay empty
