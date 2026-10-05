"""The miaai preset's metadata conventions and --extra_attributes (real conversions, no cluster)."""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import tifffile

from tensorswitch_v2.__main__ import main as cli_main
from tensorswitch_v2.utils import miaai_metadata as mm
from tensorswitch_v2.utils import verify as v

VOX = ["--voxel_size", "8,8,40", "--voxel_unit", "nanometer"]


def convert(src, out, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", src, "-o", out, *VOX, "--quiet", *extra])


def read(path):
    return json.load(open(path))["attributes"]


def ms(path):
    return read(path)["ome"]["multiscales"][0]


@pytest.fixture
def tifs(temp_dir):
    rng = np.random.default_rng(1)
    out = {}
    for name, shape in (("raw", (8, 16, 16)), ("lab", (8, 16, 16)), ("big", (64, 512, 512))):
        path = os.path.join(temp_dir, f"{name}.tif")
        tifffile.imwrite(path, rng.integers(1, 5, shape, dtype=np.uint8), metadata={"axes": "ZYX"})
        out[name] = path
    return out


@pytest.fixture
def container(temp_dir, tifs):
    out = os.path.join(temp_dir, "c.zarr")
    convert(tifs["raw"], out, "--preset", "miaai")
    convert(tifs["lab"], out, "--preset", "miaai", "--add-to-existing", "--is_label", "--label-key", "seg")
    return out


class TestConventions:
    def test_name_removed_and_identity_outer_transform_everywhere(self, container):
        for rel in ("zarr.json", "raw/zarr.json", "labels/seg/zarr.json"):
            m = ms(os.path.join(container, rel))
            assert "name" not in m, rel
            assert m["coordinateTransformations"] == [{"type": "scale", "scale": [1.0, 1.0, 1.0]}], rel

    def test_dataset_level_scales_are_untouched(self, container):
        assert ms(os.path.join(container, "raw", "zarr.json"))["datasets"][0]["coordinateTransformations"][0]["scale"] == [40.0, 8.0, 8.0]

    def test_marker_records_the_preset(self, container):
        assert read(os.path.join(container, "zarr.json"))["tensorswitch"] == {"preset": "miaai"}

    def test_old_preset_name_does_the_same(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "old.zarr")
        convert(tifs["raw"], out, "--preset", "mia_lmvd")
        assert "name" not in ms(os.path.join(out, "raw", "zarr.json"))

    def test_without_the_preset_nothing_changes(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "plain.zarr")
        convert(tifs["raw"], out)
        m = ms(os.path.join(out, "raw", "zarr.json"))
        assert m.get("name") == "raw" and "coordinateTransformations" not in m
        assert "tensorswitch" not in read(os.path.join(out, "zarr.json"))

    def test_expansion_factor_scales_the_spatial_axes_only(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "exm.zarr")
        convert(tifs["raw"], out, "--preset", "miaai", "--expansion_factor", "4")
        assert ms(os.path.join(out, "raw", "zarr.json"))["coordinateTransformations"][0]["scale"] == [0.25, 0.25, 0.25]
        assert read(os.path.join(out, "zarr.json"))["tensorswitch"] == {"preset": "miaai", "expansion_factor": 4.0}

    def test_idempotent(self, container):
        assert mm.finalize_container(container) == []

    def test_pyramid_keeps_the_conventions_and_passes_verification(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "pyr.zarr")
        convert(tifs["big"], out, "--preset", "miaai", "--auto_multiscale")
        assert len([d for d in os.listdir(os.path.join(out, "raw")) if d.startswith("s")]) >= 2
        for rel in ("zarr.json", "raw/zarr.json"):
            m = ms(os.path.join(out, rel))
            assert "name" not in m and m["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
            assert len(m["datasets"]) >= 2
        report = v.verify_output(out, tifs["big"], {"voxel_size": "8,8,40"}, write_report=False)
        assert {c["name"]: c["status"] for c in report["checks"]}["levels_listed"] == "pass"


class TestLaterRewritesApplyTheMarker:
    def test_a_stale_group_is_repaired_when_a_pyramid_step_touches_it(self, container):
        raw = os.path.join(container, "raw")
        path = os.path.join(raw, "zarr.json")
        doc = json.load(open(path))
        doc["attributes"]["ome"]["multiscales"][0]["name"] = "back_again"
        json.dump(doc, open(path, "w"))
        assert mm.apply_marker_if_any(raw) == ["raw/zarr.json"]
        assert "name" not in ms(path)

    def test_container_without_marker_is_left_alone(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "plain.zarr")
        convert(tifs["raw"], out)
        assert mm.apply_marker_if_any(os.path.join(out, "raw")) == []


class TestExtraAttributes:
    def test_goes_to_the_group_that_was_written_and_keeps_ome(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "x.zarr")
        convert(tifs["raw"], out, "--extra_attributes", json.dumps({"source_path": "/a/b.tif", "publication": None}))
        convert(tifs["lab"], out, "--add-to-existing", "--is_label", "--label-key", "seg",
                "--extra_attributes", json.dumps({"label_class": "nucleus", "provenance": "public_gt"}))
        raw, lab = read(os.path.join(out, "raw", "zarr.json")), read(os.path.join(out, "labels", "seg", "zarr.json"))
        assert raw["source_path"] == "/a/b.tif" and raw["publication"] is None and "label_class" not in raw
        assert lab["label_class"] == "nucleus" and lab["provenance"] == "public_gt" and "source_path" not in lab
        assert "multiscales" in raw["ome"] and "multiscales" in lab["ome"] and raw["_software"]["name"] == "TensorSwitch"

    def test_accepts_a_json_file(self, temp_dir, tifs):
        extra = os.path.join(temp_dir, "extra.json")
        json.dump({"dataset": "Mouse skull"}, open(extra, "w"))
        out = os.path.join(temp_dir, "f.zarr")
        convert(tifs["raw"], out, "--extra_attributes", extra)
        assert read(os.path.join(out, "raw", "zarr.json"))["dataset"] == "Mouse skull"

    def test_survives_a_pyramid(self, temp_dir, tifs):
        out = os.path.join(temp_dir, "p.zarr")
        convert(tifs["big"], out, "--preset", "miaai", "--auto_multiscale", "--extra_attributes", '{"source_path": "/x.tif"}')
        assert read(os.path.join(out, "raw", "zarr.json"))["source_path"] == "/x.tif"

    @pytest.mark.parametrize("bad", ['{"ome": {}}', '{"_software": 1}', '{"tensorswitch": {}}', "[1]", "not json"])
    def test_protected_keys_and_bad_input_are_refused(self, bad):
        with pytest.raises(ValueError):
            mm.load_extra_attributes(bad)
