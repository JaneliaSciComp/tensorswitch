"""Whole-dataset planning: files come from an injected zip listing, no network."""
from types import SimpleNamespace

import pytest

from tensorswitch_v2.utils import record_planner as rp

ZIP = "https://example.org/files/data.zip"
OUT = "/groups/miaai/miaai/out"


def entries(*names, size=1000):
    return [SimpleNamespace(name=n, size=size) for n in names]


def lister(*names, size=1000):
    return lambda url: entries(*names, size=size)


def record(**kw):
    rec = {
        "id": "rec", "title": "t", "repository": "Zenodo",
        "imaging": {"dimensionality": "3D", "voxel_size_nm": {"x": 8.0, "y": 8.0, "z": 40.0}},
        "data": {"access": "open", "size_bytes": 10**6, "download_url": ZIP},
        "technical": {
            "sample": {"urls": [f"{ZIP}::set/test/images/X2.tif", f"{ZIP}::set/test/masks/Y2.tif"], "size_bytes": 10},
            "arrays": [
                {"role": "raw", "format": "tiff", "axes": "zyx", "shape": [5, 6, 7], "shape_varies": True,
                 "path_pattern": "set/{train,test}/images/X*.tif"},
                {"role": "label", "format": "tiff", "axes": "zyx", "alignment": "same-grid",
                 "path_pattern": "set/{train,test}/masks/Y*.tif"},
            ],
        },
    }
    rec.update(kw)
    return rec


FILES = ["set/train/images/X1.tif", "set/train/masks/Y1.tif", "set/test/images/X2.tif", "set/test/masks/Y2.tif",
         "set/readme.txt"]


def converts(plan):
    return [s["args"] for s in plan["steps"] if s["tool"] == "convert"]


class TestPairing:
    def test_pairs_by_folder_and_name(self):
        plan = rp.plan_dataset(record(), OUT, lister=lister(*FILES))
        assert plan["status"] == "ready" and plan["unpaired"] == []
        assert sorted(s["name"] for s in plan["samples"]) == ["test_X2", "train_X1"]
        sample = {s["name"]: s["files"] for s in plan["samples"]}
        assert sample["train_X1"] == ["set/train/images/X1.tif", "set/train/masks/Y1.tif"]

    def test_one_container_per_sample_with_raw_then_label(self):
        plan = rp.plan_dataset(record(), OUT, lister=lister(*FILES))
        args = converts(plan)
        assert len(args) == 4 and {a["output_path"] for a in args} == {f"{OUT}/rec_train_X1.zarr", f"{OUT}/rec_test_X2.zarr"}
        first = [a for a in args if a["output_path"].endswith("train_X1.zarr")]
        assert "is_label" not in first[0] and first[1]["is_label"] and first[1]["add_to_existing"]
        assert [s["tool"] for s in plan["steps"]].count("verify_output") == 2

    def test_voxel_size_comes_from_the_record_for_every_sample(self):
        assert {a["voxel_size"] for a in converts(rp.plan_dataset(record(), OUT, lister=lister(*FILES)))} == {"8,8,40"}

    def test_file_without_a_partner_is_reported_and_still_converted(self):
        plan = rp.plan_dataset(record(), OUT, lister=lister(*FILES, "set/test/images/X3.tif"))
        assert any("X3.tif" in u and "no label partner" in u for u in plan["unpaired"])
        assert any(s["name"] == "test_X3" for s in plan["samples"]) and plan["status"] == "partial"

    def test_role_words_and_leading_letter_are_ignored_but_other_words_must_match(self):
        names = ["a/image/train-input.tif", "a/seg/train-labels.tif", "a/image/test-input.tif"]
        rec = record()
        rec["technical"]["arrays"][0]["path_pattern"] = "a/image/*-input.tif"
        rec["technical"]["arrays"][1]["path_pattern"] = "a/seg/*-labels.tif"
        plan = rp.plan_dataset(rec, OUT, lister=lister(*names))
        assert [s["files"] for s in plan["samples"] if len(s["files"]) == 2] == [["a/image/train-input.tif", "a/seg/train-labels.tif"]]
        assert any("test-input.tif" in u for u in plan["unpaired"])

    def test_two_files_with_the_same_key_are_not_guessed(self):
        rec = record()
        rec["technical"]["arrays"][0]["path_pattern"] = "set/*X*.tif"
        plan = rp.plan_dataset(rec, OUT, lister=lister(*FILES, "set/train/X1.tif"))
        assert any("same sample key" in u for u in plan["unpaired"])


class TestPairingNeedsAlignment:
    def test_label_without_same_grid_is_not_paired(self):
        rec = record()
        rec["technical"]["arrays"][1]["alignment"] = "unknown"
        plan = rp.plan_dataset(rec, OUT, lister=lister(*FILES))
        assert not any(len(s["files"]) == 2 for s in plan["samples"])
        assert any("not 'same-grid'" in u for u in plan["unpaired"])

    def test_same_grid_label_is_paired(self):
        rec = record()
        rec["technical"]["arrays"][1]["alignment"] = "same-grid"
        plan = rp.plan_dataset(rec, OUT, lister=lister(*FILES))
        assert sorted(len(s["files"]) for s in plan["samples"]) == [2, 2]

    def test_data_and_mask_files_pair(self):
        rec = record()
        rec["technical"]["arrays"][0]["path_pattern"] = "set/{train,test}/data_NNN.tif"
        rec["technical"]["arrays"][1].update(path_pattern="set/{train,test}/mask_NNN.tif", alignment="same-grid")
        names = ["set/train/data_000.tif", "set/train/mask_000.tif", "set/train/data_001.tif", "set/train/mask_001.tif"]
        plan = rp.plan_dataset(rec, OUT, lister=lister(*names))
        assert plan["unpaired"] == [] and sorted(len(s["files"]) for s in plan["samples"]) == [2, 2]


class TestLimitsAndBlocking:
    def test_over_50_gb_is_skipped_and_listed(self):
        plan = rp.plan_dataset(record(), OUT, lister=lister(*FILES, size=20 * 1024 ** 3))
        assert plan["status"] == "skipped" and plan["steps"] == [] and plan["skipped"][0]["record"] == "rec"

    def test_no_zip_to_list_is_blocked_with_a_reason(self):
        rec = record()
        rec["data"]["download_url"] = "https://example.org/landing"
        rec["technical"]["sample"]["urls"] = ["https://example.org/a.tif"]
        plan = rp.plan_dataset(rec, OUT, lister=lister())
        assert plan["status"] == "blocked" and "zip" in plan["warnings"][0]

    def test_listing_failure_is_reported(self):
        def broken(url):
            raise RuntimeError("HTTP 404")
        plan = rp.plan_dataset(record(), OUT, lister=broken)
        assert plan["status"] == "blocked" and "HTTP 404" in plan["warnings"][0]

    def test_no_matching_files_is_blocked(self):
        plan = rp.plan_dataset(record(), OUT, lister=lister("other/file.tif"))
        assert plan["status"] == "blocked"

    def test_no_voxel_size_still_stops_untrusted_formats(self):
        rec = record()
        rec["imaging"]["voxel_size_nm"] = None
        for a in rec["technical"]["arrays"]:
            a["format"] = "nifti"
            a["path_pattern"] = a["path_pattern"].replace(".tif", ".nii.gz")
        names = [n.replace(".tif", ".nii.gz") for n in FILES]
        plan = rp.plan_dataset(rec, OUT, lister=lister(*names))
        assert plan["steps"] == [] and plan["status"] == "blocked"
