"""
Tests for utils/record_planner.py: catalog record -> conversion plan.

Records here are small synthetic ones shaped like mia-agentic-search YAML.
"""

import copy
import json
import os

import pytest

from tensorswitch_v2.utils import record_planner as rp
from tensorswitch_v2.utils.fetch import safe_destination

ZIP = "https://zenodo.org/records/1/files/data.zip"
OUT = "/groups/miaai/miaai/out"


def record(**overrides):
    base = {
        "id": "rec-1",
        "title": "t",
        "repository": "Zenodo",
        "license": {"spdx": "CC-BY-4.0"},
        "imaging": {"dimensionality": "3D", "voxel_size_nm": {"x": 8.0, "y": 8.0, "z": 40.0}},
        "data": {"access": "open", "size_bytes": 10**8},
        "technical": {
            "sample": {"urls": [f"{ZIP}::set/images/a.tif", f"{ZIP}::set/masks/a.tif"], "size_bytes": 10**6},
            "arrays": [
                {"role": "raw", "format": "tiff", "axes": "zyx", "dtype": "uint8",
                 "path_pattern": "data.zip::set/images/*.tif"},
                {"role": "label", "format": "tiff", "axes": "zyx", "dtype": "uint8",
                 "path_pattern": "data.zip::set/masks/*.tif", "alignment": "same-grid"},
            ],
        },
    }
    for key, value in overrides.items():
        base[key] = value
    return base


def tools(plan):
    return [(s["tool"], s["args"]) for s in plan["steps"]]


class TestScopeAndBlocking:
    @pytest.mark.parametrize("dim", ["2D", "2D+t"])
    def test_out_of_scope_dimensionality_is_blocked(self, dim):
        rec = record()
        rec["imaging"]["dimensionality"] = dim
        plan = rp.plan_record(rec, OUT)
        assert plan["status"] == "blocked" and not plan["steps"]
        assert "out of scope" in plan["warnings"][0]

    def test_no_arrays_is_blocked(self):
        rec = record()
        rec["technical"]["arrays"] = []
        plan = rp.plan_record(rec, OUT)
        assert plan["status"] == "blocked" and "technical.arrays" in plan["warnings"][0]

    def test_no_sample_urls_is_blocked_and_says_why(self):
        rec = record()
        rec["technical"]["sample"] = None
        rec["data"]["download_url"] = "https://zenodo.org/records/1"
        plan = rp.plan_record(rec, OUT)
        assert plan["status"] == "blocked" and "no concrete file" in plan["warnings"][0]

    def test_3d_plus_t_warns(self):
        rec = record()
        rec["imaging"]["dimensionality"] = "3D+t"
        assert any("time axis" in w for w in rp.plan_record(rec, OUT)["warnings"])

    def test_registration_access_warns(self):
        rec = record()
        rec["data"]["access"] = "registration"
        assert any("login" in w for w in rp.plan_record(rec, OUT)["warnings"])

    def test_plan_is_json_serializable(self):
        json.dumps(rp.plan_record(record(), OUT))


class TestRawAndLabel:
    def test_fetch_then_convert_for_each_array(self):
        plan = rp.plan_record(record(), OUT)
        assert plan["status"] == "ready"
        assert [t for t, _ in tools(plan)] == ["fetch_dataset", "convert", "fetch_dataset", "convert", "verify_output"]

    def test_raw_creates_the_container_and_label_is_added_to_it(self):
        steps = tools(rp.plan_record(record(), OUT))
        raw, label = steps[1][1], steps[3][1]
        assert raw["output_path"] == label["output_path"] == f"{OUT}/rec-1.zarr"
        assert "add_to_existing" not in raw and "is_label" not in raw
        assert label["is_label"] and label["data_type"] == "labels" and label["add_to_existing"]
        assert label["label_key"] == "segmentation"

    def test_voxel_size_comes_from_the_record(self):
        assert tools(rp.plan_record(record(), OUT))[1][1]["voxel_size"] == "8,8,40"

    def test_label_listed_before_raw_still_runs_after_it(self):
        rec = record()
        rec["technical"]["arrays"].reverse()
        steps = tools(rp.plan_record(rec, OUT))
        convert = [a for t, a in steps if t == "convert"]
        assert "is_label" not in convert[0] and convert[1]["add_to_existing"]

    def test_second_label_gets_its_own_key(self):
        rec = record()
        rec["technical"]["sample"]["urls"].append(f"{ZIP}::set/masks2/b.tif")
        rec["technical"]["arrays"].append(
            {"role": "label", "format": "tiff", "path_pattern": "data.zip::set/masks2/*.tif"})
        labels = [a for t, a in tools(rp.plan_record(rec, OUT)) if t == "convert" and a.get("is_label")]
        assert [a["label_key"] for a in labels] == ["segmentation", "segmentation_2"]

    def test_label_without_raw_creates_its_own_container(self):
        rec = record()
        rec["technical"]["arrays"] = [rec["technical"]["arrays"][1]]
        convert = [a for t, a in tools(rp.plan_record(rec, OUT)) if t == "convert"][0]
        assert convert["output_path"].endswith("rec-1_labels_only.zarr") and "add_to_existing" not in convert

    def test_restoration_target_goes_to_its_own_container(self):
        rec = record()
        rec["technical"]["arrays"][1]["role"] = "target"
        outputs = {a["output_path"] for t, a in tools(rp.plan_record(rec, OUT)) if t == "convert"}
        assert len(outputs) == 2


class TestVoxelSize:
    def test_missing_voxel_size_warns_and_is_not_invented(self):
        rec = record()
        rec["imaging"]["voxel_size_nm"] = None
        plan = rp.plan_record(rec, OUT)
        assert all("voxel_size" not in a for t, a in tools(plan) if t == "convert")
        assert any("no voxel size" in w for w in plan["warnings"])

    def test_missing_z_is_called_out(self):
        rec = record()
        rec["imaging"]["voxel_size_nm"] = {"x": 8.0, "y": 8.0, "z": None}
        plan = rp.plan_record(rec, OUT)
        assert all("voxel_size" not in a for t, a in tools(plan) if t == "convert")
        assert any("no z" in w for w in plan["warnings"])

    def test_scaled_alignment_warns(self):
        rec = record()
        rec["technical"]["arrays"][1]["alignment"] = "scaled"
        notes = rp.plan_record(rec, OUT)["arrays"][1]["notes"]
        assert any("voxel size may not apply" in n for n in notes)


class TestMatching:
    def test_plain_urls_are_told_apart_by_their_folder(self):
        rec = record()
        rec["technical"]["sample"]["urls"] = [
            "https://h.org/data/images/x.tiff", "https://h.org/data/masks/x.tiff"]
        rec["technical"]["arrays"][0]["path_pattern"] = "data/images/<uid>.tiff"
        rec["technical"]["arrays"][1]["path_pattern"] = "data/masks/<uid>.tiff"
        plan = rp.plan_record(rec, OUT)
        assert plan["arrays"][0]["files"] == ["https://h.org/data/images/x.tiff"]
        assert plan["arrays"][1]["files"] == ["https://h.org/data/masks/x.tiff"]

    def test_braces_and_placeholders_in_patterns(self):
        rec = record()
        rec["technical"]["sample"]["urls"] = [f"{ZIP}::DS2/imagesTr/s01_0000.nii.gz", f"{ZIP}::DS2/labelsTs/s01.nii.gz"]
        rec["technical"]["arrays"] = [
            {"role": "raw", "format": "nifti", "path_pattern": "data.zip/DS?/images{Tr,Ts}/*_0000.nii.gz"},
            {"role": "label", "format": "nifti", "path_pattern": "data.zip/DS?/labels{Tr,Ts}/*.nii.gz"}]
        plan = rp.plan_record(rec, OUT)
        assert plan["status"] == "ready"

    def test_unmatched_sample_file_is_reported_not_silently_used(self):
        rec = record()
        rec["technical"]["sample"]["urls"].append(f"{ZIP}::set/readme.txt")
        assert any("readme.txt" in w for w in rp.plan_record(rec, OUT)["warnings"])

    def test_file_extension_must_fit_the_array_format(self):
        rec = record()
        rec["technical"]["arrays"][0]["format"] = "mrc"
        assert rp.plan_record(rec, OUT)["arrays"][0]["files"] == []


class TestHdf5:
    def hdf5(self, pattern_raw, pattern_label):
        rec = record()
        url = "https://zenodo.org/records/3/files/nuclei.zip::nuclei/a.h5"
        rec["technical"]["sample"]["urls"] = [url]
        rec["technical"]["arrays"] = [
            {"role": "raw", "format": "hdf5", "path_pattern": pattern_raw},
            {"role": "label", "format": "hdf5", "path_pattern": pattern_label}]
        return rp.plan_record(rec, OUT)

    def test_dataset_names_come_from_the_pattern_and_the_file_is_fetched_once(self):
        plan = self.hdf5("nuclei.zip::nuclei/*.h5 (volumes/raw)", "nuclei.zip::nuclei/*.h5 (volumes/labels/seg)")
        assert [t for t, _ in tools(plan)] == ["fetch_dataset", "convert", "convert", "verify_output"]
        converts = [a for t, a in tools(plan) if t == "convert"]
        assert [c["dataset_path"] for c in converts] == ["volumes/raw", "volumes/labels/seg"]
        assert not any("dataset_path=null" in w for w in plan["warnings"])

    def test_double_colon_form(self):
        plan = self.hdf5("nuclei.zip::nuclei/*.h5 :: volumes/raw", "nuclei.zip::nuclei/*.h5 :: volumes/lab")
        assert [a["dataset_path"] for t, a in tools(plan) if t == "convert"] == ["volumes/raw", "volumes/lab"]

    def test_unknown_dataset_name_warns(self):
        plan = self.hdf5("nuclei.zip::nuclei/*.h5", "nuclei.zip::nuclei/*.h5")
        assert any("dataset_path=null" in w for w in plan["warnings"])


class TestUnconvertible:
    def one(self, fmt, role="raw", pattern="data.zip::set/images/*.tif", url=f"{ZIP}::set/images/a.tif"):
        rec = record()
        rec["technical"]["sample"]["urls"] = [url]
        rec["technical"]["arrays"] = [{"role": role, "format": fmt, "path_pattern": pattern}]
        return rp.plan_record(rec, OUT)

    def test_tables_are_not_converted(self):
        plan = self.one("csv", role="label", pattern="data.zip::set/*.csv", url=f"{ZIP}::set/t.csv")
        assert plan["status"] == "blocked" and "table" in plan["arrays"][0]["notes"][0]

    def test_png_stacks_need_the_whole_folder(self):
        plan = self.one("png", pattern="data.zip::set/*.png", url=f"{ZIP}::set/s0.png")
        assert "whole folder" in plan["arrays"][0]["notes"][0]

    def test_unsupported_format(self):
        plan = self.one("dicom", pattern="data.zip::set/*.dcm", url=f"{ZIP}::set/a.dcm")
        assert "not supported" in plan["arrays"][0]["notes"][0]

    def test_zarr_inside_a_zip_cannot_be_fetched(self):
        plan = self.one("zarr", pattern="data.zip::set/*.zarr", url=f"{ZIP}::set/a.zarr")
        assert "folders" in plan["arrays"][0]["notes"][0]

    def test_remote_zarr_is_read_in_place_without_fetching(self):
        url = "https://bucket.s3.amazonaws.com/vol.zarr"
        plan = self.one("zarr", pattern="vol.zarr", url=url)
        assert [t for t, _ in tools(plan)] == ["convert", "verify_output"]
        assert tools(plan)[0][1]["input_path"] == url

    def test_one_unconvertible_array_makes_the_plan_partial(self):
        rec = record()
        rec["technical"]["sample"]["urls"].append(f"{ZIP}::set/t.csv")
        rec["technical"]["arrays"].append({"role": "label", "format": "csv", "path_pattern": "data.zip::set/*.csv"})
        plan = rp.plan_record(rec, OUT)
        assert plan["status"] == "partial" and len(tools(plan)) == 5


class TestVerifyStep:
    def verify_steps(self, plan):
        return [a for t, a in tools(plan) if t == "verify_output"]

    def test_one_verify_step_per_container_comes_last(self):
        plan = rp.plan_record(record(), OUT)
        assert tools(plan)[-1][0] == "verify_output" and len(self.verify_steps(plan)) == 1

    def test_verify_step_carries_what_was_asked_for(self):
        args = self.verify_steps(rp.plan_record(record(), OUT))[0]
        assert args["output_path"] == f"{OUT}/rec-1.zarr" and args["image_key"] == "raw"
        assert args["source_path"] == f"{OUT}/source/set/images/a.tif"
        assert args["voxel_size"] == "8,8,40"
        assert args["labels"] == f"segmentation={OUT}/source/set/masks/a.tif"

    def test_no_voxel_size_in_the_record_means_none_to_compare(self):
        rec = record()
        rec["imaging"]["voxel_size_nm"] = None
        assert "voxel_size" not in self.verify_steps(rp.plan_record(rec, OUT))[0]

    def test_hdf5_labels_carry_their_dataset(self):
        rec = record()
        rec["technical"]["sample"]["urls"] = ["https://zenodo.org/records/3/files/n.zip::nuclei/a.h5"]
        rec["technical"]["arrays"] = [
            {"role": "raw", "format": "hdf5", "path_pattern": "n.zip::nuclei/*.h5 (volumes/raw)"},
            {"role": "label", "format": "hdf5", "path_pattern": "n.zip::nuclei/*.h5 (volumes/labels/seg)"}]
        args = self.verify_steps(rp.plan_record(rec, OUT))[0]
        assert args["dataset_path"] == "volumes/raw"
        assert args["labels"] == f"segmentation={OUT}/source/nuclei/a.h5::volumes/labels/seg"

    def test_labels_only_container_expects_no_image(self):
        rec = record()
        rec["technical"]["arrays"] = [rec["technical"]["arrays"][1]]
        args = self.verify_steps(rp.plan_record(rec, OUT))[0]
        assert args["image_key"] == "" and args["source_path"] == ""

    def test_separate_containers_get_separate_verify_steps(self):
        rec = record()
        rec["technical"]["arrays"][1]["role"] = "target"
        assert len(self.verify_steps(rp.plan_record(rec, OUT))) == 2

    def test_verify_arguments_are_real_tool_parameters(self):
        import inspect

        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        params = inspect.signature(mcp_server.verify_output).parameters
        assert set(self.verify_steps(rp.plan_record(record(), OUT))[0]) <= set(params)


class TestSizeAndPaths:
    def test_large_sample_uses_submit_job_with_the_project(self):
        rec = record()
        rec["technical"]["sample"]["size_bytes"] = 5 * 1024 ** 3
        converts = [(t, a) for t, a in tools(rp.plan_record(rec, OUT, project="miaai"))
                    if t not in ("fetch_dataset", "verify_output")]
        assert all(t == "submit_job" and a["project"] == "miaai" for t, a in converts)

    def test_large_sample_without_project_says_so(self):
        rec = record()
        rec["technical"]["sample"]["size_bytes"] = 5 * 1024 ** 3
        plan = rp.plan_record(rec, OUT)
        assert any("LSF project" in w for w in plan["warnings"])

    def test_huge_dataset_is_flagged(self):
        rec = record()
        rec["data"]["size_bytes"] = 70 * 1024 ** 3
        assert any("over 50 GB" in w for w in rp.plan_record(rec, OUT)["warnings"])

    def test_staging_is_inside_the_output_folder(self):
        for tool, args in tools(rp.plan_record(record(), OUT)):
            if tool == "fetch_dataset":
                assert args["dest_dir"] == f"{OUT}/source"

    def test_planned_input_path_is_where_fetch_will_write(self, temp_dir):
        plan = rp.plan_record(record(), temp_dir)
        fetch = [a for t, a in tools(plan) if t == "fetch_dataset"][0]
        convert = [a for t, a in tools(plan) if t == "convert"][0]
        member = fetch["spec"].split("::", 1)[1]
        assert convert["input_path"] == os.path.join(fetch["dest_dir"], *member.split("/"))
        assert safe_destination(fetch["dest_dir"], member) == os.path.realpath(convert["input_path"])

    def test_plain_url_path_matches_fetch(self, temp_dir):
        rec = record()
        rec["technical"]["sample"]["urls"] = ["https://h.org/a/my%20vol.tif"]
        rec["technical"]["arrays"] = [{"role": "raw", "format": "tiff", "path_pattern": "a/*.tif"}]
        plan = rp.plan_record(rec, temp_dir)
        convert = [a for t, a in tools(plan) if t == "convert"][0]
        assert convert["input_path"] == os.path.join(temp_dir, "source", "my vol.tif")


class TestInputsAreNotMutated:
    def test_record_is_left_unchanged(self):
        rec = record()
        before = copy.deepcopy(rec)
        rp.plan_record(rec, OUT)
        assert rec == before


class TestLoadRecord:
    def test_local_file(self, temp_dir):
        import yaml

        path = os.path.join(temp_dir, "r.yaml")
        yaml.safe_dump(record(), open(path, "w"))
        assert rp.load_record(path)["id"] == "rec-1"

    def test_not_a_record(self, temp_dir):
        path = os.path.join(temp_dir, "x.yaml")
        open(path, "w").write("- just\n- a list\n")
        with pytest.raises(rp.RecordError, match="not a catalog record"):
            rp.load_record(path)

    def test_invalid_yaml(self, temp_dir):
        path = os.path.join(temp_dir, "x.yaml")
        open(path, "w").write("a: [unclosed\n")
        with pytest.raises(rp.RecordError, match="valid YAML"):
            rp.load_record(path)

    def test_neither_file_url_nor_id(self):
        with pytest.raises(rp.RecordError, match="not a file"):
            rp.load_record("some weird/thing here")

    def test_github_blob_url_becomes_raw(self):
        url = "https://github.com/AI-HHMI/mia-agentic-search/blob/main/datasets/3D/FIB-SEM/x.yaml"
        assert rp._github_raw(url) == \
            "https://raw.githubusercontent.com/AI-HHMI/mia-agentic-search/main/datasets/3D/FIB-SEM/x.yaml"


class TestMcpTool:
    @pytest.fixture
    def tool(self):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        return lambda *a, **k: json.loads(mcp_server.plan_conversion_from_yaml(*a, **k))

    def test_returns_the_plan_for_a_local_record(self, tool, temp_dir):
        import yaml

        path = os.path.join(temp_dir, "r.yaml")
        yaml.safe_dump(record(), open(path, "w"))
        plan = tool(path, OUT)
        assert plan["status"] == "ready" and plan["record"]["id"] == "rec-1"
        assert [s["tool"] for s in plan["steps"]] == ["fetch_dataset", "convert", "fetch_dataset", "convert", "verify_output"]

    def test_bad_record_is_an_error_not_an_exception(self, tool):
        result = tool("definitely not a record!", OUT)
        assert result["error"] == "bad_record"

    def test_steps_are_runnable_with_the_real_tool_signatures(self, tool, temp_dir):
        import inspect

        import yaml
        from tensorswitch_v2 import mcp_server

        path = os.path.join(temp_dir, "r.yaml")
        rec = record()
        rec["technical"]["sample"]["size_bytes"] = 5 * 1024 ** 3
        yaml.safe_dump(rec, open(path, "w"))
        for step in tool(path, OUT, project="miaai")["steps"]:
            params = inspect.signature(getattr(mcp_server, step["tool"])).parameters
            assert set(step["args"]) <= set(params), (step["tool"], set(step["args"]) - set(params))


class TestShortHdf5:
    def plan(self, shape):
        rec = record()
        rec["technical"]["sample"]["urls"] = ["https://zenodo.org/records/3/files/n.zip::nuclei/a.h5"]
        rec["technical"]["arrays"] = [
            {"role": "raw", "format": "hdf5", "shape": shape, "path_pattern": "n.zip::nuclei/*.h5 (volumes/raw)"},
            {"role": "label", "format": "hdf5", "shape": shape, "path_pattern": "n.zip::nuclei/*.h5 (volumes/lab)"}]
        return rp.plan_record(rec, OUT)

    def test_short_first_axis_gets_relabel_axis_and_a_note(self):
        plan = self.plan([8, 100, 100])
        converts = [a for t, a in tools(plan) if t == "convert"]
        assert all(c["relabel_axis"] == "c=z" for c in converts)
        assert any("relabel_axis='c=z'" in n for a in plan["arrays"] for n in a["notes"])

    @pytest.mark.parametrize("shape", [[11, 100, 100], [104, 350, 350], None, [8, 100], [3, 4, 100, 100]])
    def test_other_shapes_are_left_alone(self, shape):
        assert all("relabel_axis" not in a for t, a in tools(self.plan(shape)) if t == "convert")

    def test_boundary_is_ten(self):
        assert all(a.get("relabel_axis") == "c=z" for t, a in tools(self.plan([10, 9, 9])) if t == "convert")

    def test_other_formats_never_get_it(self):
        rec = record()
        rec["technical"]["arrays"][0]["shape"] = [4, 50, 50]
        assert all("relabel_axis" not in a for t, a in tools(rp.plan_record(rec, OUT)) if t == "convert")

    def test_relabel_axis_is_a_real_convert_parameter(self):
        import inspect

        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        assert "relabel_axis" in inspect.signature(mcp_server.convert).parameters
