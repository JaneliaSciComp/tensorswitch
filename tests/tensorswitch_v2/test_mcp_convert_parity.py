"""
MCP convert and the CLI run the same code (parse_args -> _apply_preset -> run_conversion),
so the same request must give byte-identical outputs. Each case runs through both and the
resulting stores are compared file by file.
"""

import contextlib
import hashlib
import io
import json
import os

import h5py
import numpy as np
import pytest

pytest.importorskip("mcp")
tifffile = pytest.importorskip("tifffile")

from tensorswitch_v2 import mcp_server as m
from tensorswitch_v2.__main__ import main as cli_main

VOX = "8,8,40"
JSON_NAMES = ("zarr.json", ".zattrs", ".zarray", ".zgroup", "attributes.json")


def fingerprint(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            path = os.path.join(d, f)
            data = open(path, "rb").read().replace(root.encode(), b"<ROOT>")
            if f in JSON_NAMES:
                data = json.dumps(json.loads(data), sort_keys=True).encode()
            out[os.path.relpath(path, root)] = hashlib.sha1(data).hexdigest()
    return out


@pytest.fixture
def sources(temp_dir):
    rng = np.random.default_rng(3)
    src = {}
    for name, shape, high in (("raw", (8, 16, 16), 200), ("lab", (8, 16, 16), 5), ("small", (4, 8, 8), 5),
                              ("big", (64, 512, 512), 200)):
        path = os.path.join(temp_dir, f"{name}.tif")
        tifffile.imwrite(path, rng.integers(1, high, shape, dtype=np.uint8), metadata={"axes": "ZYX"})
        src[name] = path
    h5 = os.path.join(temp_dir, "v.h5")
    with h5py.File(h5, "w") as f:
        d = f.create_dataset("vol/raw", data=rng.integers(0, 200, (8, 16, 16), dtype=np.uint8))
        d.attrs["voxel_size_x"] = d.attrs["voxel_size_y"] = d.attrs["voxel_size_z"] = 0.01
    src["h5"] = h5
    return src


CASES = {
    "zarr3": [("raw", [], {})],
    "zarr2": [("raw", ["--output_format", "zarr2"], {"output_format": "zarr2"})],
    "n5": [("raw", ["--output_format", "n5"], {"output_format": "n5"})],
    "webknossos": [("raw", ["--preset", "webknossos"], {"preset": "webknossos"})],
    "mia_lmvd": [("raw", ["--preset", "mia_lmvd"], {"preset": "mia_lmvd"})],
    "miaai": [("raw", ["--preset", "miaai"], {"preset": "miaai"})],
    "is_label": [("lab", ["--is_label"], {"is_label": True})],
    "bbox": [("raw", ["--bbox", "0,0,0,4,8,8"], {"bbox": "0,0,0,4,8,8"})],
    "bbox_nd_axes": [("raw", ["--bbox", "0,0,2,4,4,4", "--bbox_axes", "0,1,2"],
                      {"bbox": "0,0,2,4,4,4", "bbox_axes": "0,1,2"})],
    "multiscale": [("big", ["--auto_multiscale"], {"auto_multiscale": True})],
    "add_label": [("raw", [], {}),
                  ("lab", ["--add-to-existing", "--is_label", "--label-key", "seg"],
                   {"add_to_existing": True, "is_label": True, "label_key": "seg"})],
    "replace_image": [("raw", [], {}),
                      ("lab", ["--add-to-existing", "--data-type", "image"],
                       {"add_to_existing": True, "data_type": "image"})],
    "sparse_label": [("raw", [], {}),
                     ("small", ["--add-to-existing", "--is_label", "--label-key", "seg", "--output-offset", "2", "4", "4",
                                "--target-shape", "8", "16", "16"],
                      {"add_to_existing": True, "is_label": True, "label_key": "seg", "output_offset": "2,4,4",
                       "target_shape": "8,16,16"})],
    "label_pyramid": [("big", [], {}),
                      ("big", ["--add-to-existing", "--is_label", "--label-key", "seg", "--auto_multiscale"],
                       {"add_to_existing": True, "is_label": True, "label_key": "seg", "auto_multiscale": True})],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_mcp_convert_matches_the_cli(temp_dir, sources, case):
    out = {"cli": os.path.join(temp_dir, "cli.zarr"), "mcp": os.path.join(temp_dir, "mcp.zarr")}
    if case == "n5":
        out = {k: v.replace(".zarr", ".n5") for k, v in out.items()}
    for src_key, cli_args, mcp_kwargs in CASES[case]:
        with contextlib.redirect_stdout(io.StringIO()):
            cli_main(["-i", sources[src_key], "-o", out["cli"], "--voxel_size", VOX, "--quiet", *cli_args])
        result = json.loads(m.convert(sources[src_key], out["mcp"], voxel_size=VOX, **mcp_kwargs))
        assert result.get("status") == "success", result
    assert fingerprint(out["mcp"]) == fingerprint(out["cli"])


def test_hdf5_matches_the_cli(temp_dir, sources):
    cli_out, mcp_out = os.path.join(temp_dir, "c.zarr"), os.path.join(temp_dir, "m.zarr")
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", sources["h5"], "-o", cli_out, "--dataset_path", "vol/raw", "--quiet"])
    assert json.loads(m.convert(sources["h5"], mcp_out, dataset_path="vol/raw"))["status"] == "success"
    assert fingerprint(mcp_out) == fingerprint(cli_out)


class TestResponse:
    def test_reports_chunks_time_and_size(self, temp_dir, sources):
        r = json.loads(m.convert(sources["raw"], os.path.join(temp_dir, "o.zarr"), voxel_size=VOX))
        assert r["format"] == "zarr3" and r["chunks_processed"] >= 1
        assert isinstance(r["time_seconds"], float) and r["dataset_size_gb"] == 0.0

    def test_pyramid_details(self, temp_dir, sources):
        r = json.loads(m.convert(sources["big"], os.path.join(temp_dir, "o.zarr"), voxel_size=VOX, auto_multiscale=True))
        assert r["auto_multiscale"] and r["pyramid_levels"] >= 1 and r["pyramid_info"][0]["level"] == "s1"

    def test_output_path_is_the_final_path_not_the_tmp_path(self, temp_dir, sources):
        out = os.path.join(temp_dir, "o.zarr")
        assert json.loads(m.convert(sources["raw"], out, voxel_size=VOX))["output"] == out


class TestGuardsAndErrors:
    def test_nd_bbox_size_guard_uses_the_bbox_volume(self, temp_dir, sources, monkeypatch):
        monkeypatch.setattr(m, "MCP_CONVERT_MAX_GB", 0.0000001)   # ~100 bytes
        tiny = json.loads(m.convert(sources["raw"], os.path.join(temp_dir, "a.zarr"), voxel_size=VOX,
                                    bbox="0,0,0,2,2,2"))
        assert tiny["status"] == "success"            # 8 voxels fits
        whole = json.loads(m.convert(sources["raw"], os.path.join(temp_dir, "b.zarr"), voxel_size=VOX))
        assert whole["error"] == "dataset_too_large"

    def test_bad_value_is_a_validation_error(self, temp_dir, sources):
        r = json.loads(m.convert(sources["raw"], os.path.join(temp_dir, "o.zarr"), voxel_size=VOX,
                                 compression_level="abc"))
        assert r["error"] == "validation_error"

    def test_missing_voxel_size_is_refused_with_the_usual_message(self, temp_dir, sources):
        assert "No voxel size metadata" in m.convert(sources["raw"], os.path.join(temp_dir, "o.zarr"))

    def test_add_to_existing_without_container(self, temp_dir, sources):
        r = json.loads(m.convert(sources["lab"], os.path.join(temp_dir, "none.zarr"), voxel_size=VOX,
                                 is_label=True, data_type="labels", add_to_existing=True))
        assert "does not exist" in r["error"]

    def test_stdout_stays_clean(self, temp_dir, sources, capsys):
        m.convert(sources["raw"], os.path.join(temp_dir, "o.zarr"), voxel_size=VOX, auto_multiscale=True)
        assert capsys.readouterr().out == ""
