"""
HDF5 has no axis names; a first dimension of 10 or less is guessed to be a channel, so a
short z-stack comes out as axes c,y,x with no z. That must be reported, not silent.
"""

import contextlib
import io
import warnings

import h5py
import numpy as np
import pytest

from tensorswitch_v2.__main__ import _get_input_metadata, main, parse_args


def h5(path, shape):
    with h5py.File(path, "w") as f:
        f.create_dataset("vol/raw", data=np.random.default_rng(0).integers(1, 200, shape, dtype=np.uint8))
    return path


def args_for(path, *extra):
    return parse_args(["-i", path, "-o", path + ".zarr", "--dataset_path", "vol/raw", "--voxel_size", "8,8,40", *extra])


def warned(path, *extra):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _get_input_metadata(args_for(path, *extra))
    return [str(w.message) for w in caught if "channel" in str(w.message)]


def test_short_hdf5_volume_gets_a_warning_naming_the_fix(temp_dir):
    messages = warned(h5(f"{temp_dir}/short.h5", (8, 16, 16)))
    assert len(messages) == 1 and "relabel_axis='c=z'" in messages[0] and "--relabel_axis c=z" in messages[0]


def test_tall_hdf5_volume_is_fine(temp_dir):
    assert warned(h5(f"{temp_dir}/tall.h5", (12, 16, 16))) == []


def test_no_warning_once_the_axis_is_relabelled(temp_dir):
    assert warned(h5(f"{temp_dir}/short.h5", (8, 16, 16)), "--relabel_axis", "c=z") == []


def test_other_formats_are_not_judged(temp_dir):
    import tifffile

    path = f"{temp_dir}/a.tif"
    tifffile.imwrite(path, np.zeros((8, 16, 16), dtype=np.uint8), metadata={"axes": "ZYX"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _get_input_metadata(parse_args(["-i", path, "-o", path + ".zarr", "--voxel_size", "8,8,40"]))
    assert not [w for w in caught if "channel" in str(w.message)]


def test_conversion_warns_and_relabel_really_fixes_the_axes(temp_dir):
    import json

    path = h5(f"{temp_dir}/short.h5", (8, 16, 16))
    out = f"{temp_dir}/o.zarr"
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        main(["-i", path, "-o", out, "--dataset_path", "vol/raw", "--voxel_size", "8,8,40", "--quiet"])
    assert any("channel" in str(w.message) for w in caught)
    axes = [a["name"] for a in json.load(open(f"{out}/raw/zarr.json"))["attributes"]["ome"]["multiscales"][0]["axes"]]
    assert axes == ["c", "y", "x"]

    fixed = f"{temp_dir}/fixed.zarr"
    with contextlib.redirect_stdout(io.StringIO()):
        main(["-i", path, "-o", fixed, "--dataset_path", "vol/raw", "--voxel_size", "8,8,40", "--quiet",
              "--relabel_axis", "c=z"])
    ms = json.load(open(f"{fixed}/raw/zarr.json"))["attributes"]["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [40.0, 8.0, 8.0]


class TestMcpReportsWarnings:
    @pytest.fixture
    def mcp(self):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        return mcp_server

    def test_convert_response_carries_the_warning(self, mcp, temp_dir):
        import json

        path = h5(f"{temp_dir}/short.h5", (8, 16, 16))
        result = json.loads(mcp.convert(path, f"{temp_dir}/o.zarr", dataset_path="vol/raw", voxel_size="8,8,40"))
        assert result["status"] == "success"
        assert any("relabel_axis='c=z'" in w for w in result["warnings"])

    def test_relabel_axis_through_the_mcp_gives_z_and_no_warning(self, mcp, temp_dir):
        import json

        path = h5(f"{temp_dir}/short.h5", (8, 16, 16))
        out = f"{temp_dir}/o.zarr"
        result = json.loads(mcp.convert(path, out, dataset_path="vol/raw", voxel_size="8,8,40", relabel_axis="c=z"))
        assert "warnings" not in result
        axes = [a["name"] for a in json.load(open(f"{out}/raw/zarr.json"))["attributes"]["ome"]["multiscales"][0]["axes"]]
        assert axes == ["z", "y", "x"]

    def test_clean_conversion_has_no_warnings_key(self, mcp, temp_dir):
        import json

        path = h5(f"{temp_dir}/tall.h5", (12, 16, 16))
        result = json.loads(mcp.convert(path, f"{temp_dir}/o.zarr", dataset_path="vol/raw", voxel_size="8,8,40"))
        assert "warnings" not in result

    def test_stdout_stays_clean(self, mcp, temp_dir, capsys):
        path = h5(f"{temp_dir}/short.h5", (8, 16, 16))
        mcp.convert(path, f"{temp_dir}/o.zarr", dataset_path="vol/raw", voxel_size="8,8,40")
        assert capsys.readouterr().out == ""

    def test_submit_job_reports_the_warning_before_queueing(self, mcp, temp_dir):
        import json
        import os
        import shutil
        import subprocess
        import tempfile
        from unittest import mock

        work = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".ts_warn_")
        try:
            path = h5(f"{work}/short.h5", (8, 16, 16))
            ok = subprocess.CompletedProcess([], 0, stdout="Job <5> is submitted.", stderr="")
            with mock.patch("subprocess.run", return_value=ok):
                result = json.loads(mcp.submit_job(path, f"{work}/o.zarr", project="proj", dataset_path="vol/raw",
                                                   voxel_size="8,8,40", memory=15, wall_time="0:10", cores=1))
            assert result["status"] == "submitted" and any("relabel_axis" in w for w in result["warnings"])
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_no_stale_voxel_warning_after_the_user_relabelled_the_axis(temp_dir, capsys):
    """run_conversion used to warn 'Z will not be applied' even when relabel_axis already fixed it."""
    path = h5(f"{temp_dir}/short.h5", (8, 16, 16))
    base = ["-i", path, "-o", f"{temp_dir}/a.zarr", "--dataset_path", "vol/raw", "--voxel_size", "8,8,40"]
    with contextlib.redirect_stdout(io.StringIO()):
        main(base + ["--relabel_axis", "c=z"])
    assert "WARNING" not in capsys.readouterr().err
    with contextlib.redirect_stdout(io.StringIO()):
        main(["-i", path, "-o", f"{temp_dir}/b.zarr", "--dataset_path", "vol/raw", "--voxel_size", "8,8,40"])
    assert "WARNING: --voxel_size gave 3 values" in capsys.readouterr().err   # still warns when nothing was fixed
