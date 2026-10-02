"""
Cluster submission must fail fast on a source with no usable voxel size.

The converter refuses such a source, but inside an LSF job that would only show
up after the job had waited in the queue. submit_job (CLI and MCP) therefore
checks the reader first, whether or not resources are auto-calculated.
"""

import json
import os
from argparse import Namespace
from unittest import mock

import h5py
import numpy as np
import pytest
import tifffile

from tensorswitch_v2.__main__ import _require_voxel_metadata_before_submit


def _h5(path, attrs=None):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("main", data=np.zeros((4, 8, 8), dtype=np.uint8))
        for key, value in (attrs or {}).items():
            ds.attrs[key] = value
    return path


def _args(path, **kw):
    return Namespace(input=path, dataset_path="main", voxel_size=None, **kw)


FULL = {"voxel_size_x": 0.02, "voxel_size_y": 0.02, "voxel_size_z": 0.025}


class TestRequireVoxelMetadataBeforeSubmit:
    def test_refuses_source_without_voxel_sizes(self, temp_dir):
        with pytest.raises(ValueError, match="No voxel size metadata"):
            _require_voxel_metadata_before_submit(_args(_h5(os.path.join(temp_dir, "bare.h5"))))

    def test_names_the_missing_axis_for_partial_metadata(self, temp_dir):
        path = _h5(os.path.join(temp_dir, "partial.h5"), {"voxel_size_x": 0.02, "voxel_size_y": 0.02})
        with pytest.raises(ValueError, match=r"axis z"):
            _require_voxel_metadata_before_submit(_args(path))

    def test_accepts_calibrated_source(self, temp_dir):
        _require_voxel_metadata_before_submit(_args(_h5(os.path.join(temp_dir, "ok.h5"), FULL)))

    def test_explicit_voxel_size_skips_the_check(self, temp_dir):
        args = _args(_h5(os.path.join(temp_dir, "bare.h5")))
        args.voxel_size = "20,20,25"
        _require_voxel_metadata_before_submit(args)

    def test_unreadable_source_is_left_to_the_job(self, temp_dir):
        _require_voxel_metadata_before_submit(_args(os.path.join(temp_dir, "missing.h5")))


class TestSubmitEntryPoints:
    def test_cli_submit_refuses_before_calling_bsub_even_with_explicit_resources(self, temp_dir):
        from tensorswitch_v2.__main__ import submit_job

        args = _args(
            _h5(os.path.join(temp_dir, "bare.h5")),
            project="proj", memory=16, wall_time="1:00", cores=4,
        )
        with mock.patch("subprocess.run") as run, mock.patch("subprocess.Popen") as popen:
            with pytest.raises(ValueError, match="No voxel size metadata"):
                submit_job(args)
        run.assert_not_called()
        popen.assert_not_called()

    def test_mcp_submit_reports_validation_error_before_bsub(self, temp_dir):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server as m

        src = _h5(os.path.join(temp_dir, "bare.h5"))
        # the MCP rejects /tmp paths (not visible to LSF nodes) before anything else;
        # pretend the files live on shared storage so the voxel check is reached
        shared = lambda p: "/groups/shared/" + os.path.basename(p)  # noqa: E731
        with mock.patch("subprocess.run") as run, mock.patch("subprocess.Popen") as popen, \
                mock.patch.object(m.os.path, "realpath", side_effect=shared):
            result = json.loads(
                m.submit_job(src, os.path.join(temp_dir, "out.zarr"), project="proj",
                             dataset_path="main", memory=16, wall_time="1:00", cores=4)
            )
        assert result["error"] == "validation_error"
        assert "voxel size" in result["message"].lower()
        run.assert_not_called()
        popen.assert_not_called()


def test_submit_with_all_resources_explicit_does_not_crash():
    """-M, -W and -n all given used to raise UnboundLocalError (is_native)."""
    import shutil
    import subprocess
    import tempfile

    from tensorswitch_v2.__main__ import parse_args, submit_job

    work = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".ts_submit_test_")
    try:
        src = _h5(os.path.join(work, "ok.h5"), FULL)
        args = parse_args(["-i", src, "-o", os.path.join(work, "o.zarr"), "--dataset_path", "main",
                           "--submit", "-P", "proj", "--memory", "30", "--wall_time", "1:00", "--cores", "2"])
        done = subprocess.CompletedProcess([], 0, stdout="Job <7> is submitted.", stderr="")
        with mock.patch("subprocess.run", return_value=done) as run:
            assert submit_job(args, return_job_id=True) == "7"
        cmd = [c for c in (call.args[0] for call in run.call_args_list) if c and c[0] == "bsub"][0]
        assert cmd[cmd.index("-n") + 1] == "2"
        assert cmd[cmd.index("-M") + 1] == "30GB"
    finally:
        shutil.rmtree(work, ignore_errors=True)
