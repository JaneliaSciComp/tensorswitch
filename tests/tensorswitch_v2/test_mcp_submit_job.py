"""
MCP submit_job runs the CLI's own parser, preset and submit code, so every CLI
option reaches the cluster command. bsub is mocked; nothing is queued.
"""

import grp
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("mcp")
tifffile = pytest.importorskip("tifffile")

from tensorswitch_v2 import mcp_server as m

GROUP = grp.getgrgid(os.getgid()).gr_name   # a project name that has a matching group


@pytest.fixture
def work():
    path = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".ts_submit_job_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def src(work):
    path = os.path.join(work, "in.tif")
    tifffile.imwrite(path, np.zeros((4, 8, 8), dtype=np.uint8), metadata={"axes": "ZYX"})
    return path


def submit(src, work, **kw):
    """Run submit_job with a mocked bsub; return (result, [bsub commands])."""
    commands = []
    counter = iter(range(1001, 1100))

    def fake_run(cmd, *a, **k):
        if cmd and cmd[0] == "bsub":
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=f"Job <{next(counter)}> is submitted.", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    params = dict(project=GROUP, voxel_size="1,2,3", memory=15, wall_time="0:10", cores=1)
    params.update(kw)
    with mock.patch("subprocess.run", side_effect=fake_run):
        raw = m.submit_job(src, os.path.join(work, "out.zarr"), **params)
    return json.loads(raw), commands


def reinvoke(cmd):
    """The CLI command a bsub job runs (unwraps the project-group sg wrapper)."""
    tail = cmd[-1]
    if "sg" in cmd:
        tail = tail.split("; ", 1)[1]   # drop 'export TENSORSWITCH_GROUP_APPLIED=1; '
    return shlex.split(tail)


def flag_value(argv, flag, n=1):
    i = argv.index(flag)
    return argv[i + 1:i + 1 + n]


class TestNewFlagsReachTheJob:
    def test_nd_bbox_with_axes_and_squeeze(self, src, work):
        result, cmds = submit(src, work, bbox="0,0,0,0,0,2,4,4,0,0", bbox_axes="2,3,4",
                              squeeze_singleton_axes=True)
        argv = reinvoke(cmds[0])
        assert result["status"] == "submitted"
        assert flag_value(argv, "--bbox_axes") == ["2,3,4"] and "--squeeze_singleton_axes" in argv
        assert flag_value(argv, "--bbox") == ["0,0,0,0,0,2,4,4,0,0"]

    def test_relabel_axis_several(self, src, work):
        _, cmds = submit(src, work, relabel_axis="t=z;c=y")
        argv = reinvoke(cmds[0])
        assert [argv[i + 1] for i, a in enumerate(argv) if a == "--relabel_axis"] == ["t=z", "c=y"]

    def test_sparse_label_ingest_flags(self, src, work):
        _, cmds = submit(src, work, is_label=True, data_type="labels", add_to_existing=True,
                         output_offset="0,0,64,64,64", target_shape="1,1,128,128,128")
        argv = reinvoke(cmds[0])
        assert flag_value(argv, "--output-offset", 5) == ["0", "0", "64", "64", "64"]
        assert flag_value(argv, "--target-shape", 5) == ["1", "1", "128", "128", "128"]
        assert "--add-to-existing" in argv

    def test_job_runs_under_the_project_group(self, src, work):
        _, cmds = submit(src, work)
        assert cmds[0][-4:-1] == ["sg", GROUP, "-c"]
        assert cmds[0][cmds[0].index("-P") + 1] == GROUP


class TestPresets:
    def test_mia_lmvd_preset_is_applied(self, src, work):
        _, cmds = submit(src, work, preset="mia_lmvd")
        argv = reinvoke(cmds[0])
        assert flag_value(argv, "--chunk_shape") == ["128,128,128"]
        assert flag_value(argv, "--shard_shape") == ["512,512,512"]
        assert "--force_c_order" in argv

    def test_paintera_preset(self, src, work):
        result, cmds = submit(src, work, preset="paintera")
        argv = reinvoke(cmds[0])
        assert result["format"] == "n5" and flag_value(argv, "--output_format") == ["n5"]
        assert flag_value(argv, "--axes_order") == ["xyz"]

    def test_explicit_setting_beats_preset(self, src, work):
        _, cmds = submit(src, work, preset="webknossos", chunk_shape="16,16,16")
        assert flag_value(reinvoke(cmds[0]), "--chunk_shape") == ["16,16,16"]


class TestAutoMultiscale:
    def test_conversion_then_dependent_coordinator(self, src, work):
        result, cmds = submit(src, work, auto_multiscale=True)
        assert result["mode"] == "convert_and_pyramid"
        assert result["conversion_job_id"] == "1001" and result["coordinator_job_id"] == "1002"
        assert len(cmds) == 2
        assert cmds[1][cmds[1].index("-w") + 1] == "done(1001)"
        assert "--auto_multiscale" in reinvoke(cmds[1])


class TestBehaviourKept:
    def test_default_omero_and_force_order(self, src, work):
        _, cmds = submit(src, work, omero=False, force_order="f")
        argv = reinvoke(cmds[0])
        assert "--no-omero" in argv and "--force_f_order" in argv

    def test_tmp_paths_are_still_refused(self, work):
        result = json.loads(m.submit_job("/tmp/x.tif", os.path.join(work, "o.zarr"), project=GROUP))
        assert result["error"] == "local_path"

    def test_missing_project_is_a_validation_error(self, src, work):
        result, cmds = submit(src, work, project="")
        assert result["error"] == "validation_error" and not cmds

    def test_bad_cli_value_is_a_validation_error_not_a_crash(self, src, work):
        result, cmds = submit(src, work, compression_level="not-a-number")
        assert result["error"] == "validation_error" and not cmds

    def test_missing_voxel_size_still_stops_before_queueing(self, work):
        bare = os.path.join(work, "bare.tif")
        tifffile.imwrite(bare, np.zeros((4, 8, 8), dtype=np.uint8))
        result, cmds = submit(bare, work, voxel_size="")
        assert result["error"] == "validation_error" and "voxel size" in result["message"].lower() and not cmds
