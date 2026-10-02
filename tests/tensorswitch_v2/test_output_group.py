"""
Output group handling: cluster jobs run under the project's group, local outputs
take the group of their parent folder.
"""

import grp
import os
import shlex
import subprocess
import warnings

import pytest

from tensorswitch_v2.utils import output_group as og


def _other_group():
    """A group the current user belongs to that is not the primary one, if any."""
    for gid in os.getgroups():
        if gid != os.getgid():
            return gid, grp.getgrgid(gid).gr_name
    return None, None


class TestProjectGroup:
    def test_member_group_is_found(self):
        name = grp.getgrgid(os.getgid()).gr_name
        assert og.project_group(name) == name

    def test_unknown_group_is_none(self):
        assert og.project_group("no_such_group_xyz") is None

    def test_empty_project_is_none(self):
        assert og.project_group(None) is None and og.project_group("") is None

    def test_group_the_user_is_not_in_is_refused(self):
        # sg would ask for a password for these, which hangs a batch job
        outsider = next((g.gr_name for g in grp.getgrall() if g.gr_gid not in og._member_gids()), None)
        if outsider is None:
            pytest.skip("user belongs to every group")
        assert og.project_group(outsider) is None


class TestWrap:
    def test_wraps_in_sg_with_marker_env(self):
        name = grp.getgrgid(os.getgid()).gr_name
        wrapped = og.wrap_for_project(name, ["/bin/bash", "-c", "echo 'hi there'"])
        assert wrapped[:3] == ["sg", name, "-c"]
        assert shlex.split(wrapped[3]) == ["env", "TENSORSWITCH_GROUP_APPLIED=1",
                                           "/bin/bash", "-c", "echo 'hi there'"]

    def test_unchanged_with_warning_when_no_group(self):
        og._warned.discard("nope_project")
        argv = ["/bin/bash", "-c", "x"]
        with pytest.warns(UserWarning, match="nope_project"):
            assert og.wrap_for_project("nope_project", argv) == argv

    def test_warns_only_once_per_project(self):
        og._warned.discard("once_project")
        with pytest.warns(UserWarning):
            og.wrap_for_project("once_project", ["a"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            og.wrap_for_project("once_project", ["a"])

    def test_no_project_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert og.wrap_for_project(None, ["a"]) == ["a"]

    def test_wrapped_command_really_runs_and_keeps_quoting(self, temp_dir):
        name = grp.getgrgid(os.getgid()).gr_name
        out = os.path.join(temp_dir, "dir with space", "o.txt")
        os.makedirs(os.path.dirname(out))
        argv = ["/bin/bash", "-c", shlex.join(["touch", out])]
        subprocess.run(og.wrap_for_project(name, argv), check=True)
        assert os.path.exists(out)

    def test_script_line_form(self):
        name = grp.getgrgid(os.getgid()).gr_name
        line = og.wrap_script_line(name, ["/bin/bash", "run.sh"])
        assert line.startswith(f"sg {name} -c ")


class TestApplyParentGroup:
    def test_noop_when_group_already_matches(self, temp_dir):
        out = os.path.join(temp_dir, "o.zarr")
        os.makedirs(out)
        assert og.apply_parent_group(out) == 0

    def test_changes_group_to_the_parents(self, temp_dir):
        gid, _ = _other_group()
        if gid is None:
            pytest.skip("user has no secondary group")
        parent = os.path.join(temp_dir, "p")
        os.makedirs(parent)
        os.chown(parent, -1, gid)
        out = os.path.join(parent, "o.zarr")
        os.makedirs(os.path.join(out, "raw", "s0"))
        open(os.path.join(out, "raw", "s0", "c0"), "w").write("x")
        assert os.stat(out).st_gid != gid  # created with the primary group (parent is not setgid)
        changed = og.apply_parent_group(out)
        assert changed >= 4
        for root, dirs, files in os.walk(out):
            for name in dirs + files:
                assert os.stat(os.path.join(root, name)).st_gid == gid
        assert os.stat(out).st_gid == gid

    def test_not_applied_inside_a_job_already_under_the_project_group(self, temp_dir, monkeypatch):
        gid, _ = _other_group()
        if gid is None:
            pytest.skip("user has no secondary group")
        parent = os.path.join(temp_dir, "p")
        os.makedirs(parent)
        os.chown(parent, -1, gid)
        out = os.path.join(parent, "o.zarr")
        os.makedirs(out)
        monkeypatch.setenv(og.APPLIED_ENV, "1")
        assert og.apply_parent_group(out) == 0
        assert os.stat(out).st_gid != gid

    def test_missing_output_and_missing_parent_are_ignored(self, temp_dir):
        assert og.apply_parent_group(os.path.join(temp_dir, "nope.zarr")) == 0

    def test_parent_group_user_is_not_in_is_left_alone(self, temp_dir, monkeypatch):
        out = os.path.join(temp_dir, "o.zarr")
        os.makedirs(out)
        monkeypatch.setattr(og, "_member_gids", lambda: set())
        assert og.apply_parent_group(out) == 0


class TestSubmitUsesProjectGroup:
    """The real submit path builds a bsub command; check what it would run."""

    @pytest.fixture
    def shared_dir(self):
        import shutil
        import tempfile

        # the MCP refuses /tmp paths (invisible to LSF nodes), so work under $HOME
        path = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".ts_group_test_")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def _submit(self, shared_dir, project):
        import json
        from unittest import mock

        import numpy as np
        import tifffile

        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server as m

        src = os.path.join(shared_dir, "a.tif")
        tifffile.imwrite(src, np.zeros((4, 8, 8), dtype=np.uint8))
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="Job <4242> is submitted to queue <x>.", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            result = json.loads(m.submit_job(src, os.path.join(shared_dir, "o.zarr"), project=project,
                                             voxel_size="1,2,3", memory=15, wall_time="0:10", cores=1))
        assert result["status"] == "submitted", result
        return [c for c in calls if c and c[0] == "bsub"][0]

    def test_job_runs_under_the_projects_group(self, shared_dir):
        name = grp.getgrgid(os.getgid()).gr_name
        cmd = self._submit(shared_dir, name)
        assert cmd[cmd.index("-P") + 1] == name
        assert cmd[-4:-1] == ["sg", name, "-c"]
        inner = shlex.split(cmd[-1])
        assert inner[:3] == ["env", "TENSORSWITCH_GROUP_APPLIED=1", "/bin/bash"]

    def test_project_without_a_group_is_submitted_unchanged(self, shared_dir):
        og._warned.discard("proj_without_group")
        with pytest.warns(UserWarning, match="proj_without_group"):
            cmd = self._submit(shared_dir, "proj_without_group")
        assert cmd[-3:-1] == ["/bin/bash", "-c"]
        assert "sg" not in cmd


class TestLocalConversionTakesParentGroup:
    """A local conversion into a folder with another group ends up in that group."""

    def _tiff(self, folder):
        import numpy as np
        import tifffile

        path = os.path.join(folder, "a.tif")
        tifffile.imwrite(path, np.zeros((4, 8, 8), dtype=np.uint8), metadata={"axes": "ZYX"})
        return path

    def test_mcp_convert_output_is_regrouped(self, temp_dir):
        import json

        gid, _ = _other_group()
        if gid is None:
            pytest.skip("user has no secondary group")
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server as m

        outdir = os.path.join(temp_dir, "dest")
        os.makedirs(outdir)
        os.chown(outdir, -1, gid)
        out = os.path.join(outdir, "o.zarr")
        assert json.loads(m.convert(self._tiff(temp_dir), out, voxel_size="1,2,3"))["status"] == "success"
        for root, dirs, files in os.walk(out):
            for name in dirs + files:
                assert os.stat(os.path.join(root, name)).st_gid == gid, os.path.join(root, name)

    def test_cli_finalize_regroups_the_renamed_output(self, temp_dir):
        gid, _ = _other_group()
        if gid is None:
            pytest.skip("user has no secondary group")
        from tensorswitch_v2.__main__ import _finalize_tmp_path

        outdir = os.path.join(temp_dir, "dest")
        os.makedirs(outdir)
        os.chown(outdir, -1, gid)
        tmp = os.path.join(outdir, "o.zarr.tmp")
        os.makedirs(os.path.join(tmp, "raw"))
        _finalize_tmp_path(tmp, os.path.join(outdir, "o.zarr"), verbose=False)
        assert os.stat(os.path.join(outdir, "o.zarr", "raw")).st_gid == gid
