"""--preset miaai gives every finished file and folder group miaai, group read+execute and no group write (directories 755,
files 654, the corpus rule of guideline section 2 and of the QC check ownership_permissions); no other preset changes the defaults.

The umask is applied from the preset (and through TENSORSWITCH_UMASK into worker processes), so new files start at 644 and new
directories at 755; the group and group execute are applied when an output is finished (apply_parent_group, and
apply_preset_to_container for the labels/ folder and the zarr.json files a run rewrites), including the levels a pyramid adds.
"""
import grp
import os
import stat
from types import SimpleNamespace

import pytest

from tensorswitch_v2.utils import output_group as og


@pytest.fixture
def restore(monkeypatch):
    old = os.umask(0o002)
    os.umask(old)
    for name in (og.UMASK_ENV, og.GROUP_ENV, og.APPLIED_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    os.umask(old)
    for name in (og.UMASK_ENV, og.GROUP_ENV, og.APPLIED_ENV):      # apply_preset_permissions sets these directly: never leak them
        os.environ.pop(name, None)


def test_the_default_umask_is_team_write(restore):
    assert og.default_umask() == 0o002


def test_a_bad_value_falls_back_to_the_default(restore, monkeypatch):
    monkeypatch.setenv(og.UMASK_ENV, "not-octal")
    assert og.default_umask() == 0o002


@pytest.mark.parametrize("preset", ["miaai", "mia_lmvd"])
def test_the_miaai_presets_start_files_at_644_and_directories_at_755(restore, preset, tmp_path):
    og.apply_preset_permissions(preset)
    assert og.default_umask() == 0o022 and os.environ[og.GROUP_ENV] == "miaai"
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")
    assert stat.S_IMODE((tmp_path / "d").stat().st_mode) & 0o777 == 0o755             # (the setgid bit of a sandbox folder is not a permission)
    assert stat.S_IMODE((tmp_path / "d" / "f").stat().st_mode) & 0o777 == 0o644


@pytest.mark.parametrize("preset", [None, "", "webknossos", "paintera"])
def test_other_presets_and_no_preset_change_nothing(restore, preset, monkeypatch):
    before = os.umask(0o002)
    os.umask(before)
    og.apply_preset_permissions(preset)
    assert og.UMASK_ENV not in os.environ and og.GROUP_ENV not in os.environ and og.default_umask() == 0o002
    now = os.umask(before)
    os.umask(now)
    assert now == before


def test_the_shared_preset_function_does_not_touch_the_process(restore):
    from tensorswitch_v2.__main__ import _apply_preset
    args = SimpleNamespace(preset="miaai", chunk_shape=None, shard_shape=None, force_c_order=False, force_f_order=False, quiet=True,
                           output_format="zarr3", axes_order=None, compression=None)
    _apply_preset(args)                                           # shared with the MCP tools: it only fills in settings
    assert og.UMASK_ENV not in os.environ and og.GROUP_ENV not in os.environ and og.default_umask() == 0o002


def test_a_conversion_applies_the_preset_only_while_it_runs(restore):
    before = os.umask(0o002)
    os.umask(before)
    with og.preset_permissions("miaai"):
        assert og.default_umask() == 0o022 and os.environ[og.GROUP_ENV] == "miaai"
        inside = os.umask(0o022)
        os.umask(inside)
        assert inside == 0o022
    assert og.UMASK_ENV not in os.environ and og.GROUP_ENV not in os.environ
    after = os.umask(before)
    os.umask(after)
    assert after == before                                         # the next conversion starts from what was there


def test_run_conversion_leaves_the_process_as_it_found_it(restore, monkeypatch):
    import tensorswitch_v2.__main__ as m
    seen = {}

    def fake(args):
        seen["umask"], seen["group"] = og.default_umask(), os.environ.get(og.GROUP_ENV)
        return "done"
    monkeypatch.setattr(m, "_run_conversion", fake)
    assert m.run_conversion(SimpleNamespace(preset="miaai")) == "done"
    assert seen == {"umask": 0o022, "group": "miaai"}
    assert og.UMASK_ENV not in os.environ and og.GROUP_ENV not in os.environ
    assert m.run_conversion(SimpleNamespace(preset="webknossos")) == "done" and seen == {"umask": 0o002, "group": None}


def _other_group():
    """A group the user belongs to that is not the one new files get, or None."""
    primary = os.getgid()
    for gid in os.getgroups():
        if gid != primary:
            return gid
    return None


def test_apply_parent_group_uses_the_preset_group_when_it_is_named_and_the_user_is_in_it(restore, tmp_path, monkeypatch):
    gid = _other_group()
    if gid is None:
        pytest.skip("the user belongs to no second group")
    name = grp.getgrgid(gid).gr_name
    out = tmp_path / "label"
    (out / "s1").mkdir(parents=True)
    (out / "s1" / "c").write_text("x")
    assert out.stat().st_gid != gid
    monkeypatch.setenv(og.GROUP_ENV, name)
    changed = og.apply_parent_group(str(out))
    assert changed >= 3 and out.stat().st_gid == gid and (out / "s1" / "c").stat().st_gid == gid


def test_a_preset_group_the_user_is_not_in_falls_back_to_the_folders_group(restore, tmp_path, monkeypatch):
    out = tmp_path / "label"
    out.mkdir()
    monkeypatch.setenv(og.GROUP_ENV, "no-such-group-xyz")
    og.apply_parent_group(str(out))                                   # the group stays the folder's own (only the modes are finished under a preset)
    assert out.stat().st_gid == tmp_path.stat().st_gid


def test_nothing_is_changed_inside_a_job_already_running_under_the_group(restore, tmp_path, monkeypatch):
    gid = _other_group()
    if gid is None:
        pytest.skip("the user belongs to no second group")
    out = tmp_path / "label"
    out.mkdir()
    monkeypatch.setenv(og.GROUP_ENV, grp.getgrgid(gid).gr_name)
    monkeypatch.setenv(og.APPLIED_ENV, "1")
    og.apply_parent_group(str(out))
    assert out.stat().st_gid != gid                                   # the group is the job's own; only the modes are finished


def _my_group():
    return grp.getgrgid(os.getgid()).gr_name


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode) & 0o777


def test_a_finished_output_under_the_preset_gets_group_read_execute_and_no_group_write(restore, tmp_path, monkeypatch):
    monkeypatch.setenv(og.GROUP_ENV, _my_group())
    out = tmp_path / "label"
    (out / "s1" / "c").mkdir(parents=True)
    f = out / "s1" / "c" / "0"
    f.write_text("x")
    os.chmod(out, 0o775)                                       # a folder made with the team umask
    os.chmod(f, 0o664)                                         # a file with group write
    og.apply_parent_group(str(out))
    assert _mode(out) == 0o755 and _mode(f) == 0o654           # group r-x, no group write, the other bits kept


def test_the_modes_are_fixed_even_inside_a_job_that_already_runs_under_the_project_group(restore, tmp_path, monkeypatch):
    monkeypatch.setenv(og.GROUP_ENV, _my_group())
    monkeypatch.setenv(og.APPLIED_ENV, "1")
    f = tmp_path / "f"
    f.write_text("x")
    os.chmod(f, 0o644)
    og.apply_parent_group(str(tmp_path))
    assert _mode(f) == 0o654


def test_without_a_preset_no_mode_is_changed(restore, tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    os.chmod(f, 0o664)
    og.apply_parent_group(str(tmp_path))
    og.apply_preset_to_container(str(tmp_path))
    assert _mode(f) == 0o664


def test_the_container_files_a_run_rewrites_and_the_labels_folder_it_creates_are_finished(restore, tmp_path, monkeypatch):
    monkeypatch.setenv(og.GROUP_ENV, _my_group())
    c = tmp_path / "crop.zarr"
    for rel in ("zarr.json", "raw/zarr.json", "labels/zarr.json", "labels/manual_gt-cell-snap_10162026/zarr.json"):
        (c / rel).parent.mkdir(parents=True, exist_ok=True)
        (c / rel).write_text("{}")
        os.chmod(c / rel, 0o644)                               # a rewritten file: a new inode, 644
    os.chmod(c / "labels", 0o775)                              # the folder the run created for a first label
    changed = og.apply_preset_to_container(str(c))
    assert changed >= 4
    for rel in ("zarr.json", "raw/zarr.json", "labels/zarr.json", "labels/manual_gt-cell-snap_10162026/zarr.json"):
        assert _mode(c / rel) == 0o654, rel
    assert _mode(c / "labels") == 0o755
    assert og.apply_preset_to_container(str(c)) == 0           # nothing more to do the second time


def test_a_symlink_is_never_touched(restore, tmp_path, monkeypatch):
    monkeypatch.setenv(og.GROUP_ENV, _my_group())
    target = tmp_path / "t"
    target.write_text("x")
    os.chmod(target, 0o600)
    link = tmp_path / "l"
    link.symlink_to(target)
    assert og.apply_preset_entry(str(link)) is False and _mode(target) == 0o600
