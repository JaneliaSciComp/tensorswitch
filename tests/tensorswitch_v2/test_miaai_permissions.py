"""--preset miaai creates files 644 and directories 755 and gives them group miaai; no other preset changes the defaults.

The umask is applied from the preset (and through TENSORSWITCH_UMASK into worker processes); the group is applied when an
output is finished (apply_parent_group), including the levels a pyramid adds after the label itself was written.
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
def test_the_miaai_presets_make_files_644_and_directories_755(restore, preset, tmp_path):
    og.apply_preset_permissions(preset)
    assert og.default_umask() == 0o022 and os.environ[og.GROUP_ENV] == "miaai"
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")
    assert stat.S_IMODE((tmp_path / "d").stat().st_mode) == 0o755
    assert stat.S_IMODE((tmp_path / "d" / "f").stat().st_mode) == 0o644


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
    assert og.apply_parent_group(str(out)) == 0                       # the parent's group already matches: nothing to change
    assert out.stat().st_gid == tmp_path.stat().st_gid


def test_nothing_is_changed_inside_a_job_already_running_under_the_group(restore, tmp_path, monkeypatch):
    gid = _other_group()
    if gid is None:
        pytest.skip("the user belongs to no second group")
    out = tmp_path / "label"
    out.mkdir()
    monkeypatch.setenv(og.GROUP_ENV, grp.getgrgid(gid).gr_name)
    monkeypatch.setenv(og.APPLIED_ENV, "1")
    assert og.apply_parent_group(str(out)) == 0 and out.stat().st_gid != gid
