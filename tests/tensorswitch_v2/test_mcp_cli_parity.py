"""
Parity between the CLI and the MCP tools.

Every CLI option must either be produced by an MCP parameter (mcp_args.SPEC) or
be listed in mcp_args.NOT_EXPOSED with a reason. A new CLI flag fails this test
until someone decides which.
"""

import argparse

import pytest

from tensorswitch_v2 import mcp_args
from tensorswitch_v2.__main__ import parse_args


def _parser():
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def grab(self, argv=None, namespace=None):
        captured["parser"] = self
        raise SystemExit

    argparse.ArgumentParser.parse_args = grab
    try:
        try:
            parse_args([])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


PARSER = _parser()
ACTIONS = [a for a in PARSER._actions if a.dest != "help"]
ALL_FLAGS = {flag for a in ACTIONS for flag in a.option_strings}


def test_every_spec_flag_is_a_real_cli_option():
    assert not (mcp_args.exposed_flags() - ALL_FLAGS)


def test_every_cli_option_is_exposed_or_explicitly_not():
    exposed = mcp_args.exposed_flags()
    missing = [a.option_strings[-1] if a.option_strings else a.dest for a in ACTIONS
               if not (set(a.option_strings) & exposed) and a.dest not in mcp_args.NOT_EXPOSED]
    assert not missing, (
        f"CLI options with no MCP parameter and no entry in mcp_args.NOT_EXPOSED: {missing}. "
        f"Expose them in mcp_args.SPEC and the tools, or add them to NOT_EXPOSED with a reason.")


def test_not_exposed_entries_are_real_options_with_reasons():
    dests = {a.dest for a in ACTIONS}
    assert not (set(mcp_args.NOT_EXPOSED) - dests), "stale NOT_EXPOSED entries"
    assert all(reason.strip() for reason in mcp_args.NOT_EXPOSED.values())


def test_nothing_is_both_exposed_and_not_exposed():
    exposed = mcp_args.exposed_flags()
    both = [a.dest for a in ACTIONS if set(a.option_strings) & exposed and a.dest in mcp_args.NOT_EXPOSED
            and a.dest not in ("omero", "no_omero", "force_c_order", "force_f_order")]
    assert not both


SAMPLES = {
    "input_path": ("in.tif", "input", "in.tif"), "output_path": ("o.zarr", "output", "o.zarr"),
    "project": ("miaai", "project", "miaai"), "output_format": ("zarr2", "output_format", "zarr2"),
    "preset": ("webknossos", "preset", "webknossos"), "dataset_path": ("main", "dataset_path", "main"),
    "level_path": ("s1", "level_path", "s1"), "chunk_shape": ("1,2,3", "chunk_shape", "1,2,3"),
    "shard_shape": ("4,5,6", "shard_shape", "4,5,6"), "no_sharding": (True, "no_sharding", True),
    "compression": ("gzip", "compression", "gzip"), "compression_level": (9, "compression_level", 9),
    "output_dtype": ("uint16", "dtype", "uint16"), "voxel_size": ("1,2,3", "voxel_size", "1,2,3"),
    "voxel_unit": ("micrometer", "voxel_unit", "micrometer"), "is_label": (True, "is_label", True),
    "data_type": ("labels", "data_type", "labels"), "add_to_existing": (True, "add_to_existing", True),
    "output_offset": ("0,10,20", "output_offset", [0, 10, 20]), "target_shape": ("1,2,3", "target_shape", [1, 2, 3]),
    "image_key": ("img", "image_key", "img"), "label_key": ("seg", "label_key", "seg"),
    "view_index": (2, "view_index", 2), "auto_multiscale": (True, "auto_multiscale", True),
    "per_level_factors": ("1,2,2;1,2,2", "per_level_factors", "1,2,2;1,2,2"),
    "downsample_method": ("mode", "downsample_method", "mode"), "no_translation": (True, "no_translation", True),
    "expansion_factor": (4.0, "expansion_factor", 4.0), "extra_attributes": ("{\"a\": 1}", "extra_attributes", "{\"a\": 1}"),
    "bbox": ("0,0,0,4,4,4", "bbox", "0,0,0,4,4,4"), "bbox_axes": ("2,3,4", "bbox_axes", "2,3,4"),
    "squeeze_singleton_axes": (True, "squeeze_singleton_axes", True),
    "relabel_axis": ("t=z;c=y", "relabel_axis", ["t=z", "c=y"]),
    "expand_to_5d": (True, "expand_to_5d", True), "axes_order": ("xyz", "axes_order", "xyz"),
    "use_bioio": (True, "use_bioio", True), "use_bioformats": (True, "use_bioformats", True),
    "no_ome_meta_export": (True, "no_ome_meta_export", True), "no_ome_xml_attr": (True, "no_ome_xml_attr", True),
    "memory": (32, "memory", 32), "wall_time": ("2:00", "wall_time", "2:00"), "cores": (4, "cores", 4),
    "job_group": ("/g/x", "job_group", "/g/x"), "log_dir": ("/l", "log_dir", "/l"),
}


def test_every_spec_parameter_has_a_round_trip_sample():
    assert {name for name, *_ in mcp_args.SPEC} == set(SAMPLES)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_parameter_reaches_the_parsed_args(name):
    value, dest, expected = SAMPLES[name]
    params = {name: value}
    if name == "voxel_unit":
        params["voxel_size"] = "1,2,3"
    args = parse_args(mcp_args.build_argv(params))
    assert getattr(args, dest) == expected


def test_unset_values_add_nothing():
    unset = {"input_path": "", "output_format": "zarr3", "level_path": "s0", "compression": "zstd",
             "compression_level": 5, "view_index": -1, "memory": 0, "cores": 0, "data_type": "auto",
             "downsample_method": "auto", "image_key": "raw", "label_key": "segmentation",
             "no_sharding": False, "output_offset": "", "relabel_axis": "", "omero": True, "force_order": ""}
    assert mcp_args.build_argv(unset) == ["--quiet"]


def test_force_order_and_omero_special_cases():
    assert "--force_c_order" in mcp_args.build_argv({"force_order": "C"})
    assert "--force_f_order" in mcp_args.build_argv({"force_order": "f"})
    assert "--no-omero" in mcp_args.build_argv({"omero": False})
    with pytest.raises(ValueError, match="force_order"):
        mcp_args.build_argv({"force_order": "x"})
    args = parse_args(mcp_args.build_argv({"force_order": "c", "omero": False}))
    assert args.force_c_order and args.no_omero


def test_voxel_unit_alone_is_ignored():
    assert "--voxel_unit" not in mcp_args.build_argv({"voxel_unit": "micrometer"})


def test_paths_with_spaces_stay_one_argument():
    argv = mcp_args.build_argv({"input_path": "/a b/c d.nd2", "output_path": "/o p/x.zarr"})
    assert argv[argv.index("-i") + 1] == "/a b/c d.nd2"
    assert parse_args(argv).output == "/o p/x.zarr"


# ---- the tool signatures and docs follow mcp_args.SPEC --------------------------------

CLUSTER_ONLY = {"project", "memory", "wall_time", "cores", "job_group", "log_dir"}
SPEC_PARAMS = {name for name, *_ in mcp_args.SPEC} | set(mcp_args.SPECIAL)


def _tools():
    pytest.importorskip("mcp")
    import inspect

    from tensorswitch_v2 import mcp_server

    return {name: (inspect.signature(getattr(mcp_server, name)).parameters, inspect.getdoc(getattr(mcp_server, name)))
            for name in ("convert", "submit_job")}


def test_submit_job_has_every_spec_parameter():
    params, _ = _tools()["submit_job"]
    assert not (SPEC_PARAMS - set(params))


def test_convert_has_every_spec_parameter_except_cluster_ones():
    params, _ = _tools()["convert"]
    assert not (SPEC_PARAMS - CLUSTER_ONLY - set(params))
    assert not (CLUSTER_ONLY & set(params)), "convert runs locally, it must not take cluster options"


@pytest.mark.parametrize("tool", ["convert", "submit_job"])
def test_every_parameter_is_documented(tool):
    params, doc = _tools()[tool]
    undocumented = [name for name in params if f"{name}:" not in doc]
    assert not undocumented, f"{tool}: parameters missing from the docstring: {undocumented}"


@pytest.mark.parametrize("tool", ["convert", "submit_job"])
def test_parameter_defaults_agree_with_the_cli_for_shared_options(tool):
    params, _ = _tools()[tool]
    cli_defaults = {"output_format": "zarr3", "level_path": "s0", "compression": "zstd", "compression_level": 5,
                    "data_type": "auto", "image_key": "raw", "label_key": "segmentation"}
    for name, value in cli_defaults.items():
        assert params[name].default == value, (tool, name)
