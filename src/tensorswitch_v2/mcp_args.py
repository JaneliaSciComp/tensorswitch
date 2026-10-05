"""
Translate MCP tool parameters into command-line arguments for the real CLI parser.

The MCP tools do not build their own argument objects. They build an argv list
here, parse it with ``__main__.parse_args`` and run the same code as the CLI, so
every CLI option, default and validation applies to the MCP as well.

SPEC lists the MCP parameter behind each CLI option. NOT_EXPOSED lists the
options that are deliberately not available as MCP parameters, with the reason;
tests/tensorswitch_v2/test_mcp_cli_parity.py fails if a CLI option is in neither,
so a new CLI flag cannot be forgotten.
"""

from typing import Any, Dict, List

# (mcp parameter, cli flag, kind, values that mean "not set")
#   value : [flag, str(v)]            flag : [flag] when truthy
#   list  : [flag, *items]            repeat: [flag, item] for each item
SPEC = [
    ("input_path", "-i", "value", ("", None)),
    ("output_path", "-o", "value", ("", None)),
    ("project", "-P", "value", ("", None)),
    ("output_format", "--output_format", "value", ("", None, "zarr3")),
    ("preset", "--preset", "value", ("", None)),
    ("dataset_path", "--dataset_path", "value", ("", None)),
    ("level_path", "--level_path", "value", ("", None, "s0")),
    ("chunk_shape", "--chunk_shape", "value", ("", None)),
    ("shard_shape", "--shard_shape", "value", ("", None)),
    ("no_sharding", "--no_sharding", "flag", (False, None)),
    ("compression", "--compression", "value", ("", None, "zstd")),
    ("compression_level", "--compression_level", "value", (None, 5)),
    ("output_dtype", "--dtype", "value", ("", None)),
    ("voxel_size", "--voxel_size", "value", ("", None)),
    ("voxel_unit", "--voxel_unit", "value", ("", None)),
    ("is_label", "--is_label", "flag", (False, None)),
    ("data_type", "--data-type", "value", ("", None, "auto")),
    ("add_to_existing", "--add-to-existing", "flag", (False, None)),
    ("output_offset", "--output-offset", "list", ("", None, [])),
    ("target_shape", "--target-shape", "list", ("", None, [])),
    ("image_key", "--image-key", "value", ("", None, "raw")),
    ("label_key", "--label-key", "value", ("", None, "segmentation")),
    ("view_index", "--view_index", "value", (None, -1)),
    ("auto_multiscale", "--auto_multiscale", "flag", (False, None)),
    ("per_level_factors", "--per_level_factors", "value", ("", None)),
    ("downsample_method", "--downsample_method", "value", ("", None, "auto")),
    ("no_translation", "--no-translation", "flag", (False, None)),
    ("expansion_factor", "--expansion_factor", "value", (None, 0, 0.0)),
    ("extra_attributes", "--extra_attributes", "value", ("", None)),
    ("bbox", "--bbox", "value", ("", None)),
    ("bbox_axes", "--bbox_axes", "value", ("", None)),
    ("squeeze_singleton_axes", "--squeeze_singleton_axes", "flag", (False, None)),
    ("relabel_axis", "--relabel_axis", "repeat", ("", None, [])),
    ("expand_to_5d", "--expand-to-5d", "flag", (False, None)),
    ("axes_order", "--axes_order", "value", ("", None)),
    ("use_bioio", "--use_bioio", "flag", (False, None)),
    ("use_bioformats", "--use_bioformats", "flag", (False, None)),
    ("no_ome_meta_export", "--no_ome_meta_export", "flag", (False, None)),
    ("no_ome_xml_attr", "--no_ome_xml_attr", "flag", (False, None)),
    ("memory", "--memory", "value", (None, 0)),
    ("wall_time", "--wall_time", "value", ("", None)),
    ("cores", "--cores", "value", (None, 0)),
    ("job_group", "--job_group", "value", ("", None)),
    ("log_dir", "--log_dir", "value", ("", None)),
]

# Parameters that map to a pair of flags and are handled in build_argv().
SPECIAL = {
    "force_order": ("--force_c_order", "--force_f_order"),   # "c" / "f"
    "omero": ("--no-omero",),                                  # omero=False
}

# CLI options (by dest) that are not MCP parameters, and why.
NOT_EXPOSED = {
    "version": "prints the version",
    "use_nested_structure": "internal: nested OME layout is always on",
    "image_only": "folder batch mode, not exposed yet",
    "labels_only": "folder batch mode, not exposed yet",
    "pattern": "folder batch mode, not exposed yet (see discover_datasets)",
    "recursive": "folder batch mode, not exposed yet",
    "max_concurrent": "folder batch mode, not exposed yet",
    "skip_existing": "folder batch mode, not exposed yet",
    "no_skip_existing": "folder batch mode, not exposed yet",
    "dry_run": "only applies to folder batch mode, not exposed yet",
    "status": "folder batch status check, not exposed yet",
    "batch_worker": "internal: LSF array worker",
    "index_file": "internal: LSF array worker",
    "start_idx": "internal: manual chunk-range worker",
    "stop_idx": "internal: manual chunk-range worker",
    "write_metadata": "internal: manual chunk-range worker",
    "downsample": "pyramid level mode: use generate_pyramid",
    "target_level": "pyramid level mode: use generate_pyramid",
    "single_level_factor": "pyramid level mode: use generate_pyramid",
    "cumulative_factors": "pyramid level mode: use generate_pyramid",
    "cumulative_factor_for_metadata": "pyramid level mode: use generate_pyramid",
    "use_shard": "pyramid level mode: use generate_pyramid",
    "upsample": "isotropic upsampling: use upsample_to_isotropic",
    "target_voxel_size": "isotropic upsampling: use upsample_to_isotropic",
    "upsample_method": "isotropic upsampling: use upsample_to_isotropic",
    "auto_resources": "automatic resource estimation is always on",
    "submit": "always set by submit_job",
    "sync": "blocks until the job ends, which would hang a tool call; use check_job_status",
    "quiet": "always set",
    "show_spec": "debug output",
    "omero": "handled by the omero parameter (--no-omero)",
    "no_omero": "handled by the omero parameter",
    "force_c_order": "handled by the force_order parameter",
    "force_f_order": "handled by the force_order parameter",
}


def _is_unset(value: Any, unset: tuple) -> bool:
    return any(value is u or (type(value) is type(u) and value == u) for u in unset)


def _items(value: Any) -> List[str]:
    """'0,0,64' / '0 0 64' / [0, 0, 64] -> ['0', '0', '64']"""
    if isinstance(value, str):
        return [t for t in value.replace(",", " ").split() if t]
    return [str(v) for v in value]


def build_argv(params: Dict[str, Any]) -> List[str]:
    """CLI argument list for the given MCP parameters (unset values are left out)."""
    argv: List[str] = ["--quiet"]
    for name, flag, kind, unset in SPEC:
        if name not in params:
            continue
        value = params[name]
        if _is_unset(value, unset):
            continue
        if name == "voxel_unit" and not params.get("voxel_size"):
            continue  # a unit without a size is meaningless (the old tools ignored it too)
        if kind == "flag":
            if value:
                argv.append(flag)
        elif kind == "value":
            argv += [flag, str(value)]
        elif kind == "list":
            argv += [flag, *_items(value)]
        elif kind == "repeat":
            specs = [s.strip() for s in (value.replace(";", " ").split() if isinstance(value, str) else value)]
            for spec in specs:
                argv += [flag, spec]
    order = (params.get("force_order") or "").lower()
    if order == "c":
        argv.append("--force_c_order")
    elif order == "f":
        argv.append("--force_f_order")
    elif order:
        raise ValueError(f"force_order must be 'c', 'f' or empty, got {order!r}")
    if params.get("omero") is False:
        argv.append("--no-omero")
    return argv


def exposed_flags() -> set:
    """Every CLI option string that some MCP parameter produces."""
    flags = {flag for _, flag, _, _ in SPEC}
    for pair in SPECIAL.values():
        flags.update(pair)
    return flags
