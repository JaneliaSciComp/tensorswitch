"""_apply_preset is shared by main() and the MCP tools; pin what each preset sets."""

from tensorswitch_v2.__main__ import _apply_preset, parse_args


def apply(*argv):
    args = parse_args(["-i", "in.tif", "-o", "out", "--quiet", *argv])
    _apply_preset(args)
    return args


def test_no_preset_changes_nothing():
    args = apply()
    assert args.chunk_shape is None and args.output_format == "zarr3"


def test_webknossos():
    args = apply("--preset", "webknossos")
    assert (args.chunk_shape, args.shard_shape, args.output_format) == ("32,32,32", "1024,1024,1024", "zarr3")


def test_paintera_defaults_to_n5_xyz_gzip():
    args = apply("--preset", "paintera")
    assert (args.output_format, args.axes_order, args.compression, args.chunk_shape) == ("n5", "xyz", "gzip", "64,64,64")


def test_paintera_with_zarr2_uses_zyx():
    args = apply("--preset", "paintera", "--output_format", "zarr2")
    assert (args.output_format, args.axes_order) == ("zarr2", "zyx")


def test_mia_lmvd():
    args = apply("--preset", "mia_lmvd")
    assert (args.chunk_shape, args.shard_shape, args.force_c_order) == ("128,128,128", "512,512,512", True)


def test_miaai_is_the_same_preset_as_mia_lmvd():
    new, old = apply("--preset", "miaai"), apply("--preset", "mia_lmvd")
    assert (new.chunk_shape, new.shard_shape, new.force_c_order, new.output_format) == \
           (old.chunk_shape, old.shard_shape, old.force_c_order, old.output_format) == ("128,128,128", "512,512,512", True, "zarr3")


def test_explicit_settings_win_over_the_preset():
    args = apply("--preset", "mia_lmvd", "--chunk_shape", "64,64,64", "--force_f_order")
    assert args.chunk_shape == "64,64,64" and not args.force_c_order and args.force_f_order
