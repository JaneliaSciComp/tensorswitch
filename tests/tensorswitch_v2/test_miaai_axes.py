"""Preset miaai drops singleton t / c / s axes (only where the axis identity is known) and never reorders space."""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import tensorstore as ts
import tifffile

from tensorswitch_v2.__main__ import main as cli_main

RNG = np.random.default_rng(11)


def convert(src, out, *extra):
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main(["-i", src, "-o", out, "--voxel_size", "8,8,40", "--voxel_unit", "nanometer", "--quiet", *extra])


def axes(out, group="raw"):
    ms = json.load(open(os.path.join(out, group, "zarr.json")))["attributes"]["ome"]["multiscales"][0]
    return [a["name"] for a in ms["axes"]], [d for d in ms["datasets"][0]["coordinateTransformations"][0]["scale"]]


def data(out, group="raw"):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(out, group, "s0")}}).result().read().result()


def ome_tiff(path, arr, order):
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": order})
    return str(path)


def zarr_source(path, arr, names):
    """A zarr3 store with named axes, which (unlike a TIFF read by tifffile) keeps length-1 axes."""
    spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {"shape": list(arr.shape), "data_type": str(arr.dtype), "dimension_names": list(names),
                         "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(arr.shape)}}}}
    ts.open(spec, create=True, delete_existing=True).result()[...] = arr
    return str(path)


@pytest.mark.parametrize("shape,expected", [
    ((1, 1, 6, 32, 48), ["z", "y", "x"]),
    ((1, 2, 6, 32, 48), ["c", "z", "y", "x"]),
    ((3, 1, 6, 32, 48), ["t", "z", "y", "x"]),
    ((3, 2, 6, 32, 48), ["t", "c", "z", "y", "x"]),
])
def test_singleton_t_and_c_are_dropped_larger_ones_kept(temp_dir, shape, expected):
    arr = RNG.integers(1, 250, shape, dtype=np.uint16)
    src = zarr_source(os.path.join(temp_dir, "src.zarr"), arr, "tczyx")
    out = os.path.join(temp_dir, "o.zarr")
    convert(src, out, "--preset", "miaai")
    names, scale = axes(out)
    assert names == expected and len(scale) == len(expected) and scale[-3:] == [40.0, 8.0, 8.0]
    assert np.array_equal(data(out), arr.reshape(data(out).shape))


def test_without_the_preset_a_5d_source_stays_5d(temp_dir):
    arr = RNG.integers(1, 250, (1, 1, 6, 32, 48), dtype=np.uint16)
    out = os.path.join(temp_dir, "n.zarr")
    convert(zarr_source(os.path.join(temp_dir, "src.zarr"), arr, "tczyx"), out)
    assert axes(out)[0] == ["t", "c", "z", "y", "x"]


def test_singleton_samples_axis_is_dropped_too(temp_dir):
    arr = RNG.integers(1, 250, (2, 1, 6, 32, 48), dtype=np.uint8)
    out = os.path.join(temp_dir, "s.zarr")
    convert(zarr_source(os.path.join(temp_dir, "src.zarr"), arr, "cszyx"), out, "--preset", "miaai")
    assert axes(out)[0] == ["c", "z", "y", "x"]


def test_a_label_added_to_a_container_is_squeezed_the_same_way(temp_dir):
    raw = RNG.integers(1, 250, (1, 1, 6, 32, 48), dtype=np.uint16)
    lab = RNG.integers(0, 5, (1, 1, 6, 32, 48), dtype=np.uint16)
    out = os.path.join(temp_dir, "c.zarr")
    convert(zarr_source(os.path.join(temp_dir, "r.zarr"), raw, "tczyx"), out, "--preset", "miaai")
    convert(zarr_source(os.path.join(temp_dir, "l.zarr"), lab, "tczyx"), out, "--preset", "miaai",
            "--add-to-existing", "--is_label", "--label-key", "seg")
    assert axes(out)[0] == axes(out, "labels/seg")[0] == ["z", "y", "x"]
    assert np.array_equal(data(out, "labels/seg"), lab.reshape(6, 32, 48))


def test_spatial_order_is_never_changed(temp_dir):
    arr = RNG.integers(1, 250, (6, 32, 48), dtype=np.uint8)
    src = os.path.join(temp_dir, "plain.tif")
    tifffile.imwrite(src, arr, shaped=False)
    out = os.path.join(temp_dir, "x.zarr")
    convert(src, out, "--preset", "miaai", "--input_axes", "xyz")       # names only: the file says x,y,z
    assert axes(out)[0] == ["x", "y", "z"] and np.array_equal(data(out), arr)


def test_a_yxz_mask_stays_yxz_with_the_preset(temp_dir):
    arr = RNG.integers(0, 5, (32, 48, 6), dtype=np.uint16)
    src = os.path.join(temp_dir, "mask.tif")
    tifffile.imwrite(src, arr, photometric="minisblack", planarconfig="contig", shaped=False)
    out = os.path.join(temp_dir, "m.zarr")
    convert(src, out, "--preset", "miaai", "--input_axes", "yxz")
    assert axes(out)[0] == ["y", "x", "z"] and np.array_equal(data(out), arr)


def test_a_source_without_axis_names_is_not_squeezed_and_does_not_fail(temp_dir):
    arr = RNG.integers(1, 250, (1, 1, 6, 32, 48), dtype=np.uint16)
    spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(temp_dir, "nn.zarr")},
            "metadata": {"shape": list(arr.shape), "data_type": "uint16",
                         "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(arr.shape)}}}}
    ts.open(spec, create=True, delete_existing=True).result()[...] = arr
    out = os.path.join(temp_dir, "nn_out.zarr")
    convert(os.path.join(temp_dir, "nn.zarr"), out, "--preset", "miaai")
    assert os.path.isdir(os.path.join(out, "raw", "s0"))
