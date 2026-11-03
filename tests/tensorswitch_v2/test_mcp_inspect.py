"""
Regression tests for the MCP ``inspect_dataset`` tool.

Stdout is the JSON-RPC stream under stdio transport, so nothing may be printed
to it. ``inspect_dataset`` used to leak "Detected OME-NGFF zarr container..."
and reader warnings. It also reported the channel axis' (missing) unit for
channel-first stores.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("mcp")
tifffile = pytest.importorskip("tifffile")
nibabel = pytest.importorskip("nibabel")

from tensorswitch_v2 import mcp_server as m


@pytest.fixture
def czyx_zarr(temp_dir):
    src = os.path.join(temp_dir, "czyx.tif")
    tifffile.imwrite(src, np.zeros((2, 4, 8, 8), dtype=np.uint16), metadata={"axes": "CZYX"})
    out = os.path.join(temp_dir, "czyx.zarr")
    result = json.loads(m.convert(src, out, voxel_size="100,100,200"))
    assert result["status"] == "success"
    return out


def test_inspect_container_writes_nothing_to_stdout(czyx_zarr, capsys):
    m.inspect_dataset(czyx_zarr)
    assert capsys.readouterr().out == ""


def test_inspect_file_with_reader_warning_writes_nothing_to_stdout(temp_dir, capsys):
    path = os.path.join(temp_dir, "vol.nii.gz")
    nibabel.save(nibabel.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), np.eye(4)), path)
    m.inspect_dataset(path)
    assert capsys.readouterr().out == ""


def test_discover_datasets_writes_nothing_to_stdout(czyx_zarr, capsys):
    m.discover_datasets(os.path.dirname(czyx_zarr))
    assert capsys.readouterr().out == ""


def test_unit_comes_from_a_spatial_axis_not_the_channel(czyx_zarr):
    info = json.loads(m.inspect_dataset(czyx_zarr))
    assert info["axes"] == ["c", "z", "y", "x"]
    assert info["unit"] == "nanometer"
