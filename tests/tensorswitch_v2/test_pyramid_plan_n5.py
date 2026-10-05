"""The N5 pyramid plan must pair voxel sizes with the right axes.

The writer stores pixelResolution as [x, y, z] while the array and its axes are z, y, x.
"""
import json
import os

import pytest
import tensorstore as ts

from tensorswitch_v2.utils.pyramid_utils import (
    _n5_voxel_sizes_in_array_order,
    calculate_pyramid_plan,
)
from tensorswitch_v2.writers.n5 import N5Writer


def _n5(root, shape, voxel, axes=("z", "y", "x")):
    spec = {"driver": "n5", "kvstore": {"driver": "file", "path": str(root)}, "path": "s0",
            "metadata": {"dimensions": list(shape), "blockSize": [16, 32, 32][-len(shape):],
                         "dataType": "uint8", "compression": {"type": "raw"}}}
    ts.open(spec, create=True, delete_existing=True).result()
    attrs = N5Writer._build_root_attributes(object.__new__(N5Writer), "s0", voxel, list(axes))
    with open(os.path.join(str(root), "attributes.json"), "w") as handle:
        json.dump(attrs, handle)
    return os.path.join(str(root), "s0")


def test_voxel_sizes_follow_the_axis_names():
    assert _n5_voxel_sizes_in_array_order([8.0, 8.0, 40.0], ["z", "y", "x"]) == [40.0, 8.0, 8.0]
    assert _n5_voxel_sizes_in_array_order([8.0, 9.0, 40.0], ["c", "z", "y", "x"]) == [1.0, 40.0, 9.0, 8.0]
    assert _n5_voxel_sizes_in_array_order([8.0, 9.0, 40.0], None) == [40.0, 9.0, 8.0]


def test_plan_reads_the_anisotropic_axis_as_z(tmp_path):
    plan = calculate_pyramid_plan(_n5(tmp_path / "v.n5", (64, 128, 256), {"x": 8.0, "y": 8.0, "z": 40.0}))
    assert plan["axes_names"] == ["z", "y", "x"] and plan["voxel_sizes"] == [40.0, 8.0, 8.0]
    assert plan["pyramid_plan"][0]["factor"] == [1, 2, 2]     # z is already coarse; x and y shrink


def test_isotropic_volume_is_halved_everywhere(tmp_path):
    plan = calculate_pyramid_plan(_n5(tmp_path / "i.n5", (64, 128, 256), {"x": 8.0, "y": 8.0, "z": 8.0}))
    assert plan["pyramid_plan"][0]["factor"] == [2, 2, 2]
