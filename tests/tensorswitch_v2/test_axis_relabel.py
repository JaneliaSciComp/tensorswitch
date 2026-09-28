"""
Tests for --relabel_axis (explicit axis identity correction).

Background: a plain (non-ImageJ-hyperstack) multi-page TIFF -- e.g. one written
by ITK/InsightToolkit with no `slices=`/`spacing=`/`unit=` tags -- gets its
leading dimension reported by tifffile as a generic index axis `'i'` instead of
`'z'`. Previously, `--axes_order zyx` had an implicit fallback that silently
reinterpreted a detected `'t'` axis as `'z'`, but did not cover `'i'`, and --
more importantly -- that automatic guess was itself unsafe: it could silently
turn a genuine time axis into a spatial one on any dataset with real timepoint
data, just because `--axes_order zyx` happened to be passed.

`--relabel_axis OLD=NEW` replaces that implicit guess with an explicit,
always-opt-in mechanism: it never fires unless the caller names the exact
axis to correct.
"""

import os
import numpy as np
import pytest
import tifffile
import tensorstore as ts

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter


@pytest.fixture
def iyx_tiff_path(temp_dir):
    """A multi-page TIFF written the way ITK/InsightToolkit does -- no ImageJ
    hyperstack metadata -- which makes tifffile report axes='IYX' instead of
    'ZYX'. This reproduces the real Lucchi/EPFL dataset bug exactly."""
    arr = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
    path = os.path.join(temp_dir, "iyx.tif")
    tifffile.imwrite(path, arr, metadata=None, software="InsightToolkit")
    # Sanity-check the fixture itself reproduces the real-world condition.
    with tifffile.TiffFile(path) as tf:
        assert tf.series[0].axes == "IYX"
    return path


@pytest.fixture
def tyx_tiff_path(temp_dir):
    """A TIFF with a genuine ImageJ-declared time axis 'T' (not a mislabeled
    Z-stack) -- used to prove real time axes are never silently reinterpreted."""
    arr = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint16)
    path = os.path.join(temp_dir, "tyx.tif")
    tifffile.imwrite(path, arr, imagej=True, metadata={"axes": "TYX"})
    with tifffile.TiffFile(path) as tf:
        assert tf.series[0].axes == "TYX"
    return path


class TestRelabelAxisRequired:
    """--axes_order alone must never reinterpret a non-spatial axis as spatial."""

    def test_iyx_axes_order_zyx_without_relabel_raises(self, iyx_tiff_path, temp_dir):
        reader = Readers.tiff(iyx_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))
        converter = DistributedConverter(reader, writer)

        with pytest.raises(ValueError, match="doesn't match source spatial axes"):
            converter.convert(
                voxel_size_override={"x": 5.0, "y": 5.0, "z": 5.0},
                axes_order_override=["z", "y", "x"],
                verbose=False,
            )

    def test_tyx_axes_order_zyx_without_relabel_raises(self, tyx_tiff_path, temp_dir):
        """Regression guard: a genuine 't' axis must NOT be silently guessed as
        'z' anymore -- this is the exact behavior that was removed."""
        reader = Readers.tiff(tyx_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))
        converter = DistributedConverter(reader, writer)

        with pytest.raises(ValueError, match="doesn't match source spatial axes"):
            converter.convert(
                voxel_size_override={"x": 1.0, "y": 1.0, "z": 1.0},
                axes_order_override=["z", "y", "x"],
                verbose=False,
            )


class TestRelabelAxisExplicit:
    """--relabel_axis explicitly fixes a mis-detected axis, and only that axis."""

    def test_relabel_i_to_z_succeeds(self, iyx_tiff_path, temp_dir):
        reader = Readers.tiff(iyx_tiff_path)
        out_path = os.path.join(temp_dir, "out.zarr")
        writer = Writers.zarr3(out_path, use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)

        converter.convert(
            chunk_shape=(10, 32, 32),
            voxel_size_override={"x": 5.0, "y": 5.0, "z": 5.0},
            axis_relabel={"i": "z"},
            verbose=False,
        )

        arr = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": os.path.join(out_path, "s0")},
        }).result()
        assert arr.shape == (10, 32, 32)

    def test_relabel_only_touches_named_axis(self, iyx_tiff_path, temp_dir):
        """--relabel_axis i=z must not affect y or x."""
        reader = Readers.tiff(iyx_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"),
                                use_sharding=False, use_nested_structure=False)
        converter = DistributedConverter(reader, writer)

        # 'y' isn't in the relabel map, so an override expecting z,x,y (with
        # y untouched) must still resolve correctly against the corrected axes.
        converter.convert(
            chunk_shape=(10, 32, 32),
            voxel_size_override={"x": 5.0, "y": 5.0, "z": 5.0},
            axis_relabel={"i": "z"},
            axes_order_override=["x", "y", "z"],
            verbose=False,
        )  # should not raise


class TestRelabelAxisDuplicateGuard:
    """Relabeling must never silently create two axes with the same name."""

    def test_relabel_creating_duplicate_raises(self, temp_dir):
        # A source that already has a genuine 'z' plus an 'x' we try to also
        # rename to 'z' -- must fail loudly, not silently collide.
        arr = np.random.randint(0, 255, (4, 8, 16, 16), dtype=np.uint8)
        path = os.path.join(temp_dir, "tzyx.tif")
        tifffile.imwrite(path, arr, imagej=True, metadata={"axes": "TZYX"})

        reader = Readers.tiff(path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))
        converter = DistributedConverter(reader, writer)

        with pytest.raises(ValueError, match="duplicate axis"):
            converter.convert(
                axis_relabel={"x": "z"},
                verbose=False,
            )
