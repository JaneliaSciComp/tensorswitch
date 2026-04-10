"""Tests for dimension name inference and the pyramid planner 5D fix."""

import json
import os
import pytest
import numpy as np
import tensorstore as ts

from tensorswitch_v2.utils.metadata_utils import (
    infer_dimension_names,
    normalize_axis_name,
)
from tensorswitch_v2.utils.pyramid_utils import calculate_pyramid_plan


# ============================================================================
# normalize_axis_name
# ============================================================================

class TestNormalizeAxisName:
    def test_channel_long(self):
        assert normalize_axis_name("channel") == "c"

    def test_channel_upper(self):
        assert normalize_axis_name("Channel") == "c"

    def test_v_to_t(self):
        assert normalize_axis_name("v") == "t"

    def test_uppercase(self):
        assert normalize_axis_name("Z") == "z"

    def test_passthrough(self):
        assert normalize_axis_name("x") == "x"
        assert normalize_axis_name("t") == "t"


# ============================================================================
# infer_dimension_names — basic shapes
# ============================================================================

class TestInferDimensionNames:
    """Core heuristic tests."""

    def test_1d(self):
        assert infer_dimension_names((512,)) == ["x"]

    def test_2d(self):
        assert infer_dimension_names((512, 512)) == ["y", "x"]

    def test_3d_all_spatial(self):
        assert infer_dimension_names((100, 512, 512)) == ["z", "y", "x"]

    def test_3d_channel_first(self):
        assert infer_dimension_names((3, 512, 512)) == ["c", "y", "x"]

    def test_3d_channel_last(self):
        assert infer_dimension_names((512, 512, 3)) == ["y", "x", "c"]

    def test_4d_standard(self):
        assert infer_dimension_names((3, 100, 512, 512)) == ["c", "z", "y", "x"]

    def test_4d_channel_last(self):
        assert infer_dimension_names((100, 512, 512, 3)) == ["z", "y", "x", "c"]

    def test_4d_channel_middle(self):
        assert infer_dimension_names((100, 3, 512, 512)) == ["z", "c", "y", "x"]

    def test_5d_standard(self):
        assert infer_dimension_names((1, 1, 4300, 14336, 16384)) == [
            "t", "c", "z", "y", "x"
        ]

    def test_5d_nonspatial_at_ends(self):
        assert infer_dimension_names((1, 100, 512, 512, 3)) == [
            "t", "z", "y", "x", "c"
        ]

    def test_6d_fallback(self):
        result = infer_dimension_names((1, 2, 3, 4, 5, 6))
        assert result == [f"dim_{i}" for i in range(6)]

    def test_empty(self):
        assert infer_dimension_names(()) == []


# ============================================================================
# infer_dimension_names — threshold edge cases
# ============================================================================

class TestInferThreshold:
    def test_at_threshold_is_nonspatial(self):
        """Dim size == threshold (10) is classified as non-spatial."""
        assert infer_dimension_names((10, 512, 512)) == ["c", "y", "x"]

    def test_above_threshold_is_spatial(self):
        """Dim size == threshold+1 (11) is classified as spatial."""
        assert infer_dimension_names((11, 512, 512)) == ["z", "y", "x"]

    def test_custom_threshold(self):
        assert infer_dimension_names((20, 512, 512), non_spatial_threshold=25) == [
            "c", "y", "x"
        ]


# ============================================================================
# infer_dimension_names — voxel_sizes hint
# ============================================================================

class TestInferWithVoxelSizes:
    def test_anisotropic_z_first(self):
        """Largest voxel at position 0 → that's z."""
        result = infer_dimension_names(
            (100, 512, 512), voxel_sizes=[0.4, 0.116, 0.116]
        )
        assert result == ["z", "y", "x"]

    def test_anisotropic_z_last(self):
        """Largest voxel at position 2 → that's z, others are y,x."""
        result = infer_dimension_names(
            (512, 512, 100), voxel_sizes=[0.116, 0.116, 0.4]
        )
        assert result == ["y", "x", "z"]

    def test_anisotropic_z_middle(self):
        """Largest voxel at position 1 → that's z."""
        result = infer_dimension_names(
            (512, 100, 512), voxel_sizes=[0.116, 0.4, 0.116]
        )
        assert result == ["y", "z", "x"]

    def test_isotropic_ignores_hint(self):
        """All voxels equal → fall back to position-based z,y,x."""
        result = infer_dimension_names(
            (100, 512, 512), voxel_sizes=[0.116, 0.116, 0.116]
        )
        assert result == ["z", "y", "x"]

    def test_voxel_hint_with_nonspatial(self):
        """Voxel hint only applies to spatial dims."""
        result = infer_dimension_names(
            (3, 512, 512, 100), voxel_sizes=[1.0, 0.116, 0.116, 0.4]
        )
        assert result == ["c", "y", "x", "z"]


# ============================================================================
# infer_dimension_names — fallback (ambiguous classification)
# ============================================================================

class TestInferFallback:
    def test_all_dims_small_3d(self):
        """All dims <= threshold → fallback to conventional z,y,x."""
        assert infer_dimension_names((5, 5, 5)) == ["z", "y", "x"]

    def test_all_dims_small_5d(self):
        """All dims small 5D → fallback to conventional t,c,z,y,x."""
        assert infer_dimension_names((1, 2, 3, 4, 5)) == ["t", "c", "z", "y", "x"]

    def test_four_spatial_dims(self):
        """4+ spatial dims → ambiguous → fallback."""
        assert infer_dimension_names((100, 200, 300, 400)) == ["c", "z", "y", "x"]


# ============================================================================
# End-to-end: calculate_pyramid_plan with null dimension_names
# ============================================================================

class TestPyramidPlan5DFix:
    """Verify the original bug is fixed: 5D zarr3 with dimension_names=null."""

    @pytest.fixture
    def zarr3_5d_no_dimnames(self, tmp_path):
        """Create a 5D zarr3 dataset with dimension_names: null in metadata."""
        root = tmp_path / "fused.zarr"
        s0_dir = root / "s0"
        shape = [1, 1, 64, 128, 128]

        data = np.random.randint(0, 255, shape, dtype=np.uint16)

        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(s0_dir)},
            "metadata": {
                "shape": shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1, 32, 32, 32]},
                },
                "data_type": "uint16",
                "node_type": "array",
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": 1}},
                ],
                # dimension_names intentionally omitted (will be null in zarr.json)
            },
        }
        store = ts.open(spec, create=True, delete_existing=True).result()
        store[...] = data

        # Overwrite zarr.json to ensure dimension_names is null
        zarr_json = s0_dir / "zarr.json"
        meta = json.loads(zarr_json.read_text())
        meta["dimension_names"] = None
        zarr_json.write_text(json.dumps(meta, indent=2))

        # Write root group metadata with OME multiscales (no axes — simulates the bug)
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "multiscales": [
                        {
                            "axes": [
                                {"name": "t", "type": "time"},
                                {"name": "c", "type": "channel"},
                                {"name": "z", "type": "space", "unit": "nanometer"},
                                {"name": "y", "type": "space", "unit": "nanometer"},
                                {"name": "x", "type": "space", "unit": "nanometer"},
                            ],
                            "datasets": [
                                {
                                    "path": "s0",
                                    "coordinateTransformations": [
                                        {
                                            "type": "scale",
                                            "scale": [1.0, 1.0, 1.0, 0.5, 0.5],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "zarr.json").write_text(json.dumps(group_meta, indent=2))

        return str(s0_dir)

    @pytest.fixture
    def zarr3_5d_no_dimnames_no_ome(self, tmp_path):
        """5D zarr3 with BOTH dimension_names=null AND no OME metadata."""
        root = tmp_path / "bare.zarr"
        s0_dir = root / "s0"
        shape = [1, 1, 64, 128, 128]

        data = np.random.randint(0, 255, shape, dtype=np.uint16)
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(s0_dir)},
            "metadata": {
                "shape": shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1, 32, 32, 32]},
                },
                "data_type": "uint16",
                "node_type": "array",
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": 1}},
                ],
            },
        }
        store = ts.open(spec, create=True, delete_existing=True).result()
        store[...] = data

        # Overwrite zarr.json to ensure dimension_names is null
        zarr_json = s0_dir / "zarr.json"
        meta = json.loads(zarr_json.read_text())
        meta["dimension_names"] = None
        zarr_json.write_text(json.dumps(meta, indent=2))

        # Minimal root metadata — no OME, no voxel sizes
        root.mkdir(parents=True, exist_ok=True)
        (root / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group"}, indent=2)
        )

        return str(s0_dir)

    def test_5d_with_ome_metadata(self, zarr3_5d_no_dimnames):
        """5D dataset with OME metadata but null dimension_names → >0 levels."""
        plan = calculate_pyramid_plan(zarr3_5d_no_dimnames)
        assert plan["num_levels"] > 0
        assert len(plan["axes_names"]) == 5

    def test_5d_no_ome_metadata(self, zarr3_5d_no_dimnames_no_ome):
        """5D dataset with NO metadata at all → inferred axes, >0 levels."""
        plan = calculate_pyramid_plan(zarr3_5d_no_dimnames_no_ome)
        assert plan["num_levels"] > 0
        assert len(plan["axes_names"]) == 5
        # Non-spatial dims (t, c) should have been inferred for size-1 dims
        non_spatial = [ax for ax in plan["axes_names"] if ax in ("t", "c")]
        assert len(non_spatial) == 2

    def test_4d_no_dimnames(self, tmp_path):
        """4D dataset with null dimension_names → >0 levels."""
        root = tmp_path / "vol4d.zarr"
        s0_dir = root / "s0"
        shape = [3, 64, 128, 128]

        data = np.random.randint(0, 255, shape, dtype=np.uint16)
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(s0_dir)},
            "metadata": {
                "shape": shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 32, 32, 32]},
                },
                "data_type": "uint16",
                "node_type": "array",
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": 1}},
                ],
            },
        }
        store = ts.open(spec, create=True, delete_existing=True).result()
        store[...] = data

        zarr_json = s0_dir / "zarr.json"
        meta = json.loads(zarr_json.read_text())
        meta["dimension_names"] = None
        zarr_json.write_text(json.dumps(meta, indent=2))

        root.mkdir(parents=True, exist_ok=True)
        (root / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group"}, indent=2)
        )

        plan = calculate_pyramid_plan(str(s0_dir))
        assert plan["num_levels"] > 0
        assert len(plan["axes_names"]) == 4
        assert plan["axes_names"][0] == "c"  # size-3 dim inferred as channel
