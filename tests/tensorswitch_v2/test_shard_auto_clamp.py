"""Tests for smart shard shape auto-clamping to data extent."""

import pytest
from tensorswitch_v2.utils.tensorstore_utils import (
    _clamp_shard_to_shape,
    zarr3_store_spec,
)


class TestClampShardToShape:
    """Unit tests for _clamp_shard_to_shape."""

    def test_small_tile_3d(self):
        """Tile 501x1536x2466 with default 1024 shard → 512x1024x1024."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[501, 1536, 2466],
            chunk=[64, 64, 64],
        )
        assert result == [512, 1024, 1024]

    def test_large_tile_no_clamp(self):
        """Tile larger than default shard → no clamping."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[2000, 2000, 2000],
            chunk=[64, 64, 64],
        )
        assert result == [1024, 1024, 1024]

    def test_all_dims_small(self):
        """All dims smaller than 1024 → all clamped."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[50, 80, 120],
            chunk=[64, 64, 64],
        )
        assert result == [64, 128, 128]

    def test_5d_non_spatial(self):
        """5D with non-spatial dims (size 1 and 3) → non-spatial untouched."""
        result = _clamp_shard_to_shape(
            shard=[1, 1, 1024, 1024, 1024],
            shape=[1, 3, 501, 1536, 2466],
            chunk=[1, 1, 64, 64, 64],
        )
        assert result == [1, 1, 512, 1024, 1024]

    def test_exact_chunk_multiple(self):
        """Tile that is exact chunk multiple → clamped to tile size."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[512, 1024, 1024],
            chunk=[64, 64, 64],
        )
        assert result == [512, 1024, 1024]

    def test_tiny_dim(self):
        """Dim=1 rounds up to one chunk."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[1, 512, 512],
            chunk=[64, 64, 64],
        )
        assert result == [64, 512, 512]

    def test_dim_equals_chunk(self):
        """Dim exactly equals chunk size → shard = chunk."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[64, 64, 64],
            chunk=[64, 64, 64],
        )
        assert result == [64, 64, 64]

    def test_non_spatial_dim_3(self):
        """Non-spatial dim=3, chunk=1 → aligned to 3, clamped (3 < 1024 but chunk=1 so aligned=3)."""
        result = _clamp_shard_to_shape(
            shard=[1, 1024, 1024],
            shape=[3, 512, 512],
            chunk=[1, 64, 64],
        )
        assert result == [1, 512, 512]

    def test_mixed_small_large(self):
        """One dim small, others large."""
        result = _clamp_shard_to_shape(
            shard=[1024, 1024, 1024],
            shape=[100, 2000, 3000],
            chunk=[64, 64, 64],
        )
        assert result == [128, 1024, 1024]


def _get_shard_shape(spec):
    """Extract shard shape from zarr3 store spec."""
    return spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"]


class TestZarr3StoreSpecShardClamping:
    """Integration tests: zarr3_store_spec auto-clamps default shard."""

    def test_small_shape_auto_clamp(self):
        """Default shard is clamped when shape < 1024."""
        spec = zarr3_store_spec(
            path="/tmp/test.zarr",
            shape=[501, 1536, 2466],
            dtype="uint8",
            use_shard=True,
        )
        assert _get_shard_shape(spec) == [512, 1024, 1024]

    def test_large_shape_no_clamp(self):
        """Default shard unchanged when shape >= 1024."""
        spec = zarr3_store_spec(
            path="/tmp/test.zarr",
            shape=[2000, 2000, 2000],
            dtype="uint8",
            use_shard=True,
        )
        assert _get_shard_shape(spec) == [1024, 1024, 1024]

    def test_custom_shard_not_clamped(self):
        """User-provided shard shape is NOT clamped."""
        spec = zarr3_store_spec(
            path="/tmp/test.zarr",
            shape=[100, 100, 100],
            dtype="uint8",
            use_shard=True,
            custom_shard_shape=[512, 512, 512],
        )
        assert _get_shard_shape(spec) == [512, 512, 512]

    def test_5d_with_axes_auto_clamp(self):
        """5D data with axes_order → non-spatial=1, spatial clamped."""
        spec = zarr3_store_spec(
            path="/tmp/test.zarr",
            shape=[1, 3, 501, 1536, 2466],
            dtype="uint16",
            use_shard=True,
            axes_order=["t", "c", "z", "y", "x"],
        )
        assert _get_shard_shape(spec) == [1, 1, 512, 1024, 1024]
