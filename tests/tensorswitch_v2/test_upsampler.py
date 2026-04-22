"""Tests for the Upsampler module (anisotropic → isotropic resampling)."""

import json
import os
import tempfile
import shutil

import numpy as np
import pytest
import zarr

from tensorswitch_v2.core.upsampler import Upsampler, upsample_to_isotropic


@pytest.fixture
def aniso_zarr2(tmp_path):
    """Create a small anisotropic Zarr2 dataset mimicking NISB structure.

    Shape: 32×32×16×1 (x, y, z, c)
    Voxels: 9×9×20 nm
    Axes: x, y, z, c
    """
    root = str(tmp_path / "aniso.zarr")
    img_group = os.path.join(root, "img")
    s0_path = os.path.join(img_group, "s0")

    # Create data with a Z gradient so interpolation is verifiable
    data = np.zeros((32, 32, 16, 1), dtype=np.uint8)
    for z in range(16):
        data[:, :, z, 0] = int(z * 15)  # 0, 15, 30, ..., 225

    # Write array
    arr = zarr.open_array(s0_path, mode="w", shape=data.shape,
                          chunks=(16, 16, 16, 1), dtype=data.dtype,
                          zarr_format=2)
    arr[:] = data

    # Write .zgroup at root
    for d in [root, img_group]:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)

    # Write OME metadata
    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": "img",
            "axes": [
                {"name": "x", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "c", "type": "channel"},
            ],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [9.0, 9.0, 20.0, 1.0]}
                ],
            }],
        }]
    }
    with open(os.path.join(img_group, ".zattrs"), "w") as f:
        json.dump(metadata, f)

    return {
        "root": root,
        "s0_path": s0_path,
        "img_group": img_group,
        "data": data,
    }


@pytest.fixture
def aniso_labels_zarr2(tmp_path):
    """Create a small anisotropic label dataset.

    Shape: 32×32×16 (x, y, z)
    Voxels: 9×9×20 nm
    """
    root = str(tmp_path / "aniso_labels.zarr")
    seg_group = os.path.join(root, "labels", "seg")
    s0_path = os.path.join(seg_group, "s0")

    # Create label data with distinct regions
    data = np.zeros((32, 32, 16), dtype=np.uint16)
    data[:16, :, :] = 1
    data[16:, :, :] = 2
    data[:, :, 8:] = 3

    arr = zarr.open_array(s0_path, mode="w", shape=data.shape,
                          chunks=(16, 16, 16), dtype=data.dtype,
                          zarr_format=2)
    arr[:] = data

    for d in [root, os.path.join(root, "labels"), seg_group]:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)

    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": "seg",
            "axes": [
                {"name": "x", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "z", "type": "space", "unit": "nanometer"},
            ],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [9.0, 9.0, 20.0]}
                ],
            }],
        }]
    }
    with open(os.path.join(seg_group, ".zattrs"), "w") as f:
        json.dump(metadata, f)

    return {
        "root": root,
        "s0_path": s0_path,
        "seg_group": seg_group,
        "data": data,
    }


class TestUpsampler:
    """Tests for the Upsampler class."""

    def test_output_shape(self, aniso_zarr2, tmp_path):
        """Upsampled shape should match expected isotropic dimensions."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        up = Upsampler(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0, 1.0],
        )
        stats = up.upsample(verbose=False)

        # Z should be upsampled: 16 * (20/9) = 35.6 → 36
        assert stats["output_shape"][0] == 32  # x unchanged
        assert stats["output_shape"][1] == 32  # y unchanged
        assert stats["output_shape"][2] == round(16 * 20.0 / 9.0)  # z upsampled
        assert stats["output_shape"][3] == 1   # c unchanged

    def test_dtype_preserved(self, aniso_zarr2, tmp_path):
        """Output dtype should match input dtype."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        up = Upsampler(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0, 1.0],
        )
        up.upsample(verbose=False)

        out = zarr.open(out_s0, mode="r")
        assert out.dtype == np.uint8

    def test_trilinear_intensity_range(self, aniso_zarr2, tmp_path):
        """Trilinear interpolation should not exceed source intensity range."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        up = Upsampler(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0, 1.0],
            upsample_method="trilinear",
        )
        up.upsample(verbose=False)

        src = zarr.open(aniso_zarr2["s0_path"], mode="r")
        out = zarr.open(out_s0, mode="r")

        assert out[:].min() >= src[:].min()
        assert out[:].max() <= src[:].max()

    def test_trilinear_interpolates_z_gradient(self, aniso_zarr2, tmp_path):
        """Z gradient should produce smoothly interpolated values."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        up = Upsampler(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0, 1.0],
            upsample_method="trilinear",
        )
        up.upsample(verbose=False)

        out = zarr.open(out_s0, mode="r")
        # Z profile should be monotonically non-decreasing (was a gradient)
        z_profile = out[16, 16, :, 0]
        diffs = np.diff(z_profile.astype(np.int16))
        assert np.all(diffs >= 0), "Z gradient should be monotonically non-decreasing"

    def test_nearest_neighbor_labels(self, aniso_labels_zarr2, tmp_path):
        """Nearest-neighbor should preserve exact label values."""
        out_s0 = str(tmp_path / "out.zarr" / "labels" / "seg" / "s0")
        up = Upsampler(
            input_path=aniso_labels_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0],
            is_label=True,
        )
        up.upsample(verbose=False)

        out = zarr.open(out_s0, mode="r")
        unique_labels = set(np.unique(out[:]))
        expected_labels = set(np.unique(aniso_labels_zarr2["data"]))
        assert unique_labels == expected_labels, \
            f"Labels changed: expected {expected_labels}, got {unique_labels}"

    def test_is_label_forces_nearest(self, aniso_labels_zarr2, tmp_path):
        """is_label=True should force nearest-neighbor even if method=trilinear."""
        out_s0 = str(tmp_path / "out.zarr" / "labels" / "seg" / "s0")
        up = Upsampler(
            input_path=aniso_labels_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0],
            upsample_method="trilinear",  # Would normally be trilinear
            is_label=True,   # Should override to nearest
        )
        assert up.upsample_method == "nearest"
        assert up.interp_order == 0

    def test_3d_array_no_channel(self, aniso_labels_zarr2, tmp_path):
        """Should work with 3D arrays (no channel dimension)."""
        out_s0 = str(tmp_path / "out.zarr" / "labels" / "seg" / "s0")
        up = Upsampler(
            input_path=aniso_labels_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0],
            is_label=True,
        )
        stats = up.upsample(verbose=False)

        assert len(stats["output_shape"]) == 3  # No channel dim
        assert stats["output_shape"][0] == 32
        assert stats["output_shape"][1] == 32

    def test_zoom_factors(self, aniso_zarr2, tmp_path):
        """Zoom factors should be correct: 1.0 for xy, 20/9 for z, 1.0 for c."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        up = Upsampler(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_sizes=[9.0, 9.0, 9.0, 1.0],
        )
        stats = up.upsample(verbose=False)

        zf = stats["zoom_factors"]
        assert abs(zf[0] - 1.0) < 1e-6  # x
        assert abs(zf[1] - 1.0) < 1e-6  # y
        assert abs(zf[2] - 20.0 / 9.0) < 1e-4  # z
        assert abs(zf[3] - 1.0) < 1e-6  # c


class TestUpsampleToIsotropic:
    """Tests for the upsample_to_isotropic convenience function."""

    def test_auto_target_voxel(self, aniso_zarr2, tmp_path):
        """With target_voxel_size=None, should use smallest voxel (9nm)."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        stats = upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            verbose=False,
        )
        # Z zoom should be 20/9
        z_zoom = stats["zoom_factors"][2]
        assert abs(z_zoom - 20.0 / 9.0) < 1e-4

    def test_explicit_target_voxel(self, aniso_zarr2, tmp_path):
        """With explicit target, zoom factors should match."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        stats = upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            target_voxel_size=10.0,
            verbose=False,
        )
        # X zoom: 9/10 = 0.9 (downsampling!), Z zoom: 20/10 = 2.0
        assert abs(stats["zoom_factors"][0] - 0.9) < 1e-4
        assert abs(stats["zoom_factors"][2] - 2.0) < 1e-4

    def test_writes_ome_metadata(self, aniso_zarr2, tmp_path):
        """Should write correct OME-NGFF 0.4 metadata at both group and root level."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            verbose=False,
        )
        out_group = os.path.dirname(out_s0)
        out_root = os.path.dirname(out_group)

        # Group-level metadata (img/.zattrs — OME-NGFF 0.4)
        with open(os.path.join(out_group, ".zattrs")) as f:
            meta = json.load(f)
        ms = meta["multiscales"][0]
        assert ms["version"] == "0.4"
        assert ms["type"] == "image"
        assert ms["datasets"][0]["path"] == "s0"
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale == [9.0, 9.0, 9.0, 1.0]

        # Root-level metadata (out.zarr/.zgroup + .zattrs)
        assert os.path.exists(os.path.join(out_root, ".zgroup"))
        with open(os.path.join(out_root, ".zattrs")) as f:
            root_meta = json.load(f)
        root_ms = root_meta["multiscales"][0]
        assert root_ms["version"] == "0.4"
        assert root_ms["type"] == "image"
        assert root_ms["datasets"][0]["path"] == "img/s0"
        root_scale = root_ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert root_scale == [9.0, 9.0, 9.0, 1.0]
        assert "_software" in root_meta

    def test_writes_root_metadata_zarr3(self, aniso_zarr2, tmp_path):
        """Should write correct OME-NGFF 0.5 zarr.json metadata for zarr3 output."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            output_format="zarr3",
            no_sharding=True,
            verbose=False,
        )
        out_group = str(tmp_path / "out.zarr" / "img")
        out_root = str(tmp_path / "out.zarr")

        # Group-level zarr.json (img/zarr.json — OME-NGFF 0.5)
        group_zarr_json = os.path.join(out_group, "zarr.json")
        assert os.path.exists(group_zarr_json)
        with open(group_zarr_json) as f:
            group_meta = json.load(f)
        assert group_meta["zarr_format"] == 3
        group_ome = group_meta["attributes"]["ome"]
        assert group_ome["version"] == "0.5"
        group_ms = group_ome["multiscales"][0]
        assert group_ms["type"] == "image"
        assert group_ms["datasets"][0]["path"] == "s0"

        # Root-level zarr.json (out.zarr/zarr.json — OME-NGFF 0.5)
        root_zarr_json = os.path.join(out_root, "zarr.json")
        assert os.path.exists(root_zarr_json)
        with open(root_zarr_json) as f:
            root_meta = json.load(f)
        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"
        root_ome = root_meta["attributes"]["ome"]
        assert root_ome["version"] == "0.5"
        root_ms = root_ome["multiscales"][0]
        assert root_ms["type"] == "image"
        assert root_ms["datasets"][0]["path"] == "img/s0"
        assert "_software" in root_meta["attributes"]

    def test_already_isotropic_raises(self, tmp_path):
        """Should raise ValueError if source is already isotropic."""
        # Create isotropic dataset
        root = str(tmp_path / "iso.zarr")
        s0_path = os.path.join(root, "img", "s0")
        data = np.zeros((16, 16, 16), dtype=np.uint8)
        arr = zarr.open_array(s0_path, mode="w", shape=data.shape,
                              chunks=(16, 16, 16), dtype=data.dtype)
        arr[:] = data

        for d in [root, os.path.join(root, "img")]:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, ".zgroup"), "w") as f:
                json.dump({"zarr_format": 2}, f)

        metadata = {
            "multiscales": [{
                "version": "0.4",
                "axes": [
                    {"name": "x", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "z", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [{"path": "s0", "coordinateTransformations": [
                    {"type": "scale", "scale": [9.0, 9.0, 9.0]}
                ]}],
            }]
        }
        with open(os.path.join(root, "img", ".zattrs"), "w") as f:
            json.dump(metadata, f)

        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        with pytest.raises(ValueError, match="already isotropic"):
            upsample_to_isotropic(input_path=s0_path, output_path=out_s0, verbose=False)


class TestUpsampleOutputFormats:
    """Tests for TensorStore-based output format support."""

    def test_zarr3_non_sharded_output(self, aniso_zarr2, tmp_path):
        """Should produce valid zarr3 output when output_format='zarr3'."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        stats = upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            output_format="zarr3",
            no_sharding=True,
            verbose=False,
        )
        # Verify zarr3 format: zarr.json exists, .zarray does not
        assert os.path.exists(os.path.join(out_s0, "zarr.json"))
        assert not os.path.exists(os.path.join(out_s0, ".zarray"))
        assert stats["output_shape"][2] == round(16 * 20.0 / 9.0)

    def test_zarr3_sharded_output(self, aniso_zarr2, tmp_path):
        """Should produce valid zarr3 sharded output."""
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        stats = upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            output_format="zarr3",
            no_sharding=False,
            verbose=False,
        )
        # Verify zarr3 format
        assert os.path.exists(os.path.join(out_s0, "zarr.json"))
        # Verify shape
        assert stats["output_shape"][0] == 32
        assert stats["output_shape"][2] == round(16 * 20.0 / 9.0)

        # Verify sharding codec in metadata
        with open(os.path.join(out_s0, "zarr.json")) as f:
            meta = json.load(f)
        codecs = meta["codecs"]
        has_sharding = any(c.get("name") == "sharding_indexed" for c in codecs)
        assert has_sharding, "zarr3 output should have sharding_indexed codec"

    def test_zarr3_sharded_data_correctness(self, aniso_zarr2, tmp_path):
        """Zarr3 sharded output should contain correct data."""
        import tensorstore as ts
        out_s0 = str(tmp_path / "out.zarr" / "img" / "s0")
        upsample_to_isotropic(
            input_path=aniso_zarr2["s0_path"],
            output_path=out_s0,
            output_format="zarr3",
            no_sharding=False,
            verbose=False,
        )
        # Read back with TensorStore
        store = ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': out_s0},
        }, open=True).result()
        data = store.read().result()
        # Z gradient should be preserved
        z_profile = data[16, 16, :, 0]
        diffs = np.diff(z_profile.astype(np.int16))
        assert np.all(diffs >= 0), "Z gradient should be monotonically non-decreasing"
