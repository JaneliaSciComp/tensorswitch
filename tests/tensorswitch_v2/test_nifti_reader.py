"""
Tests for the NIfTI reader (Tier 2, nibabel-backed).

Covers the two things that are easy to get silently wrong with NIfTI:

1. **Axis order.** NIfTI stores voxels column-major with the fastest-varying
   axis first, so nibabel hands back an (x, y, z) array while every other
   TensorSwitch reader produces (z, y, x). ``load_nifti_stack`` transposes;
   these tests pin that down against an independently-constructed reference.

2. **Voxel size trust.** NIfTI's header has no nanometer unit, so EM-scale
   exports routinely carry a meaningless ``pixdim`` (the real UroCell dataset
   declares 224980.1875 mm). The reader must ignore implausible values rather
   than writing them into OME-NGFF metadata.
"""

import os

import numpy as np
import pytest

nibabel = pytest.importorskip("nibabel")

from tensorswitch_v2.api import Readers
from tensorswitch_v2.readers import NIfTIReader
from tensorswitch_v2.utils import load_nifti_stack, extract_nifti_metadata


def _write_nifti(path, arr_xyz, zooms=None, units=None):
    """Write a NIfTI file from an (x, y, z)-ordered array."""
    img = nibabel.Nifti1Image(arr_xyz, affine=np.eye(4))
    if zooms is not None:
        img.header.set_zooms(zooms)
    if units is not None:
        img.header.set_xyzt_units(*units)
    nibabel.save(img, path)
    return path


@pytest.fixture
def nifti_3d(temp_dir):
    """Distinct-per-axis 3D volume so a wrong transpose cannot pass."""
    rng = np.random.default_rng(0)
    arr_xyz = rng.integers(0, 255, (7, 11, 13), dtype=np.uint8)
    path = _write_nifti(os.path.join(temp_dir, "vol.nii.gz"), arr_xyz)
    return path, arr_xyz


class TestNIfTIAxisOrder:
    def test_shape_is_transposed_to_zyx(self, nifti_3d):
        path, arr_xyz = nifti_3d
        arr, dims = load_nifti_stack(path)
        assert dims == ["z", "y", "x"]
        # source (x=7, y=11, z=13) must become (z=13, y=11, x=7)
        assert tuple(arr.shape) == (13, 11, 7)

    def test_values_match_transposed_reference(self, nifti_3d):
        path, arr_xyz = nifti_3d
        arr, _ = load_nifti_stack(path)
        np.testing.assert_array_equal(np.asarray(arr), arr_xyz.T)

    def test_dtype_preserved_not_upcast_to_float64(self, nifti_3d):
        """get_fdata() would return float64; the dataobj proxy must not."""
        path, arr_xyz = nifti_3d
        arr, _ = load_nifti_stack(path)
        assert arr.dtype == np.uint8

    def test_tensorstore_domain_labels(self, nifti_3d):
        path, _ = nifti_3d
        store = NIfTIReader(path).get_tensorstore()
        assert list(store.domain.labels) == ["z", "y", "x"]


class TestNIfTIAutoDetect:
    def test_auto_detect_gz(self, nifti_3d):
        path, _ = nifti_3d
        assert isinstance(Readers.auto_detect(path), NIfTIReader)

    def test_auto_detect_plain_nii(self, temp_dir):
        arr = np.zeros((3, 4, 5), dtype=np.uint8)
        path = _write_nifti(os.path.join(temp_dir, "plain.nii"), arr)
        assert isinstance(Readers.auto_detect(path), NIfTIReader)


class TestNIfTIVoxelSizes:
    def test_micron_units_converted_to_nm(self, temp_dir):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        path = _write_nifti(os.path.join(temp_dir, "um.nii.gz"), arr,
                            zooms=(0.016, 0.016, 0.015), units=("micron", "sec"))
        _, voxel = extract_nifti_metadata(path)
        assert voxel is not None
        assert voxel["x"] == pytest.approx(16.0)
        assert voxel["z"] == pytest.approx(15.0)

    def test_implausible_pixdim_is_rejected(self, temp_dir):
        """Reproduces the real UroCell header: 224980.1875 declared as mm."""
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        path = _write_nifti(os.path.join(temp_dir, "bogus.nii.gz"), arr,
                            zooms=(224980.1875,) * 3, units=("mm", "sec"))
        _, voxel = extract_nifti_metadata(path)
        assert voxel is None, "implausible pixdim must not be trusted"

    def test_unknown_unit_is_rejected(self, temp_dir):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        path = _write_nifti(os.path.join(temp_dir, "nounit.nii.gz"), arr,
                            zooms=(1.0, 1.0, 1.0), units=("unknown", "unknown"))
        _, voxel = extract_nifti_metadata(path)
        assert voxel is None


class TestNIfTIMetadata:
    def test_reports_shape_dtype_and_version(self, nifti_3d):
        path, _ = nifti_3d
        md = NIfTIReader(path).get_metadata()
        assert md["shape"] == (13, 11, 7)   # already ZYX
        assert md["dtype"] == "uint8"
        assert md["nifti_version"] == 1

    def test_missing_file_raises(self, temp_dir):
        with pytest.raises(ValueError, match="does not exist"):
            load_nifti_stack(os.path.join(temp_dir, "nope.nii.gz"))
