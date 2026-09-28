"""
Tests for the PNG Z-stack reader (Tier 2).

Covers the three things that are easy to get silently wrong with PNG stacks:

1. **Slice ordering.** Published EM stacks are routinely named without zero
   padding (``im0 … im1039``). A lexicographic sort orders those
   ``im0, im1, im10, im100``, which builds a volume with shuffled sections and
   no error anywhere. ``_find_png_files`` sorts naturally; these tests pin it.

2. **Zip equivalence.** PNG volumes ship as one archive (PyTC's EM30-H is 1040
   slices in a 24 GB zip), read lazily rather than extracted. The zip path must
   produce exactly the array the directory path does.

3. **Voxel size honesty.** PNG carries no voxel size at all, so the reader must
   always fall through to the default-with-warning rather than inventing one
   from the pHYs print-resolution chunk.
"""

import os
import warnings
import zipfile

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from tensorswitch_v2.api import Readers
from tensorswitch_v2.readers import PngReader
from tensorswitch_v2.utils import load_png_stack


def _write_stack(directory, n=12, shape=(8, 10), pad=False):
    """Write n single-value PNG slices; slice z has every pixel == z."""
    for i in range(n):
        name = f"im{i:04d}.png" if pad else f"im{i}.png"
        Image.fromarray(np.full(shape, i, dtype=np.uint8)).save(
            os.path.join(directory, name))
    return directory


def test_directory_stack_shape_and_order(tmp_path):
    d = str(tmp_path); _write_stack(d, n=12)
    arr = np.asarray(PngReader(d).get_tensorstore())
    assert arr.shape == (12, 8, 10)
    assert arr.dtype == np.uint8
    # the natural-sort assertion: slice z must hold value z
    assert [int(arr[z, 0, 0]) for z in range(12)] == list(range(12))


def test_unpadded_names_are_not_sorted_lexicographically(tmp_path):
    """im0, im1, im10, im100 must NOT be the resulting order."""
    d = str(tmp_path); _write_stack(d, n=101)
    arr = np.asarray(load_png_stack(d))
    assert arr.shape[0] == 101
    assert [int(arr[z, 0, 0]) for z in range(101)] == list(range(101))
    # lexicographic order would have put slice 1 at index 2
    assert int(arr[2, 0, 0]) == 2


def test_zip_matches_directory(tmp_path):
    d = str(tmp_path / "d"); os.makedirs(d); _write_stack(d, n=12)
    zp = str(tmp_path / "stack.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for i in range(12):
            zf.write(os.path.join(d, f"im{i}.png"), f"im_pad/im{i}.png")
    from_dir = np.asarray(PngReader(d).get_tensorstore())
    from_zip = np.asarray(PngReader(zp).get_tensorstore())
    assert np.array_equal(from_dir, from_zip)


def test_zip_is_not_extracted(tmp_path):
    """Reading a zip must not leave extracted files behind."""
    d = str(tmp_path / "d"); os.makedirs(d); _write_stack(d, n=4)
    zp = str(tmp_path / "stack.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for i in range(4):
            zf.write(os.path.join(d, f"im{i}.png"), f"im_pad/im{i}.png")
    before = set(os.listdir(tmp_path))
    np.asarray(PngReader(zp).get_tensorstore())
    assert set(os.listdir(tmp_path)) == before


def test_auto_detect_routes_to_png_reader(tmp_path):
    d = str(tmp_path / "d"); os.makedirs(d); _write_stack(d, n=3)
    assert isinstance(Readers.auto_detect(d), PngReader)
    assert isinstance(Readers.auto_detect(os.path.join(d, "im0.png")), PngReader)
    zp = str(tmp_path / "s.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(os.path.join(d, "im0.png"), "im0.png")
    assert isinstance(Readers.auto_detect(zp), PngReader)


def test_single_png_is_2d(tmp_path):
    d = str(tmp_path); _write_stack(d, n=1)
    r = PngReader(os.path.join(d, "im0.png"))
    arr = np.asarray(r.get_tensorstore())
    assert arr.shape == (8, 10)
    assert r._dimension_names == ["y", "x"]


def test_voxel_size_always_defaults_with_warning(tmp_path):
    d = str(tmp_path); _write_stack(d, n=3)
    r = PngReader(d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vs = r.get_voxel_sizes()
    assert vs == {"x": 1.0, "y": 1.0, "z": 1.0}
    assert any("PNG" in str(w.message) for w in caught)


def test_dimension_names_for_stack(tmp_path):
    d = str(tmp_path); _write_stack(d, n=5)
    r = PngReader(d); r.get_tensorstore()
    assert r._dimension_names == ["z", "y", "x"]


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No PNG files"):
        load_png_stack(str(tmp_path))


def test_zip_with_mismatched_slice_raises(tmp_path):
    """A slice of a different shape means a corrupt archive -- fail loudly."""
    d = str(tmp_path / "d"); os.makedirs(d); _write_stack(d, n=3)
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(os.path.join(d, "im3.png"))
    zp = str(tmp_path / "bad.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for i in range(4):
            zf.write(os.path.join(d, f"im{i}.png"), f"im{i}.png")
    with pytest.raises(ValueError, match="does not match"):
        np.asarray(PngReader(zp).get_tensorstore())


def test_batch_mode_treats_png_directory_as_one_volume(tmp_path):
    """A slice directory must be one dataset, not N datasets.

    Without the is_png_zstack_directory check in batch.py this falls through to
    'batch_directory' and every slice is converted as its own volume -- silently,
    and catastrophically for a 1040-slice stack.
    """
    from tensorswitch_v2.core.batch import detect_input_mode
    d = str(tmp_path / "slices"); os.makedirs(d); _write_stack(d, n=8)
    assert detect_input_mode(d, str(tmp_path / "out.zarr")) == "single_file"


def test_zip_handle_is_cached_not_reopened(tmp_path):
    """Repeated slice reads must reuse one ZipFile handle (the CZI PR #15 fix)."""
    import tensorswitch_v2.utils.format_loaders as fl
    d = str(tmp_path / "d"); os.makedirs(d); _write_stack(d, n=6)
    zp = str(tmp_path / "s.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for i in range(6):
            zf.write(os.path.join(d, f"im{i}.png"), f"im{i}.png")
    fl._png_zip_cache.pop(zp, None)
    opened = []
    real = zipfile.ZipFile

    class _Counting(zipfile.ZipFile):
        def __init__(self, *a, **k):
            opened.append(a[0] if a else k.get("file"))
            super().__init__(*a, **k)

    zipfile.ZipFile = _Counting
    try:
        np.asarray(PngReader(zp).get_tensorstore())
    finally:
        zipfile.ZipFile = real
    # one open for this archive, not one per slice
    assert opened.count(zp) == 1, f"reopened {opened.count(zp)}x"


def test_is_png_zstack_directory_rejects_single_file(tmp_path):
    from tensorswitch_v2.utils import is_png_zstack_directory
    d = str(tmp_path / "one"); os.makedirs(d); _write_stack(d, n=1)
    assert is_png_zstack_directory(d) is False
    _write_stack(d, n=2)
    assert is_png_zstack_directory(d) is True
