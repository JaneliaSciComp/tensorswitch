"""Tests for --auto_multiscale CLI routing.

Regression tests for Konrad Rokicki's 2026-04-16 bug report: running
``--auto_multiscale -i dataset.zarr/s0 -o dataset.zarr --submit`` (the
pattern documented in README lines 200-201) previously fell through to
the standard conversion branch, producing a "No voxel size metadata
found" error. After the routing fix the same command should dispatch to
the pyramid-only branch.

Covers all three zarr/n5 formats and both workflows:
    1. pyramid-only (with / without -o; zarr3, zarr2, n5; nested and flat)
    2. convert+pyramid (raw source files still route to the conversion path)
"""

import os

from tensorswitch_v2.__main__ import _is_pyramid_only_intent


# ---------------------------------------------------------------------------
# Fixtures: build minimal fake zarr/n5 directory layouts on disk.
# ---------------------------------------------------------------------------

def _make_zarr3_level(level_dir: str) -> None:
    """Create a zarr3 level directory with a minimal zarr.json."""
    os.makedirs(level_dir, exist_ok=True)
    with open(os.path.join(level_dir, 'zarr.json'), 'w') as f:
        f.write('{"zarr_format":3,"node_type":"array","shape":[1]}')


def _make_zarr3_container(root_dir: str, level_name: str = 's0',
                          nested: str = '') -> str:
    """Create a zarr3 container with a root zarr.json and an s0 level.

    If ``nested`` (e.g. ``'raw'`` or ``'labels/segmentation'``), the level
    lives under that nested group (OME-NGFF structure).
    Returns the path to the level directory.
    """
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, 'zarr.json'), 'w') as f:
        f.write('{"zarr_format":3,"node_type":"group"}')
    if nested:
        os.makedirs(os.path.join(root_dir, nested), exist_ok=True)
        with open(os.path.join(root_dir, nested, 'zarr.json'), 'w') as f:
            f.write('{"zarr_format":3,"node_type":"group"}')
        level_dir = os.path.join(root_dir, nested, level_name)
    else:
        level_dir = os.path.join(root_dir, level_name)
    _make_zarr3_level(level_dir)
    return level_dir


def _make_zarr2_level(level_dir: str) -> None:
    """Create a zarr2 level directory with a minimal .zarray."""
    os.makedirs(level_dir, exist_ok=True)
    with open(os.path.join(level_dir, '.zarray'), 'w') as f:
        f.write('{"zarr_format":2,"shape":[1]}')


def _make_zarr2_container(root_dir: str, level_name: str = 's0') -> str:
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, '.zgroup'), 'w') as f:
        f.write('{"zarr_format":2}')
    level_dir = os.path.join(root_dir, level_name)
    _make_zarr2_level(level_dir)
    return level_dir


def _make_n5_level(level_dir: str) -> None:
    os.makedirs(level_dir, exist_ok=True)
    with open(os.path.join(level_dir, 'attributes.json'), 'w') as f:
        f.write('{"dimensions":[1]}')


def _make_n5_container(root_dir: str, level_name: str = 's0') -> str:
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, 'attributes.json'), 'w') as f:
        f.write('{}')
    level_dir = os.path.join(root_dir, level_name)
    _make_n5_level(level_dir)
    return level_dir


# ---------------------------------------------------------------------------
# Konrad's reported scenario (the regression): case 4 in the matrix.
# ---------------------------------------------------------------------------

def test_konrad_scenario_routes_to_pyramid_only(temp_dir):
    """-i zarr/s0 -o zarr --auto_multiscale must be pyramid-only (not convert)."""
    root = os.path.join(temp_dir, 'dataset.zarr')
    s0 = _make_zarr3_container(root)

    assert _is_pyramid_only_intent(s0, root) is True


# ---------------------------------------------------------------------------
# Pyramid-only, no -o (existing behavior must keep working).
# ---------------------------------------------------------------------------

def test_level_input_no_output_is_pyramid_only(temp_dir):
    s0 = _make_zarr3_container(os.path.join(temp_dir, 'a.zarr'))
    assert _is_pyramid_only_intent(s0, None) is True
    assert _is_pyramid_only_intent(s0, '') is True


def test_container_input_no_output_is_pyramid_only(temp_dir):
    root = os.path.join(temp_dir, 'a.zarr')
    _make_zarr3_container(root)
    assert _is_pyramid_only_intent(root, None) is True


# ---------------------------------------------------------------------------
# Zarr3 / Zarr2 / N5 parity: case 4 (-i level -o container) across formats.
# ---------------------------------------------------------------------------

def test_zarr3_level_output_equals_container(temp_dir):
    root = os.path.join(temp_dir, 'z3.zarr')
    s0 = _make_zarr3_container(root)
    assert _is_pyramid_only_intent(s0, root) is True


def test_zarr2_level_output_equals_container(temp_dir):
    root = os.path.join(temp_dir, 'z2.zarr')
    s0 = _make_zarr2_container(root)
    assert _is_pyramid_only_intent(s0, root) is True


def test_n5_level_output_equals_container(temp_dir):
    root = os.path.join(temp_dir, 'x.n5')
    s0 = _make_n5_container(root)
    assert _is_pyramid_only_intent(s0, root) is True


# ---------------------------------------------------------------------------
# Container-as-input with -o (case 5): input == output.
# ---------------------------------------------------------------------------

def test_container_input_same_output_is_pyramid_only(temp_dir):
    root = os.path.join(temp_dir, 'a.zarr')
    _make_zarr3_container(root)
    assert _is_pyramid_only_intent(root, root) is True


# ---------------------------------------------------------------------------
# Nested OME-NGFF structure: -i out.zarr/raw/s0 -o out.zarr.
# ---------------------------------------------------------------------------

def test_nested_raw_structure_routes_to_pyramid_only(temp_dir):
    root = os.path.join(temp_dir, 'nested.zarr')
    s0 = _make_zarr3_container(root, nested='raw')
    assert _is_pyramid_only_intent(s0, root) is True


def test_nested_labels_structure_routes_to_pyramid_only(temp_dir):
    root = os.path.join(temp_dir, 'labels.zarr')
    s0 = _make_zarr3_container(root, nested='labels/segmentation')
    assert _is_pyramid_only_intent(s0, root) is True


# ---------------------------------------------------------------------------
# Cross-format zarr→zarr conversion (case 7) must NOT be pyramid-only.
# ---------------------------------------------------------------------------

def test_different_output_container_is_not_pyramid_only(temp_dir):
    old_root = os.path.join(temp_dir, 'old.zarr')
    new_root = os.path.join(temp_dir, 'new.zarr')
    _make_zarr3_container(old_root)
    # new_root does not need to exist; convert will create it
    assert _is_pyramid_only_intent(old_root, new_root) is False


def test_level_input_with_unrelated_output_is_not_pyramid_only(temp_dir):
    old_s0 = _make_zarr3_container(os.path.join(temp_dir, 'old.zarr'))
    new_root = os.path.join(temp_dir, 'new.zarr')
    assert _is_pyramid_only_intent(old_s0, new_root) is False


# ---------------------------------------------------------------------------
# Raw source files (case 2): convert+pyramid path must be preserved.
# ---------------------------------------------------------------------------

def test_raw_tiff_file_is_not_pyramid_only(temp_dir):
    tiff = os.path.join(temp_dir, 'raw.tif')
    with open(tiff, 'wb') as f:
        f.write(b'\x00')
    assert _is_pyramid_only_intent(tiff, os.path.join(temp_dir, 'out.zarr')) is False


def test_raw_czi_file_is_not_pyramid_only(temp_dir):
    czi = os.path.join(temp_dir, 'raw.czi')
    with open(czi, 'wb') as f:
        f.write(b'\x00')
    assert _is_pyramid_only_intent(czi, os.path.join(temp_dir, 'out.zarr')) is False


def test_nonexistent_input_is_not_pyramid_only(temp_dir):
    missing = os.path.join(temp_dir, 'missing.zarr')
    assert _is_pyramid_only_intent(missing, os.path.join(temp_dir, 'out.zarr')) is False


def test_empty_directory_is_not_pyramid_only(temp_dir):
    empty = os.path.join(temp_dir, 'empty_dir')
    os.makedirs(empty)
    assert _is_pyramid_only_intent(empty, os.path.join(temp_dir, 'out.zarr')) is False


def test_empty_input_returns_false():
    assert _is_pyramid_only_intent(None, '/tmp/out.zarr') is False
    assert _is_pyramid_only_intent('', '/tmp/out.zarr') is False


# ---------------------------------------------------------------------------
# Trailing slashes must not confuse the ancestor comparison.
# ---------------------------------------------------------------------------

def test_trailing_slashes_are_normalized(temp_dir):
    root = os.path.join(temp_dir, 'a.zarr')
    s0 = _make_zarr3_container(root)
    assert _is_pyramid_only_intent(s0 + '/', root + '/') is True
    assert _is_pyramid_only_intent(s0 + '/', root) is True
    assert _is_pyramid_only_intent(s0, root + '/') is True
