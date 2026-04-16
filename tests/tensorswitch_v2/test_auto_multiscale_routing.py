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

import argparse
import os

from tensorswitch_v2.__main__ import (
    _is_pyramid_only_intent,
    _resolve_conversion_subgroup,
    find_base_level,
)


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


# ---------------------------------------------------------------------------
# _resolve_conversion_subgroup — regression tests for the 2026-04-16 bug where
# --auto_multiscale routed the pyramid at the container root instead of the
# freshly written subgroup, overwriting a prior stage's levels.
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Build a minimal argparse.Namespace mimicking the CLI defaults."""
    defaults = dict(
        output_format='zarr3',
        use_nested_structure=True,
        data_type='auto',
        is_label=False,
        image_key='raw',
        label_key='segmentation',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_subgroup_labels_via_is_label():
    """--is_label routes pyramid to labels/segmentation."""
    args = _make_args(is_label=True)
    assert _resolve_conversion_subgroup(args) == 'labels/segmentation'


def test_resolve_subgroup_labels_via_data_type():
    """--data-type labels (no is_label flag) also routes to labels/<key>."""
    args = _make_args(data_type='labels')
    assert _resolve_conversion_subgroup(args) == 'labels/segmentation'


def test_resolve_subgroup_labels_custom_key():
    """--label-key lands in labels/<custom_key>."""
    args = _make_args(is_label=True, label_key='mito')
    assert _resolve_conversion_subgroup(args) == 'labels/mito'


def test_resolve_subgroup_image_explicit():
    """--data-type image routes pyramid to the image key."""
    args = _make_args(data_type='image')
    assert _resolve_conversion_subgroup(args) == 'raw'


def test_resolve_subgroup_image_custom_key():
    """--image-key overrides 'raw'."""
    args = _make_args(data_type='image', image_key='em')
    assert _resolve_conversion_subgroup(args) == 'em'


def test_resolve_subgroup_auto_returns_none():
    """--data-type auto without --is_label is ambiguous; return None."""
    args = _make_args(data_type='auto', is_label=False)
    assert _resolve_conversion_subgroup(args) is None


def test_resolve_subgroup_no_nested_structure_returns_none():
    """--no-nested-structure means flat output; no subgroup to route to."""
    args = _make_args(is_label=True, use_nested_structure=False)
    assert _resolve_conversion_subgroup(args) is None


def test_resolve_subgroup_n5_returns_none():
    """N5 is always flat — even with --is_label we must not return a subgroup."""
    args = _make_args(is_label=True, output_format='n5')
    assert _resolve_conversion_subgroup(args) is None


def test_resolve_subgroup_zarr2_labels():
    """Zarr2 uses the same nested structure as Zarr3."""
    args = _make_args(is_label=True, output_format='zarr2')
    assert _resolve_conversion_subgroup(args) == 'labels/segmentation'


def test_resolve_subgroup_is_label_beats_data_type_image():
    """--is_label forces labels routing even when --data-type image is set.

    This is a defensive check — these flags should not normally both be set,
    but if they are, --is_label wins (matches the converter's precedence).
    """
    args = _make_args(is_label=True, data_type='image')
    assert _resolve_conversion_subgroup(args) == 'labels/segmentation'


# ---------------------------------------------------------------------------
# End-to-end subgroup routing: container already has raw/ multiscales from a
# prior stage. The labels stage must NOT have its pyramid land on raw/s0.
# ---------------------------------------------------------------------------

def _make_zarr2_container_with_raw_multiscales(root_dir: str) -> str:
    """Build a Zarr2 container whose root .zattrs points at raw/s0..s5.

    Mimics the state after Stage 1 of Molly's conversion: raw/ has been
    written and the container root .zattrs carries the raw multiscales block.
    """
    import json
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    # Root .zattrs — stale from a prior raw-only stage
    with open(os.path.join(root_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'name': 'test',
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [
                    {'path': f'raw/s{i}',
                     'coordinateTransformations':
                         [{'type': 'scale', 'scale': [1.0, 1.0, 1.0]}]}
                    for i in range(6)
                ],
            }],
        }, f)
    # raw subgroup with its own .zattrs + s0..s5
    raw_dir = os.path.join(root_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(raw_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'name': 'raw',
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [{'path': f's{i}'} for i in range(6)],
            }],
        }, f)
    for i in range(6):
        _make_zarr2_level(os.path.join(raw_dir, f's{i}'))
    return root_dir


def _write_zarr2_labels_s0(root_dir: str) -> str:
    """Simulate a labels conversion: write labels/segmentation/s0 only."""
    import json
    labels_dir = os.path.join(root_dir, 'labels')
    seg_dir = os.path.join(labels_dir, 'segmentation')
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(labels_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(seg_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    # labels/segmentation/.zattrs — only s0 so far (pyramid yet to run)
    with open(os.path.join(seg_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'name': 'segmentation',
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [{'path': 's0'}],
            }],
        }, f)
    _make_zarr2_level(os.path.join(seg_dir, 's0'))
    return seg_dir


def test_find_base_level_on_container_returns_stale_raw(temp_dir):
    """Guard: confirm the bug root cause — find_base_level on the container
    returns raw/s0 because root .zattrs still has the raw multiscales.
    """
    root = os.path.join(temp_dir, 'oocyte.zarr')
    _make_zarr2_container_with_raw_multiscales(root)
    _write_zarr2_labels_s0(root)

    s0, root_out = find_base_level(root)
    # Without the subgroup fix, the pyramid would read raw/s0 — which is
    # exactly the Stage-2 failure we hit.
    assert s0.endswith(os.path.join('raw', 's0'))


def test_subgroup_routing_lands_on_labels_s0(temp_dir):
    """The fix: passing the resolved subgroup to find_base_level lands on
    labels/segmentation/s0 instead of raw/s0.
    """
    root = os.path.join(temp_dir, 'oocyte.zarr')
    _make_zarr2_container_with_raw_multiscales(root)
    _write_zarr2_labels_s0(root)

    args = _make_args(is_label=True, output_format='zarr2')
    subgroup = _resolve_conversion_subgroup(args)
    assert subgroup == 'labels/segmentation'

    s0, _ = find_base_level(os.path.join(root, subgroup))
    assert s0.endswith(os.path.join('labels', 'segmentation', 's0'))


def test_subgroup_routing_image_lands_on_raw_s0(temp_dir):
    """Image stage into a container with only labels present: subgroup
    routing picks raw/s0 (which the converter just wrote).
    """
    import json
    root = os.path.join(temp_dir, 'oocyte.zarr')
    # Container with ONLY labels written first (reverse of Molly's case)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(root, '.zattrs'), 'w') as f:
        json.dump({'labels': ['segmentation']}, f)
    _write_zarr2_labels_s0(root)

    # Now simulate raw/s0 just written
    raw_dir = os.path.join(root, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(raw_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'name': 'raw',
                'axes': [{'name': a} for a in ('z', 'y', 'x')],
                'datasets': [{'path': 's0'}],
            }],
        }, f)
    _make_zarr2_level(os.path.join(raw_dir, 's0'))

    args = _make_args(data_type='image', output_format='zarr2')
    subgroup = _resolve_conversion_subgroup(args)
    assert subgroup == 'raw'

    s0, _ = find_base_level(os.path.join(root, subgroup))
    assert s0.endswith(os.path.join('raw', 's0'))
