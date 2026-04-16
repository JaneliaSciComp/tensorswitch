"""Tests for _warn_if_axis_voxel_mismatch.

Fires when ``--voxel_size`` gives 3 values but source axes have fewer than 3
spatial dimensions (typical TIFF mis-label case: ``axes='CYX'`` but user
intended a Z-stack).
"""

import argparse
import sys

import pytest

from tensorswitch_v2.__main__ import _warn_if_axis_voxel_mismatch


def _make_args(voxel_size=None, axes_order=None):
    return argparse.Namespace(voxel_size=voxel_size)


def test_warn_fires_on_cyx_with_3_voxel_values(capsys):
    args = _make_args(voxel_size="5.99,5.99,5.96")
    _warn_if_axis_voxel_mismatch(args, ['c', 'y', 'x'])
    err = capsys.readouterr().err
    assert 'WARNING' in err
    assert '5.96' in err  # Z value dropped
    assert '--axes_order zyx' in err  # suggested workaround
    assert "['c', 'y', 'x']" in err


def test_warn_fires_on_iyx(capsys):
    """IYX is another tifffile mis-label family."""
    args = _make_args(voxel_size="1,1,1")
    _warn_if_axis_voxel_mismatch(args, ['i', 'y', 'x'])
    err = capsys.readouterr().err
    assert 'WARNING' in err


def test_warn_fires_on_syx(capsys):
    """SYX: sample axis."""
    args = _make_args(voxel_size="1,1,1")
    _warn_if_axis_voxel_mismatch(args, ['s', 'y', 'x'])
    err = capsys.readouterr().err
    assert 'WARNING' in err


def test_no_warn_on_zyx(capsys):
    args = _make_args(voxel_size="5.99,5.99,5.96")
    _warn_if_axis_voxel_mismatch(args, ['z', 'y', 'x'])
    err = capsys.readouterr().err
    assert err == ''


def test_no_warn_on_4d_czyx(capsys):
    """CZYX has 3 spatial axes even though it starts with C."""
    args = _make_args(voxel_size="5.99,5.99,5.96")
    _warn_if_axis_voxel_mismatch(args, ['c', 'z', 'y', 'x'])
    err = capsys.readouterr().err
    assert err == ''


def test_no_warn_without_voxel_size(capsys):
    args = _make_args(voxel_size=None)
    _warn_if_axis_voxel_mismatch(args, ['c', 'y', 'x'])
    err = capsys.readouterr().err
    assert err == ''


def test_no_warn_with_2_voxel_values(capsys):
    """--voxel_size with 2 values is invalid elsewhere, but here we just
    don't trigger the 3-spatial warning logic."""
    args = _make_args(voxel_size="5.99,5.99")
    _warn_if_axis_voxel_mismatch(args, ['c', 'y', 'x'])
    err = capsys.readouterr().err
    assert err == ''


def test_no_warn_without_axes_order(capsys):
    args = _make_args(voxel_size="5.99,5.99,5.96")
    _warn_if_axis_voxel_mismatch(args, None)
    err = capsys.readouterr().err
    assert err == ''
    _warn_if_axis_voxel_mismatch(args, [])
    err = capsys.readouterr().err
    assert err == ''
