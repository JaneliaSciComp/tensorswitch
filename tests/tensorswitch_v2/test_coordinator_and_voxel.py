"""
Tests for:
1. Coordinator script generation (batch.py)
2. Voxel size path matching fix (zarr.py readers)
"""

import os
import json
import tempfile
import shutil
import sys

import pytest
import numpy as np
import tensorstore as ts

from tensorswitch_v2.core.batch import (
    _generate_coordinator_script,
    _build_s0_worker_cmd,
    _build_pyramid_worker_cmd,
    _calculate_step_resources,
)
from tensorswitch_v2.readers.zarr import (
    Zarr2Reader,
    _extract_voxel_sizes_from_multiscales,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def zarr2_nested_voxel(tmp_path):
    """Create a zarr2 structure mimicking NISB: data.zarr/img/s0 with
    multiscales metadata at both the group (img/) and root level.
    Voxel sizes stored as coordinateTransformations.
    """
    root = tmp_path / "data.zarr"

    # Create s0 array via tensorstore
    s0_path = str(root / "img" / "s0")
    data = np.random.randint(0, 255, (16, 32, 32), dtype=np.uint8)
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': s0_path},
        'metadata': {
            'shape': list(data.shape),
            'chunks': [8, 16, 16],
            'dtype': '|u1',
            'compressor': {'id': 'gzip', 'level': 1},
            'order': 'C',
            'dimension_separator': '/',
        },
    }
    store = ts.open(spec, create=True, delete_existing=True).result()
    store[...] = data

    # s0-level .zattrs — axes only, no multiscales (like real NISB data)
    with open(os.path.join(s0_path, '.zattrs'), 'w') as f:
        json.dump({'axes': ['z', 'y', 'x']}, f)

    # img group .zattrs — multiscales with RELATIVE paths
    img_dir = str(root / "img")
    os.makedirs(img_dir, exist_ok=True)
    with open(os.path.join(img_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(img_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [{
                    'path': 's0',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [20.0, 9.0, 9.0]}
                    ],
                }],
            }],
        }, f)

    # Root .zattrs — multiscales with FULL paths (img/s0)
    root_dir = str(root)
    with open(os.path.join(root_dir, '.zgroup'), 'w') as f:
        json.dump({'zarr_format': 2}, f)
    with open(os.path.join(root_dir, '.zattrs'), 'w') as f:
        json.dump({
            'multiscales': [{
                'version': '0.4',
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [{
                    'path': 'img/s0',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [20.0, 9.0, 9.0]}
                    ],
                }],
            }],
        }, f)

    return root


# ── Voxel Size Tests ──────────────────────────────────────────────────


class TestVoxelSizePathMatching:
    """Tests for voxel size extraction from nested zarr structures."""

    def test_basename_match(self):
        """_extract_voxel_sizes_from_multiscales matches basename of nested paths."""
        metadata = {
            'multiscales': [{
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'},
                ],
                'datasets': [{
                    'path': 'img/s0',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [20.0, 9.0, 9.0]}
                    ],
                }],
            }],
        }
        result = _extract_voxel_sizes_from_multiscales(metadata, '', ('', '0', 's0'))
        assert result is not None
        assert result['z'] == 20.0
        assert result['x'] == 9.0
        assert result['y'] == 9.0

    def test_direct_match_still_works(self):
        """Direct path match (e.g. 's0') still works."""
        metadata = {
            'multiscales': [{
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
                ],
                'datasets': [{
                    'path': 's0',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [1.0, 0.5, 0.5]}
                    ],
                }],
            }],
        }
        result = _extract_voxel_sizes_from_multiscales(metadata, '', ('', '0', 's0'))
        assert result is not None
        # micrometer -> nanometer: 0.5 * 1000 = 500
        assert result['x'] == 500.0

    def test_no_match_returns_none(self):
        """Returns None when no dataset path matches."""
        metadata = {
            'multiscales': [{
                'axes': [{'name': 'z'}, {'name': 'y'}, {'name': 'x'}],
                'datasets': [{
                    'path': 'completely/different/path',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [1.0, 1.0, 1.0]}
                    ],
                }],
            }],
        }
        result = _extract_voxel_sizes_from_multiscales(metadata, '', ('', '0', 's0'))
        assert result is None

    def test_reader_from_s0_subpath(self, zarr2_nested_voxel):
        """Zarr2Reader opened at img/s0 finds voxel sizes from parent .zattrs."""
        s0_path = str(zarr2_nested_voxel / "img" / "s0")
        reader = Zarr2Reader(s0_path)
        voxels = reader.get_voxel_sizes()
        assert voxels['z'] == 20.0
        assert voxels['x'] == 9.0
        assert voxels['y'] == 9.0

    def test_reader_parent_walkup(self, zarr2_nested_voxel):
        """Metadata search walks up to parent dir when s0/.zattrs lacks multiscales."""
        s0_path = str(zarr2_nested_voxel / "img" / "s0")
        reader = Zarr2Reader(s0_path)
        meta = reader.get_metadata()
        assert 'multiscales' in meta
        # Should have found it from img/.zattrs (parent)
        assert meta.get('_multiscales_source') == str(zarr2_nested_voxel / "img")


# ── Coordinator Script Tests ──────────────────────────────────────────


class TestCoordinatorScript:
    """Tests for coordinator bash script generation."""

    def test_generate_script_structure(self):
        """Generated script has correct bash structure and step count."""
        steps = [
            {
                'name': 'image s0 conversion',
                'job_name': 'tsv2_image_s0_test',
                'worker_cmd': [sys.executable, '-m', 'tensorswitch_v2', '--input', '/in', '--output', '/out'],
                'memory_gb': 30,
                'wall_time': '1:00',
                'cores': 2,
            },
            {
                'name': 'image pyramid',
                'job_name': 'tsv2_image_pyramid_test',
                'worker_cmd': [sys.executable, '-m', 'tensorswitch_v2', '--input', '/out/raw', '--auto_multiscale'],
                'memory_gb': 30,
                'wall_time': '1:00',
                'cores': 2,
            },
        ]
        script = _generate_coordinator_script(steps, 'myproject', '/mygroup', '/tmp/logs')

        assert script.startswith('#!/bin/bash\n')
        assert 'set -e' in script
        assert '\nbsub -K' in script
        # Count actual bsub command lines (not the comment)
        assert sum(1 for line in script.splitlines() if line.startswith('bsub -K')) == 2
        assert 'Step 1/2: image s0 conversion' in script
        assert 'Step 2/2: image pyramid' in script
        assert 'All steps complete' in script

    def test_generate_script_four_steps(self):
        """Full coordinator with image+labels has 4 steps."""
        steps = [
            {'name': 'image s0', 'job_name': 'j1', 'worker_cmd': ['echo', '1'],
             'memory_gb': 30, 'wall_time': '1:00', 'cores': 2},
            {'name': 'image pyramid', 'job_name': 'j2', 'worker_cmd': ['echo', '2'],
             'memory_gb': 30, 'wall_time': '1:00', 'cores': 2},
            {'name': 'labels s0', 'job_name': 'j3', 'worker_cmd': ['echo', '3'],
             'memory_gb': 15, 'wall_time': '0:30', 'cores': 2},
            {'name': 'labels pyramid', 'job_name': 'j4', 'worker_cmd': ['echo', '4'],
             'memory_gb': 15, 'wall_time': '0:30', 'cores': 2},
        ]
        script = _generate_coordinator_script(steps, 'proj', '/grp', '/logs')
        assert sum(1 for line in script.splitlines() if line.startswith('bsub -K')) == 4
        assert 'Step 1/4' in script
        assert 'Step 4/4' in script


class TestBuildWorkerCommands:
    """Tests for s0 and pyramid worker command builders."""

    def test_s0_image_cmd(self):
        """Image s0 command has correct flags."""
        cmd = _build_s0_worker_cmd(
            input_path='/data/img/s0',
            output_path='/out/data.zarr',
            data_type='image',
            output_format='zarr3',
            compression='zstd',
            compression_level=5,
            image_key='raw',
            label_key='seg',
            use_nested_structure=True,
            chunk_shape='256,256,64',
            shard_shape=None,
        )
        assert '--input' in cmd
        assert '/data/img/s0' in cmd
        assert '--output' in cmd
        assert '--data-type' in cmd
        idx = cmd.index('--data-type')
        assert cmd[idx + 1] == 'image'
        assert '--use-nested-structure' in cmd
        assert '--chunk_shape' in cmd
        assert '--is-label' not in cmd
        assert '--add-to-existing' not in cmd

    def test_s0_label_add_to_existing(self):
        """Label s0 command includes --is-label and --add-to-existing."""
        cmd = _build_s0_worker_cmd(
            input_path='/data/seg/s0',
            output_path='/out/data.zarr',
            data_type='labels',
            output_format='zarr3',
            compression='zstd',
            compression_level=5,
            image_key='raw',
            label_key='seg',
            use_nested_structure=True,
            chunk_shape=None,
            shard_shape=None,
            add_to_existing=True,
        )
        assert '--is-label' in cmd
        assert '--add-to-existing' in cmd

    def test_s0_label_standalone(self):
        """Label-only s0 command has --is-label but NOT --add-to-existing."""
        cmd = _build_s0_worker_cmd(
            input_path='/data/seg/s0',
            output_path='/out/data.zarr',
            data_type='labels',
            output_format='zarr3',
            compression='zstd',
            compression_level=5,
            image_key='raw',
            label_key='seg',
            use_nested_structure=True,
            chunk_shape=None,
            shard_shape=None,
            add_to_existing=False,
        )
        assert '--is-label' in cmd
        assert '--add-to-existing' not in cmd

    def test_pyramid_cmd(self):
        """Pyramid command uses --auto_multiscale without --submit."""
        cmd = _build_pyramid_worker_cmd('/out/data.zarr/raw')
        assert '--auto_multiscale' in cmd
        assert '--submit' not in cmd
        assert '--input' in cmd
        idx = cmd.index('--input')
        assert cmd[idx + 1] == '/out/data.zarr/raw'

    def test_pyramid_label_subgroup(self):
        """Pyramid command for labels points to labels/seg subgroup."""
        cmd = _build_pyramid_worker_cmd('/out/data.zarr/labels/seg')
        idx = cmd.index('--input')
        assert cmd[idx + 1] == '/out/data.zarr/labels/seg'
