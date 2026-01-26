"""
Pytest fixtures for tensorswitch_v2 tests.

Provides synthetic test data and temporary directories for testing
readers, writers, and converters.
"""

import os
import sys

# Add src to path for imports - MUST be before any other imports
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import json
import tempfile
import shutil
import pytest
import numpy as np
import tensorstore as ts


def pytest_configure(config):
    """Pytest hook to configure paths before collection."""
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="tensorswitch_test_")
    yield tmpdir
    # Cleanup after test
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def sample_3d_array():
    """Create a small 3D test array (32x64x64)."""
    np.random.seed(42)
    return np.random.randint(0, 255, (32, 64, 64), dtype=np.uint8)


@pytest.fixture
def sample_4d_array():
    """Create a small 4D test array (2x16x32x32) for CZYX."""
    np.random.seed(42)
    return np.random.randint(0, 255, (2, 16, 32, 32), dtype=np.uint16)


@pytest.fixture
def sample_tiff_path(temp_dir, sample_3d_array):
    """Create a test TIFF file."""
    import tifffile
    path = os.path.join(temp_dir, "test_input.tif")
    tifffile.imwrite(path, sample_3d_array)
    return path


@pytest.fixture
def sample_zarr3_path(temp_dir, sample_3d_array):
    """Create a test Zarr3 dataset."""
    path = os.path.join(temp_dir, "test_input.zarr")
    array_path = os.path.join(path, "s0")

    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': array_path},
        'metadata': {
            'shape': list(sample_3d_array.shape),
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [16, 32, 32]}},
            'data_type': 'uint8',
            'node_type': 'array',
            'codecs': [
                {'name': 'bytes', 'configuration': {'endian': 'little'}},
                {'name': 'gzip', 'configuration': {'level': 1}}
            ],
            'dimension_names': ['z', 'y', 'x']
        }
    }

    store = ts.open(spec, create=True, delete_existing=True).result()
    store[...] = sample_3d_array

    # Write group metadata
    group_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{
                    "axes": [
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"}
                    ],
                    "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]}]}],
                    "name": "test_image"
                }]
            }
        }
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), 'w') as f:
        json.dump(group_metadata, f, indent=2)

    return path


@pytest.fixture
def sample_zarr2_path(temp_dir, sample_3d_array):
    """Create a test Zarr2 dataset."""
    path = os.path.join(temp_dir, "test_input_v2.zarr")
    array_path = os.path.join(path, "s0")

    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': array_path},
        'metadata': {
            'shape': list(sample_3d_array.shape),
            'chunks': [16, 32, 32],
            'dtype': '|u1',
            'compressor': {'id': 'gzip', 'level': 1},
            'order': 'C',
            'dimension_separator': '/'
        }
    }

    store = ts.open(spec, create=True, delete_existing=True).result()
    store[...] = sample_3d_array

    # Write Zarr2 group files
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ".zgroup"), 'w') as f:
        json.dump({"zarr_format": 2}, f)

    with open(os.path.join(path, ".zattrs"), 'w') as f:
        json.dump({
            "multiscales": [{
                "version": "0.4",
                "name": "test_image",
                "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                "datasets": [{"path": "s0"}]
            }]
        }, f)

    return path


@pytest.fixture
def sample_n5_path(temp_dir, sample_3d_array):
    """Create a test N5 dataset."""
    path = os.path.join(temp_dir, "test_input.n5")
    array_path = os.path.join(path, "s0")

    spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': array_path},
        'metadata': {
            'dataType': 'uint8',
            'dimensions': list(sample_3d_array.shape),
            'blockSize': [16, 32, 32],
            'compression': {'type': 'gzip', 'level': 1}
        }
    }

    store = ts.open(spec, create=True, delete_existing=True).result()
    store[...] = sample_3d_array

    # Write N5 root attributes
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "attributes.json"), 'w') as f:
        json.dump({
            "n5": "2.5.0",
            "pixelResolution": {"dimensions": [0.5, 0.5, 1.0], "unit": "um"}
        }, f)

    return path


def validate_zarr3_output(output_path: str) -> dict:
    """
    Validate Zarr3 output structure.

    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check zarr.json exists at root
    zarr_json_path = os.path.join(output_path, "zarr.json")
    if not os.path.exists(zarr_json_path):
        results['errors'].append("Missing zarr.json at root")
        results['valid'] = False
    else:
        with open(zarr_json_path) as f:
            metadata = json.load(f)

        # Check zarr_format
        if metadata.get('zarr_format') != 3:
            results['errors'].append(f"Wrong zarr_format: {metadata.get('zarr_format')}")
            results['valid'] = False

        # Check for OME metadata
        if 'attributes' not in metadata or 'ome' not in metadata.get('attributes', {}):
            results['warnings'].append("Missing OME metadata")

    # Check s0 array exists
    s0_path = os.path.join(output_path, "s0", "zarr.json")
    if not os.path.exists(s0_path):
        results['errors'].append("Missing s0/zarr.json")
        results['valid'] = False

    return results


def validate_zarr2_output(output_path: str) -> dict:
    """
    Validate Zarr2 output structure.

    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check .zgroup exists at root
    zgroup_path = os.path.join(output_path, ".zgroup")
    if not os.path.exists(zgroup_path):
        results['errors'].append("Missing .zgroup at root")
        results['valid'] = False

    # Check .zattrs exists
    zattrs_path = os.path.join(output_path, ".zattrs")
    if not os.path.exists(zattrs_path):
        results['warnings'].append("Missing .zattrs at root")

    # Check s0 array exists
    s0_zarray = os.path.join(output_path, "s0", ".zarray")
    if not os.path.exists(s0_zarray):
        results['errors'].append("Missing s0/.zarray")
        results['valid'] = False

    return results


def validate_n5_output(output_path: str) -> dict:
    """
    Validate N5 output structure.

    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check attributes.json exists at root
    attrs_path = os.path.join(output_path, "attributes.json")
    if not os.path.exists(attrs_path):
        results['warnings'].append("Missing attributes.json at root")

    # Check s0 dataset exists
    s0_attrs = os.path.join(output_path, "s0", "attributes.json")
    if not os.path.exists(s0_attrs):
        results['errors'].append("Missing s0/attributes.json")
        results['valid'] = False

    return results
