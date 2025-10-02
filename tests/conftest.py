"""
Pytest configuration and shared fixtures for TensorSwitch tests.
"""

import os
import sys
import shutil
import pytest
import tensorstore as ts

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from tensorswitch.utils import (
    load_nd2_stack,
    load_ims_stack,
    zarr3_store_spec,
    zarr2_store_spec,
    get_total_chunks_from_store
)
from test_data_config import OUTPUTS_DIR


@pytest.fixture(scope="session", autouse=True)
def setup_outputs_dir():
    """Create outputs directory for tests."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    yield
    # Cleanup after all tests


@pytest.fixture
def cleanup_output():
    """Cleanup fixture for individual tests."""
    outputs = []

    def _register(path):
        outputs.append(path)
        return path

    yield _register

    # Cleanup registered outputs
    for output in outputs:
        if os.path.exists(output):
            if os.path.isdir(output):
                shutil.rmtree(output)
            else:
                os.remove(output)


def get_middle_chunks(file_path, file_type, chunk_count=10, use_shard=False):
    """
    Calculate middle chunk range for testing.

    Args:
        file_path: Path to data file
        file_type: Type of file ('nd2', 'ims', 'tif', 'zarr3', 'zarr2', 'n5')
        chunk_count: Number of chunks to process (default: 10)
        use_shard: Whether using sharded format

    Returns:
        (start_idx, stop_idx): Middle chunk range
    """
    # Load data to get shape
    h5_file = None
    if file_type == 'nd2':
        volume = load_nd2_stack(file_path)
    elif file_type == 'ims':
        volume, h5_file = load_ims_stack(file_path)
    elif file_type in ['tif', 'tiff']:
        import tifffile
        volume = tifffile.imread(file_path)
    elif file_type == 'zarr3':
        store = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file', 'path': file_path}}).result()
        volume = store
    elif file_type == 'zarr2':
        store = ts.open({'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': file_path}}).result()
        volume = store
    elif file_type == 'n5':
        store = ts.open({'driver': 'n5', 'kvstore': {'driver': 'file', 'path': file_path}}).result()
        volume = store
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Get shape and dtype
    shape = volume.shape
    # Handle numpy dtype objects (from ND2/TIFF/IMS)
    if hasattr(volume.dtype, 'name'):
        dtype = volume.dtype.name
    else:
        dtype = str(volume.dtype)

    # Calculate chunk shape based on dimensions
    if len(shape) == 3:
        chunk_shape = (1, min(2304, shape[1]), min(2304, shape[2]))
    elif len(shape) == 4:
        chunk_shape = (1, 1, min(2304, shape[2]), min(2304, shape[3]))
    elif len(shape) == 5:
        chunk_shape = (1, 1, 1, min(2304, shape[3]), min(2304, shape[4]))
    else:
        chunk_shape = (64, 64, 64)  # Default

    # Create temporary store to calculate chunks
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp.zarr')

    try:
        if file_type in ['zarr3', 'n5'] or use_shard:
            store_spec = zarr3_store_spec(temp_path, shape, dtype, use_shard=use_shard)
        else:
            store_spec = zarr2_store_spec(temp_path, shape, chunk_shape)

        temp_store = ts.open(store_spec, create=True, delete_existing=True).result()
        total_chunks = get_total_chunks_from_store(temp_store)

        # Calculate middle range
        middle_start = (total_chunks // 2) - (chunk_count // 2)
        middle_end = middle_start + chunk_count

        # Ensure valid range
        middle_start = max(0, middle_start)
        middle_end = min(total_chunks, middle_end)

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Close h5 file if it was opened for IMS
        if h5_file is not None:
            try:
                h5_file.close()
            except:
                pass

        # Close volume if needed
        if hasattr(volume, '_meta') and hasattr(volume._meta, 'close'):
            try:
                volume._meta.close()
            except:
                pass

    return middle_start, middle_end, total_chunks


def validate_zarr3_ome_metadata(zarr_path):
    """
    Validate OME-Zarr v0.5 metadata structure.

    Args:
        zarr_path: Path to zarr3 group

    Returns:
        bool: True if valid OME-Zarr metadata
    """
    import json

    # Check for multiscale subdirectory (new structure)
    if os.path.exists(os.path.join(zarr_path, 'multiscale')):
        zarr_path = os.path.join(zarr_path, 'multiscale')

    # Check group zarr.json
    group_json = os.path.join(zarr_path, 'zarr.json')
    assert os.path.exists(group_json), f"Group zarr.json not found: {group_json}"

    with open(group_json, 'r') as f:
        metadata = json.load(f)

    assert metadata.get('zarr_format') == 3, "Not zarr3 format"
    assert metadata.get('node_type') == 'group', "Not a group node"
    assert 'ome' in metadata.get('attributes', {}), "OME metadata missing"

    ome = metadata['attributes']['ome']
    assert 'multiscales' in ome, "Multiscales metadata missing"

    multiscales = ome['multiscales'][0]
    # Version field is optional, check if present
    if 'version' in multiscales:
        assert multiscales['version'] in ['0.4', '0.5'], f"Unsupported OME-Zarr version: {multiscales.get('version')}"
    assert 'axes' in multiscales, "Axes metadata missing"
    assert 'datasets' in multiscales, "Datasets metadata missing"
    assert len(multiscales['datasets']) > 0, "No datasets in multiscales"

    # Check s0 array exists
    s0_path = os.path.join(zarr_path, 's0')
    assert os.path.exists(s0_path), "s0 array directory not found"

    s0_json = os.path.join(s0_path, 'zarr.json')
    assert os.path.exists(s0_json), "s0 zarr.json not found"

    return True


def validate_zarr2_ome_metadata(zarr_path):
    """
    Validate Zarr2 OME-Zarr metadata structure.

    Args:
        zarr_path: Path to zarr2 group

    Returns:
        bool: True if valid OME-Zarr metadata
    """
    import json

    # Check for multiscale subdirectory (new structure)
    if os.path.exists(os.path.join(zarr_path, 'multiscale')):
        zarr_path = os.path.join(zarr_path, 'multiscale')

    # Check group .zattrs
    zattrs = os.path.join(zarr_path, '.zattrs')
    assert os.path.exists(zattrs), f"Group .zattrs not found: {zattrs}"

    with open(zattrs, 'r') as f:
        metadata = json.load(f)

    assert 'multiscales' in metadata, "Multiscales metadata missing"

    multiscales = metadata['multiscales'][0]
    assert 'axes' in multiscales, "Axes metadata missing"
    assert 'datasets' in multiscales, "Datasets metadata missing"

    # Check s0 array exists
    s0_path = os.path.join(zarr_path, 's0')
    assert os.path.exists(s0_path), "s0 array directory not found"

    s0_zarray = os.path.join(s0_path, '.zarray')
    assert os.path.exists(s0_zarray), "s0 .zarray not found"

    return True
