"""
Tests for folder_discovery.py — both old (Precomputed) and new (Zarr/N5/file) paths.

Run standalone: python tests/tensorswitch_v2/test_folder_discovery.py
"""

import os
import sys
import json
import shutil
import tempfile

# Direct import to bypass heavy package __init__.py (dask, tensorstore, etc.)
import importlib.util
_mod_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src',
    'tensorswitch_v2', 'utils', 'folder_discovery.py'
)
_spec = importlib.util.spec_from_file_location('folder_discovery', os.path.abspath(_mod_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

discover_datasets = _mod.discover_datasets
classify_dataset = _mod.classify_dataset
classify_dataset_generic = _mod.classify_dataset_generic
is_neuroglancer_precomputed = _mod.is_neuroglancer_precomputed
is_zarr_dataset = _mod.is_zarr_dataset
is_n5_dataset = _mod.is_n5_dataset
validate_discovery_for_conversion = _mod.validate_discovery_for_conversion
DiscoveredDataset = _mod.DiscoveredDataset
DiscoveryResult = _mod.DiscoveryResult
SEGMENTATION_KEYWORDS = _mod.SEGMENTATION_KEYWORDS


def _make_precomputed(path, dtype='uint8', encoding='jpeg'):
    """Create a mock Precomputed dataset."""
    os.makedirs(path, exist_ok=True)
    info = {
        '@type': 'neuroglancer_multiscale_volume',
        'data_type': dtype,
        'scales': [{
            'encoding': encoding,
            'resolution': [9, 9, 12],
            'size': [1024, 1024, 512],
            'chunk_sizes': [[64, 64, 64]],
        }],
    }
    with open(os.path.join(path, 'info'), 'w') as f:
        json.dump(info, f)


def _make_zarr3(path, dtype='uint16', shape=None):
    """Create a mock Zarr3 dataset (zarr.json at s0/)."""
    s0 = os.path.join(path, 's0')
    os.makedirs(s0, exist_ok=True)
    # Group zarr.json at root
    with open(os.path.join(path, 'zarr.json'), 'w') as f:
        json.dump({'node_type': 'group', 'zarr_format': 3}, f)
    # Array zarr.json at s0
    meta = {
        'node_type': 'array',
        'zarr_format': 3,
        'data_type': dtype,
        'shape': shape or [512, 1024, 1024],
        'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [32, 256, 256]}},
    }
    with open(os.path.join(s0, 'zarr.json'), 'w') as f:
        json.dump(meta, f)


def _make_zarr2(path, dtype='<u2', shape=None):
    """Create a mock Zarr2 dataset (.zarray at root)."""
    os.makedirs(path, exist_ok=True)
    meta = {
        'zarr_format': 2,
        'dtype': dtype,
        'shape': shape or [512, 1024, 1024],
        'chunks': [64, 256, 256],
        'compressor': {'id': 'zstd', 'level': 5},
        'order': 'C',
    }
    with open(os.path.join(path, '.zarray'), 'w') as f:
        json.dump(meta, f)


def _make_n5(path, dtype='uint16', dims=None):
    """Create a mock N5 dataset (attributes.json)."""
    os.makedirs(path, exist_ok=True)
    meta = {
        'dataType': dtype,
        'dimensions': dims or [1024, 1024, 512],
        'blockSize': [128, 128, 128],
        'compression': {'type': 'gzip', 'level': 5},
    }
    with open(os.path.join(path, 'attributes.json'), 'w') as f:
        json.dump(meta, f)


def _make_file(path):
    """Create an empty file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('')


# ===========================================================================
# Tests
# ===========================================================================

passed = 0
failed = 0


def check(name, condition, detail=''):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_classify_dataset_generic():
    print("\n--- classify_dataset_generic ---")
    # 1. Precomputed encoding wins
    check("compressed_segmentation -> seg",
          classify_dataset_generic('image', 'uint8', 'compressed_segmentation') == 'segmentation')
    # 2. Filename keywords
    check("labels.zarr -> seg",
          classify_dataset_generic('labels.zarr', 'uint8') == 'segmentation')
    check("mask.tif -> seg",
          classify_dataset_generic('mask.tif', 'uint8') == 'segmentation')
    check("segmentation_v2 -> seg",
          classify_dataset_generic('segmentation_v2', 'uint8') == 'segmentation')
    check("annotation_result -> seg",
          classify_dataset_generic('annotation_result', 'uint8') == 'segmentation')
    # 3. dtype
    check("uint64 -> seg",
          classify_dataset_generic('volume', 'uint64') == 'segmentation')
    check("uint32 -> seg",
          classify_dataset_generic('data', 'uint32') == 'segmentation')
    # 4. Default
    check("raw.zarr uint8 -> image",
          classify_dataset_generic('raw.zarr', 'uint8') == 'image')
    check("volume.tif unknown -> image",
          classify_dataset_generic('volume.tif', 'unknown') == 'image')


def test_classify_dataset_legacy():
    print("\n--- classify_dataset (legacy Precomputed) ---")
    info_seg = {
        'data_type': 'uint8',
        'scales': [{'encoding': 'compressed_segmentation', 'size': [100, 100, 100]}],
    }
    info_img = {
        'data_type': 'uint8',
        'scales': [{'encoding': 'jpeg', 'size': [100, 100, 100]}],
    }
    info_dtype_seg = {
        'data_type': 'uint64',
        'scales': [{'encoding': 'raw', 'size': [100, 100, 100]}],
    }
    check("compressed_segmentation -> seg", classify_dataset(info_seg) == 'segmentation')
    check("jpeg uint8 -> image", classify_dataset(info_img) == 'image')
    check("uint64 -> seg", classify_dataset(info_dtype_seg) == 'segmentation')


def test_format_detection():
    print("\n--- Format detection helpers ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        # Precomputed
        pc = os.path.join(tmpdir, 'precomp')
        _make_precomputed(pc)
        check("is_neuroglancer_precomputed(precomp)", is_neuroglancer_precomputed(pc))
        check("not is_zarr_dataset(precomp)", not is_zarr_dataset(pc))
        check("not is_n5_dataset(precomp)", not is_n5_dataset(pc))

        # Zarr3
        z3 = os.path.join(tmpdir, 'data.zarr')
        _make_zarr3(z3)
        check("is_zarr_dataset(zarr3)", is_zarr_dataset(z3))
        check("not is_neuroglancer_precomputed(zarr3)", not is_neuroglancer_precomputed(z3))
        check("not is_n5_dataset(zarr3)", not is_n5_dataset(z3))

        # Zarr2
        z2 = os.path.join(tmpdir, 'old.zarr')
        _make_zarr2(z2)
        check("is_zarr_dataset(zarr2)", is_zarr_dataset(z2))

        # N5
        n5 = os.path.join(tmpdir, 'data.n5')
        _make_n5(n5)
        check("is_n5_dataset(n5)", is_n5_dataset(n5))
        check("not is_zarr_dataset(n5)", not is_zarr_dataset(n5))

        # Empty dir
        empty = os.path.join(tmpdir, 'empty')
        os.makedirs(empty)
        check("not is_neuroglancer_precomputed(empty)", not is_neuroglancer_precomputed(empty))
        check("not is_zarr_dataset(empty)", not is_zarr_dataset(empty))
        check("not is_n5_dataset(empty)", not is_n5_dataset(empty))
    finally:
        shutil.rmtree(tmpdir)


def test_discover_precomputed():
    """Old functionality — Precomputed image + segmentation."""
    print("\n--- discover_datasets: Precomputed (old) ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_precomputed(os.path.join(tmpdir, 'image_data'), dtype='uint8', encoding='jpeg')
        _make_precomputed(os.path.join(tmpdir, 'seg_data'), dtype='uint64', encoding='compressed_segmentation')

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("has_segmentation", result.has_segmentation)
        check("has_both", result.has_both)
        check("image name", result.image.name == 'image_data', f"got {result.image.name}")
        check("seg name", result.segmentation.name == 'seg_data', f"got {result.segmentation.name}")
        check("image format=precomputed", result.image.source_format == 'precomputed')
        check("seg format=precomputed", result.segmentation.source_format == 'precomputed')
        check("image dtype=uint8", result.image.dtype == 'uint8')
        check("seg dtype=uint64", result.segmentation.dtype == 'uint64')
        check("image shape", result.image.shape == [1024, 1024, 512])
    finally:
        shutil.rmtree(tmpdir)


def test_discover_zarr3():
    """New functionality — Zarr3 image + segmentation."""
    print("\n--- discover_datasets: Zarr3 (new) ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_zarr3(os.path.join(tmpdir, 'raw.zarr'), dtype='uint16')
        _make_zarr3(os.path.join(tmpdir, 'labels.zarr'), dtype='uint64')

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("has_segmentation", result.has_segmentation)
        check("image name=raw.zarr", result.image.name == 'raw.zarr')
        check("seg name=labels.zarr", result.segmentation.name == 'labels.zarr')
        check("image format=zarr3", result.image.source_format == 'zarr3')
        check("seg format=zarr3", result.segmentation.source_format == 'zarr3')
        check("image dtype=uint16", result.image.dtype == 'uint16')
        # labels.zarr: keyword 'label' in name -> segmentation regardless of dtype
        check("seg classified by name", result.segmentation.data_type == 'segmentation')
    finally:
        shutil.rmtree(tmpdir)


def test_discover_zarr2():
    """New functionality — Zarr2."""
    print("\n--- discover_datasets: Zarr2 (new) ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_zarr2(os.path.join(tmpdir, 'volume.zarr'), dtype='<u2')

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("image name=volume.zarr", result.image.name == 'volume.zarr')
        check("image format=zarr2", result.image.source_format == 'zarr2')
        check("dtype normalized to uint16", result.image.dtype == 'uint16')
    finally:
        shutil.rmtree(tmpdir)


def test_discover_n5():
    """New functionality — N5."""
    print("\n--- discover_datasets: N5 (new) ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_n5(os.path.join(tmpdir, 'data.n5'), dtype='uint16')
        _make_n5(os.path.join(tmpdir, 'seg_mask.n5'), dtype='uint32')

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("has_segmentation", result.has_segmentation)
        check("image name=data.n5", result.image.name == 'data.n5')
        check("seg name=seg_mask.n5", result.segmentation.name == 'seg_mask.n5')
        check("image format=n5", result.image.source_format == 'n5')
        check("seg classified by name (mask keyword)", result.segmentation.data_type == 'segmentation')
    finally:
        shutil.rmtree(tmpdir)


def test_discover_files():
    """New functionality — file-based formats (TIFF, ND2, etc.)."""
    print("\n--- discover_datasets: Files (new) ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_file(os.path.join(tmpdir, 'volume.tif'))
        _make_file(os.path.join(tmpdir, 'segmentation_result.nd2'))

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("has_segmentation", result.has_segmentation)
        check("image name=volume.tif", result.image.name == 'volume.tif')
        check("image format=tiff", result.image.source_format == 'tiff')
        check("seg name=segmentation_result.nd2", result.segmentation.name == 'segmentation_result.nd2')
        check("seg format=nd2", result.segmentation.source_format == 'nd2')
        check("image dtype=unknown (no file open)", result.image.dtype == 'unknown')
    finally:
        shutil.rmtree(tmpdir)


def test_discover_mixed():
    """New functionality — mixed Precomputed + Zarr + files."""
    print("\n--- discover_datasets: Mixed formats ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_precomputed(os.path.join(tmpdir, 'image_precomp'), dtype='uint8', encoding='jpeg')
        _make_zarr3(os.path.join(tmpdir, 'labels.zarr'), dtype='uint8')  # keyword: labels

        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None, result.error or '')
        check("has_image", result.has_image)
        check("has_segmentation", result.has_segmentation)
        check("image from precomputed", result.image.source_format == 'precomputed')
        check("seg from zarr3", result.segmentation.source_format == 'zarr3')
    finally:
        shutil.rmtree(tmpdir)


def test_discover_empty():
    """Empty directory."""
    print("\n--- discover_datasets: Empty dir ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        result = discover_datasets(tmpdir, verbose=False)
        check("no error", result.error is None)
        check("no image", not result.has_image)
        check("no seg", not result.has_segmentation)
        check("0 images", len(result.all_images) == 0)
    finally:
        shutil.rmtree(tmpdir)


def test_discover_nonexistent():
    """Non-existent directory."""
    print("\n--- discover_datasets: Non-existent ---")
    result = discover_datasets('/nonexistent/path/xyz123', verbose=False)
    check("has error", result.error is not None)
    check("error message", 'does not exist' in result.error)


def test_validate():
    """Test validate_discovery_for_conversion."""
    print("\n--- validate_discovery_for_conversion ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_precomputed(os.path.join(tmpdir, 'image'), dtype='uint8', encoding='jpeg')
        _make_precomputed(os.path.join(tmpdir, 'seg'), dtype='uint64', encoding='compressed_segmentation')

        result = discover_datasets(tmpdir, verbose=False)

        # Default: both
        img, seg, err = validate_discovery_for_conversion(result)
        check("default: no error", err is None)
        check("default: has image", img is not None)
        check("default: has seg", seg is not None)

        # image-only
        img, seg, err = validate_discovery_for_conversion(result, image_only=True)
        check("image_only: no error", err is None)
        check("image_only: has image", img is not None)
        check("image_only: no seg", seg is None)

        # labels-only
        img, seg, err = validate_discovery_for_conversion(result, labels_only=True)
        check("labels_only: no error", err is None)
        check("labels_only: no image", img is None)
        check("labels_only: has seg", seg is not None)
    finally:
        shutil.rmtree(tmpdir)


def test_validate_multiple():
    """Test validation error when multiple datasets of same type."""
    print("\n--- validate: multiple images ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        _make_precomputed(os.path.join(tmpdir, 'image1'), dtype='uint8', encoding='jpeg')
        _make_precomputed(os.path.join(tmpdir, 'image2'), dtype='uint8', encoding='raw')

        result = discover_datasets(tmpdir, verbose=False)
        check("2 images found", len(result.all_images) == 2)
        check("no primary image (ambiguous)", result.image is None)

        img, seg, err = validate_discovery_for_conversion(result)
        check("error for multiple images", err is not None)
        check("error mentions 'multiple'", 'multiple' in err.lower() if err else False)
    finally:
        shutil.rmtree(tmpdir)


def test_num_scales_zarr3():
    """Test scale counting for Zarr3 with s0, s1, s2."""
    print("\n--- num_scales for Zarr3 ---")
    tmpdir = tempfile.mkdtemp(prefix='ts_test_')
    try:
        z = os.path.join(tmpdir, 'data.zarr')
        _make_zarr3(z, dtype='uint16')
        # Add s1 and s2
        for i in [1, 2]:
            si = os.path.join(z, f's{i}')
            os.makedirs(si, exist_ok=True)
            with open(os.path.join(si, 'zarr.json'), 'w') as f:
                json.dump({'node_type': 'array', 'data_type': 'uint16', 'shape': [256, 512, 512]}, f)

        result = discover_datasets(tmpdir, verbose=False)
        check("has_image", result.has_image)
        check("num_scales=3", result.image.num_scales == 3, f"got {result.image.num_scales}")
    finally:
        shutil.rmtree(tmpdir)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("folder_discovery.py — Test Suite")
    print("=" * 60)

    test_classify_dataset_generic()
    test_classify_dataset_legacy()
    test_format_detection()
    test_discover_precomputed()
    test_discover_zarr3()
    test_discover_zarr2()
    test_discover_n5()
    test_discover_files()
    test_discover_mixed()
    test_discover_empty()
    test_discover_nonexistent()
    test_validate()
    test_validate_multiple()
    test_num_scales_zarr3()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
