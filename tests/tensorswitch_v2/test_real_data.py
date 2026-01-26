"""
Real data integration tests for tensorswitch_v2.

Tests the new unified architecture with actual data files to validate
production readiness before migration.
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter

# Test data paths
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(TESTS_DIR, 'test_data', 'synthetic')
REAL_DATA_DIR = os.path.join(TESTS_DIR, 'real_test_data')


def test_synthetic_tiff_to_zarr3():
    """Test TIFF -> Zarr3 conversion with synthetic data."""
    input_path = os.path.join(SYNTHETIC_DIR, 'test_stack.tif')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== TIFF -> Zarr3 (synthetic) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        reader = Readers.tiff(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=True)

        print(f"\nStats: {stats}")

        # Validate output
        assert os.path.exists(os.path.join(output_path, 'zarr.json'))
        assert os.path.exists(os.path.join(output_path, 's0', 'zarr.json'))

        with open(os.path.join(output_path, 'zarr.json')) as f:
            metadata = json.load(f)
        assert metadata['zarr_format'] == 3

        print("PASS: TIFF -> Zarr3 (synthetic)")
        return True


def test_synthetic_n5_to_zarr3():
    """Test N5 -> Zarr3 conversion with synthetic data."""
    input_path = os.path.join(SYNTHETIC_DIR, 'test_volume.n5')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== N5 -> Zarr3 (synthetic) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # N5 synthetic data uses root level (no subdataset)
        reader = Readers.n5(input_path, dataset_path="")
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 16, 16), verbose=True)

        print(f"\nStats: {stats}")

        # Validate output
        assert os.path.exists(os.path.join(output_path, 'zarr.json'))

        print("PASS: N5 -> Zarr3 (synthetic)")
        return True


def test_synthetic_zarr3_rechunk():
    """Test Zarr3 -> Zarr3 rechunking with synthetic data."""
    input_path = os.path.join(SYNTHETIC_DIR, 'test_zarr3.zarr')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    # Check if zarr.json exists (valid zarr3 structure)
    if not os.path.exists(os.path.join(input_path, 'zarr.json')):
        # Check for s0 subdirectory
        if os.path.exists(os.path.join(input_path, 's0', 'zarr.json')):
            pass  # Valid structure with s0
        elif os.path.exists(os.path.join(input_path, 'multiscale', 's0', 'zarr.json')):
            pass  # Valid multiscale structure
        else:
            pytest.skip(f"{input_path} has invalid zarr3 structure (no zarr.json)")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== Zarr3 -> Zarr3 rechunk (synthetic) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Check if input has s0 subdirectory
        if os.path.exists(os.path.join(input_path, 's0')):
            reader = Readers.zarr3(input_path, dataset_path="s0")
        else:
            reader = Readers.zarr3(input_path, dataset_path="")

        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(8, 16, 16), verbose=True)

        print(f"\nStats: {stats}")

        print("PASS: Zarr3 -> Zarr3 rechunk (synthetic)")
        return True


def test_real_ims_to_zarr3(chunk_count=5):
    """Test IMS -> Zarr3 conversion with real data (partial)."""
    input_path = os.path.join(REAL_DATA_DIR, 'ims', 'Pat16_MF6_2024-09-25_14.02.48_F0.ims')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== IMS -> Zarr3 (real data, {chunk_count} chunks) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        reader = Readers.ims(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)

        # Get total chunks and process only a subset
        total = converter.get_total_chunks(chunk_shape=(1, 512, 512))
        print(f"Total chunks: {total}")

        # Process middle chunks for testing
        start_idx = max(0, (total // 2) - (chunk_count // 2))
        stop_idx = min(total, start_idx + chunk_count)

        print(f"Processing chunks {start_idx} to {stop_idx}")

        stats = converter.convert(
            start_idx=start_idx,
            stop_idx=stop_idx,
            chunk_shape=(1, 512, 512),
            write_metadata=True,
            verbose=True
        )

        print(f"\nStats: {stats}")

        # Validate output - for partial processing, check that s0 array was created
        # (zarr.json is only written when all chunks are processed)
        s0_path = os.path.join(output_path, 's0')
        assert os.path.exists(s0_path), f"s0 directory not created: {s0_path}"
        assert os.path.exists(os.path.join(s0_path, 'zarr.json')), "s0/zarr.json not created"
        assert stats['chunks_processed'] == chunk_count

        print("PASS: IMS -> Zarr3 (real data, partial)")
        return True


def test_real_nd2_to_zarr3(chunk_count=3):
    """Test ND2 -> Zarr3 conversion with real data (partial)."""
    input_path = os.path.join(REAL_DATA_DIR, 'nd2', '20250903_FlyID08-01_2ndGel_1%_0.8%_1hr_Atto488_40XW_006.nd2')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== ND2 -> Zarr3 (real data, {chunk_count} chunks) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        reader = Readers.nd2(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)

        # Get total chunks and process only a subset
        total = converter.get_total_chunks(chunk_shape=(1, 512, 512))
        print(f"Total chunks: {total}")

        # Process middle chunks for testing
        start_idx = max(0, (total // 2) - (chunk_count // 2))
        stop_idx = min(total, start_idx + chunk_count)

        print(f"Processing chunks {start_idx} to {stop_idx}")

        stats = converter.convert(
            start_idx=start_idx,
            stop_idx=stop_idx,
            chunk_shape=(1, 512, 512),
            write_metadata=True,
            verbose=True
        )

        print(f"\nStats: {stats}")

        # Validate output
        s0_path = os.path.join(output_path, 's0')
        assert os.path.exists(s0_path), f"s0 directory not created: {s0_path}"
        assert stats['chunks_processed'] == chunk_count

        print("PASS: ND2 -> Zarr3 (real data, partial)")
        return True


def test_real_tiff_to_zarr3(chunk_count=3):
    """Test TIFF -> Zarr3 conversion with real data (partial)."""
    input_path = os.path.join(REAL_DATA_DIR, 'tif', '20250414_1p75-fold_8xbin_nuclear_segmentation.tif')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== TIFF -> Zarr3 (real data, {chunk_count} chunks) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        reader = Readers.tiff(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)

        # Get total chunks and process only a subset
        total = converter.get_total_chunks(chunk_shape=(1, 512, 512))
        print(f"Total chunks: {total}")

        # Process middle chunks for testing
        start_idx = max(0, (total // 2) - (chunk_count // 2))
        stop_idx = min(total, start_idx + chunk_count)

        print(f"Processing chunks {start_idx} to {stop_idx}")

        stats = converter.convert(
            start_idx=start_idx,
            stop_idx=stop_idx,
            chunk_shape=(1, 512, 512),
            write_metadata=True,
            verbose=True
        )

        print(f"\nStats: {stats}")

        # Validate output
        s0_path = os.path.join(output_path, 's0')
        assert os.path.exists(s0_path), f"s0 directory not created: {s0_path}"
        assert stats['chunks_processed'] == chunk_count

        print("PASS: TIFF -> Zarr3 (real data, partial)")
        return True


def test_real_czi_to_zarr3(chunk_count=3):
    """Test CZI -> Zarr3 conversion with real data using BIOIO reader (partial)."""
    input_path = os.path.join(REAL_DATA_DIR, 'czi', 'Gel3_DE_Ribo647_nDapi_075x_4x9.czi')

    if not os.path.exists(input_path):
        pytest.skip(f"{input_path} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'output.zarr')

        print(f"\n=== CZI -> Zarr3 (real data, {chunk_count} chunks) ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Use BIOIO reader for CZI files
        reader = Readers.bioio(input_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)

        # Get total chunks and process only a subset
        total = converter.get_total_chunks(chunk_shape=(1, 512, 512))
        print(f"Total chunks: {total}")

        # Process middle chunks for testing
        start_idx = max(0, (total // 2) - (chunk_count // 2))
        stop_idx = min(total, start_idx + chunk_count)

        print(f"Processing chunks {start_idx} to {stop_idx}")

        stats = converter.convert(
            start_idx=start_idx,
            stop_idx=stop_idx,
            chunk_shape=(1, 512, 512),
            write_metadata=True,
            verbose=True
        )

        print(f"\nStats: {stats}")

        # Validate output
        s0_path = os.path.join(output_path, 's0')
        assert os.path.exists(s0_path), f"s0 directory not created: {s0_path}"
        assert os.path.exists(os.path.join(s0_path, 'zarr.json')), "s0/zarr.json not created"
        assert stats['chunks_processed'] == chunk_count

        print("PASS: CZI -> Zarr3 (real data, partial)")
        return True


def run_all_tests():
    """Run all integration tests."""
    results = {}

    # Synthetic data tests (fast)
    results['synthetic_tiff_to_zarr3'] = test_synthetic_tiff_to_zarr3()
    results['synthetic_n5_to_zarr3'] = test_synthetic_n5_to_zarr3()
    results['synthetic_zarr3_rechunk'] = test_synthetic_zarr3_rechunk()

    # Real data tests (slower)
    results['real_tiff_to_zarr3'] = test_real_tiff_to_zarr3(chunk_count=5)
    results['real_nd2_to_zarr3'] = test_real_nd2_to_zarr3(chunk_count=5)
    results['real_ims_to_zarr3'] = test_real_ims_to_zarr3(chunk_count=5)
    # results['real_czi_to_zarr3'] = test_real_czi_to_zarr3(chunk_count=3)  # Requires bioio

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for test_name, passed in results.items():
        status = "PASS" if passed else "SKIP/FAIL"
        print(f"  {test_name}: {status}")

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} passed")

    return all(results.values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorSwitch v2 real data tests")
    parser.add_argument("--test", choices=['all', 'synthetic', 'real', 'tiff', 'n5', 'zarr3', 'ims', 'nd2', 'czi', 'real_tiff', 'real_nd2'],
                       default='all', help="Which test to run")
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks for real data test")

    args = parser.parse_args()

    if args.test == 'all':
        run_all_tests()
    elif args.test == 'synthetic':
        test_synthetic_tiff_to_zarr3()
        test_synthetic_n5_to_zarr3()
        test_synthetic_zarr3_rechunk()
    elif args.test == 'real':
        test_real_tiff_to_zarr3(chunk_count=args.chunks)
        test_real_nd2_to_zarr3(chunk_count=args.chunks)
        test_real_ims_to_zarr3(chunk_count=args.chunks)
    elif args.test == 'tiff':
        test_synthetic_tiff_to_zarr3()
    elif args.test == 'n5':
        test_synthetic_n5_to_zarr3()
    elif args.test == 'zarr3':
        test_synthetic_zarr3_rechunk()
    elif args.test == 'ims':
        test_real_ims_to_zarr3(chunk_count=args.chunks)
    elif args.test == 'czi':
        test_real_czi_to_zarr3(chunk_count=args.chunks)
    elif args.test == 'nd2':
        test_real_nd2_to_zarr3(chunk_count=args.chunks)
    elif args.test == 'real_tiff':
        test_real_tiff_to_zarr3(chunk_count=args.chunks)
    elif args.test == 'real_nd2':
        test_real_nd2_to_zarr3(chunk_count=args.chunks)
