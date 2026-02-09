#!/usr/bin/env python
"""
Test F-order (Fortran/column-major) and C-order (row-major) for tensorswitch_v2.

Tests using real IMS data:
1. C-order output (default) - write and verify data integrity
2. F-order output - write with transpose codec and verify data integrity
3. Verify both orders produce identical data when read back

Usage:
    pixi run python tests/tensorswitch_v2/test_fc_order.py
"""

import os
import sys
import json
import numpy as np
import tensorstore as ts

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tensorswitch_v2.readers.ims import IMSReader
from tensorswitch_v2.utils import get_tensorstore_context, zarr3_store_spec


def test_c_order_real_data(test_data, output_path):
    """Test C-order (default) with real IMS data subset."""
    print("\n" + "=" * 60)
    print("TEST 1: C-ORDER (REAL DATA)")
    print("=" * 60)

    shape = test_data.shape
    dtype = str(test_data.dtype)

    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Output: {output_path}")

    # Create zarr3 spec with C-order (no 5D expansion for direct test)
    spec = zarr3_store_spec(
        path=output_path,
        shape=list(shape),
        dtype=dtype,
        use_shard=True,
        level_path="",
        use_ome_structure=False,
        custom_chunk_shape=[32, 128, 128],
        custom_shard_shape=[64, 256, 256],
        use_fortran_order=False,  # C-order
        axes_order=['z', 'y', 'x']
    )
    spec['context'] = get_tensorstore_context()

    # Write data
    print("\nWriting C-order data...")
    store = ts.open(spec, create=True, delete_existing=True).result()
    store.write(test_data).result()
    print(f"  Wrote {shape[0]} slices")

    # Verify zarr.json metadata
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'r') as f:
        zarr_metadata = json.load(f)

    codecs = zarr_metadata.get('codecs', [])
    has_transpose = any(c.get('name') == 'transpose' for c in codecs)

    print(f"\nZarr metadata check:")
    print(f"  Has transpose codec: {has_transpose}")
    print(f"  Expected: False (C-order)")

    if has_transpose:
        print("  ERROR: C-order should NOT have transpose codec!")
        return False, None

    # Read back and verify
    read_store = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': output_path},
        'context': get_tensorstore_context()
    }, read=True).result()

    read_data = read_store.read().result()

    if np.array_equal(test_data, read_data):
        print("  Data verification: PASS")
        return True, read_data
    else:
        print("  Data verification: FAIL")
        diff = np.sum(test_data != read_data)
        print(f"    Different elements: {diff}")
        return False, None


def test_f_order_real_data(test_data, output_path):
    """Test F-order (Fortran) with real IMS data subset."""
    print("\n" + "=" * 60)
    print("TEST 2: F-ORDER (REAL DATA)")
    print("=" * 60)

    shape = test_data.shape
    dtype = str(test_data.dtype)

    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Output: {output_path}")

    # Create zarr3 spec with F-order (no sharding - transpose codec has issues with sharding)
    spec = zarr3_store_spec(
        path=output_path,
        shape=list(shape),
        dtype=dtype,
        use_shard=False,  # Disable sharding for F-order test
        level_path="",
        use_ome_structure=False,
        custom_chunk_shape=[32, 128, 128],
        use_fortran_order=True,  # F-order
        axes_order=['z', 'y', 'x']
    )
    spec['context'] = get_tensorstore_context()

    # Write data
    print("\nWriting F-order data...")
    store = ts.open(spec, create=True, delete_existing=True).result()
    store.write(test_data).result()
    print(f"  Wrote {shape[0]} slices")

    # Verify zarr.json metadata
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'r') as f:
        zarr_metadata = json.load(f)

    codecs = zarr_metadata.get('codecs', [])
    transpose_codec = next((c for c in codecs if c.get('name') == 'transpose'), None)

    print(f"\nZarr metadata check:")
    print(f"  Has transpose codec: {transpose_codec is not None}")
    print(f"  Expected: True (F-order)")

    if transpose_codec:
        order = transpose_codec.get('configuration', {}).get('order', [])
        expected_order = list(range(len(shape) - 1, -1, -1))  # Reversed
        print(f"  Transpose order: {order}")
        print(f"  Expected order: {expected_order}")
        if order == expected_order:
            print("  Transpose order: CORRECT")
        else:
            print("  WARNING: Transpose order mismatch!")
    else:
        print("  ERROR: F-order should have transpose codec!")
        return False, None

    # Read back and verify
    read_store = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': output_path},
        'context': get_tensorstore_context()
    }, read=True).result()

    read_data = read_store.read().result()

    if np.array_equal(test_data, read_data):
        print("  Data verification: PASS")
        return True, read_data
    else:
        print("  Data verification: FAIL")
        diff = np.sum(test_data != read_data)
        print(f"    Different elements: {diff}")
        return False, None


def test_f_order_with_sharding(test_data, output_path):
    """Test F-order (Fortran) WITH sharding - previously broken, now fixed."""
    print("\n" + "=" * 60)
    print("TEST 2b: F-ORDER WITH SHARDING (REAL DATA)")
    print("=" * 60)

    shape = test_data.shape
    dtype = str(test_data.dtype)

    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Output: {output_path}")

    # Create zarr3 spec with F-order AND sharding (the fix!)
    spec = zarr3_store_spec(
        path=output_path,
        shape=list(shape),
        dtype=dtype,
        use_shard=True,  # Enable sharding - now works with F-order!
        level_path="",
        use_ome_structure=False,
        custom_chunk_shape=[32, 128, 128],
        custom_shard_shape=[64, 256, 256],
        use_fortran_order=True,  # F-order
        axes_order=['z', 'y', 'x']
    )
    spec['context'] = get_tensorstore_context()

    # Write data
    print("\nWriting F-order + sharding data...")
    try:
        store = ts.open(spec, create=True, delete_existing=True).result()
        store.write(test_data).result()
        print(f"  Wrote {shape[0]} slices")
    except Exception as e:
        print(f"  ERROR creating/writing store: {e}")
        return False, None

    # Verify zarr.json metadata - transpose should be INSIDE sharding's inner codecs
    zarr_json_path = os.path.join(output_path, "zarr.json")
    with open(zarr_json_path, 'r') as f:
        zarr_metadata = json.load(f)

    codecs = zarr_metadata.get('codecs', [])
    sharding_codec = next((c for c in codecs if c.get('name') == 'sharding_indexed'), None)

    print(f"\nZarr metadata check:")
    print(f"  Has sharding codec: {sharding_codec is not None}")

    if sharding_codec:
        inner_codecs = sharding_codec.get('configuration', {}).get('codecs', [])
        transpose_in_inner = next((c for c in inner_codecs if c.get('name') == 'transpose'), None)
        print(f"  Transpose inside sharding: {transpose_in_inner is not None}")
        if transpose_in_inner:
            order = transpose_in_inner.get('configuration', {}).get('order', [])
            expected_order = list(range(len(shape) - 1, -1, -1))
            print(f"  Transpose order: {order}")
            print(f"  Expected order: {expected_order}")
            if order == expected_order:
                print("  Transpose order: CORRECT")
            else:
                print("  WARNING: Transpose order mismatch!")
    else:
        print("  ERROR: Should have sharding codec!")
        return False, None

    # Read back and verify
    read_store = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': output_path},
        'context': get_tensorstore_context()
    }, read=True).result()

    read_data = read_store.read().result()

    if np.array_equal(test_data, read_data):
        print("  Data verification: PASS")
        return True, read_data
    else:
        print("  Data verification: FAIL")
        diff = np.sum(test_data != read_data)
        print(f"    Different elements: {diff}")
        return False, None


def test_data_equality(c_data, f_data):
    """Test that C-order and F-order produce identical data when read back."""
    print("\n" + "=" * 60)
    print("TEST 3: DATA EQUALITY CHECK")
    print("=" * 60)

    if c_data is None or f_data is None:
        print("  Cannot compare: missing data from previous tests")
        return False

    if np.array_equal(c_data, f_data):
        print("  C-order read == F-order read: PASS")
        print(f"  Both arrays shape: {c_data.shape}")
        print(f"  Both arrays dtype: {c_data.dtype}")
        return True
    else:
        print("  C-order read == F-order read: FAIL")
        print(f"  C-order shape: {c_data.shape}")
        print(f"  F-order shape: {f_data.shape}")
        diff = np.sum(c_data != f_data)
        print(f"  Number of different elements: {diff}")
        return False


def main():
    """Run all F/C order tests with real IMS data."""
    # Test file
    input_path = "/groups/liconn/liconn/FlyLICONN/For_Training/ExPID06B_Brain-1_PFA-AA_epox_18.47.15.ims"
    output_base = "tests/tensorswitch_v2/real_data_output/order_test"

    # Check input exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    print("=" * 60)
    print("TENSORSWITCH V2 - F/C ORDER TEST (REAL DATA)")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output base: {output_base}")

    # Create output directories
    os.makedirs(output_base, exist_ok=True)

    # Load real IMS data subset
    print("\nLoading IMS data...")
    reader = IMSReader(input_path)
    spec = reader.get_tensorstore_spec()
    dask_array = spec['array']

    # Extract a 64-slice subset for testing (keeps test fast)
    test_slices = 64
    test_data = dask_array[:test_slices, :512, :512].compute()
    print(f"Test subset shape: {test_data.shape}")
    print(f"Test subset dtype: {test_data.dtype}")
    print(f"Test subset size: {test_data.nbytes / 1024 / 1024:.1f} MB")

    # Run tests
    results = []

    # Test 1: C-order with real data
    c_pass, c_read_data = test_c_order_real_data(
        test_data,
        os.path.join(output_base, "c_order_real.zarr")
    )
    results.append(("C-order (real data)", c_pass))

    # Test 2: F-order with real data (no sharding)
    f_pass, f_read_data = test_f_order_real_data(
        test_data,
        os.path.join(output_base, "f_order_real.zarr")
    )
    results.append(("F-order (real data)", f_pass))

    # Test 2b: F-order WITH sharding (new test after fix)
    f_shard_pass, f_shard_read_data = test_f_order_with_sharding(
        test_data,
        os.path.join(output_base, "f_order_sharding_real.zarr")
    )
    results.append(("F-order + sharding (real data)", f_shard_pass))

    # Test 3: Data equality - verify both orders read back identical data
    equality_pass = test_data_equality(c_read_data, f_read_data)
    results.append(("Data equality", equality_pass))

    # Test 3b: Data equality - verify F-order with sharding also matches
    equality_shard_pass = test_data_equality(c_read_data, f_shard_read_data)
    results.append(("Data equality (C vs F+shard)", equality_shard_pass))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
