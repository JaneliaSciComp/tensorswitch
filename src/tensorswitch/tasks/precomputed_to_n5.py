import tensorstore as ts
import os
import time
import psutil
from ..utils import (
    fetch_precomputed_info,
    ensure_precomputed_info_decompressed,
    precomputed_store_spec,
    n5_output_spec_with_compression,
    calculate_downsample_factors,
    get_chunk_domains,
    commit_tasks,
    print_processing_info,
    get_total_chunks_from_store
)


def convert(base_path, output_path, level, start_idx=0, stop_idx=None,
            memory_limit=50, custom_chunk_shape=None, **kwargs):
    """
    Convert single scale of Neuroglancer Precomputed to N5 format.

    This task converts a single resolution level from Neuroglancer Precomputed
    format to N5 format with proper attributes.json including downsampling factors.
    It uses chunk-based processing compatible with cluster job submission.

    Args:
        base_path: Path to Precomputed dataset base directory
                   Supports: local paths, gs:// (GCS), http:// (HTTP), s3:// (S3)
                   Examples:
                   - Local: "/groups/tavakoli/data/ExPID124_image"
                   - GCS: "gs://liconn-public/ExPID124/image"
                   - HTTP: "http://server.com/data"

        output_path: Output N5 path for this specific scale
                     Format: /path/to/output.n5/ch0tp0/s{level}
                     Example: "/groups/tavakoli/ExPID124.n5/ch0tp0/s0"

        level: Scale level index to convert (0 = highest resolution)
               Must be valid index in the Precomputed info.json scales array

        start_idx: Starting chunk index for parallel processing (default: 0)
                   Used for cluster job splitting

        stop_idx: Ending chunk index (default: None = all chunks)
                  Used for cluster job splitting

        memory_limit: Memory limit percentage for transaction commits (default: 50)
                     Commits when psutil.virtual_memory().percent exceeds this

        custom_chunk_shape: Override default [128, 128, 128] chunk size
                           Format: list of 3 integers [X, Y, Z]
                           Example: [256, 256, 256]

    Returns:
        None (writes N5 dataset to disk)

    Output Structure:
        output_path/
        ├── attributes.json      # N5 metadata with dimensions, blockSize, etc.
        └── 0/, 1/, 2/, ...      # Data block directories

    Examples:
        >>> # Local conversion of scale 0 (highest resolution)
        >>> convert(
        ...     base_path="/groups/tavakoli/data/ExPID124_image",
        ...     output_path="/groups/tavakoli/ExPID124.n5/ch0tp0/s0",
        ...     level=0
        ... )

        >>> # Convert from GCS with custom chunk size
        >>> convert(
        ...     base_path="gs://liconn-public/ExPID124/image",
        ...     output_path="/local/output.n5/ch0tp0/s1",
        ...     level=1,
        ...     custom_chunk_shape=[256, 256, 256]
        ... )

        >>> # Parallel processing: chunks 0-100
        >>> convert(
        ...     base_path="/groups/tavakoli/data/ExPID124_segmentation",
        ...     output_path="/groups/tavakoli/ExPID124_seg.n5/ch0tp0/s0",
        ...     level=0,
        ...     start_idx=0,
        ...     stop_idx=100
        ... )

    Cluster Submission:
        Use with tensorswitch CLI for automatic job splitting:

        $ python -m tensorswitch --task precomputed_to_n5 \\
            --base_path "gs://liconn-public/ExPID124/image" \\
            --output_path "/groups/tavakoli/ExPID124.n5/ch0tp0/s0" \\
            --level 0 \\
            --num_volumes 8 \\
            --cores 4 \\
            --wall_time 2:00 \\
            --project tavakoli \\
            --submit
    """
    print("=" * 80)
    print("NEUROGLANCER PRECOMPUTED TO N5 CONVERSION")
    print("=" * 80)

    # Step 1: Ensure info file is decompressed (for local paths)
    print(f"\n[1/7] Preparing Precomputed dataset...")
    ensure_precomputed_info_decompressed(base_path)

    # Step 2: Fetch Precomputed metadata
    print(f"\n[2/7] Fetching Precomputed metadata from: {base_path}")
    try:
        info = fetch_precomputed_info(base_path)
    except Exception as e:
        print(f"ERROR: Failed to fetch Precomputed info: {e}")
        raise

    if level >= len(info['scales']):
        raise ValueError(
            f"Scale level {level} not found. "
            f"Dataset has {len(info['scales'])} scales (0-{len(info['scales'])-1})"
        )

    scale = info['scales'][level]
    scale_key = scale['key']
    shape = scale['size']  # [X, Y, Z] in Neuroglancer format
    dtype = info['data_type']  # e.g., 'uint8', 'uint64'

    print(f"\n  Dataset type: {info['@type']}")
    print(f"  Data type: {dtype}")
    print(f"  Total scales: {len(info['scales'])}")
    print(f"\n  Converting scale {level}:")
    print(f"    Key: {scale_key}")
    print(f"    Resolution: {scale['resolution']} nm")
    print(f"    Dimensions: {shape[0]:,} × {shape[1]:,} × {shape[2]:,}")
    print(f"    Encoding: {scale.get('encoding', 'unknown')}")

    # Step 3: Calculate downsampling factors
    print(f"\n[3/7] Calculating downsampling factors...")
    downsample_factors = calculate_downsample_factors(level, info)
    print(f"  Downsampling factors (relative to s0): {downsample_factors}")

    # Step 4: Open Precomputed source
    print(f"\n[4/7] Opening Precomputed source...")
    print(f"  Scale key: {scale_key}")
    try:
        precomputed_spec = precomputed_store_spec(base_path, scale_key)
        source_store = ts.open(precomputed_spec).result()
        print(f"  ✓ Source opened successfully")
        print(f"  Source shape: {source_store.shape}")
        print(f"  Source dtype: {source_store.dtype}")

        # Neuroglancer Precomputed includes channel dimension (X, Y, Z, C)
        # For single-channel data, squeeze the channel dimension to get 3D (X, Y, Z)
        if len(source_store.shape) == 4 and source_store.shape[3] == 1:
            print(f"  Squeezing channel dimension: {source_store.shape} → {source_store.shape[:3]}")
            source_store = source_store[..., 0]  # Remove channel dimension
            print(f"  ✓ Final source shape: {source_store.shape}")

    except Exception as e:
        print(f"ERROR: Failed to open Precomputed source: {e}")
        raise

    # Step 5: Determine chunk size
    if custom_chunk_shape is not None:
        chunk_shape = custom_chunk_shape
        print(f"\n[5/7] Using custom chunk size: {chunk_shape}")
    else:
        chunk_shape = [128, 128, 128]
        print(f"\n[5/7] Using default chunk size: {chunk_shape}")

    # Calculate estimated data size
    import numpy as np
    dtype_bytes = np.dtype(dtype).itemsize
    total_voxels = np.prod(shape)
    uncompressed_gb = (total_voxels * dtype_bytes) / (1024**3)
    print(f"  Uncompressed size: {uncompressed_gb:.2f} GB")

    # Step 6: Create N5 output
    print(f"\n[6/7] Creating N5 output...")
    print(f"  Output path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    try:
        n5_spec = n5_output_spec_with_compression(
            output_path,
            shape,
            dtype,
            chunk_shape=chunk_shape,
            compression_level=3,
            downsample_factors=downsample_factors
        )
        output_store = ts.open(n5_spec, create=True, open=True, delete_existing=False).result()
        print(f"  ✓ N5 dataset created/opened")
        print(f"  Compression: zstd level 3")
        print(f"  Block size: {chunk_shape}")

        # Verify attributes.json was created
        attrs_path = os.path.join(output_path, "attributes.json")
        if os.path.exists(attrs_path):
            print(f"  ✓ attributes.json created")
        else:
            print(f"  WARNING: attributes.json not found at {attrs_path}")

    except Exception as e:
        print(f"ERROR: Failed to create N5 output: {e}")
        raise

    # Step 7: Process chunks with transaction management
    print(f"\n[7/7] Processing data chunks...")
    total_chunks = get_total_chunks_from_store(output_store, chunk_shape=chunk_shape)

    if stop_idx is None:
        stop_idx = total_chunks

    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Processing range: {start_idx:,} to {stop_idx:,} ({stop_idx - start_idx:,} chunks)")

    if stop_idx > total_chunks:
        print(f"  WARNING: stop_idx ({stop_idx}) exceeds total_chunks ({total_chunks})")
        stop_idx = total_chunks

    print_processing_info(level, start_idx, stop_idx, total_chunks)

    # Process chunks with transaction management
    tasks = []
    txn = ts.Transaction()
    linear_indices = range(start_idx, stop_idx)

    start_time = time.time()
    processed_chunks = 0

    print(f"\n  Starting chunk processing...")
    for i, chunk_domain in enumerate(get_chunk_domains(chunk_shape, output_store, linear_indices)):
        # Write chunk
        task = output_store[chunk_domain].with_transaction(txn).write(
            source_store[chunk_domain]
        )
        tasks.append(task)

        # Commit when memory limit reached
        txn = commit_tasks(tasks, txn, memory_limit)
        processed_chunks += 1

        # Progress update every 100 chunks
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            chunks_per_sec = processed_chunks / elapsed if elapsed > 0 else 0
            remaining_chunks = stop_idx - start_idx - processed_chunks
            eta_seconds = remaining_chunks / chunks_per_sec if chunks_per_sec > 0 else 0

            print(f"    Progress: {processed_chunks:,}/{stop_idx - start_idx:,} chunks "
                  f"({100.0 * processed_chunks / (stop_idx - start_idx):.1f}%) | "
                  f"Speed: {chunks_per_sec:.1f} chunks/s | "
                  f"ETA: {eta_seconds/60:.1f} min")

    # Final commit
    if txn.open:
        print(f"\n  Committing final transaction...")
        txn.commit_sync()
        print(f"  ✓ Transaction committed")

    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"CONVERSION COMPLETE")
    print(f"=" * 80)
    print(f"  Source: {base_path} (scale {level})")
    print(f"  Output: {output_path}")
    print(f"  Processed: {processed_chunks:,} chunks in {elapsed_time/60:.2f} minutes")
    print(f"  Speed: {processed_chunks/elapsed_time:.2f} chunks/second")
    print(f"\n✅ Successfully converted Precomputed scale {level} to N5 format")
    print("=" * 80)
