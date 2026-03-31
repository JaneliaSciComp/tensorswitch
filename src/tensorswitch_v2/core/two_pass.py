"""
Two-pass conversion for DaskReader sources with large native chunks.

When DaskReader source chunks are much larger than output chunks (e.g., ND2 with
6 GB full-frame dask chunks vs 4 MB output tiles), the virtual_chunked pipeline
re-reads the same huge source chunk for every small output tile.

Solution:
- Pass 1: DaskReader -> intermediate zarr2 (source-aligned chunks, sequential I/O)
- Pass 2: Zarr2Reader (Tier 1) -> final output (native TensorStore, fast)

Inspired by pangeo's rechunker which uses the same two-pass intermediate approach.
"""

import os
import sys
import re
import math
import time
import shutil
import shlex
import subprocess
from typing import Dict, Optional, Tuple, Any

import numpy as np

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

# Thresholds for auto-detection
_SOURCE_CHUNK_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
_RATIO_THRESHOLD = 100


def should_use_two_pass(reader, chunk_shape, shard_shape, output_format, no_sharding=False):
    """Detect if two-pass conversion is needed for this reader.

    Triggers when source dask chunks are much larger than output chunks,
    which would cause massive redundant re-reads via virtual_chunked.

    Args:
        reader: A BaseReader instance
        chunk_shape: Target chunk shape tuple or None
        shard_shape: Target shard shape tuple or None
        output_format: 'zarr2', 'zarr3', or 'n5'
        no_sharding: If True, zarr3 uses non-sharded mode

    Returns:
        (bool, str or None): (should_use_two_pass, reason_message)
    """
    from ..readers.base import DaskReader

    if not isinstance(reader, DaskReader):
        return False, None

    # Load dask array to get native chunk shape
    reader._load()
    dask_array = reader._dask_array
    native_chunk_shape = tuple(int(c[0]) for c in dask_array.chunks)
    dtype_bytes = dask_array.dtype.itemsize

    source_chunk_bytes = int(np.prod(native_chunk_shape)) * dtype_bytes

    # Determine target chunk size
    if output_format == 'zarr3' and not no_sharding:
        target_shape = shard_shape or tuple(min(s, 1024) for s in dask_array.shape)
    else:
        target_shape = chunk_shape or tuple(min(s, 64) for s in dask_array.shape)

    target_chunk_bytes = int(np.prod(target_shape)) * dtype_bytes

    ratio = source_chunk_bytes / max(target_chunk_bytes, 1)

    if source_chunk_bytes > _SOURCE_CHUNK_THRESHOLD_BYTES:
        reason = (
            f"Source chunk size ({source_chunk_bytes / (1024**3):.1f} GB) exceeds "
            f"500 MB threshold (native dask chunks: {native_chunk_shape}). "
            f"Two-pass conversion avoids redundant re-reads."
        )
        return True, reason

    if ratio > _RATIO_THRESHOLD:
        reason = (
            f"Source/target chunk ratio is {ratio:.0f}x (>{_RATIO_THRESHOLD}x). "
            f"Two-pass conversion avoids {ratio:.0f}x redundant re-reads per source chunk."
        )
        return True, reason

    return False, None


def compute_intermediate_path(output_path, input_path):
    """Compute path for the ephemeral intermediate zarr2 store.

    Places intermediate in same parent directory as output (same filesystem).
    Uses hidden dot-prefix and timestamp for uniqueness.
    """
    output_parent = os.path.dirname(os.path.abspath(output_path))
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    # Sanitize stem for filesystem
    input_stem = input_stem.replace(" ", "_")[:80]
    timestamp = int(time.time())
    return os.path.join(output_parent,
                        f".tensorswitch_intermediate_{input_stem}_{timestamp}.zarr")


def compute_intermediate_chunk_shape(reader):
    """Get source-aligned chunk shape from DaskReader's dask array.

    Each intermediate chunk = one dask partition = one source read.
    This ensures Pass 1 reads each source chunk exactly once.
    """
    reader._load()
    return tuple(int(c[0]) for c in reader._dask_array.chunks)


def run_two_pass_local(
    reader,
    writer,
    args,
    chunk_shape,
    shard_shape,
    verbose=True,
    force_order=None,
    voxel_size_override=None,
    voxel_unit=None,
    is_label=False,
    expand_to_5d=False,
    bbox=None,
):
    """Run two-pass conversion locally (no LSF).

    Pass 1: reader -> intermediate zarr2 (source-aligned chunks, fast sequential I/O)
    Pass 2: intermediate zarr2 -> final output (Tier 1 native TensorStore, any target chunks)

    Returns:
        dict: Combined stats from both passes
    """
    from .converter import DistributedConverter
    from ..readers.zarr import Zarr2Reader
    from ..writers.zarr2 import Zarr2Writer

    intermediate_path = compute_intermediate_path(args.output, args.input)
    intermediate_chunk_shape = compute_intermediate_chunk_shape(reader)

    reader._load()
    source_dtype = str(reader._dask_array.dtype)

    # Extract voxel sizes from original reader BEFORE Pass 1 (needed for Pass 2 metadata)
    if voxel_size_override is None:
        try:
            original_voxel_sizes = reader.get_voxel_sizes()
            # Check if these are real voxel sizes (not defaults)
            if original_voxel_sizes and not all(v == 1.0 for v in original_voxel_sizes.values()):
                voxel_size_override = original_voxel_sizes
        except Exception:
            pass

    if verbose:
        source_chunk_bytes = int(np.prod(intermediate_chunk_shape)) * np.dtype(source_dtype).itemsize
        print("=" * 72)
        print("TWO-PASS CONVERSION")
        print("=" * 72)
        print(f"  Source:            {args.input}")
        print(f"  Intermediate:      {intermediate_path}")
        print(f"  Output:            {args.output}")
        print(f"  Source chunks:     {intermediate_chunk_shape} ({source_chunk_bytes / (1024**2):.1f} MB each)")
        print(f"  Target chunks:     {chunk_shape or 'auto'}")
        if shard_shape:
            print(f"  Target shards:     {shard_shape}")
        print()

    stats_pass1 = None
    stats_pass2 = None

    try:
        # ===== PASS 1: DaskReader -> Intermediate Zarr2 =====
        if verbose:
            print("-" * 72)
            print("PASS 1: Source -> Intermediate Zarr2 (source-aligned chunks)")
            print("-" * 72)

        intermediate_writer = Zarr2Writer(
            output_path=intermediate_path,
            compression="zstd",
            compression_level=3,
            level_path="data",
            use_nested_structure=False,
        )

        converter_pass1 = DistributedConverter(reader, intermediate_writer)
        stats_pass1 = converter_pass1.convert(
            chunk_shape=intermediate_chunk_shape,
            shard_shape=None,
            write_metadata=True,
            skip_voxel_validation=True,
            verbose=verbose,
            expand_to_5d=False,
            bbox=bbox,
            voxel_size_override=voxel_size_override,
            voxel_unit=voxel_unit,
        )

        if verbose:
            print(f"\nPass 1 complete: {stats_pass1['chunks_processed']} chunks "
                  f"in {stats_pass1['elapsed_seconds']:.1f}s")

        # ===== PASS 2: Intermediate Zarr2 -> Final Output =====
        if verbose:
            print()
            print("-" * 72)
            print("PASS 2: Intermediate Zarr2 -> Final Output (target chunks)")
            print("-" * 72)

        intermediate_reader = Zarr2Reader(
            path=os.path.join(intermediate_path, "data"),
        )

        converter_pass2 = DistributedConverter(intermediate_reader, writer)
        stats_pass2 = converter_pass2.convert(
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            write_metadata=True,
            verbose=verbose,
            force_order=force_order,
            voxel_size_override=voxel_size_override,
            voxel_unit=voxel_unit,
            is_label=is_label,
            expand_to_5d=expand_to_5d,
            # bbox already applied in Pass 1; intermediate is already cropped
        )

        if verbose:
            print(f"\nPass 2 complete: {stats_pass2['chunks_processed']} chunks "
                  f"in {stats_pass2['elapsed_seconds']:.1f}s")

    except Exception:
        if os.path.exists(intermediate_path):
            print(f"\nWARNING: Intermediate preserved for debugging: {intermediate_path}")
        raise

    # Cleanup intermediate
    if os.path.exists(intermediate_path):
        if verbose:
            print(f"\nCleaning up intermediate: {intermediate_path}")
        shutil.rmtree(intermediate_path, ignore_errors=True)

    total_elapsed = (stats_pass1['elapsed_seconds'] if stats_pass1 else 0) + \
                    (stats_pass2['elapsed_seconds'] if stats_pass2 else 0)
    if verbose:
        print()
        print("=" * 72)
        print("TWO-PASS CONVERSION COMPLETE")
        print(f"  Pass 1: {stats_pass1['elapsed_seconds']:.1f}s" if stats_pass1 else "  Pass 1: N/A")
        print(f"  Pass 2: {stats_pass2['elapsed_seconds']:.1f}s" if stats_pass2 else "  Pass 2: N/A")
        print(f"  Total:  {total_elapsed:.1f}s")
        print("=" * 72)

    return {
        'pass1_stats': stats_pass1,
        'pass2_stats': stats_pass2,
        'total_elapsed': total_elapsed,
    }


# ============================================================================
# LSF Two-Pass Submission
# ============================================================================


def _calculate_pass1_resources(volume_shape, dtype_str, source_chunk_shape):
    """Calculate LSF resources for Pass 1 (file-decoded -> intermediate zarr2).

    Returns:
        (memory_gb, wall_time_str, cores)
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024**3)
    source_chunk_gb = (np.prod(source_chunk_shape) * dtype_bytes) / (1024**3)

    # Memory: source chunk read buffer + zarr2 write buffer + overhead
    base_mem = source_chunk_gb * 2 + 4
    memory_gb = int(math.ceil(base_mem * 1.5 / 5) * 5)
    memory_gb = max(memory_gb, 15)

    # Cores: I/O-bound, some parallelism for compression
    cores = 4

    # Enforce cluster policy: 15 GB per core minimum
    memory_gb = max(memory_gb, cores * 15)

    # Wall time: dataset_size / file-decoded throughput (10 MB/s) with 2x safety
    throughput_mb_s = 10
    base_minutes = (dataset_size_gb * 1024) / throughput_mb_s / 60
    overhead = 10 if dataset_size_gb > 100 else 5
    safe_minutes = int(math.ceil((base_minutes + overhead) * 2 / 30) * 30)
    safe_minutes = max(30, min(safe_minutes, 96 * 60))
    hours = safe_minutes // 60
    minutes = safe_minutes % 60
    wall_time = f"{hours}:{minutes:02d}"

    return memory_gb, wall_time, cores


def _calculate_pass2_resources(volume_shape, dtype_str, chunk_shape, shard_shape,
                                output_format, no_sharding):
    """Calculate LSF resources for Pass 2 (native zarr2 -> final output).

    Delegates to existing resource_utils with is_native_source=True.
    """
    from ..utils.resource_utils import calculate_memory, calculate_wall_time, estimate_shard_info

    # Estimate shard/chunk info for resource calculation
    shard_info_shape, total_shards = estimate_shard_info(
        volume_shape, dtype_str, output_format,
        shard_shape=shard_shape, chunk_shape=chunk_shape,
        no_sharding=no_sharding,
    )

    memory_gb = calculate_memory(
        volume_shape, dtype_str, shard_info_shape, total_shards,
        output_format=output_format,
    )

    # Cores
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024**3)
    use_sharding = output_format == 'zarr3' and not no_sharding
    if use_sharding and total_shards == 1:
        cores = 1
    elif not use_sharding:
        cores = min(8, max(4, int(math.ceil(dataset_size_gb / 25)) * 2))
    else:
        cores = min(8, max(1, int(math.ceil(memory_gb / 15)) * 2))

    wall_time = calculate_wall_time(
        volume_shape, dtype_str, shard_info_shape, total_shards,
        output_format=output_format, no_sharding=no_sharding,
        cores=cores, is_native_source=True,
    )

    # Enforce cluster policy
    memory_gb = max(memory_gb, cores * 15)

    return memory_gb, wall_time, cores


def submit_two_pass_job(args, reader):
    """Submit two-pass conversion as chained LSF jobs via coordinator script.

    Generates a bash coordinator script that:
    1. Submits Pass 1 (source -> intermediate zarr2)
    2. Waits for Pass 1 via bwait
    3. Submits Pass 2 (intermediate zarr2 -> final output)
    4. Waits for Pass 2 via bwait
    5. Cleans up intermediate

    Returns:
        Coordinator job ID string
    """
    reader._load()
    source_shape = tuple(reader._dask_array.shape)
    source_dtype = str(reader._dask_array.dtype)
    intermediate_chunk_shape = compute_intermediate_chunk_shape(reader)
    intermediate_path = compute_intermediate_path(args.output, args.input)

    # Extract voxel sizes from original reader for Pass 2
    voxel_size_str = None
    voxel_unit = getattr(args, 'voxel_unit', None)
    if getattr(args, 'voxel_size', None):
        # User provided explicit voxel sizes — pass through
        voxel_size_str = args.voxel_size
    else:
        try:
            vs = reader.get_voxel_sizes()
            if vs and not all(v == 1.0 for v in vs.values()):
                voxel_size_str = f"{vs['x']},{vs['y']},{vs['z']}"
                if voxel_unit is None:
                    voxel_unit = 'nanometer'
        except Exception:
            pass

    # Calculate resources for each pass
    pass1_mem, pass1_wall, pass1_cores = _calculate_pass1_resources(
        source_shape, source_dtype, intermediate_chunk_shape
    )

    # Pass 2 shape must match the output dimension order (after Zarr2Writer reordering).
    # Zarr2Writer reorders non-spatial dims before spatial (e.g., Z,C,Y,X -> C,Z,Y,X).
    # Compute the reordered shape so resource estimation uses correct chunk counts.
    chunk_shape_tuple = tuple(int(x) for x in args.chunk_shape.split(',')) if args.chunk_shape else None
    shard_shape_tuple = tuple(int(x) for x in args.shard_shape.split(',')) if getattr(args, 'shard_shape', None) else None

    pass2_shape = source_shape
    if args.output_format in ('zarr2', 'zarr3', 'n5'):
        # Compute the reorder that Zarr2Writer._reorder_axes_for_ome would apply
        try:
            dim_names = reader._get_dimension_names()
            if dim_names:
                SPATIAL = {'x', 'y', 'z'}
                non_spatial_idx = [i for i, d in enumerate(dim_names) if d.lower() not in SPATIAL]
                spatial_idx = [i for i, d in enumerate(dim_names) if d.lower() in SPATIAL]
                reorder = non_spatial_idx + spatial_idx
                if reorder != list(range(len(source_shape))):
                    pass2_shape = tuple(source_shape[i] for i in reorder)
        except Exception:
            pass

    pass2_mem, pass2_wall, pass2_cores = _calculate_pass2_resources(
        pass2_shape, source_dtype,
        chunk_shape=chunk_shape_tuple,
        shard_shape=shard_shape_tuple,
        output_format=args.output_format,
        no_sharding=args.no_sharding,
    )

    # Generate coordinator script
    script = _generate_coordinator_script(
        args=args,
        intermediate_path=intermediate_path,
        intermediate_chunk_shape=intermediate_chunk_shape,
        pass1_resources=(pass1_mem, pass1_wall, pass1_cores),
        pass2_resources=(pass2_mem, pass2_wall, pass2_cores),
        voxel_size_str=voxel_size_str,
        voxel_unit=voxel_unit,
    )

    # Write script to file
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    input_stem = os.path.splitext(os.path.basename(args.input))[0].replace(" ", "_")[:80]
    script_path = os.path.join(log_dir, f"two_pass_coordinator_{input_stem}.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    # Calculate coordinator wall time (sum of both passes + overhead)
    def _parse_wall_time(wt):
        parts = wt.split(':')
        return int(parts[0]) * 60 + int(parts[1])

    total_minutes = _parse_wall_time(pass1_wall) + _parse_wall_time(pass2_wall) + 60
    coord_wall_hours = min((total_minutes + 59) // 60, 48)
    coord_wall_time = f"{coord_wall_hours}:00"

    # Submit coordinator
    job_name = f"tsv2_two_pass_{input_stem}"[:128]
    log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
    error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

    bsub_cmd = [
        "bsub",
        "-J", job_name,
        "-n", "1",
        "-W", coord_wall_time,
        "-M", "8GB",
        "-R", "rusage[mem=8192]",
        "-P", args.project,
        "-o", log_path,
        "-e", error_path,
        "/bin/bash", script_path,
    ]

    print("=" * 72)
    print("TWO-PASS CONVERSION (LSF)")
    print("=" * 72)
    print(f"  Source:            {args.input}")
    print(f"  Intermediate:      {intermediate_path}")
    print(f"  Output:            {args.output}")
    print(f"  Source chunks:     {intermediate_chunk_shape}")
    print(f"  Pass 1: mem={pass1_mem}GB, wall={pass1_wall}, cores={pass1_cores}")
    print(f"  Pass 2: mem={pass2_mem}GB, wall={pass2_wall}, cores={pass2_cores}")
    print(f"  Coordinator wall:  {coord_wall_time}")
    print(f"  Script:            {script_path}")
    print("=" * 72)

    result = subprocess.run(bsub_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        raise RuntimeError(f"bsub failed: {result.stderr}")

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        coord_job_id = match.group(1)
        print(f"Coordinator job submitted: {coord_job_id}")
        print(result.stdout.strip())
        print(f"\nMonitor with: bjobs {coord_job_id}")
        print(f"View log:     tail -f {log_path.replace('%J', coord_job_id)}")
        return coord_job_id

    raise RuntimeError(f"Could not extract job ID from: {result.stdout}")


def _generate_coordinator_script(
    args,
    intermediate_path,
    intermediate_chunk_shape,
    pass1_resources,
    pass2_resources,
    voxel_size_str,
    voxel_unit,
):
    """Generate bash coordinator script for two-pass LSF submission."""
    python_path = sys.executable
    tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    q_python = shlex.quote(python_path)
    q_tsdir = shlex.quote(tensorswitch_dir)
    q_input = shlex.quote(args.input)
    q_intermediate = shlex.quote(intermediate_path)
    q_output = shlex.quote(args.output)

    pass1_mem, pass1_wall, pass1_cores = pass1_resources
    pass2_mem, pass2_wall, pass2_cores = pass2_resources

    intermediate_chunk_str = ",".join(map(str, intermediate_chunk_shape))

    # Build Pass 2 flags
    pass2_flags = []
    pass2_flags += ["--output_format", args.output_format]
    if args.chunk_shape:
        pass2_flags += ["--chunk_shape", args.chunk_shape]
    if args.shard_shape:
        pass2_flags += ["--shard_shape", args.shard_shape]
    if args.no_sharding:
        pass2_flags.append("--no_sharding")
    if args.compression != "zstd":
        pass2_flags += ["--compression", args.compression]
    if args.compression_level != 5:
        pass2_flags += ["--compression_level", str(args.compression_level)]
    if voxel_size_str:
        pass2_flags += ["--voxel_size", voxel_size_str]
    if voxel_unit:
        pass2_flags += ["--voxel_unit", voxel_unit]
    if getattr(args, 'is_label', False):
        pass2_flags.append("--is-label")
    if getattr(args, 'expand_to_5d', False):
        pass2_flags.append("--expand-to-5d")
    if getattr(args, 'data_type', 'auto') != 'auto':
        pass2_flags += ["--data-type", args.data_type]
    if getattr(args, 'force_c_order', False):
        pass2_flags.append("--force_c_order")
    if getattr(args, 'force_f_order', False):
        pass2_flags.append("--force_f_order")
    if getattr(args, 'use_nested_structure', True) is False:
        pass2_flags.append("--no-nested-structure")

    pass2_flags_str = " ".join(shlex.quote(f) for f in pass2_flags)

    # Build Pass 1 flags (minimal)
    # Pass 1 extra flags: voxel info (for OME metadata on intermediate) and bbox
    pass1_extra = ""
    if voxel_size_str:
        pass1_extra += f" --voxel_size {shlex.quote(voxel_size_str)}"
    if voxel_unit:
        pass1_extra += f" --voxel_unit {shlex.quote(voxel_unit)}"
    if getattr(args, 'bbox', None):
        pass1_extra += f" --bbox {shlex.quote(args.bbox)}"

    script = f"""#!/bin/bash
set -e

echo "========================================================================"
echo "TWO-PASS CONVERSION"
echo "========================================================================"
echo "Source:       {args.input}"
echo "Intermediate: {intermediate_path}"
echo "Output:       {args.output}"
echo "Started:      $(date)"
echo ""

# Helper to extract job IDs from bsub output
extract_job_ids() {{
    grep -oP 'Job <\\K[0-9]+(?=>)' || true
}}

# ===== PASS 1: Source -> Intermediate Zarr2 =====
echo "--------------------------------------------------------------------"
echo "PASS 1: Source -> Intermediate Zarr2 (source-aligned chunks)"
echo "--------------------------------------------------------------------"
echo "  Chunk shape: {intermediate_chunk_str}"
echo "  Memory: {pass1_mem}GB, Wall time: {pass1_wall}, Cores: {pass1_cores}"
echo "  Submitting..."

PASS1_OUTPUT=$(cd {q_tsdir} && {q_python} -m tensorswitch_v2 \\
    -i {q_input} \\
    -o {q_intermediate} \\
    --output_format zarr2 \\
    --chunk_shape {intermediate_chunk_str} \\
    --level_path data \\
    --no-nested-structure \\
    --no_two_pass \\
    --submit \\
    -P {args.project} \\
    --memory {pass1_mem} \\
    --wall_time {pass1_wall} \\
    --cores {pass1_cores}{pass1_extra} \\
    2>&1)

PASS1_JOB=$(echo "$PASS1_OUTPUT" | extract_job_ids | head -1)

if [ -z "$PASS1_JOB" ]; then
    echo "ERROR: No job ID returned for Pass 1"
    echo "Output was: $PASS1_OUTPUT"
    exit 1
fi

echo "  Pass 1 job: $PASS1_JOB"
echo "  Waiting for Pass 1..."
bwait -w "done($PASS1_JOB)" 2>&1 || true

# Verify Pass 1 completed successfully
sleep 5
for attempt in 1 2 3 4 5; do
    PASS1_STAT=$(bjobs -noheader -o "stat" $PASS1_JOB 2>/dev/null | tr -d ' ')
    if [ "$PASS1_STAT" = "DONE" ] || [ "$PASS1_STAT" = "EXIT" ]; then
        break
    fi
    sleep 5
done

if [ "$PASS1_STAT" = "EXIT" ]; then
    echo ""
    echo "ERROR: Pass 1 FAILED (EXIT status). Cleaning up intermediate."
    echo "Check error log: bjobs -l $PASS1_JOB"
    rm -rf {q_intermediate}
    exit 1
elif [ "$PASS1_STAT" != "DONE" ]; then
    echo ""
    echo "WARNING: Pass 1 unexpected status after retries: $PASS1_STAT"
    echo "Cleaning up intermediate and aborting."
    rm -rf {q_intermediate}
    exit 1
fi

echo "  Pass 1 complete (verified DONE): $(date)"

# ===== PASS 2: Intermediate Zarr2 -> Final Output =====
echo ""
echo "--------------------------------------------------------------------"
echo "PASS 2: Intermediate Zarr2 -> Final Output (target chunks)"
echo "--------------------------------------------------------------------"
echo "  Memory: {pass2_mem}GB, Wall time: {pass2_wall}, Cores: {pass2_cores}"
echo "  Submitting..."

PASS2_OUTPUT=$(cd {q_tsdir} && {q_python} -m tensorswitch_v2 \\
    -i {q_intermediate}/data \\
    -o {q_output} \\
    {pass2_flags_str} \\
    --no_two_pass \\
    --submit \\
    -P {args.project} \\
    --memory {pass2_mem} \\
    --wall_time {pass2_wall} \\
    --cores {pass2_cores} \\
    2>&1)

PASS2_JOB=$(echo "$PASS2_OUTPUT" | extract_job_ids | head -1)

if [ -z "$PASS2_JOB" ]; then
    echo "ERROR: No job ID returned for Pass 2"
    echo "Output was: $PASS2_OUTPUT"
    echo "Intermediate preserved for debugging: {intermediate_path}"
    exit 1
fi

echo "  Pass 2 job: $PASS2_JOB"
echo "  Waiting for Pass 2..."
bwait -w "done($PASS2_JOB)" 2>&1 || true

# Verify Pass 2
sleep 5
for attempt in 1 2 3 4 5; do
    PASS2_STAT=$(bjobs -noheader -o "stat" $PASS2_JOB 2>/dev/null | tr -d ' ')
    if [ "$PASS2_STAT" = "DONE" ] || [ "$PASS2_STAT" = "EXIT" ]; then
        break
    fi
    sleep 5
done

if [ "$PASS2_STAT" = "EXIT" ]; then
    echo ""
    echo "ERROR: Pass 2 FAILED (EXIT status)."
    echo "Intermediate preserved for debugging: {intermediate_path}"
    echo "Check error log: bjobs -l $PASS2_JOB"
    exit 1
elif [ "$PASS2_STAT" != "DONE" ]; then
    echo ""
    echo "WARNING: Pass 2 unexpected status after retries: $PASS2_STAT"
    echo "Intermediate preserved for debugging: {intermediate_path}"
    exit 1
fi

echo "  Pass 2 complete (verified DONE): $(date)"

# ===== CLEANUP =====
echo ""
echo "Cleaning up intermediate..."
rm -rf {q_intermediate}

echo ""
echo "========================================================================"
echo "TWO-PASS CONVERSION COMPLETE"
echo "========================================================================"
echo "Output: {args.output}"
echo "Completed: $(date)"
"""

    return script
