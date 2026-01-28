"""
CLI entry point for tensorswitch_v2 conversion.

Usage:
    python -m tensorswitch_v2 -i input.tif -o output.zarr
    python -m tensorswitch_v2 -i input.n5 -o output.zarr --chunk_shape 32,256,256
    python -m tensorswitch_v2 -i input.tif -o output.zarr --start_idx 0 --stop_idx 100
    python -m tensorswitch_v2 -i input.tif -o output.zarr --submit -P scicompsoft
"""

import os
import sys
import subprocess
import math
import argparse
from typing import Optional, Tuple

import numpy as np

os.umask(0o0002)  # Team permissions: rwxrwxr-x


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="tensorswitch_v2",
        description="TensorSwitch v2: Convert microscopy data between formats.",
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input path (format auto-detected from extension)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path",
    )

    # Output format
    parser.add_argument(
        "--output_format", default="zarr3",
        choices=["zarr3", "zarr2", "n5"],
        help="Output format (default: zarr3)",
    )

    # Dataset paths
    parser.add_argument(
        "--dataset_path", default="",
        help="Path within input container (e.g., 's0' for N5)",
    )
    parser.add_argument(
        "--level_path", default="s0",
        help="Level subdirectory name in output (default: s0)",
    )

    # Chunk/shard configuration
    parser.add_argument(
        "--chunk_shape", default=None,
        help="Comma-separated chunk shape (e.g., '32,256,256')",
    )
    parser.add_argument(
        "--shard_shape", default=None,
        help="Comma-separated shard shape (Zarr3 only, e.g., '64,512,512')",
    )
    parser.add_argument(
        "--no_sharding", action="store_true",
        help="Disable sharding (Zarr3 only)",
    )

    # Compression
    parser.add_argument(
        "--compression", default="zstd",
        help="Compression codec (default: zstd)",
    )
    parser.add_argument(
        "--compression_level", type=int, default=5,
        help="Compression level (default: 5)",
    )

    # Manual bsub chunk-range mode
    parser.add_argument(
        "--start_idx", type=int, default=None,
        help="Starting chunk index (for manual bsub worker)",
    )
    parser.add_argument(
        "--stop_idx", type=int, default=None,
        help="Ending chunk index (for manual bsub worker)",
    )
    parser.add_argument(
        "--write_metadata", action="store_true",
        help="Force metadata write (for manual bsub last job)",
    )

    # CZI-specific
    parser.add_argument(
        "--view_index", type=int, default=None,
        help="CZI view index (None = all views as 5D VCZYX)",
    )

    # Output control
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    # LSF cluster submission
    # NOTE: When adding new conversion arguments above, also add passthrough
    # logic in submit_job() so they are forwarded to the cluster job.
    parser.add_argument(
        "--submit", action="store_true",
        help="Submit as a single LSF bsub job instead of running locally",
    )
    parser.add_argument(
        "--project", "-P", type=str, default=None,
        help="LSF project name (required when --submit is used)",
    )
    parser.add_argument(
        "--memory", type=int, default=None,
        help="Memory in GB for LSF job (default: auto-calculated from source data)",
    )
    parser.add_argument(
        "--wall_time", type=str, default=None,
        help="Wall time for LSF job, format H:MM (default: auto-calculated from source data)",
    )
    parser.add_argument(
        "--cores", type=int, default=None,
        help="Number of cores for LSF job (default: auto-calculated from memory)",
    )
    parser.add_argument(
        "--job_group", type=str, default="/scicompsoft/chend/tensorstore",
        help="LSF job group path (default: /scicompsoft/chend/tensorstore)",
    )

    return parser.parse_args(argv)


def parse_shape(s: str) -> Tuple[int, ...]:
    """Parse a comma-separated shape string into a tuple of ints.

    Args:
        s: Shape string like '32,256,256'

    Returns:
        Tuple of ints, e.g. (32, 256, 256)
    """
    return tuple(int(x.strip()) for x in s.split(","))


def create_reader(args):
    """Create a reader from CLI arguments using auto-detection."""
    from .api.readers import Readers

    path = args.input
    path_lower = path.lower()

    # CZI: use dedicated Tier 2 reader with optional view_index
    if path_lower.endswith(".czi"):
        return Readers.czi(path, view_index=args.view_index)

    # If dataset_path is provided, use explicit reader for formats that need it
    if args.dataset_path:
        if path_lower.endswith(".n5"):
            return Readers.n5(path, dataset_path=args.dataset_path)
        elif path_lower.endswith(".zarr"):
            return Readers.auto_detect(path)
        elif path_lower.endswith((".h5", ".hdf5")):
            return Readers.hdf5(path, dataset_path=args.dataset_path)

    return Readers.auto_detect(path)


def create_writer(args):
    """Create a writer from CLI arguments."""
    from .api.writers import Writers

    fmt = args.output_format

    if fmt == "zarr3":
        return Writers.zarr3(
            output_path=args.output,
            use_sharding=not args.no_sharding,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=args.level_path,
        )
    elif fmt == "zarr2":
        return Writers.zarr2(
            output_path=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            level_path=args.level_path,
        )
    elif fmt == "n5":
        return Writers.n5(
            output_path=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            dataset_path=args.level_path,
        )
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def _get_input_metadata(args):
    """Read input shape and dtype for resource estimation."""
    reader = create_reader(args)
    spec = reader.get_tensorstore_spec()
    if spec.get('driver') == 'array' and 'array' in spec:
        arr = spec['array']
        return tuple(arr.shape), str(arr.dtype)
    else:
        import tensorstore as ts
        from tensorswitch.utils import get_tensorstore_context
        spec['context'] = get_tensorstore_context()
        store = ts.open(spec, read=True).result()
        return tuple(store.shape), store.dtype.name


def _estimate_shard_info(args, volume_shape, dtype_str):
    """Estimate shard shape and total shards for resource calculation.

    Returns:
        (shard_shape, total_shards) tuple. If no sharding, shard_shape is the
        chunk_shape and total_shards is the total number of chunks.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    writer = create_writer(args)

    # Determine chunk shape
    if args.chunk_shape:
        chunk_shape = parse_shape(args.chunk_shape)
    else:
        chunk_shape = writer.get_default_chunk_shape(volume_shape, dtype_size=dtype_bytes)

    # Determine shard shape
    use_sharding = args.output_format == "zarr3" and not args.no_sharding
    if use_sharding:
        if args.shard_shape:
            shard_shape = parse_shape(args.shard_shape)
        else:
            shard_shape = writer.get_default_shard_shape(chunk_shape, volume_shape)
            if shard_shape is None:
                shard_shape = chunk_shape
    else:
        shard_shape = chunk_shape

    total_shards = int(np.prod(np.ceil(np.array(volume_shape) / np.array(shard_shape)).astype(int)))
    return shard_shape, total_shards


def _calculate_memory(volume_shape, dtype_str, shard_shape, total_shards):
    """Calculate memory in GB using v1 formula.

    Formula: base (file loading) + shard buffers + task overhead, then x1.3 safety.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

    # Base memory for file loading
    if dataset_size_gb < 10:
        base_mem = 2
    elif dataset_size_gb < 100:
        base_mem = min(dataset_size_gb * 0.02, 10)
    else:
        base_mem = min(dataset_size_gb * 0.005, 20)

    # Shard processing buffers (up to 3 concurrent shards, 2x for read+write+compress)
    concurrent_shards = min(total_shards, 3)
    shard_buffer_mem = shard_size_gb * 2.0 * concurrent_shards

    # Task overhead (conversion)
    task_overhead = 2

    # Total with 30% safety margin, rounded to nearest 5 GB
    total_mem = base_mem + shard_buffer_mem + task_overhead
    recommended = int(math.ceil(total_mem * 1.3 / 5) * 5)
    return max(5, min(recommended, 500))


def _calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards):
    """Calculate wall time string (H:MM) using v1 formula.

    Formula: (per-shard time x total_shards + overhead) x 2 safety, rounded to 30 min.
    """
    dtype_bytes = np.dtype(dtype_str).itemsize
    dataset_size_gb = (np.prod(volume_shape) * dtype_bytes) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_bytes) / (1024 ** 3)

    # Per-shard time estimate (empirical)
    if shard_size_gb < 0.1:
        minutes_per_shard = 0.5
    elif shard_size_gb < 1.0:
        minutes_per_shard = 2
    else:
        minutes_per_shard = 3

    base_minutes = minutes_per_shard * total_shards

    # Overhead for file loading
    if dataset_size_gb > 1000:
        overhead = 10
    elif dataset_size_gb > 100:
        overhead = 5
    else:
        overhead = 2

    # 2x safety, round to nearest 30 min, cap at 4 hours
    safe_minutes = int(math.ceil((base_minutes + overhead) * 2 / 30) * 30)
    safe_minutes = max(30, safe_minutes)
    hours = min(safe_minutes // 60, 4)
    minutes = safe_minutes % 60
    if hours >= 4:
        return "4:00"
    return f"{hours}:{minutes:02d}"


def submit_job(args):
    """Submit a single LSF bsub job that re-invokes tensorswitch_v2.

    Constructs a bsub command with the same conversion arguments (minus
    --submit and LSF-only flags) so the job runs conversion on a cluster node.
    """
    if not args.project:
        raise ValueError(
            "--project is required when using --submit. "
            "Example: --submit --project scicompsoft"
        )

    # Auto-calculate resources from source data when not explicitly provided
    memory_gb = args.memory
    wall_time = args.wall_time

    if memory_gb is None or wall_time is None:
        print("Reading input metadata for resource estimation...")
        volume_shape, dtype_str = _get_input_metadata(args)
        shard_shape, total_shards = _estimate_shard_info(args, volume_shape, dtype_str)
        print(f"  Shape: {volume_shape}, dtype: {dtype_str}")
        print(f"  Shard shape: {shard_shape}, total shards: {total_shards}")

        if memory_gb is None:
            memory_gb = _calculate_memory(volume_shape, dtype_str, shard_shape, total_shards)
        if wall_time is None:
            wall_time = _calculate_wall_time(volume_shape, dtype_str, shard_shape, total_shards)

    cores = args.cores if args.cores is not None else max(1, int(math.ceil(memory_gb / 15)) * 2)

    # Job name: tsv2_{src_ext}_to_{out_format}_{input_stem}
    input_name = os.path.basename(args.input)
    input_stem = os.path.splitext(input_name)[0]
    input_ext = os.path.splitext(input_name)[1].lstrip(".")
    job_name = f"tsv2_{input_ext}_to_{args.output_format}_{input_stem}"
    job_name = job_name.replace(" ", "_")[:128]

    # Log directory next to output
    output_parent = os.path.dirname(os.path.abspath(args.output))
    log_dir = os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
    error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

    # Build re-invocation command (conversion args only, no LSF-only args)
    reinvoke = [
        sys.executable, "-m", "tensorswitch_v2",
        "--input", args.input,
        "--output", args.output,
        "--output_format", args.output_format,
    ]
    if args.dataset_path:
        reinvoke += ["--dataset_path", args.dataset_path]
    if args.level_path != "s0":
        reinvoke += ["--level_path", args.level_path]
    if args.chunk_shape:
        reinvoke += ["--chunk_shape", args.chunk_shape]
    if args.shard_shape:
        reinvoke += ["--shard_shape", args.shard_shape]
    if args.no_sharding:
        reinvoke.append("--no_sharding")
    if args.compression != "zstd":
        reinvoke += ["--compression", args.compression]
    if args.compression_level != 5:
        reinvoke += ["--compression_level", str(args.compression_level)]
    if args.start_idx is not None:
        reinvoke += ["--start_idx", str(args.start_idx)]
    if args.stop_idx is not None:
        reinvoke += ["--stop_idx", str(args.stop_idx)]
    if args.write_metadata:
        reinvoke.append("--write_metadata")
    if args.view_index is not None:
        reinvoke += ["--view_index", str(args.view_index)]
    if args.quiet:
        reinvoke.append("--quiet")

    # Build bsub command
    command = [
        "bsub",
        "-J", job_name,
        "-n", str(cores),
        "-W", wall_time,
        "-M", f"{memory_gb}GB",
        "-R", f"rusage[mem={memory_gb * 1024 * 1024 * 1024}]",
        "-P", args.project,
        "-g", args.job_group,
        "-o", log_path,
        "-e", error_path,
    ] + reinvoke

    # Print summary and submit
    print("=" * 72)
    print("LSF Single-Job Submission")
    print("=" * 72)
    print(f"  Job name:  {job_name}")
    print(f"  Cores:     {cores}")
    print(f"  Memory:    {memory_gb} GB")
    print(f"  Wall time: {wall_time}")
    print(f"  Project:   {args.project}")
    print(f"  Job group: {args.job_group}")
    print(f"  Log:       {log_path}")
    print(f"  Error:     {error_path}")
    print(f"  Command:   {' '.join(reinvoke)}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Job submitted successfully.")
        print(result.stdout.strip())
    else:
        print(f"Job submission failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"bsub failed with exit code {result.returncode}")


def main(argv=None):
    """Run single-process conversion or submit LSF job."""
    args = parse_args(argv)

    if args.submit:
        submit_job(args)
        return

    reader = create_reader(args)
    writer = create_writer(args)

    from .core.converter import DistributedConverter

    converter = DistributedConverter(reader, writer)

    # Parse optional shapes
    chunk_shape = parse_shape(args.chunk_shape) if args.chunk_shape else None
    shard_shape = parse_shape(args.shard_shape) if args.shard_shape else None

    verbose = not args.quiet

    if args.start_idx is not None:
        # Manual chunk-range mode (for bsub workers)
        converter.convert(
            start_idx=args.start_idx,
            stop_idx=args.stop_idx,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            write_metadata=args.write_metadata,
            delete_existing=False,
            verbose=verbose,
        )
    else:
        # Full single-process conversion
        converter.convert(
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            write_metadata=True,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
