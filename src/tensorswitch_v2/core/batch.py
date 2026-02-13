"""
Batch processing module for tensorswitch_v2.

Handles batch conversion of multiple files with LSF job array support.
Supports both traditional file-based batch processing and folder-based
dataset discovery (for directories containing image and segmentation datasets).

Usage:
    # Batch convert directory of files (LSF)
    python -m tensorswitch_v2 -i /path/to/tiffs/ -o /path/to/output/ --submit -P project

    # Convert folder with auto-detected image and segmentation
    python -m tensorswitch_v2 -i /path/to/folder/ -o output.zarr

    # Convert only image or labels from discovered folder
    python -m tensorswitch_v2 -i /path/to/folder/ -o output.zarr --image-only
    python -m tensorswitch_v2 -i /path/to/folder/ -o output.zarr --labels-only

    # Check status
    python -m tensorswitch_v2 -i /path/to/dir/ -o /path/to/output/ --status
"""

import os
import sys
import glob
import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)


# Supported input format extensions
SUPPORTED_EXTENSIONS = {
    '.tif', '.tiff',  # TIFF
    '.nd2',           # Nikon ND2
    '.czi',           # Zeiss CZI
    '.ims',           # Imaris HDF5
    '.h5', '.hdf5',   # Generic HDF5
    '.n5',            # N5
    '.zarr',          # Zarr (for re-conversion)
}

# Output format extensions
OUTPUT_EXTENSIONS = {
    'zarr3': '.zarr',
    'zarr2': '.zarr',
    'n5': '.n5',
}


def detect_input_mode(input_path: str) -> str:
    """
    Determine if input is single file, batch directory, or discovered folder.

    Args:
        input_path: Input path from CLI

    Returns:
        'single_file': Single file or single dataset (e.g., one Precomputed dir)
        'batch_directory': Directory with multiple files to batch process
        'discovered_folder': Directory with discoverable datasets (image/segmentation)
    """
    from ..readers.base import is_local_precomputed
    from ..utils.folder_discovery import discover_datasets, is_neuroglancer_precomputed

    # Check if path ends with known format extension
    ext = os.path.splitext(input_path)[1].lower()

    if ext in SUPPORTED_EXTENSIONS:
        return 'single_file'
    elif is_local_precomputed(input_path):
        # Local precomputed directory (has info file) - treat as single file
        return 'single_file'
    elif os.path.isdir(input_path) or input_path.endswith('/'):
        # Check if directory contains discoverable datasets (e.g., image + segmentation)
        # This takes priority over batch_directory mode
        result = discover_datasets(input_path, verbose=False)
        if result.has_image or result.has_segmentation:
            return 'discovered_folder'
        return 'batch_directory'
    else:
        # Could be a path that doesn't exist yet or ambiguous
        # Check if parent exists and path looks like a file
        if os.path.exists(os.path.dirname(input_path)):
            return 'single_file'
        raise ValueError(
            f"Cannot determine input mode for: {input_path}\n"
            f"For single file, use path with extension (e.g., .tif, .nd2)\n"
            f"For batch mode, use directory path ending with /"
        )


def discover_files(
    input_dir: str,
    pattern: str = "*",
    recursive: bool = True,
) -> List[str]:
    """
    Discover files matching pattern in input directory.

    Args:
        input_dir: Input directory path
        pattern: Glob pattern (e.g., "*.tif", "scan_*.tif")
        recursive: Search subdirectories

    Returns:
        Sorted list of absolute file paths
    """
    input_dir = os.path.abspath(input_dir)

    if recursive:
        # Use ** for recursive search
        search_pattern = os.path.join(input_dir, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)

    # Filter to only supported formats
    supported_files = []
    for f in files:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                supported_files.append(f)

    return sorted(supported_files)


def generate_output_path(
    input_file: str,
    input_dir: str,
    output_dir: str,
    output_format: str = "zarr3",
) -> str:
    """
    Generate output path preserving subdirectory structure.

    Args:
        input_file: Absolute path to input file
        input_dir: Base input directory
        output_dir: Base output directory
        output_format: Output format (zarr3, zarr2, n5)

    Returns:
        Output path with preserved subdirectory structure

    Example:
        input_file:  /data/input/subdir/tile0.tif
        input_dir:   /data/input/
        output_dir:  /data/output/
        → /data/output/subdir/tile0.zarr
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    input_file = os.path.abspath(input_file)

    # Get relative path from input_dir
    rel_path = os.path.relpath(input_file, input_dir)

    # Change extension
    base_name = os.path.splitext(rel_path)[0]
    out_ext = OUTPUT_EXTENSIONS.get(output_format, '.zarr')

    # Combine with output_dir
    output_path = os.path.join(output_dir, base_name + out_ext)

    return output_path


@dataclass
class BatchResult:
    """Result of batch operation."""
    total: int = 0
    submitted: int = 0
    skipped: int = 0
    failed: int = 0
    completed: int = 0
    in_progress: int = 0
    job_ids: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)  # (file, reason)


@dataclass
class BatchFileInfo:
    """Information about a file in batch."""
    index: int
    input_path: str
    output_path: str
    status: str = "pending"  # pending, completed, failed, in_progress


class BatchConverter:
    """
    Batch converter for processing multiple files.

    Supports:
    - LSF job array submission
    - Local sequential processing
    - Status checking

    Example:
        batch = BatchConverter(
            input_dir="/data/tiffs/",
            output_dir="/data/zarr/",
            pattern="*.tif",
            output_format="zarr3",
        )
        files = batch.discover()
        result = batch.submit_lsf(project="myproject")
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.tif",
        output_format: str = "zarr3",
        recursive: bool = False,
    ):
        """
        Initialize batch converter.

        Args:
            input_dir: Input directory containing files
            output_dir: Output directory for converted files
            pattern: Glob pattern for file matching
            output_format: Output format (zarr3, zarr2, n5)
            recursive: Search subdirectories
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.pattern = pattern
        self.output_format = output_format
        self.recursive = recursive

        self._files: Optional[List[BatchFileInfo]] = None
        self._index_file: Optional[str] = None

    def discover(self) -> List[BatchFileInfo]:
        """
        Discover files to process.

        Returns:
            List of BatchFileInfo with input/output paths
        """
        input_files = discover_files(
            self.input_dir,
            self.pattern,
            self.recursive,
        )

        self._files = []
        for i, input_file in enumerate(input_files, start=1):
            output_path = generate_output_path(
                input_file,
                self.input_dir,
                self.output_dir,
                self.output_format,
            )
            self._files.append(BatchFileInfo(
                index=i,
                input_path=input_file,
                output_path=output_path,
            ))

        return self._files

    def _is_completed(self, file_info: BatchFileInfo) -> bool:
        """Check if output file exists and is valid."""
        output_path = file_info.output_path
        # Check for zarr.json in s0 subdirectory (indicates successful conversion)
        marker_file = os.path.join(output_path, "s0", "zarr.json")
        return os.path.exists(marker_file)

    def _create_index_file(self, files: List[BatchFileInfo]) -> str:
        """
        Create index file mapping job index to input/output paths.

        Format (TSV):
            1    /input/tile0.tif    /output/tile0.zarr
            2    /input/tile1.tif    /output/tile1.zarr

        Returns:
            Path to index file
        """
        index_dir = os.path.join(self.output_dir, ".batch")
        os.makedirs(index_dir, exist_ok=True)

        index_file = os.path.join(index_dir, "index.tsv")

        with open(index_file, 'w') as f:
            for file_info in files:
                f.write(f"{file_info.index}\t{file_info.input_path}\t{file_info.output_path}\n")

        self._index_file = index_file
        return index_file

    def _save_batch_config(self, args_dict: Dict) -> str:
        """Save batch configuration for status checking and worker jobs."""
        config_dir = os.path.join(self.output_dir, ".batch")
        os.makedirs(config_dir, exist_ok=True)

        config_file = os.path.join(config_dir, "config.json")

        config = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "pattern": self.pattern,
            "output_format": self.output_format,
            "recursive": self.recursive,
            "total_files": len(self._files) if self._files else 0,
            "args": args_dict,
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config_file

    def submit_lsf(
        self,
        project: str,
        chunk_shape: Optional[str] = None,
        shard_shape: Optional[str] = None,
        compression: str = "zstd",
        compression_level: int = 5,
        memory_gb: int = 30,
        wall_time: str = "1:00",
        cores: int = 2,
        max_concurrent: int = 100,
        job_group: Optional[str] = None,
        skip_existing: bool = True,
        dry_run: bool = False,
    ) -> BatchResult:
        """
        Submit batch conversion as LSF job array.

        Args:
            project: LSF project name
            chunk_shape: Chunk shape string (e.g., "32,32,32")
            shard_shape: Shard shape string (e.g., "256,1024,1024")
            compression: Compression codec
            compression_level: Compression level
            memory_gb: Memory per job in GB
            wall_time: Wall time per job (H:MM)
            cores: Cores per job
            max_concurrent: Max concurrent jobs in array
            job_group: LSF job group path
            skip_existing: Skip files that already have output
            dry_run: Print commands without submitting

        Returns:
            BatchResult with submission status
        """
        if self._files is None:
            self.discover()

        result = BatchResult(total=len(self._files))

        # Filter files to process
        files_to_process = []
        for file_info in self._files:
            if skip_existing and self._is_completed(file_info):
                result.skipped += 1
                file_info.status = "completed"
            else:
                files_to_process.append(file_info)

        if not files_to_process:
            print("All files already converted. Nothing to submit.")
            return result

        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)

        # Create index file
        index_file = self._create_index_file(files_to_process)

        # Save config
        args_dict = {
            "chunk_shape": chunk_shape,
            "shard_shape": shard_shape,
            "compression": compression,
            "compression_level": compression_level,
            "output_format": self.output_format,
        }
        self._save_batch_config(args_dict)

        # Create log directory
        log_dir = os.path.join(self.output_dir, ".batch", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Build job array indices (1-based)
        # Re-index files_to_process for array
        for i, file_info in enumerate(files_to_process, start=1):
            file_info.index = i

        # Rewrite index file with new indices
        index_file = self._create_index_file(files_to_process)

        num_jobs = len(files_to_process)
        array_spec = f"[1-{num_jobs}]%{max_concurrent}"

        # Job name based on output directory
        job_name = f"tsv2_batch_{os.path.basename(self.output_dir.rstrip('/'))}"
        job_name = job_name.replace(" ", "_")[:64]

        # Build worker command list
        worker_cmd_list = [
            sys.executable, "-m", "tensorswitch_v2",
            "--batch_worker",
            "--index_file", index_file,
            "--output_format", self.output_format,
            "--compression", compression,
            "--compression_level", str(compression_level),
        ]
        if chunk_shape:
            worker_cmd_list += ["--chunk_shape", chunk_shape]
        if shard_shape:
            worker_cmd_list += ["--shard_shape", shard_shape]

        # Convert to properly quoted shell command string
        # This handles paths with spaces correctly when bsub creates its wrapper
        worker_cmd_str = shlex.join(worker_cmd_list)

        # Build bsub command - use bash -c to run the quoted command
        # This ensures paths with spaces are handled correctly
        bsub_cmd = [
            "bsub",
            "-J", f"{job_name}{array_spec}",
            "-n", str(cores),
            "-W", wall_time,
            "-M", f"{memory_gb}GB",
            "-R", f"rusage[mem={memory_gb * 1024}]",
            "-P", project,
            "-o", os.path.join(log_dir, "job_%I.log"),
            "-e", os.path.join(log_dir, "job_%I.err"),
        ]

        if job_group:
            bsub_cmd += ["-g", job_group]

        # Use bash -c with quoted command to handle paths with spaces
        bsub_cmd += ["/bin/bash", "-c", worker_cmd_str]

        # Print summary
        print("=" * 72)
        print("LSF Batch Job Array Submission")
        print("=" * 72)
        print(f"  Input dir:     {self.input_dir}")
        print(f"  Output dir:    {self.output_dir}")
        print(f"  Pattern:       {self.pattern}")
        print(f"  Total files:   {result.total}")
        print(f"  To process:    {num_jobs}")
        print(f"  Skipped:       {result.skipped} (already exist)")
        print(f"  Job name:      {job_name}{array_spec}")
        print(f"  Resources:     {cores} cores, {memory_gb} GB, {wall_time}")
        print(f"  Max concurrent: {max_concurrent}")
        print(f"  Project:       {project}")
        print(f"  Index file:    {index_file}")
        print(f"  Log dir:       {log_dir}")
        print("-" * 72)
        print(f"  Worker cmd:    {worker_cmd_str}")
        print("=" * 72)

        if dry_run:
            print("\n[DRY RUN] Would submit:")
            print(f"  {' '.join(bsub_cmd)}")
            result.submitted = num_jobs
            return result

        # Submit job array
        submit_result = subprocess.run(bsub_cmd, capture_output=True, text=True)

        if submit_result.returncode == 0:
            output = submit_result.stdout.strip()
            print(f"\nJob array submitted successfully!")
            print(output)

            # Extract job ID
            # Format: "Job <12345> is submitted to queue <normal>."
            import re
            match = re.search(r'Job <(\d+)>', output)
            if match:
                result.job_ids.append(match.group(1))

            result.submitted = num_jobs
        else:
            print(f"\nJob submission failed!")
            print(submit_result.stderr)
            result.failed = num_jobs

        print(f"\nMonitor with: bjobs -A {result.job_ids[0] if result.job_ids else '<jobid>'}")
        print(f"Check status: python -m tensorswitch_v2 --status -i {self.input_dir} -o {self.output_dir}")

        return result

    def run_local(
        self,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        shard_shape: Optional[Tuple[int, ...]] = None,
        compression: str = "zstd",
        compression_level: int = 5,
        skip_existing: bool = True,
        verbose: bool = True,
    ) -> BatchResult:
        """
        Run batch conversion locally (sequential).

        Args:
            chunk_shape: Chunk shape tuple
            shard_shape: Shard shape tuple
            compression: Compression codec
            compression_level: Compression level
            skip_existing: Skip files that already have output
            verbose: Print progress

        Returns:
            BatchResult with conversion status
        """
        from ..api.readers import Readers
        from ..api.writers import Writers
        from .converter import DistributedConverter

        if self._files is None:
            self.discover()

        result = BatchResult(total=len(self._files))

        for i, file_info in enumerate(self._files):
            if verbose:
                print(f"\n[{i+1}/{result.total}] {os.path.basename(file_info.input_path)}")

            # Check if already exists
            if skip_existing and self._is_completed(file_info):
                if verbose:
                    print(f"  SKIP: output exists")
                result.skipped += 1
                continue

            try:
                # Create output directory
                os.makedirs(os.path.dirname(file_info.output_path), exist_ok=True)

                # Create reader (auto-detect format)
                reader = Readers.auto_detect(file_info.input_path)

                # Create writer
                if self.output_format == "zarr3":
                    writer = Writers.zarr3(
                        output_path=file_info.output_path,
                        use_sharding=True,
                        compression=compression,
                        compression_level=compression_level,
                    )
                elif self.output_format == "zarr2":
                    writer = Writers.zarr2(
                        output_path=file_info.output_path,
                        compression=compression,
                        compression_level=compression_level,
                    )
                else:
                    writer = Writers.n5(
                        output_path=file_info.output_path,
                        compression=compression,
                        compression_level=compression_level,
                    )

                # Convert
                converter = DistributedConverter(reader, writer)
                converter.convert(
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    write_metadata=True,
                    verbose=verbose,
                )

                result.completed += 1
                file_info.status = "completed"

            except Exception as e:
                if verbose:
                    print(f"  FAILED: {e}")
                result.failed += 1
                result.failed_files.append((file_info.input_path, str(e)))
                file_info.status = "failed"

        # Print summary
        print("\n" + "=" * 72)
        print("Batch Conversion Complete")
        print("=" * 72)
        print(f"  Total:     {result.total}")
        print(f"  Completed: {result.completed}")
        print(f"  Skipped:   {result.skipped}")
        print(f"  Failed:    {result.failed}")

        if result.failed_files:
            print("\nFailed files:")
            for f, reason in result.failed_files[:10]:
                print(f"  - {os.path.basename(f)}: {reason}")
            if len(result.failed_files) > 10:
                print(f"  ... and {len(result.failed_files) - 10} more")

        return result

    def check_status(self) -> BatchResult:
        """
        Check batch conversion status.

        Returns:
            BatchResult with current status
        """
        if self._files is None:
            self.discover()

        result = BatchResult(total=len(self._files))

        # Check each file's status
        for file_info in self._files:
            if self._is_completed(file_info):
                result.completed += 1
                file_info.status = "completed"
            else:
                # Check if output directory exists but incomplete
                if os.path.exists(file_info.output_path):
                    # Might be in progress or failed
                    file_info.status = "in_progress"
                    result.in_progress += 1
                else:
                    file_info.status = "pending"

        # Try to get LSF job status
        config_file = os.path.join(self.output_dir, ".batch", "config.json")
        lsf_status = {}

        if os.path.exists(config_file):
            # Check for running jobs
            try:
                bjobs_result = subprocess.run(
                    ["bjobs", "-noheader", "-o", "stat"],
                    capture_output=True, text=True, timeout=10
                )
                if bjobs_result.returncode == 0:
                    lines = bjobs_result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            stat = line.strip()
                            lsf_status[stat] = lsf_status.get(stat, 0) + 1
            except Exception:
                pass

        # Print status
        print("=" * 72)
        print("Batch Conversion Status")
        print("=" * 72)
        print(f"  Input:     {self.input_dir}")
        print(f"  Output:    {self.output_dir}")
        print(f"  Pattern:   {self.pattern}")
        print("-" * 72)
        print(f"  Total files:   {result.total}")
        print(f"  Completed:     {result.completed} ({100*result.completed/result.total:.1f}%)")
        print(f"  In progress:   {result.in_progress}")
        print(f"  Pending:       {result.total - result.completed - result.in_progress}")

        if lsf_status:
            print("-" * 72)
            print("  LSF Jobs:")
            for stat, count in sorted(lsf_status.items()):
                print(f"    {stat}: {count}")

        # Check for failed jobs (logs with errors)
        log_dir = os.path.join(self.output_dir, ".batch", "logs")
        if os.path.exists(log_dir):
            err_files = glob.glob(os.path.join(log_dir, "*.err"))
            non_empty_errs = [f for f in err_files if os.path.getsize(f) > 0]
            if non_empty_errs:
                print("-" * 72)
                print(f"  Error logs: {len(non_empty_errs)} files with errors")
                print(f"  Check: ls {log_dir}/*.err")

        print("=" * 72)

        return result


def read_index_file(index_file: str, job_index: int) -> Tuple[str, str]:
    """
    Read input/output paths for a specific job index from index file.

    Args:
        index_file: Path to index.tsv file
        job_index: Job array index (1-based)

    Returns:
        (input_path, output_path) tuple
    """
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                idx = int(parts[0])
                if idx == job_index:
                    return parts[1], parts[2]

    raise ValueError(f"Job index {job_index} not found in {index_file}")


@dataclass
class DiscoveredConversionResult:
    """Result of discovered folder conversion."""
    image_converted: bool = False
    segmentation_converted: bool = False
    output_path: str = ""
    error: Optional[str] = None


def convert_discovered_folder(
    input_dir: str,
    output_path: str,
    image_only: bool = False,
    labels_only: bool = False,
    image_key: str = "raw",
    label_key: str = "segmentation",
    use_nested_structure: bool = True,
    output_format: str = "zarr3",
    chunk_shape: Optional[Tuple[int, ...]] = None,
    shard_shape: Optional[Tuple[int, ...]] = None,
    compression: str = "zstd",
    compression_level: int = 5,
    verbose: bool = True,
) -> DiscoveredConversionResult:
    """
    Convert a discovered folder containing image and/or segmentation datasets.

    Uses folder_discovery to find datasets, validates them, and converts
    to a single zarr output with OME-NGFF nested structure.

    Args:
        input_dir: Directory containing datasets to discover
        output_path: Output zarr path
        image_only: Only convert image dataset (skip segmentation)
        labels_only: Only convert segmentation dataset (skip image)
        image_key: Name for image group in output (default: 'raw')
        label_key: Name for label image in output (default: 'segmentation')
        use_nested_structure: Use OME-NGFF nested structure (default: True)
        output_format: Output format (zarr3, zarr2, n5)
        chunk_shape: Optional chunk shape
        shard_shape: Optional shard shape
        compression: Compression codec
        compression_level: Compression level
        verbose: Print progress

    Returns:
        DiscoveredConversionResult with conversion status

    Example:
        >>> result = convert_discovered_folder(
        ...     '/data/230130b/',
        ...     '/output/230130b.zarr',
        ...     image_only=False,
        ...     labels_only=False,
        ... )
        >>> if result.error:
        ...     print(f"Error: {result.error}")
    """
    from ..utils.folder_discovery import (
        discover_datasets,
        validate_discovery_for_conversion,
        print_discovery_summary,
    )
    from ..api.readers import Readers
    from ..api.writers import Writers
    from .converter import DistributedConverter

    result = DiscoveredConversionResult(output_path=output_path)

    # Discover datasets in the folder
    if verbose:
        print(f"\nDiscovering datasets in: {input_dir}")

    discovery = discover_datasets(input_dir, verbose=verbose)

    if verbose:
        print_discovery_summary(discovery)

    # Validate and get datasets to convert
    image_ds, seg_ds, error = validate_discovery_for_conversion(
        discovery,
        image_only=image_only,
        labels_only=labels_only,
    )

    if error:
        result.error = error
        return result

    # Check we have something to convert
    if not image_ds and not seg_ds:
        result.error = "No datasets found to convert in directory."
        return result

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert image dataset if found
    if image_ds and not labels_only:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Converting IMAGE: {image_ds.name}")
            print(f"{'='*60}")

        try:
            reader = Readers.auto_detect(image_ds.path)

            if output_format == "zarr3":
                writer = Writers.zarr3(
                    output_path=output_path,
                    use_sharding=True,
                    compression=compression,
                    compression_level=compression_level,
                    use_nested_structure=use_nested_structure,
                    data_type='image',
                    image_key=image_key,
                    label_key=label_key,
                )
            elif output_format == "zarr2":
                writer = Writers.zarr2(
                    output_path=output_path,
                    compression=compression,
                    compression_level=compression_level,
                )
            else:
                writer = Writers.n5(
                    output_path=output_path,
                    compression=compression,
                    compression_level=compression_level,
                )

            converter = DistributedConverter(reader, writer)
            converter.convert(
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                write_metadata=True,
                verbose=verbose,
            )
            result.image_converted = True

        except Exception as e:
            result.error = f"Failed to convert image: {e}"
            return result

    # Convert segmentation dataset if found
    if seg_ds and not image_only:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Converting SEGMENTATION: {seg_ds.name}")
            print(f"{'='*60}")

        try:
            reader = Readers.auto_detect(seg_ds.path)

            if output_format == "zarr3":
                writer = Writers.zarr3(
                    output_path=output_path,
                    use_sharding=True,
                    compression=compression,
                    compression_level=compression_level,
                    use_nested_structure=use_nested_structure,
                    data_type='labels',
                    image_key=image_key,
                    label_key=label_key,
                )
            elif output_format == "zarr2":
                writer = Writers.zarr2(
                    output_path=output_path,
                    compression=compression,
                    compression_level=compression_level,
                )
            else:
                writer = Writers.n5(
                    output_path=output_path,
                    compression=compression,
                    compression_level=compression_level,
                )

            converter = DistributedConverter(reader, writer)
            converter.convert(
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                write_metadata=True,
                verbose=verbose,
                is_label=True,
            )
            result.segmentation_converted = True

        except Exception as e:
            result.error = f"Failed to convert segmentation: {e}"
            return result

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("DISCOVERED FOLDER CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        if result.image_converted:
            print(f"  Image:  {image_key}/ ✓")
        if result.segmentation_converted:
            print(f"  Labels: labels/{label_key}/ ✓")
        print(f"{'='*60}")

    return result


def submit_discovered_folder_lsf(
    input_dir: str,
    output_path: str,
    project: str,
    image_only: bool = False,
    labels_only: bool = False,
    image_key: str = "raw",
    label_key: str = "segmentation",
    use_nested_structure: bool = True,
    output_format: str = "zarr3",
    chunk_shape: Optional[str] = None,
    shard_shape: Optional[str] = None,
    compression: str = "zstd",
    compression_level: int = 5,
    memory_gb: int = 30,
    wall_time: str = "2:00",
    cores: int = 4,
    job_group: str = "/scicompsoft/chend/tensorstore",
    dry_run: bool = False,
) -> Dict:
    """
    Submit LSF jobs for discovered folder conversion.

    Submits separate jobs for image and segmentation datasets if both are found.

    Args:
        input_dir: Directory containing datasets to discover
        output_path: Output zarr path
        project: LSF project name
        image_only: Only convert image dataset
        labels_only: Only convert segmentation dataset
        image_key: Name for image group
        label_key: Name for label image
        use_nested_structure: Use OME-NGFF nested structure
        output_format: Output format
        chunk_shape: Chunk shape string
        shard_shape: Shard shape string
        compression: Compression codec
        compression_level: Compression level
        memory_gb: Memory per job
        wall_time: Wall time per job
        cores: Cores per job
        job_group: LSF job group
        dry_run: Print commands without submitting

    Returns:
        Dict with job_ids for submitted jobs
    """
    from ..utils.folder_discovery import (
        discover_datasets,
        validate_discovery_for_conversion,
    )

    result = {'job_ids': [], 'error': None}

    # Discover and validate
    discovery = discover_datasets(input_dir, verbose=True)
    image_ds, seg_ds, error = validate_discovery_for_conversion(
        discovery,
        image_only=image_only,
        labels_only=labels_only,
    )

    if error:
        result['error'] = error
        print(f"\nError: {error}")
        return result

    # Create log directory
    output_parent = os.path.dirname(os.path.abspath(output_path))
    log_dir = os.path.join(output_parent, "output")
    os.makedirs(log_dir, exist_ok=True)

    jobs_to_submit = []

    # Prepare image job
    if image_ds and not labels_only:
        jobs_to_submit.append({
            'input_path': image_ds.path,
            'data_type': 'image',
            'name': image_ds.name,
        })

    # Prepare segmentation job
    if seg_ds and not image_only:
        jobs_to_submit.append({
            'input_path': seg_ds.path,
            'data_type': 'labels',
            'name': seg_ds.name,
        })

    # Submit jobs
    for job_info in jobs_to_submit:
        job_name = f"tsv2_disc_{job_info['data_type']}_{job_info['name']}"
        job_name = job_name.replace(" ", "_")[:64]

        log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
        error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

        # Build worker command
        worker_cmd = [
            sys.executable, "-m", "tensorswitch_v2",
            "--input", job_info['input_path'],
            "--output", output_path,
            "--output_format", output_format,
            "--data-type", job_info['data_type'],
            "--compression", compression,
            "--compression_level", str(compression_level),
            "--image-key", image_key,
            "--label-key", label_key,
        ]
        if use_nested_structure:
            worker_cmd.append("--use-nested-structure")
        else:
            worker_cmd.append("--no-nested-structure")
        if chunk_shape:
            worker_cmd += ["--chunk_shape", chunk_shape]
        if shard_shape:
            worker_cmd += ["--shard_shape", shard_shape]
        if job_info['data_type'] == 'labels':
            worker_cmd.append("--is-label")

        # Convert to properly quoted shell command string
        # This handles paths with spaces correctly when bsub creates its wrapper
        worker_cmd_str = shlex.join(worker_cmd)

        # Build bsub command - use bash -c to run the quoted command
        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", str(cores),
            "-W", wall_time,
            "-M", f"{memory_gb}GB",
            "-R", f"rusage[mem={memory_gb * 1024}]",
            "-P", project,
            "-g", job_group,
            "-o", log_path,
            "-e", error_path,
            "/bin/bash", "-c", worker_cmd_str,
        ]

        print(f"\n{'='*60}")
        print(f"LSF Job: {job_info['data_type'].upper()}")
        print(f"{'='*60}")
        print(f"  Job name:  {job_name}")
        print(f"  Input:     {job_info['input_path']}")
        print(f"  Output:    {output_path}")
        print(f"  Data type: {job_info['data_type']}")
        print(f"  Resources: {cores} cores, {memory_gb} GB, {wall_time}")

        if dry_run:
            print(f"\n[DRY RUN] Would submit:")
            print(f"  {worker_cmd_str}")
            result['job_ids'].append('DRY_RUN')
        else:
            submit_result = subprocess.run(bsub_cmd, capture_output=True, text=True)
            if submit_result.returncode == 0:
                print(f"\nJob submitted: {submit_result.stdout.strip()}")
                import re
                match = re.search(r'Job <(\d+)>', submit_result.stdout)
                if match:
                    result['job_ids'].append(match.group(1))
            else:
                print(f"\nJob submission failed: {submit_result.stderr}")
                result['error'] = submit_result.stderr

    return result
