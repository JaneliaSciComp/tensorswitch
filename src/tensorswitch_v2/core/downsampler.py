"""
Downsampler for TensorSwitch Phase 5 architecture.

Provides parallel downsampling from s0 to any target level using cumulative factors.
This is a key innovation over v1's sequential approach (s0→s1→s2).

Key Features:
- All levels downsample directly from s0 (parallel execution)
- Cumulative factor calculation for each level
- LSF multi-job mode support with chunk-range processing
- Uses TensorStore's downsample driver for efficient processing
"""

import os
import json
import time
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import tensorstore as ts

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

# Import utility functions from v2 utils (independent from v1)
from tensorswitch_v2.utils import (
    get_chunk_domains,
    get_total_chunks_from_store,
    get_tensorstore_context,
    downsample_spec,
    zarr3_store_spec,
    get_input_driver,
    get_chunk_linear_indices_in_shard,
)


def calculate_cumulative_factors(
    per_level_factors: List[List[int]],
    target_level: int
) -> List[int]:
    """
    Calculate cumulative downsampling factors from s0 to target level.

    This is the key function enabling parallel downsampling from s0.
    Instead of applying factors sequentially (s0→s1, s1→s2), we multiply
    all factors up to the target level to get a single cumulative factor.

    Args:
        per_level_factors: List of per-level downsampling factors.
            per_level_factors[0] = factors for s0→s1
            per_level_factors[1] = factors for s1→s2
            etc.
        target_level: Target level to downsample to (1-indexed).
            Level 1 = s1, Level 2 = s2, etc.

    Returns:
        List of cumulative factors for direct s0→s{target_level} downsampling.

    Example:
        >>> per_level_factors = [
        ...     [1, 2, 2],  # s0→s1: no Z, 2x Y, 2x X
        ...     [1, 2, 2],  # s1→s2: no Z, 2x Y, 2x X
        ...     [2, 2, 2],  # s2→s3: 2x Z, 2x Y, 2x X (Z catches up)
        ... ]
        >>> calculate_cumulative_factors(per_level_factors, 1)
        [1, 2, 2]  # Direct s0→s1
        >>> calculate_cumulative_factors(per_level_factors, 2)
        [1, 4, 4]  # Direct s0→s2 (2*2 = 4)
        >>> calculate_cumulative_factors(per_level_factors, 3)
        [2, 8, 8]  # Direct s0→s3 (1*1*2=2 for Z, 2*2*2=8 for Y,X)
    """
    if target_level < 1:
        raise ValueError(f"target_level must be >= 1, got {target_level}")
    if target_level > len(per_level_factors):
        raise ValueError(
            f"target_level {target_level} exceeds available factors "
            f"(max level: {len(per_level_factors)})"
        )

    # Start with identity factors
    num_dims = len(per_level_factors[0])
    cumulative = [1] * num_dims

    # Multiply factors for each level up to target
    for level_idx in range(target_level):
        for dim_idx in range(num_dims):
            cumulative[dim_idx] *= per_level_factors[level_idx][dim_idx]

    return cumulative


class Downsampler:
    """
    Downsample directly from s0 to any target level with cumulative factors.

    This class enables parallel pyramid generation by allowing each level
    to be computed independently from s0, without waiting for previous levels.

    Design Principles:
    - Each level job is self-contained and independent
    - Uses TensorStore's downsample driver for efficient processing
    - Supports chunk-range processing for LSF multi-job mode
    - Pre-creation of output is handled separately (before job submission)

    Example (single level):
        >>> downsampler = Downsampler(
        ...     s0_path="/data/dataset.zarr/s0",
        ...     output_path="/data/dataset.zarr",
        ...     target_level=2
        ... )
        >>> downsampler.downsample(cumulative_factors=[1, 4, 4])

    Example (LSF worker processing chunk range):
        >>> downsampler.downsample(
        ...     cumulative_factors=[1, 4, 4],
        ...     start_idx=0,
        ...     stop_idx=100
        ... )
    """

    def __init__(
        self,
        s0_path: str,
        output_path: str,
        target_level: int,
        use_shard: bool = True,
        custom_shard_shape: Optional[List[int]] = None,
        custom_chunk_shape: Optional[List[int]] = None,
        downsample_method: str = "mean",
    ):
        """
        Initialize Downsampler.

        Args:
            s0_path: Path to s0 array (e.g., "/data/dataset.zarr/s0")
            output_path: Root output path (e.g., "/data/dataset.zarr")
            target_level: Target level to create (1 = s1, 2 = s2, etc.)
            use_shard: Whether to use sharding for output (default: True)
            custom_shard_shape: Override shard shape (uses defaults if None)
            custom_chunk_shape: Override chunk shape (uses defaults if None)
            downsample_method: TensorStore downsample method (default: "mean").
                Options: "mean" (intensity images), "mode" (segmentation/labels),
                "median" (noise reduction), "stride" (fastest), "min", "max".
        """
        self.s0_path = s0_path
        self.output_path = output_path
        self.target_level = target_level
        self.use_shard = use_shard
        self.custom_shard_shape = custom_shard_shape
        self.custom_chunk_shape = custom_chunk_shape
        self.downsample_method = downsample_method

        # Will be populated when downsample() is called
        self._s0_store = None
        self._output_store = None
        self._downsampled_store = None
        self._s0_metadata = None

    def _load_s0_metadata(self) -> Dict[str, Any]:
        """Load metadata from s0 array."""
        zarr_json_path = os.path.join(self.s0_path, 'zarr.json')
        if not os.path.exists(zarr_json_path):
            raise FileNotFoundError(f"s0 zarr.json not found at {zarr_json_path}")

        with open(zarr_json_path, 'r') as f:
            return json.load(f)

    def _get_output_level_path(self) -> str:
        """Get the output path for the target level."""
        return os.path.join(self.output_path, f"s{self.target_level}")

    def downsample(
        self,
        cumulative_factors: List[int],
        start_idx: int = 0,
        stop_idx: Optional[int] = None,
        shard_coord: Optional[List[int]] = None,
        use_fortran_order: bool = False,
        progress_interval: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Downsample s0 directly to target level using cumulative factors.

        Args:
            cumulative_factors: Cumulative downsampling factors from s0.
                Example: [1, 4, 4] for direct s0→s2 (4x in Y,X)
            start_idx: Starting chunk index (for LSF multi-job mode)
            stop_idx: Ending chunk index (None = process all remaining)
            shard_coord: Optional 3D shard coordinate [z, y, x] for shard-based processing
            use_fortran_order: Whether to use Fortran (F) order for output
            progress_interval: Report progress every N chunks
            verbose: Print progress messages

        Returns:
            dict: Processing statistics (chunks_processed, elapsed_time, etc.)
        """
        start_time = time.time()

        # Load s0 metadata
        self._s0_metadata = self._load_s0_metadata()
        dimension_names = self._s0_metadata.get('dimension_names')

        if verbose:
            print(f"\n{'='*60}")
            print(f"DOWNSAMPLING: s0 → s{self.target_level}")
            print(f"{'='*60}")
            print(f"Source: {self.s0_path}")
            print(f"Output: {self._get_output_level_path()}")
            print(f"Cumulative factors: {cumulative_factors}")
            print(f"Downsample method: {self.downsample_method}")
            print(f"Dimension names: {dimension_names}")

        # Open s0 store
        input_driver = get_input_driver(self.s0_path)
        s0_spec = {
            'driver': input_driver,
            'kvstore': {'driver': 'file', 'path': self.s0_path},
            'context': get_tensorstore_context()
        }
        self._s0_store = ts.open(s0_spec).result()
        s0_shape = list(self._s0_store.shape)
        s0_dtype = self._s0_store.dtype.name

        if verbose:
            print(f"s0 shape: {s0_shape}")
            print(f"s0 dtype: {s0_dtype}")

        # Create downsampled virtual view of s0 using TensorStore downsample driver
        downsampled_spec = downsample_spec(
            s0_spec,
            s0_shape,
            dimension_names,
            custom_factors=cumulative_factors,
            method=self.downsample_method
        )
        downsampled_spec['context'] = get_tensorstore_context()
        self._downsampled_store = ts.open(downsampled_spec).result()
        # IMPORTANT: Use TensorStore's actual shape, not the predicted shape from pyramid planner
        # TensorStore uses ceiling division: (dim + factor - 1) // factor
        downsampled_shape = list(self._downsampled_store.shape)

        if verbose:
            print(f"Downsampled shape: {downsampled_shape}")

        # Set default chunk and shard shapes if not provided
        chunk_shape = self.custom_chunk_shape
        if chunk_shape is None:
            chunk_shape = [32, 32, 32]
            if verbose:
                print(f"Using default chunk shape: {chunk_shape}")

        shard_shape = self.custom_shard_shape
        if shard_shape is None and self.use_shard:
            shard_shape = [1024, 1024, 1024]
            if verbose:
                print(f"Using default shard shape: {shard_shape}")

        # Create output store spec using the ACTUAL downsampled shape from TensorStore
        output_level_path = self._get_output_level_path()
        output_spec = zarr3_store_spec(
            self.output_path,
            downsampled_shape,  # Use actual shape, not predicted
            s0_dtype,
            self.use_shard,
            level_path=f"s{self.target_level}",
            use_ome_structure=True,
            custom_shard_shape=shard_shape,
            custom_chunk_shape=chunk_shape,
            use_fortran_order=use_fortran_order,
            axes_order=dimension_names
        )
        output_spec['context'] = get_tensorstore_context()

        # Create output directory
        os.makedirs(output_level_path, exist_ok=True)

        # Check if pre-created metadata exists and has wrong shape
        # If so, delete and recreate with correct shape
        zarr_json_path = os.path.join(output_level_path, 'zarr.json')
        need_recreate = False
        if os.path.exists(zarr_json_path):
            with open(zarr_json_path, 'r') as f:
                existing_metadata = json.load(f)
            existing_shape = existing_metadata.get('shape', [])
            if existing_shape != downsampled_shape:
                if verbose:
                    print(f"  Fixing pre-created shape: {existing_shape} -> {downsampled_shape}")
                # Delete the incorrect zarr.json so TensorStore will recreate it
                os.remove(zarr_json_path)
                need_recreate = True

        # Open output store
        # Note: Can't use delete_existing with open=True, so we delete manually above
        self._output_store = ts.open(output_spec, create=True, open=True).result()

        # Get chunk shape from output store
        output_chunk_shape = self._output_store.chunk_layout.write_chunk.shape

        if verbose:
            print(f"Output chunk shape: {output_chunk_shape}")

        # Handle shard_coord if provided
        if shard_coord is not None:
            linear_indices_to_process = self._get_shard_chunk_indices(
                shard_coord, shard_shape, output_chunk_shape, downsampled_shape
            )
            if verbose:
                print(f"Processing shard {shard_coord}: {len(linear_indices_to_process)} chunks")
        else:
            linear_indices_to_process = None

        # Calculate total chunks
        total_chunks = get_total_chunks_from_store(
            self._output_store,
            chunk_shape=output_chunk_shape
        )

        if stop_idx is None:
            stop_idx = total_chunks

        if linear_indices_to_process is None:
            linear_indices_to_process = range(start_idx, stop_idx)

        if verbose:
            print(f"Processing chunks {start_idx} to {stop_idx} of {total_chunks}")

        # Process chunks with transaction-per-chunk pattern
        chunks_processed = 0
        for chunk_domain in get_chunk_domains(
            output_chunk_shape,
            self._output_store,
            linear_indices_to_process=linear_indices_to_process
        ):
            # Create transaction per chunk to prevent OOM
            with ts.Transaction() as txn:
                self._output_store[chunk_domain].with_transaction(txn).write(
                    self._downsampled_store[chunk_domain]
                ).result()

            chunks_processed += 1
            if verbose and chunks_processed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = chunks_processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {chunks_processed} chunks ({rate:.1f} chunks/s)")

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nDownsampling complete for s{self.target_level}")
            print(f"  Chunks processed: {chunks_processed}")
            print(f"  Time: {elapsed_time:.1f}s")
            print(f"  Rate: {chunks_processed/elapsed_time:.1f} chunks/s")

        return {
            'target_level': self.target_level,
            'chunks_processed': chunks_processed,
            'total_chunks': total_chunks,
            'elapsed_time': elapsed_time,
            'cumulative_factors': cumulative_factors,
            'output_shape': downsampled_shape,
        }

    def _get_shard_chunk_indices(
        self,
        shard_coord: List[int],
        shard_shape: List[int],
        chunk_shape: Tuple[int, ...],
        output_shape: List[int]
    ) -> List[int]:
        """Get linear chunk indices within a specific shard."""
        import math

        # Calculate chunk grid dimensions
        chunk_grid = [
            (output_shape[i] + chunk_shape[i] - 1) // chunk_shape[i]
            for i in range(len(output_shape))
        ]

        # Use N-D utility function from v1
        return get_chunk_linear_indices_in_shard(
            shard_coord=shard_coord,
            shard_shape=shard_shape,
            chunk_shape=list(chunk_shape),
            chunk_grid=chunk_grid
        )


def downsample_level(
    s0_path: str,
    output_path: str,
    target_level: int,
    cumulative_factors: List[int],
    start_idx: int = 0,
    stop_idx: Optional[int] = None,
    use_shard: bool = True,
    custom_shard_shape: Optional[List[int]] = None,
    custom_chunk_shape: Optional[List[int]] = None,
    downsample_method: str = "mean",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to downsample a single level.

    This is the main entry point for CLI and job scripts.

    Args:
        s0_path: Path to s0 array
        output_path: Root output path
        target_level: Target level to create (1 = s1, 2 = s2, etc.)
        cumulative_factors: Cumulative downsampling factors from s0
        start_idx: Starting chunk index (for LSF multi-job mode)
        stop_idx: Ending chunk index
        use_shard: Whether to use sharding
        custom_shard_shape: Override shard shape
        custom_chunk_shape: Override chunk shape
        downsample_method: TensorStore downsample method (default: "mode").
            Options: "mean", "mode", "median", "stride", "min", "max".
        verbose: Print progress messages

    Returns:
        dict: Processing statistics

    Example:
        >>> # Downsample s0 directly to s2 with 4x factor in Y,X
        >>> stats = downsample_level(
        ...     s0_path="/data/dataset.zarr/s0",
        ...     output_path="/data/dataset.zarr",
        ...     target_level=2,
        ...     cumulative_factors=[1, 4, 4]
        ... )
    """
    downsampler = Downsampler(
        s0_path=s0_path,
        output_path=output_path,
        target_level=target_level,
        use_shard=use_shard,
        custom_shard_shape=custom_shard_shape,
        custom_chunk_shape=custom_chunk_shape,
        downsample_method=downsample_method,
    )

    return downsampler.downsample(
        cumulative_factors=cumulative_factors,
        start_idx=start_idx,
        stop_idx=stop_idx,
        verbose=verbose,
    )
