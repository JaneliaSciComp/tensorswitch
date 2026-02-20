"""
Pyramid Planner for TensorSwitch Phase 5 architecture.

Provides multi-level pyramid planning and chained job submission.
Uses the v1 chained approach where each level reads from the previous level,
which is much more efficient for deep pyramids.

Key Features:
- Pre-calculate complete pyramid plan before submission
- Chained downsampling: s1←s0, s2←s1, s3←s2, etc. (not all from s0)
- Sequential job submission with bwait between levels
- Per-level factors (constant ~8 voxels read per output) vs cumulative (exponential)
- Reuse Yurii Zubov's anisotropic downsampling algorithm from v1
"""

import os
import json
import shlex
import subprocess
import sys
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

# Import utility functions from v2 utils (independent from v1)
from tensorswitch_v2.utils import (
    calculate_pyramid_plan as v1_calculate_pyramid_plan,
    calculate_anisotropic_downsample_factors,
    precreate_zarr3_output,
    get_tensorstore_context,
    update_ome_metadata_if_needed,
)

from .downsampler import calculate_cumulative_factors
from tensorswitch_v2.utils.pyramid_utils import resolve_downsample_method


def extract_source_spatial_unit(root_path: str, zarr_format: str) -> str:
    """
    Extract the spatial unit from the root OME-NGFF metadata.

    For downsampling, we preserve the source s0 unit to maintain consistency.

    Args:
        root_path: Path to the zarr root (parent of s0/0 directory)
        zarr_format: 'zarr3' or 'zarr2'

    Returns:
        str: Spatial unit string (e.g., 'nanometer', 'micrometer').
             Defaults to 'nanometer' if not found.
    """
    try:
        if zarr_format == 'zarr3':
            zarr_json_path = os.path.join(root_path, 'zarr.json')
            if os.path.exists(zarr_json_path):
                with open(zarr_json_path, 'r') as f:
                    metadata = json.load(f)
                axes = metadata.get('attributes', {}).get('ome', {}).get('multiscales', [{}])[0].get('axes', [])
                for axis in axes:
                    if axis.get('type') == 'space' and 'unit' in axis:
                        return axis['unit']
        else:  # zarr2
            zattrs_path = os.path.join(root_path, '.zattrs')
            if os.path.exists(zattrs_path):
                with open(zattrs_path, 'r') as f:
                    metadata = json.load(f)
                axes = metadata.get('multiscales', [{}])[0].get('axes', [])
                for axis in axes:
                    if axis.get('type') == 'space' and 'unit' in axis:
                        return axis['unit']
    except Exception as e:
        print(f"Warning: Could not extract source spatial unit: {e}")

    # Default to nanometer for new data
    return 'nanometer'


def _calculate_downsample_memory(
    source_shape: List[int],
    level_shape: List[int],
    shard_shape: List[int],
    dtype_size: int = 2,
) -> int:
    """
    Calculate memory in GB for a downsampling job (chained mode).

    In chained mode, each level reads from the previous level (not s0).
    Memory needed:
    - Read buffer: portion of source that maps to output shard (~8x for 2x downsample)
    - Write buffer: output shard
    - Processing overhead

    Args:
        source_shape: Shape of source array (previous level, e.g., s1 shape for s2 job)
        level_shape: Shape of target level
        shard_shape: Shard shape for output
        dtype_size: Bytes per element (default: 2 for uint16)

    Returns:
        Memory in GB (rounded to nearest 5, min 8, max 128)
    """
    import math

    # Calculate source data size
    source_size_gb = (np.prod(source_shape) * dtype_size) / (1024 ** 3)

    # Calculate output shard size
    shard_size_gb = (np.prod(shard_shape) * dtype_size) / (1024 ** 3)

    # In chained mode, per-level factor is small (typically 2x per spatial dim)
    # Each output voxel needs ~8 input voxels (2x2x2 for 3D spatial)
    per_level_factor = [source_shape[i] / level_shape[i] if level_shape[i] > 0 else 1
                        for i in range(len(source_shape))]
    total_factor = np.prod(per_level_factor)

    # Read buffer: shard_size * factor (typically ~8x, much smaller than cumulative)
    read_buffer_gb = min(shard_size_gb * total_factor, source_size_gb * 0.2)

    # Write buffer: shard size * 2 (read + write)
    write_buffer_gb = shard_size_gb * 2

    # Base overhead
    base_overhead = 4  # GB for Python, TensorStore, etc.

    # Total with 1.5x safety margin
    total_gb = read_buffer_gb + write_buffer_gb + base_overhead
    recommended = int(math.ceil(total_gb * 1.5 / 5) * 5)

    # Clamp to reasonable range (min 8GB for overhead, max 128GB)
    return max(8, min(recommended, 128))


def _calculate_downsample_wall_time(
    source_shape: List[int],
    level_shape: List[int],
    shard_shape: List[int],
    dtype_size: int = 2,
) -> str:
    """
    Calculate wall time for a downsampling job (chained mode).

    In chained mode, each level reads from the previous level which is
    much smaller than s0, so wall time is more predictable.

    Args:
        source_shape: Shape of source array (previous level)
        level_shape: Shape of target level
        shard_shape: Shard shape for output
        dtype_size: Bytes per element

    Returns:
        Wall time string in H:MM format
    """
    import math

    # Calculate sizes
    source_size_gb = (np.prod(source_shape) * dtype_size) / (1024 ** 3)
    level_size_gb = (np.prod(level_shape) * dtype_size) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_size) / (1024 ** 3)

    # Calculate total shards
    total_shards = int(np.prod(np.ceil(np.array(level_shape) / np.array(shard_shape)).astype(int)))

    # Time estimation based on empirical observations:
    # In chained mode, we're reading from smaller source levels
    # - Small shards (<0.1 GB): ~0.5 min each (faster since source is smaller)
    # - Medium shards (0.1-1 GB): ~2 min each
    # - Large shards (>1 GB): ~4 min each

    if shard_size_gb < 0.1:
        minutes_per_shard = 0.5
    elif shard_size_gb < 1.0:
        minutes_per_shard = 2
    else:
        minutes_per_shard = 4

    # Base time for processing
    base_minutes = minutes_per_shard * total_shards

    # Add overhead for source reading (based on source size, not s0)
    if source_size_gb > 100:
        read_overhead = 5
    elif source_size_gb > 10:
        read_overhead = 3
    else:
        read_overhead = 1

    # Total with 2x safety margin, round to 15 min, cap at 12 hours
    total_minutes = int(math.ceil((base_minutes + read_overhead) * 2 / 15) * 15)
    total_minutes = max(15, min(total_minutes, 12 * 60))  # 15 min to 12 hours

    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours}:{minutes:02d}"


def _calculate_downsample_cores(memory_gb: int) -> int:
    """
    Calculate number of cores based on memory allocation.

    LSF typically allocates ~15GB per core, so we scale accordingly.

    Args:
        memory_gb: Memory in GB

    Returns:
        Number of cores (min 1, max 8)
    """
    import math
    cores = max(1, int(math.ceil(memory_gb / 15)) * 2)
    return min(cores, 8)


class PyramidPlanner:
    """
    Plan and coordinate multi-level pyramid generation with parallel execution.

    Uses chained downsampling approach (like v1) where each level reads from
    the previous level. This is much more efficient than parallel-from-s0 for
    deep pyramids because:
    - Each level reads only ~8 voxels per output (2x2x2) instead of exponential
    - Memory and time requirements are constant per level, not exponential
    - s4/s5 complete in minutes instead of hours

    Design Principles:
    - Chained: s1←s0, s2←s1, s3←s2, etc. (sequential with dependencies)
    - Per-level factors used (constant ~8x read amplification)
    - Coordinator script handles bwait between levels
    - Pre-creation happens before any job submission

    Example:
        >>> planner = PyramidPlanner("/data/dataset.zarr/s0")
        >>> plan = planner.calculate_pyramid_plan()
        >>> print(f"Need {plan['num_levels']} levels")
        >>> for level in plan['levels']:
        ...     print(f"s{level['level']}: factor={level['per_level_factor']}")
        >>>
        >>> # Submit chained jobs with coordinator
        >>> job_id = planner.submit_chained_pyramid(plan, project="scicompsoft")
    """

    def __init__(self, s0_path: str):
        """
        Initialize PyramidPlanner.

        Args:
            s0_path: Path to s0 array (e.g., "/data/dataset.zarr/s0")
        """
        self.s0_path = s0_path
        self.root_path = os.path.dirname(s0_path)
        self._s0_metadata = None

    def _load_s0_metadata(self) -> Dict[str, Any]:
        """Load metadata from s0 array (supports both Zarr3 and Zarr2)."""
        if self._s0_metadata is not None:
            return self._s0_metadata

        # Try Zarr3 first (zarr.json)
        zarr_json_path = os.path.join(self.s0_path, 'zarr.json')
        if os.path.exists(zarr_json_path):
            with open(zarr_json_path, 'r') as f:
                self._s0_metadata = json.load(f)
            self._s0_metadata['_format'] = 'zarr3'
            # Extract source spatial unit from root metadata
            self._s0_metadata['source_spatial_unit'] = extract_source_spatial_unit(self.root_path, 'zarr3')
            return self._s0_metadata

        # Try Zarr2 (.zarray + .zattrs)
        zarray_path = os.path.join(self.s0_path, '.zarray')
        if os.path.exists(zarray_path):
            with open(zarray_path, 'r') as f:
                zarray = json.load(f)

            # Load .zattrs for dimension names (check level first, then root)
            zattrs = {}
            zattrs_path = os.path.join(self.s0_path, '.zattrs')
            if os.path.exists(zattrs_path):
                with open(zattrs_path, 'r') as f:
                    zattrs = json.load(f)

            # Also check root .zattrs for OME-NGFF metadata
            root_zattrs = {}
            root_zattrs_path = os.path.join(self.root_path, '.zattrs')
            if os.path.exists(root_zattrs_path):
                with open(root_zattrs_path, 'r') as f:
                    root_zattrs = json.load(f)

            # Convert Zarr2 format to unified format similar to Zarr3
            shape = zarray.get('shape', [])
            chunks = zarray.get('chunks', shape)

            # Get dimension names - try multiple sources
            dimension_names = None

            # 1. Try _ARRAY_DIMENSIONS from level .zattrs (xarray convention)
            dimension_names = zattrs.get('_ARRAY_DIMENSIONS')

            # 2. Try OME-NGFF multiscales axes from root .zattrs
            if not dimension_names:
                multiscales = root_zattrs.get('multiscales', [])
                if multiscales and 'axes' in multiscales[0]:
                    axes = multiscales[0]['axes']
                    dimension_names = [ax.get('name', f'dim_{i}') for i, ax in enumerate(axes)]

            # 3. Infer from shape as fallback
            if not dimension_names:
                if len(shape) == 5:
                    dimension_names = ['t', 'c', 'z', 'y', 'x']
                elif len(shape) == 4:
                    dimension_names = ['c', 'z', 'y', 'x']
                elif len(shape) == 3:
                    dimension_names = ['z', 'y', 'x']
                else:
                    dimension_names = [f'dim_{i}' for i in range(len(shape))]

            # Extract source spatial unit from root metadata
            source_spatial_unit = extract_source_spatial_unit(self.root_path, 'zarr2')

            self._s0_metadata = {
                '_format': 'zarr2',
                'shape': shape,
                'dimension_names': dimension_names,
                'chunk_grid': {
                    'configuration': {
                        'chunk_shape': chunks
                    }
                },
                'data_type': zarray.get('dtype', 'uint16'),
                'source_spatial_unit': source_spatial_unit,
                # Zarr2 doesn't have sharding, use chunks as both
                '_zarray': zarray,
                '_zattrs': zattrs
            }
            return self._s0_metadata

        raise FileNotFoundError(
            f"s0 metadata not found. Expected zarr.json (Zarr3) or .zarray (Zarr2) at {self.s0_path}"
        )

    def calculate_pyramid_plan(
        self,
        min_array_nbytes: Optional[int] = None,
        min_array_shape: Optional[List[int]] = None,
        custom_per_level_factors: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate complete pyramid plan with cumulative factors for parallel execution.

        This extends v1's calculate_pyramid_plan by adding cumulative factors
        that enable direct s0→sN downsampling for each level.

        Args:
            min_array_nbytes: Stop when array size < this (default: chunk_nbytes)
            min_array_shape: Stop when all dims < this (default: chunk_shape)
            custom_per_level_factors: Custom per-level factors (bypasses auto-calculation).
                                      E.g., [[1,2,2], [1,2,2], [1,2,2]] for 3 levels.

        Returns:
            dict with keys:
                'format': 'zarr3' or 'zarr2'
                'shape': s0 shape
                'voxel_sizes': s0 voxel sizes
                'axes_names': dimension names
                'chunk_shape': inner chunk shape
                'shard_shape': outer shard shape
                'dtype_size': bytes per element
                'num_levels': total number of levels needed
                'levels': list of dicts for each level:
                    [
                        {
                            "level": 1,
                            "per_level_factor": [1,2,2],      # s0→s1 factor
                            "cumulative_factor": [1,2,2],     # Same for level 1
                            "predicted_shape": [100, 512, 512],
                            "predicted_voxel_sizes": [0.4, 0.325, 0.325],
                            "shard_shape": [1024, 1024, 1024],
                            "chunk_shape": [32, 32, 32]
                        },
                        {
                            "level": 2,
                            "per_level_factor": [1,2,2],      # s1→s2 factor
                            "cumulative_factor": [1,4,4],     # Cumulative from s0
                            ...
                        },
                        ...
                    ]
        """
        if custom_per_level_factors:
            # Use custom factors instead of auto-calculation
            return self._calculate_plan_with_custom_factors(custom_per_level_factors)

        # Use v1's calculate_pyramid_plan as base
        v1_plan = v1_calculate_pyramid_plan(
            self.s0_path,
            min_array_nbytes=min_array_nbytes,
            min_array_shape=min_array_shape
        )

        # Extract per-level factors from v1 plan
        per_level_factors = [level['factor'] for level in v1_plan['pyramid_plan']]

        # Enhance with cumulative factors
        enhanced_levels = []
        for level_info in v1_plan['pyramid_plan']:
            level = level_info['level']

            # Calculate cumulative factor for direct s0→sN downsampling
            cumulative_factor = calculate_cumulative_factors(per_level_factors, level)

            enhanced_level = {
                'level': level,
                'per_level_factor': level_info['factor'],
                'cumulative_factor': cumulative_factor,
                'predicted_shape': level_info['predicted_shape'],
                'predicted_voxel_sizes': level_info['predicted_voxel_sizes'],
                'shard_shape': level_info.get('shard_shape'),
                'chunk_shape': level_info.get('chunk_shape'),
            }
            enhanced_levels.append(enhanced_level)

        # Get source spatial unit for preservation during downsampling
        source_spatial_unit = extract_source_spatial_unit(self.root_path, v1_plan['format'])

        return {
            'format': v1_plan['format'],
            'shape': v1_plan['shape'],
            'voxel_sizes': v1_plan['voxel_sizes'],
            'axes_names': v1_plan['axes_names'],
            'chunk_shape': v1_plan['chunk_shape'],
            'shard_shape': v1_plan.get('shard_shape'),
            'inner_chunk_shape': v1_plan.get('inner_chunk_shape'),
            'dtype_size': v1_plan['dtype_size'],
            'num_levels': v1_plan['num_levels'],
            'levels': enhanced_levels,
            # Keep original v1 plan for compatibility
            'v1_pyramid_plan': v1_plan['pyramid_plan'],
            'source_spatial_unit': source_spatial_unit,
        }

    def _calculate_plan_with_custom_factors(
        self,
        per_level_factors: List[List[int]],
    ) -> Dict[str, Any]:
        """
        Calculate pyramid plan using custom per-level factors.

        This bypasses the automatic anisotropic factor calculation and uses
        user-provided factors for each level. Useful when voxel size info
        is not in the zarr metadata (e.g., stored in BigStitcher XML).

        Args:
            per_level_factors: List of per-level factors, e.g., [[1,2,2], [1,2,2], [1,2,2]]

        Returns:
            Pyramid plan dict (same format as calculate_pyramid_plan)
        """
        # Load s0 metadata to get shape, dtype, axes, etc.
        s0_metadata = self._load_s0_metadata()
        zarr_format = s0_metadata.get('_format', 'zarr3')

        # Get s0 shape
        if zarr_format == 'zarr3':
            s0_shape = s0_metadata.get('shape', [])
        else:  # zarr2
            s0_shape = s0_metadata.get('shape', [])

        # Get axes names
        axes_names = s0_metadata.get('dimension_names')
        if not axes_names:
            # Infer from shape
            ndim = len(s0_shape)
            if ndim == 5:
                axes_names = ['t', 'c', 'z', 'y', 'x']
            elif ndim == 4:
                axes_names = ['c', 'z', 'y', 'x']
            elif ndim == 3:
                axes_names = ['z', 'y', 'x']
            else:
                axes_names = [f'dim_{i}' for i in range(ndim)]

        # Get dtype size
        dtype_str = s0_metadata.get('data_type', 'uint16')
        dtype_size = np.dtype(dtype_str.replace('<', '').replace('>', '').replace('|', '')).itemsize

        # Get chunk/shard shapes
        if zarr_format == 'zarr3':
            chunk_config = s0_metadata.get('chunk_grid', {}).get('configuration', {})
            chunk_shape = chunk_config.get('chunk_shape', [256] * len(s0_shape))
            # For sharding, check codecs
            shard_shape = chunk_shape  # Default to chunk if no sharding
            codecs = s0_metadata.get('codecs', [])
            for codec in codecs:
                if codec.get('name') == 'sharding_indexed':
                    inner_config = codec.get('configuration', {}).get('chunk_shape')
                    if inner_config:
                        shard_shape = chunk_shape
                        chunk_shape = inner_config
                    break
        else:  # zarr2
            chunk_shape = s0_metadata.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', [256] * len(s0_shape))
            shard_shape = chunk_shape  # No sharding in zarr2

        # Validate factor dimensions match s0 shape
        for i, factors in enumerate(per_level_factors):
            if len(factors) != len(s0_shape):
                raise ValueError(
                    f"Factor dimensions mismatch at level {i+1}: "
                    f"got {len(factors)} factors but s0 has {len(s0_shape)} dimensions.\n"
                    f"s0 shape: {s0_shape}, axes: {axes_names}\n"
                    f"Factors: {factors}"
                )

        # Build enhanced levels with cumulative factors and predicted shapes
        enhanced_levels = []
        current_shape = list(s0_shape)
        cumulative = [1] * len(s0_shape)

        for level_idx, factors in enumerate(per_level_factors):
            level = level_idx + 1  # s1, s2, s3, ...

            # Update cumulative factors
            cumulative = [c * f for c, f in zip(cumulative, factors)]

            # Calculate predicted shape
            predicted_shape = [max(1, s // f) for s, f in zip(s0_shape, cumulative)]

            # Calculate predicted voxel sizes (assume 1.0 if unknown - custom factors mean user knows best)
            # We don't have voxel info, so use cumulative factors as relative voxel size
            predicted_voxel_sizes = [float(c) for c in cumulative]

            enhanced_level = {
                'level': level,
                'per_level_factor': factors,
                'cumulative_factor': list(cumulative),
                'predicted_shape': predicted_shape,
                'predicted_voxel_sizes': predicted_voxel_sizes,
                'shard_shape': list(shard_shape),
                'chunk_shape': list(chunk_shape),
            }
            enhanced_levels.append(enhanced_level)

            # Update current shape for next iteration (chained mode)
            current_shape = predicted_shape

        # Get source spatial unit for preservation during downsampling
        source_spatial_unit = s0_metadata.get('source_spatial_unit', 'nanometer')

        return {
            'format': zarr_format,
            'shape': list(s0_shape),
            'voxel_sizes': [1.0] * len(s0_shape),  # Unknown with custom factors
            'axes_names': axes_names,
            'chunk_shape': list(chunk_shape),
            'shard_shape': list(shard_shape),
            'inner_chunk_shape': list(chunk_shape),
            'dtype_size': dtype_size,
            'num_levels': len(per_level_factors),
            'levels': enhanced_levels,
            'custom_factors': True,  # Flag to indicate custom factors were used
            'source_spatial_unit': source_spatial_unit,
        }

    def precreate_all_levels(
        self,
        pyramid_plan: Dict[str, Any],
        use_shard: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Pre-create all level directories and metadata before job submission.

        This is critical for multi-job mode to prevent race conditions.
        All levels are pre-created so jobs can write immediately.

        Supports both Zarr3 (zarr.json) and Zarr2 (.zarray/.zattrs) formats.

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            use_shard: Whether to use sharding (ignored for Zarr2)
            verbose: Print progress messages
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"PRE-CREATING {pyramid_plan['num_levels']} PYRAMID LEVELS")
            print(f"{'='*60}")

        # Get s0 metadata for dtype, format, and compression
        s0_metadata = self._load_s0_metadata()
        dtype = s0_metadata.get('data_type', 'uint16')
        zarr_format = s0_metadata.get('_format', 'zarr3')

        # Detect level naming format (s0/s1 vs 0/1)
        from tensorswitch_v2.utils.metadata_utils import (
            detect_level_format, get_level_name,
            extract_compression_from_zarr3_metadata, extract_compression_from_zarr2_metadata
        )
        prefix = detect_level_format(self.root_path)

        # Extract compression from level 0
        if zarr_format == 'zarr2':
            compressor = extract_compression_from_zarr2_metadata(s0_metadata.get('_zarray', {}))
        else:
            compressor = extract_compression_from_zarr3_metadata(s0_metadata)

        for level_info in pyramid_plan['levels']:
            level = level_info['level']
            level_name = get_level_name(level, prefix)

            if verbose:
                print(f"\nPre-creating {level_name}...")
                print(f"  Shape: {level_info['predicted_shape']}")
                if zarr_format == 'zarr2':
                    # Zarr2 has no sharding, only chunks
                    print(f"  Chunk shape: {pyramid_plan['chunk_shape']}")
                else:
                    # Zarr3 has both shard and inner chunk
                    print(f"  Shard shape: {level_info['shard_shape']}")
                    print(f"  Chunk shape: {level_info['chunk_shape']}")

            if zarr_format == 'zarr2':
                # Pre-create Zarr2 level (directory + .zarray + .zattrs)
                # Zarr2 has no sharding - use chunk_shape from plan
                self._precreate_zarr2_level(
                    level=level,
                    shape=level_info['predicted_shape'],
                    chunk_shape=pyramid_plan['chunk_shape'],
                    dtype=dtype,
                    axes_names=pyramid_plan['axes_names'],
                    prefix=prefix,
                    compressor=compressor,
                    verbose=verbose,
                )
            else:
                # Use Zarr3 precreation with inherited compression
                # Fall back to plan's top-level shapes if level-specific not set
                level_shard = level_info.get('shard_shape') or pyramid_plan.get('shard_shape')
                level_chunk = level_info.get('chunk_shape') or pyramid_plan.get('chunk_shape')
                precreate_zarr3_output(
                    output_path=self.root_path,
                    level=level,
                    output_shape=level_info['predicted_shape'],
                    shard_shape=level_shard if use_shard else level_chunk,
                    chunk_shape=level_chunk,
                    dtype=dtype,
                    use_ome_structure=True,
                    use_v2_encoding=False,
                    axes_order=pyramid_plan['axes_names'],
                    compression=compressor,
                )

            if verbose:
                print(f"  Pre-created {level_name}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"PRE-CREATION COMPLETE")
            print(f"{'='*60}")

    def _precreate_zarr2_level(
        self,
        level: int,
        shape: List[int],
        chunk_shape: List[int],
        dtype: str,
        axes_names: List[str],
        prefix: str = "",
        compressor: Optional[Dict] = None,
        verbose: bool = True,
    ) -> None:
        """
        Pre-create a Zarr2 pyramid level directory with .zarray and .zattrs.

        Args:
            level: Pyramid level number (1, 2, 3, ...)
            shape: Array shape for this level
            chunk_shape: Chunk shape
            dtype: Data type string
            axes_names: Dimension names
            prefix: Level naming prefix ("s" for s1/s2, "" for 1/2)
            compressor: Compressor config from level 0 (default: zstd level 5)
            verbose: Print progress
        """
        # Use detected level naming format
        from tensorswitch_v2.utils.metadata_utils import get_level_name
        level_name = get_level_name(level, prefix)
        level_path = os.path.join(self.root_path, level_name)
        os.makedirs(level_path, exist_ok=True)

        if verbose:
            print(f"  Creating Zarr2 level at: {level_path}")

        # Use dtype from source - preserve endianness if already in Zarr2 format
        dtype_str = str(dtype)
        if dtype_str.startswith(('<', '>', '|')):
            # Already in Zarr2 format (e.g., '>u2', '<u2', '|u1'), use as-is
            zarr_dtype = dtype_str
        else:
            # Convert numpy-style dtype to Zarr2 format (default to little-endian)
            dtype_map = {
                'uint8': '|u1', 'uint16': '<u2', 'uint32': '<u4', 'uint64': '<u8',
                'int8': '|i1', 'int16': '<i2', 'int32': '<i4', 'int64': '<i8',
                'float32': '<f4', 'float64': '<f8',
            }
            zarr_dtype = dtype_map.get(dtype_str, '<u2')

        # Use compression from level 0, or default
        if compressor is None:
            compressor = {"id": "zstd", "level": 5}

        # Write .zarray
        zarray = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunk_shape),
            "dtype": zarr_dtype,
            "compressor": compressor,
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "dimension_separator": "/"
        }
        with open(os.path.join(level_path, '.zarray'), 'w') as f:
            json.dump(zarray, f, indent=2)

        # Write .zattrs with _ARRAY_DIMENSIONS
        zattrs = {
            "_ARRAY_DIMENSIONS": list(axes_names) if axes_names else None
        }
        with open(os.path.join(level_path, '.zattrs'), 'w') as f:
            json.dump(zattrs, f, indent=2)

    def generate_parallel_submission_script(
        self,
        pyramid_plan: Dict[str, Any],
        project: str,
        memory: int = 32,
        wall_time: str = "2:00",
        cores: int = 2,
        use_shard: bool = True,
        downsample_method: str = "auto",
    ) -> str:
        """
        Generate bash script for parallel submission of all levels.

        Unlike v1's bwait-chained approach, this generates a script that
        submits all levels simultaneously since each level reads from s0.

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            project: LSF project to charge
            memory: Memory per job in GB
            wall_time: Wall time per job
            cores: Cores per job
            downsample_method: TensorStore downsample method (default: "auto")
            use_shard: Whether to use sharding

        Returns:
            Bash script as string
        """
        tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        python_path = sys.executable

        # Quote paths for safe shell usage (handles spaces)
        q_tensorswitch_dir = shlex.quote(tensorswitch_dir)
        q_python_path = shlex.quote(python_path)
        q_root_path = shlex.quote(self.root_path)
        q_s0_path = shlex.quote(self.s0_path)

        script = f"""#!/bin/bash
set -e

echo "================================================================"
echo "PARALLEL PYRAMID GENERATION (All levels from s0)"
echo "================================================================"
echo "Dataset: {self.root_path}"
echo "Levels: s1 to s{pyramid_plan['num_levels']}"
echo "Mode: All levels submitted simultaneously (no waiting)"
echo ""

# Submit all levels in parallel - each level downsamples directly from s0
ALL_JOB_IDS=""

"""

        for level_info in pyramid_plan['levels']:
            level = level_info['level']
            cumulative_factors = ",".join(map(str, level_info['cumulative_factor']))
            # Use level-specific shape, or fall back to plan's top-level shape (preserves source dimensions)
            level_shard = level_info.get('shard_shape') or pyramid_plan.get('shard_shape') or []
            level_chunk = level_info.get('chunk_shape') or pyramid_plan.get('chunk_shape') or []
            shard_shape = ",".join(map(str, level_shard)) if level_shard else ""
            chunk_shape = ",".join(map(str, level_chunk)) if level_chunk else ""

            script += f"""
# Submit s{level} (directly from s0 with cumulative factor {level_info['cumulative_factor']})
echo "Submitting s{level}..."
S{level}_OUTPUT=$(cd {q_tensorswitch_dir} && {q_python_path} -m tensorswitch_v2 \\
    --downsample \\
    -i {q_s0_path} \\
    -o {q_root_path} \\
    --target_level {level} \\
    --cumulative_factors "{cumulative_factors}" \\
    --use_shard {1 if use_shard else 0} \\
    --shard_shape "{shard_shape}" \\
    --chunk_shape "{chunk_shape}" \\
    --downsample_method {downsample_method} \\
    --submit \\
    -P {project} \\
    --memory {memory} \\
    --wall_time {wall_time} \\
    --cores {cores} \\
    2>&1)

S{level}_JOBS=$(echo "$S{level}_OUTPUT" | grep -oP 'Job <\\K[0-9]+(?=>)' | tr '\\n' ' ')
echo "  s{level} jobs: $S{level}_JOBS"
ALL_JOB_IDS="$ALL_JOB_IDS $S{level}_JOBS"
"""

        script += f"""
echo ""
echo "================================================================"
echo "ALL LEVELS SUBMITTED"
echo "================================================================"
echo "Total jobs: $(echo $ALL_JOB_IDS | wc -w)"
echo "Job IDs: $ALL_JOB_IDS"
echo ""
echo "Waiting for all jobs to complete..."

# Build wait condition for all jobs
WAIT_CONDITION=""
for JOB_ID in $ALL_JOB_IDS; do
    if [ -z "$WAIT_CONDITION" ]; then
        WAIT_CONDITION="done($JOB_ID)"
    else
        WAIT_CONDITION="$WAIT_CONDITION && done($JOB_ID)"
    fi
done

if [ -n "$WAIT_CONDITION" ]; then
    echo "Wait condition: $WAIT_CONDITION"
    bwait -w "$WAIT_CONDITION" 2>&1 || echo "Warning: bwait returned non-zero"
fi

echo ""
echo "All jobs complete. Updating root metadata..."
{q_python_path} -c "from tensorswitch_v2.utils import update_ome_metadata_if_needed; update_ome_metadata_if_needed({repr(self.root_path)}, use_ome_structure=True)"

echo ""
echo "================================================================"
echo "PYRAMID GENERATION COMPLETE: {self.root_path}"
echo "================================================================"
"""

        return script

    def _submit_metadata_coordinator_job(
        self,
        level_job_ids: List[str],
        project: str,
        verbose: bool = True,
    ) -> Optional[str]:
        """
        Submit a coordinator job that waits for all level jobs and updates metadata.

        This job uses bwait to wait for all downsampling jobs to complete,
        then updates the root zarr.json with all pyramid levels.

        Args:
            level_job_ids: List of job IDs from level submissions
            project: LSF project to charge
            verbose: Print progress messages

        Returns:
            Coordinator job ID, or None if submission failed
        """
        if not level_job_ids:
            if verbose:
                print("No level jobs to coordinate - skipping metadata coordinator")
            return None

        # Build wait condition
        wait_conditions = [f"done({jid})" for jid in level_job_ids]
        wait_expr = " && ".join(wait_conditions)

        # Build the coordinator script - write to shared filesystem (next to output)
        tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        python_path = sys.executable

        # Job name
        dataset_name = os.path.basename(self.root_path)
        job_name = f"tsv2_coord_{dataset_name}"[:128]

        # Log/script directory - use shared filesystem
        log_dir = os.path.join(os.path.dirname(self.root_path), "output")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
        error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

        # Write script to shared filesystem (not /tmp which is local)
        script_path = os.path.join(log_dir, f"coordinator_{dataset_name}.sh")

        # Quote paths for safe shell usage (handles spaces)
        q_tensorswitch_dir = shlex.quote(tensorswitch_dir)
        q_python_path = shlex.quote(python_path)
        q_root_path = shlex.quote(self.root_path)

        coordinator_script = f"""#!/bin/bash
set -e

echo "=========================================="
echo "METADATA COORDINATOR JOB"
echo "=========================================="
echo "Waiting for {len(level_job_ids)} downsampling jobs..."
echo "Job IDs: {' '.join(level_job_ids)}"
echo ""

# Wait for all level jobs to complete
bwait -w "{wait_expr}" 2>&1 || echo "Warning: bwait returned non-zero (jobs may have failed)"

echo ""
echo "All level jobs completed. Updating root metadata..."

# Update OME-NGFF metadata
cd {q_tensorswitch_dir}
{q_python_path} -c "from tensorswitch_v2.utils import update_ome_metadata_if_needed; update_ome_metadata_if_needed({repr(self.root_path)}, use_ome_structure=True)"

echo ""
echo "=========================================="
echo "PYRAMID COMPLETE: {self.root_path}"
echo "=========================================="
"""

        with open(script_path, 'w') as f:
            f.write(coordinator_script)
        os.chmod(script_path, 0o755)

        # Submit coordinator job
        # Use /bin/bash to run script - handles paths with spaces properly
        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", "1",
            "-W", "0:30",  # 30 min should be plenty for bwait + metadata update
            "-M", "4GB",
            "-P", project,
            "-o", log_path,
            "-e", error_path,
            "/bin/bash", script_path
        ]

        if verbose:
            print(f"\nSubmitting metadata coordinator job...")
            print(f"  Waits for: {len(level_job_ids)} jobs")
            print(f"  Then updates: {self.root_path}/zarr.json")

        result = subprocess.run(bsub_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if verbose:
                print(f"  ERROR: {result.stderr}")
            return None

        # Extract job ID
        import re
        match = re.search(r'Job <(\d+)>', result.stdout)
        if match:
            coord_job_id = match.group(1)
            if verbose:
                print(f"  Coordinator job submitted: {coord_job_id}")
            return coord_job_id

        return None

    def submit_chained_pyramid(
        self,
        pyramid_plan: Dict[str, Any],
        project: str,
        memory: Optional[int] = None,
        wall_time: Optional[str] = None,
        cores: Optional[int] = None,
        use_shard: bool = True,
        downsample_method: str = "auto",
        dry_run: bool = False,
        verbose: bool = True,
    ) -> str:
        """
        Submit chained pyramid generation via a coordinator job.

        Uses the v1 chained approach where each level reads from the previous
        level (s1←s0, s2←s1, etc.). A coordinator job runs the entire sequence
        with bwait between levels.

        This is more efficient than parallel-from-s0 for deep pyramids because
        each level only reads ~8 voxels per output instead of exponentially more.

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            project: LSF project to charge
            memory: Memory per job in GB (None = auto-calculate per level)
            wall_time: Wall time per job (None = auto-calculate per level)
            cores: Cores per job (None = auto-calculate based on memory)
            use_shard: Whether to use sharding
            downsample_method: TensorStore downsample method (default: "auto")
            dry_run: If True, print commands but don't execute
            verbose: Print progress messages

        Returns:
            Coordinator job ID
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"CHAINED PYRAMID SUBMISSION")
            print(f"{'='*60}")
            print(f"Dataset: {self.root_path}")
            print(f"Levels: s1 to s{pyramid_plan['num_levels']}")
            print(f"Mode: Chained (s1←s0, s2←s1, s3←s2, ...)")
            print(f"Project: {project}")
            print(f"Resource mode: {'User-specified' if memory else 'Auto-calculated per level'}")

        # Generate coordinator script
        script = self._generate_chained_coordinator_script(
            pyramid_plan=pyramid_plan,
            project=project,
            memory=memory,
            wall_time=wall_time,
            cores=cores,
            use_shard=use_shard,
            downsample_method=downsample_method,
        )

        # Write script to shared filesystem
        log_dir = os.path.join(os.path.dirname(self.root_path), "output")
        os.makedirs(log_dir, exist_ok=True)

        dataset_name = os.path.basename(self.root_path)
        script_path = os.path.join(log_dir, f"chained_pyramid_{dataset_name}.sh")

        if verbose:
            print(f"\nCoordinator script: {script_path}")

        if dry_run:
            print(f"\n[DRY RUN] Would write script to: {script_path}")
            print(f"\n--- Script content ---\n{script}\n--- End script ---")
            return "DRY_RUN"

        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        # Calculate total wall time for coordinator (sum of all level wall times + overhead)
        # This ensures coordinator stays alive long enough for all bwaits to complete
        dtype_size = pyramid_plan.get('dtype_size', 2)
        shapes = [pyramid_plan['shape']]
        for level_info in pyramid_plan['levels']:
            shapes.append(level_info['predicted_shape'])

        total_level_minutes = 0
        for level_info in pyramid_plan['levels']:
            level = level_info['level']
            source_shape = shapes[level - 1]
            level_shape = level_info['predicted_shape']
            level_shard_shape = level_info.get('shard_shape') or pyramid_plan.get('shard_shape') or [1, 1, 1024, 1024, 1024]

            if wall_time is None:
                level_wall_time = _calculate_downsample_wall_time(
                    source_shape, level_shape, level_shard_shape, dtype_size
                )
            else:
                level_wall_time = wall_time

            # Parse H:MM format and sum
            parts = level_wall_time.split(':')
            total_level_minutes += int(parts[0]) * 60 + int(parts[1])

        # Add 1 hour overhead for bwait operations and metadata update
        total_minutes = total_level_minutes + 60
        # Convert to hours, round up, cap at 48 hours (typical LSF max)
        coord_wall_hours = min((total_minutes + 59) // 60, 48)
        coord_wall_time = f"{coord_wall_hours}:00"

        # Submit coordinator job
        job_name = f"tsv2_pyramid_{dataset_name}"[:128]
        log_path = os.path.join(log_dir, f"output__{job_name}_%J.log")
        error_path = os.path.join(log_dir, f"error__{job_name}_%J.log")

        # Use /bin/bash to run script - handles paths with spaces properly
        # When bsub creates wrapper script, it writes args verbatim; bash properly
        # handles the quoted script path as an argument
        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", "1",
            "-W", coord_wall_time,
            "-M", "8GB",
            "-R", "rusage[mem=8192]",
            "-P", project,
            "-o", log_path,
            "-e", error_path,
            "/bin/bash", script_path
        ]

        if verbose:
            print(f"\nSubmitting coordinator job...")
            print(f"  Wall time: {coord_wall_time}")
            print(f"  Log: {log_path}")

        result = subprocess.run(bsub_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            raise RuntimeError(f"bsub failed: {result.stderr}")

        # Extract job ID
        import re
        match = re.search(r'Job <(\d+)>', result.stdout)
        if match:
            coord_job_id = match.group(1)
            if verbose:
                print(f"  Coordinator job submitted: {coord_job_id}")
                print(f"\n{'='*60}")
                print(f"SUBMISSION COMPLETE")
                print(f"{'='*60}")
                print(f"Coordinator job: {coord_job_id}")
                print(f"Monitor with: bjobs {coord_job_id}")
                print(f"View log: tail -f {log_path}")
            return coord_job_id

        raise RuntimeError(f"Could not extract job ID from: {result.stdout}")

    def _generate_chained_coordinator_script(
        self,
        pyramid_plan: Dict[str, Any],
        project: str,
        memory: Optional[int] = None,
        wall_time: Optional[str] = None,
        cores: Optional[int] = None,
        use_shard: bool = True,
        downsample_method: str = "auto",
    ) -> str:
        """
        Generate bash script for chained pyramid generation.

        The script submits each level sequentially, waiting (bwait) for the
        previous level to complete before submitting the next.

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            project: LSF project to charge
            memory: Memory per job in GB (None = auto-calculate)
            wall_time: Wall time per job (None = auto-calculate)
            cores: Cores per job (None = auto-calculate)
            use_shard: Whether to use sharding
            downsample_method: TensorStore downsample method (default: "auto")

        Returns:
            Bash script as string
        """
        tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        python_path = sys.executable

        # Quote paths for safe shell usage (handles spaces)
        q_tensorswitch_dir = shlex.quote(tensorswitch_dir)
        q_python_path = shlex.quote(python_path)
        q_root_path = shlex.quote(self.root_path)

        num_levels = pyramid_plan['num_levels']
        dtype_size = pyramid_plan.get('dtype_size', 2)

        # Detect level naming format
        from tensorswitch_v2.utils.metadata_utils import detect_level_format, get_level_name
        prefix = detect_level_format(self.root_path)
        first_level = get_level_name(1, prefix)
        last_level = get_level_name(num_levels, prefix)

        # Build the script header
        script = f"""#!/bin/bash
set -e

echo "========================================================================"
echo "CHAINED PYRAMID GENERATION"
echo "========================================================================"
echo "Dataset: {self.root_path}"
echo "Levels: {first_level} to {last_level}"
echo "Mode: Chained (each level reads from previous level)"
echo "Started: $(date)"
echo ""

# Helper function to extract job IDs from bsub output
extract_job_ids() {{
    grep -oP 'Job <\\K[0-9]+(?=>)' || true
}}

# Helper function to submit a level and wait for completion
submit_and_wait() {{
    local level=$1
    local source_path=$2
    local factor=$3
    local mem=$4
    local wtime=$5
    local ncores=$6
    local shard_shape=$7
    local chunk_shape=$8

    echo ""
    echo "============================================================"
    echo "LEVEL s$level (reading from $source_path)"
    echo "============================================================"
    echo "  Factor: $factor"
    echo "  Memory: ${{mem}}GB, Wall time: $wtime, Cores: $ncores"
    echo "  Submitting..."

    output=$(cd {q_tensorswitch_dir} && {q_python_path} -m tensorswitch_v2 \\
        --downsample \\
        -i "$source_path" \\
        -o {q_root_path} \\
        --target_level $level \\
        --cumulative_factors "$factor" \\
        --use_shard {1 if use_shard else 0} \\
        --shard_shape "$shard_shape" \\
        --chunk_shape "$chunk_shape" \\
        --downsample_method {downsample_method} \\
        --submit \\
        -P {project} \\
        --memory $mem \\
        --wall_time $wtime \\
        --cores $ncores \\
        2>&1)

    job_ids=$(echo "$output" | extract_job_ids | tr '\\n' ' ')

    if [ -z "$job_ids" ]; then
        echo "ERROR: No job IDs returned for s$level"
        echo "Output was: $output"
        exit 1
    fi

    echo "  Jobs submitted: $job_ids"

    # Build wait condition
    wait_condition=""
    for job_id in $job_ids; do
        if [ -z "$wait_condition" ]; then
            wait_condition="done($job_id)"
        else
            wait_condition="$wait_condition && done($job_id)"
        fi
    done

    echo "  Waiting for jobs to complete..."
    bwait -w "$wait_condition" 2>&1 || true

    echo "  s$level complete: $(date)"
}}

"""

        # Add each level with chained source path
        # Build list of shapes for resource calculation
        shapes = [pyramid_plan['shape']]  # s0 shape
        for level_info in pyramid_plan['levels']:
            shapes.append(level_info['predicted_shape'])

        for i, level_info in enumerate(pyramid_plan['levels']):
            level = level_info['level']
            per_level_factor = level_info['per_level_factor']
            factor_str = ",".join(map(str, per_level_factor))

            # Source is previous level (0 for 1, 1 for 2, etc. - or s0 for s1 if using s-prefix)
            source_level = level - 1
            source_level_name = get_level_name(source_level, prefix)
            source_path = os.path.join(self.root_path, source_level_name)
            source_shape = shapes[source_level]  # Shape of source level

            # Use level-specific shape, or fall back to plan's top-level shape (preserves source dimensions)
            level_shard_shape = level_info.get('shard_shape') or pyramid_plan.get('shard_shape') or [1024, 1024, 1024]
            level_chunk_shape = level_info.get('chunk_shape') or pyramid_plan.get('chunk_shape') or [256, 256, 256]
            level_shape = level_info['predicted_shape']

            shard_shape_str = ",".join(map(str, level_shard_shape))
            chunk_shape_str = ",".join(map(str, level_chunk_shape))

            # Auto-calculate resources based on source level (not s0!)
            if memory is None:
                level_memory = _calculate_downsample_memory(
                    source_shape, level_shape, level_shard_shape, dtype_size
                )
            else:
                level_memory = memory

            if wall_time is None:
                level_wall_time = _calculate_downsample_wall_time(
                    source_shape, level_shape, level_shard_shape, dtype_size
                )
            else:
                level_wall_time = wall_time

            if cores is None:
                level_cores = _calculate_downsample_cores(level_memory)
            else:
                level_cores = cores

            # Enforce cluster policy: 15 GB per core minimum
            level_memory = max(level_memory, level_cores * 15)

            level_name = get_level_name(level, prefix)
            script += f"""# {level_name}: reads from {source_level_name}, factor {per_level_factor}
submit_and_wait {level} "{source_path}" "{factor_str}" {level_memory} "{level_wall_time}" {level_cores} "{shard_shape_str}" "{chunk_shape_str}"

"""

        # Add final metadata update and completion message
        script += f"""
echo ""
echo "============================================================"
echo "UPDATING ROOT METADATA"
echo "============================================================"
echo "All levels complete, updating root zarr.json..."
{q_python_path} -c "from tensorswitch_v2.utils import update_ome_metadata_if_needed; update_ome_metadata_if_needed({repr(self.root_path)}, use_ome_structure=True)"
echo "Metadata update complete."

echo ""
echo "========================================================================"
echo "PYRAMID GENERATION COMPLETE"
echo "========================================================================"
echo "Dataset: {self.root_path}"
echo "Levels: {get_level_name(0, prefix)} to {get_level_name(num_levels, prefix)}"
echo "Completed: $(date)"
echo ""
"""

        return script

    # Keep old method name as alias for backward compatibility
    def submit_all_levels_parallel(self, *args, **kwargs):
        """Deprecated: Use submit_chained_pyramid instead."""
        print("WARNING: submit_all_levels_parallel is deprecated, using submit_chained_pyramid")
        return self.submit_chained_pyramid(*args, **kwargs)

    def update_root_metadata(self, verbose: bool = True) -> None:
        """
        Update root metadata (zarr.json or .zattrs) with all pyramid levels.

        This should be called after all downsampling jobs complete.
        It scans for existing s* directories and updates the multiscales metadata.

        Args:
            verbose: Print progress messages
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"UPDATING ROOT METADATA")
            print(f"{'='*60}")

        # Use v1's unified metadata update function which handles both zarr3 and zarr2
        update_ome_metadata_if_needed(self.root_path, use_ome_structure=True)

        if verbose:
            print(f"{'='*60}\n")

    def print_pyramid_plan(self, pyramid_plan: Dict[str, Any]) -> None:
        """Print a human-readable summary of the pyramid plan."""
        print(f"\n{'='*70}")
        print(f"PYRAMID PLAN: {self.root_path}")
        print(f"{'='*70}")
        print(f"Format: {pyramid_plan['format']}")
        print(f"s0 shape: {pyramid_plan['shape']}")
        print(f"s0 voxel sizes: {pyramid_plan['voxel_sizes']}")
        print(f"Source spatial unit: {pyramid_plan.get('source_spatial_unit', 'nanometer')} (preserved)")
        print(f"Dimension names: {pyramid_plan['axes_names']}")
        print(f"Shard shape: {pyramid_plan.get('shard_shape')}")
        print(f"Chunk shape: {pyramid_plan['chunk_shape']}")
        print(f"Levels needed: {pyramid_plan['num_levels']}")
        print()
        print(f"{'Level':<6} {'Per-Level Factor':<20} {'Cumulative Factor':<20} {'Shape':<25}")
        print(f"{'-'*70}")

        for level_info in pyramid_plan['levels']:
            level = f"s{level_info['level']}"
            per_level = str(level_info['per_level_factor'])
            cumulative = str(level_info['cumulative_factor'])
            shape = str(level_info['predicted_shape'])
            print(f"{level:<6} {per_level:<20} {cumulative:<20} {shape:<25}")

        print(f"{'='*70}\n")


def create_pyramid_parallel(
    s0_path: str,
    project: str,
    min_array_nbytes: Optional[int] = None,
    min_array_shape: Optional[List[int]] = None,
    memory: Optional[int] = None,
    wall_time: Optional[str] = None,
    cores: Optional[int] = None,
    use_shard: bool = True,
    downsample_method: str = "auto",
    dry_run: bool = False,
    verbose: bool = True,
    custom_per_level_factors: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to create full pyramid with chained job submission.

    This is the main entry point for CLI --auto_multiscale mode.
    Uses the chained approach where each level reads from the previous level
    (s1←s0, s2←s1, etc.) which is much more efficient than parallel-from-s0.

    Args:
        s0_path: Path to s0 array
        project: LSF project to charge
        min_array_nbytes: Stop when array size < this
        min_array_shape: Stop when all dims < this
        memory: Memory per job in GB (None = auto-calculate per level)
        wall_time: Wall time per job (None = auto-calculate per level)
        cores: Cores per job (None = auto-calculate based on memory)
        use_shard: Whether to use sharding
        downsample_method: TensorStore downsample method (default: "auto")
        dry_run: If True, print commands but don't execute
        verbose: Print progress messages
        custom_per_level_factors: Custom per-level factors (bypasses auto-calculation).
                                  E.g., [[1,2,2], [1,2,2], [1,2,2]] for 3 levels with Z-skip.

    Returns:
        dict with 'pyramid_plan' and 'coordinator_job_id' keys

    Example:
        >>> result = create_pyramid_parallel(
        ...     s0_path="/data/dataset.zarr/s0",
        ...     project="scicompsoft"
        ... )
        >>> print(f"Coordinator job: {result['coordinator_job_id']}")

        >>> # With custom factors (skip Z downsampling)
        >>> result = create_pyramid_parallel(
        ...     s0_path="/data/dataset.zarr/s0",
        ...     project="scicompsoft",
        ...     custom_per_level_factors=[[1,2,2], [1,2,2], [1,2,2], [1,2,2]]
        ... )
    """
    planner = PyramidPlanner(s0_path)

    # Resolve 'auto' downsample method based on input path
    resolved_method = resolve_downsample_method(downsample_method, s0_path)
    if verbose and downsample_method == 'auto':
        print(f"Auto-detected downsample method: {resolved_method}")

    # Calculate pyramid plan
    pyramid_plan = planner.calculate_pyramid_plan(
        min_array_nbytes=min_array_nbytes,
        min_array_shape=min_array_shape,
        custom_per_level_factors=custom_per_level_factors,
    )

    if verbose:
        planner.print_pyramid_plan(pyramid_plan)

    # Pre-create all levels (directories and per-level zarr.json)
    if not dry_run:
        planner.precreate_all_levels(pyramid_plan, use_shard=use_shard, verbose=verbose)
        # NOTE: Root metadata update is now done by coordinator after all levels complete

    # Submit chained pyramid via coordinator job
    # Coordinator handles: submit s1 → bwait → submit s2 → bwait → ... → update metadata
    coordinator_job_id = planner.submit_chained_pyramid(
        pyramid_plan,
        project=project,
        memory=memory,
        wall_time=wall_time,
        cores=cores,
        use_shard=use_shard,
        downsample_method=resolved_method,
        dry_run=dry_run,
        verbose=verbose
    )

    return {
        'pyramid_plan': pyramid_plan,
        'coordinator_job_id': coordinator_job_id,
    }
