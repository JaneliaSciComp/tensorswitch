"""
Pyramid Planner for TensorSwitch Phase 5 architecture.

Provides multi-level pyramid planning and parallel job submission.
The key innovation is calculating cumulative factors that enable
all levels to downsample directly from s0 in parallel.

Key Features:
- Pre-calculate complete pyramid plan before submission
- Calculate cumulative factors for each level (s0→sN direct)
- Submit all level jobs in parallel (no bwait chaining)
- Reuse Yurii Zubov's anisotropic downsampling algorithm from v1
"""

import os
import json
import subprocess
import sys
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Import utility functions from existing tensorswitch
from tensorswitch.utils import (
    calculate_pyramid_plan as v1_calculate_pyramid_plan,
    calculate_anisotropic_downsample_factors,
    precreate_zarr3_output,
    get_tensorstore_context,
    update_ome_metadata_if_needed,
)

from .downsampler import calculate_cumulative_factors


def _calculate_downsample_memory(
    s0_shape: List[int],
    level_shape: List[int],
    shard_shape: List[int],
    dtype_size: int = 2,
) -> int:
    """
    Calculate memory in GB for a downsampling job.

    Downsampling reads from s0 and writes to the target level.
    Memory needed:
    - Read buffer: portion of s0 that maps to output shard
    - Write buffer: output shard
    - Processing overhead

    Args:
        s0_shape: Shape of s0 array
        level_shape: Shape of target level
        shard_shape: Shard shape for output
        dtype_size: Bytes per element (default: 2 for uint16)

    Returns:
        Memory in GB (rounded to nearest 5, min 8, max 128)
    """
    import math

    # Calculate s0 data size
    s0_size_gb = (np.prod(s0_shape) * dtype_size) / (1024 ** 3)

    # Calculate output shard size
    shard_size_gb = (np.prod(shard_shape) * dtype_size) / (1024 ** 3)

    # Calculate total shards in output
    total_shards = int(np.prod(np.ceil(np.array(level_shape) / np.array(shard_shape)).astype(int)))

    # Memory estimation:
    # 1. TensorStore downsample driver needs to read s0 data proportional to output
    #    For each output chunk, it reads corresponding s0 region (factor x larger)
    # 2. Write buffer for output shard
    # 3. Processing overhead

    # Estimate read buffer based on cumulative factor
    # If level is s3 with factor [4,8,8], each output voxel needs 4*8*8=256 input voxels
    cumulative_factor = [s0_shape[i] / level_shape[i] for i in range(len(s0_shape))]
    total_factor = np.prod(cumulative_factor)

    # Read buffer: shard_size * factor (but capped by s0 size)
    read_buffer_gb = min(shard_size_gb * total_factor, s0_size_gb * 0.1)

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
    s0_shape: List[int],
    level_shape: List[int],
    shard_shape: List[int],
    dtype_size: int = 2,
) -> str:
    """
    Calculate wall time for a downsampling job.

    Args:
        s0_shape: Shape of s0 array
        level_shape: Shape of target level
        shard_shape: Shard shape for output
        dtype_size: Bytes per element

    Returns:
        Wall time string in H:MM format
    """
    import math

    # Calculate sizes
    s0_size_gb = (np.prod(s0_shape) * dtype_size) / (1024 ** 3)
    level_size_gb = (np.prod(level_shape) * dtype_size) / (1024 ** 3)
    shard_size_gb = (np.prod(shard_shape) * dtype_size) / (1024 ** 3)

    # Calculate total shards
    total_shards = int(np.prod(np.ceil(np.array(level_shape) / np.array(shard_shape)).astype(int)))

    # Time estimation based on empirical observations:
    # - Small shards (<0.1 GB): ~1 min each
    # - Medium shards (0.1-1 GB): ~3 min each
    # - Large shards (>1 GB): ~5 min each
    # Plus overhead for reading s0 data

    if shard_size_gb < 0.1:
        minutes_per_shard = 1
    elif shard_size_gb < 1.0:
        minutes_per_shard = 3
    else:
        minutes_per_shard = 5

    # Base time for processing
    base_minutes = minutes_per_shard * total_shards

    # Add overhead for s0 reading (larger s0 = more read time)
    if s0_size_gb > 100:
        read_overhead = 10
    elif s0_size_gb > 10:
        read_overhead = 5
    else:
        read_overhead = 2

    # Total with 2x safety margin, round to 15 min
    total_minutes = int(math.ceil((base_minutes + read_overhead) * 2 / 15) * 15)
    total_minutes = max(15, min(total_minutes, 240))  # 15 min to 4 hours

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

    The key innovation over v1 is that instead of sequential downsampling
    (s0→s1, wait, s1→s2, wait, ...), this planner enables parallel downsampling
    where all levels are computed directly from s0 simultaneously.

    Design Principles:
    - All levels downsample from s0 (not from previous level)
    - Jobs are independent and can run in parallel
    - Pre-creation happens before any job submission
    - Cumulative factors are calculated for direct s0→sN downsampling

    Example:
        >>> planner = PyramidPlanner("/data/dataset.zarr/s0")
        >>> plan = planner.calculate_pyramid_plan()
        >>> print(f"Need {plan['num_levels']} levels")
        >>> for level in plan['levels']:
        ...     print(f"s{level['level']}: cumulative_factor={level['cumulative_factor']}")
        >>>
        >>> # Submit all jobs in parallel
        >>> job_ids = planner.submit_all_levels_parallel(plan, project="scicompsoft")
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
        """Load metadata from s0 array."""
        if self._s0_metadata is not None:
            return self._s0_metadata

        zarr_json_path = os.path.join(self.s0_path, 'zarr.json')
        if not os.path.exists(zarr_json_path):
            raise FileNotFoundError(f"s0 zarr.json not found at {zarr_json_path}")

        with open(zarr_json_path, 'r') as f:
            self._s0_metadata = json.load(f)

        return self._s0_metadata

    def calculate_pyramid_plan(
        self,
        min_array_nbytes: Optional[int] = None,
        min_array_shape: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate complete pyramid plan with cumulative factors for parallel execution.

        This extends v1's calculate_pyramid_plan by adding cumulative factors
        that enable direct s0→sN downsampling for each level.

        Args:
            min_array_nbytes: Stop when array size < this (default: chunk_nbytes)
            min_array_shape: Stop when all dims < this (default: chunk_shape)

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

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            use_shard: Whether to use sharding
            verbose: Print progress messages
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"PRE-CREATING {pyramid_plan['num_levels']} PYRAMID LEVELS")
            print(f"{'='*60}")

        # Get s0 metadata for dtype
        s0_metadata = self._load_s0_metadata()
        dtype = s0_metadata.get('data_type', 'uint16')

        for level_info in pyramid_plan['levels']:
            level = level_info['level']

            if verbose:
                print(f"\nPre-creating s{level}...")
                print(f"  Shape: {level_info['predicted_shape']}")
                print(f"  Shard shape: {level_info['shard_shape']}")
                print(f"  Chunk shape: {level_info['chunk_shape']}")

            # Use unified precreate_zarr3_output which handles both directories and metadata
            # Function signature: (output_path, level, output_shape, shard_shape, chunk_shape,
            #                       dtype, use_ome_structure, use_v2_encoding, axes_order, ...)
            precreate_zarr3_output(
                output_path=self.root_path,
                level=level,
                output_shape=level_info['predicted_shape'],
                shard_shape=level_info['shard_shape'] if use_shard else level_info['chunk_shape'],
                chunk_shape=level_info['chunk_shape'],
                dtype=dtype,
                use_ome_structure=True,
                use_v2_encoding=False,
                axes_order=pyramid_plan['axes_names'],
            )

            if verbose:
                print(f"  Pre-created s{level}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"PRE-CREATION COMPLETE")
            print(f"{'='*60}")

    def generate_parallel_submission_script(
        self,
        pyramid_plan: Dict[str, Any],
        project: str,
        memory: int = 32,
        wall_time: str = "2:00",
        cores: int = 2,
        use_shard: bool = True,
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
            use_shard: Whether to use sharding

        Returns:
            Bash script as string
        """
        tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        python_path = sys.executable

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
            shard_shape = ",".join(map(str, level_info['shard_shape'])) if level_info.get('shard_shape') else ""
            chunk_shape = ",".join(map(str, level_info['chunk_shape'])) if level_info.get('chunk_shape') else ""

            script += f"""
# Submit s{level} (directly from s0 with cumulative factor {level_info['cumulative_factor']})
echo "Submitting s{level}..."
S{level}_OUTPUT=$(cd {tensorswitch_dir} && {python_path} -m tensorswitch_v2 \\
    --downsample \\
    -i "{self.s0_path}" \\
    -o "{self.root_path}" \\
    --target_level {level} \\
    --cumulative_factors "{cumulative_factors}" \\
    --use_shard {1 if use_shard else 0} \\
    --shard_shape "{shard_shape}" \\
    --chunk_shape "{chunk_shape}" \\
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
{python_path} -c "from tensorswitch.utils import update_ome_metadata_if_needed; update_ome_metadata_if_needed('{self.root_path}', use_ome_structure=True)"

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
cd {tensorswitch_dir}
{python_path} -c "from tensorswitch.utils import update_ome_metadata_if_needed; update_ome_metadata_if_needed('{self.root_path}', use_ome_structure=True)"

echo ""
echo "=========================================="
echo "PYRAMID COMPLETE: {self.root_path}"
echo "=========================================="
"""

        with open(script_path, 'w') as f:
            f.write(coordinator_script)
        os.chmod(script_path, 0o755)

        # Submit coordinator job
        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", "1",
            "-W", "0:30",  # 30 min should be plenty for bwait + metadata update
            "-M", "4GB",
            "-P", project,
            "-o", log_path,
            "-e", error_path,
            script_path
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

    def submit_all_levels_parallel(
        self,
        pyramid_plan: Dict[str, Any],
        project: str,
        memory: Optional[int] = None,
        wall_time: Optional[str] = None,
        cores: Optional[int] = None,
        use_shard: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        """
        Submit all downsampling jobs for all levels in parallel.

        Each level is submitted independently since they all read from s0.
        A coordinator job is submitted at the end to wait for all levels
        and update the root metadata.

        Args:
            pyramid_plan: Output from calculate_pyramid_plan()
            project: LSF project to charge
            memory: Memory per job in GB (None = auto-calculate per level)
            wall_time: Wall time per job (None = auto-calculate per level)
            cores: Cores per job (None = auto-calculate based on memory)
            use_shard: Whether to use sharding
            dry_run: If True, print commands but don't execute
            verbose: Print progress messages

        Returns:
            List of all submitted job IDs (including coordinator job)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"PARALLEL JOB SUBMISSION")
            print(f"{'='*60}")
            print(f"Dataset: {self.root_path}")
            print(f"Levels: s1 to s{pyramid_plan['num_levels']}")
            print(f"Project: {project}")
            print(f"Resource mode: {'User-specified' if memory else 'Auto-calculated per level'}")
            print(f"Dry run: {dry_run}")

        tensorswitch_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        python_path = sys.executable

        all_job_ids = []

        # Get s0 shape and dtype for resource calculation
        s0_shape = pyramid_plan['shape']
        dtype_size = pyramid_plan.get('dtype_size', 2)

        for level_info in pyramid_plan['levels']:
            level = level_info['level']
            cumulative_factors = ",".join(map(str, level_info['cumulative_factor']))
            level_shard_shape = level_info.get('shard_shape') or [1024, 1024, 1024]
            level_chunk_shape = level_info.get('chunk_shape') or [32, 32, 32]
            level_shape = level_info['predicted_shape']

            shard_shape_str = ",".join(map(str, level_shard_shape))
            chunk_shape_str = ",".join(map(str, level_chunk_shape))

            # Auto-calculate resources if not specified
            if memory is None:
                level_memory = _calculate_downsample_memory(
                    s0_shape, level_shape, level_shard_shape, dtype_size
                )
            else:
                level_memory = memory

            if wall_time is None:
                level_wall_time = _calculate_downsample_wall_time(
                    s0_shape, level_shape, level_shard_shape, dtype_size
                )
            else:
                level_wall_time = wall_time

            if cores is None:
                level_cores = _calculate_downsample_cores(level_memory)
            else:
                level_cores = cores

            cmd = [
                python_path, "-m", "tensorswitch_v2",
                "--downsample",
                "-i", self.s0_path,
                "-o", self.root_path,
                "--target_level", str(level),
                "--cumulative_factors", cumulative_factors,
                "--use_shard", "1" if use_shard else "0",
                "--submit",
                "-P", project,
                "--memory", str(level_memory),
                "--wall_time", level_wall_time,
                "--cores", str(level_cores),
            ]

            if shard_shape_str:
                cmd.extend(["--shard_shape", shard_shape_str])
            if chunk_shape_str:
                cmd.extend(["--chunk_shape", chunk_shape_str])

            if verbose:
                print(f"\nSubmitting s{level} (factor: {level_info['cumulative_factor']}, "
                      f"mem: {level_memory}GB, time: {level_wall_time}, cores: {level_cores})...")

            if dry_run:
                print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=tensorswitch_dir
                )

                if result.returncode != 0:
                    print(f"  ERROR: {result.stderr}")
                    continue

                # Extract job IDs from output
                import re
                job_ids = re.findall(r'Job <(\d+)>', result.stdout)
                all_job_ids.extend(job_ids)

                if verbose:
                    print(f"  Submitted {len(job_ids)} jobs: {job_ids}")

        # Submit coordinator job to wait for all levels and update metadata
        coordinator_job_id = None
        if not dry_run and all_job_ids:
            coordinator_job_id = self._submit_metadata_coordinator_job(
                level_job_ids=all_job_ids,
                project=project,
                verbose=verbose
            )
            if coordinator_job_id:
                all_job_ids.append(coordinator_job_id)

        if verbose:
            print(f"\n{'='*60}")
            print(f"SUBMISSION COMPLETE")
            print(f"{'='*60}")
            print(f"Level jobs submitted: {len(all_job_ids) - (1 if coordinator_job_id else 0)}")
            if coordinator_job_id:
                print(f"Coordinator job: {coordinator_job_id} (updates metadata after all levels complete)")
            if not dry_run:
                print(f"All job IDs: {all_job_ids}")
                print(f"\nMonitor with: bjobs")
                print(f"Metadata will be updated automatically when all jobs complete.")

        return all_job_ids

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
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to create full pyramid with parallel job submission.

    This is the main entry point for CLI --auto_multiscale mode.

    Args:
        s0_path: Path to s0 array
        project: LSF project to charge
        min_array_nbytes: Stop when array size < this
        min_array_shape: Stop when all dims < this
        memory: Memory per job in GB (None = auto-calculate per level)
        wall_time: Wall time per job (None = auto-calculate per level)
        cores: Cores per job (None = auto-calculate based on memory)
        use_shard: Whether to use sharding
        dry_run: If True, print commands but don't execute
        verbose: Print progress messages

    Returns:
        dict with 'pyramid_plan' and 'job_ids' keys

    Example:
        >>> result = create_pyramid_parallel(
        ...     s0_path="/data/dataset.zarr/s0",
        ...     project="scicompsoft"
        ... )
        >>> print(f"Submitted {len(result['job_ids'])} jobs")
    """
    planner = PyramidPlanner(s0_path)

    # Calculate pyramid plan
    pyramid_plan = planner.calculate_pyramid_plan(
        min_array_nbytes=min_array_nbytes,
        min_array_shape=min_array_shape
    )

    if verbose:
        planner.print_pyramid_plan(pyramid_plan)

    # Pre-create all levels
    if not dry_run:
        planner.precreate_all_levels(pyramid_plan, use_shard=use_shard, verbose=verbose)

    # Submit all levels in parallel
    job_ids = planner.submit_all_levels_parallel(
        pyramid_plan,
        project=project,
        memory=memory,
        wall_time=wall_time,
        cores=cores,
        use_shard=use_shard,
        dry_run=dry_run,
        verbose=verbose
    )

    return {
        'pyramid_plan': pyramid_plan,
        'job_ids': job_ids,
    }
