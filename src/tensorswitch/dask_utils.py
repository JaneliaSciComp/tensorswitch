#!/usr/bin/env python3
"""
Dask utilities for TensorSwitch LSF cluster integration.
"""

import os
import logging
from dask.distributed import Client
from dask_jobqueue import LSFCluster

logger = logging.getLogger(__name__)


def create_lsf_cluster(
    num_workers: int = 96,
    cores: int = 4,
    memory: str = "60GB", 
    walltime: str = "1:00",
    project: str = "scicompsoft"
) -> LSFCluster:
    """Create LSFCluster using proven configuration with output files in output/ directory."""
    
    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    cluster = LSFCluster(
        num_workers, 
        project=project, 
        ncpus=cores, 
        cores=cores, 
        memory=memory, 
        walltime=walltime, 
        job_extra_directives=[f"-o {output_dir}/output_%J.txt", f"-e {output_dir}/error_%J.txt"], 
        use_stdin=True
    )
    return cluster


def simple_test_function(x):
    """Simple test function for testing cluster connectivity."""
    return x * x


def run_tensorswitch_cli(cmd_args):
    """Run TensorSwitch as CLI subprocess on worker."""
    import subprocess
    import sys
    
    print(f"Worker executing: {' '.join(cmd_args)}")
    
    try:
        # Run as subprocess to avoid serialization issues
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"Worker completed successfully")
            return {"success": True, "stdout": result.stdout}
        else:
            print(f"Worker failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return {"success": False, "stderr": result.stderr}
            
    except subprocess.TimeoutExpired:
        print("Worker timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"Worker error: {e}")
        return {"success": False, "error": str(e)}


def submit_dask_job(args, total_chunks):
    """Submit TensorSwitch job using Dask JobQueue."""
    
    print("Using Dask JobQueue for task submission")
    print(f"Total chunks: {total_chunks}")
    print(f"Workers: {args.num_volumes}, Cores per worker: {args.cores}")
    
    cluster = None
    client = None
    
    try:
        # Create cluster
        cluster = create_lsf_cluster(
            num_workers=args.num_volumes,
            cores=int(args.cores),
            memory="60GB",
            walltime=args.wall_time,
            project=args.project
        )
        
        client = Client(cluster)
        print(f"Dashboard: {client.dashboard_link}")
        
        # Parse custom shapes
        custom_shard_shape = None
        if hasattr(args, 'custom_shard_shape') and args.custom_shard_shape:
            custom_shard_shape = [int(x) for x in args.custom_shard_shape.split(',')]
        
        custom_chunk_shape = None
        if hasattr(args, 'custom_chunk_shape') and args.custom_chunk_shape:
            custom_chunk_shape = [int(x) for x in args.custom_chunk_shape.split(',')]
        
        # Calculate chunk distribution
        num_workers = min(total_chunks, args.num_volumes)
        chunks_per_worker = total_chunks // num_workers
        
        print(f"Distributing {total_chunks} chunks across {num_workers} workers")
        
        # Submit tasks
        futures = []
        for i in range(num_workers):
            start_idx = i * chunks_per_worker
            if i == num_workers - 1:
                stop_idx = total_chunks
            else:
                stop_idx = (i + 1) * chunks_per_worker
            
            print(f"Submitting worker {i+1}/{num_workers}: chunks {start_idx}-{stop_idx}")
            
            # Build CLI command for worker
            cmd_args = [
                "python", "-m", "tensorswitch",
                "--task", args.task,
                "--base_path", args.base_path,
                "--output_path", args.output_path,
                "--level", str(args.level),
                "--start_idx", str(start_idx),
                "--stop_idx", str(stop_idx),
                "--memory_limit", str(args.memory_limit)
            ]
            
            # Add optional arguments
            if getattr(args, 'downsample', False):
                cmd_args.extend(["--downsample", "1"])
            if args.use_shard:
                cmd_args.extend(["--use_shard", "1"])
            if args.use_ome_structure:
                cmd_args.extend(["--use_ome_structure", "1"])
            if custom_shard_shape:
                cmd_args.extend(["--custom_shard_shape", ",".join(map(str, custom_shard_shape))])
            if custom_chunk_shape:
                cmd_args.extend(["--custom_chunk_shape", ",".join(map(str, custom_chunk_shape))])
            
            future = client.submit(run_tensorswitch_cli, cmd_args)
            futures.append(future)
        
        # Wait for completion
        print("Waiting for task completion...")
        completed = 0
        failed = 0
        for future in futures:
            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    print(f"Task {completed}/{len(futures)} completed successfully")
                else:
                    failed += 1
                    print(f"Task failed: {result.get('stderr', result.get('error', 'unknown'))}")
            except Exception as e:
                failed += 1
                print(f"Task failed with exception: {e}")
        
        if failed > 0:
            print(f"Warning: {failed} tasks failed, {completed} succeeded")
        else:
            print(f"All {completed} tasks completed successfully")
        
        # Apply OME metadata if needed
        if hasattr(args, 'use_ome_structure') and args.use_ome_structure:
            try:
                from tensorswitch.utils import update_ome_metadata_if_needed
                update_ome_metadata_if_needed(args.output_path, use_ome_structure=True)
                print("OME metadata updated")
            except Exception as e:
                print(f"OME metadata update failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Dask job submission failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # FIXED: Original code below was causing "killed by owner" errors:
        if client is not None:
             client.close()
        if cluster is not None:
             cluster.close()

        # Let LSF jobs run independently. The cluster will auto-cleanup when jobs complete.
        print("Jobs submitted - cluster will manage LSF execution independently")
        print(f"Logs in: {os.path.abspath('./output')}")

