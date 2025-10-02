"""
Cost estimation for TensorSwitch GUI.

Janelia Cluster CPU: $0.05/slot/hour
AI Assistant: $0.10/session

Note: TensorSwitch uses CPU-only jobs (no GPU).
"""

import math


def estimate_processing_time(total_chunks: int, num_cores: int) -> float:
    """
    Estimate processing time in hours.

    Args:
        total_chunks: Total number of chunks to process
        num_cores: Number of CPU cores allocated

    Returns:
        Estimated time in hours
    """
    # Conservative estimate: 1 chunk/second/core
    chunks_per_second = num_cores * 1.0
    time_seconds = total_chunks / chunks_per_second
    return time_seconds / 3600  # Convert to hours


def estimate_cluster_cost(num_cores: int, wall_time_hours: float) -> float:
    """
    Estimate maximum cluster cost based on wall time limit.

    Args:
        num_cores: Number of CPU cores/slots requested
        wall_time_hours: Wall time limit in hours

    Returns:
        Maximum cost in dollars (CPU only)
    """
    return num_cores * wall_time_hours * 0.05


def get_ai_cost() -> float:
    """Get AI assistant cost per session."""
    return 0.10


def calculate_total_chunks(shape: tuple, chunk_shape: tuple) -> int:
    """
    Calculate total number of chunks.

    Args:
        shape: Array dimensions (e.g., (512, 2048, 2048))
        chunk_shape: Chunk dimensions (e.g., (64, 64, 64))

    Returns:
        Total number of chunks
    """
    total = 1
    for dim_size, chunk_size in zip(shape, chunk_shape):
        num_chunks = math.ceil(dim_size / chunk_size)
        total *= num_chunks
    return total


def format_time(hours: float) -> str:
    """Format time in human-readable format."""
    if hours >= 1.0:
        return f"{hours:.2f} hours"
    elif hours >= 1/60:
        return f"{hours * 60:.1f} minutes"
    else:
        return f"{hours * 3600:.0f} seconds"


def get_cost_summary(total_chunks: int, num_cores: int, wall_time_hours: float) -> str:
    """
    Get simple cost and time summary.

    Args:
        total_chunks: Total chunks to process
        num_cores: Number of CPU cores
        wall_time_hours: Wall time limit in hours

    Returns:
        Formatted summary string
    """
    estimated_time = estimate_processing_time(total_chunks, num_cores)
    estimated_cost = estimate_cluster_cost(num_cores, estimated_time)
    max_cost = estimate_cluster_cost(num_cores, wall_time_hours)
    ai_cost = get_ai_cost()

    return f"""
COST & TIME ESTIMATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processing Time:     {format_time(estimated_time)}
Total Chunks:        {total_chunks:,}
CPU Cores:           {num_cores}

Cluster Cost:        ${estimated_cost:.4f} - ${max_cost:.4f}
  (Estimated - Maximum based on {wall_time_hours:.1f}h wall time)

AI Assistant:        ${ai_cost:.2f} per session

TOTAL COST:          ${estimated_cost + ai_cost:.4f} - ${max_cost + ai_cost:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Test example
if __name__ == "__main__":
    # Example: (512, 2048, 2048) array with (64, 64, 64) chunks
    chunks = calculate_total_chunks((512, 2048, 2048), (64, 64, 64))
    print(get_cost_summary(chunks, num_cores=32, wall_time_hours=1.0))
