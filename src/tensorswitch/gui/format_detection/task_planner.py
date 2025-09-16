"""
Task planning logic for TensorSwitch conversions
"""

from typing import Dict, List, Optional, Tuple


class TaskPlanner:
    """Plans conversion tasks based on input/output formats and user preferences"""

    def __init__(self):
        # Mapping of input->output format combinations to tasks
        self.conversion_tasks = {
            ('tiff', 'zarr2'): 'tiff_to_zarr2_s0',
            ('tiff', 'zarr3'): 'tiff_to_zarr3_s0',
            ('nd2', 'zarr2'): 'nd2_to_zarr2_s0',
            ('nd2', 'zarr3'): 'nd2_to_zarr3_s0',
            ('ims', 'zarr2'): 'ims_to_zarr2_s0',
            ('ims', 'zarr3'): 'ims_to_zarr3_s0',
            ('n5', 'zarr2'): 'n5_to_zarr2',
            ('n5', 'n5'): 'n5_to_n5',
            ('zarr2', 'zarr2'): 'downsample_zarr2',
            ('zarr3', 'zarr3'): 'downsample_shard_zarr3'
        }

        # Downsampling tasks for each format
        self.downsample_tasks = {
            'zarr2': 'downsample_zarr2',
            'zarr3': 'downsample_shard_zarr3'
        }

        # Compatible output formats for each input format
        self.compatible_outputs = {
            'tiff': ['zarr2', 'zarr3'],
            'nd2': ['zarr2', 'zarr3'],
            'ims': ['zarr2', 'zarr3'],
            'n5': ['zarr2', 'n5'],
            'zarr2': ['zarr2'],
            'zarr3': ['zarr3']
        }

    def get_compatible_outputs(self, input_format: str) -> List[str]:
        """Get list of compatible output formats for input format"""
        return self.compatible_outputs.get(input_format, [])

    def create_conversion_plan(self, input_format: str, output_format: str,
                             downsample_levels: List[int] = None) -> Dict:
        """
        Create a complete conversion plan

        Args:
            input_format: Source format (tiff, nd2, ims, etc.)
            output_format: Target format (zarr2, zarr3, n5)
            downsample_levels: List of levels to generate (e.g., [0,1,2,3])

        Returns:
            Dictionary with planned tasks and execution order
        """
        plan = {
            'input_format': input_format,
            'output_format': output_format,
            'tasks': [],
            'estimated_outputs': [],
            'requires_multiscale': False,
            'error': None
        }

        try:
            # Get primary conversion task
            primary_task = self.conversion_tasks.get((input_format, output_format))
            if not primary_task:
                plan['error'] = f"No conversion available from {input_format} to {output_format}"
                return plan

            # Always include primary conversion
            plan['tasks'].append({
                'task': primary_task,
                'purpose': 'Primary conversion',
                'generates_level': 0,
                'order': 1
            })

            plan['estimated_outputs'].append(f"s0 level ({output_format} format)")

            # Check if we need downsampling
            if downsample_levels and len(downsample_levels) > 1:
                plan['requires_multiscale'] = True

                # Get downsampling task
                downsample_task = self.downsample_tasks.get(output_format)
                if downsample_task:
                    max_level = max(downsample_levels)
                    plan['tasks'].append({
                        'task': downsample_task,
                        'purpose': f'Generate pyramid levels 1-{max_level}',
                        'generates_level': f'1-{max_level}',
                        'order': 2
                    })

                    # Add expected outputs
                    for level in downsample_levels[1:]:  # Skip s0 (already included)
                        plan['estimated_outputs'].append(f"s{level} level (downsampled)")

        except Exception as e:
            plan['error'] = f"Planning failed: {str(e)}"

        return plan

    def estimate_processing_time(self, file_size_mb: float, tasks: List[Dict]) -> str:
        """Estimate total processing time based on file size and tasks"""
        if not tasks:
            return "Unknown"

        # Basic time estimation (very rough)
        base_time_per_gb = 5  # minutes per GB for primary conversion
        downsample_time_per_gb = 2  # minutes per GB for downsampling

        total_minutes = 0
        file_size_gb = file_size_mb / 1024

        for task_info in tasks:
            task_name = task_info['task']
            if 'downsample' in task_name:
                total_minutes += file_size_gb * downsample_time_per_gb
            else:
                total_minutes += file_size_gb * base_time_per_gb

        if total_minutes < 1:
            return "< 1 minute"
        elif total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            hours = total_minutes / 60
            return f"~{hours:.1f} hours"

    def format_plan_summary(self, plan: Dict, file_size_mb: float = 0) -> str:
        """Format conversion plan into readable summary"""
        if plan.get('error'):
            return f"Error: {plan['error']}"

        if not plan.get('tasks'):
            return "No tasks planned"

        lines = []
        lines.append(f"Conversion Plan: {plan['input_format'].upper()} → {plan['output_format'].upper()}")
        lines.append("")

        # List tasks in execution order
        lines.append("Execution Steps:")
        for task_info in plan['tasks']:
            lines.append(f"{task_info['order']}. {task_info['task']}")
            lines.append(f"   Purpose: {task_info['purpose']}")
            lines.append("")

        # List expected outputs
        if plan.get('estimated_outputs'):
            lines.append("Expected Outputs:")
            for i, output in enumerate(plan['estimated_outputs'], 1):
                lines.append(f"• {output}")

        # Add time estimate
        if file_size_mb > 0:
            time_est = self.estimate_processing_time(file_size_mb, plan['tasks'])
            lines.append("")
            lines.append(f"Estimated time: {time_est}")

        return "\n".join(lines)

    def get_task_parameters(self, task_name: str, input_analysis: Dict,
                          output_config: Dict) -> Dict:
        """
        Get recommended parameters for a specific task

        Args:
            task_name: Name of the task
            input_analysis: Results from format detector
            output_config: User's output preferences

        Returns:
            Dictionary of recommended parameters
        """
        params = {
            'use_shard': True,
            'use_ome_structure': True,
            'level': 0,
            'cores': '4',
            'wall_time': '2:00',
            'num_volumes': 8,
            'memory_limit': 70
        }

        # Adjust based on file size
        file_size_mb = input_analysis.get('size_mb', 0)
        if file_size_mb > 5000:  # > 5GB
            params.update({
                'cores': '8',
                'wall_time': '4:00',
                'num_volumes': 16,
                'memory_limit': 80
            })
        elif file_size_mb > 1000:  # > 1GB
            params.update({
                'cores': '6',
                'wall_time': '3:00',
                'num_volumes': 12,
                'memory_limit': 75
            })

        # Disable sharding for zarr2
        if 'zarr2' in task_name:
            params['use_shard'] = False

        # Adjust for specific formats
        input_format = input_analysis.get('format')
        if input_format == 'nd2':
            # ND2 files can be memory intensive
            params['memory_limit'] = min(85, params['memory_limit'] + 10)
        elif input_format == 'ims':
            # IMS files benefit from more parallel processing
            params['num_volumes'] = min(32, params['num_volumes'] * 2)

        return params