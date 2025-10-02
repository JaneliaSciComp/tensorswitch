#!/usr/bin/env python3
"""
TensorSwitch AI Assistant - Focused helper for data conversion tasks
"""

import json
from .ai_config import ai_config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_tensorswitch_knowledge():
    """Comprehensive TensorSwitch knowledge base for AI responses"""
    return {
        "overview": {
            "purpose": "TensorSwitch converts scientific imaging data between formats with automatic optimization",
            "gui_workflow": "Welcome Page → Conversion Page (with AI assistant) → Progress Monitoring",
            "key_feature": "Smart Mode automatically detects file formats and suggests optimal conversion settings"
        },
        "tasks": {
            "tiff_to_zarr3_s0": "Converts TIFF stacks to Zarr3 with OME metadata. Best for microscopy image stacks.",
            "nd2_to_zarr3_s0": "Converts Nikon ND2 files to Zarr3 with full metadata preservation. Ideal for brain imaging.",
            "ims_to_zarr3_s0": "Converts Imaris IMS files to Zarr3 format. Handles 3D/4D/5D datasets with rich metadata.",
            "tiff_to_zarr2_s0": "Converts TIFF to Zarr2 for broader tool compatibility.",
            "nd2_to_zarr2_s0": "Converts ND2 to Zarr2 format. Choose when you need legacy compatibility.",
            "ims_to_zarr2_s0": "Converts IMS to Zarr2 format for wider analysis tool support.",
            "downsample_shard_zarr3": "Creates multiscale pyramids (s0→s1→s2→s3→s4) for Zarr3 datasets. Essential for large data visualization.",
            "downsample_zarr2": "Creates multiscale pyramids for Zarr2 datasets.",
            "n5_to_zarr2": "Converts BigDataViewer N5 format to Zarr2.",
            "n5_to_n5": "Re-chunks and optimizes existing N5 datasets."
        },
        "formats": {
            "ND2": "Nikon microscopy format. Contains rich metadata including acquisition parameters, channel info, and calibration data.",
            "IMS": "Imaris format for 3D/4D imaging. Stores complex multidimensional datasets with spatial and temporal information.",
            "TIFF": "Standard image format. Good for image stacks but limited metadata compared to specialized formats.",
            "Zarr3": "Modern array format with sharding. Best performance for large datasets, faster random access, custom chunk shapes.",
            "Zarr2": "Legacy Zarr format. Better compatibility with analysis tools like napari, but no sharding support.",
            "N5": "BigDataViewer format optimized for very large datasets. Good compression and performance."
        },
        "smart_mode": {
            "how_it_works": "Automatically detects file format, extracts metadata (shape, dimensions, size), suggests optimal conversion task and parameters",
            "detection_capability": "Analyzes file extensions, directory structure, and file headers to identify format and version",
            "benefits": "Eliminates guesswork, provides format-specific optimization, suggests resource requirements",
            "when_to_use": "Always start with Smart Mode unless you have specific advanced requirements"
        },
        "workflow_guidance": {
            "smart_mode_steps": [
                "1. Enter your input file/directory path",
                "2. Smart Mode automatically detects format and shows metadata",
                "3. Choose your desired output format (Zarr3 recommended)",
                "4. Set downsampling levels if needed",
                "5. Review the execution plan",
                "6. Execute locally or submit to cluster"
            ],
            "path_examples": [
                "/groups/[lab_name]/data/file.nd2 - PRFS storage",
                "/nrs/[lab_name]/datasets/ - NRS storage",
                "Directory paths work too - Smart Mode will find compatible files"
            ]
        },
        "parameters": {
            "cores": "2 cores recommended for most tasks - TensorSwitch is optimized for efficient processing",
            "memory": "16GB memory sufficient for most conversions",
            "wall_time": "1-2 hours covers most tasks when correct volume parameters are selected",
            "volume_selection": "Key to performance - proper volume parameters more important than high core count",
            "sharding": "Enable for Zarr3 when files >1GB. Improves random access performance significantly.",
            "downsampling": "Level 0=full resolution, 1-5=pyramid levels. Use 2-3 levels for visualization."
        },
        "cluster_features": {
            "dask_jobqueue": "Advanced cluster submission with automatic scaling and error recovery",
            "lsf_integration": "Direct LSF job submission with resource management",
            "progress_monitoring": "Real-time tracking of job status and completion"
        },
        "troubleshooting": {
            "format_not_detected": "Try entering the full file path. Smart Mode works best with specific file paths.",
            "memory_errors": "Increase memory allocation or reduce chunk size. Consider using cluster submission.",
            "slow_performance": "Enable sharding for Zarr3, use cluster submission for large files.",
            "job_failures": "Check file permissions, verify paths exist, ensure sufficient cluster resources."
        },
        "best_practices": {
            "format_choice": "Use Zarr3 for new projects (better performance), Zarr2 for legacy tool compatibility",
            "resource_planning": "Local execution for <1GB, cluster submission for larger files",
            "metadata_preservation": "Always enable OME structure to preserve acquisition metadata",
            "path_organization": "Use HHMI lab path helper to find correct storage locations"
        }
    }


def get_resource_recommendation(file_size_gb):
    """Provide resource recommendations based on file size"""
    if file_size_gb < 1:
        return "small"
    elif file_size_gb < 5:
        return "medium"
    elif file_size_gb < 20:
        return "large"
    else:
        return "xlarge"


def get_tensorswitch_help_with_openai(user_question, context_data=None):
    """
    Main AI function for TensorSwitch-specific help
    """
    if not OPENAI_AVAILABLE:
        return "AI Assistant requires OpenAI library. Please install with: pip install openai"

    if not ai_config.is_ai_available():
        return "AI Assistant not configured. Please set your OpenAI API key."

    try:
        client = OpenAI(api_key=ai_config.get_api_key())
        knowledge = get_tensorswitch_knowledge()

        # Add GUI context if available
        context_str = ""
        if context_data:
            context_str = f"\n\nCurrent GUI state:\n{json.dumps(context_data, indent=2)}"

        # System prompt focused only on TensorSwitch with Smart Mode guidance
        system_prompt = f"""You are a TensorSwitch data conversion assistant. Your job is to help scientists use TensorSwitch GUI for converting imaging data efficiently.

CORE GUIDANCE PRINCIPLES:
1. ALWAYS guide users to Smart Mode first - it automatically detects formats and suggests optimal settings
2. When users provide file paths, immediately guide them through the Smart Mode workflow
3. Smart Mode eliminates the need to manually specify formats or guess parameters
4. Only suggest Manual Mode for advanced users with specific requirements

IMPORTANT: When user explicitly requests specific parameter values (like "use 4 cores" or "set memory to 32GB"), HONOR their request even if it differs from recommendations. Users know their workflow needs. Provide the values they ask for.

SMART MODE WORKFLOW (Guide users through these steps):
1. "Enter your file path in the 'Input File/Directory Path' field"
2. "Make sure 'Smart Mode' is selected (recommended for most users)"
3. "Smart Mode will automatically detect your file format and show metadata"
4. "Choose your desired output format (Zarr3 recommended for new projects)"
5. "Review the suggested parameters and execution plan"
6. "Click Execute or Submit to Cluster"

WHEN USERS PROVIDE PATHS:
- Immediately explain that Smart Mode will detect their format automatically
- Guide them to enter the path in the GUI first
- Explain what Smart Mode will show them (format, size, dimensions, etc.)
- Don't ask them to specify format manually - Smart Mode handles this

WHEN USERS SHARE AUTO-DETECTED INFO:
- If user shares detected file info (format, size, shape, etc.), provide SPECIFIC recommendations
- Give practical recommendations based on real TensorSwitch usage:
  • **Cores**: "Use 2 cores for most tasks - TensorSwitch is optimized for efficient processing"
  • **Memory**: "16GB memory is sufficient for most conversions"
  • **Wall time**: "1-2 hours covers most tasks when correct volume parameters are selected"
  • **Key tip**: "Proper volume selection is more important than high core count"
- For the 1.7GB ND2 file example: "Use 2 cores, 16GB memory, 2 hours wall time"

IMPORTANT RULES:
- ONLY help with TensorSwitch-related questions
- Always recommend Smart Mode first unless user specifically needs Manual Mode
- When users share detected file info, give SPECIFIC parameter recommendations immediately
- Use the comprehensive TensorSwitch knowledge provided below
- Be specific about resource recommendations based on file size
- Guide users step-by-step through the GUI workflow
- Focus on: Smart Mode guidance, format detection, parameter optimization, cluster submission

TensorSwitch Comprehensive Knowledge:
{json.dumps(knowledge, indent=2)}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_question}{context_str}"}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI Assistant error: {str(e)}"


def analyze_file_and_suggest(file_path, file_size_gb=None):
    """Analyze file and provide TensorSwitch-specific suggestions"""
    suggestions = {
        "detected_format": "Unknown",
        "recommended_task": None,
        "recommended_resources": None,
        "tips": []
    }

    if not file_path:
        return suggestions

    # Format detection based on extension
    file_lower = file_path.lower()
    if file_lower.endswith('.nd2'):
        suggestions["detected_format"] = "ND2"
        suggestions["recommended_task"] = "nd2_to_zarr3_s0"
        suggestions["tips"].append("ND2 files work great with Smart Mode auto-detection")
        suggestions["tips"].append("Consider Zarr3 format for better performance with large files")
    elif file_lower.endswith('.ims'):
        suggestions["detected_format"] = "IMS"
        suggestions["recommended_task"] = "ims_to_zarr3_s0"
        suggestions["tips"].append("IMS files contain rich metadata that will be preserved")
    elif file_lower.endswith(('.tif', '.tiff')):
        suggestions["detected_format"] = "TIFF"
        suggestions["recommended_task"] = "tiff_to_zarr3_s0"
        suggestions["tips"].append("TIFF stacks convert efficiently to Zarr format")

    # Resource recommendations
    if file_size_gb:
        rec_level = get_resource_recommendation(file_size_gb)
        knowledge = get_tensorswitch_knowledge()
        suggestions["recommended_resources"] = knowledge["recommended_resources"][rec_level]

    return suggestions


def get_ai_benefits():
    """Return list of AI assistant benefits for display"""
    return [
        "🎯 **Format Detection**: Get instant analysis of your input files",
        "⚙️ **Parameter Recommendations**: Optimal cores, memory, and time settings",
        "📁 **Lab Path Guidance**: Find your lab's storage locations quickly",
        "🔄 **Workflow Help**: Smart vs Manual mode explanations",
        "🚀 **Conversion Advice**: Best practices for your specific data type",
        "❓ **Real-time Support**: Ask questions about any TensorSwitch feature"
    ]