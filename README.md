# TensorSwitch

This package provides a unified entry point for managing N5/Zarr dataset conversions and downsampling with both command-line and web GUI interfaces. It centralizes your workflow into a single pipeline to reduce manual work and errors.

## Features

- **Web GUI Interface**: Production-ready web interface for scientists (no programming required)
- **AI Assistant**: Integrated OpenAI-powered assistant with context-aware responses and conversion guidance
- **Cost Estimation**: Real-time cluster resource cost estimates (AI + cluster billing) before job submission
- **Command-Line Interface**: Full-featured CLI for automated workflows
- **Advanced Dask JobQueue**: Hybrid Dask-LSF execution with automatic scaling and error recovery
- **Smart Workflow System**: Auto-detect input formats and intelligently plan conversions
- **Enhanced OME-ZARR Metadata**: Automatic preservation of rich metadata from TIFF, ND2, and IMS files
- **Dual Zarr Format Support**: Create files compatible with both Zarr v2 and v3 tools
- **Remote Data Support**: Direct conversion from HTTP, Google Cloud Storage (GCS), and S3-served datasets
- **Lab Path Integration**: Built-in HHMI lab storage paths (131 labs, 126 projects)
- **Cluster Integration**: LSF job submission with resource management and real-time monitoring
- **Multiple Format Support**: Complete Zarr2, Zarr3, N5, TIFF, ND2, and IMS conversions
- **Optimized N5 Compression**: Native zstd compression for better compatibility and performance
- **Comprehensive Test Suite**: 31 verified tests covering CLI, GUI, and OME-Zarr workflows

## Folder Structure

```
tensorswitch/
├── pixi.lock
├── pyproject.toml
├── README.md
├── src
│   └── tensorswitch
│       ├── __init__.py
│       ├── __main__.py                   # Main dispatcher script
│       ├── tasks
│       │   ├── __init__.py
│       │   ├── downsample_shard_zarr3.py # Downsample using shards
│       │   ├── downsample_zarr2.py       # Downsample existing Zarr V2 datasets
│       │   ├── n5_to_n5.py               # N5 to N5 conversion logic
│       │   ├── n5_to_zarr2.py            # N5 to Zarr V2 conversion logic
│       │   ├── tiff_to_zarr2_s0.py       # TIFF to Zarr V2 level s0 with OME-Zarr metadata
│       │   ├── tiff_to_zarr3_s0.py       # TIFF to Zarr V3 level s0 with OME-Zarr metadata
│       │   ├── nd2_to_zarr2_s0.py        # ND2 to Zarr V2 level s0 with OME-Zarr metadata
│       │   ├── nd2_to_zarr3_s0.py        # ND2 to Zarr V3 level s0 with OME-Zarr metadata
│       │   ├── ims_to_zarr2_s0.py        # IMS to Zarr V2 level s0 with OME-Zarr metadata
│       │   ├── ims_to_zarr3_s0.py        # IMS to Zarr V3 level s0 with OME-Zarr metadata
│       ├── utils.py                      # Common utilities and OME-Zarr metadata functions
│       ├── dask_utils.py                 # Dask JobQueue integration for cluster execution
│       └── gui/                          # Web GUI interface
│           ├── app.py                    # Main GUI application
│           ├── launch_gui.py             # GUI server launcher
│           ├── cost_estimator.py         # Cost and time estimation for cluster jobs
│           ├── README_GUI.md             # GUI documentation
│           ├── ai/                       # AI assistant system
│           │   ├── __init__.py
│           │   ├── ai_config.py          # AI configuration and environment setup
│           │   ├── ai_interface.py       # AI chat interface for GUI
│           │   └── tensorswitch_assistant.py # AI knowledge base and OpenAI integration
│           ├── format_detection/         # Smart workflow and format auto-detection
│           │   ├── __init__.py
│           │   ├── format_detector.py    # Auto-detect file formats and metadata
│           │   └── task_planner.py       # Intelligent conversion planning
│           └── lab_paths_system/         # HHMI lab paths integration
│               ├── __init__.py
│               ├── lab_paths.py          # Lab path management
│               ├── path_selector.py      # Path selection UI
│               ├── hierarchical_lab_paths.json
│               ├── lab_paths_data.json
│               └── Lab_and_project_file_share_path.xlsx
├── contrib
│   ├── add_dimension_names.py            # Add dimension names to zarr.json files for all levels
│   ├── bleaching_correction_task.py      # Z-direction bleaching correction for microscopy datasets
│   ├── re_submit_jobs.ipynb              # Jupyter notebook to re-submit failed chunk jobs
│   ├── start_neuroglancer_server.py      # Start a CORS-enabled web server
│   ├── submit_bleaching_correction_general.py # Generic bleaching correction submission script
│   ├── update_metadata.py                # Update OME-Zarr multiscale metadata and add ome_xml
│   ├── update_metadata_zarr2.py          # Update OME-Zarr multiscale metadata for zarr2 datasets
│   └── z_to_chunk_index.py               # Print chunk index ranges for resubmit failed or left over jobs
└── tests                                   # Comprehensive test suite
    ├── conftest.py                         # Pytest fixtures and utilities
    ├── test_data_config.py                 # Real and synthetic data configuration
    ├── test_http_support.py                # HTTP path handling tests (7 tests)
    ├── cli/                                # CLI tests (18 tests)
    │   ├── test_cli_conversions_real.py    # Real data tests with middle chunks (6 tests)
    │   ├── test_cli_conversions_synthetic.py # All 10 conversion tasks (8 tests)
    │   └── test_cli_ome_zarr.py            # OME-Zarr metadata validation (4 tests)
    └── gui/                                # GUI tests (13 tests)
        ├── test_gui_conversions_synthetic.py # GUI conversion workflows (9 tests)
        └── test_gui_ome_zarr.py            # GUI OME-Zarr metadata (4 tests)

```

## Installation

### Pip

```
pip install git+https://github.com/JaneliaSciComp/tensorswitch
```

### Pixi

```
pixi add python
pixi add --pypi "tensorswitch @ git+https://github.com/JaneliaSciComp/tensorswitch"
```

## Quick Start

### Option 1: Web GUI (Recommended for Scientists)

Launch the web interface for easy, no-programming data conversions:

```bash
# Start the GUI server
pixi run python src/tensorswitch/gui/launch_gui.py

# Access in browser at:
# http://[your-hostname]:5000
```

**Features:**
- **AI Assistant**: Integrated OpenAI-powered chat for conversion guidance and parameter optimization
- **Smart Mode**: Auto-detect file formats and suggest optimal conversion settings
- **Simple 3-step workflow**: Select input → Configure → Convert
- **Advanced Dask Submission**: Hybrid Dask-LSF execution with automatic scaling and error recovery
- **Built-in path suggestions**: 131 HHMI labs with direct storage access
- **Real-time progress monitoring**: Job status tracking and log viewing
- **Dual format support**: Create Zarr files compatible with both v2 and v3 tools
- **Local or cluster job submission**: Flexible execution options
- **All conversion tasks supported**: TIFF, ND2, IMS to Zarr2/Zarr3

See [GUI Documentation](src/tensorswitch/gui/README_GUI.md) for detailed usage.

### Option 2: Command Line Interface

## How to Use

### 1. Run the Main Pipeline

Use the `python -m tensorswitch` as the entry point. Example:

```bash
python -m tensorswitch --task n5_to_zarr2     --base_path /path/to/n5     --output_path /path/to/zarr2     --level 0
```

### 2. Submit Cluster Jobs (Optional)

You can also use the `--submit` flag to dispatch jobs to an LSF cluster:

```bash
python -m tensorswitch --task downsample_shard_zarr3 \
  --base_path /path/to/zarr/s0 \
  --output_path /path/to/zarr \
  --level 1 \
  --downsample 1 \
  --use_shard 1 \
  --submit \
  --project your_project_name
```

### 3. Supported Tasks

| Task Name                 | Description |
|---------------------------|-----------------------------------------|
| `n5_to_zarr2`            | Convert N5 to Zarr V2                   |
| `n5_to_n5`               | Convert N5 to N5 (new chunking)         |
| `downsample_shard_zarr3` | Downsample existing Zarr V3 dataset with sharding |
| `downsample_zarr2`       | Downsample existing Zarr V2 dataset     |
| `tiff_to_zarr2_s0`       | Convert TIFF stack to Zarr V2 (s0) with OME-Zarr metadata |
| `tiff_to_zarr3_s0`       | Convert TIFF stack to Zarr V3 (s0) with OME-Zarr metadata |
| `nd2_to_zarr2_s0`        | Convert ND2 file to Zarr V2 (s0) with OME-Zarr metadata |
| `nd2_to_zarr3_s0`        | Convert ND2 file to Zarr V3 (s0) with OME-Zarr metadata |
| `ims_to_zarr2_s0`        | Convert IMS file to Zarr V2 (s0) with OME-Zarr metadata |
| `ims_to_zarr3_s0`        | Convert IMS file to Zarr V3 (s0) with OME-Zarr metadata |

All s0 conversion tasks create multiscale-compatible Zarr structures with proper OME-Zarr metadata and correct dimension ordering.

### Enhanced OME-ZARR Metadata

All conversion tasks now support both Zarr V2 and Zarr V3 formats with automatic metadata preservation:

**Zarr V2 Tasks:** `tiff_to_zarr2_s0`, `nd2_to_zarr2_s0`, `ims_to_zarr2_s0`, `downsample_zarr2`
**Zarr V3 Tasks:** `tiff_to_zarr3_s0`, `nd2_to_zarr3_s0`, `ims_to_zarr3_s0`, `downsample_shard_zarr3`

- **Automatic OME XML Extraction**: Source metadata is automatically preserved
- **Correct Dimension Ordering**: Fixes dimension mapping issues (e.g., XYCTZ → [c,y,x])
- **Complete Zarr2 Support**: Creates both .zattrs and .zgroup files
- **No Manual Steps Required**: Metadata is handled automatically during conversion

**Recent Fixes:**
- Fixed dimension ordering for ND2 files with multi-channel data
- Added missing .zgroup files for zarr2 format compliance
- Universal metadata handling across all formats

### 4. Dual Zarr v2/v3 Format Compatibility

Create Zarr files that work with both v2 and v3 tools for maximum compatibility.

#### Available Options

Use `--dual_zarr_approach` with any Zarr v3 conversion task:

| Option | Description | When to Use |
|--------|-------------|-------------|
| `none` | **Default**. Pure Zarr v3 | Standard workflows |
| `v2_chunks` | Dual format with v2 chunk structure | Maximum compatibility |
| `v3_chunks` | Dual format with v3 chunk structure | Specific research needs |

#### Key Features

- **Auto-disable sharding**: Dual format prevents sharding (shown in logs)
- **Cross-tool compatibility**: Same data readable by both Zarr v2 and v3 libraries
- **Multiple entry points**: Root and multiscale access for flexible tool support

#### Examples

```bash
# Default: Pure Zarr v3
python -m tensorswitch --task nd2_to_zarr3_s0 --base_path file.nd2 --output_path output.zarr

# Dual format for maximum compatibility
python -m tensorswitch --task nd2_to_zarr3_s0 --base_path file.nd2 --output_path output.zarr --dual_zarr_approach v2_chunks

# Dual format with v3 chunk structure
python -m tensorswitch --task nd2_to_zarr3_s0 --base_path file.nd2 --output_path output.zarr --dual_zarr_approach v3_chunks
```

### 5. AI Assistant Features

The integrated AI assistant provides intelligent guidance for data conversion tasks.

#### Enabling AI Assistant

Set your OpenAI API key before launching the GUI:

```bash
export OPENAI_API_KEY="your-api-key-here"
pixi run python src/tensorswitch/gui/launch_gui.py
```

#### AI Capabilities

- **Format Detection Guidance**: Get recommendations based on your file types
- **Parameter Optimization**: Optimal cores, memory, and time settings for your data size
- **Lab Path Assistance**: Find your lab's storage locations quickly
- **Smart vs Manual Mode**: Understand when to use each approach
- **Conversion Best Practices**: Format-specific optimization tips
- **Real-time Support**: Ask questions about any TensorSwitch feature

#### Example AI Interactions

```
User: "I have a 5GB ND2 file, what settings should I use?"
AI: "For a 5GB ND2 file, I recommend:
     - Use Smart Mode for auto-detection
     - 2 cores, 16GB memory, 2 hours wall time
     - Enable OME structure for metadata preservation
     - Consider Zarr3 format for better performance"

User: "How do I find my lab's storage path?"
AI: "Use the Lab Path Helper in the GUI. Your lab's paths are:
     - NRS: /nrs/yourlab/
     - Groups: /groups/yourlab/
     - Click on the path suggestions for quick access"
```

### 6. Cluster Job Submission

Submit conversions to LSF cluster with automatic job splitting.

#### Traditional LSF Submission

```bash
# Submit ND2 to Zarr conversion to cluster
python -m tensorswitch --task nd2_to_zarr3_s0 \
  --base_path /path/to/file.nd2 \
  --output_path /path/to/output.zarr \
  --num_volumes 4 \
  --cores 2 \
  --wall_time 2:00 \
  --project your_project \
  --submit

# Submit with dual format
python -m tensorswitch --task nd2_to_zarr3_s0 \
  --base_path /path/to/file.nd2 \
  --output_path /path/to/output.zarr \
  --dual_zarr_approach v2_chunks \
  --num_volumes 8 \
  --project your_project \
  --submit
```

#### Dask JobQueue Submission

```bash
# Use Dask for advanced cluster scheduling
python -m tensorswitch --task nd2_to_zarr3_s0 \
  --base_path /path/to/file.nd2 \
  --output_path /path/to/output.zarr \
  --use_dask_jobqueue \
  --project your_project \
  --submit
```

#### Cluster Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_volumes` | Number of parallel jobs | 8 |
| `--cores` | Cores per job | 4 |
| `--wall_time` | Time limit (HH:MM) | 1:00 |
| `--project` | Billing project (required) | None |
| `--use_dask_jobqueue` | Use Dask scheduling | False |

#### Advanced Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--custom_chunk_shape` | Custom chunk size for output (comma-separated) | 128,128,128 |
| `--custom_shard_shape` | Custom shard size for Zarr v3 (comma-separated) | 256,256,256 |
| `--memory_limit` | Memory limit in GB per job | 50 |
| `--dual_zarr_approach` | Create dual v2/v3 compatible format | v2_chunks, v3_chunks, none |

### 6. Example Commands

#### Convert N5 to Zarr V2 locally
```bash
python -m tensorswitch --task n5_to_zarr2     --base_path /path/to/n5     --output_path /path/to/zarr2     --level 0
```

#### Downsample Zarr v2 locally
```bash
python -m tensorswitch --task downsample_zarr2 --base_path /path/to/zarr2/s0 --output_path /path/to/zarr2 --level 1
```

#### Downsample Zarr v3 locally
```bash
python -m tensorswitch --task downsample_shard_zarr3 --base_path /path/to/zarr/s0 --output_path /path/to/zarr --level 1 --use_shard 1
```

#### Submit N5 to Zarr jobs to cluster
```bash
python -m tensorswitch --task n5_to_zarr2 --base_path /path/to/n5 --output_path /path/to/zarr2 --submit --project your_project_name
```

#### Convert N5 to N5 with custom chunk size (rechunking)
```bash
# Transfer and rechunk N5 from remote HTTP server to local storage
python -m tensorswitch --task n5_to_n5 \
  --base_path "http://remote-server/dataset.n5/setup0/timepoint0/s0" \
  --output_path "/local/storage/dataset.n5/setup0/timepoint0/s0" \
  --custom_chunk_shape 128,128,128 \
  --level 0 \
  --memory_limit 50 \
  --num_volumes 1 \
  --cores 8 \
  --submit \
  --project your_project_name
```

#### Convert TIFF to Zarr v2 s0 with automatic OME metadata
```bash
python -m tensorswitch --task tiff_to_zarr2_s0 --base_path /path/to/tiff_folder --output_path /path/to/zarr2 --use_ome_structure 1
```

#### Convert TIFF to Zarr v3 s0 with automatic OME metadata
```bash
python -m tensorswitch --task tiff_to_zarr3_s0 --base_path /path/to/tiff_folder --output_path /path/to/zarr3 --use_shard 0 --use_ome_structure 1
```

#### Convert ND2 to Zarr v2 s0 with automatic OME metadata (local)
```bash
python -m tensorswitch --task nd2_to_zarr2_s0 --base_path /path/to/file.nd2 --output_path /path/to/output.zarr --use_ome_structure 1
```

#### Convert ND2 to Zarr v3 s0 with automatic OME metadata (local)
```bash
python -m tensorswitch --task nd2_to_zarr3_s0 --base_path /path/to/file.nd2 --output_path /path/to/output.zarr --use_shard 0 --use_ome_structure 1
```

#### Submit ND2 to Zarr v3 conversion to cluster with OME metadata
```bash
python -m tensorswitch --task nd2_to_zarr3_s0 \
  --base_path /path/to/file.nd2 \
  --output_path /path/to/output.zarr \
  --use_ome_structure 1 \
  --num_volumes 8 \
  --cores 4 \
  --wall_time 2:00 \
  --project your_project_name \
  --submit
```

#### Convert IMS to Zarr v3 s0 with automatic OME-Zarr metadata (local)
```bash
python -m tensorswitch --task ims_to_zarr3_s0 --base_path /path/to/file.ims --output_path /path/to/output.zarr --use_shard 0 --use_ome_structure 1
```

#### Submit IMS to Zarr v3 conversion to cluster with OME metadata
```bash
python -m tensorswitch --task ims_to_zarr3_s0 \
  --base_path /path/to/file.ims \
  --output_path /path/to/output.zarr \
  --use_ome_structure 1 \
  --num_volumes 8 \
  --cores 4 \
  --wall_time 2:00 \
  --project your_project_name \
  --submit
```

#### Create Complete Multiscale OME-Zarr Dataset

1. **Convert to s0**: Use any s0 conversion task (tiff, nd2, or ims) with `--use_ome_structure 1`
2. **Generate levels s1-s4**: Use downsampling for each level  
3. **Update multiscale metadata**: Optionally update multiscale structure

```bash
# Step 1: Convert source to s0 (example with IMS) - OME metadata automatically preserved
python -m tensorswitch --task ims_to_zarr3_s0 \
  --base_path /path/to/file.ims \
  --output_path /path/to/output.zarr \
  --use_ome_structure 1 \
  --submit --project your_project

# Step 2: Generate downsampled levels s1-s4
for level in {1..4}; do
  python -m tensorswitch --task downsample_shard_zarr3 \
    --base_path /path/to/output.zarr/multiscale/s$((level-1)) \
    --output_path /path/to/output.zarr \
    --level $level \
    --submit --project your_project
done

# Step 3: Update multiscale metadata (optional - source metadata already preserved)
python contrib/update_metadata.py /path/to/output.zarr
```

#### Resubmit Failed Chunks
Use `re_submit_jobs.ipynb` to debug chunk failures and resubmit specific chunk ranges using `z_to_chunk_index.py`.

### 7. Remote Data Support

Process N5/Zarr data directly from remote sources without downloading.

**Supported sources:**
- HTTP/HTTPS URLs (e.g., Keller lab servers)
- Google Cloud Storage (`gs://` URLs)
- AWS S3 (`s3://` URLs)

**Setup for Google Cloud Storage:**
```bash
gcloud auth application-default login
```

**Example - Convert from Google Cloud:**
```bash
python -m tensorswitch --task n5_to_zarr2 \
  --base_path "gs://bucket-name/path/to/data.n5/s0" \
  --output_path /local/output.zarr \
  --level 0
```

**Inspect remote data:**
```bash
pixi run python inspect_gcs_neuroglancer.py gs://bucket-name/path/to/data
```

Benefits: No download required, saves storage space, faster start.

---

## Contrib Scripts

### update_metadata.py
Update OME-Zarr multiscale metadata and optionally add ome_xml information:

```bash
# Update metadata for levels s0-s4
python contrib/update_metadata.py /path/to/output.zarr

# Auto-detect max level and add ome_xml from corresponding ND2 files
python contrib/update_metadata.py /path/to/output.zarr --check-ome-xml

# Dry run to see what would be done
python contrib/update_metadata.py /path/to/output.zarr --check-ome-xml --dry-run
```

### bleaching_correction_task.py
Z-direction bleaching correction for microscopy datasets:

```bash
# Apply bleaching correction to Zarr dataset
python contrib/bleaching_correction_task.py --input_path /path/to/input.zarr --output_path /path/to/corrected.zarr
```


### add_dimension_names.py
Add dimension names to zarr.json files for all levels in a dataset:

```bash
# Add dimension names to all levels in a zarr dataset
python contrib/add_dimension_names.py /path/to/dataset.zarr
```

### submit_bleaching_correction_general.py
Generic bleaching correction submission script for s0 level:

```bash
# Submit bleaching correction with default settings
python contrib/submit_bleaching_correction_general.py /path/to/input.zarr /path/to/corrected.zarr

# Submit with custom volumes and project
python contrib/submit_bleaching_correction_general.py /path/to/input.zarr /path/to/corrected.zarr 4 your_project
```

### update_metadata_zarr2.py
Update zarr v2 metadata to fix common issues:

```bash
# Fix ome_xml placement in zarr v2 metadata
python contrib/update_metadata_zarr2.py /path/to/dataset.zarr --check-ome-xml

# Update multiscale metadata structure
python contrib/update_metadata_zarr2.py /path/to/dataset.zarr --update-multiscale
```

### Other Contrib Scripts
- **start_neuroglancer_server.py**: Start a CORS-enabled web server for neuroglancer viewing
- **z_to_chunk_index.py**: Calculate chunk index ranges for resubmitting specific z-slices
- **re_submit_jobs.ipynb**: Interactive notebook for debugging and resubmitting failed chunks

---

## Testing

The `tests/` directory contains validation scripts for different conversion workflows:

- **test_ims_to_zarr3_middle_chunks.py**: Validates IMS to Zarr3 conversion with OME-Zarr metadata structure
- **test_nd2_to_zarr3_middle_chunks.py**: Validates ND2 to Zarr3 conversion with OME-Zarr metadata structure
- **test_n5_to_*.py**: Various N5 conversion and downsampling tests
- **test_zarr3_to_downsample_*.py**: Zarr3 downsampling tests with and without sharding

Run tests individually:
```bash
python tests/test_ims_to_zarr3_middle_chunks.py
```

---

## OME-Zarr Structure

All s0 conversion tasks create a consistent multiscale structure:

```
output.zarr/
└── multiscale/
    ├── zarr.json              # OME-Zarr metadata with relative paths ("s0", "s1", etc.)
    ├── s0/                    # Full resolution data
    ├── s1/                    # 2x downsampled
    ├── s2/                    # 4x downsampled  
    ├── s3/                    # 8x downsampled
    └── s4/                    # 16x downsampled
```

This structure is compatible with neuroglancer and other OME-Zarr viewers.

---

## Requirements

- Python 3.10+
- TensorStore
- NumPy
- psutil
- requests (for N5 over HTTP)
- Dask + tifffile (for TIFF conversion)
- nd2 + ome-zarr (for ND2 conversion with OME metadata)
- h5py (for IMS conversion)
- json (for metadata handling)
- glob (for file pattern matching)

---

## Support

If you need more enhancements (like adding logging or progress tracking), feel free to extend the `tasks` modules.

For resubmissions, consider using the interactive notebook `re_submit_jobs.ipynb` and CLI helpers in `z_to_chunk_index.py`.

---
