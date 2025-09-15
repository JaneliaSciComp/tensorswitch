# TensorSwitch

This package provides a unified entry point for managing N5/Zarr dataset conversions and downsampling with both command-line and web GUI interfaces. It centralizes your workflow into a single pipeline to reduce manual work and errors.

## Features

- **Web GUI Interface**: Production-ready web interface for scientists (no programming required)
- **Command-Line Interface**: Full-featured CLI for automated workflows
- **Enhanced OME-ZARR Metadata**: Automatic preservation of rich metadata from TIFF, ND2, and IMS files
- **Cluster Integration**: LSF job submission with resource management
- **Multiple Format Support**: TIFF, ND2, IMS, N5, and Zarr conversions

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
│       └── gui/                          # Web GUI interface
│           ├── app.py                    # Main GUI application
│           ├── launch_gui.py             # GUI server launcher
│           ├── README_GUI.md             # GUI documentation
│           └── lab_paths_system/         # HHMI lab paths integration
├── contrib
│   ├── bleaching_correction_task.py      # Z-direction bleaching correction for microscopy datasets
│   ├── re_submit_jobs.ipynb              # Jupyter notebook to re-submit failed chunk jobs
│   ├── start_neuroglancer_server.py      # Start a CORS-enabled web server
│   ├── submit_bleaching_correction_s0.py # Submit bleaching correction jobs to LSF cluster
│   ├── update_metadata.py                # Update OME-Zarr multiscale metadata and add ome_xml
│   └── z_to_chunk_index.py               # Print chunk index ranges for resubmit failed or left over jobs
└── tests
    ├── test_n5_to_n5.py                       # Test N5 to N5 conversion
    ├── test_n5_to_zarr2.py                    # Test N5 to Zarr2 conversion
    ├── test_n5_to_zarr3_downsample_shard.py   # Test N5 to Zarr3 downsampling with shards
    ├── test_zarr3_to_downsample_noshard_zarr3.py # Test Zarr3 downsampling without shards
    ├── test_zarr3_to_downsample_shard_zarr3.py   # Test Zarr3 downsampling with shards
    ├── test_nd2_to_zarr3_middle_chunks.py     # Test ND2 to Zarr3 middle chunk verification
    └── test_ims_to_zarr3_middle_chunks.py     # Test IMS to Zarr3 middle chunk verification

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
- Simple 3-step workflow: Select input → Configure → Convert
- Built-in path suggestions for 131 HHMI labs
- Real-time progress monitoring
- Local or cluster job submission
- All 6 conversion tasks supported

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


### 4. Example Commands

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

### submit_bleaching_correction_s0.py
Submit bleaching correction jobs to LSF cluster:

```bash
# Submit bleaching correction to cluster
python contrib/submit_bleaching_correction_s0.py --input_path /path/to/input.zarr --output_path /path/to/corrected.zarr --project your_project
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
