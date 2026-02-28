# TensorSwitch v2

**Version**: 2.0.0-beta
**Status**: Production Ready
**Branch**: `unified`

A high-performance microscopy data conversion tool with TensorStore as the unified intermediate format. Supports 200+ input formats, automatic multi-scale pyramid generation, and distributed processing on LSF clusters.

> **Looking for v1?** The original task-based TensorSwitch (v1) is on the [`develop` branch](https://github.com/JaneliaSciComp/tensorswitch/tree/develop). Use `git checkout develop` to access it.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Python API](#python-api)
7. [Supported Formats](#supported-formats)
8. [Multi-Scale Pyramids](#multi-scale-pyramids)
9. [Batch Processing](#batch-processing)
10. [LSF Cluster Submission](#lsf-cluster-submission)
11. [Auto-Calculation](#auto-calculation)
12. [Examples](#examples)

---

<p align="center">
  <a href="https://drive.google.com/file/d/1HWX7P-EZfj_NOKsv0YdbXXnlL8DXirSW/view?usp=sharing">
    <img src="PPT_v2/video_thumbnail.png" width="100%" alt="TensorSwitch v2 Demo Video">
  </a>
  <br>
  <em>Click the image to watch the Data Translator demo video on Google Drive</em>
</p>

---

## Features

- **Universal Format Support**: Read 200+ microscopy formats via three-tier reader strategy
- **TensorStore Backend**: High-performance intermediate format for efficient chunk processing
- **Auto-Detection**: Automatic format detection and optimal reader selection
- **Multi-Scale Pyramids**: Automatic pyramid generation with chained downsampling
- **Batch Processing**: Convert thousands of files with LSF job arrays
- **LSF Cluster Support**: Auto-calculated resources (memory, wall time, cores)
- **Preserve Source Layout**: Maintains source dimensionality (3D/4D/5D) and axis order per OME-NGFF RFC-3
- **Compression**: zstd compression with configurable levels

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    CLI (python -m tensorswitch_v2)              │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Processing Layer                       │
│  DistributedConverter │ PyramidPlanner │ BatchConverter         │
└─────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
┌─────────────────────┐  ┌─────────────┐  ┌─────────────────────┐
│      Readers        │  │ TensorStore │  │      Writers        │
│  (120+ formats)     │◀─│   Array     │─▶│  (3 formats)        │
│  Tier 1/2/3/4       │  │ (Unified)   │  │  Zarr3/Zarr2/N5     │
└─────────────────────┘  └─────────────┘  └─────────────────────┘
```

### Three-Tier Reader Strategy

| Tier | Performance | Formats | Description |
|------|-------------|---------|-------------|
| **Tier 1** | Maximum | N5, Zarr2, Zarr3, Precomputed | Native TensorStore drivers |
| **Tier 2** | Optimized | TIFF, ND2, IMS, HDF5, CZI | Custom optimized readers |
| **Tier 3** | Compatible | LIF + 20 more | BIOIO Python plugins |
| **Tier 4** | Universal | 150+ formats | Bio-Formats Java (via bioio-bioformats) |

---

## Installation

TensorSwitch v2 is part of the TensorSwitch package.

### Option 1: Pixi Environment (Recommended for Janelia)

```bash
cd /path/to/tensorswitch
pixi install
```

### Option 2: Pip Install

```bash
# From local checkout (editable mode for development)
pip install -e /path/to/tensorswitch

# From GitHub
pip install git+https://github.com/JaneliaSciComp/tensorswitch.git@unified
```

### Verify Installation

```bash
# Using pixi
pixi run tensorswitch-v2 --version
# Output: tensorswitch_v2 2.0.0-beta

# Or using module syntax
pixi run python -m tensorswitch_v2 --version

# After pip install (no pixi needed)
tensorswitch-v2 --version
```

### Python API

```python
from tensorswitch_v2 import __version__, TensorSwitchDataset, Readers, Writers
print(__version__)  # 2.0.0-beta
```

---

## Quick Start

> **Note**: Examples below show `pixi run` prefix for Janelia cluster. If you installed via pip, omit `pixi run`.

### Single File Conversion

```bash
# Convert TIFF to Zarr3 (auto-detect format)
pixi run tensorswitch-v2 -i input.tif -o output.zarr

# Convert ND2 to Zarr3 with custom chunk shape
pixi run tensorswitch-v2 -i input.nd2 -o output.zarr --chunk_shape 32,256,256

# Convert to Zarr2 format
pixi run tensorswitch-v2 -i input.tif -o output.zarr --output_format zarr2
```

### Generate Multi-Scale Pyramid

```bash
# After s0 conversion, generate full pyramid
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i output.zarr/s0 -o output.zarr
```

### Submit to LSF Cluster

```bash
# Submit conversion job to cluster
pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr \
  --submit -P scicompsoft
```

---

## CLI Reference

### Basic Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input` | Input file or directory | Required |
| `-o, --output` | Output path | Required |
| `--output_format` | Output format: `zarr3`, `zarr2`, `n5` | `zarr3` |
| `-V, --version` | Show version | - |
| `--quiet` | Suppress progress output | False |
| `--show_spec` | Preview conversion specs without running | False |
| `--omero` | Include structured omero channel metadata | False |

### Presets

| Argument | Description |
|----------|-------------|
| `--preset webknossos` | WebKnossos-optimized settings: zarr3, chunk 32x32x32, shard 1024x1024x1024, zstd |

```bash
# Example: Convert for WebKnossos viewing
pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr --preset webknossos
```

### Chunk/Shard Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--chunk_shape` | Inner chunk shape (e.g., `32,256,256`) | Auto |
| `--shard_shape` | Shard shape for Zarr3 (e.g., `64,512,512`) | Auto |
| `--no_sharding` | Disable sharding (Zarr3 only) | False |
| `--compression` | Compression codec | `zstd` |
| `--compression_level` | Compression level (1-22) | `5` |

### Downsampling

| Argument | Description |
|----------|-------------|
| `--auto_multiscale` | Generate full pyramid from s0 |
| `--downsample` | Single-level downsampling mode |
| `--target_level` | Target level for single-level mode |
| `--single_level_factor` | Factor for single-level mode (e.g., `1,4,4` for s2) |
| `--per_level_factors` | Custom per-level factors for pyramid (e.g., `1,2,2;1,2,2;1,2,2`) |
| `--downsample_method` | Downsampling method: `auto` (default), `mean`, `mode`, `median`, `stride`, `min`, `max` |

### LSF Cluster

| Argument | Description | Default |
|----------|-------------|---------|
| `--submit` | Submit as LSF job | False |
| `-P, --project` | LSF project name | Required with `--submit` |
| `--memory` | Memory in GB | Auto-calculated |
| `--wall_time` | Wall time (H:MM) | Auto-calculated |
| `--cores` | Number of cores | Auto-calculated |
| `--job_group` | LSF job group | `/scicompsoft/chend/tensorstore` |

### Batch Processing

| Argument | Description | Default |
|----------|-------------|---------|
| `--pattern` | File pattern for batch mode | `*.tif` |
| `--recursive` | Search subdirectories | False |
| `--max_concurrent` | Max concurrent LSF jobs | `100` |
| `--skip_existing` | Skip completed files | True |
| `--status` | Check batch status | - |
| `--dry_run` | Preview without executing | False |

### Format-Specific

| Argument | Description |
|----------|-------------|
| `--view_index` | CZI view index (None = all views as 5D) |
| `--use_bioio` | Force BIOIO adapter (Tier 3, Python plugins) |
| `--use_bioformats` | Force Bio-Formats reader (Tier 4, Java-backed, 150+ formats) |
| `--dataset_path` | Path within container (e.g., `s0` for N5) |

### Layout Preservation

| Argument | Description |
|----------|-------------|
| `--expand-to-5d` | Force 5D TCZYX expansion (legacy behavior) |

**Default behavior (RFC-3 compliant)**: Source dimensionality and axis order are preserved:
- 3D XYZ → 3D XYZ (not expanded to 5D)
- 4D CZYX → 4D CZYX
- Axis order preserved (XYZ stays XYZ, ZYX stays ZYX)

**Singleton channel squeeze**: When reading neuroglancer precomputed format with `num_channels=1`, the implicit 4th channel dimension is automatically squeezed out to preserve true 3D output.

Use `--expand-to-5d` for compatibility with tools requiring strict 5D TCZYX format.

### Memory Order

| Argument | Description |
|----------|-------------|
| `--force_c_order` | Force C-order (row-major) output |
| `--force_f_order` | Force F-order (column-major) output |

**Default behavior**: Source order is auto-detected and preserved. Use these flags only to override.

### Metadata Override

| Argument | Description |
|----------|-------------|
| `--voxel_size` | Override voxel size in nanometers, comma-separated X,Y,Z (e.g., `160,160,400`) |

**Use case**: When source files lack embedded voxel size metadata (e.g., raw TIFF stacks, BigStitcher output).

```bash
# Example: Set voxel size for a TIFF without embedded metadata (160nm x 160nm x 400nm)
pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr \
  --voxel_size 160,160,400
```

### Unit Handling

TensorSwitch v2 follows these rules for spatial units:

| Operation | Unit Behavior |
|-----------|---------------|
| **New conversions** (TIFF/ND2/CZI → Zarr) | Always converts to **nanometers**. Source units (µm, mm, etc.) are auto-detected and converted. |
| **Downsampling/Pyramid** (existing Zarr → pyramid levels) | **Preserves source s0 unit** (micrometer, nanometer, etc.) for consistency. |

**Examples:**
- TIFF with 0.116 µm voxels → Zarr with `"unit": "nanometer"` and scale `[116.0, 116.0, 400.0]`
- BigStitcher Zarr2 with `"unit": "micrometer"` → Pyramid levels keep `"unit": "micrometer"`

---

## Python API

### TensorSwitchDataset

```python
from tensorswitch_v2.api import TensorSwitchDataset, Readers, Writers

# Auto-detect format and create dataset
dataset = TensorSwitchDataset("/path/to/data.tif")

# Access properties
print(dataset.shape)      # (100, 2048, 2048)
print(dataset.dtype)      # uint16
print(dataset.ndim)       # 3

# Get TensorStore array
ts_array = dataset.get_tensorstore_array(mode='open')

# Get OME-NGFF metadata
metadata = dataset.get_ome_ngff_metadata()
voxel_sizes = dataset.get_voxel_sizes()  # {'x': nm, 'y': nm, 'z': nm} in nanometers
```

### Readers Factory

```python
from tensorswitch_v2.api import Readers

# Auto-detect (recommended)
reader = Readers.auto_detect("/path/to/data.tif")

# Explicit reader selection
reader = Readers.tiff("/path/to/data.tif")      # Tier 2
reader = Readers.nd2("/path/to/data.nd2")       # Tier 2
reader = Readers.czi("/path/to/data.czi")       # Tier 2
reader = Readers.ims("/path/to/data.ims")       # Tier 2
reader = Readers.n5("/path/to/data.n5")         # Tier 1
reader = Readers.zarr3("/path/to/data.zarr")    # Tier 1
reader = Readers.bioio("/path/to/data.lif")     # Tier 3
reader = Readers.bioformats("/path/to/data.vsi")  # Tier 4 (Java)

# Get TensorStore spec
spec = reader.get_tensorstore_spec()
metadata = reader.get_metadata()
```

### Writers Factory

```python
from tensorswitch_v2.api import Writers

# Create writers
writer = Writers.zarr3("/path/to/output.zarr",
                       use_sharding=True,
                       compression="zstd",
                       compression_level=5)

writer = Writers.zarr2("/path/to/output.zarr",
                       compression="blosc")

writer = Writers.n5("/path/to/output.n5")
```

### DistributedConverter

```python
from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter

# Create reader and writer
reader = Readers.auto_detect("/path/to/input.tif")
writer = Writers.zarr3("/path/to/output.zarr")

# Create converter
converter = DistributedConverter(reader, writer)

# Full conversion (chunk/shard shapes are optional - auto-calculated if omitted)
converter.convert(
    chunk_shape=(32, 256, 256),   # Optional: example value
    shard_shape=(64, 512, 512),   # Optional: example value
    write_metadata=True,
    verbose=True
)

# Or let it auto-calculate optimal shapes (recommended)
converter.convert(write_metadata=True, verbose=True)

# Partial conversion (for LSF multi-job mode)
converter.convert(
    start_idx=0,
    stop_idx=100,
    write_metadata=False  # Last job writes metadata
)
```

### PyramidPlanner

```python
from tensorswitch_v2.core import PyramidPlanner, create_pyramid_parallel

# Plan pyramid levels
planner = PyramidPlanner("/path/to/dataset.zarr/s0")
plan = planner.calculate_pyramid_plan()

# Print plan
planner.print_pyramid_plan(plan)

# Submit chained pyramid jobs to LSF
planner.submit_chained_pyramid(
    pyramid_plan=plan,
    project="scicompsoft",
    output_format="zarr3"
)

# Or use convenience function
create_pyramid_parallel(
    s0_path="/path/to/dataset.zarr/s0",
    root_path="/path/to/dataset.zarr",
    project="scicompsoft"
)
```

### BatchConverter

```python
from tensorswitch_v2.core import BatchConverter

# Create batch converter
batch = BatchConverter(
    input_dir="/path/to/tiffs/",
    output_dir="/path/to/output/",
    pattern="*.tif",
    output_format="zarr3",
    recursive=False
)

# Discover files
files = batch.discover()
print(f"Found {len(files)} files")

# Submit to LSF
batch.submit_lsf(
    project="scicompsoft",
    memory_gb=30,
    wall_time="1:00",
    max_concurrent=100
)

# Check status
result = batch.check_status()
print(f"Completed: {result.completed}/{result.total}")
```

---

## Supported Formats

### Input Formats

| Format | Extension | Tier | Reader |
|--------|-----------|------|--------|
| TIFF | `.tif`, `.tiff` | 2 | TiffReader |
| ND2 (Nikon) | `.nd2` | 2 | ND2Reader |
| IMS (Imaris) | `.ims` | 2 | IMSReader |
| CZI (Zeiss) | `.czi` | 2 | CZIReader |
| HDF5 | `.h5`, `.hdf5` | 2 | HDF5Reader |
| N5 | `.n5` | 1 | N5Reader |
| Zarr v3 | `.zarr` | 1 | Zarr3Reader |
| Zarr v2 | `.zarr` | 1 | Zarr2Reader |
| Precomputed | directory | 1 | PrecomputedReader |
| LIF (Leica) | `.lif` | 3 | BIOIOReader |
| OME-TIFF | `.ome.tif` | 3 | BIOIOReader |
| 20+ more | various | 3 | BIOIOReader |
| Olympus VSI, Leica SCN, etc. | various | 3+ | BioFormatsReader (Java) |
| 150+ formats | various | 3+ | BioFormatsReader (Java) |

### Source Layout Preservation

TensorSwitch v2 preserves source data layout by default (RFC-3 compliant):

**1. Dimension Preservation:**
- 3D → 3D, 4D → 4D (no automatic 5D expansion)
- Precomputed with `num_channels=1`: singleton channel squeezed (`[X,Y,Z,1]` → `[X,Y,Z]`)

**2. Axis Order Rules:**
- Source order preserved: XYZ→XYZ, ZYX→ZYX
- Non-spatial axes (T, C) always come before spatial axes (Z, Y, X) in OME-NGFF output
- Example: `[X,Y,Z,channel]` → reordered to `[channel,X,Y,Z]` for OME-NGFF compliance

**3. Memory Order (F-order vs C-order):**
- Auto-detected from source and preserved by default
- Override with `--force_c_order` or `--force_f_order`
- F-order (column-major): common in N5, Fortran, MATLAB
- C-order (row-major): common in Python, NumPy, most image formats

**4. Reader-Specific Behavior:**

| Reader | Shape Handling | Notes |
|--------|---------------|-------|
| N5, Zarr | Direct pass-through | TensorStore native |
| Precomputed (1ch) | `[X,Y,Z,1]` → `[X,Y,Z]` | Singleton squeezed |
| Precomputed (Nch) | `[X,Y,Z,N]` preserved | Multi-channel kept |
| TIFF, ND2, CZI, IMS | Shape preserved | Tier 2 readers |

**5. Legacy Mode (`--expand-to-5d`):**
- Input: `[Z, Y, X]` → Output: `[1, 1, Z, Y, X]` (T=1, C=1)
- Input: `[C, Z, Y, X]` → Output: `[1, C, Z, Y, X]` (T=1)
- Use for tools requiring strict OME-NGFF v0.4/v0.5 5D format

### Output Formats

| Format | Description | Features |
|--------|-------------|----------|
| **Zarr v3** | Primary format | Sharding, OME-NGFF v0.5, zstd compression |
| **Zarr v2** | Legacy format | OME-NGFF v0.4, blosc compression |
| **N5** | Java tools | BigDataViewer compatible |

### OME-NGFF Nested Structure

Zarr3 output uses OME-NGFF spec-compliant nested structure:

```
output.zarr/
├── zarr.json                 # Root metadata
├── raw/                      # Image data (for --data-type image)
│   ├── zarr.json
│   └── s0/, s1/...
└── labels/                   # Labels (for --data-type labels)
    ├── zarr.json
    └── segmentation/
        ├── zarr.json         # Includes image-label colors
        └── s0/, s1/...
```

**CLI Options:**

| Argument | Description |
|----------|-------------|
| `--data-type {image,labels,auto}` | Data type (default: auto-detect) |
| `--image PATH` | Explicit image path for combined conversion |
| `--labels PATH` | Explicit labels path for combined conversion |
| `--image-only` | Only convert image when folder has both |
| `--labels-only` | Only convert labels when folder has both |
| `--image-key NAME` | Name for image group (default: "raw") |
| `--label-key NAME` | Name for label image (default: "segmentation") |
| `--no-nested-structure` | Disable nested structure |

**Examples:**

```bash
# Convert as image (default)
tensorswitch -i data.tif -o output.zarr

# Convert as labels/segmentation
tensorswitch -i segmentation.tif -o output.zarr --data-type labels

# Auto-detect from folder (Neuroglancer Precomputed)
tensorswitch -i /path/to/folder/ -o output.zarr
```

See [OME_NGFF_STRUCTURE_PLAN.md](docs/OME_NGFF_STRUCTURE_PLAN.md) for full documentation.

---

## Multi-Scale Pyramids

TensorSwitch v2 uses **chained downsampling** for efficient pyramid generation:

```
Chained Downsampling:
┌───┐
│ 0 │ (original)
└─┬─┘
  │ 2x
  ▼
┌───┐
│ 1 │ ──bwait
└─┬─┘
  │ 2x
  ▼
┌───┐
│ 2 │ ──bwait
└─┬─┘
  ...
```

**Benefits**:
- Constant ~8x read amplification per level
- Deep levels (4, 5) complete in minutes
- Automatic anisotropic factor calculation
- OME-NGFF compliant metadata with translation transforms (Neuroglancer compatible)
- **S-prefixed level naming** (s0/s1/s2) follows Janelia house style (OME-NGFF compatible)
- **Compression inheritance** from level 0 (consistent settings across pyramid)

### Generate Pyramid

**Flexible input paths** - both formats work:
```bash
# Option 1: Root zarr path (auto-detects s0 from metadata or common patterns)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/dataset.zarr \
  --submit -P scicompsoft

# Option 2: Explicit s0 path (also works)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/dataset.zarr/s0 \
  --submit -P scicompsoft
```

**Auto-detection logic**:
1. If path ends with level pattern (`0`, `s0`, `1`, `s1`, etc.) → use as-is
2. Check OME-NGFF metadata (`multiscales[0].datasets[0].path`)
3. Fallback to common subdirectories: `0`, `s0`

**Level naming**: New conversions use s-prefixed format (`s0/s1/s2`) following Janelia house style. When downsampling existing data, the source format is auto-detected and followed:
- Source has `s0/` → creates `s1/`, `s2/`, `s3/`...
- Source has `0/` → creates `1/`, `2/`, `3/`...

**Compression inheritance**: Downsampled levels inherit compression settings from level 0 (e.g., if level 0 uses zstd level 3, all pyramid levels will use zstd level 3).

```bash
# Single level only (e.g., just create s2)
pixi run python -m tensorswitch_v2 --downsample \
  -i /path/to/dataset.zarr/s0 \
  -o /path/to/dataset.zarr \
  --target_level 2 \
  --single_level_factor 1,4,4
```

### Downsampling Factor Arguments

There are two factor arguments for different use cases:

| Argument | Mode | Purpose | Format |
|----------|------|---------|--------|
| `--single_level_factor` | `--downsample` | Create **one** level | `z,y,x` (cumulative from s0) |
| `--per_level_factors` | `--auto_multiscale` | Create **full pyramid** | `z,y,x;z,y,x;...` (per-level) |

**Key difference:**
- `--single_level_factor 1,4,4` → Total 4x reduction on Y,X from s0 (creates one level)
- `--per_level_factors "1,2,2;1,2,2"` → 2x per level (creates multiple levels, cumulative calculated automatically)

### Downsampling Method

The `--downsample_method` option controls how TensorStore computes downsampled values:

| Method | Description | Best For |
|--------|-------------|----------|
| `auto` | Auto-detect from filename (default) | Most use cases |
| `mean` | Average of values | Intensity images (fluorescence, brightfield) |
| `mode` | Most frequent value | Segmentation masks, labels |
| `median` | Median value | Noise reduction |
| `stride` | Simple striding (fastest) | Quick previews |
| `min` | Minimum value | Specific use cases |
| `max` | Maximum value | Specific use cases |

**Auto-detection heuristics:**
- If filename/path contains `label`, `mask`, `seg`, `annotation`, `roi`, `binary`, `instance` → uses `mode`
- Otherwise → uses `mean` (best for most microscopy data)

```bash
# Auto-detect method (default - usually picks 'mean' for microscopy)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/dataset.zarr/s0 \
  -o /path/to/dataset.zarr \
  --submit -P scicompsoft

# Explicitly use mode for segmentation data
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/labels.zarr/s0 \
  -o /path/to/labels.zarr \
  --downsample_method mode \
  --submit -P scicompsoft
```

### Custom Factors for Anisotropic Data

When voxel sizes are not in zarr metadata (e.g., raw TIFF, BigStitcher data), use `--per_level_factors`:

```bash
# Full pyramid with custom factors: skip Z, downsample Y,X by 2x per level
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/dataset.zarr/s0 \
  -o /path/to/dataset.zarr \
  --per_level_factors "1,2,2;1,2,2;1,2,2;1,2,2" \
  --submit -P scicompsoft
```

**How `--per_level_factors` works:**

Each semicolon-separated entry is the factor **from previous level to current level**:

```
--per_level_factors "1,2,2;1,2,2;1,2,2;1,2,2"
                     ─────  ─────  ─────  ─────
                     s0→s1  s1→s2  s2→s3  s3→s4
```

| Level | Per-Level Factor | Cumulative (auto-calculated) | Shape Change |
|-------|------------------|------------------------------|--------------|
| s1 | `[1,2,2]` | `[1,2,2]` | Z same, Y,X ÷2 |
| s2 | `[1,2,2]` | `[1,4,4]` | Z same, Y,X ÷4 |
| s3 | `[1,2,2]` | `[1,8,8]` | Z same, Y,X ÷8 |
| s4 | `[1,2,2]` | `[1,16,16]` | Z same, Y,X ÷16 |

**Example: 4D CZYX data (never downsample channels)**

```bash
--per_level_factors "1,1,2,2;1,1,2,2;1,1,2,2;1,2,2,2"
#                    C Z Y X
```

### Batch Pyramid Generation

Generate pyramids for multiple datasets at once. Supports both regular directories and multi-tile `.zarr` directories (e.g., BigStitcher format).

**Case 1: Directory containing .zarr files**
```bash
# Directory structure:
# /path/to/zarr_output/
# ├── tile001.zarr/
# ├── tile002.zarr/
# └── tile003.zarr/

pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/zarr_output/ \
  --submit -P scicompsoft
```

**Case 2: Multi-tile .zarr directory (BigStitcher format)**
```bash
# Directory structure:
# dataset.ome.zarr/
# ├── s0-t0.zarr/
# ├── s1-t0.zarr/
# ├── s2-t0.zarr/
# └── ...

pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/dataset.ome.zarr \
  --per_level_factors "1,1,1,2,2;1,1,2,2,2;1,1,2,2,2" \
  --submit -P tavakoli
```

**With custom anisotropic factors:**
```bash
# Apply same factors to all tiles (e.g., skip Z, downsample Y,X)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /path/to/zarr_output/ \
  --per_level_factors "1,2,2;1,2,2;1,2,2;1,2,2" \
  --max_concurrent 50 \
  --submit -P scicompsoft
```

**Paths with spaces are supported:**
```bash
# Paths containing spaces work correctly (use quotes)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i "/path/to/my data/dataset.ome.zarr" \
  --submit -P scicompsoft
```

Each dataset gets its own coordinator job that spawns chained downsampling jobs. When using `--per_level_factors`, the same factors are applied to all datasets in the batch.

---

## Batch Processing

Convert multiple files using LSF job arrays:

```bash
# Discover and convert all TIFFs in directory
pixi run python -m tensorswitch_v2 \
  -i /path/to/tiff_directory/ \
  -o /path/to/output_directory/ \
  --pattern "*.tif" \
  --submit -P scicompsoft \
  --max_concurrent 100

# Check status
pixi run python -m tensorswitch_v2 --status \
  -i /path/to/tiff_directory/ \
  -o /path/to/output_directory/

# Dry run (preview only)
pixi run python -m tensorswitch_v2 \
  -i /path/to/tiff_directory/ \
  -o /path/to/output_directory/ \
  --pattern "*.tif" \
  --dry_run
```

### Rechunking (Same Format, Different Chunks)

For rechunking existing data (e.g., N5 → N5 with different chunk shape), the **output path determines the behavior**:

```bash
# FORMAT CONVERSION: N5 → Zarr3 (different format)
# Output to .zarr → converts to Zarr3 s0, then use --auto_multiscale for pyramid
pixi run python -m tensorswitch_v2 \
  -i /source/dataset.n5/setup0/timepoint0/s0 \
  -o /output/dataset.zarr/s0 \
  --output_format zarr3 \
  --submit -P scicompsoft

# RECHUNK: N5 → N5 (same format, new chunks)
# Output to same level path (s0, s1, etc.) → rechunks with new chunk shape
pixi run python -m tensorswitch_v2 \
  -i /source/dataset.n5/setup0/timepoint0/s0 \
  -o /output/dataset.n5/setup0/timepoint0/s0 \
  --output_format n5 \
  --chunk_shape 128,128,128 \
  --submit -P liconn
```

**Key distinction**: When output format matches input format AND output path ends with a level (s0, s1, etc.), this is a rechunk operation.

For datasets with multiple levels (like Keller lab N5), use a shell script to rechunk all levels in parallel:

```bash
# Rechunk all levels in parallel (shell script approach)
for level in {0..4}; do
  case $level in
    0) cores=8 ;;
    1) cores=4 ;;
    *) cores=2 ;;
  esac

  pixi run python -m tensorswitch_v2 \
    -i "/source/dataset.n5/setup0/timepoint0/s${level}" \
    -o "/output/dataset.n5/setup0/timepoint0/s${level}" \
    --output_format n5 \
    --chunk_shape 128,128,128 \
    --cores ${cores} \
    --submit -P liconn
done
```

---

## LSF Cluster Submission

> **Note**: The `-P` flag specifies your LSF project for job accounting. Replace `scicompsoft` with your own lab's project code (e.g., `-P ahrens`, `-P liconn`, `-P tavakoli`). Contact Scientific Computing if you don't know your project code.

### Auto-Calculated Resources

TensorSwitch v2 automatically calculates optimal resources based on input data:

| Resource | Formula | Constraints |
|----------|---------|-------------|
| **Memory** | Based on shard size × concurrent buffers | 5-500 GB, 15 GB/core minimum |
| **Wall Time** | Based on shard count × per-shard time | Capped at 96 hours (4 days) |
| **Cores** | Based on memory (I/O bound) | Capped at 8 cores |

### Manual Override

```bash
# Override auto-calculated values
pixi run python -m tensorswitch_v2 -i input.tif -o output.zarr \
  --submit -P scicompsoft \
  --memory 60 \
  --wall_time 4:00 \
  --cores 4
```

### Check Job Status

```bash
# Check running job
bjobs <job_id>

# View job details
bjobs -l <job_id>

# Extend wall time (if needed)
bmod -W 96:00 <job_id>
```

---

## Auto-Calculation

### Format Auto-Detection

```python
# Extension-based detection
.tif, .tiff  → TiffReader (Tier 2)
.nd2         → ND2Reader (Tier 2)
.ims         → IMSReader (Tier 2)
.czi         → CZIReader (Tier 2)
.n5          → N5Reader (Tier 1)
.zarr        → Zarr3Reader or Zarr2Reader (Tier 1)

# Directory detection
zarr.json    → Zarr v3
.zgroup      → Zarr v2
attributes.json → N5
info         → Precomputed
```

### Chunk Shape Auto-Calculation

- Non-spatial axes (t, c): chunk = 1 (per-channel access)
- **Zarr3 sharded** (default): inner chunk = 256, shard = 1024
- **Zarr3 non-sharded / Zarr2**: adaptive spatial chunk based on dataset size:
  - < 20 GB → 64, 20–100 GB → 128, > 100 GB → 256

### LSF Resource Auto-Calculation

When using `--submit`, memory, wall time, and cores are auto-calculated per format:

| | Zarr3 Sharded | Zarr3 No-Shard / Zarr2 |
|---|---|---|
| **Cores** | ceil(mem/15) × 2, cap 8 | min 4, ceil(size_gb/25) × 2, cap 8 |
| **Memory** | shard buffers + base | reader overhead + base (then max with cores × 15 GB) |
| **Wall time** | per-shard / (cores × 0.85) | max(throughput, per-chunk) / (cores × 0.7) |

Override with `--memory`, `--wall_time`, `--cores`. See [docs/RESOURCE_AUTO_CALCULATION.md](docs/RESOURCE_AUTO_CALCULATION.md) for details.

---

## Examples

### Example 1: Large TIFF to Zarr3 with Pyramid

```bash
# Step 1: Convert TIFF to Zarr3 s0
pixi run python -m tensorswitch_v2 \
  -i /data/large_dataset.tif \
  -o /output/large_dataset.zarr \
  --submit -P scicompsoft

# Step 2: Generate pyramid (after s0 completes)
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /output/large_dataset.zarr/s0 \
  -o /output/large_dataset.zarr \
  --submit -P scicompsoft
```

### Example 2: Batch Convert ND2 Files

```bash
pixi run python -m tensorswitch_v2 \
  -i /data/nd2_files/ \
  -o /output/zarr_files/ \
  --pattern "*.nd2" \
  --submit -P scicompsoft \
  --max_concurrent 50
```

### Example 3: Batch Pyramid Generation

```bash
# Generate pyramids for all zarr files in a directory
pixi run python -m tensorswitch_v2 --auto_multiscale \
  -i /output/zarr_files/ \
  --pattern '*.zarr' \
  --max_concurrent 50 \
  --submit -P scicompsoft
```

### Example 5: CZI Multi-View Conversion

```bash
# Convert all views as 5D TCZYX (views mapped to time axis for viewer compatibility)
pixi run python -m tensorswitch_v2 \
  -i /data/multiview.czi \
  -o /output/multiview.zarr \
  --submit -P scicompsoft

# Extract single view
pixi run python -m tensorswitch_v2 \
  -i /data/multiview.czi \
  -o /output/view0.zarr \
  --view_index 0 \
  --submit -P scicompsoft
```

### Example 6: Python API Conversion

```python
from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter

# Read TIFF, write Zarr3
reader = Readers.tiff("/data/input.tif")
writer = Writers.zarr3("/output/result.zarr")

converter = DistributedConverter(reader, writer)
converter.convert(
    chunk_shape=(1, 128, 128, 128),
    shard_shape=(1, 512, 512, 512),
    write_metadata=True,
    verbose=True
)
```

---

## Module Structure

```
tensorswitch_v2/
├── __init__.py              # Package root
├── __main__.py              # CLI entry point
├── api/
│   ├── __init__.py          # Public API exports
│   ├── dataset.py           # TensorSwitchDataset
│   ├── readers.py           # Readers factory
│   └── writers.py           # Writers factory
├── core/
│   ├── __init__.py          # Core exports
│   ├── converter.py         # DistributedConverter
│   ├── downsampler.py       # Downsampler, cumulative factors
│   ├── pyramid.py           # PyramidPlanner, chained submission
│   └── batch.py             # BatchConverter, LSF job arrays
├── readers/
│   ├── __init__.py          # Reader exports
│   ├── base.py              # BaseReader abstract class
│   ├── n5.py                # N5Reader (Tier 1)
│   ├── zarr.py              # Zarr3Reader, Zarr2Reader (Tier 1)
│   ├── precomputed.py       # PrecomputedReader (Tier 1)
│   ├── tiff.py              # TiffReader (Tier 2)
│   ├── nd2.py               # ND2Reader (Tier 2)
│   ├── ims.py               # IMSReader (Tier 2)
│   ├── hdf5.py              # HDF5Reader (Tier 2)
│   ├── czi.py               # CZIReader (Tier 2)
│   ├── bioio_adapter.py     # BIOIOReader (Tier 3)
│   └── bioformats.py        # BioFormatsReader (Tier 4, Java)
└── writers/
    ├── __init__.py          # Writer exports
    ├── base.py              # BaseWriter abstract class
    ├── zarr3.py             # Zarr3Writer (OME-NGFF v0.5)
    ├── zarr2.py             # Zarr2Writer (OME-NGFF v0.4)
    └── n5.py                # N5Writer
```

---

## Performance

Tested on Janelia LSF cluster:

| Dataset | Size | Output | Time | Compression |
|---------|------|--------|------|-------------|
| Lila Batch (1890 TIFFs) | 6.6 TB | 758 GB Zarr3 | 15 min | 8.7x |
| CZI Pyramid (6 levels) | 446 GB | 571 GB total | 73 min | - |
| Ahrens TIFF | 1.9 TB | ~800 GB Zarr3 | ~70 hrs | ~2.4x |

---

## License

Internal Janelia Research Campus tool.

---

## Authors

- Diyi Chen (SciComp)

---

## Links

- **TensorStore Docs**: https://google.github.io/tensorstore/
- **BIOIO Docs**: https://github.com/bioio-devs/bioio
