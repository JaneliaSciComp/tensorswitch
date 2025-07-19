# TensorSwitch

This package provides a unified entry point and organized task modules for managing N5/Zarr dataset conversions and downsampling. It centralizes your workflow into a single pipeline to reduce manual work and errors.

## Folder Structure

```
tensorswitch/
├── pixi.lock
├── pyproject.toml
├── README.md
├── src
│   └── tensorswitch
│       ├── __init__.py
│       ├── __main__.py # Main dispatcher script
│       ├── tasks
│       │   ├── __init__.py
│       │   ├── downsample_shard_zarr3.py # Downsample using shards
│       │   ├── n5_to_n5.py               # N5 to N5 conversion logic
│       │   ├── n5_to_zarr2.py            # N5 to Zarr V2 conversion logic
│       │   ├── tiff_to_zarr3_s0.py       # TIFF to Zarr V3 level s0 conversion logic  
│       ├── utils.py                      # Common utilities (chunk domain calculation)
│       ├── z_to_chunk_index.py           # Print chunk index ranges for resubmit failed or left over jobs
├── re_submit_jobs.ipynb                  # Jupyter notebook to re-submit failed chunk jobs
└── tests
    ├── test_n5_to_n5.py
    ├── test_n5_to_zarr2.py
    ├── test_n5_to_zarr3_downsample_shard.py
    ├── test_zarr3_to_downsample_noshard_zarr3.py
    └── test_zarr3_to_downsample_shard_zarr3.py

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

| Task Name            | Description |
|----------------------|-----------------------------------------|
| `n5_to_zarr2`        | Convert N5 to Zarr V2                   |
| `n5_to_n5`           | Convert N5 to N5 (new chunking)         |
| `downsample_zarr`    | Downsample existing Zarr dataset        |
| `submit_downsample`  | Submit downsampling jobs to the cluster |
| `submit_n5_zarr`     | Submit N5 to Zarr jobs to the cluster   |
| `submit_n5_n5`       | Submit N5 to N5 jobs to the cluster     |
| `tiff_to_zarr3_s0`   | Convert TIFF stack to Zarr V3 (s0)      |


### 4. Example Commands

#### Convert N5 to Zarr V2 locally
```bash
python -m tensorswitch --task n5_to_zarr2     --base_path /path/to/n5     --output_path /path/to/zarr2     --level 0
```

#### Downsample Zarr locally
```bash
python -m tensorswitch  --task downsample_shard_zarr3     --base_path /path/to/zarr/s0     --output_path /path/to/zarr     --level 1     --use_shard 1
```

#### Submit N5 to Zarr jobs to cluster
```bash
python -m tensorswitch --task n5_to_zarr2 --base_path /path/to/n5 --output_path /path/to/zarr2 --submit --project your_project_name
```

#### Convert TIFF to Zarr v3 s0
```bash
python -m tensorswitch --task tiff_to_zarr3_s0 --base_path /path/to/tiff_folder --output_path /path/to/zarr3 --use_shard 0
```

#### Resubmit Failed Chunks
Use `re_submit_jobs.ipynb` to debug chunk failures and resubmit specific chunk ranges using `z_to_chunk_index.py`.


---

## Requirements

- Python 3.10+
- TensorStore
- NumPy
- psutil
- requests (for N5 over HTTP)
- Dask + tifffile (for TIFF conversion)

---

## Support

If you need more enhancements (like adding logging or progress tracking), feel free to extend the `tasks` modules.

For resubmissions, consider using the interactive notebook `re_submit_jobs.ipynb` and CLI helpers in `z_to_chunk_index.py`.

---
