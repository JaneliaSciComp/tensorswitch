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
│       │   ├── n5_to_n5.py # N5 to N5 conversion logic
│       │   ├── n5_to_zarr2.py # N5 to Zarr V2 conversion logic
│       └── utils.py # Common utilities (chunk domain calculation)
└── tests
    ├── test_n5_to_n5.py
    ├── test_n5_to_zarr2.py
    ├── test_n5_to_zarr3_downsample_shard.py
    ├── test_zarr3_to_downsample_noshard_zarr3.py
    └── test_zarr3_to_downsample_shard_zarr3.py
```

## How to Use

### 1. Run the Main Pipeline

Use the `python -m tensorswitch` as the entry point. Example:

```bash
python -m tensorswitch --task n5_to_zarr2     --base_path /path/to/n5     --output_path /path/to/zarr2     --level 0
```

### 2. Supported Tasks

| Task Name            | Description |
|------------------|----|
| `n5_to_zarr2`        | Convert N5 to Zarr V2 |
| `n5_to_n5`           | Convert N5 to N5 (new chunking) |
| `downsample_zarr`    | Downsample existing Zarr dataset |
| `submit_downsample`  | Submit downsampling jobs to the cluster |
| `submit_n5_zarr`     | Submit N5 to Zarr jobs to the cluster |
| `submit_n5_n5`       | Submit N5 to N5 jobs to the cluster |

### 3. Example Commands

#### Convert N5 to Zarr V2 locally
```bash
python -m tensorswitch --task n5_to_zarr2     --base_path /path/to/n5     --output_path /path/to/zarr2     --level 0
```

#### Downsample Zarr locally
```bash
python -m tensorswitch  --task downsample_zarr     --base_path /path/to/zarr     --level 1     --use_shard 1
```

#### Submit N5 to Zarr jobs to cluster
```bash
python -m tensorswitch --task submit_n5_zarr
```

---

## Requirements

- Python 3.10+
- TensorStore
- NumPy
- psutil
- requests (for N5 over HTTP)

---

## Support

If you need more enhancements (like adding logging or progress tracking), feel free to extend the `tasks` modules.

---
