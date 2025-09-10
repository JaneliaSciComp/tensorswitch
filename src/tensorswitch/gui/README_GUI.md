# TensorSwitch GUI

A simple web interface for converting scientific data formats. No programming required!

## Quick Start

### 1. Launch the GUI
```bash
# From tensorswitch root directory
pixi run python src/tensorswitch/gui/launch_gui.py
```

### 2. Access the GUI
Open your web browser and go to:
- **JupyterHub**: `http://[your-hostname].int.janelia.org:5000`
- **Local**: `http://localhost:5000`

Replace `[your-hostname]` with your actual computer name (like `e10u02`).

**Custom Port**: Set `export TENSORSWITCH_GUI_PORT=XXXX` if needed

## What It Does

The GUI helps you convert between different scientific data formats:

### Supported Conversions
- **TIFF to Zarr3**: Convert TIFF image stacks to modern Zarr format
- **ND2 to Zarr3**: Convert Nikon ND2 microscopy files to Zarr format
- **IMS to Zarr3**: Convert Imaris IMS files to Zarr format
- **N5 to N5**: Re-chunk existing N5 formats
- **N5 Conversions**: Convert between N5 and Zarr2 format
- **Downsampling**: Create multi-resolution pyramids for large datasets

### Key Features
- **Simple Interface**: Just select input/output paths and click convert
- **Lab Path Helper**: Built-in suggestions for 131 HHMI lab storage paths
- **Project Billing**: Automatic project selection for cluster jobs
- **Progress Tracking**: Watch conversion progress in real-time
- **Local or Cluster**: Run jobs locally or submit to LSF cluster

## How to Use

1. **Select Input**: Choose your source data file or folder
2. **Select Output**: Choose where to save the converted data
3. **Pick Task**: Select the conversion type you need
4. **Configure**: Set any advanced options if needed
5. **Submit**: Click to start the conversion

The GUI will show progress and let you know when it's done.

## Current Status

**Production Ready** - The GUI is fully functional and ready to use.

- All 6 TensorSwitch conversion tasks supported
- Real-time progress monitoring for local jobs
- Full LSF cluster integration
- Custom chunk/shard configuration
- 131 HHMI lab paths integrated
- 126 project options for billing

## File Structure

```
  src/tensorswitch/gui/
  ├── app.py                    # Main GUI
  ├── launch_gui.py             # Server launcher
  ├── __init__.py               # Package init
  ├── README_GUI.md             # User guide
  └── lab_paths_system/         # Lab paths integration
      ├── __init__.py
      ├── lab_paths.py          # Core lab paths functionality
      ├── simple_path_selector.py  # Path selector widget
      ├── hierarchical_lab_paths.json
      ├── lab_paths_data.json
      └── Lab_and_project_file_share_path.xlsx
```

## Installation

Dependencies are managed in `pyproject.toml`:
- `panel` - Web GUI framework
- `param` - Parameter validation
- `sqlalchemy` - Job database

## Troubleshooting

**Can't connect to GUI:**
- Make sure the server is running (check terminal output)
- Verify you're using the correct hostname and port
- Try the alternative access URLs above

**GUI won't start:**
- Check dependencies: `pixi install`
- Make sure port 5000 is available
- Kill existing processes: `pkill -f "python.*gui"`

**Path validation errors:**
- Use absolute paths starting with `/nrs` or `/groups`
- Make sure directories exist and are accessible

## For Developers

The GUI uses a clean, modular architecture:
- **Main App**: `app.py` contains the complete working GUI
- **Components**: Modular pieces available in `components/` for future expansion
- **Lab Paths**: Complete HHMI integration in `lab_paths_system/`
- **Testing**: Small test datasets available in `test_data/`

The code is production-ready and well-documented for future development.