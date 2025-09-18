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

### Supported Conversions (10 Tasks)
- **TIFF to Zarr3/Zarr2**: Convert TIFF image stacks to modern Zarr format
- **ND2 to Zarr3/Zarr2**: Convert Nikon ND2 microscopy files to Zarr format
- **IMS to Zarr3/Zarr2**: Convert Imaris IMS files to Zarr format
- **N5 to N5**: Re-chunk existing N5 formats
- **N5 to Zarr2**: Convert N5 to Zarr2 format
- **Downsampling**: Create multi-resolution pyramids for Zarr2 and Zarr3 datasets

### Key Features
- **Smart Workflow Mode**: Auto-detect input formats and intelligently plan conversions
- **Manual Mode**: Traditional task selection for advanced users
- **Lab Path Helper**: Built-in suggestions for 131 HHMI lab storage paths
- **Project Billing**: Automatic project selection for cluster jobs (126 projects)
- **Dask JobQueue**: Advanced cluster scheduling with hybrid Dask-LSF execution
- **Progress Tracking**: Watch conversion progress in real-time
- **Local or Cluster**: Run jobs locally or submit to LSF cluster

## How to Use

### Smart Workflow (Recommended)
1. **Enter Input Path**: The system auto-detects your file format and shows metadata
2. **Choose Output Format**: Select Zarr2, Zarr3, or N5 as desired output
3. **Configure Downsampling**: Set pyramid levels (0-5) if needed
4. **Review Plan**: See the execution plan with estimated steps
5. **Execute**: Click to start the conversion

### Manual Workflow (Advanced)
1. **Select Input/Output**: Choose your source and destination paths
2. **Pick Task**: Manually select the specific conversion task
3. **Configure**: Set advanced options and parameters
4. **Submit**: Click to start the conversion

The GUI shows real-time progress and completion status for both workflows.

## Current Status

**Production Ready** - The GUI is fully functional and ready to use.

- All 10 TensorSwitch conversion tasks supported
- Real-time progress monitoring for local jobs
- Full LSF cluster integration
- Custom chunk/shard configuration
- 131 HHMI lab paths integrated
- 126 project options for billing

## File Structure

```
  src/tensorswitch/gui/
  ├── app.py                    # Main GUI application (production-ready)
  ├── launch_gui.py             # GUI server launcher
  ├── __init__.py               # Package initialization
  ├── README_GUI.md             # User documentation
  ├── format_detection/         # Smart workflow system
  │   ├── __init__.py
  │   ├── format_detector.py    # Auto-detect file formats and extract metadata
  │   └── task_planner.py       # Intelligent conversion planning
  └── lab_paths_system/         # HHMI lab integration
      ├── __init__.py
      ├── lab_paths.py          # Lab path management (131 labs)
      ├── path_selector.py      # Path selection UI components
      ├── hierarchical_lab_paths.json  # Structured lab directory data
      ├── lab_paths_data.json   # Lab path database
      └── Lab_and_project_file_share_path.xlsx  # Source Excel data (126 projects)
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