"""
Input file format detection and metadata extraction for TensorSwitch
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class FormatDetector:
    """Detects file formats and extracts metadata from input files"""

    def __init__(self):
        self.supported_formats = {
            'tiff': ['.tif', '.tiff'],
            'nd2': ['.nd2'],
            'ims': ['.ims'],
            'n5': [],  # directory format
            'zarr2': [],  # directory format
            'zarr3': []  # directory format
        }

    def analyze_input(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze input file/directory and extract all metadata

        Returns:
            Dictionary with format, shape, size, dimensions, etc.
        """
        result = {
            'path': file_path,
            'exists': False,
            'format': None,
            'shape': None,
            'dimensions': None,
            'dtype': None,
            'size_mb': 0,
            'num_channels': None,
            'num_timepoints': None,
            'num_z_slices': None,
            'pixel_size': None,
            'metadata': {},
            'error': None
        }

        try:
            if not file_path or not os.path.exists(file_path):
                result['error'] = "File does not exist"
                return result

            result['exists'] = True
            result['size_mb'] = self._get_size_mb(file_path)

            # Detect format
            format_type = self._detect_format(file_path)
            result['format'] = format_type

            if not format_type:
                result['error'] = "Unsupported file format"
                return result

            # Extract metadata based on format
            if format_type == 'tiff':
                self._extract_tiff_metadata(file_path, result)
            elif format_type == 'nd2':
                self._extract_nd2_metadata(file_path, result)
            elif format_type == 'ims':
                self._extract_ims_metadata(file_path, result)
            elif format_type in ['zarr2', 'zarr3']:
                self._extract_zarr_metadata(file_path, result)
            elif format_type == 'n5':
                self._extract_n5_metadata(file_path, result)

        except Exception as e:
            result['error'] = f"Analysis failed: {str(e)}"

        return result

    def _detect_format(self, path: str) -> Optional[str]:
        """Detect file format from path"""
        if os.path.isfile(path):
            ext = Path(path).suffix.lower()
            for format_name, extensions in self.supported_formats.items():
                if ext in extensions:
                    return format_name
        elif os.path.isdir(path):
            return self._detect_directory_format(path)
        return None

    def _detect_directory_format(self, dir_path: str) -> Optional[str]:
        """Detect format of directory (N5 or Zarr)"""
        path = Path(dir_path)

        # Check for N5
        if (path / 'attributes.json').exists():
            return 'n5'

        # Check for Zarr v3
        if (path / 'zarr.json').exists():
            return 'zarr3'

        # Check for Zarr v2/v3
        if (path / '.zattrs').exists() or (path / '.zgroup').exists():
            return self._detect_zarr_version(path)

        # Look for Zarr arrays in subdirectories
        for item in path.iterdir():
            if item.is_dir() and (item / '.zarray').exists():
                return self._detect_zarr_version(path)

        return None

    def _detect_zarr_version(self, zarr_path: Path) -> str:
        """Determine if Zarr is v2 or v3"""
        # v3 indicators
        if (zarr_path / 'zarr.json').exists():
            return 'zarr3'

        # Look for sharding (v3 feature)
        if any(zarr_path.rglob('*.shard')):
            return 'zarr3'

        # Check .zarray files
        for zarray in zarr_path.rglob('.zarray'):
            try:
                with open(zarray) as f:
                    data = json.load(f)
                    if data.get('zarr_format') == 3 or 'codecs' in data:
                        return 'zarr3'
            except:
                pass

        return 'zarr2'

    def _extract_tiff_metadata(self, file_path: str, result: Dict):
        """Extract metadata from TIFF files"""
        try:
            # Try to use tifffile if available
            try:
                import tifffile
                with tifffile.TiffFile(file_path) as tif:
                    result['shape'] = tif.series[0].shape
                    result['dtype'] = str(tif.series[0].dtype)
                    result['dimensions'] = len(tif.series[0].shape)

                    # Extract dimension info if available
                    if len(tif.series[0].shape) >= 2:
                        shape = tif.series[0].shape
                        if len(shape) == 2:  # Y, X
                            result['num_z_slices'] = 1
                        elif len(shape) == 3:  # Z, Y, X or C, Y, X
                            result['num_z_slices'] = shape[0]
                        elif len(shape) == 4:  # C, Z, Y, X or T, Z, Y, X
                            result['num_channels'] = shape[0]
                            result['num_z_slices'] = shape[1]
                        elif len(shape) == 5:  # T, C, Z, Y, X
                            result['num_timepoints'] = shape[0]
                            result['num_channels'] = shape[1]
                            result['num_z_slices'] = shape[2]

                    # Get pixel size if available
                    if hasattr(tif, 'pages') and tif.pages:
                        page = tif.pages[0]
                        if hasattr(page, 'tags'):
                            x_res = page.tags.get('XResolution')
                            if x_res and x_res.value:
                                result['pixel_size'] = f"{1/x_res.value[0]:.4f} units/pixel"

            except ImportError:
                # Fallback: basic file info only
                result['error'] = "tifffile not available - limited metadata"

        except Exception as e:
            result['error'] = f"TIFF analysis failed: {str(e)}"

    def _extract_nd2_metadata(self, file_path: str, result: Dict):
        """Extract metadata from ND2 files"""
        try:
            # Try to use nd2 library if available
            try:
                import nd2
                with nd2.ND2File(file_path) as f:
                    result['shape'] = f.shape
                    result['dtype'] = str(f.dtype)
                    result['dimensions'] = len(f.shape)

                    # Parse dimension info from shape (safer than accessing sizes)
                    if result['shape']:
                        self._parse_shape_dimensions(result['shape'], result)

                    # Skip advanced metadata for now - focus on basic info that always works

            except ImportError:
                result['error'] = "nd2 library not available - limited metadata"

        except Exception as e:
            result['error'] = f"ND2 analysis failed: {str(e)}"

    def _extract_ims_metadata(self, file_path: str, result: Dict):
        """Extract metadata from IMS files"""
        try:
            # IMS files are HDF5 format
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    # Look for dataset in standard IMS structure
                    if 'DataSet' in f:
                        dataset_keys = list(f['DataSet'].keys())
                        if dataset_keys:
                            ds_path = f'DataSet/{dataset_keys[0]}/ResolutionLevel/0/TimePoint/0/Channel/0/Data'
                            if ds_path in f:
                                dataset = f[ds_path]
                                result['shape'] = dataset.shape
                                result['dtype'] = str(dataset.dtype)
                                result['dimensions'] = len(dataset.shape)

            except ImportError:
                result['error'] = "h5py not available - limited metadata"

        except Exception as e:
            result['error'] = f"IMS analysis failed: {str(e)}"

    def _extract_zarr_metadata(self, dir_path: str, result: Dict):
        """Extract metadata from Zarr directories"""
        try:
            import zarr
            store = zarr.DirectoryStore(dir_path)
            root = zarr.group(store=store)

            # Look for main array (often at root level or in subdirectory)
            arrays = self._find_zarr_arrays(root)
            if arrays:
                main_array = arrays[0]  # Use first/largest array
                result['shape'] = main_array.shape
                result['dtype'] = str(main_array.dtype)
                result['dimensions'] = len(main_array.shape)

                # Extract dimension info from shape
                self._parse_shape_dimensions(main_array.shape, result)

        except Exception as e:
            result['error'] = f"Zarr analysis failed: {str(e)}"

    def _extract_n5_metadata(self, dir_path: str, result: Dict):
        """Extract metadata from N5 directories"""
        try:
            # Read attributes.json
            attrs_path = Path(dir_path) / 'attributes.json'
            if attrs_path.exists():
                with open(attrs_path) as f:
                    attrs = json.load(f)
                    if 'dimensions' in attrs:
                        result['shape'] = tuple(attrs['dimensions'])
                        result['dimensions'] = len(attrs['dimensions'])
                    if 'dataType' in attrs:
                        result['dtype'] = attrs['dataType']

        except Exception as e:
            result['error'] = f"N5 analysis failed: {str(e)}"

    def _find_zarr_arrays(self, group) -> List:
        """Find arrays in Zarr group"""
        arrays = []
        try:
            for key in group.keys():
                item = group[key]
                if hasattr(item, 'shape'):  # It's an array
                    arrays.append(item)
                elif hasattr(item, 'keys'):  # It's a group
                    arrays.extend(self._find_zarr_arrays(item))
        except:
            pass
        return arrays

    def _parse_shape_dimensions(self, shape: Tuple, result: Dict):
        """Parse shape to extract dimension info"""
        if not shape:
            return

        # Common patterns for microscopy data
        if len(shape) == 2:  # Y, X
            result['num_z_slices'] = 1
        elif len(shape) == 3:  # Z, Y, X
            result['num_z_slices'] = shape[0]
        elif len(shape) == 4:  # C, Z, Y, X
            result['num_channels'] = shape[0]
            result['num_z_slices'] = shape[1]
        elif len(shape) == 5:  # T, C, Z, Y, X
            result['num_timepoints'] = shape[0]
            result['num_channels'] = shape[1]
            result['num_z_slices'] = shape[2]

    def _get_size_mb(self, path: str) -> float:
        """Get file/directory size in MB"""
        try:
            if os.path.isfile(path):
                size_bytes = os.path.getsize(path)
            else:
                size_bytes = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, dirs, files in os.walk(path)
                    for file in files
                )
            return round(size_bytes / (1024 * 1024), 1)
        except:
            return 0

    def format_summary(self, analysis: Dict) -> str:
        """Format analysis results into readable summary with bullet points"""
        if analysis.get('error'):
            return f"❌ **Error**: {analysis['error']}"

        if not analysis.get('format'):
            return "❓ **Format**: Unknown"

        lines = []
        lines.append(f"• **Format**: {analysis['format'].upper()}")
        lines.append(f"• **Size**: {analysis['size_mb']} MB")

        if analysis.get('shape'):
            lines.append(f"• **Shape**: {analysis['shape']}")
            # Add dimension order information
            dimension_order = self._infer_dimension_order(analysis['shape'])
            if dimension_order:
                lines.append(f"• **Dimension Order**: {' × '.join(dimension_order)}")

        if analysis.get('dtype'):
            lines.append(f"• **Data Type**: {analysis['dtype']}")

        if analysis.get('num_channels'):
            lines.append(f"• **Channels**: {analysis['num_channels']}")

        if analysis.get('num_timepoints'):
            lines.append(f"• **Timepoints**: {analysis['num_timepoints']}")

        if analysis.get('num_z_slices'):
            lines.append(f"• **Z Slices**: {analysis['num_z_slices']}")

        if analysis.get('pixel_size'):
            lines.append(f"• **Pixel Size**: {analysis['pixel_size']}")

        return "  \n".join(lines)  # Use markdown line breaks (two spaces + newline)

    def _infer_dimension_order(self, shape: Tuple) -> List[str]:
        """Infer dimension order based on shape using same logic as utils.py"""
        if not shape:
            return []

        # Use same logic as zarr3_store_spec in utils.py lines 143-155
        if len(shape) == 3:
            # For 3D, assume channels if first dimension is small, otherwise Z
            if shape[0] <= 10:
                return ["C", "Y", "X"]
            else:
                return ["Z", "Y", "X"]
        elif len(shape) == 4:
            return ["C", "Z", "Y", "X"]
        elif len(shape) == 5:
            return ["T", "C", "Z", "Y", "X"]
        elif len(shape) == 2:
            return ["Y", "X"]
        else:
            return [f"dim_{i}" for i in range(len(shape))]