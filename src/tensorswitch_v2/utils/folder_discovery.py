"""
Folder-based dataset discovery and classification module.

Auto-detects and classifies datasets in a directory as 'image' or 'segmentation'
based on encoding, dtype, and filename keywords.

Supported formats:
- Neuroglancer Precomputed (info file)
- Zarr3 (zarr.json)
- Zarr2 (.zarray / .zgroup)
- N5 (attributes.json)
- TIFF, ND2, CZI, IMS, HDF5 (by extension, classified by filename)

Usage:
    from tensorswitch_v2.utils.folder_discovery import discover_datasets

    # Scan directory for datasets
    result = discover_datasets('/path/to/data/')

    # Result contains classified datasets
    print(result.image)        # DiscoveredDataset or None
    print(result.segmentation) # DiscoveredDataset or None
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# Encodings that indicate segmentation data (Precomputed-specific)
SEGMENTATION_ENCODINGS = {'compressed_segmentation'}

# Data types that suggest segmentation data (ID-based)
SEGMENTATION_DTYPES = {'uint64', 'int64', 'uint32', 'int32'}

# Keywords in filename/path that indicate segmentation/label data
SEGMENTATION_KEYWORDS = {'label', 'mask', 'seg', 'annotation', 'roi', 'binary', 'instance'}

# Supported file extensions for discovery
DISCOVERABLE_EXTENSIONS = {'.tif', '.tiff', '.nd2', '.czi', '.ims', '.h5', '.hdf5'}

# Extension to format name mapping
_FORMAT_MAP = {
    '.tif': 'tiff', '.tiff': 'tiff',
    '.nd2': 'nd2', '.czi': 'czi', '.ims': 'ims',
    '.h5': 'hdf5', '.hdf5': 'hdf5',
}


@dataclass
class DiscoveredDataset:
    """Information about a discovered dataset."""
    path: str
    name: str  # Directory or file name
    data_type: str  # 'image' or 'segmentation'
    dtype: str  # e.g., 'uint8', 'uint64', 'unknown'
    source_format: str = 'unknown'  # 'precomputed', 'zarr3', 'zarr2', 'n5', 'tiff', etc.
    encoding: str = 'unknown'  # Precomputed encoding, 'unknown' for others
    shape: List[int] = field(default_factory=list)
    resolution: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    num_scales: int = 1
    info: Dict = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of dataset discovery."""
    image: Optional[DiscoveredDataset] = None
    segmentation: Optional[DiscoveredDataset] = None
    all_images: List[DiscoveredDataset] = field(default_factory=list)
    all_segmentations: List[DiscoveredDataset] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation is not None

    @property
    def has_both(self) -> bool:
        return self.has_image and self.has_segmentation

    @property
    def has_multiple_images(self) -> bool:
        return len(self.all_images) > 1

    @property
    def has_multiple_segmentations(self) -> bool:
        return len(self.all_segmentations) > 1


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_dataset_generic(name: str, dtype: str, encoding: str = '') -> str:
    """
    Classify any dataset as 'image' or 'segmentation'.

    Priority chain:
      1. Precomputed encoding (compressed_segmentation → segmentation)
      2. Filename keywords (label, mask, seg, etc.)
      3. dtype (uint32, uint64 → segmentation)
      4. Default → image

    Args:
        name: Dataset filename or directory name
        dtype: Data type string (e.g., 'uint8', 'uint64', 'unknown')
        encoding: Precomputed encoding string (empty for non-Precomputed)

    Returns:
        'image' or 'segmentation'
    """
    # 1. Strong signal: Precomputed encoding
    if encoding and encoding.lower() in SEGMENTATION_ENCODINGS:
        return 'segmentation'

    # 2. Filename keywords
    name_lower = name.lower()
    for keyword in SEGMENTATION_KEYWORDS:
        if keyword in name_lower:
            return 'segmentation'

    # 3. dtype heuristic
    if dtype.lower() in SEGMENTATION_DTYPES:
        return 'segmentation'

    return 'image'


def classify_dataset(info: Dict) -> str:
    """
    Classify a Precomputed dataset as 'image' or 'segmentation' based on metadata.

    This is a thin wrapper around classify_dataset_generic() for backward
    compatibility with code that passes Precomputed info dicts.

    Args:
        info: Precomputed dataset info dict

    Returns:
        'image' or 'segmentation'
    """
    scales = info.get('scales', [])
    if not scales:
        return 'image'

    first_scale = scales[0]
    encoding = first_scale.get('encoding', '')
    dtype = info.get('data_type', '')

    return classify_dataset_generic('', dtype, encoding)


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

def is_neuroglancer_precomputed(path: str) -> bool:
    """
    Check if a path is a valid Neuroglancer Precomputed dataset.

    A valid precomputed dataset has an 'info' file with '@type' field
    containing 'neuroglancer' or has recognizable scale structure.
    """
    if not os.path.isdir(path):
        return False

    info_path = os.path.join(path, 'info')
    if not os.path.isfile(info_path):
        return False

    try:
        with open(info_path, 'r') as f:
            info = json.load(f)

        # Check for neuroglancer precomputed markers
        type_field = info.get('@type', '')
        if 'neuroglancer' in type_field.lower():
            return True

        # Also accept if it has scales array (common precomputed structure)
        if 'scales' in info and isinstance(info['scales'], list):
            return True

        return False
    except (json.JSONDecodeError, IOError):
        return False


def is_zarr_dataset(path: str) -> bool:
    """Check if path is a Zarr dataset (v2 or v3)."""
    if not os.path.isdir(path):
        return False
    # Zarr3: zarr.json at root or in s0/0
    if os.path.isfile(os.path.join(path, 'zarr.json')):
        return True
    # Zarr2: .zarray or .zgroup at root or in s0/0
    if os.path.isfile(os.path.join(path, '.zarray')) or os.path.isfile(os.path.join(path, '.zgroup')):
        return True
    for subdir in ['s0', '0']:
        sub = os.path.join(path, subdir)
        if os.path.isfile(os.path.join(sub, 'zarr.json')):
            return True
        if os.path.isfile(os.path.join(sub, '.zarray')):
            return True
    return False


def is_n5_dataset(path: str) -> bool:
    """Check if path is an N5 dataset."""
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, 'attributes.json'))


# ---------------------------------------------------------------------------
# Lightweight metadata readers (JSON only, no TensorStore)
# ---------------------------------------------------------------------------

def _read_precomputed_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Read Neuroglancer Precomputed info file and create DiscoveredDataset."""
    info_path = os.path.join(path, 'info')
    if not os.path.isfile(info_path):
        return None

    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    scales = info.get('scales', [])
    if not scales:
        return None

    first_scale = scales[0]
    dtype = info.get('data_type', 'unknown')
    encoding = first_scale.get('encoding', 'unknown')
    resolution = first_scale.get('resolution', [1, 1, 1])
    size = first_scale.get('size', [0, 0, 0])
    name = os.path.basename(path)

    data_type = classify_dataset_generic(name, dtype, encoding)

    return DiscoveredDataset(
        path=path,
        name=name,
        data_type=data_type,
        dtype=dtype,
        source_format='precomputed',
        encoding=encoding,
        shape=size,
        resolution=resolution,
        num_scales=len(scales),
        info=info,
    )


def _find_metadata_file(path: str, filename: str) -> Optional[str]:
    """Find a metadata file at root or in s0/0 subdirectory."""
    candidate = os.path.join(path, filename)
    if os.path.isfile(candidate):
        return candidate
    for subdir in ['s0', '0']:
        candidate = os.path.join(path, subdir, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_zarr3_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Read Zarr3 metadata from zarr.json without opening TensorStore."""
    meta_path = _find_metadata_file(path, 'zarr.json')
    if not meta_path:
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # zarr.json for arrays has node_type='array' and data_type/shape fields
    if meta.get('node_type') != 'array':
        # Could be a group zarr.json — check if s0/ has an array
        if meta_path == os.path.join(path, 'zarr.json'):
            # Try s0/zarr.json
            sub_path = _find_metadata_file(path, os.path.join('s0', 'zarr.json'))
            if sub_path:
                try:
                    with open(sub_path, 'r') as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, IOError):
                    return None
            else:
                return None

    dtype = str(meta.get('data_type', 'unknown'))
    shape = meta.get('shape', [])
    name = os.path.basename(path)
    data_type = classify_dataset_generic(name, dtype)

    # Count scales by checking s0, s1, s2, ... subdirectories
    num_scales = 1
    for i in range(1, 20):
        if os.path.isdir(os.path.join(path, f's{i}')):
            num_scales += 1
        else:
            break

    return DiscoveredDataset(
        path=path,
        name=name,
        data_type=data_type,
        dtype=dtype,
        source_format='zarr3',
        shape=shape,
        num_scales=num_scales,
    )


def _read_zarr2_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Read Zarr2 metadata from .zarray without opening TensorStore."""
    meta_path = _find_metadata_file(path, '.zarray')
    if not meta_path:
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    dtype = str(meta.get('dtype', 'unknown'))
    # Strip numpy dtype prefix (e.g., '<u2' -> 'uint16')
    dtype = _normalize_numpy_dtype(dtype)
    shape = meta.get('shape', [])
    name = os.path.basename(path)
    data_type = classify_dataset_generic(name, dtype)

    num_scales = 1
    for i in range(1, 20):
        if os.path.isdir(os.path.join(path, f's{i}')):
            num_scales += 1
        else:
            break

    return DiscoveredDataset(
        path=path,
        name=name,
        data_type=data_type,
        dtype=dtype,
        source_format='zarr2',
        shape=shape,
        num_scales=num_scales,
    )


def _read_n5_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Read N5 metadata from attributes.json without opening TensorStore."""
    attrs_path = os.path.join(path, 'attributes.json')
    if not os.path.isfile(attrs_path):
        return None

    try:
        with open(attrs_path, 'r') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # N5 attributes.json may be at the group level (no dataType) or dataset level
    dtype = str(meta.get('dataType', 'unknown'))
    dims = meta.get('dimensions', [])

    # If no dataType at root, try s0/attributes.json
    if dtype == 'unknown':
        s0_attrs = os.path.join(path, 's0', 'attributes.json')
        if os.path.isfile(s0_attrs):
            try:
                with open(s0_attrs, 'r') as f:
                    s0_meta = json.load(f)
                dtype = str(s0_meta.get('dataType', 'unknown'))
                dims = s0_meta.get('dimensions', dims)
            except (json.JSONDecodeError, IOError):
                pass

    name = os.path.basename(path)
    data_type = classify_dataset_generic(name, dtype)

    num_scales = 1
    for i in range(1, 20):
        if os.path.isdir(os.path.join(path, f's{i}')):
            num_scales += 1
        else:
            break

    return DiscoveredDataset(
        path=path,
        name=name,
        data_type=data_type,
        dtype=dtype,
        source_format='n5',
        shape=dims,
        num_scales=num_scales,
    )


def _read_file_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Create DiscoveredDataset for a supported file. Classification by name only."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in DISCOVERABLE_EXTENSIONS:
        return None

    source_format = _FORMAT_MAP.get(ext, 'unknown')
    name = os.path.basename(path)
    data_type = classify_dataset_generic(name, 'unknown')

    return DiscoveredDataset(
        path=path,
        name=name,
        data_type=data_type,
        dtype='unknown',
        source_format=source_format,
    )


def _normalize_numpy_dtype(dtype_str: str) -> str:
    """Convert numpy dtype string (e.g., '<u2') to readable name (e.g., 'uint16')."""
    mapping = {
        '|u1': 'uint8', '<u2': 'uint16', '<u4': 'uint32', '<u8': 'uint64',
        '|i1': 'int8', '<i2': 'int16', '<i4': 'int32', '<i8': 'int64',
        '<f4': 'float32', '<f8': 'float64',
        '>u2': 'uint16', '>u4': 'uint32', '>u8': 'uint64',
        '>i2': 'int16', '>i4': 'int32', '>i8': 'int64',
        '>f4': 'float32', '>f8': 'float64',
    }
    return mapping.get(dtype_str, dtype_str)


# ---------------------------------------------------------------------------
# Main discovery
# ---------------------------------------------------------------------------

def _add_dataset(result: DiscoveryResult, dataset: DiscoveredDataset, verbose: bool):
    """Add a discovered dataset to the result."""
    if verbose:
        print(f"  Found: {dataset.name} ({dataset.data_type}, {dataset.source_format}, {dataset.dtype})")
    if dataset.data_type == 'image':
        result.all_images.append(dataset)
    else:
        result.all_segmentations.append(dataset)


def discover_datasets(
    directory: str,
    verbose: bool = True
) -> DiscoveryResult:
    """
    Discover and classify datasets in a directory.

    Scans the given directory for subdirectories and files that are valid
    datasets, classifies each as 'image' or 'segmentation', and returns
    a DiscoveryResult.

    Supported formats:
    - Neuroglancer Precomputed directories (info file)
    - Zarr3 directories (zarr.json)
    - Zarr2 directories (.zarray / .zgroup)
    - N5 directories (attributes.json)
    - Supported files: .tif, .tiff, .nd2, .czi, .ims, .h5, .hdf5

    Args:
        directory: Path to directory to scan
        verbose: Print discovery progress

    Returns:
        DiscoveryResult with discovered datasets

    Example:
        >>> result = discover_datasets('/path/to/data/')
        >>> if result.has_both:
        ...     print(f"Image: {result.image.name}")
        ...     print(f"Segmentation: {result.segmentation.name}")
    """
    result = DiscoveryResult()

    if not os.path.isdir(directory):
        result.error = f"Directory does not exist: {directory}"
        return result

    # Check if directory itself is a single dataset
    dataset = _try_read_directory_dataset(directory)
    if dataset:
        _add_dataset(result, dataset, verbose)
        if verbose:
            print(f"Found single dataset: {dataset.name} ({dataset.data_type})")
        return result

    # Scan all entries in directory
    entries = sorted(os.listdir(directory))

    if verbose:
        print(f"Scanning {len(entries)} entries in {directory}...")

    for entry in entries:
        entry_path = os.path.join(directory, entry)

        if os.path.isdir(entry_path):
            dataset = _try_read_directory_dataset(entry_path)
            if dataset:
                _add_dataset(result, dataset, verbose)

        elif os.path.isfile(entry_path):
            dataset = _read_file_dataset(entry_path)
            if dataset:
                _add_dataset(result, dataset, verbose)

    # Set primary image/segmentation if exactly one of each
    if len(result.all_images) == 1:
        result.image = result.all_images[0]
    if len(result.all_segmentations) == 1:
        result.segmentation = result.all_segmentations[0]

    if verbose:
        print(f"Discovery complete: {len(result.all_images)} images, {len(result.all_segmentations)} segmentations")

    return result


def _try_read_directory_dataset(path: str) -> Optional[DiscoveredDataset]:
    """Try to read a directory as a dataset in priority order: Precomputed → Zarr → N5."""
    if is_neuroglancer_precomputed(path):
        return _read_precomputed_dataset(path)

    if is_zarr_dataset(path):
        return _read_zarr3_dataset(path) or _read_zarr2_dataset(path)

    if is_n5_dataset(path):
        return _read_n5_dataset(path)

    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_discovery_for_conversion(
    result: DiscoveryResult,
    image_only: bool = False,
    labels_only: bool = False
) -> Tuple[Optional[DiscoveredDataset], Optional[DiscoveredDataset], Optional[str]]:
    """
    Validate discovery result and return datasets for conversion.

    Applies the error-with-guidance policy when ambiguous datasets are found.

    Args:
        result: DiscoveryResult from discover_datasets()
        image_only: Only return image dataset
        labels_only: Only return segmentation dataset

    Returns:
        Tuple of (image_dataset, segmentation_dataset, error_message)
        error_message is None if validation succeeds
    """
    # Check for errors in discovery
    if result.error:
        return None, None, result.error

    # Handle --image-only
    if image_only:
        if result.has_multiple_images:
            msg = _format_multiple_datasets_error(result.all_images, 'image')
            return None, None, msg
        if not result.has_image:
            return None, None, "No image datasets found in directory."
        return result.image, None, None

    # Handle --labels-only
    if labels_only:
        if result.has_multiple_segmentations:
            msg = _format_multiple_datasets_error(result.all_segmentations, 'segmentation')
            return None, None, msg
        if not result.has_segmentation:
            return None, None, "No segmentation datasets found in directory."
        return None, result.segmentation, None

    # Default: convert both if available
    error_parts = []

    if result.has_multiple_images:
        error_parts.append(_format_multiple_datasets_error(result.all_images, 'image'))

    if result.has_multiple_segmentations:
        error_parts.append(_format_multiple_datasets_error(result.all_segmentations, 'segmentation'))

    if error_parts:
        return None, None, '\n\n'.join(error_parts)

    return result.image, result.segmentation, None


def _format_multiple_datasets_error(datasets: List[DiscoveredDataset], data_type: str) -> str:
    """Format error message for multiple datasets of same type."""
    lines = [f"Error: Found multiple {data_type} datasets:"]

    for ds in datasets:
        lines.append(f"  - {ds.name} ({ds.source_format}, {ds.dtype})")

    lines.append("")
    lines.append("To convert only one type:")
    lines.append(f"  python -m tensorswitch_v2 -i /path/to/dir/ -o output.zarr "
                 f"--{data_type.replace('segmentation', 'labels')}-only")

    return '\n'.join(lines)


def print_discovery_summary(result: DiscoveryResult) -> None:
    """Print a formatted summary of discovery results."""
    print("=" * 60)
    print("Dataset Discovery Summary")
    print("=" * 60)

    if result.error:
        print(f"Error: {result.error}")
        return

    print(f"Images found: {len(result.all_images)}")
    for ds in result.all_images:
        primary = " (primary)" if ds == result.image else ""
        print(f"  - {ds.name}{primary}")
        print(f"    format: {ds.source_format}, dtype: {ds.dtype}")
        if ds.shape:
            print(f"    shape: {ds.shape}")
        if ds.num_scales > 1:
            print(f"    scales: {ds.num_scales}")

    print(f"\nSegmentations found: {len(result.all_segmentations)}")
    for ds in result.all_segmentations:
        primary = " (primary)" if ds == result.segmentation else ""
        print(f"  - {ds.name}{primary}")
        print(f"    format: {ds.source_format}, dtype: {ds.dtype}")
        if ds.shape:
            print(f"    shape: {ds.shape}")
        if ds.num_scales > 1:
            print(f"    scales: {ds.num_scales}")

    print("=" * 60)
