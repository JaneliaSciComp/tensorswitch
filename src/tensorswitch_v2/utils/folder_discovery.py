"""
Folder-based dataset discovery and classification module.

Auto-detects and classifies datasets in a directory as 'image' or 'segmentation'
based on encoding and dtype. Currently supports Neuroglancer Precomputed format.

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


# Encodings that indicate segmentation data
SEGMENTATION_ENCODINGS = {'compressed_segmentation'}

# Data types that suggest segmentation data (ID-based)
SEGMENTATION_DTYPES = {'uint64', 'int64', 'uint32', 'int32'}


@dataclass
class DiscoveredDataset:
    """Information about a discovered dataset."""
    path: str
    name: str  # Directory name
    data_type: str  # 'image' or 'segmentation'
    dtype: str  # e.g., 'uint8', 'uint64'
    encoding: str  # e.g., 'jpeg', 'raw', 'compressed_segmentation'
    shape: List[int]  # XYZ shape at highest resolution
    resolution: List[float]  # Voxel sizes in nanometers [x, y, z]
    num_scales: int  # Number of resolution levels
    info: Dict  # Full info/metadata content


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


def is_neuroglancer_precomputed(path: str) -> bool:
    """
    Check if a path is a valid Neuroglancer Precomputed dataset.

    A valid precomputed dataset has an 'info' file with '@type' field
    containing 'neuroglancer' or has recognizable scale structure.

    Args:
        path: Path to check

    Returns:
        bool: True if path is a valid precomputed dataset
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


def classify_dataset(info: Dict) -> str:
    """
    Classify a dataset as 'image' or 'segmentation' based on metadata.

    Classification logic:
    1. Strong signal: encoding='compressed_segmentation' -> segmentation
    2. Secondary signal: dtype in [uint64, int64, uint32, int32] -> segmentation
    3. Default: image

    Args:
        info: Dataset info/metadata as dict

    Returns:
        str: 'image' or 'segmentation'
    """
    # Check scales for encoding and dtype
    scales = info.get('scales', [])
    if not scales:
        return 'image'  # Can't determine, default to image

    # Get first scale (highest resolution) for classification
    first_scale = scales[0]
    encoding = first_scale.get('encoding', '')
    dtype = info.get('data_type', '')

    # Strong signal: compressed_segmentation encoding
    if encoding.lower() in SEGMENTATION_ENCODINGS:
        return 'segmentation'

    # Secondary signal: integer dtype typically used for IDs
    if dtype.lower() in SEGMENTATION_DTYPES:
        return 'segmentation'

    return 'image'


def _read_precomputed_dataset(path: str) -> Optional[DiscoveredDataset]:
    """
    Read Neuroglancer Precomputed info file and create DiscoveredDataset.

    Args:
        path: Path to precomputed dataset directory

    Returns:
        DiscoveredDataset or None if reading fails
    """
    info_path = os.path.join(path, 'info')
    if not os.path.isfile(info_path):
        return None

    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # Extract basic info
    scales = info.get('scales', [])
    if not scales:
        return None

    first_scale = scales[0]
    dtype = info.get('data_type', 'unknown')
    encoding = first_scale.get('encoding', 'unknown')
    resolution = first_scale.get('resolution', [1, 1, 1])
    size = first_scale.get('size', [0, 0, 0])

    # Classify
    data_type = classify_dataset(info)

    return DiscoveredDataset(
        path=path,
        name=os.path.basename(path),
        data_type=data_type,
        dtype=dtype,
        encoding=encoding,
        shape=size,
        resolution=resolution,
        num_scales=len(scales),
        info=info
    )


def discover_datasets(
    directory: str,
    verbose: bool = True
) -> DiscoveryResult:
    """
    Discover and classify datasets in a directory.

    Scans the given directory for subdirectories that are valid datasets
    (currently supports Neuroglancer Precomputed format), classifies each
    as 'image' or 'segmentation', and returns a DiscoveryResult.

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

    # Check if directory itself is a dataset
    if is_neuroglancer_precomputed(directory):
        dataset = _read_precomputed_dataset(directory)
        if dataset:
            if dataset.data_type == 'image':
                result.image = dataset
                result.all_images.append(dataset)
            else:
                result.segmentation = dataset
                result.all_segmentations.append(dataset)
            if verbose:
                print(f"Found single dataset: {dataset.name} ({dataset.data_type})")
            return result

    # Scan subdirectories
    subdirs = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])

    if verbose:
        print(f"Scanning {len(subdirs)} subdirectories in {directory}...")

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)

        # Try Neuroglancer Precomputed format
        if is_neuroglancer_precomputed(subdir_path):
            dataset = _read_precomputed_dataset(subdir_path)
            if dataset:
                if verbose:
                    print(f"  Found: {dataset.name} ({dataset.data_type}, {dataset.dtype}, {dataset.encoding})")

                if dataset.data_type == 'image':
                    result.all_images.append(dataset)
                else:
                    result.all_segmentations.append(dataset)

    # Set primary image/segmentation if exactly one of each
    if len(result.all_images) == 1:
        result.image = result.all_images[0]
    if len(result.all_segmentations) == 1:
        result.segmentation = result.all_segmentations[0]

    if verbose:
        print(f"Discovery complete: {len(result.all_images)} images, {len(result.all_segmentations)} segmentations")

    return result


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

    Example:
        >>> result = discover_datasets('/path/to/data/')
        >>> image, seg, error = validate_discovery_for_conversion(result)
        >>> if error:
        ...     print(error)
        ...     sys.exit(1)
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

    # Check for multiple images
    if result.has_multiple_images:
        error_parts.append(_format_multiple_datasets_error(result.all_images, 'image'))

    # Check for multiple segmentations
    if result.has_multiple_segmentations:
        error_parts.append(_format_multiple_datasets_error(result.all_segmentations, 'segmentation'))

    if error_parts:
        return None, None, '\n\n'.join(error_parts)

    # No errors - return whatever we found
    return result.image, result.segmentation, None


def _format_multiple_datasets_error(datasets: List[DiscoveredDataset], data_type: str) -> str:
    """Format error message for multiple datasets of same type."""
    lines = [f"Error: Found multiple {data_type} datasets:"]

    for ds in datasets:
        lines.append(f"  - {ds.name}/ ({ds.dtype}, {ds.encoding})")

    lines.append("")
    lines.append("Please specify explicitly:")

    if data_type == 'image':
        lines.append(f"  tensorswitch convert output.zarr --image /path/to/{datasets[0].name}")
    else:
        lines.append(f"  tensorswitch convert output.zarr --labels /path/to/{datasets[0].name}")

    lines.append("")
    lines.append("Or to convert only one type:")
    lines.append(f"  tensorswitch convert /path/to/dir/ output.zarr --{data_type.replace('segmentation', 'labels')}-only")

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
        print(f"    dtype: {ds.dtype}, encoding: {ds.encoding}")
        print(f"    shape: {ds.shape}, resolution: {ds.resolution} nm")
        print(f"    scales: {ds.num_scales}")

    print(f"\nSegmentations found: {len(result.all_segmentations)}")
    for ds in result.all_segmentations:
        primary = " (primary)" if ds == result.segmentation else ""
        print(f"  - {ds.name}{primary}")
        print(f"    dtype: {ds.dtype}, encoding: {ds.encoding}")
        print(f"    shape: {ds.shape}, resolution: {ds.resolution} nm")
        print(f"    scales: {ds.num_scales}")

    print("=" * 60)
