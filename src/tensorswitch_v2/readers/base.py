"""
Base reader class for TensorSwitch Phase 5 architecture.

All format-specific readers inherit from BaseReader and must implement
the abstract methods to convert their format into TensorStore arrays.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import os
import json
import numpy as np


# ============================================================================
# Shared kvstore utilities for all Tier 1 readers
# ============================================================================

def build_kvstore(path: str) -> Dict:
    """
    Build TensorStore kvstore specification from a path.

    Handles local paths and remote URLs (HTTP, HTTPS, S3, GCS).
    This is used by all Tier 1 readers (N5, Zarr, Precomputed) to create
    proper kvstore specs for TensorStore.

    Args:
        path: Local path or remote URL
            - Local: /path/to/data or file:///path/to/data
            - GCS: gs://bucket/path
            - S3: s3://bucket/path
            - HTTP: http://example.com/data or https://...

    Returns:
        dict: TensorStore kvstore specification

    Examples:
        >>> build_kvstore('/local/path/data')
        {'driver': 'file', 'path': '/local/path/data'}

        >>> build_kvstore('gs://my-bucket/data')
        {'driver': 'gcs', 'bucket': 'my-bucket', 'path': 'data'}

        >>> build_kvstore('s3://my-bucket/data')
        {'driver': 's3', 'bucket': 'my-bucket', 'path': 'data'}

        >>> build_kvstore('https://example.com/data')
        {'driver': 'http', 'base_url': 'https://example.com/data'}
    """
    parsed = urlparse(path)
    scheme = parsed.scheme.lower()

    if scheme in ('http', 'https'):
        # HTTP/HTTPS URL
        return {'driver': 'http', 'base_url': path}

    elif scheme == 's3':
        # S3 URL: s3://bucket/path
        bucket = parsed.netloc
        s3_path = parsed.path.lstrip('/')
        return {'driver': 's3', 'bucket': bucket, 'path': s3_path}

    elif scheme == 'gs':
        # Google Cloud Storage URL: gs://bucket/path
        bucket = parsed.netloc
        gcs_path = parsed.path.lstrip('/')
        return {'driver': 'gcs', 'bucket': bucket, 'path': gcs_path}

    elif scheme == 'file':
        # Explicit file:// URL
        return {'driver': 'file', 'path': parsed.path}

    else:
        # Local file path (no scheme or unknown scheme)
        local_path = path if not scheme else parsed.path
        return {'driver': 'file', 'path': local_path}


def is_remote_path(path: str) -> bool:
    """Check if path is a remote URL (HTTP, S3, GCS)."""
    parsed = urlparse(path)
    return parsed.scheme.lower() in ('http', 'https', 's3', 'gs')


def is_local_precomputed(path: str) -> bool:
    """
    Check if path is a local Neuroglancer Precomputed directory.

    Detects precomputed format by checking for an 'info' file with
    the '@type': 'neuroglancer_multiscale_volume' field.

    Args:
        path: Path to check

    Returns:
        bool: True if path is a local precomputed directory

    Example:
        >>> is_local_precomputed('/data/image_230130b')
        True  # if contains valid info file
    """
    # Skip remote URLs
    if is_remote_path(path):
        return False

    # Check if directory exists and contains info file
    info_path = os.path.join(path, 'info')
    if not os.path.isdir(path) or not os.path.isfile(info_path):
        return False

    # Validate info file content
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        return info.get('@type') == 'neuroglancer_multiscale_volume'
    except (json.JSONDecodeError, IOError, KeyError):
        return False


class BaseReader(ABC):
    """
    Abstract base class for all format readers.

    Readers are the foundation of the Phase 5 architecture (Layer 1).
    Each reader converts a specific format (TIFF, N5, ND2, etc.) into
    a TensorStore spec/array, providing a unified interface for all formats.

    Architecture Layer: 1 (Foundation - No Dependencies)
    - Independent, can be used standalone without wrapper
    - Each reader converts its format → TensorStore spec/array

    Design Principles:
    - Format-specific implementation (one reader per format)
    - Returns TensorStore specs (virtual, lazy, zero-copy when possible)
    - Extracts format-specific metadata
    - No knowledge of writers or output formats

    Example Usage:
        >>> from tensorswitch_v2.readers import TiffReader
        >>> reader = TiffReader("/path/to/data.tif")
        >>> ts_spec = reader.get_tensorstore_spec()
        >>> metadata = reader.get_metadata()
        >>> voxel_sizes = reader.get_voxel_sizes()

    Subclass Requirements:
        Must implement all @abstractmethod methods.
        Optionally override get_ome_metadata() for custom OME-NGFF generation.
    """

    def __init__(self, path: str):
        """
        Initialize reader with data path.

        Args:
            path: Path to input data (local file path, HTTP URL, GCS/S3 URI)
        """
        self.path = path

    @abstractmethod
    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore specification for this data source.

        This is the core conversion method - converts format-specific data
        into a TensorStore spec (virtual or open).

        Returns:
            dict: TensorStore spec with keys like:
                - 'driver': TensorStore driver name (e.g., 'zarr3', 'n5', 'array')
                - 'kvstore': Key-value store specification
                - 'schema': Array schema (dtype, shape, dimension_names)
                - 'open': True if opening existing data

        Example (N5 - native TensorStore):
            {
                'driver': 'n5',
                'kvstore': {'driver': 'file', 'path': '/data.n5'},
                'open': True,
                'schema': {'dimension_names': ['z', 'y', 'x']}
            }

        Example (TIFF - wrapped Dask array):
            {
                'driver': 'array',
                'array': dask_array,  # Lazy Dask array
                'schema': {
                    'dtype': 'uint16',
                    'shape': [100, 1024, 1024],
                    'dimension_names': ['z', 'y', 'x']
                }
            }

        Notes:
            - Prefer virtual/lazy specs when possible (avoid loading data)
            - For non-TensorStore formats, wrap in Dask then use 'array' driver
            - Include dimension_names for dimension-aware processing
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """
        Return format-specific metadata.

        Extracts all available metadata from the source format.
        This is format-specific and not standardized.

        Returns:
            dict: Format-specific metadata containing:
                - Original format metadata (OME-XML, HDF5 attributes, etc.)
                - Any embedded metadata from the file
                - Format-specific properties

        Example (TIFF):
            {
                'ome_xml': '<OME>...</OME>',
                'imagej_metadata': {...},
                'tiff_tags': {...}
            }

        Example (N5):
            {
                'dimensions': [100, 1024, 1024],
                'blockSize': [32, 32, 32],
                'dataType': 'uint16',
                'compression': {...},
                'pixelResolution': {...}
            }

        Notes:
            - This is the "raw" metadata from the source format
            - For standardized OME-NGFF metadata, use get_ome_metadata()
        """
        pass

    @abstractmethod
    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return physical pixel/voxel dimensions in nanometers.

        Extracts the physical size of each voxel from format metadata.
        Critical for anisotropic downsampling and coordinate transformations.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' (and 't' if applicable):
                {
                    'x': 116.0,  # nanometers per pixel in X
                    'y': 116.0,  # nanometers per pixel in Y
                    'z': 500.0   # nanometers per pixel in Z
                }

        Example (isotropic):
            {'x': 100.0, 'y': 100.0, 'z': 100.0}

        Example (anisotropic - common in microscopy):
            {'x': 116.0, 'y': 116.0, 'z': 500.0}  # Z is coarser

        Notes:
            - Return 1.0 if physical size is unknown
            - Units MUST be nanometers (convert from other units)
            - Anisotropic voxels will trigger warnings and smart downsampling
        """
        pass

    def get_source_info(self) -> Dict:
        """
        Return comprehensive source metadata for auto-detection.

        Extracts all available properties from the source format including
        dimension names, memory order, chunk shape, voxel sizes, and more.
        This unified structure allows the converter to auto-configure output
        to match source settings unless explicitly overridden.

        Returns:
            dict: Unified source metadata structure:
                {
                    # Dimension info
                    'dimension_names': ['z', 'y', 'x'],  # Actual names from source
                    'shape': (498, 2000, 2000),
                    'dtype': 'uint16',

                    # Memory layout
                    'is_fortran_order': False,  # F-order vs C-order
                    'memory_order': 'C',        # 'C' or 'F'

                    # Chunking (native from source)
                    'chunk_shape': (64, 64, 64),  # Source's native chunks
                    'is_chunked': True,

                    # Physical metadata
                    'voxel_sizes': {'x': 0.16, 'y': 0.16, 'z': 0.4},
                    'voxel_unit': 'nanometer',

                    # Compression (if applicable)
                    'compression': 'zstd',
                    'compression_level': 5,

                    # Format-specific
                    'ome_xml': '...',        # Raw OME-XML if available
                    'source_format': 'nd2',  # Source format identifier
                }

        Notes:
            - Default implementation provides basic extraction
            - Subclasses should override to extract format-specific info
            - Missing fields will be None or have sensible defaults
        """
        # Get basic info from spec
        spec = self.get_tensorstore_spec()
        schema = spec.get('schema', {})

        shape = tuple(schema.get('shape', []))
        dtype = schema.get('dtype', 'unknown')
        dimension_names = schema.get('dimension_names', [])

        # Get voxel sizes
        try:
            voxel_sizes = self.get_voxel_sizes()
        except Exception:
            voxel_sizes = {'x': 1.0, 'y': 1.0, 'z': 1.0}

        # Get raw metadata
        try:
            raw_metadata = self.get_metadata()
        except Exception:
            raw_metadata = {}

        # Extract OME-XML if available
        ome_xml = raw_metadata.get('ome_xml') or raw_metadata.get('raw_xml')

        # Auto-detect suggested downsample method based on filename and dtype
        suggested_downsample_method = self._detect_downsample_method(dtype)

        # Default source info - subclasses should override for format-specific extraction
        return {
            # Dimension info
            'dimension_names': dimension_names if dimension_names else None,
            'shape': shape,
            'dtype': dtype,

            # Memory layout (default to C-order, subclasses should detect)
            'is_fortran_order': False,
            'memory_order': 'C',

            # Chunking (default unknown, subclasses should extract)
            'chunk_shape': None,
            'is_chunked': None,

            # Physical metadata
            'voxel_sizes': voxel_sizes,
            'voxel_unit': 'nanometer',

            # Compression (default unknown)
            'compression': None,
            'compression_level': None,

            # Format-specific
            'ome_xml': ome_xml,
            'source_format': self.__class__.__name__.replace('Reader', '').lower(),
            'raw_metadata': raw_metadata,

            # Downsampling hint
            'suggested_downsample_method': suggested_downsample_method,
        }

    def get_ome_metadata(self) -> Dict:
        """
        Return OME-NGFF compatible metadata (v0.4 or v0.5).

        Converts format-specific metadata into standardized OME-NGFF format.
        Default implementation calls _convert_to_ome_ngff(), which can be
        overridden by subclasses for custom conversion logic.

        Returns:
            dict: OME-NGFF metadata structure:
                {
                    'multiscales': [{
                        'axes': [
                            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                            {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
                        ],
                        'datasets': [{
                            'path': '0',
                            'coordinateTransformations': [{
                                'type': 'scale',
                                'scale': [0.5, 0.116, 0.116]
                            }]
                        }]
                    }]
                }

        Notes:
            - Default implementation provides basic OME-NGFF structure
            - Override for format-specific OME-NGFF generation
            - Used by TensorSwitchDataset.get_ome_ngff_metadata()
        """
        return self._convert_to_ome_ngff(self.get_metadata(), self.get_voxel_sizes())

    def supports_remote(self) -> bool:
        """
        Check if this reader supports remote data (HTTP, GCS, S3).

        Returns:
            bool: True if reader can handle remote URLs, False otherwise

        Example:
            >>> reader = N5Reader("http://example.com/data.n5")
            >>> reader.supports_remote()
            True

            >>> reader = TiffReader("data.tif")
            >>> reader.supports_remote()
            False

        Notes:
            - N5, Zarr2, Zarr3, Precomputed support remote (TensorStore native)
            - TIFF, ND2, IMS typically don't support remote efficiently
            - BIOIO support depends on underlying plugin
        """
        return False

    def _convert_to_ome_ngff(self, metadata: Dict, voxel_sizes: Dict[str, float]) -> Dict:
        """
        Helper method to convert format-specific metadata to OME-NGFF.

        Subclasses can override this for custom OME-NGFF generation.
        Default implementation provides basic structure.

        Args:
            metadata: Format-specific metadata from get_metadata()
            voxel_sizes: Voxel dimensions from get_voxel_sizes()

        Returns:
            dict: Basic OME-NGFF metadata structure
        """
        # Get array shape from TensorStore spec
        spec = self.get_tensorstore_spec()
        shape = spec.get('schema', {}).get('shape', [])
        dimension_names = spec.get('schema', {}).get('dimension_names', [])

        # Infer dimension names if not provided
        if not dimension_names:
            ndim = len(shape)
            if ndim == 3:
                dimension_names = ['z', 'y', 'x']
            elif ndim == 4:
                dimension_names = ['c', 'z', 'y', 'x']
            elif ndim == 5:
                dimension_names = ['t', 'c', 'z', 'y', 'x']
            else:
                dimension_names = [f'dim_{i}' for i in range(ndim)]

        # Build OME-NGFF axes
        axes = []
        for dim_name in dimension_names:
            if dim_name in ['z', 'y', 'x']:
                axes.append({
                    'name': dim_name,
                    'type': 'space',
                    'unit': 'nanometer'
                })
            elif dim_name == 't':
                axes.append({
                    'name': 't',
                    'type': 'time',
                    'unit': 'second'
                })
            elif dim_name == 'c':
                axes.append({
                    'name': 'c',
                    'type': 'channel'
                })
            else:
                axes.append({'name': dim_name})

        # Build scale transformation from voxel sizes
        scale = []
        for dim_name in dimension_names:
            if dim_name in voxel_sizes:
                scale.append(voxel_sizes[dim_name])
            else:
                scale.append(1.0)  # Default scale

        # Basic OME-NGFF structure
        # Note: "version" belongs at the "ome" level only, NOT inside multiscales
        # Use actual filename for name instead of generic reader class name
        import os
        dataset_name = os.path.splitext(os.path.basename(self.path))[0]

        ome_metadata = {
            'multiscales': [{
                'axes': axes,
                'datasets': [{
                    'path': 's0',
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': scale
                    }]
                }],
                'name': dataset_name
            }]
        }

        return ome_metadata

    def _detect_downsample_method(self, dtype: str) -> str:
        """
        Auto-detect the best downsampling method based on filename and data type.

        Uses heuristics to determine if data is likely segmentation/labels (use 'mode')
        or intensity data (use 'mean').

        Args:
            dtype: Data type string (e.g., 'uint16', 'float32')

        Returns:
            str: Suggested downsample method ('mean' or 'mode')

        Heuristics:
            1. Filename/path contains label-related keywords → 'mode'
            2. Otherwise → 'mean' (default for intensity images)

        Note:
            - 'mean' is appropriate for most microscopy data (fluorescence, brightfield)
            - 'mode' preserves discrete values, best for segmentation masks and labels
        """
        import os

        # Keywords that suggest segmentation/label data
        label_keywords = ['label', 'mask', 'seg', 'annotation', 'roi', 'binary', 'instance']

        # Check multiple levels of the path (handles /data/labels/dataset.zarr/s0)
        # Normalize and split path into components
        path_parts = self.path.lower().replace('\\', '/').split('/')
        # Check last 4 components (covers most directory structures)
        check_parts = ' '.join(path_parts[-4:]) if len(path_parts) >= 4 else ' '.join(path_parts)

        for keyword in label_keywords:
            if keyword in check_parts:
                return 'mode'

        # Default to 'mean' for intensity images (most common case)
        return 'mean'

    def __repr__(self) -> str:
        """String representation of reader."""
        return f"{self.__class__.__name__}(path='{self.path}')"
