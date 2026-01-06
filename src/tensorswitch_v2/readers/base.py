"""
Base reader class for TensorSwitch Phase 5 architecture.

All format-specific readers inherit from BaseReader and must implement
the abstract methods to convert their format into TensorStore arrays.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


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
        Return physical pixel/voxel dimensions in micrometers.

        Extracts the physical size of each voxel from format metadata.
        Critical for anisotropic downsampling and coordinate transformations.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' (and 't' if applicable):
                {
                    'x': 0.116,  # micrometers per pixel in X
                    'y': 0.116,  # micrometers per pixel in Y
                    'z': 0.5     # micrometers per pixel in Z
                }

        Example (isotropic):
            {'x': 0.1, 'y': 0.1, 'z': 0.1}

        Example (anisotropic - common in microscopy):
            {'x': 0.116, 'y': 0.116, 'z': 0.5}  # Z is coarser

        Notes:
            - Return 1.0 if physical size is unknown
            - Units MUST be micrometers (convert from other units)
            - Anisotropic voxels will trigger warnings and smart downsampling
        """
        pass

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
                            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                            {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
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
                    'unit': 'micrometer'
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
        ome_metadata = {
            'multiscales': [{
                'version': '0.5',
                'axes': axes,
                'datasets': [{
                    'path': '0',
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': scale
                    }]
                }],
                'name': f'Data from {self.__class__.__name__}'
            }]
        }

        return ome_metadata

    def __repr__(self) -> str:
        """String representation of reader."""
        return f"{self.__class__.__name__}(path='{self.path}')"
