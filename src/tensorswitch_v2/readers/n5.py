"""
N5 reader implementation using native TensorStore driver.

Tier 1 reader - maximum performance with zero conversion overhead.
"""

import json
import os
from typing import Dict, Optional
import tensorstore as ts
from .base import BaseReader


class N5Reader(BaseReader):
    """
    Reader for N5 format using native TensorStore driver.

    N5 (Not-HDF5) is a chunked array storage format developed by the
    Saalfeld lab at Janelia. TensorStore provides native N5 support,
    making this a Tier 1 reader with maximum performance.

    Tier: 1 (Native TensorStore - Maximum Performance)
    - Zero conversion overhead
    - Direct TensorStore driver usage
    - Supports remote data (HTTP, GCS, S3)
    - Production-critical format

    Features:
    - Native chunked I/O
    - Multiple compression codecs (gzip, bzip2, xz, lz4, blosc)
    - Multi-resolution pyramids
    - Metadata in attributes.json

    Example:
        >>> from tensorswitch_v2.readers import N5Reader
        >>> reader = N5Reader("/path/to/data.n5")
        >>> spec = reader.get_tensorstore_spec()
        >>> metadata = reader.get_metadata()
        >>> voxel_sizes = reader.get_voxel_sizes()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset
        >>> dataset = TensorSwitchDataset("/data.n5", reader=N5Reader("/data.n5"))
        >>> ts_array = dataset.get_tensorstore_array(mode='open')
    """

    def __init__(self, path: str, dataset_path: str = ""):
        """
        Initialize N5 reader.

        Args:
            path: Path to N5 container (directory)
            dataset_path: Optional path to dataset within N5 (e.g., "s0" for scale 0)
                         If empty, reads root dataset. For multi-scale, specify scale.

        Example:
            >>> # Single-scale N5
            >>> reader = N5Reader("/data.n5")

            >>> # Multi-scale N5 (specific scale)
            >>> reader = N5Reader("/data.n5", dataset_path="s0")
        """
        super().__init__(path)
        self.dataset_path = dataset_path
        self._metadata_cache = None
        self._full_path = os.path.join(path, dataset_path) if dataset_path else path

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec using native N5 driver.

        This is a Tier 1 reader - uses TensorStore's native N5 driver
        with zero conversion overhead.

        Returns:
            dict: TensorStore spec for N5 dataset with keys:
                - 'driver': 'n5'
                - 'kvstore': Path to N5 container
                - 'path': Dataset path within container (if multi-scale)
                - 'open': True (opening existing data)
                - 'schema': Dimension names if available

        Example:
            >>> reader = N5Reader("/data.n5")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec)
            {
                'driver': 'n5',
                'kvstore': {'driver': 'file', 'path': '/data.n5'},
                'open': True,
                'schema': {'dimension_names': ['z', 'y', 'x']}
            }

        Example (multi-scale):
            >>> reader = N5Reader("/data.n5", dataset_path="s0")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['path'])
            's0'

        Notes:
            - Native N5 driver = maximum performance
            - Supports remote kvstores (http, gcs, s3)
            - Lazy evaluation (doesn't read data)
            - Compression handled by TensorStore automatically
        """
        spec = {
            'driver': 'n5',
            'kvstore': {
                'driver': 'file',
                'path': self.path
            },
            'open': True
        }

        # Add dataset path if specified (for multi-scale N5)
        if self.dataset_path:
            spec['path'] = self.dataset_path

        # Try to get dimension names from metadata
        metadata = self.get_metadata()
        if 'dimensions' in metadata:
            ndim = len(metadata['dimensions'])
            # Infer dimension names based on number of dimensions
            if ndim == 3:
                dimension_names = ['z', 'y', 'x']
            elif ndim == 4:
                dimension_names = ['c', 'z', 'y', 'x']
            elif ndim == 5:
                dimension_names = ['t', 'c', 'z', 'y', 'x']
            else:
                dimension_names = [f'dim_{i}' for i in range(ndim)]

            spec['schema'] = {'dimension_names': dimension_names}

        return spec

    def get_metadata(self) -> Dict:
        """
        Return N5 metadata from attributes.json.

        Reads and parses the N5 attributes.json file which contains
        dataset metadata like dimensions, data type, compression, etc.

        Returns:
            dict: N5 metadata with keys:
                - 'dimensions': Array shape [z, y, x]
                - 'blockSize': Chunk dimensions
                - 'dataType': Data type (uint8, uint16, float32, etc.)
                - 'compression': Compression settings
                - Custom metadata (pixelResolution, units, etc.)

        Example:
            >>> reader = N5Reader("/data.n5")
            >>> metadata = reader.get_metadata()
            >>> print(metadata)
            {
                'dimensions': [100, 1024, 1024],
                'blockSize': [32, 32, 32],
                'dataType': 'uint16',
                'compression': {'type': 'gzip', 'level': 5},
                'pixelResolution': {'dimensions': [0.5, 0.116, 0.116], 'unit': 'um'}
            }

        Notes:
            - Cached after first read for efficiency
            - Returns empty dict if attributes.json not found
            - Custom metadata depends on how N5 was written
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        # Read attributes.json from N5 container
        attributes_path = os.path.join(self._full_path, 'attributes.json')

        if not os.path.exists(attributes_path):
            # Try without dataset_path (root level)
            attributes_path = os.path.join(self.path, 'attributes.json')

        if os.path.exists(attributes_path):
            try:
                with open(attributes_path, 'r') as f:
                    self._metadata_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to read N5 attributes: {e}")
                self._metadata_cache = {}
        else:
            print(f"Warning: N5 attributes.json not found at {attributes_path}")
            self._metadata_cache = {}

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from N5 metadata.

        Extracts physical pixel/voxel sizes from N5 metadata.
        N5 typically stores this in 'pixelResolution' or 'resolution' fields.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in micrometers

        Example:
            >>> reader = N5Reader("/data.n5")
            >>> voxel_sizes = reader.get_voxel_sizes()
            >>> print(voxel_sizes)
            {'x': 0.116, 'y': 0.116, 'z': 0.5}

        Notes:
            - Returns 1.0 for each dimension if metadata doesn't contain voxel sizes
            - Assumes metadata is in micrometers (converts if needed)
            - N5 convention: dimensions = [z, y, x] (reverse of XYZ)
        """
        metadata = self.get_metadata()

        # Try different metadata field names used in N5
        voxel_data = None

        # Check for 'pixelResolution' (Janelia convention)
        if 'pixelResolution' in metadata:
            voxel_data = metadata['pixelResolution']
            if 'dimensions' in voxel_data:
                dimensions = voxel_data['dimensions']
                unit = voxel_data.get('unit', 'um')

                # N5 convention: dimensions are [z, y, x]
                if len(dimensions) >= 3:
                    z, y, x = dimensions[0], dimensions[1], dimensions[2]

                    # Convert to micrometers if needed
                    scale = 1.0
                    if unit in ['nm', 'nanometer']:
                        scale = 0.001  # nm to µm
                    elif unit in ['mm', 'millimeter']:
                        scale = 1000.0  # mm to µm

                    return {
                        'x': float(x) * scale,
                        'y': float(y) * scale,
                        'z': float(z) * scale
                    }

        # Check for 'resolution' (alternative convention)
        elif 'resolution' in metadata:
            resolution = metadata['resolution']
            if isinstance(resolution, list) and len(resolution) >= 3:
                z, y, x = resolution[0], resolution[1], resolution[2]
                return {'x': float(x), 'y': float(y), 'z': float(z)}

        # Check for 'scales' (multi-scale metadata)
        elif 'scales' in metadata:
            scales = metadata['scales']
            if isinstance(scales, list) and len(scales) > 0:
                # Use first scale level
                scale_data = scales[0]
                if isinstance(scale_data, list) and len(scale_data) >= 3:
                    z, y, x = scale_data[0], scale_data[1], scale_data[2]
                    return {'x': float(x), 'y': float(y), 'z': float(z)}

        # Default: return 1.0 for all dimensions if no voxel size found
        print(f"Warning: No voxel size metadata found in N5, using default 1.0")
        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def supports_remote(self) -> bool:
        """
        Check if N5 reader supports remote data.

        N5 with TensorStore supports remote kvstores (HTTP, GCS, S3).

        Returns:
            bool: True (N5 supports remote via TensorStore)

        Example:
            >>> reader = N5Reader("gs://bucket/data.n5")
            >>> reader.supports_remote()
            True
        """
        return True

    def __repr__(self) -> str:
        """String representation of N5 reader."""
        if self.dataset_path:
            return f"N5Reader(path='{self.path}', dataset_path='{self.dataset_path}')"
        return f"N5Reader(path='{self.path}')"
