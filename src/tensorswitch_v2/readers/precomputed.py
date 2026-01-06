"""
Neuroglancer Precomputed reader implementation using native TensorStore driver.

Tier 1 reader - maximum performance with zero conversion overhead.
"""

from typing import Dict
import tensorstore as ts
from .base import BaseReader


class PrecomputedReader(BaseReader):
    """
    Reader for Neuroglancer Precomputed format using native TensorStore driver.

    Precomputed is a chunked multi-resolution format developed by Google for
    Neuroglancer visualization. TensorStore provides native support, making
    this a Tier 1 reader with maximum performance.

    Tier: 1 (Native TensorStore - Maximum Performance)
    - Zero conversion overhead
    - Direct TensorStore driver usage
    - Excellent remote data support (HTTP, GCS, S3)
    - Multi-resolution pyramids native

    Features:
    - Multi-scale pyramids
    - Sharded format option
    - Optimized for web visualization
    - Commonly used for connectomics data

    Example:
        >>> from tensorswitch_v2.readers import PrecomputedReader
        >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
        >>> spec = reader.get_tensorstore_spec()

    Example (HTTP):
        >>> reader = PrecomputedReader("precomputed://https://example.com/data")
        >>> dataset = TensorSwitchDataset("precomputed://...", reader=reader)
    """

    def __init__(self, path: str, scale_index: int = 0):
        """
        Initialize Precomputed reader.

        Args:
            path: Precomputed URL (precomputed://http://..., precomputed://gs://...)
            scale_index: Which resolution level to read (0 = highest resolution)

        Example:
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data", scale_index=1)
        """
        super().__init__(path)
        self.scale_index = scale_index

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec using native Neuroglancer Precomputed driver.

        This is a Tier 1 reader - uses TensorStore's native driver
        with zero conversion overhead.

        Returns:
            dict: TensorStore spec for Precomputed dataset

        Example:
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec)
            {
                'driver': 'neuroglancer_precomputed',
                'kvstore': 'gs://bucket/data',
                'scale_index': 0,
                'open': True
            }

        Notes:
            - Native driver = maximum performance
            - Excellent for remote data (HTTP, GCS, S3)
            - Handles multi-resolution automatically
        """
        spec = {
            'driver': 'neuroglancer_precomputed',
            'kvstore': self.path.replace('precomputed://', ''),
            'scale_index': self.scale_index,
            'open': True
        }

        return spec

    def get_metadata(self) -> Dict:
        """
        Return Precomputed metadata.

        Reads the info file from the Precomputed dataset which contains
        multi-scale metadata, data type, voxel sizes, etc.

        Returns:
            dict: Precomputed metadata

        Notes:
            - Metadata fetched from TensorStore
            - Contains multi-scale information
            - Includes voxel sizes and coordinate space
        """
        # Open the dataset to get metadata
        spec = self.get_tensorstore_spec()
        try:
            store = ts.open(spec).result()
            # Extract basic metadata from TensorStore
            metadata = {
                'shape': list(store.shape),
                'dtype': str(store.dtype),
                'scale_index': self.scale_index
            }
            return metadata
        except Exception as e:
            print(f"Warning: Failed to read Precomputed metadata: {e}")
            return {}

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from Precomputed metadata.

        Precomputed format includes voxel sizes in the info file.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers
                  (converted to micrometers)

        Example:
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
            >>> voxel_sizes = reader.get_voxel_sizes()
            >>> print(voxel_sizes)
            {'x': 8.0, 'y': 8.0, 'z': 40.0}  # in micrometers

        Notes:
            - Precomputed stores in nanometers, converted to micrometers
            - Returns 1.0 for each dimension if metadata unavailable
        """
        # For now, return defaults
        # TODO: Parse Precomputed info file for actual voxel sizes
        print(f"Warning: Precomputed voxel size parsing not yet implemented, using default 1.0")
        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def supports_remote(self) -> bool:
        """
        Check if Precomputed reader supports remote data.

        Precomputed is designed for remote data (HTTP, GCS, S3).

        Returns:
            bool: True (Precomputed is primarily for remote data)

        Example:
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
            >>> reader.supports_remote()
            True
        """
        return True

    def __repr__(self) -> str:
        """String representation of Precomputed reader."""
        return f"PrecomputedReader(path='{self.path}', scale_index={self.scale_index})"
