"""
Neuroglancer Precomputed reader implementation using native TensorStore driver.

Tier 1 reader - maximum performance with zero conversion overhead.
Supports local paths and remote URLs (HTTP, GCS, S3).
"""

import os
import json
from typing import Dict, Optional
import tensorstore as ts
from .base import BaseReader, build_kvstore, is_remote_path
from ..utils.format_loaders import extract_precomputed_metadata
from ..utils import get_tensorstore_context


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
        >>> store = reader.get_tensorstore()

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
        self._ts_store_cache = None

    def _build_spec(self) -> Dict:
        """Build TensorStore spec dict for Precomputed (without opening)."""
        clean_path = self.path
        if clean_path.startswith('precomputed://'):
            clean_path = clean_path[len('precomputed://'):]

        kvstore = build_kvstore(clean_path)

        return {
            'driver': 'neuroglancer_precomputed',
            'kvstore': kvstore,
            'scale_index': self.scale_index,
            'open': True
        }

    def get_tensorstore(self) -> ts.TensorStore:
        """Return opened TensorStore using native Precomputed driver."""
        if self._ts_store_cache is not None:
            return self._ts_store_cache

        spec = self._build_spec()
        spec['context'] = get_tensorstore_context()
        self._ts_store_cache = ts.open(spec, read=True).result()
        return self._ts_store_cache

    def get_metadata(self) -> Dict:
        """
        Return Precomputed metadata.

        Reads the info file from the Precomputed dataset which contains
        multi-scale metadata, data type, voxel sizes, etc.

        Returns:
            dict: Precomputed metadata including:
                - shape: Array dimensions
                - dtype: Data type
                - scale_index: Current resolution level
                - info: Full precomputed info file content
                - scales: Multi-resolution scale information

        Notes:
            - Reads info file directly for full metadata
            - Also fetches shape/dtype from TensorStore
            - Contains multi-scale information and voxel sizes
        """
        # Get full info file content
        info, _ = extract_precomputed_metadata(self.path, self.scale_index)

        # Also get basic info from TensorStore for shape/dtype
        try:
            store = self.get_tensorstore()
            metadata = {
                'shape': list(store.shape),
                'dtype': str(store.dtype),
                'scale_index': self.scale_index
            }
        except Exception as e:
            print(f"Warning: Failed to read Precomputed shape from TensorStore: {e}")
            metadata = {'scale_index': self.scale_index}

        # Add info file content
        if info:
            metadata['info'] = info
            metadata['scales'] = info.get('scales', [])

        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from Precomputed metadata.

        Precomputed format includes voxel sizes in the info file as 'resolution'
        in nanometers with XYZ order.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = PrecomputedReader("precomputed://gs://bucket/data")
            >>> voxel_sizes = reader.get_voxel_sizes()
            >>> print(voxel_sizes)
            {'x': 8.0, 'y': 8.0, 'z': 40.0}  # in nanometers

        Notes:
            - Precomputed stores resolution in nanometers (preserved as-is)
            - Resolution is in XYZ order in the info file
            - Returns 1.0 for each dimension if metadata unavailable
        """
        _, voxel_sizes = extract_precomputed_metadata(self.path, self.scale_index)

        if voxel_sizes is None:
            print(f"Warning: Could not extract voxel sizes from precomputed info, using default 1.0")
            return {'x': 1.0, 'y': 1.0, 'z': 1.0}

        return voxel_sizes

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
