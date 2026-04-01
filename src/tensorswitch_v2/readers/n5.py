"""
N5 reader implementation using native TensorStore driver.

Tier 1 reader - maximum performance with zero conversion overhead.
Supports local paths and remote URLs (HTTP, S3, GCS).
"""

import json
import os
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import tensorstore as ts
from .base import BaseReader, _default_voxel_sizes
from ..utils.format_loaders import convert_to_nanometers
from ..utils import get_tensorstore_context


def _parse_remote_url(url: str) -> Tuple[str, dict]:
    """
    Parse a URL and return the appropriate kvstore driver and spec.

    Args:
        url: Local path or remote URL (http://, https://, s3://, gs://)

    Returns:
        Tuple of (driver_name, kvstore_spec)

    Examples:
        >>> _parse_remote_url('/local/path/data.n5')
        ('file', {'driver': 'file', 'path': '/local/path/data.n5'})

        >>> _parse_remote_url('http://server.com/data.n5')
        ('http', {'driver': 'http', 'base_url': 'http://server.com/data.n5'})

        >>> _parse_remote_url('s3://bucket/data.n5')
        ('s3', {'driver': 's3', 'bucket': 'bucket', 'path': 'data.n5'})
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in ('http', 'https'):
        # HTTP/HTTPS URL
        return 'http', {'driver': 'http', 'base_url': url}

    elif scheme == 's3':
        # S3 URL: s3://bucket/path
        bucket = parsed.netloc
        path = parsed.path.lstrip('/')
        return 's3', {'driver': 's3', 'bucket': bucket, 'path': path}

    elif scheme == 'gs':
        # Google Cloud Storage URL: gs://bucket/path
        bucket = parsed.netloc
        path = parsed.path.lstrip('/')
        return 'gcs', {'driver': 'gcs', 'bucket': bucket, 'path': path}

    else:
        # Local file path (no scheme or file://)
        path = url if not scheme else parsed.path
        return 'file', {'driver': 'file', 'path': path}


def _is_remote_url(path: str) -> bool:
    """Check if path is a remote URL."""
    parsed = urlparse(path)
    return parsed.scheme.lower() in ('http', 'https', 's3', 'gs')


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
        >>> store = reader.get_tensorstore()
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
        self._ts_store_cache = None

    def _build_spec(self) -> Dict:
        """Build TensorStore spec dict for N5 (without opening)."""
        driver_type, kvstore_spec = _parse_remote_url(self.path)

        spec = {
            'driver': 'n5',
            'kvstore': kvstore_spec,
            'open': True
        }

        if self.dataset_path:
            spec['path'] = self.dataset_path

        return spec

    def get_tensorstore(self) -> ts.TensorStore:
        """Return opened TensorStore using native N5 driver."""
        if self._ts_store_cache is not None:
            return self._ts_store_cache

        spec = self._build_spec()
        spec['context'] = get_tensorstore_context()
        self._ts_store_cache = ts.open(spec, read=True).result()
        return self._ts_store_cache

    def get_metadata(self) -> Dict:
        """
        Return N5 metadata from attributes.json.

        Reads and parses the N5 attributes.json file which contains
        dataset metadata like dimensions, data type, compression, etc.
        Supports both local paths and remote URLs (HTTP, S3, GCS).

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
            - For remote URLs, fetches via HTTP/S3/GCS
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._metadata_cache = {}

        if _is_remote_url(self.path):
            # Remote URL - fetch attributes.json via HTTP
            self._metadata_cache = self._fetch_remote_metadata()
        else:
            # Local path - read from filesystem
            self._metadata_cache = self._read_local_metadata()

        return self._metadata_cache

    def _read_local_metadata(self) -> Dict:
        """Read metadata from local filesystem."""
        # Read attributes.json from N5 container
        attributes_path = os.path.join(self._full_path, 'attributes.json')

        if not os.path.exists(attributes_path):
            # Try without dataset_path (root level)
            attributes_path = os.path.join(self.path, 'attributes.json')

        if os.path.exists(attributes_path):
            try:
                with open(attributes_path, 'r') as f:
                    metadata = json.load(f)
                # Add 'shape' for consistency with other readers (N5 uses 'dimensions')
                if 'dimensions' in metadata:
                    metadata['shape'] = tuple(metadata['dimensions'])
                return metadata
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to read N5 attributes: {e}")
                return {}
        else:
            print(f"Warning: N5 attributes.json not found at {attributes_path}")
            return {}

    def _fetch_remote_metadata(self) -> Dict:
        """Fetch metadata from remote URL (HTTP/S3/GCS)."""
        import urllib.request
        import urllib.error

        # Build URL to attributes.json
        if self.dataset_path:
            # URL with dataset path
            base_url = self.path.rstrip('/')
            attributes_url = f"{base_url}/{self.dataset_path}/attributes.json"
        else:
            attributes_url = f"{self.path.rstrip('/')}/attributes.json"

        try:
            with urllib.request.urlopen(attributes_url, timeout=30) as response:
                content = response.read().decode('utf-8')
                metadata = json.loads(content)
                # Add 'shape' for consistency with other readers (N5 uses 'dimensions')
                if 'dimensions' in metadata:
                    metadata['shape'] = tuple(metadata['dimensions'])
                return metadata
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"Warning: N5 attributes.json not found at {attributes_url}")
            else:
                print(f"Warning: HTTP error fetching N5 attributes: {e}")
            return {}
        except urllib.error.URLError as e:
            print(f"Warning: URL error fetching N5 attributes: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in N5 attributes: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Failed to fetch N5 attributes: {e}")
            return {}

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from N5 metadata.

        Extracts physical pixel/voxel sizes from N5 metadata.
        N5 typically stores this in 'pixelResolution' or 'resolution' fields.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = N5Reader("/data.n5")
            >>> voxel_sizes = reader.get_voxel_sizes()
            >>> print(voxel_sizes)
            {'x': 116.0, 'y': 116.0, 'z': 500.0}

        Notes:
            - Returns 1.0 for each dimension if metadata doesn't contain voxel sizes
            - Detects unit from metadata and converts to nanometers
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

                    # Convert to nanometers using helper
                    return {
                        'x': convert_to_nanometers(x, unit),
                        'y': convert_to_nanometers(y, unit),
                        'z': convert_to_nanometers(z, unit)
                    }

        # Check for 'resolution' (alternative convention) - assume nanometers if no unit
        elif 'resolution' in metadata:
            resolution = metadata['resolution']
            if isinstance(resolution, list) and len(resolution) >= 3:
                z, y, x = resolution[0], resolution[1], resolution[2]
                # No unit specified, assume already nanometers
                return {'x': float(x), 'y': float(y), 'z': float(z)}

        # Check for 'scales' (multi-scale metadata) - assume nanometers if no unit
        elif 'scales' in metadata:
            scales = metadata['scales']
            if isinstance(scales, list) and len(scales) > 0:
                # Use first scale level
                scale_data = scales[0]
                if isinstance(scale_data, list) and len(scale_data) >= 3:
                    z, y, x = scale_data[0], scale_data[1], scale_data[2]
                    return {'x': float(x), 'y': float(y), 'z': float(z)}

        return _default_voxel_sizes("N5")

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
