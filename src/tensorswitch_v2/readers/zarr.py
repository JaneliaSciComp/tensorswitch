"""
Zarr reader implementations using native TensorStore drivers.

Tier 1 readers - maximum performance via direct TensorStore integration.
Contains both Zarr3Reader and Zarr2Reader.
"""

from typing import Dict, Optional, List
import os
import json
import tensorstore as ts
from .base import BaseReader
from ..utils import get_tensorstore_context


class Zarr3Reader(BaseReader):
    """
    Reader for Zarr v3 format using native TensorStore driver.

    Uses TensorStore's native 'zarr3' driver for maximum performance.
    No intermediate layers - direct read from Zarr3 store.

    Tier: 1 (Native TensorStore - Maximum Performance)
    - Zero conversion overhead
    - Direct TensorStore driver
    - Supports local and remote (HTTP, GCS, S3) stores
    - Native sharding support

    Features:
    - OME-NGFF metadata support
    - Multi-resolution pyramid access
    - Sharded chunk support
    - Remote data access (GCS, S3, HTTP)

    Example:
        >>> from tensorswitch_v2.readers import Zarr3Reader
        >>> reader = Zarr3Reader("/path/to/data.zarr")
        >>> spec = reader.get_tensorstore_spec()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset
        >>> dataset = TensorSwitchDataset("/path/to/data.zarr")
        >>> ts_array = dataset.get_tensorstore_array()

    Example (remote):
        >>> reader = Zarr3Reader("gs://bucket/data.zarr")
    """

    def __init__(self, path: str, dataset_path: str = ""):
        """
        Initialize Zarr3 reader.

        Args:
            path: Path to Zarr3 store (local path, GCS, S3, or HTTP URL)
            dataset_path: Path within Zarr store to specific array (e.g., "s0" for scale 0)

        Example:
            >>> # Root of Zarr store
            >>> reader = Zarr3Reader("/data.zarr")
            >>>
            >>> # Specific resolution level
            >>> reader = Zarr3Reader("/data.zarr", dataset_path="s0")
            >>>
            >>> # Remote store
            >>> reader = Zarr3Reader("gs://bucket/data.zarr", dataset_path="s0")
        """
        super().__init__(path)
        self._dataset_path = dataset_path
        self._metadata_cache = None

    def get_tensorstore(self) -> ts.TensorStore:
        """
        Return an opened TensorStore using the native Zarr3 driver.

        Returns:
            ts.TensorStore: Opened store for the Zarr3 dataset.

        Example:
            >>> reader = Zarr3Reader("/data.zarr", dataset_path="s0")
            >>> store = reader.get_tensorstore()
            >>> print(store.shape)

        Notes:
            - Tier 1: Direct TensorStore driver, zero conversion overhead
            - Supports local files, GCS, S3, and HTTP
        """
        spec = {
            'driver': 'zarr3',
            'kvstore': self._build_kvstore(),
            'open': True,
            'context': get_tensorstore_context(),
        }
        return ts.open(spec, read=True).result()

    def _build_kvstore(self) -> Dict:
        """
        Build kvstore spec based on path type (local, GCS, S3, HTTP).
        """
        full_path = os.path.join(self.path, self._dataset_path) if self._dataset_path else self.path

        if full_path.startswith('gs://'):
            # Google Cloud Storage
            parts = full_path[5:].split('/', 1)
            bucket = parts[0]
            path = parts[1] if len(parts) > 1 else ''
            return {'driver': 'gcs', 'bucket': bucket, 'path': path}
        elif full_path.startswith('s3://'):
            # AWS S3
            parts = full_path[5:].split('/', 1)
            bucket = parts[0]
            path = parts[1] if len(parts) > 1 else ''
            return {'driver': 's3', 'bucket': bucket, 'path': path}
        elif full_path.startswith('http://') or full_path.startswith('https://'):
            # HTTP
            return {'driver': 'http', 'base_url': full_path}
        else:
            # Local file
            return {'driver': 'file', 'path': full_path}

    def get_metadata(self) -> Dict:
        """
        Return Zarr3 metadata from zarr.json.

        Reads the zarr.json file which contains array metadata
        and OME-NGFF attributes.

        Returns:
            dict: Zarr3 metadata including OME-NGFF if present

        Example:
            >>> reader = Zarr3Reader("/data.zarr")
            >>> metadata = reader.get_metadata()
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata = {}

        # Build path to zarr.json
        if self._dataset_path:
            zarr_json_path = os.path.join(self.path, self._dataset_path, 'zarr.json')
        else:
            zarr_json_path = os.path.join(self.path, 'zarr.json')

        try:
            with open(zarr_json_path, 'r') as f:
                zarr_metadata = json.load(f)
                metadata['zarr_metadata'] = zarr_metadata

                # Extract shape and dtype (convert to tuple for consistency)
                if 'shape' in zarr_metadata:
                    metadata['shape'] = tuple(zarr_metadata['shape'])
                if 'data_type' in zarr_metadata:
                    metadata['dtype'] = zarr_metadata['data_type']
                if 'chunk_grid' in zarr_metadata:
                    chunk_config = zarr_metadata['chunk_grid'].get('configuration', {})
                    metadata['chunk_shape'] = chunk_config.get('chunk_shape')

                # Extract dimension names
                if 'dimension_names' in zarr_metadata:
                    metadata['dimension_names'] = zarr_metadata['dimension_names']

                # Extract attributes (OME-NGFF)
                if 'attributes' in zarr_metadata:
                    metadata['attributes'] = zarr_metadata['attributes']
                    if 'multiscales' in zarr_metadata['attributes']:
                        metadata['multiscales'] = zarr_metadata['attributes']['multiscales']
        except FileNotFoundError:
            # Try reading from root zarr.json for OME-NGFF metadata
            root_zarr_json = os.path.join(self.path, 'zarr.json')
            try:
                with open(root_zarr_json, 'r') as f:
                    root_metadata = json.load(f)
                    if 'attributes' in root_metadata:
                        metadata['attributes'] = root_metadata['attributes']
                        if 'multiscales' in root_metadata['attributes']:
                            metadata['multiscales'] = root_metadata['attributes']['multiscales']
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to read Zarr3 metadata: {e}")

        self._metadata_cache = metadata
        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from OME-NGFF coordinateTransformations.

        Extracts voxel sizes from the multiscales metadata if available.
        Converts to nanometers based on the unit specified in axes metadata.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = Zarr3Reader("/data.zarr")
            >>> voxel_sizes = reader.get_voxel_sizes()
        """
        from ..utils.format_loaders import convert_to_nanometers

        metadata = self.get_metadata()

        # Try to get from OME-NGFF multiscales
        multiscales = metadata.get('multiscales', [])
        if multiscales:
            datasets = multiscales[0].get('datasets', [])
            axes = multiscales[0].get('axes', [])

            # Build axis name to unit mapping
            axis_units = {}
            for axis in axes:
                axis_name = axis.get('name', '').lower()
                axis_units[axis_name] = axis.get('unit', 'micrometer')

            # Find the dataset matching our path
            for ds in datasets:
                ds_path = ds.get('path', '')
                if ds_path == self._dataset_path or (not self._dataset_path and ds_path in ['', '0', 's0']):
                    transforms = ds.get('coordinateTransformations', [])
                    for t in transforms:
                        if t.get('type') == 'scale':
                            scales = t.get('scale', [])
                            # Map scales to axes and convert to nanometers
                            voxel_sizes = {'x': 1.0, 'y': 1.0, 'z': 1.0}
                            for i, axis in enumerate(axes):
                                axis_name = axis.get('name', '').lower()
                                if i < len(scales) and axis_name in voxel_sizes:
                                    unit = axis_units.get(axis_name, 'micrometer')
                                    voxel_sizes[axis_name] = convert_to_nanometers(scales[i], unit)
                            return voxel_sizes

        # Default
        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def supports_remote(self) -> bool:
        """Check if this is a remote store."""
        return any(self.path.startswith(p) for p in ['gs://', 's3://', 'http://', 'https://'])

    def __repr__(self) -> str:
        """String representation of Zarr3 reader."""
        return f"Zarr3Reader(path='{self.path}', dataset_path='{self._dataset_path}')"


class Zarr2Reader(BaseReader):
    """
    Reader for Zarr v2 format using native TensorStore driver.

    Uses TensorStore's native 'zarr' driver for maximum performance.
    No intermediate layers - direct read from Zarr2 store.

    Tier: 1 (Native TensorStore - Maximum Performance)
    - Zero conversion overhead
    - Direct TensorStore driver
    - Supports local and remote stores
    - Legacy Zarr v2 compatibility

    Features:
    - OME-NGFF metadata support (.zattrs)
    - Multi-resolution pyramid access
    - Blosc/zlib compression support
    - Remote data access

    Example:
        >>> from tensorswitch_v2.readers import Zarr2Reader
        >>> reader = Zarr2Reader("/path/to/data.zarr")
        >>> spec = reader.get_tensorstore_spec()

    Example (multiscale):
        >>> reader = Zarr2Reader("/data.zarr", dataset_path="0")
    """

    def __init__(self, path: str, dataset_path: str = ""):
        """
        Initialize Zarr2 reader.

        Args:
            path: Path to Zarr2 store (local path or URL)
            dataset_path: Path within Zarr store to specific array

        Example:
            >>> reader = Zarr2Reader("/data.zarr")
            >>> reader = Zarr2Reader("/data.zarr", dataset_path="0")
        """
        super().__init__(path)
        self._dataset_path = dataset_path
        self._metadata_cache = None

    def get_tensorstore(self) -> ts.TensorStore:
        """
        Return an opened TensorStore using the native Zarr2 driver.

        Returns:
            ts.TensorStore: Opened store for the Zarr2 dataset.

        Example:
            >>> reader = Zarr2Reader("/data.zarr")
            >>> store = reader.get_tensorstore()
            >>> print(store.shape)
        """
        spec = {
            'driver': 'zarr',  # TensorStore uses 'zarr' for v2
            'kvstore': self._build_kvstore(),
            'open': True,
            'context': get_tensorstore_context(),
        }
        return ts.open(spec, read=True).result()

    def _build_kvstore(self) -> Dict:
        """Build kvstore spec based on path type."""
        full_path = os.path.join(self.path, self._dataset_path) if self._dataset_path else self.path

        if full_path.startswith('gs://'):
            parts = full_path[5:].split('/', 1)
            bucket = parts[0]
            path = parts[1] if len(parts) > 1 else ''
            return {'driver': 'gcs', 'bucket': bucket, 'path': path}
        elif full_path.startswith('s3://'):
            parts = full_path[5:].split('/', 1)
            bucket = parts[0]
            path = parts[1] if len(parts) > 1 else ''
            return {'driver': 's3', 'bucket': bucket, 'path': path}
        elif full_path.startswith('http://') or full_path.startswith('https://'):
            return {'driver': 'http', 'base_url': full_path}
        else:
            return {'driver': 'file', 'path': full_path}

    def get_metadata(self) -> Dict:
        """
        Return Zarr2 metadata from .zarray and .zattrs.

        Reads the .zarray (array metadata) and .zattrs (attributes)
        files from the Zarr2 store.

        Returns:
            dict: Zarr2 metadata including OME-NGFF if present
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata = {}
        base_path = os.path.join(self.path, self._dataset_path) if self._dataset_path else self.path

        # Read .zarray
        zarray_path = os.path.join(base_path, '.zarray')
        try:
            with open(zarray_path, 'r') as f:
                zarray = json.load(f)
                metadata['zarray'] = zarray
                metadata['shape'] = zarray.get('shape')
                metadata['dtype'] = zarray.get('dtype')
                metadata['chunk_shape'] = zarray.get('chunks')
                metadata['compressor'] = zarray.get('compressor')
        except Exception as e:
            print(f"Warning: Failed to read .zarray: {e}")

        # Read .zattrs (OME-NGFF metadata typically here)
        zattrs_path = os.path.join(base_path, '.zattrs')
        try:
            with open(zattrs_path, 'r') as f:
                zattrs = json.load(f)
                metadata['attributes'] = zattrs
                if 'multiscales' in zattrs:
                    metadata['multiscales'] = zattrs['multiscales']
        except FileNotFoundError:
            # Try root .zattrs for OME-NGFF
            root_zattrs = os.path.join(self.path, '.zattrs')
            try:
                with open(root_zattrs, 'r') as f:
                    root_attrs = json.load(f)
                    metadata['root_attributes'] = root_attrs
                    if 'multiscales' in root_attrs:
                        metadata['multiscales'] = root_attrs['multiscales']
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to read .zattrs: {e}")

        self._metadata_cache = metadata
        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from OME-NGFF coordinateTransformations.

        Converts to nanometers based on the unit specified in axes metadata.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers
        """
        from ..utils.format_loaders import convert_to_nanometers

        metadata = self.get_metadata()

        # Try to get from OME-NGFF multiscales
        multiscales = metadata.get('multiscales', [])
        if multiscales:
            datasets = multiscales[0].get('datasets', [])
            axes = multiscales[0].get('axes', [])

            # Build axis name to unit mapping
            axis_units = {}
            for axis in axes:
                axis_name = axis.get('name', '').lower()
                axis_units[axis_name] = axis.get('unit', 'micrometer')

            for ds in datasets:
                ds_path = ds.get('path', '')
                if ds_path == self._dataset_path or (not self._dataset_path and ds_path in ['', '0']):
                    transforms = ds.get('coordinateTransformations', [])
                    for t in transforms:
                        if t.get('type') == 'scale':
                            scales = t.get('scale', [])
                            voxel_sizes = {'x': 1.0, 'y': 1.0, 'z': 1.0}
                            for i, axis in enumerate(axes):
                                axis_name = axis.get('name', '').lower()
                                if i < len(scales) and axis_name in voxel_sizes:
                                    unit = axis_units.get(axis_name, 'micrometer')
                                    voxel_sizes[axis_name] = convert_to_nanometers(scales[i], unit)
                            return voxel_sizes

        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def supports_remote(self) -> bool:
        """Check if this is a remote store."""
        return any(self.path.startswith(p) for p in ['gs://', 's3://', 'http://', 'https://'])

    def __repr__(self) -> str:
        """String representation of Zarr2 reader."""
        return f"Zarr2Reader(path='{self.path}', dataset_path='{self._dataset_path}')"
