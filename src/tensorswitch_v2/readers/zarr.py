"""
Zarr reader implementations using native TensorStore drivers.

Tier 1 readers - maximum performance via direct TensorStore integration.
Contains both Zarr3Reader and Zarr2Reader.
"""

from typing import Dict, Optional, List
import os
import json
import tensorstore as ts
from .base import BaseReader, _default_voxel_sizes, build_kvstore as _build_kvstore_shared
from ..utils import get_tensorstore_context
from ..utils.format_loaders import convert_to_nanometers


def _extract_voxel_sizes_from_multiscales(metadata, dataset_path, default_paths):
    """Extract voxel sizes from OME-NGFF multiscales metadata.

    Shared helper for Zarr3Reader and Zarr2Reader get_voxel_sizes().

    Args:
        metadata: Parsed group metadata dict (must have 'multiscales' key).
        dataset_path: The reader's dataset path (e.g. 's0', '0').
        default_paths: Fallback path strings to match when dataset_path is empty.

    Returns:
        dict with 'x','y','z' in nanometers, or None if not found.
    """
    multiscales = metadata.get('multiscales', [])
    if not multiscales:
        return None

    datasets = multiscales[0].get('datasets', [])
    axes = multiscales[0].get('axes', [])

    # Build axis name → unit mapping
    axis_units = {}
    for axis in axes:
        axis_name = axis.get('name', '').lower()
        axis_units[axis_name] = axis.get('unit', 'micrometer')

    # Find the dataset matching our path.
    # Try exact match first, then basename match for nested paths like "img/s0".
    for ds in datasets:
        ds_path = ds.get('path', '')
        ds_basename = ds_path.rsplit('/', 1)[-1] if '/' in ds_path else ds_path
        if (ds_path == dataset_path
                or (not dataset_path and ds_path in default_paths)
                or (not dataset_path and ds_basename in default_paths)):
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

    return None


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
        >>> store = reader.get_tensorstore()

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
        self._ts_store_cache = None

    def _build_spec(self) -> Dict:
        """Build TensorStore spec dict for Zarr3 (without opening)."""
        kvstore = self._build_kvstore()
        return {
            'driver': 'zarr3',
            'kvstore': kvstore,
            'open': True,
        }

    def get_tensorstore(self) -> ts.TensorStore:
        """Return opened TensorStore using native zarr3 driver."""
        if self._ts_store_cache is not None:
            return self._ts_store_cache

        spec = self._build_spec()
        spec['context'] = get_tensorstore_context()
        self._ts_store_cache = ts.open(spec, read=True).result()
        return self._ts_store_cache

    def _build_kvstore(self) -> Dict:
        """Build kvstore spec using shared build_kvstore (handles S3, GCS, HTTP, local)."""
        full_path = os.path.join(self.path, self._dataset_path) if self._dataset_path else self.path
        return _build_kvstore_shared(full_path)

    def get_metadata(self) -> Dict:
        """
        Return Zarr3 metadata from zarr.json.

        Reads the zarr.json file which contains array metadata
        and OME-NGFF attributes. Supports both local and remote stores.

        Returns:
            dict: Zarr3 metadata including OME-NGFF if present

        Example:
            >>> reader = Zarr3Reader("/data.zarr")
            >>> metadata = reader.get_metadata()
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        if self.supports_remote():
            metadata = self._read_remote_metadata()
        else:
            metadata = self._read_local_metadata()

        self._metadata_cache = metadata
        return metadata

    def _read_local_metadata(self) -> Dict:
        """Read zarr.json metadata from local filesystem."""
        metadata = {}

        if self._dataset_path:
            zarr_json_path = os.path.join(self.path, self._dataset_path, 'zarr.json')
        else:
            zarr_json_path = os.path.join(self.path, 'zarr.json')

        try:
            with open(zarr_json_path, 'r') as f:
                zarr_metadata = json.load(f)
                self._extract_zarr3_fields(metadata, zarr_metadata)
        except FileNotFoundError:
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

        return metadata

    def _read_remote_metadata(self) -> Dict:
        """Read zarr.json metadata from remote store via TensorStore kvstore."""
        from .base import build_kvstore

        metadata = {}
        key = (self._dataset_path + '/zarr.json') if self._dataset_path else 'zarr.json'

        try:
            kvs = ts.KvStore.open(build_kvstore(self.path)).result()
            read_result = kvs.read(key).result()
            if read_result.value is not None and len(read_result.value) > 0:
                zarr_metadata = json.loads(bytes(read_result.value))
                self._extract_zarr3_fields(metadata, zarr_metadata)
        except Exception:
            # Fall back: try root zarr.json
            if self._dataset_path:
                try:
                    kvs = ts.KvStore.open(build_kvstore(self.path)).result()
                    read_result = kvs.read('zarr.json').result()
                    if read_result.value is not None and len(read_result.value) > 0:
                        root_metadata = json.loads(bytes(read_result.value))
                        if 'attributes' in root_metadata:
                            metadata['attributes'] = root_metadata['attributes']
                            if 'multiscales' in root_metadata['attributes']:
                                metadata['multiscales'] = root_metadata['attributes']['multiscales']
                except Exception:
                    pass

        return metadata

    @staticmethod
    def _extract_zarr3_fields(metadata: Dict, zarr_metadata: Dict) -> None:
        """Extract standard fields from parsed zarr.json into metadata dict."""
        metadata['zarr_metadata'] = zarr_metadata
        if 'shape' in zarr_metadata:
            metadata['shape'] = tuple(zarr_metadata['shape'])
        if 'data_type' in zarr_metadata:
            metadata['dtype'] = zarr_metadata['data_type']
        if 'chunk_grid' in zarr_metadata:
            chunk_config = zarr_metadata['chunk_grid'].get('configuration', {})
            metadata['chunk_shape'] = chunk_config.get('chunk_shape')
        if 'dimension_names' in zarr_metadata:
            metadata['dimension_names'] = zarr_metadata['dimension_names']
        if 'attributes' in zarr_metadata:
            metadata['attributes'] = zarr_metadata['attributes']
            if 'multiscales' in zarr_metadata['attributes']:
                metadata['multiscales'] = zarr_metadata['attributes']['multiscales']

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
        result = _extract_voxel_sizes_from_multiscales(
            self.get_metadata(), self._dataset_path, ('', '0', 's0'))
        return result if result else _default_voxel_sizes("Zarr")

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
        >>> store = reader.get_tensorstore()

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
        self._ts_store_cache = None

    def _build_spec(self) -> Dict:
        """Build TensorStore spec dict for Zarr2 (without opening)."""
        kvstore = self._build_kvstore()
        return {
            'driver': 'zarr',  # TensorStore uses 'zarr' for v2
            'kvstore': kvstore,
            'open': True,
        }

    def get_tensorstore(self) -> ts.TensorStore:
        """Return opened TensorStore using native zarr (v2) driver."""
        if self._ts_store_cache is not None:
            return self._ts_store_cache

        spec = self._build_spec()
        spec['context'] = get_tensorstore_context()
        self._ts_store_cache = ts.open(spec, read=True).result()
        return self._ts_store_cache

    def _build_kvstore(self) -> Dict:
        """Build kvstore spec using shared build_kvstore (handles S3, GCS, HTTP, local)."""
        full_path = os.path.join(self.path, self._dataset_path) if self._dataset_path else self.path
        return _build_kvstore_shared(full_path)

    def get_metadata(self) -> Dict:
        """
        Return Zarr2 metadata from .zarray and .zattrs.

        Reads the .zarray (array metadata) and .zattrs (attributes)
        files from the Zarr2 store. Supports both local and remote stores.

        Returns:
            dict: Zarr2 metadata including OME-NGFF if present
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        if self.supports_remote():
            metadata = self._read_remote_metadata()
        else:
            metadata = self._read_local_metadata()

        self._metadata_cache = metadata
        return metadata

    def _read_local_metadata(self) -> Dict:
        """Read .zarray and .zattrs metadata from local filesystem."""
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
            pass
        except Exception as e:
            print(f"Warning: Failed to read .zattrs: {e}")

        # If no multiscales found at array level, walk up parent directories.
        # OME-NGFF multiscales live on the group .zattrs (e.g. img/.zattrs)
        # or the container root .zattrs (e.g. data.zarr/.zattrs).
        if 'multiscales' not in metadata:
            current = os.path.dirname(os.path.abspath(base_path))
            stop_at = os.path.dirname(os.path.abspath(self.path))
            # Walk up at most 5 levels to avoid runaway traversal
            for _ in range(5):
                parent_zattrs = os.path.join(current, '.zattrs')
                try:
                    with open(parent_zattrs, 'r') as f:
                        parent_attrs = json.load(f)
                        if 'multiscales' in parent_attrs:
                            metadata['multiscales'] = parent_attrs['multiscales']
                            metadata['_multiscales_source'] = current
                            break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if current == stop_at or current == os.path.dirname(current):
                    break
                current = os.path.dirname(current)

        return metadata

    def _read_remote_metadata(self) -> Dict:
        """Read .zarray and .zattrs metadata from remote store via TensorStore kvstore."""
        from .base import build_kvstore

        metadata = {}
        base_key = (self._dataset_path + '/') if self._dataset_path else ''

        try:
            kvs = ts.KvStore.open(build_kvstore(self.path)).result()

            # Read .zarray
            try:
                read_result = kvs.read(base_key + '.zarray').result()
                if read_result.value is not None and len(read_result.value) > 0:
                    zarray = json.loads(bytes(read_result.value))
                    metadata['zarray'] = zarray
                    metadata['shape'] = zarray.get('shape')
                    metadata['dtype'] = zarray.get('dtype')
                    metadata['chunk_shape'] = zarray.get('chunks')
                    metadata['compressor'] = zarray.get('compressor')
            except Exception:
                pass

            # Read .zattrs (try array-level first, then root for multiscales)
            found_attrs = False
            try:
                read_result = kvs.read(base_key + '.zattrs').result()
                if read_result.value is not None and len(read_result.value) > 0:
                    zattrs = json.loads(bytes(read_result.value))
                    metadata['attributes'] = zattrs
                    if 'multiscales' in zattrs:
                        metadata['multiscales'] = zattrs['multiscales']
                        found_attrs = True
            except Exception:
                pass

            # Fall back to root .zattrs for multiscales
            if not found_attrs and self._dataset_path:
                try:
                    read_result = kvs.read('.zattrs').result()
                    if read_result.value is not None and len(read_result.value) > 0:
                        root_attrs = json.loads(bytes(read_result.value))
                        metadata['root_attributes'] = root_attrs
                        if 'multiscales' in root_attrs:
                            metadata['multiscales'] = root_attrs['multiscales']
                except Exception:
                    pass
        except Exception:
            pass

        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """
        Return voxel dimensions from OME-NGFF coordinateTransformations.

        Converts to nanometers based on the unit specified in axes metadata.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers
        """
        result = _extract_voxel_sizes_from_multiscales(
            self.get_metadata(), self._dataset_path, ('', '0', 's0'))
        return result if result else _default_voxel_sizes("Zarr")

    def supports_remote(self) -> bool:
        """Check if this is a remote store."""
        return any(self.path.startswith(p) for p in ['gs://', 's3://', 'http://', 'https://'])

    def __repr__(self) -> str:
        """String representation of Zarr2 reader."""
        return f"Zarr2Reader(path='{self.path}', dataset_path='{self._dataset_path}')"
