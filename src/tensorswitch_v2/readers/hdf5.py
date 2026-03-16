"""
HDF5 reader implementation for generic HDF5 files.

Tier 2 reader - new implementation for HDF5 format.
"""

from typing import Dict, Optional, List
import dask.array as da
from .base import DaskReader


class HDF5Reader(DaskReader):
    """
    Reader for generic HDF5 files.

    Uses h5py and dask for lazy loading. DaskReader base class wraps
    the dask array via ts.virtual_chunked for a uniform TensorStore API.

    Tier: 2 (Custom Optimized - Production Ready)

    Example:
        >>> from tensorswitch_v2.readers import HDF5Reader
        >>> reader = HDF5Reader("/path/to/data.h5", dataset_path="/data/volume")
        >>> store = reader.get_tensorstore()
    """

    def __init__(
        self,
        path: str,
        dataset_path: Optional[str] = None,
        chunk_shape: Optional[tuple] = None
    ):
        super().__init__(path)
        self._dataset_path = dataset_path
        self._chunk_shape = chunk_shape
        self._metadata_cache = None
        self._h5file = None

    def _open_dataset(self):
        """Open HDF5 file and locate the dataset."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 support. Install with:\n"
                "  pip install h5py"
            )

        self._h5file = h5py.File(self.path, 'r')

        if self._dataset_path:
            if self._dataset_path not in self._h5file:
                raise ValueError(
                    f"Dataset '{self._dataset_path}' not found in {self.path}. "
                    f"Available datasets: {self._list_datasets(self._h5file)}"
                )
            return self._h5file[self._dataset_path]
        else:
            dataset = self._find_main_dataset(self._h5file)
            if dataset is None:
                raise ValueError(
                    f"Could not auto-detect dataset in {self.path}. "
                    f"Please specify dataset_path. "
                    f"Available datasets: {self._list_datasets(self._h5file)}"
                )
            return dataset

    def _find_main_dataset(self, h5file) -> Optional[object]:
        """Try to find the main dataset in an HDF5 file."""
        import h5py

        common_names = [
            'data', 'volume', 'image', 'raw', 'stack',
            'Data', 'Volume', 'Image', 'Raw', 'Stack',
            's0', 's0/data', 'volumes/raw'
        ]

        for name in common_names:
            if name in h5file:
                obj = h5file[name]
                if isinstance(obj, h5py.Dataset):
                    self._dataset_path = name
                    return obj

        largest_dataset = None
        largest_size = 0

        def find_datasets(group, prefix=''):
            nonlocal largest_dataset, largest_size
            for key in group.keys():
                path = f"{prefix}/{key}" if prefix else key
                obj = group[key]
                if isinstance(obj, h5py.Dataset):
                    size = obj.size
                    if size > largest_size:
                        largest_size = size
                        largest_dataset = obj
                        self._dataset_path = path
                elif isinstance(obj, h5py.Group):
                    find_datasets(obj, path)

        find_datasets(h5file)
        return largest_dataset

    def _list_datasets(self, h5file) -> List[str]:
        """List all datasets in HDF5 file."""
        import h5py
        datasets = []

        def find_all(group, prefix=''):
            for key in group.keys():
                path = f"{prefix}/{key}" if prefix else key
                obj = group[key]
                if isinstance(obj, h5py.Dataset):
                    datasets.append(path)
                elif isinstance(obj, h5py.Group):
                    find_all(obj, path)

        find_all(h5file)
        return datasets

    def _load(self):
        """Lazy-load the HDF5 data into a dask array."""
        if self._dask_array is not None:
            return

        dataset = self._open_dataset()

        # Determine chunk shape
        if self._chunk_shape:
            chunks = self._chunk_shape
        elif dataset.chunks:
            chunks = dataset.chunks
        else:
            chunks = 'auto'

        self._dask_array = da.from_array(dataset, chunks=chunks)

    def _get_dimension_names(self) -> List[str]:
        """Infer dimension names from array shape."""
        self._load()
        ndim = len(self._dask_array.shape)
        if ndim == 2:
            return ['y', 'x']
        elif ndim == 3:
            return ['z', 'y', 'x']
        elif ndim == 4:
            return ['c', 'z', 'y', 'x']
        elif ndim == 5:
            return ['t', 'c', 'z', 'y', 'x']
        else:
            return [f'dim_{i}' for i in range(ndim)]

    def get_metadata(self) -> Dict:
        """Return HDF5 metadata from dataset and file attributes."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        dataset = self._open_dataset()

        metadata = {
            'dataset_path': self._dataset_path,
            'shape': dataset.shape,
            'dtype': str(dataset.dtype),
            'chunks': dataset.chunks,
        }

        attrs = {}
        for key in dataset.attrs.keys():
            try:
                value = dataset.attrs[key]
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                attrs[key] = value
            except Exception:
                pass

        metadata['attributes'] = attrs

        unit_names = ['unit', 'units', 'voxel_unit', 'pixel_unit', 'resolution_unit',
                      'spatial_unit', 'voxel_units', 'pixel_units']
        detected_unit = 'micrometer'
        for unit_name in unit_names:
            if unit_name in attrs:
                detected_unit = str(attrs[unit_name])
                break
        metadata['voxel_unit'] = detected_unit

        voxel_size_names = {
            'x': ['voxel_size_x', 'pixel_size_x', 'resolution_x', 'dx', 'scale_x'],
            'y': ['voxel_size_y', 'pixel_size_y', 'resolution_y', 'dy', 'scale_y'],
            'z': ['voxel_size_z', 'pixel_size_z', 'resolution_z', 'dz', 'scale_z'],
        }

        for dim, names in voxel_size_names.items():
            for name in names:
                if name in attrs:
                    metadata[f'voxel_size_{dim}'] = float(attrs[name])
                    break
            if f'voxel_size_{dim}' not in metadata:
                metadata[f'voxel_size_{dim}'] = 1.0

        self._metadata_cache = metadata
        return metadata

    def get_voxel_sizes(self) -> Dict[str, float]:
        """Return voxel dimensions from HDF5 attributes in nanometers."""
        from ..utils.format_loaders import convert_to_nanometers

        metadata = self.get_metadata()
        unit = metadata.get('voxel_unit', 'micrometer')

        return {
            'x': convert_to_nanometers(metadata.get('voxel_size_x', 1.0), unit),
            'y': convert_to_nanometers(metadata.get('voxel_size_y', 1.0), unit),
            'z': convert_to_nanometers(metadata.get('voxel_size_z', 1.0), unit)
        }

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"HDF5Reader(path='{self.path}', dataset_path='{self._dataset_path}')"
