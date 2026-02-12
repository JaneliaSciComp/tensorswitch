"""
HDF5 reader implementation for generic HDF5 files.

Tier 2 reader - new implementation for HDF5 format.
"""

from typing import Dict, Optional, List
import dask.array as da
from .base import BaseReader


class HDF5Reader(BaseReader):
    """
    Reader for generic HDF5 files.

    Provides access to HDF5 datasets using h5py and dask for lazy loading.
    Wraps the data in TensorStore's 'array' driver for compatibility
    with the unified architecture.

    Tier: 2 (Custom Optimized - Production Ready)
    - Uses h5py for HDF5 access
    - Dask for lazy/chunked loading
    - Supports dataset path specification
    - Handles various HDF5 structures

    Features:
    - Generic HDF5 dataset access
    - Lazy loading via dask.array.from_array
    - Metadata extraction from HDF5 attributes
    - Support for nested dataset paths

    Example:
        >>> from tensorswitch_v2.readers import HDF5Reader
        >>> reader = HDF5Reader("/path/to/data.h5", dataset_path="/data/volume")
        >>> spec = reader.get_tensorstore_spec()

    Example (with TensorSwitchDataset):
        >>> from tensorswitch_v2.api import TensorSwitchDataset, Readers
        >>> reader = Readers.hdf5("/path/to/data.h5", dataset_path="/volumes/raw")
        >>> dataset = TensorSwitchDataset("/path/to/data.h5", reader=reader)

    Notes:
        - For Imaris IMS files, use IMSReader instead (specialized)
        - dataset_path must be specified if HDF5 contains multiple datasets
    """

    def __init__(
        self,
        path: str,
        dataset_path: Optional[str] = None,
        chunk_shape: Optional[tuple] = None
    ):
        """
        Initialize HDF5 reader.

        Args:
            path: Path to HDF5 file
            dataset_path: Path to dataset within HDF5 file (e.g., "/data/volume")
                         If None, will try to auto-detect the main dataset
            chunk_shape: Optional chunk shape for dask array (uses HDF5 chunks if None)

        Example:
            >>> reader = HDF5Reader("/data.h5", dataset_path="/volume")
            >>> reader = HDF5Reader("/data.h5")  # Auto-detect dataset
        """
        super().__init__(path)
        self._dataset_path = dataset_path
        self._chunk_shape = chunk_shape
        self._dask_array = None
        self._metadata_cache = None
        self._h5file = None

    def _open_dataset(self):
        """
        Open HDF5 file and locate the dataset.

        Returns:
            h5py.Dataset: The HDF5 dataset object
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 support. Install with:\n"
                "  pip install h5py"
            )

        self._h5file = h5py.File(self.path, 'r')

        if self._dataset_path:
            # Use specified dataset path
            if self._dataset_path not in self._h5file:
                raise ValueError(
                    f"Dataset '{self._dataset_path}' not found in {self.path}. "
                    f"Available datasets: {self._list_datasets(self._h5file)}"
                )
            return self._h5file[self._dataset_path]
        else:
            # Auto-detect main dataset
            dataset = self._find_main_dataset(self._h5file)
            if dataset is None:
                raise ValueError(
                    f"Could not auto-detect dataset in {self.path}. "
                    f"Please specify dataset_path. "
                    f"Available datasets: {self._list_datasets(self._h5file)}"
                )
            return dataset

    def _find_main_dataset(self, h5file) -> Optional[object]:
        """
        Try to find the main dataset in an HDF5 file.

        Searches for common dataset names or the largest dataset.
        """
        import h5py

        # Common dataset names to look for
        common_names = [
            'data', 'volume', 'image', 'raw', 'stack',
            'Data', 'Volume', 'Image', 'Raw', 'Stack',
            's0', 's0/data', 'volumes/raw'
        ]

        # Try common names first
        for name in common_names:
            if name in h5file:
                obj = h5file[name]
                if isinstance(obj, h5py.Dataset):
                    self._dataset_path = name
                    return obj

        # Find largest dataset
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

    def get_tensorstore_spec(self) -> Dict:
        """
        Return TensorStore spec wrapping Dask array from HDF5 dataset.

        Opens the HDF5 file, creates a Dask array for lazy access,
        and wraps it in TensorStore's 'array' driver.

        Returns:
            dict: TensorStore spec with 'array' driver wrapping Dask array

        Example:
            >>> reader = HDF5Reader("/data.h5", dataset_path="/volume")
            >>> spec = reader.get_tensorstore_spec()
            >>> print(spec['driver'])
            'array'
        """
        if self._dask_array is None:
            dataset = self._open_dataset()

            # Determine chunk shape
            if self._chunk_shape:
                chunks = self._chunk_shape
            elif dataset.chunks:
                chunks = dataset.chunks
            else:
                # Default chunking for non-chunked datasets
                chunks = 'auto'

            # Create dask array from HDF5 dataset
            self._dask_array = da.from_array(dataset, chunks=chunks)

        # Wrap Dask array in TensorStore 'array' driver
        spec = {
            'driver': 'array',
            'array': self._dask_array,
            'schema': {
                'dtype': str(self._dask_array.dtype),
                'shape': list(self._dask_array.shape),
                'dimension_names': self._infer_dimension_names(self._dask_array.shape)
            }
        }

        return spec

    def get_metadata(self) -> Dict:
        """
        Return HDF5 metadata from dataset and file attributes.

        Returns:
            dict: Metadata including HDF5 attributes and voxel sizes if available

        Example:
            >>> reader = HDF5Reader("/data.h5", dataset_path="/volume")
            >>> metadata = reader.get_metadata()
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        dataset = self._open_dataset()

        metadata = {
            'dataset_path': self._dataset_path,
            'shape': dataset.shape,
            'dtype': str(dataset.dtype),
            'chunks': dataset.chunks,
        }

        # Extract dataset attributes
        attrs = {}
        for key in dataset.attrs.keys():
            try:
                value = dataset.attrs[key]
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                attrs[key] = value
            except Exception:
                pass

        metadata['attributes'] = attrs

        # Try to detect unit from common attribute names
        unit_names = ['unit', 'units', 'voxel_unit', 'pixel_unit', 'resolution_unit',
                      'spatial_unit', 'voxel_units', 'pixel_units']
        detected_unit = 'micrometer'  # Default for microscopy data
        for unit_name in unit_names:
            if unit_name in attrs:
                detected_unit = str(attrs[unit_name])
                break
        metadata['voxel_unit'] = detected_unit

        # Try to extract voxel sizes from common attribute names
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
        """
        Return voxel dimensions from HDF5 attributes.

        Returns:
            dict: Voxel dimensions with keys 'x', 'y', 'z' in nanometers

        Example:
            >>> reader = HDF5Reader("/data.h5", dataset_path="/volume")
            >>> voxel_sizes = reader.get_voxel_sizes()

        Notes:
            - Returns 1.0 for each dimension if not found in attributes
            - Searches common attribute names for voxel/pixel size info
            - Auto-detects unit from attributes and converts to nanometers
        """
        from ..utils.format_loaders import convert_to_nanometers

        metadata = self.get_metadata()
        unit = metadata.get('voxel_unit', 'micrometer')

        return {
            'x': convert_to_nanometers(metadata.get('voxel_size_x', 1.0), unit),
            'y': convert_to_nanometers(metadata.get('voxel_size_y', 1.0), unit),
            'z': convert_to_nanometers(metadata.get('voxel_size_z', 1.0), unit)
        }

    def _infer_dimension_names(self, shape):
        """Infer dimension names from array shape."""
        ndim = len(shape)
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

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        """String representation of HDF5 reader."""
        return f"HDF5Reader(path='{self.path}', dataset_path='{self._dataset_path}')"
