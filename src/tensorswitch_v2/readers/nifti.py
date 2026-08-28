"""
NIfTI (.nii / .nii.gz) reader implementation wrapping load_nifti_stack.

Tier 2 reader - reuses the same DaskReader pattern as the TIFF/IMS readers.

NIfTI-1 and NIfTI-2 are supported via nibabel. Axis order is normalised from
NIfTI's native (x, y, z) column-major layout to TensorSwitch's (z, y, x)
convention inside load_nifti_stack(), so downstream code treats NIfTI sources
identically to every other format.
"""

from typing import Dict, List, Optional

# Import utility functions from v2 utils (independent from v1)
from ..utils import load_nifti_stack, extract_nifti_metadata
from .base import DaskReader, _default_voxel_sizes


class NIfTIReader(DaskReader):
    """
    Reader for NIfTI format using nibabel.

    Wraps load_nifti_stack() which returns a lazily-read Dask array in
    (z, y, x) order. DaskReader base class wraps that via ts.virtual_chunked
    for a uniform TensorStore API.

    Tier: 2 (Custom Optimized - Production Ready)
    - nibabel is pure-Python (numpy only), no compiled/native dependencies
    - Reads via nibabel's dataobj proxy: lazy, and preserves on-disk dtype
      (get_fdata() would eagerly upcast everything to float64)
    - Handles both .nii and gzipped .nii.gz transparently

    Voxel size note: NIfTI's header defines spatial units of meter/mm/micron
    only -- there is no nanometer code -- so EM-scale data very often carries a
    unit-less or meaningless ``pixdim``. get_voxel_sizes() returns the header
    value only when it declares a real unit AND converts to a plausible
    microscopy scale; otherwise it falls back to the shared default and warns,
    so the caller is expected to pass --voxel_size explicitly.

    Example:
        >>> from tensorswitch_v2.readers import NIfTIReader
        >>> reader = NIfTIReader("/path/to/volume.nii.gz")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._metadata_cache = None
        self._voxel_sizes_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the NIfTI data and extract dimension names."""
        if self._dask_array is not None:
            return

        self._dask_array, self._dimension_names = load_nifti_stack(self.path)

    def get_metadata(self) -> Dict:
        """Return NIfTI metadata (shape, dtype, header fields)."""
        if self._metadata_cache is None:
            self._load()
            try:
                info, voxel_sizes = extract_nifti_metadata(self.path)
                self._voxel_sizes_cache = voxel_sizes
                # info['shape'] is the source's native (x, y, z) order; the
                # array this reader actually produces is transposed to
                # (z, y, x). Report the transposed shape as 'shape' (matching
                # every other reader) and keep the native one under a distinct
                # key -- spreading info last would silently report the wrong
                # axis order to callers.
                self._metadata_cache = {
                    **info,
                    'source_shape_xyz': info.get('shape'),
                    'shape': tuple(self._dask_array.shape),
                    'dtype': str(self._dask_array.dtype),
                }
            except Exception as e:
                print(f"Warning: Failed to extract NIfTI metadata: {e}")
                self._metadata_cache = {
                    'shape': tuple(self._dask_array.shape),
                    'dtype': str(self._dask_array.dtype),
                }

        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """Return voxel dimensions in nanometers.

        Falls back to the shared default when the NIfTI header does not carry a
        trustworthy physical scale (see class docstring) -- pass --voxel_size.
        """
        self.get_metadata()  # populates _voxel_sizes_cache

        if self._voxel_sizes_cache:
            return self._voxel_sizes_cache

        return _default_voxel_sizes("NIfTI")

    def __repr__(self) -> str:
        return f"NIfTIReader(path='{self.path}')"
