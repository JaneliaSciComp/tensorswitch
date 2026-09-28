"""
PNG Z-stack reader.

Tier 2 reader - wraps load_png_stack() and exposes it through the uniform
TensorStore API like every other reader.

Deliberately NOT folded into TiffReader. The two look similar (both stack 2D
slices from a directory) but differ in the ways that matter:

  - TIFF carries voxel size, axis order and OME-XML in the file. PNG carries
    NONE of that -- no resolution, no axes, no units -- so --voxel_size is
    mandatory rather than a fallback, and dimension names are fixed by
    construction instead of read.
  - A single TIFF can be a 3D volume; a PNG never can. A PNG "volume" is
    always many files, so the single-file branch is a 2D image, not a stack.
  - PNG stacks are commonly published as a zip (PyTC's EM30 ships 1040 slices
    in one 24 GB archive), which TIFF stacks are not.

Keeping them apart means TiffReader's metadata paths stay honest about what
TIFF actually provides, and PNG's "there is no metadata here" is explicit.

Example:
    >>> from tensorswitch_v2.readers import PngReader
    >>> reader = PngReader("/path/to/slices/")        # or ".../stack.zip"
    >>> store = reader.get_tensorstore()
"""

import os
from typing import Dict, List, Optional

from ..utils import load_png_stack
from .base import DaskReader, _default_voxel_sizes


class PngReader(DaskReader):
    """
    Reader for PNG Z-stacks (directory of slices, zip of slices, or one PNG).

    Tier: 2 (Custom Optimized)
    - Directory or zip of 2D PNG slices -> lazy (Z, Y, X) dask array
    - Single PNG -> (Y, X)
    - Natural numeric sort, so unpadded names (im0, im1, ... im1039) stack in
      the right order rather than im0, im1, im10, im100
    - No voxel size is available from PNG: always requires --voxel_size
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _load(self):
        """Lazy-load the PNG stack and fix dimension names by construction."""
        if self._dask_array is not None:
            return

        self._dask_array = load_png_stack(self.path)

        # PNG has no axes metadata to read, so the names follow from the shape:
        # a stack is (z, y, x); a lone slice is (y, x); an RGB/RGBA slice adds
        # a channel axis, which stays 'c' so downstream code does not mistake
        # it for a spatial dimension.
        ndim = self._dask_array.ndim
        if ndim == 2:
            self._dimension_names = ['y', 'x']
        elif ndim == 3:
            # (z, y, x) for a stack; (y, x, c) for a single colour slice
            if os.path.isfile(self.path) and self.path.lower().endswith('.png'):
                self._dimension_names = ['y', 'x', 'c']
            else:
                self._dimension_names = ['z', 'y', 'x']
        elif ndim == 4:
            self._dimension_names = ['z', 'y', 'x', 'c']
        else:
            self._dimension_names = None

    def get_metadata(self) -> Dict:
        """Shape and dtype only -- PNG has no scientific metadata to report."""
        if self._metadata_cache is None:
            self._load()
            self._metadata_cache = {
                'shape': tuple(self._dask_array.shape),
                'dtype': str(self._dask_array.dtype),
                'source_format': 'png',
                'n_slices': (self._dask_array.shape[0]
                             if self._dask_array.ndim >= 3 else 1),
            }
        return self._metadata_cache

    def get_voxel_sizes(self) -> Dict[str, float]:
        """Always the default + warning: PNG stores no voxel size, ever.

        PNG's optional pHYs chunk records pixels-per-metre for *printing* and
        is absent from every scientific export seen in practice; trusting it
        would be worse than admitting there is nothing here. Pass --voxel_size.
        """
        return _default_voxel_sizes("PNG")

    def __repr__(self) -> str:
        return f"PngReader(path='{self.path}')"
