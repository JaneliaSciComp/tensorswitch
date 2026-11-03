"""
MRC / CCP4 (.mrc, .mrcs, .rec, .ali, .st, .map, .rec.nad) reader wrapping mrcfile.

Tier 2 reader - same DaskReader pattern as the TIFF/NIfTI readers.

MRC is the standard format for cryo-ET tomograms and many EM volumes. The data
block is already C-ordered (z, y, x), so no transpose is needed. The header
stores voxel size in angstroms; get_voxel_sizes() converts it to nanometers.
"""

import warnings
from typing import Dict, List, Optional

from .base import DaskReader

_ANGSTROM_TO_NM = 0.1
# Voxel sizes of exactly 1.0 A on every axis are what tools write when the
# volume was never calibrated, so they are treated as "no metadata".
_PLACEHOLDER_ANGSTROM = 1.0
# Below this the value is almost certainly micrometers written into the
# angstrom field (a 0.0926 um light-sheet pixel reads as 0.0926 A).
_MIN_PLAUSIBLE_ANGSTROM = 0.25


class MRCReader(DaskReader):
    """
    Reader for MRC files using mrcfile (memory-mapped, lazy).

    Supports 2D images and 3D volumes / image stacks. 4D volume stacks
    (ISPG 401) and complex modes are rejected rather than guessed at.

    Voxel size note: a header voxel size that is zero, exactly 1.0 A on every
    axis, or below 0.25 A is treated as missing (has_voxel_metadata() is False),
    so the converter asks for --voxel_size instead of writing a wrong scale.
    Light-microscopy MRCs (e.g. BioSR) often store micrometers in the angstrom
    field; values above the 0.25 threshold cannot be told apart from real
    angstroms, so callers that know the true voxel size should always pass it.

    Example:
        >>> from tensorswitch_v2.readers import MRCReader
        >>> reader = MRCReader("/path/to/tomogram.mrc")
        >>> store = reader.get_tensorstore()
    """

    def __init__(self, path: str):
        super().__init__(path)
        self._mrc = None
        self._metadata_cache = None
        self._dimension_names: Optional[List[str]] = None

    def _open(self):
        if self._mrc is None:
            import mrcfile

            self._mrc = mrcfile.mmap(self.path, mode="r", permissive=True)
        return self._mrc

    def _load(self):
        """Lazy-load the MRC data block as a dask array."""
        if self._dask_array is not None:
            return

        import dask.array as da

        mrc = self._open()
        data = mrc.data
        if data.ndim not in (2, 3):
            raise ValueError(
                f"MRC file has {data.ndim} dimensions (ISPG={int(mrc.header.ispg)}); "
                f"only 2D images and 3D volumes/stacks are supported: {self.path}"
            )
        if data.dtype.kind == "c":
            raise ValueError(f"Complex-valued MRC data (mode {int(mrc.header.mode)}) is not supported: {self.path}")

        self._dimension_names = list("zyx"[-data.ndim:])
        self._dask_array = da.from_array(data, chunks="auto")

    def get_metadata(self) -> Dict:
        """Return MRC header fields (shape, dtype, mode, ISPG, voxel size in angstroms)."""
        if self._metadata_cache is None:
            self._load()
            header = self._open().header
            voxel = self._open().voxel_size
            self._metadata_cache = {
                "shape": tuple(self._dask_array.shape),
                "dtype": str(self._dask_array.dtype),
                "mrc_mode": int(header.mode),
                "mrc_ispg": int(header.ispg),
                "voxel_size_angstrom": {
                    "x": float(voxel.x),
                    "y": float(voxel.y),
                    "z": float(voxel.z),
                },
            }
        return self._metadata_cache

    def _header_axes(self) -> List[str]:
        return ["x", "y"] if self._dask_array.ndim == 2 else ["x", "y", "z"]

    def _voxel_problem(self) -> Optional[str]:
        """Why the header voxel size cannot be trusted, or None if it can."""
        meta = self.get_metadata()["voxel_size_angstrom"]
        required = [meta[a] for a in self._header_axes()]
        if any(v <= 0 for v in required):
            return "the header voxel size is zero"
        if all(v == _PLACEHOLDER_ANGSTROM for v in required):
            return "the header voxel size is the uncalibrated 1.0 A default"
        if any(v < _MIN_PLAUSIBLE_ANGSTROM for v in required):
            return (
                f"the header voxel size {required} A is implausibly small "
                f"(micrometers written into the angstrom field?)"
            )
        return None

    def _read_voxel_sizes(self) -> Optional[Dict[str, float]]:
        """Header voxel size in nanometers (angstrom values / 10), or None if untrusted."""
        problem = self._voxel_problem()
        if problem:
            warnings.warn(f"MRC voxel size ignored: {problem}.", stacklevel=2)
            return None

        meta = self.get_metadata()["voxel_size_angstrom"]
        return {a: meta[a] * _ANGSTROM_TO_NM for a in self._header_axes()}

    def __del__(self):
        mrc = getattr(self, "_mrc", None)
        if mrc is not None:
            try:
                mrc.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"MRCReader(path='{self.path}')"
