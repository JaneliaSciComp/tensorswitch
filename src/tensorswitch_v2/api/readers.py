"""
Readers factory class with static methods for creating format-specific readers.

Provides both auto-detection and explicit reader selection via static methods.
"""

import os
from typing import Optional
from ..readers.base import BaseReader, is_local_precomputed as _is_local_precomputed


class Readers:
    """
    Static factory class for creating format-specific readers.

    Provides two usage patterns:
    1. Auto-detection: Intelligent tier selection based on file extension
    2. Explicit selection: Direct reader creation via static methods

    Architecture Layer: 2 (API/Factory)
    - Orchestrates reader selection
    - Implements hybrid three-tier strategy
    - Provides clean API for users

    Hybrid Three-Tier Strategy:
        Tier 1 (Native TensorStore): N5, Zarr2/3, Precomputed
            → Maximum performance, zero conversion overhead
        Tier 2 (Custom Optimized): TIFF, ND2, IMS, HDF5
            → Reuse existing code, minimal overhead
        Tier 3 (BIOIO Adapter): CZI, LIF, + 20 more formats
            → Broad compatibility via BIOIO ecosystem

    Example (Auto-detection):
        >>> from tensorswitch_v2.api import Readers, TensorSwitchDataset
        >>> reader = Readers.auto_detect("/path/to/data.tif")
        >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)

    Example (Explicit reader):
        >>> reader = Readers.n5("/path/to/data.n5")
        >>> dataset = TensorSwitchDataset("/path/to/data.n5", reader=reader)

    Example (Force BIOIO for testing):
        >>> reader = Readers.bioio("/path/to/data.tif")
        >>> dataset = TensorSwitchDataset("/path/to/data.tif", reader=reader)
    """

    @staticmethod
    def auto_detect(path: str) -> BaseReader:
        """
        Automatically detect format and select optimal reader tier.

        Implements intelligent tier selection:
        1. Check for Tier 1 formats (native TensorStore) → fastest
        2. Check for Tier 2 formats (custom optimized) → production-ready
        3. Fallback to Tier 3 (BIOIO adapter) → broad compatibility

        Args:
            path: Path to input data (local, HTTP, GCS, S3)

        Returns:
            BaseReader: Appropriate reader instance for the format

        Raises:
            ValueError: If format cannot be determined
            NotImplementedError: If reader not yet implemented (Week 1-2)

        Example:
            >>> reader = Readers.auto_detect("/data.zarr")
            >>> # Returns Zarr3Reader (Tier 1 - native TensorStore)

            >>> reader = Readers.auto_detect("/data.tif")
            >>> # Returns TiffReader (Tier 2 - custom optimized)

            >>> reader = Readers.auto_detect("/data.czi")
            >>> # Returns BIOIOReader (Tier 3 - BIOIO adapter)

        Tier Selection Logic:
            Tier 1 (Maximum Performance):
            - .n5 → N5Reader
            - .zarr (Zarr3) → Zarr3Reader
            - .zarr (Zarr2) → Zarr2Reader
            - precomputed:// → PrecomputedReader

            Tier 2 (Production Formats):
            - .tif, .tiff → TiffReader
            - .nd2 → ND2Reader
            - .ims → IMSReader
            - .h5, .hdf5 → HDF5Reader

            Tier 3 (Broad Compatibility):
            - .czi, .lif, .sldy, .dv, etc. → BIOIOReader
            - Unknown formats → BIOIOReader (fallback)

        Notes:
            - Extensions checked in order of priority (Tier 1 → 2 → 3)
            - Case-insensitive extension matching
            - For .zarr, checks for Zarr3 vs Zarr2 version
            - User can override with explicit reader if needed
        """
        from ..readers.base import is_remote_path as _is_remote
        path_lower = path.lower()

        # Remote URLs: infer format from URL pattern (no filesystem access)
        if _is_remote(path):
            if path_lower.endswith('.zarr') or '.zarr/' in path_lower:
                return _remote_zarr_reader(path)
            elif path_lower.endswith('.n5') or '.n5/' in path_lower:
                return _remote_n5_reader(path)
            elif 'precomputed://' in path_lower:
                return Readers.precomputed(path)
            else:
                # Default to zarr3 for unrecognized remote paths
                return _remote_zarr_reader(path)

        # Tier 1: Native TensorStore (maximum performance)
        if path_lower.endswith('.n5'):
            return Readers.n5(path)
        elif path_lower.endswith('.zarr'):
            # Distinguish Zarr3 vs Zarr2
            if _is_zarr3(path):
                return Readers.zarr3(path)
            else:
                return Readers.zarr2(path)
        elif 'precomputed://' in path_lower or path_lower.endswith('.precomputed'):
            return Readers.precomputed(path)
        elif _is_local_precomputed(path):
            # Local precomputed directory (has info file with @type)
            return Readers.precomputed(path)

        # Tier 2: Custom Optimized (production formats)
        elif path_lower.endswith(('.tif', '.tiff')):
            return Readers.tiff(path)
        elif path_lower.endswith('.nd2'):
            return Readers.nd2(path)
        elif path_lower.endswith('.ims'):
            return Readers.ims(path)
        elif path_lower.endswith(('.h5', '.hdf5')):
            return Readers.hdf5(path)
        elif path_lower.endswith('.czi'):
            # Prefer Tier 2 CZIReader (pylibCZIrw, multi-view support)
            # Fall back to BIOIO for simpler CZI files or if pylibCZIrw unavailable
            try:
                from ..readers.czi import CZIReader
                return CZIReader(path)
            except ImportError:
                return Readers.bioio(path)

        # Directory: check for zarr/n5 marker files before TIFF fallback
        elif os.path.isdir(path):
            # Zarr2 array (has .zarray at root)
            if os.path.exists(os.path.join(path, '.zarray')):
                return Readers.zarr2(path)
            # Zarr2 group (has .zgroup) — detect zarr3 vs zarr2
            if os.path.exists(os.path.join(path, '.zgroup')):
                if _is_zarr3(path):
                    return Readers.zarr3(path)
                return Readers.zarr2(path)
            # Zarr3 (has zarr.json)
            if os.path.exists(os.path.join(path, 'zarr.json')):
                return Readers.zarr3(path)
            # N5 (has attributes.json)
            if os.path.exists(os.path.join(path, 'attributes.json')):
                return Readers.n5(path)
            # TIFF Z-stack directory
            from ..utils.format_loaders import is_tiff_zstack_directory
            if is_tiff_zstack_directory(path):
                return Readers.tiff(path)
            return Readers.bioio(path)

        # Tier 3: BIOIO Adapter (broad compatibility)
        else:
            return Readers.bioio(path)

    # ========================================================================
    # Tier 1: Native TensorStore Readers (Week 3-4)
    # ========================================================================

    @staticmethod
    def n5(path: str, dataset_path: str = "") -> BaseReader:
        """
        Create N5 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to N5 dataset
            dataset_path: Optional path to dataset within N5 (e.g., "s0" for scale 0)

        Returns:
            N5Reader instance

        Example (single-scale):
            >>> reader = Readers.n5("/data.n5")

        Example (multi-scale):
            >>> reader = Readers.n5("/data.n5", dataset_path="s0")

        Implementation Status:
            ✅ Complete (Week 3-4)
        """
        from ..readers.n5 import N5Reader
        return N5Reader(path, dataset_path=dataset_path)

    @staticmethod
    def zarr3(path: str, dataset_path: str = "") -> BaseReader:
        """
        Create Zarr3 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to Zarr3 dataset
            dataset_path: Path within store to specific array (e.g., "s0")

        Returns:
            Zarr3Reader instance

        Example:
            >>> reader = Readers.zarr3("/data.zarr")
            >>> reader = Readers.zarr3("/data.zarr", dataset_path="s0")

        Implementation Status:
            ✅ Complete (Tier 1 - Native TensorStore)
        """
        from ..readers.zarr import Zarr3Reader
        return Zarr3Reader(path, dataset_path=dataset_path)

    @staticmethod
    def zarr2(path: str, dataset_path: str = "") -> BaseReader:
        """
        Create Zarr2 reader (Tier 1 - Native TensorStore).

        Args:
            path: Path to Zarr2 dataset
            dataset_path: Path within store to specific array (e.g., "0")

        Returns:
            Zarr2Reader instance

        Example:
            >>> reader = Readers.zarr2("/data.zarr")
            >>> reader = Readers.zarr2("/data.zarr", dataset_path="0")

        Implementation Status:
            ✅ Complete (Tier 1 - Native TensorStore)
        """
        from ..readers.zarr import Zarr2Reader
        return Zarr2Reader(path, dataset_path=dataset_path)

    @staticmethod
    def precomputed(path: str, scale_index: int = 0) -> BaseReader:
        """
        Create Neuroglancer Precomputed reader (Tier 1 - Native TensorStore).

        Args:
            path: Path or URL to Precomputed dataset
            scale_index: Which resolution level to read (0 = highest resolution)

        Returns:
            PrecomputedReader instance

        Example:
            >>> reader = Readers.precomputed("precomputed://gs://bucket/data")
            >>> reader = Readers.precomputed("precomputed://gs://bucket/data", scale_index=1)

        Implementation Status:
            ✅ Complete (Week 3-4)
        """
        from ..readers.precomputed import PrecomputedReader
        return PrecomputedReader(path, scale_index=scale_index)

    # ========================================================================
    # Tier 2: Custom Optimized Readers (Week 5-6)
    # ========================================================================

    @staticmethod
    def tiff(path: str) -> BaseReader:
        """
        Create TIFF reader (Tier 2 - Custom Optimized).

        Reuses existing load_tiff_stack() from utils.py.

        Args:
            path: Path to TIFF file or directory

        Returns:
            TiffReader instance

        Example:
            >>> reader = Readers.tiff("/data.tif")
            >>> reader = Readers.tiff("/data/stack/")

        Implementation Status:
            ✅ Complete
        """
        from ..readers.tiff import TiffReader
        return TiffReader(path)

    @staticmethod
    def nd2(path: str) -> BaseReader:
        """
        Create ND2 reader (Tier 2 - Custom Optimized).

        Reuses existing load_nd2_stack() from utils.py.

        Args:
            path: Path to ND2 file

        Returns:
            ND2Reader instance

        Example:
            >>> reader = Readers.nd2("/data.nd2")

        Implementation Status:
            ✅ Complete (Tier 2 - reuses load_nd2_stack())
        """
        from ..readers.nd2 import ND2Reader
        return ND2Reader(path)

    @staticmethod
    def ims(path: str, resolution_level: int = 0) -> BaseReader:
        """
        Create IMS reader (Tier 2 - Custom Optimized).

        Reuses existing load_ims_stack() from utils.py.

        Args:
            path: Path to IMS file
            resolution_level: Which resolution level to read (0 = highest)

        Returns:
            IMSReader instance

        Example:
            >>> reader = Readers.ims("/data.ims")

        Implementation Status:
            ✅ Complete (Tier 2 - reuses load_ims_stack())
        """
        from ..readers.ims import IMSReader
        return IMSReader(path, resolution_level=resolution_level)

    @staticmethod
    def hdf5(path: str, dataset_path: Optional[str] = None) -> BaseReader:
        """
        Create HDF5 reader (Tier 2 - Custom Optimized).

        Args:
            path: Path to HDF5 file
            dataset_path: Path to dataset within HDF5 (auto-detected if None)

        Returns:
            HDF5Reader instance

        Example:
            >>> reader = Readers.hdf5("/data.h5")
            >>> reader = Readers.hdf5("/data.h5", dataset_path="/volume")

        Implementation Status:
            ✅ Complete (Tier 2 - uses h5py + dask)
        """
        from ..readers.hdf5 import HDF5Reader
        return HDF5Reader(path, dataset_path=dataset_path)

    @staticmethod
    def czi(path: str, view_index: Optional[int] = None) -> BaseReader:
        """
        Create CZI reader (Tier 2 - Custom Optimized).

        Reuses existing load_czi_stack() from utils.py via pylibCZIrw.
        Supports multi-view CZI files (V dimension → 5D VCZYX).

        Args:
            path: Path to CZI file
            view_index: Optional specific view to load. If None and multiple
                       views exist, loads all views as 5D VCZYX array.

        Returns:
            CZIReader instance

        Example:
            >>> reader = Readers.czi("/data.czi")
            >>> reader = Readers.czi("/data.czi", view_index=0)

        Implementation Status:
            ✅ Complete (Tier 2 - reuses load_czi_stack())
        """
        from ..readers.czi import CZIReader
        return CZIReader(path, view_index=view_index)

    # ========================================================================
    # Tier 3: BIOIO Adapter (Week 7)
    # ========================================================================

    @staticmethod
    def bioio(
        path: str,
        scene_index: int = 0,
        channel_index: Optional[int] = None,
        time_index: Optional[int] = None,
        resolution_level: int = 0,
        reader: Optional[object] = None
    ) -> BaseReader:
        """
        Create BIOIO adapter reader (Tier 3 - Broad Compatibility).

        Strategic investment: One adapter unlocks 20+ formats.
        Supports: CZI, LIF, SLDY, DV, and all other BIOIO formats.

        Args:
            path: Path to input file
            scene_index: Which scene to load for multi-scene files (default: 0)
            channel_index: Optional specific channel to extract (None = all)
            time_index: Optional specific timepoint to extract (None = all)
            resolution_level: Resolution level for pyramid formats (0=full, default: 0)
            reader: Optional explicit BIOIO reader class (auto-detects if None)

        Returns:
            BIOIOReader instance

        Example:
            >>> # Auto-detect format
            >>> reader = Readers.bioio("/data.czi")

            >>> # Specific scene in multi-scene file
            >>> reader = Readers.bioio("/data.lif", scene_index=2)

            >>> # Extract single channel
            >>> reader = Readers.bioio("/data.czi", channel_index=0)

            >>> # Read lower resolution pyramid level
            >>> reader = Readers.bioio("/data.czi", resolution_level=2)

        Supported Formats:
            CZI, LIF, SLDY, DV, OME-TIFF, and 20+ more via BIOIO plugins.
            See https://github.com/bioio-devs/bioio for full list.

        Implementation Status:
            ✅ Complete - Strategic unlock: 1 adapter (~200 LOC) = 20+ formats
        """
        from ..readers.bioio_adapter import BIOIOReader
        return BIOIOReader(
            path,
            scene_index=scene_index,
            channel_index=channel_index,
            time_index=time_index,
            resolution_level=resolution_level,
            reader=reader
        )

    # ========================================================================
    # Tier 4: Bio-Formats (Java-backed, 150+ formats)
    # ========================================================================

    @staticmethod
    def bioformats(
        path: str,
        scene_index: int = 0,
        channel_index: Optional[int] = None,
        time_index: Optional[int] = None,
        resolution_level: int = 0,
    ) -> BaseReader:
        """
        Create Bio-Formats reader (Tier 4 - Maximum Format Compatibility).

        Uses the OME Bio-Formats Java library via bioio-bioformats plugin.
        Supports 150+ file formats including many proprietary vendor formats.

        Note: Requires Java and bioio-bioformats to be installed:
            conda install -c conda-forge scyjava
            pip install bioio-bioformats

        Args:
            path: Path to input file (local filesystem only)
            scene_index: Which scene to load for multi-scene files (default: 0)
            channel_index: Optional specific channel to extract (None = all)
            time_index: Optional specific timepoint to extract (None = all)
            resolution_level: Resolution level for pyramid formats (0=full)

        Returns:
            BioFormatsReader instance

        Example:
            >>> # Read Olympus VSI (only supported by Bio-Formats)
            >>> reader = Readers.bioformats("/data/slide.vsi")

            >>> # Read with specific scene
            >>> reader = Readers.bioformats("/data/multi.lif", scene_index=2)

        When to use bioformats() vs bioio():
            - bioio(): Pure Python plugins, no Java required, ~20 common formats
            - bioformats(): Java-backed, 150+ formats including obscure/legacy ones

        Formats uniquely supported by Bio-Formats:
            - Olympus VSI, OIB, OIF
            - Leica SCN
            - Volocity
            - Imspector OBF
            - Many legacy vendor formats
            - See: https://bio-formats.readthedocs.io/en/latest/supported-formats.html

        Implementation Status:
            ✅ Complete - Wraps BIOIOReader with Bio-Formats Java backend
        """
        from ..readers.bioformats import BioFormatsReader
        return BioFormatsReader(
            path,
            scene_index=scene_index,
            channel_index=channel_index,
            time_index=time_index,
            resolution_level=resolution_level,
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _is_zarr3(path: str) -> bool:
    """
    Check if Zarr dataset is Zarr3 vs Zarr2.

    Zarr3 has a zarr.json file, Zarr2 has .zarray/.zgroup files.

    Args:
        path: Path to Zarr dataset

    Returns:
        bool: True if Zarr3, False if Zarr2

    Detection Logic:
        1. Check for zarr.json at root → Zarr3
        2. Check for zarr.json in subdirectories (s0/, 0/) → Zarr3
        3. Check for .zarray or .zgroup → Zarr2
        4. Default to Zarr2 if unclear
    """
    import os

    # Check root for zarr.json (Zarr3 indicator)
    if os.path.exists(os.path.join(path, 'zarr.json')):
        return True

    # Check common subdirectories for zarr.json (multiscale Zarr3)
    for subdir in ['s0', '0', 'data']:
        subpath = os.path.join(path, subdir, 'zarr.json')
        if os.path.exists(subpath):
            return True

    # Check for Zarr2 indicators
    if os.path.exists(os.path.join(path, '.zarray')):
        return False
    if os.path.exists(os.path.join(path, '.zgroup')):
        return False

    # Check subdirectories for Zarr2
    for subdir in ['s0', '0', 'data']:
        if os.path.exists(os.path.join(path, subdir, '.zarray')):
            return False

    # Default to Zarr2 for backwards compatibility
    return False


def _s3_discover_array_path(path: str) -> str:
    """Use S3 bounded directory listing to find the first array in a container.

    Performs a breadth-first search using S3 ``ListObjectsV2`` with
    ``delimiter='/'`` to discover subdirectories, and direct GETs to check
    for array markers at each node:

    - ``.zarray`` exists → Zarr2 array
    - ``zarr.json`` with ``node_type == 'array'`` → Zarr3 array
    - ``attributes.json`` with ``dataType`` → N5 array

    Array markers are checked via direct HTTP GET — not via the listing
    response — because S3's ``max-keys`` can truncate metadata files when
    there are many chunk subdirectories (e.g. N5 ``s0/`` with 188 chunks).

    Returns the relative subpath to the first array found, or ``''`` if
    the URL is not S3 or no array is found within the search bounds.

    Bounds: max depth 4, max 10 children per level.
    """
    from ..readers.base import parse_s3_url, s3_list_children
    import urllib.request
    import json

    parsed = parse_s3_url(path)
    if parsed is None:
        return ''

    bucket, root_prefix = parsed
    base_url = f"https://{bucket}.s3.amazonaws.com/"

    def _fetch_json(key: str):
        """Fetch and parse a JSON file from S3.  Return dict or None."""
        try:
            with urllib.request.urlopen(base_url + key, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def _is_array(prefix: str) -> bool:
        """Check array markers via direct GET (not from listing)."""
        # Zarr2: .zarray exists
        try:
            req = urllib.request.Request(base_url + prefix + '.zarray', method='HEAD')
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            pass
        # Zarr3: zarr.json with node_type=array
        meta = _fetch_json(prefix + 'zarr.json')
        if meta and meta.get('node_type') == 'array':
            return True
        # N5: attributes.json with dataType
        meta = _fetch_json(prefix + 'attributes.json')
        if meta and 'dataType' in meta:
            return True
        return False

    MAX_DEPTH = 4
    MAX_CHILDREN = 10

    # Prioritise directories likely to contain raw image data so that BFS
    # finds them before exploring dozens of label subdirectories.
    _PRIORITY = {
        's0': 0, 'em': 1, 'raw': 2, 'data': 3, 'volumes': 4,
        'recon-1': 5, 'setup0': 6,
    }

    def _sort_children(dirs):
        return sorted(dirs, key=lambda d: _PRIORITY.get(d, 100))

    # BFS: each entry is (prefix, depth, relative_subpath)
    queue = [(root_prefix, 0, '')]

    try:
        while queue:
            prefix, depth, rel = queue.pop(0)

            # Skip root (depth 0) — caller already checked it
            if depth > 0 and _is_array(prefix):
                return rel

            if depth >= MAX_DEPTH:
                continue

            dirs, _files = s3_list_children(bucket, prefix)

            for child in _sort_children(dirs)[:MAX_CHILDREN]:
                child_prefix = prefix + child + '/'
                child_rel = f"{rel}/{child}" if rel else child
                queue.append((child_prefix, depth + 1, child_rel))
    except Exception:
        pass

    return ''


def _remote_zarr_reader(path: str):
    """Create the correct Zarr reader for a remote URL.

    Detects Zarr3 vs Zarr2.  Groups with OME-NGFF ``multiscales`` metadata
    are auto-resolved to the first array.  Groups without multiscales raise
    ``ValueError`` asking the user to provide the dataset sub-path.
    """
    from ..readers.base import build_kvstore
    import tensorstore as ts
    import json

    try:
        kvs = ts.KvStore.open(build_kvstore(path)).result()

        # Check for Zarr3 (zarr.json)
        result = kvs.read('zarr.json').result()
        if result.value is not None and len(result.value) > 0:
            meta = json.loads(bytes(result.value))
            # If it's a group with multiscales, find the first dataset path
            if 'node_type' in meta and meta['node_type'] == 'group':
                ds_path = _find_first_dataset_path(meta)
                if ds_path:
                    return Readers.zarr3(path, dataset_path=ds_path)
                # Try S3 bounded directory listing before giving up
                resolved = _s3_discover_array_path(path)
                if resolved:
                    return Readers.zarr3(path, dataset_path=resolved)
                raise ValueError(
                    f"Remote Zarr3 path is a group without multiscales metadata: {path}\n"
                    "Append the dataset sub-path to the URL, e.g.:\n"
                    f"  {path}/raw/s0"
                )
            return Readers.zarr3(path)

        # Check for Zarr2 group (.zgroup + .zattrs with multiscales)
        zgroup = kvs.read('.zgroup').result()
        if zgroup.value is not None and len(zgroup.value) > 0:
            # It's a zarr2 group — check .zattrs for multiscales to find first array
            zattrs = kvs.read('.zattrs').result()
            if zattrs.value is not None and len(zattrs.value) > 0:
                attrs = json.loads(bytes(zattrs.value))
                ds_path = _find_first_dataset_path(attrs)
                if ds_path:
                    return Readers.zarr2(path, dataset_path=ds_path)

            # Try S3 bounded directory listing before giving up
            resolved = _s3_discover_array_path(path)
            if resolved:
                return Readers.zarr2(path, dataset_path=resolved)
            raise ValueError(
                f"Remote Zarr path is a group without multiscales metadata: {path}\n"
                "Append the dataset sub-path to the URL, e.g.:\n"
                f"  {path}/recon-1/em/fibsem-uint8/s0"
            )

        # Check for Zarr2 array (.zarray)
        zarray = kvs.read('.zarray').result()
        if zarray.value is not None and len(zarray.value) > 0:
            return Readers.zarr2(path)

    except ValueError:
        raise
    except Exception:
        pass

    # Default fallback
    return Readers.zarr2(path)


def _find_first_dataset_path(metadata: dict) -> str:
    """Extract the first dataset path from OME-NGFF multiscales metadata."""
    # Zarr3: attributes.ome.multiscales or attributes.multiscales
    attrs = metadata.get('attributes', metadata)
    ome = attrs.get('ome', attrs)
    multiscales = ome.get('multiscales', [])
    if multiscales:
        datasets = multiscales[0].get('datasets', [])
        if datasets:
            return datasets[0].get('path', '')
    return ''


def _remote_n5_reader(path: str):
    """Create N5 reader for a remote URL.

    Checks whether the root path is an N5 array (has ``dataType`` in
    ``attributes.json``).  If the root is a group, raises ``ValueError``
    telling the user to append the dataset sub-path.
    """
    from ..readers.base import build_kvstore
    import tensorstore as ts
    import json

    try:
        kvs = ts.KvStore.open(build_kvstore(path)).result()
        result = kvs.read("attributes.json").result()
        if result.value is not None and len(result.value) > 0:
            attrs = json.loads(bytes(result.value))
            if "dataType" in attrs:
                return Readers.n5(path)
            # Root is a group — try S3 bounded directory listing
            resolved = _s3_discover_array_path(path)
            if resolved:
                return Readers.n5(path, dataset_path=resolved)
            raise ValueError(
                f"Remote N5 path is a group, not an array: {path}\n"
                "Append the dataset sub-path to the URL, e.g.:\n"
                f"  {path}/em/fibsem-uint16/s0"
            )
    except ValueError:
        raise
    except Exception:
        pass

    return Readers.n5(path)
