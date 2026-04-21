"""
Upsampler for TensorSwitch — resample anisotropic data to isotropic resolution.

Uses scipy.ndimage.zoom for interpolation:
  - order=1 (trilinear) for image data
  - order=0 (nearest-neighbor) for label/segmentation data

Processes data in XY-column chunks to keep memory bounded. Each column reads
the full Z extent, upsamples Z, and writes to the output — no chunk boundary
artifacts because the interpolation axis is never split.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import tensorstore as ts
from scipy.ndimage import zoom

from ..utils.tensorstore_utils import (
    get_zarr_store_spec,
    zarr2_store_spec,
    zarr3_store_spec,
    get_tensorstore_context,
)

# Set team permissions: rwxrwxr-x (files get rw-rw-r--)
os.umask(0o0002)

# Mapping from user-friendly method names to scipy interpolation orders
UPSAMPLE_METHODS = {
    "nearest": 0,       # Labels/segmentation — preserves exact integer IDs
    "trilinear": 1,     # Default for images — no ringing artifacts
    "cubic": 3,         # Smoother, but may ring at sharp edges (membranes)
}

# Default methods by data type (mirrors downsample: mean for images, mode for labels)
DEFAULT_IMAGE_METHOD = "trilinear"
DEFAULT_LABEL_METHOD = "nearest"


def resolve_upsample_method(method: str, is_label: bool = False) -> str:
    """Resolve 'auto' method to a concrete method name.

    Args:
        method: 'auto', 'nearest', 'linear', or 'cubic'.
        is_label: If True, 'auto' resolves to 'nearest'.

    Returns:
        Resolved method name.
    """
    if method == "auto":
        return DEFAULT_LABEL_METHOD if is_label else DEFAULT_IMAGE_METHOD
    if method not in UPSAMPLE_METHODS:
        raise ValueError(
            f"Unknown upsample method '{method}'. "
            f"Options: {', '.join(['auto'] + list(UPSAMPLE_METHODS.keys()))}"
        )
    return method


class Upsampler:
    """Upsample anisotropic data to isotropic resolution along one or more axes.

    Example:
        >>> upsampler = Upsampler(
        ...     input_path="/data/dataset.zarr/img/s0",
        ...     output_path="/data/out.zarr/img/s0",
        ...     target_voxel_sizes=[9.0, 9.0, 9.0],
        ... )
        >>> stats = upsampler.upsample()
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_voxel_sizes: List[float],
        source_voxel_sizes: Optional[List[float]] = None,
        upsample_method: str = "auto",
        is_label: bool = False,
        output_format: str = "auto",
        no_sharding: bool = False,
        shard_shape: Optional[List[int]] = None,
        compression: str = "zstd",
        compression_level: int = 5,
    ):
        """
        Args:
            input_path: Path to source s0 array (e.g., data.zarr/img/s0).
            output_path: Path for output s0 array (e.g., out.zarr/img/s0).
            target_voxel_sizes: Target voxel sizes for spatial axes (in nm).
            source_voxel_sizes: Override source voxel sizes (auto-read from
                OME-NGFF metadata if None).
            upsample_method: Interpolation method.
                "auto" = linear for images, nearest for labels.
                "nearest" = nearest-neighbor (order=0, for labels).
                "trilinear" = trilinear interpolation (order=1, for images).
                "cubic" = cubic spline (order=3, smoother but may ring).
            is_label: If True, 'auto' resolves to 'nearest'.
            output_format: Output format — "auto" (match source), "zarr2", "zarr3".
            no_sharding: Disable sharding for zarr3 output.
            shard_shape: Custom shard shape for zarr3 sharded output.
            compression: Compression codec — "zstd" (default), "gzip", "none".
            compression_level: Compression level (1-22 for zstd, 1-9 for gzip).
        """
        self.input_path = input_path.rstrip("/")
        self.output_path = output_path.rstrip("/")
        self.target_voxel_sizes = target_voxel_sizes
        self._source_voxel_sizes = source_voxel_sizes
        self.is_label = is_label
        self.output_format = output_format
        self.no_sharding = no_sharding
        self.shard_shape = shard_shape
        self.compression = compression
        self.compression_level = compression_level
        # is_label always forces nearest, regardless of explicit method
        if is_label:
            self.upsample_method = "nearest"
        else:
            self.upsample_method = resolve_upsample_method(upsample_method, is_label)
        self.interp_order = UPSAMPLE_METHODS[self.upsample_method]

    def _read_ome_metadata(self) -> Dict[str, Any]:
        """Read OME-NGFF metadata from the parent group of the input array.

        Looks for .zattrs (Zarr2) or zarr.json (Zarr3) at the group level
        (one directory up from s0).
        """
        group_path = os.path.dirname(self.input_path)

        # Try Zarr2 .zattrs
        zattrs_path = os.path.join(group_path, ".zattrs")
        if os.path.exists(zattrs_path):
            with open(zattrs_path) as f:
                attrs = json.load(f)
            return attrs

        # Try Zarr3 zarr.json
        zarr_json_path = os.path.join(group_path, "zarr.json")
        if os.path.exists(zarr_json_path):
            with open(zarr_json_path) as f:
                meta = json.load(f)
            return meta.get("attributes", {}).get("ome", meta.get("attributes", {}))

        return {}

    def _get_source_voxel_sizes_and_axes(self):
        """Extract source voxel sizes and axes from OME-NGFF metadata.

        Returns:
            (voxel_sizes, axes_info): voxel_sizes is a list of floats for
            all axes; axes_info is a list of dicts with 'name' and 'type'.
        """
        if self._source_voxel_sizes is not None:
            # Caller provided explicit voxel sizes — no metadata needed
            return self._source_voxel_sizes, None

        attrs = self._read_ome_metadata()
        multiscales = attrs.get("multiscales", [])
        if not multiscales:
            raise ValueError(
                f"No OME-NGFF multiscales metadata found for {self.input_path}. "
                f"Provide source_voxel_sizes explicitly."
            )

        ms = multiscales[0]
        axes_info = ms.get("axes", [])
        datasets = ms.get("datasets", [])

        # Find s0 entry
        s0_name = os.path.basename(self.input_path)
        scale = None
        for ds in datasets:
            if ds.get("path") == s0_name:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        scale = t["scale"]
                        break
                break

        if scale is None:
            raise ValueError(
                f"Could not find scale transform for '{s0_name}' in OME-NGFF metadata."
            )

        return scale, axes_info

    def _read_zarr2_compressor(self) -> Optional[Dict]:
        """Read compressor dict from source .zarray JSON.

        Returns the compressor in TensorStore-compatible dict format.
        Converts bare 'zstd'/'gzip' to blosc wrapper for TensorStore
        compatibility (TensorStore zarr2 driver + numcodecs).
        """
        zarray_path = os.path.join(self.input_path, ".zarray")
        if os.path.exists(zarray_path):
            with open(zarray_path) as f:
                meta = json.load(f)
            comp = meta.get("compressor")
            if comp and comp.get("id") == "zstd":
                # Bare zstd is incompatible between TensorStore and numcodecs.
                # Convert to blosc(zstd) which both support.
                return {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": comp.get("level", 5),
                    "shuffle": 1,
                    "blocksize": 0,
                }
            return comp
        return None

    def compute_zoom_factors(self) -> List[float]:
        """Compute per-axis zoom factors without running the full upsample.

        Useful for resource estimation and job submission.

        Returns:
            List of zoom factors (1.0 for non-zoomed axes).
        """
        src_voxels, axes_info = self._get_source_voxel_sizes_and_axes()
        spatial_indices = []
        if axes_info:
            for i, ax in enumerate(axes_info):
                if ax.get("type") == "space":
                    spatial_indices.append(i)
        else:
            spatial_indices = list(range(min(3, len(src_voxels))))

        zoom_factors = []
        tgt_idx = 0
        for i in range(len(src_voxels)):
            if i in spatial_indices and tgt_idx < len(self.target_voxel_sizes):
                factor = src_voxels[i] / self.target_voxel_sizes[tgt_idx]
                tgt_idx += 1
            else:
                factor = 1.0
            zoom_factors.append(factor)
        return zoom_factors

    def upsample(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the upsampling.

        Returns:
            dict with keys: input_shape, output_shape, zoom_factors,
            chunks_processed, elapsed_time.
        """
        start_time = time.time()

        # 1. Read source voxel sizes and axes
        src_voxels, axes_info = self._get_source_voxel_sizes_and_axes()

        # 2. Identify spatial axes and compute zoom factors
        spatial_indices = []
        if axes_info:
            for i, ax in enumerate(axes_info):
                if ax.get("type") == "space":
                    spatial_indices.append(i)
        else:
            # No axes info — assume all dims are spatial (up to 3)
            spatial_indices = list(range(min(3, len(src_voxels))))

        # Sanity check: warn if a "spatial" axis has voxel size 1.0 while others
        # are in a real physical range — likely a mislabeled channel/time axis
        # (e.g., TIFF CYX where tifffile reports C as spatial)
        if spatial_indices:
            spatial_voxels = [src_voxels[i] for i in spatial_indices]
            real_voxels = [v for v in spatial_voxels if v > 1.0]
            suspect_voxels = [v for v in spatial_voxels if v == 1.0]
            if suspect_voxels and real_voxels:
                import warnings
                suspect_axes = [
                    axes_info[i]["name"] if axes_info and i < len(axes_info)
                    else f"dim{i}"
                    for i in spatial_indices
                    if src_voxels[i] == 1.0
                ]
                warnings.warn(
                    f"Spatial axis {suspect_axes} has voxel size 1.0 while other "
                    f"spatial axes have sizes {real_voxels}. This may indicate a "
                    f"mislabeled channel or time axis. Check your source metadata. "
                    f"Use --voxel_size to override if needed.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build per-axis zoom factors (1.0 for non-spatial, src/tgt for spatial)
        zoom_factors = []
        tgt_idx = 0
        for i in range(len(src_voxels)):
            if i in spatial_indices and tgt_idx < len(self.target_voxel_sizes):
                factor = src_voxels[i] / self.target_voxel_sizes[tgt_idx]
                tgt_idx += 1
            else:
                factor = 1.0
            zoom_factors.append(factor)

        # 3. Open source array via TensorStore (handles zarr2/zarr3/sharded/remote)
        src_spec = get_zarr_store_spec(self.input_path)
        src = ts.open(src_spec, open=True).result()
        src_shape = list(src.shape)
        src_dtype = src.dtype.numpy_dtype
        src_chunks = list(src.chunk_layout.read_chunk.shape)

        # 4. Compute output shape
        output_shape = [
            round(src_shape[i] * zoom_factors[i]) for i in range(len(src_shape))
        ]

        # 5. Determine output chunks (keep same as source for non-zoomed axes)
        output_chunks = list(src_chunks)

        method_name = "nearest-neighbor" if self.interp_order == 0 else "trilinear"

        if verbose:
            print(f"\n{'=' * 60}")
            print("UPSAMPLING TO ISOTROPIC")
            print(f"{'=' * 60}")
            print(f"Source:       {self.input_path}")
            print(f"Output:       {self.output_path}")
            print(f"Source shape: {src_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Source voxels: {src_voxels}")
            print(f"Target voxels: {self.target_voxel_sizes}")
            print(f"Zoom factors: {[round(f, 4) for f in zoom_factors]}")
            print(f"Method:       {method_name} (order={self.interp_order})")
            print(f"dtype:        {src_dtype}")
            print(f"Chunks:       {output_chunks}")

        # 6. Determine which axes are NOT being zoomed (factor == 1.0)
        #    We iterate over these in chunks and read full extent of zoomed axes.
        non_zoom_axes = [i for i, f in enumerate(zoom_factors) if abs(f - 1.0) < 1e-6]
        zoom_axes = [i for i, f in enumerate(zoom_factors) if abs(f - 1.0) >= 1e-6]

        if not zoom_axes:
            raise ValueError("No axes need upsampling — all zoom factors are 1.0")

        # 7. Create output array via TensorStore (supports zarr2, zarr3, zarr3+sharding)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # Detect source format
        src_dir = self.input_path
        is_zarr2_source = os.path.exists(os.path.join(src_dir, ".zarray"))

        # Resolve output format
        if self.output_format == "auto":
            output_fmt = "zarr2" if is_zarr2_source else "zarr3"
        else:
            output_fmt = self.output_format

        # Build axes order from metadata
        axes_order = None
        if axes_info:
            axes_order = [ax.get("name", f"dim{i}") for i, ax in enumerate(axes_info)]

        # Build compression codec spec
        compression_spec = None
        if self.compression and self.compression != "none":
            compression_spec = {
                'name': self.compression,
                'configuration': {'level': self.compression_level}
            }

        if output_fmt == "zarr2":
            # Read source compressor from .zarray JSON (TensorStore-compatible dict)
            compressor_dict = self._read_zarr2_compressor()
            if compressor_dict is None:
                compressor_dict = {
                    'id': 'blosc', 'cname': 'zstd',
                    'clevel': self.compression_level, 'shuffle': 1, 'blocksize': 0,
                }

            # TensorStore zarr2 needs numpy dtype str (e.g., "|u1", "<u2")
            ts_dtype = np.dtype(src_dtype).str
            spec = zarr2_store_spec(
                self.output_path,
                shape=output_shape,
                chunks=output_chunks,
                axes_order=axes_order,
                dtype=ts_dtype,
                compressor=compressor_dict,
            )
        else:
            # zarr3 — sharded or non-sharded
            use_shard = not self.no_sharding
            # For sharded: output_chunks = inner chunks, shard_shape = outer
            # For non-sharded: output_chunks = chunk shape
            # TensorStore zarr3 uses numpy dtype name (e.g., "uint8", "uint16")
            ts_dtype = np.dtype(src_dtype).name
            spec = zarr3_store_spec(
                path=os.path.dirname(self.output_path),
                shape=output_shape,
                dtype=ts_dtype,
                use_shard=use_shard,
                level_path=os.path.basename(self.output_path),
                custom_chunk_shape=output_chunks,
                custom_shard_shape=self.shard_shape,
                axes_order=axes_order,
                compression=compression_spec,
            )

        spec['context'] = get_tensorstore_context()
        dst = ts.open(spec, create=True, delete_existing=True).result()

        if verbose:
            print(f"Output format: {output_fmt}"
                  f"{' (sharded)' if output_fmt == 'zarr3' and not self.no_sharding else ''}")

        # 8. Process in chunks over non-zoom axes, full extent on zoom axes
        #    For NISB: non_zoom = [0, 1, 3] (x, y, c), zoom = [2] (z)
        #    We chunk over x and y, read full z per chunk.

        # Build chunk grid for non-zoom spatial axes only
        # (skip channel and other non-spatial axes — they're usually small)
        chunk_axes = [i for i in non_zoom_axes
                      if axes_info is None or i >= len(axes_info)
                      or axes_info[i].get("type") == "space"]

        # If no spatial non-zoom axes to chunk over, process whole volume at once
        if not chunk_axes:
            chunk_axes = non_zoom_axes[:2] if len(non_zoom_axes) >= 2 else non_zoom_axes

        # Compute number of chunks per axis
        chunk_grid = {}
        for ax in chunk_axes:
            chunk_grid[ax] = math.ceil(src_shape[ax] / src_chunks[ax])

        # Generate all chunk coordinates
        from itertools import product
        axis_ranges = []
        for ax in chunk_axes:
            axis_ranges.append(range(chunk_grid[ax]))

        total_chunks = 1
        for ax in chunk_axes:
            total_chunks *= chunk_grid[ax]

        if verbose:
            print(f"Chunk axes:   {chunk_axes} ({total_chunks} chunks)")
            print(f"Zoom axes:    {zoom_axes}")
            print(f"{'=' * 60}")

        chunks_processed = 0
        for coords in product(*axis_ranges):
            # Build read/write slices
            read_slices = [slice(None)] * len(src_shape)
            write_slices = [slice(None)] * len(output_shape)

            for j, ax in enumerate(chunk_axes):
                c = coords[j]
                start = c * src_chunks[ax]
                end = min(start + src_chunks[ax], src_shape[ax])
                read_slices[ax] = slice(start, end)
                # For non-zoom axes, output indices match input
                write_slices[ax] = slice(start, end)

            # Read source chunk (full extent on zoom axes)
            slab = src[tuple(read_slices)].read().result()

            # Apply zoom
            upsampled = zoom(
                slab,
                zoom_factors,
                order=self.interp_order,
                grid_mode=True,
                mode="nearest",
            )

            # Clip to dtype range for integer types (trilinear can overshoot)
            if self.interp_order > 0 and np.issubdtype(src_dtype, np.integer):
                info = np.iinfo(src_dtype)
                upsampled = np.clip(upsampled, info.min, info.max)
            upsampled = upsampled.astype(src_dtype)

            # Trim to exact output shape on zoom axes (rounding edge case)
            for ax in zoom_axes:
                actual = upsampled.shape[ax]
                expected = output_shape[ax]
                if actual > expected:
                    trim_slices = [slice(None)] * len(output_shape)
                    trim_slices[ax] = slice(0, expected)
                    upsampled = upsampled[tuple(trim_slices)]

            # Build write slices for zoom axes (full extent)
            for ax in zoom_axes:
                write_slices[ax] = slice(0, upsampled.shape[ax])

            # Write via TensorStore transaction (memory-safe, supports sharding)
            write_domain = tuple(write_slices)
            with ts.Transaction() as txn:
                dst[write_domain].with_transaction(txn).write(
                    np.ascontiguousarray(upsampled)
                ).result()

            chunks_processed += 1
            if verbose and (chunks_processed % 10 == 0 or chunks_processed == total_chunks):
                elapsed = time.time() - start_time
                rate = chunks_processed / elapsed if elapsed > 0 else 0
                eta = (total_chunks - chunks_processed) / rate if rate > 0 else 0
                print(
                    f"  [{chunks_processed}/{total_chunks}] "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nUpsampling complete: {chunks_processed} chunks in {elapsed:.1f}s")

        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "input_shape": src_shape,
            "output_shape": output_shape,
            "zoom_factors": zoom_factors,
            "upsample_method": self.upsample_method,
            "interp_order": self.interp_order,
            "chunks_processed": chunks_processed,
            "elapsed_time": elapsed,
        }


def upsample_to_isotropic(
    input_path: str,
    output_path: str,
    target_voxel_size: Optional[float] = None,
    upsample_method: str = "auto",
    is_label: bool = False,
    verbose: bool = True,
    output_format: str = "auto",
    no_sharding: bool = False,
    shard_shape: Optional[List[int]] = None,
    compression: str = "zstd",
    compression_level: int = 5,
) -> Dict[str, Any]:
    """Upsample anisotropic s0 to isotropic resolution.

    Reads source voxel sizes from OME-NGFF metadata, computes zoom factors
    to make all spatial axes match the target resolution, then writes the
    upsampled data via TensorStore (supports zarr2, zarr3, zarr3+sharding).

    Args:
        input_path: Path to source s0 array (e.g., data.zarr/img/s0).
        output_path: Path for output s0 array (e.g., out.zarr/img/s0).
        target_voxel_size: Target isotropic voxel size in nm. If None,
            uses the smallest source voxel size (highest resolution axis).
        upsample_method: Interpolation method.
            "auto" = trilinear for images, nearest for labels.
            "nearest" = nearest-neighbor (for labels/segmentation).
            "trilinear" = trilinear interpolation (for images, default).
            "cubic" = cubic spline (smoother but may ring at edges).
        is_label: If True, 'auto' resolves to 'nearest'.
        verbose: Print progress messages.
        output_format: Output format — "auto" (match source), "zarr2", "zarr3".
        no_sharding: Disable sharding for zarr3 output.
        shard_shape: Custom shard shape for zarr3 sharded output.
        compression: Compression codec — "zstd", "gzip", or "none".
        compression_level: Compression level.

    Returns:
        dict with processing statistics.
    """
    upsampler = Upsampler(
        input_path=input_path,
        output_path="_placeholder_",
        target_voxel_sizes=[],  # Will be set below
    )

    # Read source metadata to determine voxel sizes
    src_voxels, axes_info = upsampler._get_source_voxel_sizes_and_axes()

    # Identify spatial voxel sizes
    spatial_voxels = []
    if axes_info:
        for i, ax in enumerate(axes_info):
            if ax.get("type") == "space" and i < len(src_voxels):
                spatial_voxels.append(src_voxels[i])
    else:
        spatial_voxels = list(src_voxels[:3])

    if not spatial_voxels:
        raise ValueError("No spatial axes found in source metadata.")

    # Determine target voxel size
    if target_voxel_size is None:
        target_voxel_size = min(spatial_voxels)

    target_voxels = [target_voxel_size] * len(spatial_voxels)

    if verbose:
        print(f"Source voxels (spatial): {spatial_voxels}")
        print(f"Target voxels: {target_voxels}")

    # Check if already isotropic
    all_close = all(
        abs(sv - target_voxel_size) / target_voxel_size < 0.01
        for sv in spatial_voxels
    )
    if all_close:
        raise ValueError(
            f"Source is already isotropic at {spatial_voxels}. "
            f"No upsampling needed."
        )

    # Build full target voxel list (including non-spatial axes)
    full_target = []
    spatial_idx = 0
    for i in range(len(src_voxels)):
        if axes_info and i < len(axes_info) and axes_info[i].get("type") == "space":
            full_target.append(target_voxels[spatial_idx])
            spatial_idx += 1
        else:
            full_target.append(src_voxels[i])

    # Create and run upsampler
    up = Upsampler(
        input_path=input_path,
        output_path=output_path,
        target_voxel_sizes=full_target,
        upsample_method=upsample_method,
        is_label=is_label,
        output_format=output_format,
        no_sharding=no_sharding,
        shard_shape=shard_shape,
        compression=compression,
        compression_level=compression_level,
    )
    stats = up.upsample(verbose=verbose)

    # Write OME-NGFF metadata at the group level
    _write_ome_metadata(output_path, axes_info, full_target)

    return stats


def _write_ome_metadata(
    s0_path: str,
    axes_info: Optional[List[Dict]],
    voxel_sizes: List[float],
):
    """Write OME-NGFF .zattrs for the parent group of s0.

    Creates minimal multiscales metadata with just the s0 level.
    Pyramid generation later updates this with additional levels.
    """
    group_path = os.path.dirname(s0_path)
    s0_name = os.path.basename(s0_path)

    # Build axes list
    if axes_info:
        axes = axes_info
    else:
        axes = [
            {"name": "x", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "z", "type": "space", "unit": "nanometer"},
        ]

    metadata = {
        "multiscales": [{
            "version": "0.4",
            "name": os.path.basename(group_path),
            "axes": axes,
            "datasets": [{
                "path": s0_name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": voxel_sizes}
                ],
            }],
        }]
    }

    # Write as .zattrs (Zarr2) or update zarr.json (Zarr3)
    zattrs_path = os.path.join(group_path, ".zattrs")
    zarr_json_path = os.path.join(group_path, "zarr.json")

    if os.path.exists(zarr_json_path):
        # Zarr3: update attributes.ome.multiscales
        with open(zarr_json_path) as f:
            meta = json.load(f)
        meta.setdefault("attributes", {}).setdefault("ome", {})["multiscales"] = metadata["multiscales"]
        with open(zarr_json_path, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        # Zarr2: write .zattrs
        # Ensure .zgroup exists
        zgroup_path = os.path.join(group_path, ".zgroup")
        if not os.path.exists(zgroup_path):
            os.makedirs(group_path, exist_ok=True)
            with open(zgroup_path, "w") as f:
                json.dump({"zarr_format": 2}, f)
        with open(zattrs_path, "w") as f:
            json.dump(metadata, f, indent=2)
