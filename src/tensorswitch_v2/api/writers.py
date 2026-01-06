"""
Writers factory class with static methods for creating format-specific writers.

Provides explicit writer selection via static methods.
"""

from typing import Optional, Tuple
from ..writers.base import BaseWriter


class Writers:
    """
    Static factory class for creating format-specific writers.

    Provides clean API for writer instantiation with sensible defaults.
    Writers are format-agnostic processors that accept TensorStore arrays
    from any source.

    Architecture Layer: 3 (Processing/Output)
    - Creates format-specific writer instances
    - Provides default configurations
    - Format-agnostic design (no reader knowledge)

    Output Formats:
        - Zarr3: Primary output format with sharding support
        - Zarr2: Legacy Zarr format for compatibility
        - N5: Rechunking and metadata conversion

    Example (Zarr3 with defaults):
        >>> from tensorswitch_v2.api import Writers, TensorSwitchDataset
        >>> dataset = TensorSwitchDataset("/input.tif", reader=...)
        >>> writer = Writers.zarr3("/output.zarr")
        >>> ts_array = dataset.get_tensorstore_array(mode='open')
        >>> writer.create_output_spec(ts_array, chunk_shape=(1, 1024, 1024))

    Example (Zarr3 with custom options):
        >>> writer = Writers.zarr3(
        ...     "/output.zarr",
        ...     use_sharding=True,
        ...     compression="blosc",
        ...     compression_level=5
        ... )

    Example (N5 for compatibility):
        >>> writer = Writers.n5("/output.n5")

    Design Principles (Critical):
        - Writers ONLY accept TensorStore arrays
        - Writers NEVER ask about input format
        - Complete decoupling from readers
        - Format-specific output encoding only
    """

    @staticmethod
    def zarr3(
        output_path: str,
        use_sharding: bool = True,
        compression: str = "blosc",
        compression_level: int = 5,
        **kwargs
    ) -> BaseWriter:
        """
        Create Zarr3 writer (primary output format).

        Zarr3 is the recommended output format with modern features:
        - Sharding support (reduces small file count)
        - Flexible codecs (blosc, gzip, zstd, etc.)
        - OME-NGFF v0.5 metadata
        - Remote storage support (GCS, S3)

        Args:
            output_path: Path to output Zarr3 dataset
            use_sharding: Enable sharding_indexed codec (default: True)
            compression: Compression codec ('blosc', 'gzip', 'zstd', 'none')
            compression_level: Compression level (1-9, default: 5)
            **kwargs: Additional format-specific options

        Returns:
            Zarr3Writer instance

        Example (Default - sharded):
            >>> writer = Writers.zarr3("/output.zarr")

        Example (No sharding):
            >>> writer = Writers.zarr3("/output.zarr", use_sharding=False)

        Example (Custom compression):
            >>> writer = Writers.zarr3(
            ...     "/output.zarr",
            ...     compression="zstd",
            ...     compression_level=7
            ... )

        Implementation Status:
            🚧 Week 8-9 (Phase 5.5 - Writers + Distributed)
        """
        raise NotImplementedError(
            "Zarr3Writer not yet implemented. "
            "Will be added in Week 8-9 (Phase 5.5 - Writers + Distributed). "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def zarr2(
        output_path: str,
        compression: str = "blosc",
        compression_level: int = 5,
        **kwargs
    ) -> BaseWriter:
        """
        Create Zarr2 writer (legacy format for compatibility).

        Zarr2 is the legacy format, provided for compatibility:
        - No sharding support (many small files)
        - OME-NGFF v0.4 metadata
        - Wide tool support (neuroglancer, napari, etc.)

        Args:
            output_path: Path to output Zarr2 dataset
            compression: Compression codec ('blosc', 'gzip', 'zstd', 'none')
            compression_level: Compression level (1-9, default: 5)
            **kwargs: Additional format-specific options

        Returns:
            Zarr2Writer instance

        Example:
            >>> writer = Writers.zarr2("/output.zarr")

        Example (Custom compression):
            >>> writer = Writers.zarr2(
            ...     "/output.zarr",
            ...     compression="gzip",
            ...     compression_level=6
            ... )

        Notes:
            - Use Zarr3 for new projects (sharding, better performance)
            - Use Zarr2 for compatibility with existing tools
            - No sharding means many small files (performance impact)

        Implementation Status:
            🚧 Week 8-9 (Phase 5.5 - Writers + Distributed)
        """
        raise NotImplementedError(
            "Zarr2Writer not yet implemented. "
            "Will be added in Week 8-9 (Phase 5.5 - Writers + Distributed). "
            "See PLAN_phase5.md for timeline."
        )

    @staticmethod
    def n5(
        output_path: str,
        compression: str = "gzip",
        compression_level: int = 5,
        **kwargs
    ) -> BaseWriter:
        """
        Create N5 writer (rechunking and metadata conversion).

        N5 is used for:
        - N5→N5 rechunking operations
        - N5 metadata conversion
        - Compatibility with Java tools (BigDataViewer, etc.)

        Args:
            output_path: Path to output N5 dataset
            compression: Compression type ('gzip', 'bzip2', 'xz', 'raw')
            compression_level: Compression level (1-9, default: 5)
            **kwargs: Additional format-specific options

        Returns:
            N5Writer instance

        Example (N5 rechunking):
            >>> writer = Writers.n5("/output.n5")

        Example (Custom compression):
            >>> writer = Writers.n5(
            ...     "/output.n5",
            ...     compression="bzip2",
            ...     compression_level=9
            ... )

        Notes:
            - N5 uses blocks (similar to chunks)
            - No sharding support
            - Compatible with BigDataViewer, Java ecosystem
            - Metadata in attributes.json (different from Zarr)

        Implementation Status:
            🚧 Week 8-9 (Phase 5.5 - Writers + Distributed)
        """
        raise NotImplementedError(
            "N5Writer not yet implemented. "
            "Will be added in Week 8-9 (Phase 5.5 - Writers + Distributed). "
            "See PLAN_phase5.md for timeline."
        )
