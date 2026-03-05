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
        compression: str = "zstd",
        compression_level: int = 5,
        use_ome_structure: bool = True,
        level_path: str = "s0",
        include_omero: bool = False,
        use_nested_structure: bool = True,
        data_type: str = "image",
        image_key: str = "raw",
        label_key: str = "segmentation",
        **kwargs
    ) -> BaseWriter:
        """
        Create Zarr3 writer (primary output format).

        Zarr3 is the recommended output format with modern features:
        - Sharding support (reduces small file count)
        - Flexible codecs (blosc, gzip, zstd, etc.)
        - OME-NGFF v0.5 metadata
        - Remote storage support (GCS, S3)
        - Nested structure support (raw/, labels/segmentation/)

        Args:
            output_path: Path to output Zarr3 dataset
            use_sharding: Enable sharding_indexed codec (default: True)
            compression: Compression codec ('blosc', 'gzip', 'zstd', 'none')
            compression_level: Compression level (1-9, default: 5)
            use_ome_structure: Use OME-ZARR directory structure (default: True)
            level_path: Level subdirectory name (default: "s0")
            use_nested_structure: Use OME-NGFF nested structure (default: True)
            data_type: 'image' or 'labels' - determines output subdirectory
            image_key: Name for image group (default: "raw")
            label_key: Name for label image (default: "segmentation")
            **kwargs: Additional format-specific options

        Returns:
            Zarr3Writer instance

        Example (Default - sharded):
            >>> writer = Writers.zarr3("/output.zarr")

        Example (No sharding):
            >>> writer = Writers.zarr3("/output.zarr", use_sharding=False)

        Example (Labels output):
            >>> writer = Writers.zarr3("/output.zarr", data_type="labels")
        """
        from ..writers.zarr3 import Zarr3Writer
        return Zarr3Writer(
            output_path=output_path,
            use_sharding=use_sharding,
            compression=compression,
            compression_level=compression_level,
            use_ome_structure=use_ome_structure,
            level_path=level_path,
            include_omero=include_omero,
            use_nested_structure=use_nested_structure,
            data_type=data_type,
            image_key=image_key,
            label_key=label_key
        )

    @staticmethod
    def zarr2(
        output_path: str,
        compression: str = "zstd",
        compression_level: int = 5,
        level_path: str = "s0",
        include_omero: bool = False,
        use_nested_structure: bool = True,
        data_type: str = 'image',
        image_key: str = 'raw',
        label_key: str = 'segmentation',
        **kwargs
    ) -> BaseWriter:
        """
        Create Zarr2 writer (legacy format for compatibility).

        Zarr2 is the legacy format, provided for compatibility:
        - No sharding support (many small files)
        - OME-NGFF v0.4 metadata
        - Wide tool support (neuroglancer, napari, etc.)
        - Automatic 5D TCZYX expansion for viewer compatibility

        Args:
            output_path: Path to output Zarr2 dataset
            compression: Compression codec ('blosc', 'gzip', 'zstd', 'none')
            compression_level: Compression level (1-9, default: 5)
            level_path: Level subdirectory name (default: "s0" for Janelia convention)
            use_nested_structure: Use OME-NGFF nested directory structure (default: True)
            data_type: Type of data ('image' or 'labels')
            image_key: Name for image group in nested structure (default: 'raw')
            label_key: Name for label image in nested structure (default: 'segmentation')
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
            - Data is always expanded to 5D TCZYX format for OME-NGFF viewer compatibility
        """
        from ..writers.zarr2 import Zarr2Writer
        return Zarr2Writer(
            output_path=output_path,
            compression=compression,
            compression_level=compression_level,
            level_path=level_path,
            include_omero=include_omero,
            use_nested_structure=use_nested_structure,
            data_type=data_type,
            image_key=image_key,
            label_key=label_key
        )

    @staticmethod
    def n5(
        output_path: str,
        compression: str = "gzip",
        compression_level: int = 5,
        dataset_path: str = "s0",
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
            dataset_path: Dataset path within N5 (default: "s0" for Janelia convention)
            **kwargs: Additional format-specific options

        Returns:
            N5Writer instance

        Example (N5 rechunking):
            >>> writer = Writers.n5("/output.n5")

        Example (Custom compression):
            >>> writer = Writers.n5(
            ...     "/output.n5",
            ...     compression="blosc",
            ...     compression_level=9
            ... )

        Notes:
            - N5 uses blocks (similar to chunks)
            - No sharding support
            - Compatible with BigDataViewer, Java ecosystem
            - Metadata in attributes.json (different from Zarr)
        """
        from ..writers.n5 import N5Writer
        return N5Writer(
            output_path=output_path,
            compression=compression,
            compression_level=compression_level,
            dataset_path=dataset_path
        )
