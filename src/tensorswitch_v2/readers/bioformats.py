"""
BioFormats reader for TensorSwitch Phase 5 architecture.

Tier 4 reader - wraps BIOIOReader with explicit Bio-Formats Java backend,
unlocking 150+ microscopy formats supported by the OME Bio-Formats library.

This reader requires Java and bioio-bioformats to be installed:
    conda install -c conda-forge scyjava
    pip install bioio-bioformats
"""

from typing import Dict, Optional, Any, List
from .bioio_adapter import BIOIOReader


class BioFormatsReader(BIOIOReader):
    """
    Reader using Bio-Formats Java library for 150+ microscopy formats.

    This is a specialized wrapper around BIOIOReader that explicitly uses
    the bioio-bioformats backend, which bridges to the OME Bio-Formats
    Java library via scyjava.

    Tier: 4 (Bio-Formats - Maximum Format Compatibility)
    - Supports 150+ file formats (many proprietary)
    - Requires Java runtime and bioio-bioformats package
    - Higher overhead than pure Python readers (JVM startup)
    - Use when native bioio plugins don't support your format

    When to use BioFormatsReader vs BIOIOReader:
        - BIOIOReader: Auto-detects and uses pure Python plugins (faster, no Java)
        - BioFormatsReader: Explicitly uses Bio-Formats Java (broader format support)

    Formats uniquely supported by Bio-Formats (not in pure Python plugins):
        - Olympus VSI, OIB, OIF
        - Leica SCN
        - Volocity
        - Imspector OBF
        - Many legacy/obscure vendor formats
        - See: https://bio-formats.readthedocs.io/en/latest/supported-formats.html

    Installation:
        # Install Java/Maven via conda
        conda install -c conda-forge scyjava

        # Install bioio-bioformats plugin
        pip install bioio-bioformats

        # Set JAVA_HOME if needed
        export JAVA_HOME=$CONDA_PREFIX  # Mac/Linux

    Example:
        >>> from tensorswitch_v2.readers import BioFormatsReader
        >>> reader = BioFormatsReader("/path/to/data.vsi")
        >>> spec = reader.get_tensorstore_spec()
        >>> metadata = reader.get_metadata()

    Example (via Readers factory):
        >>> from tensorswitch_v2.api import Readers
        >>> reader = Readers.bioformats("/path/to/data.vsi")

    Example (CLI):
        $ tensorswitch-v2 -i input.vsi -o output.zarr --use_bioformats

    Performance Note:
        First call has JVM startup overhead (~2-5 seconds).
        Subsequent reads are faster. For production-critical formats
        with native bioio plugins (CZI, LIF, ND2), prefer BIOIOReader.

    See Also:
        - Bio-Formats docs: https://bio-formats.readthedocs.io/
        - bioio-bioformats: https://github.com/bioio-devs/bioio-bioformats
        - Supported formats: https://bio-formats.readthedocs.io/en/latest/supported-formats.html
    """

    def __init__(
        self,
        path: str,
        scene_index: int = 0,
        channel_index: Optional[int] = None,
        time_index: Optional[int] = None,
        resolution_level: int = 0,
    ):
        """
        Initialize BioFormats reader.

        Args:
            path: Path to input file (local filesystem only - no remote URLs)
            scene_index: Which scene to load for multi-scene files (default: 0)
            channel_index: Optional specific channel to extract (None = all channels)
            time_index: Optional specific timepoint to extract (None = all timepoints)
            resolution_level: Resolution level to read (0=full, higher=lower res)

        Raises:
            ImportError: If bioio-bioformats is not installed or Java is not available

        Example:
            >>> # Basic usage
            >>> reader = BioFormatsReader("/data/image.vsi")

            >>> # Specific scene in multi-scene file
            >>> reader = BioFormatsReader("/data/slide.scn", scene_index=2)

            >>> # Lower resolution for quick preview
            >>> reader = BioFormatsReader("/data/large.svs", resolution_level=3)
        """
        # Import and validate bioio-bioformats availability
        bioformats_reader = self._get_bioformats_reader()

        # Initialize parent with explicit Bio-Formats backend
        super().__init__(
            path=path,
            scene_index=scene_index,
            channel_index=channel_index,
            time_index=time_index,
            resolution_level=resolution_level,
            reader=bioformats_reader,
        )

    def _get_bioformats_reader(self):
        """
        Import and return the bioio-bioformats Reader class.

        Returns:
            The bioio_bioformats.Reader class

        Raises:
            ImportError: If bioio-bioformats or Java dependencies are missing
        """
        try:
            # Ensure Bio-Formats JAR is loaded before importing
            BioFormatsReader._ensure_bioformats_loaded()
            from bioio_bioformats import Reader
            return Reader
        except ImportError as e:
            error_msg = str(e).lower()

            if 'java' in error_msg or 'jvm' in error_msg:
                raise ImportError(
                    "Java is required for Bio-Formats reader but was not found.\n\n"
                    "Install Java via conda:\n"
                    "  conda install -c conda-forge scyjava\n\n"
                    "Then set JAVA_HOME if needed:\n"
                    "  export JAVA_HOME=$CONDA_PREFIX  # Mac/Linux\n"
                    "  set JAVA_HOME=%CONDA_PREFIX%\\Library  # Windows\n"
                ) from e
            else:
                raise ImportError(
                    "bioio-bioformats is not installed.\n\n"
                    "Install it with:\n"
                    "  pip install bioio-bioformats\n\n"
                    "You also need Java installed:\n"
                    "  conda install -c conda-forge scyjava\n\n"
                    "See: https://github.com/bioio-devs/bioio-bioformats"
                ) from e

    @staticmethod
    def _ensure_bioformats_loaded():
        """
        Ensure Bio-Formats JAR is loaded before using bioio-bioformats.

        The bioformats_jar package needs to be called to download and add
        the Bio-Formats JAR to the classpath before the JVM starts.
        """
        try:
            import bioformats_jar
            bioformats_jar.get_loci()  # Downloads JAR if needed and adds to classpath
        except ImportError:
            pass  # bioformats_jar not installed

    @staticmethod
    def bioformats_version() -> str:
        """
        Get the version of the underlying Bio-Formats Java library.

        Returns:
            str: Bio-Formats version string (e.g., "7.0.0")

        Example:
            >>> BioFormatsReader.bioformats_version()
            '7.0.0'
        """
        try:
            BioFormatsReader._ensure_bioformats_loaded()
            from bioio_bioformats import Reader
            return Reader.bioformats_version()
        except Exception:
            return "unknown"

    @staticmethod
    def is_available() -> bool:
        """
        Check if Bio-Formats reader is available (Java + bioio-bioformats installed).

        Returns:
            bool: True if Bio-Formats can be used

        Example:
            >>> if BioFormatsReader.is_available():
            ...     reader = BioFormatsReader("/data/image.vsi")
            ... else:
            ...     print("Install bioio-bioformats and Java")
        """
        try:
            BioFormatsReader._ensure_bioformats_loaded()
            from bioio_bioformats import Reader
            # Try to access the version to verify Java works
            Reader.bioformats_version()
            return True
        except Exception:
            return False

    def supports_remote(self) -> bool:
        """
        Check if remote access is supported.

        Bio-Formats only supports local filesystem access.

        Returns:
            bool: Always False for Bio-Formats
        """
        return False

    def __repr__(self) -> str:
        """String representation of BioFormats reader."""
        return (
            f"BioFormatsReader(path='{self.path}', "
            f"scene={self._scene_index}, "
            f"resolution_level={self._resolution_level})"
        )
