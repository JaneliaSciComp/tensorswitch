"""
Unit tests for BIOIO adapter reader.

Tests Tier 3 BIOIO integration including:
- Basic reading functionality
- Multi-resolution support
- Scene support
- Metadata extraction
- Factory integration
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from tensorswitch_v2.readers import BIOIOReader
from tensorswitch_v2.api import Readers


# Test data paths
ND2_TEST_FILE = "tests/real_test_data/nd2/20250903_FlyID08-01_2ndGel_1%_0.2%_1hr_Atto488_40XW_007.nd2"
TIFF_TEST_FILE = "tests/real_test_data/tif/20250414_1p75-fold_8xbin_nuclear_segmentation.tif"


def skip_if_no_bioio():
    """Skip test if BIOIO is not installed."""
    try:
        import bioio
        return False
    except ImportError:
        return True


def skip_if_no_test_data(path):
    """Skip test if test data file doesn't exist."""
    return not os.path.exists(path)


class TestBIOIOReaderBasic:
    """Test basic BIOIO reader functionality."""

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_nd2_reader_creation(self):
        """Test creating a BIOIO reader for ND2 file."""
        reader = BIOIOReader(ND2_TEST_FILE)
        assert reader is not None
        assert reader.path == ND2_TEST_FILE

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_nd2_tensorstore_spec(self):
        """Test getting TensorStore spec from ND2 via BIOIO."""
        reader = BIOIOReader(ND2_TEST_FILE)
        spec = reader.get_tensorstore_spec()

        assert spec is not None
        assert spec['driver'] == 'array'
        assert 'schema' in spec
        assert 'shape' in spec['schema']
        assert 'dtype' in spec['schema']
        assert len(spec['schema']['shape']) >= 3  # At least ZYX

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_nd2_metadata(self):
        """Test metadata extraction from ND2 via BIOIO."""
        reader = BIOIOReader(ND2_TEST_FILE)
        metadata = reader.get_metadata()

        assert metadata is not None
        assert 'dims' in metadata
        assert 'shape' in metadata
        assert 'dtype' in metadata

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_nd2_voxel_sizes(self):
        """Test voxel size extraction from ND2 via BIOIO."""
        reader = BIOIOReader(ND2_TEST_FILE)
        voxels = reader.get_voxel_sizes()

        assert voxels is not None
        assert 'x' in voxels
        assert 'y' in voxels
        assert 'z' in voxels
        # Should have reasonable values (not all 1.0)
        assert voxels['x'] > 0
        assert voxels['y'] > 0

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(TIFF_TEST_FILE), reason="TIFF test data not found")
    def test_tiff_reader(self):
        """Test BIOIO reader with TIFF file."""
        reader = BIOIOReader(TIFF_TEST_FILE)
        spec = reader.get_tensorstore_spec()

        assert spec is not None
        assert spec['driver'] == 'array'
        assert len(spec['schema']['shape']) >= 3


class TestBIOIOReaderMultiResolution:
    """Test multi-resolution (pyramid) support."""

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_resolution_levels_property(self):
        """Test resolution_levels property returns valid list."""
        reader = BIOIOReader(ND2_TEST_FILE)
        levels = reader.resolution_levels

        assert isinstance(levels, list)
        assert len(levels) >= 1
        assert 0 in levels  # Level 0 always available

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_current_resolution_level(self):
        """Test current_resolution_level property."""
        reader = BIOIOReader(ND2_TEST_FILE)
        level = reader.current_resolution_level

        assert isinstance(level, int)
        assert level >= 0

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_set_invalid_resolution_level(self):
        """Test that setting invalid resolution level raises error."""
        reader = BIOIOReader(ND2_TEST_FILE)

        # ND2 typically has only level 0
        with pytest.raises(ValueError) as exc_info:
            reader.set_resolution_level(99)

        assert "not available" in str(exc_info.value)

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_resolution_level_in_init(self):
        """Test resolution_level parameter in constructor."""
        reader = BIOIOReader(ND2_TEST_FILE, resolution_level=0)
        assert reader._resolution_level == 0

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_get_resolution_level_shape(self):
        """Test getting shape at specific resolution level."""
        reader = BIOIOReader(ND2_TEST_FILE)
        shape = reader.get_resolution_level_shape(0)

        assert isinstance(shape, tuple)
        assert len(shape) >= 3


class TestBIOIOReaderFactory:
    """Test factory integration via Readers.bioio()."""

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_factory_creates_reader(self):
        """Test Readers.bioio() factory creates BIOIOReader."""
        reader = Readers.bioio(ND2_TEST_FILE)

        assert isinstance(reader, BIOIOReader)

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_factory_with_scene_index(self):
        """Test factory with scene_index parameter."""
        reader = Readers.bioio(ND2_TEST_FILE, scene_index=0)

        assert reader._scene_index == 0

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_factory_with_resolution_level(self):
        """Test factory with resolution_level parameter."""
        reader = Readers.bioio(ND2_TEST_FILE, resolution_level=0)

        assert reader._resolution_level == 0


class TestBIOIOReaderScenes:
    """Test multi-scene support."""

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_scenes_property(self):
        """Test scenes property returns list."""
        reader = BIOIOReader(ND2_TEST_FILE)
        scenes = reader.scenes

        assert isinstance(scenes, list)

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_channel_names_property(self):
        """Test channel_names property returns list."""
        reader = BIOIOReader(ND2_TEST_FILE)
        channels = reader.channel_names

        assert isinstance(channels, list)


class TestBIOIOReaderRepr:
    """Test string representation."""

    @pytest.mark.skipif(skip_if_no_bioio(), reason="BIOIO not installed")
    @pytest.mark.skipif(skip_if_no_test_data(ND2_TEST_FILE), reason="ND2 test data not found")
    def test_repr(self):
        """Test __repr__ returns informative string."""
        reader = BIOIOReader(ND2_TEST_FILE, scene_index=1, resolution_level=0)
        repr_str = repr(reader)

        assert 'BIOIOReader' in repr_str
        assert 'scene=' in repr_str
        assert 'resolution_level=' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
