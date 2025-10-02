"""
Tests for HTTP path handling in TensorSwitch.
Verifies that store specs correctly handle both HTTP URLs and local file paths.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from tensorswitch.utils import get_kvstore_spec, n5_store_spec


class TestHTTPPathHandling:
    """Test HTTP and local file path handling."""

    def test_get_kvstore_spec_with_http_url(self):
        """Test get_kvstore_spec with HTTP URL."""
        http_url = "http://keller-s12c.hhmi.org/s12c/samples_for_stitching/20250902%20mouse%20hipp%203%20channels/dataset.n5/setup0/timepoint0/"

        kvstore = get_kvstore_spec(http_url)

        assert kvstore['driver'] == 'http'
        assert kvstore['base_url'] == 'http://keller-s12c.hhmi.org'
        assert kvstore['path'] == '/s12c/samples_for_stitching/20250902 mouse hipp 3 channels/dataset.n5/setup0/timepoint0/'
        # Verify URL decoding (% encoding)
        assert '%20' not in kvstore['path']
        assert 'mouse hipp 3 channels' in kvstore['path']

    def test_get_kvstore_spec_with_local_path(self):
        """Test get_kvstore_spec with local file path."""
        local_path = "tests/test_data/synthetic/test_volume.n5"

        kvstore = get_kvstore_spec(local_path)

        assert kvstore['driver'] == 'file'
        assert kvstore['path'] == local_path

    def test_get_kvstore_spec_with_https_url(self):
        """Test get_kvstore_spec with HTTPS URL."""
        https_url = "https://example.com/data/brain.n5/s0"

        kvstore = get_kvstore_spec(https_url)

        assert kvstore['driver'] == 'http'
        assert kvstore['base_url'] == 'https://example.com'
        assert kvstore['path'] == '/data/brain.n5/s0'

    def test_n5_store_spec_with_http_url(self):
        """Test n5_store_spec with HTTP URL."""
        http_url = "http://keller-s12c.hhmi.org/s12c/samples_for_stitching/20250902%20mouse%20hipp%203%20channels/dataset.n5/setup0/timepoint0/"

        spec = n5_store_spec(http_url)

        assert spec['driver'] == 'n5'
        assert spec['kvstore']['driver'] == 'http'
        assert spec['kvstore']['base_url'] == 'http://keller-s12c.hhmi.org'
        assert 'mouse hipp 3 channels' in spec['kvstore']['path']

    def test_n5_store_spec_with_local_path(self):
        """Test n5_store_spec with local file path."""
        local_path = "tests/test_data/synthetic/test_volume.n5"

        spec = n5_store_spec(local_path)

        assert spec['driver'] == 'n5'
        assert spec['kvstore']['driver'] == 'file'
        assert spec['kvstore']['path'] == local_path

    def test_get_kvstore_spec_with_absolute_path(self):
        """Test get_kvstore_spec with absolute file path."""
        abs_path = "/groups/scicompsoft/home/user/data/test.n5"

        kvstore = get_kvstore_spec(abs_path)

        assert kvstore['driver'] == 'file'
        assert kvstore['path'] == abs_path

    def test_get_kvstore_spec_with_relative_path(self):
        """Test get_kvstore_spec with relative file path."""
        rel_path = "../data/test.n5"

        kvstore = get_kvstore_spec(rel_path)

        assert kvstore['driver'] == 'file'
        assert kvstore['path'] == rel_path


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
