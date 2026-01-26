"""
Integration tests for DistributedConverter.

Tests end-to-end conversion workflows with various reader/writer combinations.
"""

import os
import json
import pytest
import numpy as np
import tensorstore as ts

from tensorswitch_v2.api import Readers, Writers
from tensorswitch_v2.core import DistributedConverter
from tensorswitch_v2.readers import TiffReader, Zarr3Reader, N5Reader
from tensorswitch_v2.writers import Zarr3Writer, Zarr2Writer, N5Writer

from conftest import validate_zarr3_output, validate_zarr2_output, validate_n5_output


class TestDistributedConverterBasic:
    """Basic tests for DistributedConverter."""

    def test_init(self, sample_tiff_path, temp_dir):
        """Test DistributedConverter initialization."""
        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))

        converter = DistributedConverter(reader, writer)

        assert converter.reader == reader
        assert converter.writer == writer

    def test_get_total_chunks(self, sample_tiff_path, temp_dir):
        """Test getting total chunk count."""
        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))

        converter = DistributedConverter(reader, writer)
        total = converter.get_total_chunks(chunk_shape=(16, 32, 32))

        # (32, 64, 64) with (16, 32, 32) chunks = 2*2*2 = 8 chunks
        assert total == 8

    def test_get_chunk_ranges(self, sample_tiff_path, temp_dir):
        """Test getting chunk ranges for LSF jobs."""
        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(os.path.join(temp_dir, "out.zarr"))

        converter = DistributedConverter(reader, writer)
        ranges = converter.get_chunk_ranges(num_jobs=2, chunk_shape=(16, 32, 32))

        assert len(ranges) == 2
        # 8 chunks split into 2 jobs = 4 each
        assert ranges[0] == (0, 4)
        assert ranges[1] == (4, 8)


class TestTiffToZarr3:
    """Tests for TIFF -> Zarr3 conversion."""

    def test_basic_conversion(self, sample_tiff_path, temp_dir, sample_3d_array):
        """Test basic TIFF to Zarr3 conversion."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=False)

        assert stats['chunks_processed'] > 0
        assert stats['elapsed_seconds'] > 0

        # Validate output structure
        validation = validate_zarr3_output(output_path)
        assert validation['valid'], f"Validation failed: {validation['errors']}"

        # Verify data integrity
        output_spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0')}
        }
        output_store = ts.open(output_spec, read=True).result()
        output_data = output_store[...].read().result()

        assert output_data.shape == sample_3d_array.shape
        assert np.array_equal(output_data, sample_3d_array)

    def test_conversion_with_metadata(self, sample_tiff_path, temp_dir):
        """Test conversion preserves metadata."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        converter.convert(chunk_shape=(16, 32, 32), write_metadata=True, verbose=False)

        # Check zarr.json has OME metadata
        zarr_json_path = os.path.join(output_path, "zarr.json")
        assert os.path.exists(zarr_json_path)

        with open(zarr_json_path) as f:
            metadata = json.load(f)

        assert metadata['zarr_format'] == 3
        assert 'ome' in metadata.get('attributes', {})


class TestZarr3ToZarr3:
    """Tests for Zarr3 -> Zarr3 conversion (rechunking)."""

    def test_rechunking(self, sample_zarr3_path, temp_dir, sample_3d_array):
        """Test Zarr3 rechunking."""
        output_path = os.path.join(temp_dir, "out_rechunked.zarr")

        reader = Readers.zarr3(sample_zarr3_path, dataset_path="s0")
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        # Use different chunk shape than input
        stats = converter.convert(chunk_shape=(8, 16, 16), verbose=False)

        assert stats['chunks_processed'] > 0

        # Verify data
        output_spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0')}
        }
        output_store = ts.open(output_spec, read=True).result()
        output_data = output_store[...].read().result()

        assert np.array_equal(output_data, sample_3d_array)


class TestN5ToZarr3:
    """Tests for N5 -> Zarr3 conversion."""

    def test_basic_conversion(self, sample_n5_path, temp_dir, sample_3d_array):
        """Test N5 to Zarr3 conversion."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.n5(sample_n5_path, dataset_path="s0")
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=False)

        assert stats['chunks_processed'] > 0

        # Verify data
        output_spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0')}
        }
        output_store = ts.open(output_spec, read=True).result()
        output_data = output_store[...].read().result()

        assert np.array_equal(output_data, sample_3d_array)


class TestTiffToZarr2:
    """Tests for TIFF -> Zarr2 conversion."""

    def test_basic_conversion(self, sample_tiff_path, temp_dir, sample_3d_array):
        """Test TIFF to Zarr2 conversion."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr2(output_path)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=False)

        assert stats['chunks_processed'] > 0

        # Validate output structure
        validation = validate_zarr2_output(output_path)
        assert validation['valid'], f"Validation failed: {validation['errors']}"

        # Verify data
        output_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0')}
        }
        output_store = ts.open(output_spec, read=True).result()
        output_data = output_store[...].read().result()

        assert np.array_equal(output_data, sample_3d_array)


class TestChunkRangeProcessing:
    """Tests for LSF-style chunk range processing."""

    def test_partial_processing(self, sample_tiff_path, temp_dir, sample_3d_array):
        """Test processing a subset of chunks (LSF multi-job mode)."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)

        # Get total chunks
        total = converter.get_total_chunks(chunk_shape=(16, 32, 32))

        # Process first half
        stats1 = converter.convert(
            start_idx=0,
            stop_idx=total // 2,
            chunk_shape=(16, 32, 32),
            write_metadata=False,
            verbose=False
        )

        # Process second half
        reader2 = Readers.tiff(sample_tiff_path)
        writer2 = Writers.zarr3(output_path, use_sharding=False)
        converter2 = DistributedConverter(reader2, writer2)

        stats2 = converter2.convert(
            start_idx=total // 2,
            stop_idx=total,
            chunk_shape=(16, 32, 32),
            write_metadata=True,  # Write metadata on last job
            verbose=False
        )

        # Total processed should equal total chunks
        assert stats1['chunks_processed'] + stats2['chunks_processed'] == total

        # Verify complete data
        output_spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(output_path, 's0')}
        }
        output_store = ts.open(output_spec, read=True).result()
        output_data = output_store[...].read().result()

        assert np.array_equal(output_data, sample_3d_array)


class TestConverterStatistics:
    """Tests for conversion statistics."""

    def test_stats_returned(self, sample_tiff_path, temp_dir):
        """Test that statistics are returned."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=False)

        assert 'chunks_processed' in stats
        assert 'total_chunks' in stats
        assert 'elapsed_seconds' in stats
        assert 'chunks_per_second' in stats
        assert 'input_shape' in stats
        assert 'output_chunk_shape' in stats

    def test_stats_accuracy(self, sample_tiff_path, temp_dir):
        """Test that statistics are accurate."""
        output_path = os.path.join(temp_dir, "out.zarr")

        reader = Readers.tiff(sample_tiff_path)
        writer = Writers.zarr3(output_path, use_sharding=False)

        converter = DistributedConverter(reader, writer)
        stats = converter.convert(chunk_shape=(16, 32, 32), verbose=False)

        # Shape (32, 64, 64) with (16, 32, 32) = 8 chunks
        assert stats['total_chunks'] == 8
        assert stats['chunks_processed'] == 8
        assert stats['input_shape'] == (32, 64, 64)
        assert stats['output_chunk_shape'] == (16, 32, 32)
