# -*- coding: utf-8 -*-
"""
Created on November 10, 2025

Comprehensive pytest test suite for NIMS functionality - Simplified Version
Translation from unittest to pytest with fixtures, subtests and mock data

@author: MTH5 Development Team
"""

# =============================================================================
# Imports
# =============================================================================
import datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from mth5.io.nims import NIMS, read_nims


# =============================================================================
# Test Classes - Simplified
# =============================================================================


class TestNIMSBasicFunctionality:
    """Test basic NIMS functionality without complex data dependencies."""

    def test_initialization(self):
        """Test NIMS initialization."""
        # Test without filename
        nims = NIMS()
        assert nims.fn is None
        assert nims.block_size == 131
        assert nims.sample_rate == 8

        # Test with filename
        test_file = "test.BIN"
        nims_with_file = NIMS(test_file)
        assert str(nims_with_file.fn) == test_file or nims_with_file.fn == Path(
            test_file
        )

    def test_has_data(self):
        """Test has_data method."""
        nims = NIMS()
        assert not nims.has_data()

        # Add mock data
        nims.ts_data = pd.DataFrame({"hx": [1, 2, 3]})
        assert nims.has_data()

    def test_n_samples_with_data(self):
        """Test n_samples property when data is available."""
        nims = NIMS()
        nims.ts_data = pd.DataFrame({"hx": [1, 2, 3, 4, 5]})
        assert nims.n_samples == 5

    def test_string_representation(self):
        """Test string representation without complex property access."""
        nims = NIMS()
        nims.site_name = "Test Site"
        nims.box_id = "1105-3"
        nims.mag_id = "1305-3"

        # Just test the basic string representation works
        # without accessing complex properties that cause issues
        nims_str = str(nims)
        assert "Test Site" in nims_str
        assert "1105-3" in nims_str


class TestNIMSBasicProperties:
    """Test basic property access without complex metadata validation."""

    def test_basic_property_setting(self):
        """Test setting and getting basic properties."""
        nims = NIMS()

        # Test setting properties
        test_values = {
            "site_name": "Test Site",
            "box_id": "1105-3",
            "mag_id": "1305-3",
            "operator": "Test Op",
            "sample_rate": 8,
            "ex_length": 100.0,
            "ey_length": 90.0,
        }

        for prop, value in test_values.items():
            setattr(nims, prop, value)
            assert getattr(nims, prop) == value


class TestNIMSDataProcessing:
    """Test NIMS data processing utility methods."""

    def test_make_index_values(self):
        """Test _make_index_values method."""
        nims = NIMS()
        indices = nims._make_index_values()
        assert indices.shape == (8, 5)
        assert indices.dtype == int

    def test_find_sequence(self):
        """Test find_sequence method."""
        nims = NIMS()
        # Create test data with known sequence
        test_data = np.array([0, 1, 131, 5, 10, 1, 131, 8, 20])
        sequence = [1, 131]

        result = nims.find_sequence(test_data, sequence)
        expected = np.array([1, 5])  # Positions where [1, 131] appears
        np.testing.assert_array_equal(result, expected)

    def test_unwrap_sequence(self):
        """Test unwrap_sequence method."""
        nims = NIMS()
        # Test sequence that wraps around 256
        test_sequence = np.array([100, 150, 200, 255, 0, 50, 100])
        result = nims.unwrap_sequence(test_sequence)

        # Should be monotonically increasing after unwrapping
        assert np.all(np.diff(result) >= 0)
        assert result[0] == 0  # Should start at 0

    def test_locate_duplicate_blocks(self):
        """Test _locate_duplicate_blocks method."""
        nims = NIMS()
        # Create sequence with duplicates
        test_sequence = np.array([1, 2, 2, 3, 4, 4, 5])

        duplicates = nims._locate_duplicate_blocks(test_sequence)
        assert duplicates is not None
        assert len(duplicates) == 2  # Two duplicate pairs

        # Test sequence without duplicates
        test_sequence_no_dup = np.array([1, 2, 3, 4, 5])
        duplicates_empty = nims._locate_duplicate_blocks(test_sequence_no_dup)
        assert duplicates_empty is None

    def test_check_duplicate_blocks(self):
        """Test _check_duplicate_blocks method."""
        nims = NIMS()
        # Create identical test blocks
        block1 = np.array([1, 2, 3, 4, 5])
        block2 = np.array([1, 2, 3, 4, 5])
        info1 = np.array([10, 20])
        info2 = np.array([10, 20])

        # Should return True for identical blocks
        result = nims._check_duplicate_blocks(block1, block2, info1, info2)
        assert result is True

        # Different blocks should return False
        block3 = np.array([1, 2, 3, 4, 6])
        result = nims._check_duplicate_blocks(block1, block3, info1, info2)
        assert result is False

    def test_make_dt_index(self):
        """Test make_dt_index method."""
        nims = NIMS()
        start_time = "2019-09-26T18:33:21"
        sample_rate = 8.0
        n_samples = 100

        # Test with n_samples
        dt_index = nims.make_dt_index(start_time, sample_rate, n_samples=n_samples)
        assert len(dt_index) == n_samples
        # Check timezone properly
        assert str(dt_index.tz) in ["UTC", "+00:00"]

        # Test error case
        with pytest.raises(
            ValueError, match="Need to input either stop_time or n_samples"
        ):
            nims.make_dt_index(start_time, sample_rate)


class TestNIMSChannelResponse:
    """Test NIMS channel response functionality."""

    def test_get_channel_response(self):
        """Test get_channel_response method."""
        nims = NIMS()
        # Test for different channels
        for channel in ["hx", "hy", "hz", "ex", "ey"]:
            response = nims.get_channel_response(channel)
            assert response is not None


class TestNIMSGPSProcessing:
    """Test NIMS GPS processing methods."""

    def test_get_gps_stamp_indices_from_status(self):
        """Test _get_gps_stamp_indices_from_status method."""
        nims = NIMS()
        # Create status array with some GPS locks (status = 0)
        status_array = np.array([1, 0, 0, 1, 0, 1, 1, 0])

        indices = nims._get_gps_stamp_indices_from_status(status_array)

        # Should return indices where status changes to 0 (ignoring consecutive 0s)
        expected = np.array([1, 4, 7])  # First occurrence of each GPS lock period
        np.testing.assert_array_equal(indices, expected)

    def test_get_gps_string_list(self):
        """Test _get_gps_string_list method."""
        nims = NIMS()
        # Create test binary string with '$' markers
        test_data = bytearray(131 * 3)  # 3 blocks
        test_data[3] = ord("$")  # First block has '$'
        test_data[134] = ord("A")  # Second block has 'A'
        test_data[265] = ord("$")  # Third block has '$'

        index_values, gps_str_list = nims._get_gps_string_list(bytes(test_data))

        assert len(index_values) >= 1  # Should find at least one '$' marker
        assert len(gps_str_list) >= 3  # Should have multiple GPS string parts


class TestNIMSFileOperations:
    """Test NIMS file operations with mocking."""

    @patch("builtins.open", new_callable=mock_open)
    @patch.object(NIMS, "read_header")
    def test_read_nims_file_open(self, mock_read_header, mock_file):
        """Test that read_nims opens the file correctly."""
        nims = NIMS()
        nims.data_start_seek = 0

        # Mock the file to raise an exception early to avoid complex data processing
        mock_file.return_value.read.side_effect = Exception(
            "Test exception to stop processing"
        )

        # Just test that the file operations are called correctly
        with pytest.raises(Exception, match="Test exception"):
            nims.read_nims("test_file.BIN")

        # Verify file operations - handle both string and Path objects
        calls = mock_file.call_args_list
        assert len(calls) == 1
        call_args = calls[0][0]  # Get positional arguments
        assert str(call_args[0]) == "test_file.BIN"
        assert call_args[1] == "rb"
        mock_read_header.assert_called_once()


class TestNIMSUtilityMethods:
    """Test NIMS utility methods."""

    def test_properties_without_data(self):
        """Test property access when no data is loaded."""
        nims = NIMS()

        # Set some header values
        nims.header_gps_latitude = 45.0
        nims.header_gps_longitude = -120.0
        nims.header_gps_elevation = 1000.0

        # Should return header values when no stamps are available
        assert nims.latitude == 45.0
        assert nims.longitude == -120.0
        assert nims.elevation == 1000.0

    def test_channel_properties_without_data(self):
        """Test channel properties when no data is loaded."""
        nims = NIMS()
        # Should return None when ts_data is None
        assert nims.hx is None
        assert nims.hy is None
        assert nims.hz is None
        assert nims.ex is None
        assert nims.ey is None
        assert nims.box_temperature is None

    def test_metadata_without_data(self):
        """Test metadata properties when no data is loaded."""
        nims = NIMS()
        # Should return None when ts_data is None
        assert nims.run_metadata is None
        assert nims.station_metadata is None

    def test_to_runts_without_data(self):
        """Test to_runts when no data is available."""
        nims = NIMS()
        result = nims.to_runts()
        assert result is None


class TestNIMSPerformance:
    """Test NIMS performance and edge cases."""

    def test_large_sequence_processing(self):
        """Test processing of large sequences."""
        nims = NIMS()
        # Create large test sequence
        large_sequence = np.arange(10000)

        # Should handle large sequences efficiently
        result = nims.unwrap_sequence(large_sequence)
        assert len(result) == len(large_sequence)
        assert result[0] == 0

    def test_find_sequence_performance(self):
        """Test find_sequence with large data."""
        nims = NIMS()
        # Create large test data
        large_data = np.random.randint(0, 256, 100000)
        # Insert known sequence at specific positions
        large_data[1000:1002] = [1, 131]
        large_data[5000:5002] = [1, 131]

        result = nims.find_sequence(large_data, [1, 131])

        # Should find the sequences we inserted
        assert len(result) >= 2
        assert 1000 in result
        assert 5000 in result


class TestNIMSIntegration:
    """Test NIMS integration scenarios."""

    @patch("mth5.io.nims.nims.NIMS")
    def test_read_nims_function(self, mock_nims_class):
        """Test the read_nims convenience function."""
        mock_instance = Mock()
        mock_instance.to_runts.return_value = Mock()
        mock_nims_class.return_value = mock_instance

        result = read_nims("test_file.BIN")

        # Should create NIMS instance and call methods
        mock_nims_class.assert_called_once_with("test_file.BIN")
        mock_instance.read_nims.assert_called_once()
        mock_instance.to_runts.assert_called_once()

        assert result == mock_instance.to_runts.return_value


class TestNIMSTimingMethods:
    """Test NIMS timing-related methods."""

    def test_check_timing_calculation(self):
        """Test timing calculation methods."""
        nims = NIMS()

        # Create mock GPS stamps
        base_time = datetime.datetime(
            2019, 9, 26, 18, 33, 21, tzinfo=datetime.timezone.utc
        )
        stamps = []

        for i in range(3):
            gprmc_mock = Mock()
            gprmc_mock.gps_type = "GPRMC"
            gprmc_mock.time_stamp = base_time + datetime.timedelta(seconds=i * 10)
            gprmc_mock.index = i * 10
            gprmc_mock.valid = True

            gpgga_mock = Mock()
            gpgga_mock.gps_type = "GPGGA"

            stamps.append([i * 10, [gprmc_mock, gpgga_mock]])

        valid, gaps, difference = nims.check_timing(stamps)

        # With consistent timing, should be valid
        assert valid is True
        assert gaps is None
        assert difference == 0

    def test_align_data_basic(self):
        """Test basic align_data functionality."""
        nims = NIMS()

        # Create mock data array
        n_samples = 80
        data_array = np.zeros(
            n_samples,
            dtype=[
                ("hx", float),
                ("hy", float),
                ("hz", float),
                ("ex", float),
                ("ey", float),
            ],
        )

        # Fill with simple test data
        for i in range(n_samples):
            data_array[i] = (i, i + 1, i + 2, i + 3, i + 4)

        # Create mock stamps
        base_time = datetime.datetime(
            2019, 9, 26, 18, 33, 21, tzinfo=datetime.timezone.utc
        )
        gprmc_mock = Mock()
        gprmc_mock.gps_type = "GPRMC"
        gprmc_mock.time_stamp = base_time
        stamps = [[0, [gprmc_mock, Mock()]]]

        # Mock the check_timing method
        with patch.object(nims, "check_timing", return_value=(True, None, 0)):
            result = nims.align_data(data_array, stamps)

            # Should return a pandas DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(data_array)
            assert list(result.columns) == ["hx", "hy", "hz", "ex", "ey"]


class TestNIMSComprehensiveSuite:
    """Test coverage validation."""

    def test_essential_methods_exist(self):
        """Test that essential methods exist."""
        nims = NIMS()

        # Test essential methods exist
        essential_methods = [
            "has_data",
            "get_channel_response",
            "to_runts",
            "find_sequence",
            "unwrap_sequence",
            "read_nims",
            "make_dt_index",
        ]

        for method in essential_methods:
            assert hasattr(nims, method), f"Method {method} not found"
            assert callable(getattr(nims, method)), f"Method {method} is not callable"

    def test_essential_properties_exist(self):
        """Test that essential properties exist."""
        nims = NIMS()
        # Set minimal data to avoid errors
        nims.ts_data = pd.DataFrame({"hx": [1, 2, 3]})

        # Test properties that can be safely accessed
        essential_properties = ["latitude", "longitude", "elevation", "n_samples"]

        for prop in essential_properties:
            assert hasattr(nims, prop), f"Property {prop} not found"

        # Test that channel properties exist as methods (they're properties that might return None)
        channel_properties = ["hx", "hy", "hz", "ex", "ey"]
        for prop in channel_properties:
            # These are property methods, so we check if they exist and are callable when accessed
            assert hasattr(type(nims), prop), f"Property {prop} not found in class"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
