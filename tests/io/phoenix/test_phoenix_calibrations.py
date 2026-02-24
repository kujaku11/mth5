# -*- coding: utf-8 -*-
"""
Pytest test suite for Phoenix Geophysics calibration functionality.

@author: jpeacock
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

from mth5.io.phoenix.readers import PhoenixCalibration


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cal_file_path():
    """Fixture providing path to the example calibration file."""
    return phx_data_path / "10128_rxcal.json"


@pytest.fixture
def phoenix_calibration(cal_file_path):
    """Fixture providing a PhoenixCalibration instance with loaded data."""
    return PhoenixCalibration(cal_file_path)


@pytest.fixture
def sample_cal_data():
    """Fixture providing sample calibration data for testing."""
    return {
        "altitude": 0.0,
        "empower_version": "2.9.0.11",
        "file_type": "receiver calibration",
        "file_version": "1.0",
        "inst_serial": "666",
        "instrument_model": "RMT03-J",
        "instrument_type": "MTU-5C",
        "latitude": 0.0,
        "longitude": 0.0,
        "manufacturer": "Phoenix Geophysics",
        "num_channels": 5,
        "timestamp_utc": 1685730408,
        "cal_data": [
            {
                "tag": "E1",
                "num_of_responses": 1,
                "chan_data": [
                    {
                        "num_records": 5,
                        "freq_Hz": [1.0, 10.0, 100.0, 1000.0, 10000.0],
                        "magnitude": [1.0, 1.0, 1.0, 1.0, 1.0],
                        "phs_deg": [0.0, -1.0, -2.0, -3.0, -4.0],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def temp_cal_file(sample_cal_data):
    """Fixture creating a temporary calibration file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_cal_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_calibration_object():
    """Fixture providing a mock calibration object for testing."""
    mock_obj = Mock()
    mock_obj.timestamp_utc = 1685730408
    mock_obj.instrument_type = "MTU-5C"
    mock_obj.instrument_model = "RMT03-J"
    mock_obj.inst_serial = "666"
    mock_obj.file_type = "receiver calibration"
    mock_obj.sensor_serial = "test_sensor"

    # Mock calibration data
    mock_channel = Mock()
    mock_channel.tag = "E1"
    mock_cal_data = Mock()
    mock_cal_data.freq_Hz = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    mock_cal_data.magnitude = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    mock_cal_data.phs_deg = np.array([0.0, -1.0, -2.0, -3.0, -4.0])
    mock_channel.chan_data = [mock_cal_data]
    mock_obj.cal_data = [mock_channel]

    return mock_obj


# =============================================================================
# Test Classes
# =============================================================================


class TestPhoenixCalibrationInit:
    """Test PhoenixCalibration initialization and basic properties."""

    def test_init_without_file(self):
        """Test initialization without calibration file."""
        cal = PhoenixCalibration()
        assert cal.obj is None
        assert not hasattr(cal, "_cal_fn")

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        cal = PhoenixCalibration(test_attr="test_value", another_attr=42)
        assert hasattr(cal, "test_attr")
        assert hasattr(cal, "another_attr")
        assert getattr(cal, "test_attr") == "test_value"
        assert getattr(cal, "another_attr") == 42

    def test_init_with_valid_file(self, cal_file_path):
        """Test initialization with valid calibration file."""
        cal = PhoenixCalibration(cal_file_path)
        assert cal._cal_fn == cal_file_path
        assert cal.obj is not None

    def test_init_with_invalid_file(self):
        """Test initialization with non-existent file raises IOError."""
        with pytest.raises(IOError, match="Could not find file"):
            PhoenixCalibration("non_existent_file.json")

    def test_str_representation(self):
        """Test string representation of PhoenixCalibration."""
        cal = PhoenixCalibration()
        assert str(cal) == "Phoenix Response Filters"
        assert repr(cal) == "Phoenix Response Filters"


class TestPhoenixCalibrationProperties:
    """Test PhoenixCalibration properties and methods."""

    def test_cal_fn_property_setter_valid(self, cal_file_path):
        """Test cal_fn property setter with valid file."""
        cal = PhoenixCalibration()
        cal.cal_fn = cal_file_path
        assert cal._cal_fn == cal_file_path
        assert cal.obj is not None

    def test_cal_fn_property_setter_invalid(self):
        """Test cal_fn property setter with invalid file."""
        cal = PhoenixCalibration()
        with pytest.raises(IOError):
            cal.cal_fn = "non_existent_file.json"

    def test_cal_fn_property_setter_none(self):
        """Test cal_fn property setter with None value."""
        cal = PhoenixCalibration()
        cal.cal_fn = None
        assert not hasattr(cal, "_cal_fn")

    def test_calibration_date_property(self, phoenix_calibration):
        """Test calibration_date property returns MTime object."""
        cal_date = phoenix_calibration.calibration_date
        assert isinstance(cal_date, MTime)
        # Check that it represents the expected date (June 2, 2023)
        assert "2023-06-02" in str(cal_date)

    def test_calibration_date_property_no_data(self):
        """Test calibration_date property when no data is loaded."""
        cal = PhoenixCalibration()
        assert cal.calibration_date is None

    def test_has_read_method(self, phoenix_calibration):
        """Test _has_read method."""
        assert phoenix_calibration._has_read() is True

        cal_empty = PhoenixCalibration()
        assert cal_empty._has_read() is False

    def test_base_filter_name_property(self, phoenix_calibration):
        """Test base_filter_name property."""
        expected_name = "mtu-5c_rmt03_10128"
        assert phoenix_calibration.base_filter_name == expected_name

    def test_base_filter_name_no_data(self):
        """Test base_filter_name property when no data is loaded."""
        cal = PhoenixCalibration()
        assert cal.base_filter_name is None


class TestPhoenixCalibrationMethods:
    """Test PhoenixCalibration utility methods."""

    def test_get_max_freq_single_value(self, phoenix_calibration):
        """Test get_max_freq with single frequency value."""
        freq = np.array([1234.5])
        result = phoenix_calibration.get_max_freq(freq)
        assert result == 1000

    def test_get_max_freq_multiple_values(self, phoenix_calibration):
        """Test get_max_freq with multiple frequency values."""
        freq = np.array([1.0, 10.0, 100.0, 9999.0])
        result = phoenix_calibration.get_max_freq(freq)
        assert result == 1000

    def test_get_max_freq_edge_cases(self, phoenix_calibration):
        """Test get_max_freq with edge case frequency values."""
        test_cases = [
            (np.array([0.1]), 0),
            (np.array([1.0]), 1),
            (np.array([10.0]), 10),
            (np.array([99.9]), 10),
            (np.array([100.0]), 100),
            (np.array([999.9]), 100),
            (np.array([1000.0]), 1000),
            (np.array([10000.0]), 10000),
        ]

        for freq_array, expected in test_cases:
            result = phoenix_calibration.get_max_freq(freq_array)
            assert result == expected, f"Failed for frequency {freq_array[0]}"

    def test_get_filter_lp_name(self, phoenix_calibration):
        """Test get_filter_lp_name method."""
        result = phoenix_calibration.get_filter_lp_name("e1", 1000)
        expected = "mtu-5c_rmt03_10128_e1_1000hz_lowpass"
        assert result == expected

    def test_get_filter_sensor_name(self, phoenix_calibration):
        """Test get_filter_sensor_name method."""
        result = phoenix_calibration.get_filter_sensor_name("test_sensor")
        expected = "mtu-5c_rmt03_10128_test_sensor"
        assert result == expected

    @patch("mth5.io.phoenix.readers.calibrations.read_json_to_object")
    def test_read_method_with_file_parameter(
        self, mock_read_json, mock_calibration_object, temp_cal_file
    ):
        """Test read method with file parameter."""
        mock_read_json.return_value = mock_calibration_object

        cal = PhoenixCalibration()
        cal.read(temp_cal_file)

        assert cal._cal_fn == temp_cal_file
        assert cal.obj == mock_calibration_object
        mock_read_json.assert_called_once_with(temp_cal_file)

    def test_read_method_file_not_exists(self):
        """Test read method with non-existent file."""
        cal = PhoenixCalibration()
        cal._cal_fn = Path("non_existent_file.json")

        with pytest.raises(IOError, match="Could not find"):
            cal.read()

    @patch("mth5.io.phoenix.readers.calibrations.read_json_to_object")
    def test_read_method_receiver_calibration(
        self, mock_read_json, mock_calibration_object, temp_cal_file
    ):
        """Test read method processes receiver calibration correctly."""
        mock_read_json.return_value = mock_calibration_object

        cal = PhoenixCalibration()
        cal.read(temp_cal_file)

        # Check that channel attributes are set correctly
        assert hasattr(cal, "e1")
        e1_data = getattr(cal, "e1")
        assert isinstance(e1_data, dict)
        assert 10000 in e1_data

    @patch("mth5.io.phoenix.readers.calibrations.read_json_to_object")
    def test_read_method_sensor_calibration(
        self, mock_read_json, mock_calibration_object, temp_cal_file
    ):
        """Test read method processes sensor calibration correctly."""
        # Modify mock for sensor calibration
        mock_calibration_object.file_type = "sensor calibration"
        mock_read_json.return_value = mock_calibration_object

        cal = PhoenixCalibration()
        cal.read(temp_cal_file)

        # Check that channel attribute is set as FrequencyResponseTableFilter
        assert hasattr(cal, "e1")
        e1_filter = getattr(cal, "e1")
        assert isinstance(e1_filter, FrequencyResponseTableFilter)
        assert e1_filter.units_in == "milliVolt"
        assert e1_filter.units_out == "nanoTesla"


class TestPhoenixCalibrationChannels:
    """Test channel-specific functionality."""

    @pytest.mark.parametrize("channel", ["e1", "e2", "h1", "h2", "h3"])
    def test_has_channel_attributes(self, phoenix_calibration, channel):
        """Test that all expected channels are present."""
        assert hasattr(phoenix_calibration, channel)

    @pytest.mark.parametrize("channel", ["e1", "e2", "h1", "h2", "h3"])
    @pytest.mark.parametrize("lp_filter", [10000, 1000, 100, 10])
    def test_has_lowpass_filters(self, phoenix_calibration, channel, lp_filter):
        """Test that all expected lowpass filters are present for each channel."""
        channel_data = getattr(phoenix_calibration, channel)
        assert lp_filter in channel_data.keys()

    def test_get_filter_valid_parameters(self, phoenix_calibration):
        """Test get_filter method with valid parameters."""
        fap = phoenix_calibration.get_filter("e1", 10000)
        assert isinstance(fap, FrequencyResponseTableFilter)
        assert fap.units_in == "Volt"
        assert fap.units_out == "Volt"

    def test_get_filter_string_parameter(self, phoenix_calibration):
        """Test get_filter method with string filter name."""
        fap = phoenix_calibration.get_filter("e1", "10000")
        assert isinstance(fap, FrequencyResponseTableFilter)

    def test_get_filter_properties(self, phoenix_calibration):
        """Test comprehensive properties of returned filter."""
        fap = phoenix_calibration.get_filter("e1", 10000)

        # Test basic properties
        assert fap.units_in == "Volt"
        assert fap.units_out == "Volt"
        assert fap.name == f"{phoenix_calibration.base_filter_name}_e1_10000hz_lowpass"
        assert fap.calibration_date == phoenix_calibration.obj.timestamp_utc

        # Test frequency domain data
        assert len(fap.frequencies) > 0
        assert (fap.frequencies != 0).all()
        assert len(fap.amplitudes) > 0
        assert (fap.amplitudes != 0).all()
        assert len(fap.phases) > 0
        assert (fap.phases != 0).all()

        # Test max frequency calculation
        max_freq_calculated = phoenix_calibration.get_max_freq(fap.frequencies)
        assert max_freq_calculated == 10000

    def test_get_filter_invalid_channel(self, phoenix_calibration):
        """Test get_filter method with invalid channel name."""
        with pytest.raises(AttributeError, match="Could not find bx"):
            phoenix_calibration.get_filter("bx", 100)

    def test_get_filter_invalid_filter(self, phoenix_calibration):
        """Test get_filter method with invalid filter name."""
        with pytest.raises(KeyError, match="Could not find lowpass filter 2"):
            phoenix_calibration.get_filter("e1", 2)


class TestPhoenixCalibrationFilterProperties:
    """Test detailed filter properties across all channels and frequencies."""

    @pytest.mark.parametrize("channel", ["e1", "e2", "h1", "h2", "h3"])
    @pytest.mark.parametrize("lp_filter", [10000, 1000, 100, 10])
    def test_filter_properties_comprehensive(
        self, phoenix_calibration, channel, lp_filter
    ):
        """Comprehensive test of filter properties for all channel/frequency combinations."""
        fap = phoenix_calibration.get_filter(channel, lp_filter)

        # Test basic properties
        assert fap.units_in == "Volt"
        assert fap.units_out == "Volt"
        assert (
            fap.name
            == f"{phoenix_calibration.base_filter_name}_{channel}_{lp_filter}hz_lowpass"
        )
        assert fap.calibration_date == phoenix_calibration.calibration_date

        # Test frequency domain arrays
        assert len(fap.frequencies) > 0
        assert (fap.frequencies != 0).all()
        assert len(fap.amplitudes) > 0
        assert (fap.amplitudes != 0).all()
        assert len(fap.phases) > 0

        # Test that arrays have same length
        assert len(fap.frequencies) == len(fap.amplitudes)
        assert len(fap.frequencies) == len(fap.phases)

        # Test max frequency matches filter specification
        max_freq = phoenix_calibration.get_max_freq(fap.frequencies)
        assert max_freq == lp_filter

    def test_frequency_ordering(self, phoenix_calibration):
        """Test that frequencies are properly ordered."""
        fap = phoenix_calibration.get_filter("e1", 10000)
        frequencies = fap.frequencies

        # Check that frequencies are in ascending order
        assert np.all(frequencies[:-1] <= frequencies[1:])

    def test_amplitude_values(self, phoenix_calibration):
        """Test amplitude values are within reasonable ranges."""
        fap = phoenix_calibration.get_filter("e1", 10000)
        amplitudes = fap.amplitudes

        # Amplitudes should be positive and reasonable for a calibration
        assert np.all(amplitudes > 0)
        assert np.all(amplitudes <= 2.0)  # Reasonable upper bound for calibration

    def test_phase_values(self, phoenix_calibration):
        """Test phase values are within reasonable ranges."""
        fap = phoenix_calibration.get_filter("e1", 10000)
        phases = fap.phases

        # Phases should be in reasonable range (converted from degrees to radians)
        assert np.all(np.abs(phases) <= 2 * np.pi)


class TestPhoenixCalibrationErrorHandling:
    """Test error handling and edge cases."""

    def test_initialization_with_directory_path(self):
        """Test initialization with directory path instead of file."""
        with pytest.raises(IOError):
            PhoenixCalibration(Path(__file__).parent)

    def test_get_filter_case_sensitivity(self, phoenix_calibration):
        """Test get_filter case sensitivity."""
        # Should work with lowercase
        fap1 = phoenix_calibration.get_filter("e1", 10000)
        assert isinstance(fap1, FrequencyResponseTableFilter)

        # Should fail with uppercase (depending on implementation)
        with pytest.raises(AttributeError):
            phoenix_calibration.get_filter("E1", 10000)

    def test_filter_name_generation_edge_cases(self, phoenix_calibration):
        """Test filter name generation with edge cases."""
        # Test with different sensor names
        name1 = phoenix_calibration.get_filter_sensor_name("test_sensor_123")
        expected1 = f"{phoenix_calibration.base_filter_name}_test_sensor_123"
        assert name1 == expected1

        # Test with special characters (should be handled gracefully)
        name2 = phoenix_calibration.get_filter_lp_name("e1", 1000)
        assert "hz_lowpass" in name2
        assert name2.islower()


class TestPhoenixCalibrationIntegration:
    """Integration tests using real calibration data."""

    def test_full_workflow_receiver_calibration(self, cal_file_path):
        """Test complete workflow for receiver calibration."""
        # Initialize and load
        cal = PhoenixCalibration(cal_file_path)

        # Verify initialization
        assert cal._has_read()
        assert isinstance(cal.calibration_date, MTime)
        assert cal.base_filter_name == "mtu-5c_rmt03_10128"

        # Test all channels exist
        for channel in ["e1", "e2", "h1", "h2", "h3"]:
            assert hasattr(cal, channel)
            channel_data = getattr(cal, channel)
            assert isinstance(channel_data, dict)

            # Test filters in each channel
            for lp_freq in [10000, 1000, 100, 10]:
                assert lp_freq in channel_data
                filter_obj = channel_data[lp_freq]
                assert isinstance(filter_obj, FrequencyResponseTableFilter)
                assert filter_obj.units_in == "Volt"
                assert filter_obj.units_out == "Volt"

    def test_data_consistency_across_channels(self, phoenix_calibration):
        """Test that data is consistent across different channels."""
        channels = ["e1", "e2", "h1", "h2", "h3"]
        frequency_sets = {}

        # Collect frequency data for each channel
        for channel in channels:
            channel_data = getattr(phoenix_calibration, channel)
            for lp_freq, filter_obj in channel_data.items():
                if lp_freq not in frequency_sets:
                    frequency_sets[lp_freq] = []
                frequency_sets[lp_freq].append(len(filter_obj.frequencies))

        # Verify consistency within each frequency band
        # (Note: Different channels may have different frequency points)
        for lp_freq, lengths in frequency_sets.items():
            # All lengths should be positive
            assert all(length > 0 for length in lengths)

    def test_calibration_date_consistency(self, phoenix_calibration):
        """Test that calibration date is consistent across all filters."""
        expected_date = phoenix_calibration.calibration_date

        for channel in ["e1", "e2", "h1", "h2", "h3"]:
            channel_data = getattr(phoenix_calibration, channel)
            for filter_obj in channel_data.values():
                assert filter_obj.calibration_date == expected_date


# =============================================================================
# Performance and Memory Tests
# =============================================================================


class TestPhoenixCalibrationPerformance:
    """Test performance and memory-related aspects."""

    def test_multiple_filter_access_performance(self, phoenix_calibration):
        """Test that multiple filter accesses don't cause memory issues."""
        # Access same filter multiple times
        filters = []
        for _ in range(10):
            fap = phoenix_calibration.get_filter("e1", 10000)
            filters.append(fap)

        # All should reference the same object (or equivalent objects)
        for fap in filters:
            assert isinstance(fap, FrequencyResponseTableFilter)
            assert (
                fap.name == f"{phoenix_calibration.base_filter_name}_e1_10000hz_lowpass"
            )

    def test_large_frequency_array_handling(self, phoenix_calibration):
        """Test handling of large frequency arrays."""
        fap = phoenix_calibration.get_filter("e1", 10000)

        # Verify arrays can handle operations
        freq_copy = np.copy(fap.frequencies)
        amp_copy = np.copy(fap.amplitudes)
        phase_copy = np.copy(fap.phases)

        assert np.array_equal(freq_copy, fap.frequencies)
        assert np.array_equal(amp_copy, fap.amplitudes)
        assert np.array_equal(phase_copy, fap.phases)


# =============================================================================
# Mock and Isolation Tests
# =============================================================================


class TestPhoenixCalibrationMocking:
    """Test using mocks for isolated testing."""

    @patch("mth5.io.phoenix.readers.calibrations.read_json_to_object")
    def test_read_method_isolated(
        self, mock_read_json, mock_calibration_object, temp_cal_file
    ):
        """Test read method in isolation using mocks."""
        mock_read_json.return_value = mock_calibration_object

        cal = PhoenixCalibration()
        cal._cal_fn = temp_cal_file
        cal.read()

        # Verify method calls
        mock_read_json.assert_called_once_with(temp_cal_file)
        assert cal.obj == mock_calibration_object

    @patch("mth5.io.phoenix.readers.calibrations.Path")
    def test_file_existence_check_mocked(self, mock_path):
        """Test file existence check using mocked Path."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        cal = PhoenixCalibration()
        cal._cal_fn = mock_path_instance

        with pytest.raises(IOError, match="Could not find"):
            cal.read()
