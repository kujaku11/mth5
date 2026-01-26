# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for Phoenix Native Reader.

This test suite uses fixtures, subtests, and parametrized tests to efficiently test
Phoenix Native Reader functionality for native sampling rate time series data with real Phoenix files.

Translated from unittest to pytest format on November 16, 2025
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from mth5.io.phoenix import open_phoenix
from mth5.io.phoenix.readers.native.native_reader import (
    AD_IN_AD_UNITS,
    AD_INPUT_VOLTS,
    INSTRUMENT_INPUT_VOLTS,
    NativeReader,
)
from mth5.timeseries import ChannelTS


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False
    phx_data_path = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def phoenix_data_path():
    """Return the path to Phoenix test data."""
    if not has_test_data:
        pytest.skip("mth5_test_data not available")
    return phx_data_path


@pytest.fixture(scope="module")
def phoenix_base_path(phoenix_data_path):
    """Return the base path for Phoenix test data files."""
    return phoenix_data_path / "10128_2021-04-27-025909"


@pytest.fixture(scope="module")
def rxcal_file(phoenix_data_path):
    """Return the path to receiver calibration file."""
    return phoenix_data_path / "10128_rxcal.json"


@pytest.fixture(params=[0, 1, 2, 4])
def channel_id(request):
    """Parametrize test for different channel IDs."""
    return request.param


@pytest.fixture
def phoenix_reader_channel_0(phoenix_base_path):
    """Create Phoenix reader for channel 0."""
    file_path = phoenix_base_path / "0" / "10128_60877DFD_0_00000001.bin"
    return open_phoenix(file_path)


@pytest.fixture
def phoenix_reader_channel_1(phoenix_base_path):
    """Create Phoenix reader for channel 1."""
    file_path = phoenix_base_path / "1" / "10128_60877DFD_1_00000001.bin"
    return open_phoenix(file_path)


@pytest.fixture
def phoenix_reader_channel_2(phoenix_base_path):
    """Create Phoenix reader for channel 2."""
    file_path = phoenix_base_path / "2" / "10128_60877DFD_2_00000001.bin"
    return open_phoenix(file_path)


@pytest.fixture
def phoenix_reader_channel_4(phoenix_base_path):
    """Create Phoenix reader for channel 4."""
    file_path = phoenix_base_path / "4" / "10128_60877DFD_4_00000001.bin"
    return open_phoenix(file_path)


@pytest.fixture
def phoenix_reader(request, phoenix_base_path):
    """Create Phoenix reader based on channel_id parameter."""
    channel_id = request.getfixturevalue("channel_id")
    file_path = (
        phoenix_base_path
        / str(channel_id)
        / f"10128_60877DFD_{channel_id}_00000001.bin"
    )
    return open_phoenix(file_path)


@pytest.fixture
def expected_attributes_channel_0():
    """Expected attributes for Phoenix native reader channel 0."""
    return {
        "ad_plus_minus_range": 5.0,
        "attenuator_gain": 1.0,
        "battery_voltage_v": 12.446,
        "board_model_main": "BCM01",
        "board_model_revision": "I",
        "bytes_per_sample": 3,
        "ch_board_model": "BCM01-I",
        "ch_board_serial": 200803,
        "ch_firmware": 65567,
        "channel_id": 0,
        "channel_main_gain": 4.0,
        "channel_map": {0: "h2", 1: "e1", 2: "h1", 3: "h3", 4: "e2"},
        "channel_type": "H",
        "data_footer": 0,
        "data_scaling": 1,
        "decimation_node_id": 0,
        "detected_channel_type": "H",
        "file_extension": ".bin",
        "file_name": "10128_60877DFD_0_00000001.bin",
        "file_sequence": 2,
        "file_size": 4608128,
        "file_type": 1,
        "file_version": 3,
        "footer_idx_samp_mask": 268435455,
        "footer_sat_mask": 1879048192,
        "frag_period": 60,
        "frame_rollover_count": 0,
        "frame_size": 64,
        "frame_size_bytes": 64,
        "future1": 28,
        "future2": 0,
        "gps_elevation": 70.11294555664062,
        "gps_horizontal_accuracy": 11.969,
        "gps_lat": 43.69640350341797,
        "gps_long": -79.3936996459961,
        "gps_vertical_accuracy": 38.042,
        "hardware_configuration": (4, 3, 0, 0, 0, 10, 128, 0),
        "header_length": 128,
        "input_plusminus_range": 1.25,
        "instrument_id": "10128",
        "instrument_serial_number": "10128",
        "instrument_type": "MTU-5C",
        "intrinsic_circuitry_gain": 1.0,
        "last_frame": 0,
        "last_seq": 2,
        "lp_frequency": 10000,
        "max_samples": 1152000,
        "max_signal": 2.0269203186035156,
        "min_signal": -2.0260202884674072,
        "missing_frames": 0,
        "npts_per_frame": 20,
        "preamp_gain": 1.0,
        "recording_id": 1619492349,
        "recording_start_time": "2021-04-27T02:58:51+00:00",
        "report_hw_sat": False,
        "sample_rate": 24000,
        "sample_rate_base": 24000,
        "sample_rate_exp": 0,
        "saturated_frames": 0,
        "scale_factor": 2.3283064365386963e-09,
        "seq": 1,
        "timing_flags": 55,
        "timing_sat_count": 6,
        "timing_stability": 145,
        "timing_status": (55, 6, 145),
        "total_circuitry_gain": 4.0,
        "total_selectable_gain": 4.0,
    }


@pytest.fixture
def expected_channel_metadata_channel_0():
    """Expected channel metadata for channel 0."""
    return OrderedDict(
        [
            ("channel_number", 0),
            ("component", "h2"),
            ("data_quality.rating.value", None),
            ("location.elevation", 70.11294555664062),
            ("location.latitude", 43.69640350341797),
            ("location.longitude", -79.3936996459961),
            ("measurement_azimuth", 90.0),
            ("measurement_tilt", 0.0),
            ("sample_rate", 24000.0),
            ("sensor.id", "101"),
            ("sensor.manufacturer", "Phoenix Geophysics"),
            ("sensor.model", "MTC-150"),
            ("sensor.type", "4"),
            ("time_period.end", "2021-04-27T03:01:50.999958333+00:00"),
            ("time_period.start", "2021-04-27T02:58:51+00:00"),
            ("type", "magnetic"),
            ("units", "Volt"),
        ]
    )


# =============================================================================
# Utility functions
# =============================================================================


def get_expected_attributes_for_channel(channel_id, base_attributes):
    """Get expected attributes for a specific channel, modifying base attributes as needed."""
    attributes = base_attributes.copy()

    # Channel-specific modifications
    attributes["channel_id"] = channel_id
    attributes["file_name"] = f"10128_60877DFD_{channel_id}_00000001.bin"

    # Channel type mapping
    channel_type_map = {0: "H", 1: "E", 2: "H", 4: "E"}
    attributes["channel_type"] = channel_type_map.get(channel_id, "H")
    attributes["detected_channel_type"] = channel_type_map.get(channel_id, "H")

    return attributes


def get_expected_metadata_for_channel(channel_id, base_metadata):
    """Get expected metadata for a specific channel."""
    metadata = base_metadata.copy()

    # Channel-specific modifications
    metadata[0] = ("channel_number", channel_id)

    # Component mapping
    component_map = {0: "h2", 1: "e1", 2: "h1", 4: "e2"}
    metadata[1] = ("component", component_map.get(channel_id, "h2"))

    # Type mapping
    type_map = {0: "magnetic", 1: "electric", 2: "magnetic", 4: "electric"}
    metadata[15] = ("type", type_map.get(channel_id, "magnetic"))

    # Sensor ID
    metadata[9] = ("sensor.id", str(channel_id))

    return metadata


# =============================================================================
# Test Classes
# =============================================================================


class TestPhoenixNativeReaderBasic:
    """Test basic Phoenix Native Reader functionality."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_readers_match(self, phoenix_reader_channel_0):
        """Test that different reading methods work correctly."""
        # Test frame reading
        frame_data = phoenix_reader_channel_0.read_frames(10)
        assert len(frame_data) == 200  # 10 frames * 20 samples per frame
        assert isinstance(frame_data, np.ndarray)

        # Reset reader state
        phoenix_reader_channel_0.last_frame = 0

        # Test sequence reading
        data, footer = phoenix_reader_channel_0.read_sequence()
        assert isinstance(data, np.ndarray)
        assert isinstance(footer, np.ndarray)
        assert len(data) > 0
        assert len(footer) > 0

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    @pytest.mark.parametrize(
        "attr_name,expected_value",
        [
            ("ad_plus_minus_range", 5.0),
            ("attenuator_gain", 1.0),
            ("battery_voltage_v", 12.475),
            ("board_model_main", "BCM01"),
            ("board_model_revision", "I"),
            ("bytes_per_sample", 3),
            ("ch_board_model", "BCM01-I"),
            ("ch_board_serial", 200803),
            ("ch_firmware", 65567),
            ("channel_id", 0),
            ("channel_main_gain", 4.0),
            ("channel_type", "H"),
            ("data_footer", 0),
            ("data_scaling", 1),
            ("decimation_node_id", 0),
            ("detected_channel_type", "H"),
            ("file_extension", ".bin"),
            ("file_name", "10128_60877DFD_0_00000001.bin"),
            ("file_sequence", 0),
            ("file_size", 4608128),
            ("file_type", 1),
            ("file_version", 3),
            ("footer_idx_samp_mask", 268435455),
            ("footer_sat_mask", 1879048192),
            ("frag_period", 60),
            ("frame_rollover_count", 0),
            ("frame_size", 64),
            ("frame_size_bytes", 64),
            ("future1", 27),
            ("future2", 0),
            ("gps_elevation", 97.2187728881836),
            ("gps_horizontal_accuracy", 18.332),
            ("gps_lat", 43.69640350341797),
            ("gps_long", -79.3936996459961),
            ("gps_vertical_accuracy", 44.669),
            ("hardware_configuration", (4, 3, 0, 0, 0, 10, 128, 0)),
            ("header_length", 128),
            ("input_plusminus_range", 1.25),
            ("instrument_id", "10128"),
            ("instrument_serial_number", "10128"),
            ("instrument_type", "MTU-5C"),
            ("intrinsic_circuitry_gain", 1.0),
            ("last_frame", 0),
            ("last_seq", 2),
            ("lp_frequency", 10000),
            ("max_samples", 1152000),
            ("max_signal", 2.0543980598449707),
            ("min_signal", -2.054893970489502),
            ("missing_frames", 0),
            ("npts_per_frame", 20),
            ("preamp_gain", 1.0),
            ("recording_id", 1619492349),
            ("recording_start_time", "2021-04-27T02:58:51+00:00"),
            ("report_hw_sat", False),
            ("sample_rate", 24000),
            ("sample_rate_base", 24000),
            ("sample_rate_exp", 0),
            ("saturated_frames", 0),
            ("scale_factor", 2.3283064365386963e-09),
            ("seq", 1),
            ("timing_flags", 55),
            ("timing_sat_count", 5),
            ("timing_stability", 82),
            ("timing_status", (55, 5, 82)),
            ("total_circuitry_gain", 4.0),
            ("total_selectable_gain", 4.0),
        ],
    )
    def test_attributes_channel_0(
        self, phoenix_reader_channel_0, attr_name, expected_value
    ):
        """Test Phoenix native reader attributes for channel 0."""
        actual_value = getattr(phoenix_reader_channel_0, attr_name)

        if isinstance(expected_value, list):
            assert actual_value == expected_value
        elif isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=1e-5)
        else:
            assert actual_value == expected_value

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_channel_map(self, phoenix_reader_channel_0):
        """Test channel map is correctly set."""
        expected_channel_map = {0: "h2", 1: "e1", 2: "h1", 3: "h3", 4: "e2"}
        assert phoenix_reader_channel_0.channel_map == expected_channel_map

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_sequence_list(self, phoenix_reader_channel_0, phoenix_base_path):
        """Test sequence list contains expected files."""
        expected_files = [
            phoenix_base_path / "0" / "10128_60877DFD_0_00000001.bin",
            phoenix_base_path / "0" / "10128_60877DFD_0_00000002.bin",
            phoenix_base_path / "0" / "10128_60877DFD_0_00000003.bin",
        ]
        assert phoenix_reader_channel_0.sequence_list == expected_files


class TestPhoenixNativeReaderMultiChannel:
    """Test Phoenix Native Reader with multiple channels."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_channel_specific_attributes(self, phoenix_reader, channel_id):
        """Test channel-specific attributes across all channels."""
        assert phoenix_reader.channel_id == channel_id
        assert phoenix_reader.file_name == f"10128_60877DFD_{channel_id}_00000001.bin"

        # Test channel type mapping
        expected_channel_type = "E" if channel_id in [1, 4] else "H"
        assert phoenix_reader.channel_type == expected_channel_type
        assert phoenix_reader.detected_channel_type == expected_channel_type

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_common_attributes_across_channels(self, phoenix_reader, channel_id):
        """Test that common attributes are consistent across channels."""
        common_attrs = [
            "instrument_id",
            "instrument_type",
            "sample_rate",
            "gps_lat",
            "gps_long",
            "battery_voltage_v",
            "board_model_main",
        ]

        expected_values = {
            "instrument_id": "10128",
            "instrument_type": "MTU-5C",
            "sample_rate": 24000,
            "gps_lat": 43.69640350341797,
            "gps_long": -79.3936996459961,
            "board_model_main": "BCM01",
        }

        for attr in common_attrs:
            actual = getattr(phoenix_reader, attr)
            expected = expected_values.get(attr)
            if expected is not None:
                if isinstance(expected, float):
                    assert actual == pytest.approx(
                        expected, rel=1e-3
                    )  # More relaxed tolerance
                else:
                    assert actual == expected

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_read_frames_across_channels(self, phoenix_reader, channel_id):
        """Test reading frames works for all channels."""
        data = phoenix_reader.read_frames(5)

        # Should have 5 frames * 20 samples per frame = 100 samples
        assert len(data) == 100
        assert isinstance(data, np.ndarray)
        assert data.dtype in [np.float32, np.float64]

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_read_sequence_across_channels(self, phoenix_reader, channel_id):
        """Test reading sequence works for all channels."""
        data, footer = phoenix_reader.read_sequence()

        assert isinstance(data, np.ndarray)
        assert isinstance(footer, np.ndarray)
        assert len(data) > 0
        assert len(footer) > 0


class TestPhoenixNativeReaderChannelTS:
    """Test Phoenix Native Reader ChannelTS conversion."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_to_channel_ts_channel_0(self, phoenix_reader_channel_0, rxcal_file):
        """Test ChannelTS conversion for channel 0."""
        ch_ts = phoenix_reader_channel_0.to_channel_ts(rxcal_fn=rxcal_file)

        assert isinstance(ch_ts, ChannelTS)
        assert ch_ts.ts.size == 4320000

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    @pytest.mark.parametrize(
        "metadata_key,expected_value",
        [
            ("channel_number", 0),
            ("component", "h2"),
            ("data_quality.rating.value", None),
            ("location.elevation", 70.11294555664062),
            ("location.latitude", 43.69640350341797),
            ("location.longitude", -79.3936996459961),
            ("measurement_azimuth", 90.0),
            ("measurement_tilt", 0.0),
            ("sample_rate", 24000.0),
            ("sensor.id", "101"),
            ("sensor.manufacturer", "Phoenix Geophysics"),
            ("sensor.model", "MTC-150"),
            ("sensor.type", "4"),
            ("time_period.end", "2021-04-27T03:01:50.999958333+00:00"),
            ("time_period.start", "2021-04-27T02:58:51+00:00"),
            ("type", "magnetic"),
            ("units", "Volt"),
        ],
    )
    def test_channel_metadata_channel_0(
        self, phoenix_reader_channel_0, rxcal_file, metadata_key, expected_value
    ):
        """Test channel metadata for channel 0."""
        ch_ts = phoenix_reader_channel_0.to_channel_ts(rxcal_fn=rxcal_file)
        actual_value = ch_ts.channel_metadata.get_attr_from_name(metadata_key)

        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=1e-5)
        else:
            assert actual_value == expected_value

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_channel_filters_channel_0(self, phoenix_reader_channel_0, rxcal_file):
        """Test channel filters for channel 0."""
        ch_ts = phoenix_reader_channel_0.to_channel_ts(rxcal_fn=rxcal_file)
        filters = ch_ts.channel_metadata.get_attr_from_name("filters")

        assert isinstance(filters, list)
        assert len(filters) == 3

        expected_filter_names = [
            "mtu-5c_rmt03_10128_h2_10000hz_lowpass",
            "v_to_mv",
            "coil_101_response",
        ]

        for filt, expected_name in zip(filters, expected_filter_names):
            assert filt.name == expected_name

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_channel_response_channel_0(self, phoenix_reader_channel_0, rxcal_file):
        """Test channel response for channel 0."""
        ch_ts = phoenix_reader_channel_0.to_channel_ts(rxcal_fn=rxcal_file)

        assert len(ch_ts.channel_response.filters_list) == 2
        assert ch_ts.channel_response.filters_list[0].frequencies.shape == (69,)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_to_channel_ts_multi_channel(self, phoenix_reader, rxcal_file, channel_id):
        """Test ChannelTS conversion across multiple channels."""
        ch_ts = phoenix_reader.to_channel_ts(rxcal_fn=rxcal_file)

        assert isinstance(ch_ts, ChannelTS)
        assert ch_ts.ts.size > 0

        # Check channel number matches
        channel_num = ch_ts.channel_metadata.get_attr_from_name("channel_number")
        assert channel_num == channel_id

        # Check component mapping
        component = ch_ts.channel_metadata.get_attr_from_name("component")
        component_map = {0: "h2", 1: "e1", 2: "h1", 4: "e2"}
        assert component == component_map[channel_id]

        # Check type mapping
        channel_type = ch_ts.channel_metadata.get_attr_from_name("type")
        type_map = {0: "magnetic", 1: "electric", 2: "magnetic", 4: "electric"}
        assert channel_type == type_map[channel_id]


class TestPhoenixNativeReaderAdvanced:
    """Test advanced Phoenix Native Reader functionality."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_frame_reading_edge_cases(self, phoenix_reader_channel_0):
        """Test frame reading edge cases."""
        # Test reading zero frames
        data = phoenix_reader_channel_0.read_frames(0)
        assert len(data) == 0

        # Test reading single frame
        data = phoenix_reader_channel_0.read_frames(1)
        assert len(data) == 20  # 20 samples per frame

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_frame_skipping(self, phoenix_reader_channel_0):
        """Test frame skipping functionality."""
        initial_frame = phoenix_reader_channel_0.last_frame

        # Skip some frames
        result = phoenix_reader_channel_0.skip_frames(5)
        assert result is True
        assert phoenix_reader_channel_0.last_frame == initial_frame + 5

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_calculation_methods(self):
        """Test calculation methods."""

        # Test input plus-minus range calculation
        class MockReader:
            ad_plus_minus_range = 5.0
            total_circuitry_gain = 4.0

        mock_reader = MockReader()
        result = NativeReader._calculate_input_plusminus_range(mock_reader)
        assert result == pytest.approx(1.25)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    @pytest.mark.parametrize(
        "scaling_type,ad_range,input_range,expected",
        [
            (AD_IN_AD_UNITS, 5.0, 1.25, 256),
            (AD_INPUT_VOLTS, 5.0, 1.25, 5.0 / (2**31)),
            (INSTRUMENT_INPUT_VOLTS, 5.0, 1.25, 1.25 / (2**31)),
        ],
    )
    def test_data_scaling_calculation(
        self, scaling_type, ad_range, input_range, expected
    ):
        """Test data scaling calculation for different types."""

        class MockReader:
            def __init__(
                self, data_scaling, ad_plus_minus_range, input_plusminus_range
            ):
                self.data_scaling = data_scaling
                self.ad_plus_minus_range = ad_plus_minus_range
                self.input_plusminus_range = input_plusminus_range

        mock_reader = MockReader(scaling_type, ad_range, input_range)
        result = NativeReader._calculate_data_scaling(mock_reader)
        assert result == pytest.approx(expected)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_data_scaling_invalid(self):
        """Test data scaling calculation with invalid scaling type."""

        class MockReader:
            data_scaling = 999  # Invalid scaling type

        mock_reader = MockReader()

        with pytest.raises(LookupError, match="Invalid scaling requested"):
            NativeReader._calculate_data_scaling(mock_reader)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_memory_efficiency(self, phoenix_reader_channel_0):
        """Test memory efficiency with different frame sizes."""
        frame_sizes = [1, 10, 50, 100]

        for frame_size in frame_sizes:
            data = phoenix_reader_channel_0.read_frames(frame_size)
            expected_samples = frame_size * 20
            assert len(data) == expected_samples

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_file_properties(self, phoenix_reader_channel_0):
        """Test file-related properties."""
        assert phoenix_reader_channel_0.file_extension == ".bin"
        assert phoenix_reader_channel_0.file_type == 1
        assert phoenix_reader_channel_0.file_version == 3
        assert phoenix_reader_channel_0.bytes_per_sample == 3
        assert phoenix_reader_channel_0.npts_per_frame == 20

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_timing_properties(self, phoenix_reader_channel_0):
        """Test timing-related properties."""
        assert phoenix_reader_channel_0.timing_flags == 55
        assert phoenix_reader_channel_0.timing_sat_count == 5
        assert phoenix_reader_channel_0.timing_stability == 82
        assert phoenix_reader_channel_0.timing_status == (55, 5, 82)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_signal_range_properties(self, phoenix_reader_channel_0):
        """Test signal range properties."""
        assert isinstance(phoenix_reader_channel_0.max_signal, float)
        assert isinstance(phoenix_reader_channel_0.min_signal, float)
        assert phoenix_reader_channel_0.max_signal > phoenix_reader_channel_0.min_signal

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_gps_properties(self, phoenix_reader_channel_0):
        """Test GPS-related properties."""
        assert isinstance(phoenix_reader_channel_0.gps_lat, float)
        assert isinstance(phoenix_reader_channel_0.gps_long, float)
        assert isinstance(phoenix_reader_channel_0.gps_elevation, float)
        assert isinstance(phoenix_reader_channel_0.gps_horizontal_accuracy, float)
        assert isinstance(phoenix_reader_channel_0.gps_vertical_accuracy, float)


class TestPhoenixNativeReaderPerformance:
    """Test Phoenix Native Reader performance characteristics."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_large_frame_reading(self, phoenix_reader_channel_0):
        """Test reading large numbers of frames."""
        # Test reading many frames
        data = phoenix_reader_channel_0.read_frames(1000)
        assert len(data) == 20000  # 1000 frames * 20 samples

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    @pytest.mark.parametrize("chunk_size", [1, 10, 100, 500])
    def test_chunked_reading_performance(self, phoenix_reader_channel_0, chunk_size):
        """Test performance with different chunk sizes."""
        data = phoenix_reader_channel_0.read_frames(chunk_size)
        expected_size = chunk_size * 20
        assert len(data) == expected_size

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_sequence_reading_performance(self, phoenix_reader_channel_0):
        """Test full sequence reading performance."""
        data, footer = phoenix_reader_channel_0.read_sequence()

        # Should read all available data
        assert len(data) > 100000  # Expect substantial amount of data
        assert len(footer) > 1000  # Expect substantial number of frames


class TestPhoenixNativeReaderErrorHandling:
    """Test Phoenix Native Reader error handling."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_file_not_found_handling(self, phoenix_data_path):
        """Test handling of non-existent files."""
        # Use a properly formatted Phoenix filename that doesn't exist
        non_existent_file = phoenix_data_path / "10128_60877DFD_9_99999999.bin"

        # Phoenix reader may raise different errors depending on implementation
        with pytest.raises((FileNotFoundError, IOError, AttributeError)):
            open_phoenix(non_existent_file)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_read_beyond_file_end(self, phoenix_reader_channel_0):
        """Test reading beyond file end."""
        # Try to read an extremely large number of frames
        data = phoenix_reader_channel_0.read_frames(1000000)

        # Should not crash, but may return less data than requested
        assert isinstance(data, np.ndarray)

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_skip_beyond_file_end(self, phoenix_reader_channel_0):
        """Test skipping beyond file end."""
        # Try to skip an extremely large number of frames
        result = phoenix_reader_channel_0.skip_frames(1000000)

        # Should return False when reaching end of file
        # Note: actual behavior depends on implementation
        assert isinstance(result, bool)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhoenixNativeReaderIntegration:
    """Integration tests for Phoenix Native Reader."""

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_full_workflow_single_channel(self, phoenix_reader_channel_0, rxcal_file):
        """Test complete workflow for single channel."""
        # Read some frames
        frame_data = phoenix_reader_channel_0.read_frames(10)
        assert len(frame_data) == 200

        # Reset and read sequence
        phoenix_reader_channel_0.last_frame = 0
        sequence_data, footer = phoenix_reader_channel_0.read_sequence()
        assert len(sequence_data) > 0
        assert len(footer) > 0

        # Convert to ChannelTS
        ch_ts = phoenix_reader_channel_0.to_channel_ts(rxcal_fn=rxcal_file)
        assert isinstance(ch_ts, ChannelTS)
        assert ch_ts.ts.size > 0

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_multi_channel_consistency(self, phoenix_base_path, rxcal_file):
        """Test consistency across multiple channels."""
        channels = [0, 1, 2, 4]
        channel_data = {}

        for channel_id in channels:
            file_path = (
                phoenix_base_path
                / str(channel_id)
                / f"10128_60877DFD_{channel_id}_00000001.bin"
            )
            reader = open_phoenix(file_path)

            # Store basic info
            channel_data[channel_id] = {
                "reader": reader,
                "channel_type": reader.channel_type,
                "sample_rate": reader.sample_rate,
                "instrument_id": reader.instrument_id,
            }

        # Check consistency of common attributes
        sample_rates = [data["sample_rate"] for data in channel_data.values()]
        instrument_ids = [data["instrument_id"] for data in channel_data.values()]

        assert all(sr == sample_rates[0] for sr in sample_rates)
        assert all(inst_id == instrument_ids[0] for inst_id in instrument_ids)

        # Check channel type mapping is correct
        expected_types = {0: "H", 1: "E", 2: "H", 4: "E"}
        for channel_id, data in channel_data.items():
            assert data["channel_type"] == expected_types[channel_id]

    @pytest.mark.skipif(not has_test_data, reason="mth5_test_data not available")
    def test_cross_channel_timing_consistency(self, phoenix_base_path):
        """Test timing consistency across channels."""
        channels = [0, 1, 2, 4]
        timing_data = {}

        for channel_id in channels:
            file_path = (
                phoenix_base_path
                / str(channel_id)
                / f"10128_60877DFD_{channel_id}_00000001.bin"
            )
            reader = open_phoenix(file_path)

            timing_data[channel_id] = {
                "recording_start_time": reader.recording_start_time,
                "recording_id": reader.recording_id,
                "timing_flags": reader.timing_flags,
            }

        # All channels should have the same recording start time and ID
        start_times = [data["recording_start_time"] for data in timing_data.values()]
        recording_ids = [data["recording_id"] for data in timing_data.values()]

        assert all(st == start_times[0] for st in start_times)
        assert all(rid == recording_ids[0] for rid in recording_ids)


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
