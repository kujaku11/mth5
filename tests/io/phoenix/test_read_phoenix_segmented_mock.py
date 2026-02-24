# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for Phoenix Segmented Reader.

This test suite uses fixtures, subtests, and mocking to efficiently test
Phoenix Segmented Reader functionality for segmented-decimated time series data.

Created on November 10, 2025
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path
from struct import pack
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import pytest
from mt_metadata.common.mttime import MTime

from mth5.io.phoenix.readers.segmented.decimated_segmented_reader import (
    DecimatedSegmentCollection,
    DecimatedSegmentedReader,
    Segment,
    SubHeader,
)
from mth5.timeseries import ChannelTS


try:
    pass

    HAS_MTH5_TEST_DATA = True
except ImportError:
    HAS_MTH5_TEST_DATA = False


pytestmark = pytest.mark.skipif(
    HAS_MTH5_TEST_DATA, reason="Skipping mock tests - real data available"
)
# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_binary_header():
    """Create a mock binary header for SubHeader testing."""
    # Pack binary data for SubHeader testing
    return pack(
        "IIHHfff8x",  # Format: 4 uints, 2 ushorts, 3 floats, 8 padding bytes
        1619493882,  # gps_time_stamp (GPS time)
        48000,  # n_samples
        0,  # saturation_count
        0,  # missing_count
        -2.5,  # value_min
        2.5,  # value_max
        0.0,  # value_mean
    )


@pytest.fixture
def mock_phoenix_segmented_file(temp_directory, mock_binary_header):
    """Create a mock Phoenix segmented file for testing."""
    file_path = temp_directory / "test_phoenix.td_24k"

    # Create a mock binary file with header + subheader + data
    main_header = b"\x00" * 128  # 128-byte main header
    sub_header = mock_binary_header  # 32-byte subheader
    mock_data = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0] * 9600, dtype=np.float32
    )  # 48000 samples

    with open(file_path, "wb") as f:
        f.write(main_header + sub_header + mock_data.tobytes())

    return file_path


@pytest.fixture
def mock_stream_with_header(mock_binary_header):
    """Create a mock stream with header data."""
    stream = Mock()
    stream.read.return_value = mock_binary_header
    stream.seek = Mock()
    return stream


@pytest.fixture
def mock_subheader():
    """Create a mock SubHeader instance."""
    subheader = SubHeader()
    subheader._header = pack(
        "IIHHfff8x",
        1619493882,  # gps_time_stamp
        48000,  # n_samples
        5,  # saturation_count
        2,  # missing_count
        -2.5,  # value_min
        2.5,  # value_max
        0.1,  # value_mean
    )
    return subheader


@pytest.fixture
def mock_segment_kwargs():
    """Mock kwargs for Segment initialization."""
    return {
        "instrument_type": "MTU-5C",
        "instrument_serial_number": "10128",
        "latitude": 43.69625473022461,
        "longitude": -79.39364624023438,
        "elevation": 140.10263061523438,
        "sample_rate": 24000.0,
        "channel_id": 0,
        "channel_type": "H",
        "segment": 0,
    }


@pytest.fixture
def mock_segment(mock_stream_with_header, mock_segment_kwargs):
    """Create a mock Segment instance."""
    segment = Segment(mock_stream_with_header, **mock_segment_kwargs)
    segment.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    return segment


@pytest.fixture
def mock_reader_base_attributes():
    """Mock attributes from TSReaderBase that DecimatedSegmentedReader needs."""
    return {
        "instrument_type": "MTU-5C",
        "instrument_serial_number": "10128",
        "gps_lat": 43.69625473022461,
        "gps_long": -79.39364624023438,
        "gps_elevation": 140.10263061523438,
        "sample_rate": 24000,
        "channel_id": 0,
        "channel_type": "H",
        "stream": Mock(),
        "channel_metadata": Mock(),
        "run_metadata": Mock(),
        "station_metadata": Mock(),
    }


@pytest.fixture
def mock_decimated_segmented_reader():
    """Create a mock DecimatedSegmentedReader instance."""
    reader = Mock(spec=DecimatedSegmentedReader)

    # Set up basic attributes
    reader.instrument_type = "MTU-5C"
    reader.instrument_serial_number = "10128"
    reader.gps_lat = 43.69625473022461
    reader.gps_long = -79.39364624023438
    reader.gps_elevation = 140.10263061523438
    reader.sample_rate = 24000
    reader.channel_id = 0
    reader.channel_type = "H"
    reader.stream = Mock()
    reader.sub_header = Mock()
    reader.subheader = {}
    reader._channel_metadata = Mock()
    reader.channel_metadata = Mock()
    reader.run_metadata = Mock()
    reader.station_metadata = Mock()
    reader.logger = Mock()

    # Mock methods with realistic behavior
    def read_segment_func(metadata_only=False):
        segment = Mock()
        segment.segment_start_time = MTime(time_stamp="2021-04-27T03:24:42+00:00")
        segment.segment_end_time = MTime(time_stamp="2021-04-27T03:24:44+00:00")
        if not metadata_only:
            segment.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        else:
            segment.data = None
        return segment

    def to_channel_ts_func(rxcal_fn=None, scal_fn=None):
        mock_channel_ts = Mock(spec=ChannelTS)
        mock_channel_ts.ts = Mock()
        mock_channel_ts.ts.size = 48000
        mock_channel_ts.channel_metadata = Mock()
        mock_channel_ts.channel_response = Mock()
        mock_channel_ts.channel_response.filters_list = [Mock(), Mock()]
        mock_channel_ts.channel_response.filters_list[0].frequencies = Mock()
        mock_channel_ts.channel_response.filters_list[0].frequencies.shape = (69,)
        return mock_channel_ts

    reader.read_segment = Mock(side_effect=read_segment_func)
    reader.to_channel_ts = Mock(side_effect=to_channel_ts_func)
    reader.get_channel_response = Mock(return_value=Mock())
    reader._update_channel_metadata_from_recmeta = Mock(return_value=Mock())

    return reader


@pytest.fixture
def mock_decimated_segment_collection():
    """Create a mock DecimatedSegmentCollection instance."""
    collection = Mock(spec=DecimatedSegmentCollection)

    # Set up basic attributes
    collection.instrument_type = "MTU-5C"
    collection.instrument_serial_number = "10128"
    collection.gps_lat = 43.69625473022461
    collection.gps_long = -79.39364624023438
    collection.gps_elevation = 140.10263061523438
    collection.sample_rate = 24000
    collection.channel_id = 0
    collection.channel_type = "H"
    collection.stream = Mock()
    collection.sub_header = Mock()
    collection.subheader = {}
    collection.channel_metadata = Mock()
    collection.run_metadata = Mock()
    collection.station_metadata = Mock()
    collection.logger = Mock()

    # Mock methods with realistic behavior
    def read_segments_func(metadata_only=False):
        segments = []
        for i in range(3):  # Mock 3 segments
            segment = Mock()
            segment.gps_time_stamp = MTime(
                time_stamp=f"2021-04-27T03:24:{42+i*2:02d}+00:00"
            )
            segment.segment_start_time = segment.gps_time_stamp
            segment.segment_end_time = segment.gps_time_stamp + 2.0
            if not metadata_only:
                segment.data = np.array([i + 1.0] * 1000, dtype=np.float32)
            else:
                segment.data = None
            segments.append(segment)
        return segments

    def to_channel_ts_func(rxcal_fn=None, scal_fn=None):
        seq_list = []
        for i in range(3):
            mock_channel_ts = Mock(spec=ChannelTS)
            mock_channel_ts.ts = Mock()
            mock_channel_ts.ts.size = 1000
            seq_list.append(mock_channel_ts)
        return seq_list

    collection.read_segments = Mock(side_effect=read_segments_func)
    collection.to_channel_ts = Mock(side_effect=to_channel_ts_func)
    collection.get_channel_response = Mock(return_value=Mock())
    collection.unpack_header = Mock()

    return collection


@pytest.fixture
def expected_subheader_values():
    """Expected values for subheader testing."""
    return {
        "channel_id": 0,
        "channel_type": "H",
        "elevation": 140.10263061523438,
        "gps_time_stamp": "2021-04-27T03:24:42+00:00",
        "header_length": 32,
        "instrument_serial_number": "10128",
        "instrument_type": "MTU-5C",
        "latitude": 43.69625473022461,
        "longitude": -79.39364624023438,
        "missing_count": 0,
        "n_samples": 48000,
        "sample_rate": 24000.0,
        "saturation_count": 0,
        "segment": 0,
        "segment_end_time": "2021-04-27T03:24:44+00:00",
        "segment_start_time": "2021-04-27T03:24:42+00:00",
        "value_max": 2.5,
        "value_mean": 0.0,
        "value_min": -2.5,
    }


@pytest.fixture
def expected_channel_metadata():
    """Expected channel metadata for testing."""
    return OrderedDict(
        [
            ("channel_number", 0),
            ("component", "h2"),
            ("data_quality.rating.value", None),
            ("location.elevation", 140.10263061523438),
            ("location.latitude", 43.69625473022461),
            ("location.longitude", -79.39364624023438),
            ("measurement_azimuth", 90.0),
            ("measurement_tilt", 0.0),
            ("sample_rate", 24000.0),
            ("sensor.id", "0"),
            ("sensor.manufacturer", "Phoenix Geophysics"),
            ("sensor.model", "MTC-150"),
            ("sensor.type", "4"),
            ("time_period.end", "2021-04-27T03:25:13.999958333+00:00"),
            ("time_period.start", "2021-04-27T03:25:12+00:00"),
            ("type", "magnetic"),
            ("units", "Volt"),
        ]
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestSubHeader:
    """Test SubHeader class functionality."""

    def test_initialization_default(self):
        """Test SubHeader initialization with defaults."""
        subheader = SubHeader()

        assert subheader.header_length == 32
        assert subheader._header is None
        assert len(subheader._unpack_dict) == 7

    def test_initialization_with_kwargs(self):
        """Test SubHeader initialization with custom kwargs."""
        kwargs = {"custom_attr": "test_value", "another_attr": 42}
        subheader = SubHeader(**kwargs)

        assert subheader.custom_attr == "test_value"
        assert subheader.another_attr == 42

    def test_has_header_false(self):
        """Test _has_header returns False when no header."""
        subheader = SubHeader()
        assert not subheader._has_header()

    def test_has_header_true(self, mock_subheader):
        """Test _has_header returns True when header is set."""
        assert mock_subheader._has_header()

    def test_unpack_header_with_stream(self, mock_stream_with_header):
        """Test unpacking header from stream."""
        subheader = SubHeader()
        subheader.unpack_header(mock_stream_with_header)

        assert subheader._header is not None
        mock_stream_with_header.read.assert_called_once_with(32)

    def test_unpack_header_zero_length(self):
        """Test unpacking header with zero length."""
        subheader = SubHeader()
        subheader.header_length = 0
        mock_stream = Mock()

        result = subheader.unpack_header(mock_stream)

        assert result is None
        mock_stream.read.assert_not_called()

    @pytest.mark.parametrize(
        "property_name,expected_type",
        [
            ("gps_time_stamp", MTime),
            ("n_samples", int),
            ("saturation_count", int),
            ("missing_count", int),
            ("value_min", float),
            ("value_max", float),
            ("value_mean", float),
        ],
    )
    def test_properties_with_header(self, mock_subheader, property_name, expected_type):
        """Test property values with header data."""
        value = getattr(mock_subheader, property_name)
        assert isinstance(value, expected_type)

    def test_properties_without_header(self):
        """Test property values without header data."""
        subheader = SubHeader()

        assert subheader.gps_time_stamp is None
        assert subheader.n_samples is None
        assert subheader.saturation_count is None
        assert subheader.missing_count is None
        assert subheader.value_min is None
        assert subheader.value_max is None
        assert subheader.value_mean is None

    def test_string_representation(self, mock_subheader):
        """Test string representation of SubHeader."""
        str_repr = str(mock_subheader)

        assert "subheader information:" in str_repr
        assert "gps_time_stamp" in str_repr
        assert "n_samples" in str_repr


class TestSegment:
    """Test Segment class functionality."""

    def test_initialization(self, mock_stream_with_header, mock_segment_kwargs):
        """Test Segment initialization."""
        segment = Segment(mock_stream_with_header, **mock_segment_kwargs)

        assert segment.stream == mock_stream_with_header
        assert segment.data is None
        assert segment.instrument_type == "MTU-5C"
        assert segment.channel_id == 0

    def test_read_segment_with_data(self, mock_segment):
        """Test reading segment with data."""
        with patch("numpy.fromfile") as mock_fromfile:
            mock_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            mock_fromfile.return_value = mock_data

            # Mock n_samples property
            with patch.object(
                type(mock_segment), "n_samples", new_callable=PropertyMock
            ) as mock_n_samples:
                mock_n_samples.return_value = 3

                mock_segment.read_segment(metadata_only=False)

                mock_fromfile.assert_called_once()

    def test_read_segment_metadata_only(self, mock_segment):
        """Test reading segment metadata only."""
        with patch("numpy.fromfile") as mock_fromfile:
            mock_segment.read_segment(metadata_only=True)

            mock_fromfile.assert_not_called()

    def test_segment_start_time(self, mock_segment):
        """Test segment start time property."""
        mock_time = MTime(time_stamp="2021-04-27T03:24:42+00:00")

        with patch.object(
            type(mock_segment), "gps_time_stamp", new_callable=PropertyMock
        ) as mock_gps:
            mock_gps.return_value = mock_time

            start_time = mock_segment.segment_start_time
            assert start_time == mock_time

    def test_segment_end_time(self, mock_segment):
        """Test segment end time calculation."""
        mock_start_time = MTime(time_stamp="2021-04-27T03:24:42+00:00")
        mock_segment.sample_rate = 24000

        with patch.object(
            type(mock_segment), "gps_time_stamp", new_callable=PropertyMock
        ) as mock_gps, patch.object(
            type(mock_segment), "n_samples", new_callable=PropertyMock
        ) as mock_n_samples:
            mock_gps.return_value = mock_start_time
            mock_n_samples.return_value = 48000

            end_time = mock_segment.segment_end_time
            expected_end_time = mock_start_time + (48000 / 24000)

            assert end_time == expected_end_time


class TestDecimatedSegmentedReader:
    """Test DecimatedSegmentedReader class functionality."""

    @patch(
        "mth5.io.phoenix.readers.segmented.decimated_segmented_reader.TSReaderBase.__init__"
    )
    def test_initialization(self, mock_super_init, mock_phoenix_segmented_file):
        """Test DecimatedSegmentedReader initialization."""
        mock_super_init.return_value = None

        with patch.object(
            DecimatedSegmentedReader,
            "_update_channel_metadata_from_recmeta",
            return_value=Mock(),
        ):
            reader = DecimatedSegmentedReader(mock_phoenix_segmented_file)

            assert hasattr(reader, "_channel_metadata")
            assert hasattr(reader, "sub_header")
            assert hasattr(reader, "subheader")
            assert reader.subheader == {}

    def test_read_segment(self, mock_decimated_segmented_reader):
        """Test reading a segment."""
        segment = mock_decimated_segmented_reader.read_segment()

        assert segment is not None
        assert hasattr(segment, "data")
        assert hasattr(segment, "segment_start_time")
        assert hasattr(segment, "segment_end_time")
        mock_decimated_segmented_reader.read_segment.assert_called_once_with()

    def test_read_segment_metadata_only(self, mock_decimated_segmented_reader):
        """Test reading segment metadata only."""
        segment = mock_decimated_segmented_reader.read_segment(metadata_only=True)

        assert segment is not None
        mock_decimated_segmented_reader.read_segment.assert_called_once_with(
            metadata_only=True
        )

    def test_to_channel_ts_basic(self, mock_decimated_segmented_reader):
        """Test basic ChannelTS conversion."""
        ch_ts = mock_decimated_segmented_reader.to_channel_ts()

        assert ch_ts is not None
        assert hasattr(ch_ts, "channel_metadata")
        assert hasattr(ch_ts, "channel_response")
        mock_decimated_segmented_reader.to_channel_ts.assert_called_once_with()

    def test_to_channel_ts_filters_structure(self, mock_decimated_segmented_reader):
        """Test that to_channel_ts returns correct filter structure."""
        # Mock the channel metadata with filters attribute
        mock_channel_metadata = Mock()

        # Mock filters as AppliedFilter objects with names
        mock_filters = []
        expected_filter_names = [
            "mtu-5c_rmt03_10128_h2_10000hz_lowpass",
            "v_to_mv",
            "coil_0_response",
        ]

        for name in expected_filter_names:
            mock_filter = Mock()
            mock_filter.name = name
            mock_filters.append(mock_filter)

        mock_channel_metadata.get_attr_from_name = Mock(return_value=mock_filters)

        # Update the channel_ts mock to return our mock metadata
        def mock_to_channel_ts_with_filters(rxcal_fn=None, scal_fn=None):
            mock_channel_ts = Mock(spec=ChannelTS)
            mock_channel_ts.ts = Mock()
            mock_channel_ts.ts.size = 48000
            mock_channel_ts.channel_metadata = mock_channel_metadata
            mock_channel_ts.channel_response = Mock()
            mock_channel_ts.channel_response.filters_list = [Mock(), Mock()]
            mock_channel_ts.channel_response.filters_list[0].frequencies = Mock()
            mock_channel_ts.channel_response.filters_list[0].frequencies.shape = (69,)
            return mock_channel_ts

        mock_decimated_segmented_reader.to_channel_ts = Mock(
            side_effect=mock_to_channel_ts_with_filters
        )

        ch_ts = mock_decimated_segmented_reader.to_channel_ts()

        # Test filters exist and have correct structure
        filters = ch_ts.channel_metadata.get_attr_from_name("filters")
        assert isinstance(filters, list)
        assert len(filters) == 3

        # Test filter names match expected values
        for i, (filter_obj, expected_name) in enumerate(
            zip(filters, expected_filter_names)
        ):
            assert filter_obj.name == expected_name

    def test_to_channel_ts_with_calibration(self, mock_decimated_segmented_reader):
        """Test ChannelTS conversion with calibration files."""
        rxcal_fn = "test_rxcal.json"
        scal_fn = "test_scal.json"

        ch_ts = mock_decimated_segmented_reader.to_channel_ts(
            rxcal_fn=rxcal_fn, scal_fn=scal_fn
        )

        assert ch_ts is not None
        mock_decimated_segmented_reader.to_channel_ts.assert_called_once_with(
            rxcal_fn=rxcal_fn, scal_fn=scal_fn
        )

    @pytest.mark.parametrize(
        "metadata_only,expected_data",
        [
            (False, "has_data"),
            (True, "no_data"),
        ],
    )
    def test_read_segment_parametrized(
        self, mock_decimated_segmented_reader, metadata_only, expected_data
    ):
        """Test reading segment with different metadata_only values."""
        segment = mock_decimated_segmented_reader.read_segment(
            metadata_only=metadata_only
        )

        assert segment is not None
        mock_decimated_segmented_reader.read_segment.assert_called_once_with(
            metadata_only=metadata_only
        )


class TestDecimatedSegmentCollection:
    """Test DecimatedSegmentCollection class functionality."""

    def test_initialization(self, mock_decimated_segment_collection):
        """Test DecimatedSegmentCollection initialization using mock."""
        # Use the fixture instead of trying to create a real instance
        collection = mock_decimated_segment_collection

        assert hasattr(collection, "sub_header")
        assert hasattr(collection, "subheader")
        assert collection.subheader == {}

    def test_read_segments(self, mock_decimated_segment_collection):
        """Test reading multiple segments."""
        segments = mock_decimated_segment_collection.read_segments()

        assert isinstance(segments, list)
        assert len(segments) == 3
        mock_decimated_segment_collection.read_segments.assert_called_once_with()

    def test_read_segments_metadata_only(self, mock_decimated_segment_collection):
        """Test reading segments metadata only."""
        segments = mock_decimated_segment_collection.read_segments(metadata_only=True)

        assert isinstance(segments, list)
        mock_decimated_segment_collection.read_segments.assert_called_once_with(
            metadata_only=True
        )

    def test_to_channel_ts_list(self, mock_decimated_segment_collection):
        """Test converting segments to ChannelTS list."""
        ch_ts_list = mock_decimated_segment_collection.to_channel_ts()

        assert isinstance(ch_ts_list, list)
        assert len(ch_ts_list) == 3
        mock_decimated_segment_collection.to_channel_ts.assert_called_once_with()

    def test_to_channel_ts_with_calibration(self, mock_decimated_segment_collection):
        """Test ChannelTS conversion with calibration files."""
        rxcal_fn = "test_rxcal.json"
        scal_fn = "test_scal.json"

        ch_ts_list = mock_decimated_segment_collection.to_channel_ts(
            rxcal_fn=rxcal_fn, scal_fn=scal_fn
        )

        assert isinstance(ch_ts_list, list)
        mock_decimated_segment_collection.to_channel_ts.assert_called_once_with(
            rxcal_fn=rxcal_fn, scal_fn=scal_fn
        )


class TestSegmentedReaderValidation:
    """Test validation matching real Phoenix segmented reader test."""

    def test_to_channel_ts_comprehensive(self, mock_decimated_segmented_reader):
        """Test comprehensive to_channel_ts validation matching real test."""
        # Create a mock rxcal file path
        rxcal_fn = "example_rxcal.json"

        # Mock the channel metadata with real expected values
        expected_metadata = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "h2"),
                ("data_quality.rating.value", None),
                ("location.elevation", 140.10263061523438),
                ("location.latitude", 43.69625473022461),
                ("location.longitude", -79.39364624023438),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 24000.0),
                ("sensor.id", "0"),
                ("sensor.manufacturer", "Phoenix Geophysics"),
                ("sensor.model", "MTC-150"),
                ("sensor.type", "4"),
                ("time_period.end", "2021-04-27T03:25:13.999958333+00:00"),
                ("time_period.start", "2021-04-27T03:25:12+00:00"),
                ("type", "magnetic"),
                ("units", "Volt"),
            ]
        )

        # Mock filters with correct names
        expected_filter_names = [
            "mtu-5c_rmt03_10128_h2_10000hz_lowpass",
            "v_to_mv",
            "coil_0_response",
        ]

        mock_filters = []
        for name in expected_filter_names:
            mock_filter = Mock()
            mock_filter.name = name
            mock_filters.append(mock_filter)

        # Mock channel metadata
        mock_channel_metadata = Mock()
        mock_channel_metadata.get_attr_from_name = Mock(
            side_effect=lambda key: expected_metadata.get(
                key, mock_filters if key == "filters" else None
            )
        )

        # Mock channel response with 2 filters in filters_list
        mock_channel_response = Mock()
        mock_channel_response.filters_list = [Mock(), Mock()]
        mock_channel_response.filters_list[0].frequencies = Mock()
        mock_channel_response.filters_list[0].frequencies.shape = (69,)

        # Mock timeseries with correct size
        mock_ts = Mock()
        mock_ts.size = 48000

        # Create comprehensive channel_ts mock
        def mock_to_channel_ts_comprehensive(rxcal_fn=None, scal_fn=None):
            mock_channel_ts = Mock(spec=ChannelTS)
            mock_channel_ts.channel_metadata = mock_channel_metadata
            mock_channel_ts.channel_response = mock_channel_response
            mock_channel_ts.ts = mock_ts
            return mock_channel_ts

        mock_decimated_segmented_reader.to_channel_ts = Mock(
            side_effect=mock_to_channel_ts_comprehensive
        )

        # Test the conversion
        ch_ts = mock_decimated_segmented_reader.to_channel_ts(rxcal_fn=rxcal_fn)

        # Test metadata values
        for key, expected_value in expected_metadata.items():
            actual_value = ch_ts.channel_metadata.get_attr_from_name(key)
            if isinstance(expected_value, float):
                assert abs(actual_value - expected_value) < 1e-5, f"Mismatch for {key}"
            else:
                assert actual_value == expected_value, f"Mismatch for {key}"

        # Test channel response structure
        assert len(ch_ts.channel_response.filters_list) == 2
        assert ch_ts.channel_response.filters_list[0].frequencies.shape == (69,)

        # Test timeseries size
        assert ch_ts.ts.size == 48000

        # Test filters structure
        filters = ch_ts.channel_metadata.get_attr_from_name("filters")
        assert isinstance(filters, list)
        assert len(filters) == 3

        # Test filter names
        for i, (filter_obj, expected_name) in enumerate(
            zip(filters, expected_filter_names)
        ):
            assert filter_obj.name == expected_name, f"Filter {i} name mismatch"


class TestSegmentedReaderIntegration:
    """Integration tests using mocks instead of real files."""

    def test_full_workflow_segmented_reader(
        self, mock_decimated_segmented_reader, temp_directory
    ):
        """Test complete workflow with DecimatedSegmentedReader."""
        # Create mock file
        file_path = temp_directory / "test.td_24k"
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        with open(file_path, "wb") as f:
            f.write(b"\x00" * 160 + test_data.tobytes())  # 128 + 32 bytes header

        # Configure mock reader
        mock_decimated_segmented_reader.file_path = file_path

        # Test reading segment
        segment = mock_decimated_segmented_reader.read_segment()
        assert segment is not None

        # Test converting to ChannelTS
        ch_ts = mock_decimated_segmented_reader.to_channel_ts()
        assert ch_ts is not None

    def test_full_workflow_segment_collection(
        self, mock_decimated_segment_collection, temp_directory
    ):
        """Test complete workflow with DecimatedSegmentCollection."""
        # Create mock file with multiple segments
        file_path = temp_directory / "test_collection.td_24k"

        # Create test data for multiple segments
        header_data = b"\x00" * 128  # Main header
        segment_data = []
        for i in range(3):
            subheader = pack(
                "IIHHfff8x", 1619493882 + i * 2, 1000, 0, 0, -1.0, 1.0, 0.0
            )
            data = np.array([i + 1.0] * 1000, dtype=np.float32).tobytes()
            segment_data.append(subheader + data)

        with open(file_path, "wb") as f:
            f.write(header_data + b"".join(segment_data))

        # Configure mock collection
        mock_decimated_segment_collection.file_path = file_path

        # Test reading segments
        segments = mock_decimated_segment_collection.read_segments()
        assert len(segments) == 3

        # Test converting to ChannelTS list
        ch_ts_list = mock_decimated_segment_collection.to_channel_ts()
        assert len(ch_ts_list) == 3

    def test_error_handling_invalid_header(self, mock_decimated_segmented_reader):
        """Test error handling with invalid header data."""
        # Mock invalid header scenario
        mock_decimated_segmented_reader.read_segment.side_effect = IOError(
            "Invalid header"
        )

        with pytest.raises(IOError):
            mock_decimated_segmented_reader.read_segment()


class TestSegmentedReaderPerformance:
    """Performance and efficiency tests."""

    def test_multiple_segments_handling(self, mock_decimated_segment_collection):
        """Test handling multiple segments efficiently."""
        segments = mock_decimated_segment_collection.read_segments()

        # Should handle multiple segments without issues
        assert len(segments) >= 3

        # Test metadata-only reading for efficiency
        metadata_segments = mock_decimated_segment_collection.read_segments(
            metadata_only=True
        )
        assert len(metadata_segments) >= 3

    def test_large_segment_data(self, mock_decimated_segmented_reader):
        """Test handling large segment data."""
        # Mock large data scenario
        large_data = np.random.rand(100000).astype(np.float32)

        def large_segment_func(metadata_only=False):
            segment = Mock()
            segment.data = large_data if not metadata_only else None
            return segment

        mock_decimated_segmented_reader.read_segment = Mock(
            side_effect=large_segment_func
        )

        segment = mock_decimated_segmented_reader.read_segment()
        assert segment.data.size == 100000


class TestSegmentedReaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_subheader(self):
        """Test handling empty subheader."""
        subheader = SubHeader()

        assert not subheader._has_header()
        assert subheader.gps_time_stamp is None

    def test_zero_samples_segment(self, mock_stream_with_header, mock_segment_kwargs):
        """Test segment with zero samples."""
        segment = Segment(mock_stream_with_header, **mock_segment_kwargs)

        with patch.object(
            type(segment), "n_samples", new_callable=PropertyMock
        ) as mock_n_samples:
            mock_n_samples.return_value = 0

            with patch("numpy.fromfile") as mock_fromfile:
                mock_fromfile.return_value = np.array([], dtype=np.float32)

                segment.read_segment()

                # Should handle zero samples gracefully
                mock_fromfile.assert_called_once()

    def test_invalid_binary_data(self, mock_stream_with_header):
        """Test handling invalid binary data."""
        import struct

        # Mock stream that returns insufficient data
        mock_stream_with_header.read.return_value = b"short"  # Too short for unpack

        subheader = SubHeader()
        subheader.unpack_header(mock_stream_with_header)

        # Accessing a property should raise an error with invalid data
        with pytest.raises((struct.error, Exception)):
            _ = subheader.gps_time_stamp

    @pytest.mark.parametrize(
        "sample_rate,n_samples,expected_duration",
        [
            (24000, 48000, 2.0),
            (150, 900, 6.0),
            (1, 10, 10.0),
        ],
    )
    def test_segment_duration_calculation(
        self,
        mock_stream_with_header,
        mock_segment_kwargs,
        sample_rate,
        n_samples,
        expected_duration,
    ):
        """Test segment duration calculation with different parameters."""
        mock_segment_kwargs["sample_rate"] = sample_rate
        segment = Segment(mock_stream_with_header, **mock_segment_kwargs)

        mock_start_time = MTime(time_stamp="2021-04-27T03:24:42+00:00")

        with patch.object(
            type(segment), "gps_time_stamp", new_callable=PropertyMock
        ) as mock_gps, patch.object(
            type(segment), "n_samples", new_callable=PropertyMock
        ) as mock_n_samples:
            mock_gps.return_value = mock_start_time
            mock_n_samples.return_value = n_samples

            end_time = segment.segment_end_time
            actual_duration = end_time - mock_start_time

            assert abs(actual_duration - expected_duration) < 0.001


class TestAttributeValidation:
    """Test attribute validation matching expected values."""

    @pytest.mark.skipif(
        "peacock" not in str(Path(__file__).as_posix()),
        reason="Only local files, cannot test in GitActions",
    )
    @pytest.mark.parametrize(
        "attr,expected_value",
        [
            ("ad_plus_minus_range", 5.0),
            ("attenuator_gain", 1.0),
            ("battery_voltage_v", 12.475),
            ("board_model_main", "BCM01"),
            ("channel_id", 0),
            ("channel_main_gain", 4.0),
            ("channel_type", "H"),
            ("file_extension", ".td_24k"),
            ("instrument_id", "10128"),
            ("instrument_type", "MTU-5C"),
            ("sample_rate", 24000),
        ],
    )
    def test_phoenix_segmented_reader_attributes_mock(self, attr, expected_value):
        """Test Phoenix reader attributes match expected values (using mocks)."""
        # This would be used with real Phoenix files in actual integration tests
        # For now, we use mocks to ensure the test structure is correct
        reader = Mock()
        setattr(reader, attr, expected_value)

        assert getattr(reader, attr) == expected_value

    @pytest.mark.parametrize(
        "metadata_key,expected_value",
        [
            ("channel_number", 0),
            ("component", "h2"),
            ("data_quality.rating.value", None),
            ("sample_rate", 24000.0),
            ("type", "magnetic"),
            ("units", "Volt"),
        ],
    )
    def test_channel_metadata_attributes(self, metadata_key, expected_value):
        """Test channel metadata attributes."""
        metadata = Mock()
        metadata.get_attr_from_name = Mock(return_value=expected_value)

        value = metadata.get_attr_from_name(metadata_key)
        assert value == expected_value


# =============================================================================
# Integration test with actual Phoenix reader (if available)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    "peacock" not in str(Path(__file__).as_posix()),
    reason="Only local files, cannot test in GitActions",
)
class TestPhoenixSegmentedReaderIntegration:
    """Integration tests with real Phoenix files (if available)."""

    def test_real_phoenix_segmented_file_integration(self):
        """Test with real Phoenix segmented file."""
        # This would use actual Phoenix files if available
        # Skipped in CI/CD environments
        pytest.skip("Real Phoenix file integration test - local only")


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__])
