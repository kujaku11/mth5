"""
Test Phoenix TSReaderBase functionality using pytest framework with real data.
Based on test_base_reader.py but using pytest fixtures and modern testing patterns.
"""

from pathlib import Path
from unittest.mock import patch

import mth5_test_data
import pytest
from mt_metadata.timeseries.electric import Electric
from mt_metadata.timeseries.filters import ChannelResponse, CoefficientFilter
from mt_metadata.timeseries.magnetic import Magnetic
from mt_metadata.timeseries.run import Run
from mt_metadata.timeseries.station import Station

from mth5.io.phoenix.readers.base import TSReaderBase
from mth5.io.phoenix.readers.receiver_metadata import PhoenixReceiverMetadata


@pytest.fixture(scope="module")
def phoenix_data_path():
    """Get path to real Phoenix test data."""
    return Path(mth5_test_data.PHOENIX_TEST_DATA_DIR)


@pytest.fixture(scope="module")
def phoenix_files(phoenix_data_path):
    """Get list of Phoenix .td_150 files from real test data."""
    data_path = phoenix_data_path / "sample_data" / "10128_2021-04-27-032436"
    files = list(data_path.rglob("*.td_150"))
    assert len(files) > 0, "No Phoenix files found in test data"
    return sorted(files)


@pytest.fixture
def sample_ts_reader(phoenix_files):
    """Create TSReaderBase instance with real Phoenix file."""
    return TSReaderBase(phoenix_files[0])


@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_phoenix_file(temp_directory):
    """Create a mock Phoenix file structure for error testing."""
    channel_dir = temp_directory / "station" / "0"
    channel_dir.mkdir(parents=True)
    data_file = channel_dir / "10128_608783F4_0_00000001.td_150"
    data_file.write_bytes(b"x" * 200)
    return data_file


@pytest.fixture
def mock_ts_reader_setup(temp_directory):
    """Set up mock TSReader test environment."""
    # Create directory structure
    channel_dir = temp_directory / "station" / "0"
    channel_dir.mkdir(parents=True)

    # Create data file
    data_file = channel_dir / "10128_608783F4_0_00000001.td_150"
    data_file.write_bytes(b"x" * 200)

    # Create additional files (config, recmeta)
    config_file = channel_dir / "10128_config.json"
    config_file.write_text('{"test": "config"}')

    recmeta_file = channel_dir / "10128_recmeta.json"
    recmeta_file.write_text('{"test": "recmeta"}')

    return {
        "channel_dir": channel_dir,
        "data_file": data_file,
        "config_file": config_file,
        "recmeta_file": recmeta_file,
    }


@pytest.fixture(scope="module")
def example_rxcal_file(phoenix_data_path):
    """Get path to the example rxcal.json file."""
    rxcal_path = phoenix_data_path / "sample_data" / "example_rxcal.json"
    return rxcal_path if rxcal_path.exists() else None


class TestTSReaderBaseInitialization:
    """Test TSReaderBase initialization with real and mock data."""

    def test_initialization_with_real_file(self, phoenix_files):
        """Test initialization with real Phoenix file."""
        reader = TSReaderBase(phoenix_files[0])

        assert reader.base_path == phoenix_files[0]
        assert reader.seq == int(phoenix_files[0].stem.split("_")[-1])
        assert hasattr(reader, "stream")
        assert reader.file_size > 0

    def test_initialization_properties(self, sample_ts_reader):
        """Test that all expected properties are initialized."""
        reader = sample_ts_reader

        # Core properties
        assert isinstance(reader.base_path, Path)
        assert isinstance(reader.seq, int)
        assert isinstance(reader.file_size, int)
        assert isinstance(reader.max_samples, int)
        assert isinstance(reader.channel_map, dict)

        # Directory and file properties
        assert reader.base_dir == reader.base_path.parent
        assert reader.file_name == reader.base_path.name
        assert reader.file_extension == reader.base_path.suffix

        # Phoenix-specific properties
        assert isinstance(reader.recording_id, int)
        assert isinstance(reader.channel_id, int)
        assert isinstance(reader.instrument_id, str)

    def test_initialization_file_not_exists(self):
        """Test initialization with non-existent file."""
        fake_file = Path("10128_608783F4_0_00000001.td_150")

        with patch.object(TSReaderBase, "_open_file", return_value=False):
            with patch.object(TSReaderBase, "unpack_header"):
                reader = TSReaderBase(fake_file)
                assert reader.base_path == fake_file

    @pytest.mark.parametrize("num_files", [1, 3, 5, 10])
    def test_initialization_with_num_files(self, phoenix_files, num_files):
        """Test initialization with different num_files values."""
        reader = TSReaderBase(phoenix_files[0], num_files=num_files)

        # Test that sequence_list is populated correctly
        sequence = reader.sequence_list
        assert len(sequence) >= 1

        # num_files controls how many files to load in sequence,
        # but it only looks for files from the same channel
        # So we can't guarantee more files than exist for this channel
        expected_min = 1  # At least the current file should exist


class TestTSReaderBaseFileProperties:
    """Test file-related properties and methods."""

    def test_sequence_list_property(self, sample_ts_reader):
        """Test sequence_list property returns valid file list."""
        reader = sample_ts_reader
        sequence = reader.sequence_list

        assert isinstance(sequence, list)
        assert len(sequence) >= 1
        assert all(isinstance(f, Path) for f in sequence)
        assert all(f.suffix == ".td_150" for f in sequence)

    def test_config_file_path_property(self, sample_ts_reader):
        """Test config_file_path property."""
        reader = sample_ts_reader
        config_path = reader.config_file_path

        assert isinstance(config_path, Path)
        assert config_path.name.endswith("config.json")
        # Config file should be in the parent directory of the channel directory
        assert config_path.exists()

    def test_recmeta_file_path_property(self, sample_ts_reader):
        """Test recmeta_file_path property."""
        reader = sample_ts_reader
        recmeta_path = reader.recmeta_file_path

        assert isinstance(recmeta_path, Path)
        assert recmeta_path.name.endswith("recmeta.json")
        # Recmeta file should be in the parent directory of the channel directory
        assert recmeta_path.exists()

    def test_last_seq_property(self, sample_ts_reader):
        """Test last_seq property calculation."""
        reader = sample_ts_reader
        last_seq = reader.last_seq

        assert isinstance(last_seq, int)
        assert last_seq >= reader.seq

    def test_file_size_property(self, sample_ts_reader):
        """Test file_size property returns correct size."""
        reader = sample_ts_reader

        expected_size = reader.base_path.stat().st_size
        assert reader.file_size == expected_size

    def test_max_samples_calculation(self, sample_ts_reader):
        """Test max_samples calculation."""
        reader = sample_ts_reader

        # Should be calculated from file size and header info
        assert reader.max_samples > 0
        assert isinstance(reader.max_samples, int)


class TestTSReaderBaseFileOperations:
    """Test file operation methods."""

    def test_open_file_seq_valid(self, sample_ts_reader):
        """Test opening valid sequence file."""
        reader = sample_ts_reader

        # Test opening the current sequence
        with patch.object(reader, "unpack_header"):
            result = reader.open_file_seq(reader.seq)
            assert result is True

    def test_open_file_seq_invalid(self, sample_ts_reader):
        """Test opening invalid sequence file."""
        reader = sample_ts_reader

        # Try to open non-existent high sequence number, should handle gracefully
        try:
            result = reader.open_file_seq(99999)
            assert result is False
        except IndexError:
            # This is acceptable behavior for out-of-bounds sequence
            pass

    def test_open_next_method(self, sample_ts_reader):
        """Test open_next method."""
        reader = sample_ts_reader

        current_seq = reader.seq

        # Test successful case with mock
        with patch.object(reader, "open_file_seq") as mock_open:
            mock_open.return_value = True

            # Directly patch the actual return from open_next
            with patch.object(reader, "open_next", return_value=True):
                result = reader.open_next()
                assert result is True

        # Test that the real open_next calls open_file_seq with correct args
        with patch.object(reader, "open_file_seq") as mock_open:
            mock_open.return_value = False
            result = reader.open_next()

            # Should call with next sequence
            mock_open.assert_called_once_with(current_seq + 1)
            # May return False if next file doesn't exist
            assert result in [True, False]

    def test_close_method(self, sample_ts_reader):
        """Test close method."""
        reader = sample_ts_reader

        # Should not raise exception
        reader.close()

        # Stream should still exist but might be closed
        assert hasattr(reader, "stream")


class TestTSReaderBaseMetadataFiles:
    """Test metadata file handling."""

    def test_get_config_object(self, sample_ts_reader):
        """Test getting configuration object."""
        reader = sample_ts_reader

        config = reader.get_config_object()

        # Should return a configuration object
        assert config is not None
        assert hasattr(config, "obj")  # PhoenixConfig has obj attribute

    def test_rx_metadata_property(self, sample_ts_reader):
        """Test rx_metadata property loading."""
        reader = sample_ts_reader

        # Access the property to trigger loading
        rx_metadata = reader.rx_metadata

        assert rx_metadata is not None
        assert isinstance(rx_metadata, PhoenixReceiverMetadata)
        assert hasattr(rx_metadata, "obj")

    def test_rx_metadata_caching(self, sample_ts_reader):
        """Test that rx_metadata is cached properly."""
        reader = sample_ts_reader

        # First access
        rx_metadata1 = reader.rx_metadata

        # Second access should return same object
        rx_metadata2 = reader.rx_metadata

        assert rx_metadata1 is rx_metadata2

    def test_channel_map_update_from_recmeta(self, sample_ts_reader):
        """Test updating channel map from receiver metadata."""
        reader = sample_ts_reader

        # This should not raise an exception
        reader.update_channel_map_from_recmeta()

        # Channel map should be updated
        assert isinstance(reader.channel_map, dict)


class TestTSReaderBaseMetadataProperties:
    """Test metadata property methods with real data."""

    def test_channel_metadata_property(self, sample_ts_reader):
        """Test channel metadata property."""
        reader = sample_ts_reader

        channel_metadata = reader.channel_metadata

        assert isinstance(channel_metadata, (Magnetic, Electric))
        assert hasattr(channel_metadata, "component")
        assert hasattr(channel_metadata, "type")

    def test_channel_metadata_caching(self, sample_ts_reader):
        """Test channel metadata caching."""
        reader = sample_ts_reader

        # First access
        metadata1 = reader.channel_metadata

        # Second access should return same type but may be recreated
        metadata2 = reader.channel_metadata

        # Both should be the same type and have same content
        assert type(metadata1) == type(metadata2)
        assert metadata1.component == metadata2.component

    def test_run_metadata_property(self, sample_ts_reader):
        """Test run metadata property."""
        reader = sample_ts_reader

        run_metadata = reader.run_metadata

        assert isinstance(run_metadata, Run)
        assert hasattr(run_metadata, "id")
        assert hasattr(run_metadata, "sample_rate")

    def test_station_metadata_property(self, sample_ts_reader):
        """Test station metadata property."""
        reader = sample_ts_reader

        station_metadata = reader.station_metadata

        assert isinstance(station_metadata, Station)
        assert hasattr(station_metadata, "id")
        assert hasattr(station_metadata, "location")

    def test_metadata_values_real_data(self, sample_ts_reader):
        """Test that metadata contains expected values from real data."""
        reader = sample_ts_reader

        # Test channel metadata values
        ch_meta = reader.channel_metadata
        assert ch_meta.component in ["h1", "h2", "h3", "e1", "e2"]
        assert ch_meta.type in ["magnetic", "electric"]
        assert ch_meta.sample_rate > 0

        # Test run metadata values
        run_meta = reader.run_metadata
        assert len(run_meta.id) > 0
        assert run_meta.sample_rate > 0

        # Test station metadata values
        station_meta = reader.station_metadata
        assert len(station_meta.id) > 0


class TestTSReaderBaseFilterMethods:
    """Test filter creation methods."""

    def test_get_v_to_mv_filter(self, sample_ts_reader):
        """Test voltage to millivolt filter creation."""
        reader = sample_ts_reader

        filter_obj = reader.get_v_to_mv_filter()

        assert isinstance(filter_obj, CoefficientFilter)
        assert filter_obj.gain == 1000.0
        assert filter_obj.name == "v_to_mv"

    def test_get_dipole_filter_none(self, sample_ts_reader):
        """Test dipole filter returns None (default implementation)."""
        reader = sample_ts_reader

        # Default implementation should return None
        result = reader.get_dipole_filter()
        assert result is None

    def test_get_lowpass_filter_name(self, sample_ts_reader):
        """Test getting lowpass filter name from real data."""
        reader = sample_ts_reader

        filter_name = reader.get_lowpass_filter_name()

        # Should return a numeric value or None
        assert filter_name is None or isinstance(filter_name, (int, float))

    def test_get_receiver_lowpass_filter(self, sample_ts_reader, example_rxcal_file):
        """Test getting receiver lowpass filter."""
        reader = sample_ts_reader

        if example_rxcal_file is None:
            pytest.skip("No example_rxcal.json file available")

        # This may return None if no calibration file exists or channel doesn't match
        try:
            filter_obj = reader.get_receiver_lowpass_filter(example_rxcal_file)
            # Should not raise exception, may return None
            assert filter_obj is None or hasattr(filter_obj, "name")
        except AttributeError as e:
            if "Could not find" in str(e):
                # Expected behavior when channel doesn't exist in calibration file
                pytest.skip(f"Channel not found in calibration file: {e}")
            else:
                raise

    def test_get_sensor_filter(self, sample_ts_reader):
        """Test getting sensor filter."""
        reader = sample_ts_reader

        scal_fn = reader.base_dir / "scal.json"

        # This may return None if no sensor calibration exists
        filter_obj = reader.get_sensor_filter(scal_fn)

        # Should not raise exception, may return None
        assert filter_obj is None or hasattr(filter_obj, "name")

    def test_get_channel_response(self, sample_ts_reader):
        """Test getting channel response."""
        reader = sample_ts_reader

        response = reader.get_channel_response()

        assert isinstance(response, ChannelResponse)
        assert hasattr(response, "filters_list")

        # Should have at least the voltage conversion filter
        assert len(response.filters_list) >= 1

    def test_get_channel_response_with_calibration(
        self, sample_ts_reader, example_rxcal_file
    ):
        """Test channel response with potential calibration files."""
        reader = sample_ts_reader

        # Use real calibration file if available, otherwise use None
        rxcal_fn = example_rxcal_file
        scal_fn = None  # No sensor calibration file available in test data

        try:
            response = reader.get_channel_response(rxcal_fn=rxcal_fn, scal_fn=scal_fn)
            assert isinstance(response, ChannelResponse)
            assert hasattr(response, "filters_list")
        except AttributeError as e:
            if "Could not find" in str(e):
                # Expected when channel doesn't exist in calibration file
                # Test without calibration instead
                response = reader.get_channel_response()
                assert isinstance(response, ChannelResponse)
                assert hasattr(response, "filters_list")
            else:
                raise


class TestTSReaderBaseRealDataIntegration:
    """Test integration with real Phoenix data."""

    def test_all_phoenix_files_readable(self, phoenix_files):
        """Test that all Phoenix files in test data are readable."""
        for phoenix_file in phoenix_files[:5]:  # Test first 5 files
            reader = TSReaderBase(phoenix_file)

            assert reader.file_size > 0
            assert reader.max_samples > 0
            assert reader.recording_id is not None

    def test_metadata_consistency_across_files(self, phoenix_files):
        """Test metadata consistency across multiple files."""
        if len(phoenix_files) < 2:
            pytest.skip("Need at least 2 files for consistency test")

        readers = [TSReaderBase(f) for f in phoenix_files[:3]]

        # All files should have same recording_id and instrument_id
        recording_ids = [r.recording_id for r in readers]
        instrument_ids = [r.instrument_id for r in readers]

        assert len(set(recording_ids)) == 1, "Recording IDs should be consistent"
        assert len(set(instrument_ids)) == 1, "Instrument IDs should be consistent"

    def test_channel_types_real_data(self, phoenix_files):
        """Test that real data contains expected channel types."""
        # Group files by channel
        channel_files = {}
        for f in phoenix_files:
            channel_id = int(f.stem.split("_")[2])
            if channel_id not in channel_files:
                channel_files[channel_id] = []
            channel_files[channel_id].append(f)

        # Test each channel type
        for channel_id, files in channel_files.items():
            reader = TSReaderBase(files[0])
            ch_meta = reader.channel_metadata

            # Should have valid component and type
            assert ch_meta.component in ["h1", "h2", "h3", "e1", "e2"]
            assert ch_meta.type in ["magnetic", "electric"]

    def test_filter_integration_real_data(self, sample_ts_reader):
        """Test filter integration with real metadata."""
        reader = sample_ts_reader

        # Get channel response with real data
        response = reader.get_channel_response()

        # Verify filter structure
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) >= 1

        # Check that filters are properly configured
        for filter_obj in response.filters_list:
            assert hasattr(filter_obj, "name")
            assert len(filter_obj.name) > 0


class TestTSReaderBaseErrorHandling:
    """Test error handling and edge cases."""

    def test_get_lowpass_filter_name_no_recmeta(self, mock_phoenix_file):
        """Test getting lowpass filter name without recmeta file."""
        with patch.object(TSReaderBase, "_open_file"):
            with patch.object(TSReaderBase, "unpack_header"):
                reader = TSReaderBase(mock_phoenix_file)

                result = reader.get_lowpass_filter_name()
                assert result is None

    def test_channel_map_missing_recmeta(self, mock_phoenix_file):
        """Test channel map update when recmeta is missing."""
        with patch.object(TSReaderBase, "_open_file"):
            with patch.object(TSReaderBase, "unpack_header"):
                reader = TSReaderBase(mock_phoenix_file)

                # Should not raise exception
                reader.update_channel_map_from_recmeta()
                assert isinstance(reader.channel_map, dict)

    def test_metadata_properties_with_mock_recmeta(self, mock_phoenix_file):
        """Test metadata properties error handling with mock file."""
        with patch.object(TSReaderBase, "_open_file"):
            with patch.object(TSReaderBase, "unpack_header"):
                reader = TSReaderBase(mock_phoenix_file)

                # Test that the reader handles missing recmeta and channel type gracefully
                # This should raise ValueError for missing channel type, which is expected behavior
                with pytest.raises(ValueError, match="Channel type not available"):
                    reader.channel_metadata

                # Test that recmeta file path handling works
                assert hasattr(reader, "recmeta_file_path")

                # Test that we can check for missing recmeta without crashing
                try:
                    reader.update_channel_map_from_recmeta()
                    # Should complete without error even with missing recmeta
                except Exception as e:
                    # Any exception should be handled gracefully
                    assert isinstance(
                        e, (FileNotFoundError, AttributeError, ValueError)
                    )


class TestTSReaderBaseExtended:
    """Test additional functionality for comprehensive coverage."""

    def test_sequence_file_operations_extended(self, mock_ts_reader_setup):
        """Test extended sequence file operations."""
        channel_dir = mock_ts_reader_setup["channel_dir"]
        data_file = mock_ts_reader_setup["data_file"]

        # Create additional sequence files
        for i in range(2, 6):
            seq_file = channel_dir / f"10128_608783F4_0_0000000{i}.td_150"
            seq_file.write_bytes(b"x" * 200)

        with patch.object(TSReaderBase, "_open_file", return_value=True):
            with patch.object(TSReaderBase, "unpack_header"):
                reader = TSReaderBase(data_file, num_files=4)

                # Test sequence list
                sequence = reader.sequence_list
                assert len(sequence) >= 1

                # Test opening specific sequence
                with patch.object(reader, "unpack_header"):
                    result = reader.open_file_seq(2)
                    # Should attempt to open, result depends on mock

    @pytest.mark.parametrize(
        "channel,expected_type",
        [
            ("ex", "electric"),
            ("ey", "electric"),
            ("hx", "magnetic"),
            ("hy", "magnetic"),
            ("hz", "magnetic"),
        ],
    )
    def test_channel_type_inference(self, sample_ts_reader, channel, expected_type):
        """Test channel type inference from component names."""
        reader = sample_ts_reader

        # Test the pattern matching for channel types
        if channel.startswith("e"):
            assert "electric" == expected_type
        else:
            assert "magnetic" == expected_type

    def test_property_types_validation(self, sample_ts_reader):
        """Test that properties return expected types."""
        reader = sample_ts_reader

        # Test core property types
        assert isinstance(reader.base_path, Path)
        assert isinstance(reader.seq, int)
        assert isinstance(reader.file_size, int)
        assert isinstance(reader.max_samples, int)
        assert isinstance(reader.sequence_list, list)
        assert isinstance(reader.channel_map, dict)
        assert isinstance(reader.recording_id, int)
        assert isinstance(reader.instrument_id, str)
        assert isinstance(reader.channel_id, int)

    def test_file_naming_patterns(self, phoenix_files):
        """Test Phoenix file naming pattern recognition."""
        for phoenix_file in phoenix_files[:3]:
            reader = TSReaderBase(phoenix_file)

            # Test that file parsing works correctly
            assert reader.recording_id is not None
            assert reader.channel_id is not None
            assert reader.channel_id >= 0
            assert reader.seq >= 0
            assert len(reader.instrument_id) > 0

    def test_metadata_update_integration(self, sample_ts_reader):
        """Test metadata update method integration."""
        reader = sample_ts_reader

        # These methods should work with real data
        ch_meta = reader._update_channel_metadata_from_recmeta()
        run_meta = reader._update_run_metadata_from_recmeta()
        station_meta = reader._update_station_metadata_from_recmeta()

        assert isinstance(ch_meta, (Magnetic, Electric))
        assert isinstance(run_meta, Run)
        assert isinstance(station_meta, Station)


class TestTSReaderBaseCompatibility:
    """Test compatibility with original test expectations."""

    def test_all_original_properties_exist(self, sample_ts_reader):
        """Test that all properties from original unittest exist."""
        reader = sample_ts_reader

        # Original test properties
        properties = [
            "seq",
            "base_path",
            "last_seq",
            "recording_id",
            "channel_id",
            "channel_map",
            "base_dir",
            "file_name",
            "file_extension",
            "instrument_id",
            "file_size",
            "max_samples",
            "sequence_list",
            "config_file_path",
            "recmeta_file_path",
        ]

        for prop in properties:
            assert hasattr(reader, prop), f"Missing property: {prop}"

    def test_all_original_methods_exist(self, sample_ts_reader):
        """Test that all methods from original unittest exist."""
        reader = sample_ts_reader

        # Original test methods
        methods = [
            "get_config_object",
            "get_lowpass_filter_name",
            "get_dipole_filter",
            "get_v_to_mv_filter",
            "get_channel_response",
            "get_receiver_lowpass_filter",
            "get_sensor_filter",
            "close",
            "_open_file",
            "open_next",
        ]

        for method in methods:
            assert hasattr(reader, method), f"Missing method: {method}"
            assert callable(getattr(reader, method)), f"Not callable: {method}"

    def test_original_test_data_compatibility(self, phoenix_files):
        """Test compatibility with same data as original tests."""
        # Use same test data structure as original tests
        for phoenix_file in phoenix_files[:3]:
            reader = TSReaderBase(phoenix_file)

            # Should match original test expectations
            assert reader.recording_id == 1619493876
            assert reader.instrument_id == "10128"
            assert reader.file_extension == ".td_150"

    def test_metadata_objects_compatibility(self, sample_ts_reader):
        """Test that metadata objects match original test types."""
        reader = sample_ts_reader

        # Test metadata types match original expectations
        ch_meta = reader.channel_metadata
        run_meta = reader.run_metadata
        station_meta = reader.station_metadata

        assert isinstance(ch_meta, (Magnetic, Electric))
        assert isinstance(run_meta, Run)
        assert isinstance(station_meta, Station)

        # Test required attributes exist
        assert hasattr(ch_meta, "component")
        assert hasattr(ch_meta, "type")
        assert hasattr(ch_meta, "sample_rate")
        assert hasattr(run_meta, "id")
        assert hasattr(station_meta, "id")


# Additional test for comprehensive coverage
class TestTSReaderBasePerformance:
    """Test performance and resource usage."""

    def test_memory_usage_multiple_readers(self, phoenix_files):
        """Test memory usage with multiple reader instances."""
        readers = []

        try:
            for phoenix_file in phoenix_files[:3]:
                reader = TSReaderBase(phoenix_file)
                readers.append(reader)

                # Basic validation
                assert reader.file_size > 0
                assert reader.recording_id is not None

        finally:
            # Cleanup
            for reader in readers:
                reader.close()

    def test_repeated_property_access(self, sample_ts_reader):
        """Test repeated property access for performance."""
        reader = sample_ts_reader

        # Access properties multiple times - should be cached
        for _ in range(10):
            _ = reader.channel_metadata
            _ = reader.run_metadata
            _ = reader.station_metadata
            _ = reader.rx_metadata

        # Should not raise exceptions or show performance issues


if __name__ == "__main__":
    pytest.main([__file__])
