"""
Comprehensive pytest test suite for MTUTable class (read_tbl.py).

This test suite uses fixtures, parameterization, and is optimized for parallel execution.
Tests cover initialization, file reading, value decoding, metadata extraction,
calibration calculations, and error handling.

Author: J. Peacock
Date: 2026-01-02
"""

import struct
from pathlib import Path

import pytest
from mth5_test_data import get_test_data_path

from mth5.io.phoenix.readers.mtu.mtu_table import MTUTable


# =============================================================================
# Fixtures - Session and Module Scope for Parallel Execution Optimization
# =============================================================================


@pytest.fixture(scope="session")
def phoenix_mtu_data_path():
    """
    Get the path to Phoenix MTU test data directory (session scope).

    Returns
    -------
    pathlib.Path
        Path to the extracted Phoenix MTU test data directory.
    """
    return get_test_data_path("phoenix_mtu")


@pytest.fixture(scope="session")
def sample_tbl_file(phoenix_mtu_data_path):
    """
    Get path to a sample TBL file for testing (session scope).

    Returns
    -------
    pathlib.Path
        Path to the 1690C16C.TBL test file.
    """
    tbl_file = phoenix_mtu_data_path / "1690C16C.TBL"
    if not tbl_file.exists():
        pytest.skip(f"TBL file not found: {tbl_file}")
    return tbl_file


@pytest.fixture(scope="module")
def loaded_tbl_table(sample_tbl_file):
    """
    Create and load an MTUTable instance (module scope).

    Returns
    -------
    MTUTable
        Loaded MTUTable instance with decoded data.
    """
    tbl = MTUTable(sample_tbl_file)
    return tbl


@pytest.fixture
def empty_tbl_table():
    """
    Create an empty MTUTable instance (function scope).

    Returns
    -------
    MTUTable
        Empty MTUTable instance without file path.
    """
    return MTUTable()


@pytest.fixture
def temp_tbl_file(tmp_path):
    """
    Create a temporary TBL file for testing (function scope).

    Returns
    -------
    pathlib.Path
        Path to temporary TBL file with minimal valid content.
    """
    tbl_file = tmp_path / "test.TBL"

    # Create minimal valid TBL content (25-byte blocks)
    with open(tbl_file, "wb") as f:
        # SNUM tag (int: 1234)
        f.write(b"SNUM\x00\x00\x00\x00\x00\x00\x00\x00")  # 12 bytes
        f.write(struct.pack("<i", 1234) + b"\x00" * 9)  # 13 bytes

        # SITE tag (char: "TEST")
        f.write(b"SITE\x00\x00\x00\x00\x00\x00\x00\x00")
        f.write(b"TEST\x00\x00\x00\x00\x00\x00\x00\x00\x00")

        # ELEV tag (int: 1500)
        f.write(b"ELEV\x00\x00\x00\x00\x00\x00\x00\x00")
        f.write(struct.pack("<i", 1500) + b"\x00" * 9)

    return tbl_file


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMTUTableInitialization:
    """Test MTUTable initialization and basic setup."""

    def test_init_with_file_path(self, sample_tbl_file):
        """Test initialization with a valid file path."""
        tbl = MTUTable(sample_tbl_file)
        assert tbl.file_path == sample_tbl_file
        assert isinstance(tbl.tbl_dict, dict)
        assert len(tbl.tbl_dict) > 0  # Should be auto-loaded

    def test_init_without_file_path(self):
        """Test initialization without file path."""
        tbl = MTUTable()
        assert tbl.file_path is None
        assert tbl.tbl_dict == {}

    def test_init_with_string_path(self, sample_tbl_file):
        """Test initialization with string path (converts to Path)."""
        tbl = MTUTable(str(sample_tbl_file))
        assert isinstance(tbl.file_path, Path)
        assert tbl.file_path.exists()

    def test_tbl_tag_types_defined(self, empty_tbl_table):
        """Test that TBL_TAG_TYPES dictionary is properly defined."""
        assert hasattr(empty_tbl_table, "TBL_TAG_TYPES")
        assert isinstance(empty_tbl_table.TBL_TAG_TYPES, dict)
        assert len(empty_tbl_table.TBL_TAG_TYPES) > 0

        # Check a few key tags
        expected_tags = ["SNUM", "SITE", "EXLN", "HXSN", "SRL3", "FSCV", "HW"]
        for tag in expected_tags:
            assert tag in empty_tbl_table.TBL_TAG_TYPES
            assert len(empty_tbl_table.TBL_TAG_TYPES[tag]) == 2  # (type, description)


# =============================================================================
# File Reading Tests
# =============================================================================


class TestMTUTableFileReading:
    """Test TBL file reading functionality."""

    def test_read_tbl_loads_data(self, sample_tbl_file):
        """Test that read_tbl() successfully loads data."""
        tbl = MTUTable()
        tbl.file_path = sample_tbl_file
        tbl.read_tbl()

        assert len(tbl.tbl_dict) > 0
        assert "SNUM" in tbl.tbl_dict
        assert "SITE" in tbl.tbl_dict

    def test_read_tbl_auto_called_on_init(self, sample_tbl_file):
        """Test that read_tbl() is automatically called during init."""
        tbl = MTUTable(sample_tbl_file)
        assert len(tbl.tbl_dict) > 0

    def test_read_tbl_raises_error_without_path(self, empty_tbl_table):
        """Test that read_tbl() raises ValueError without file path."""
        with pytest.raises(ValueError, match="file_path is not set"):
            empty_tbl_table.read_tbl()

    def test_read_tbl_raises_error_for_nonexistent_file(self, tmp_path):
        """Test that read_tbl() raises FileNotFoundError for missing file."""
        tbl = MTUTable()
        tbl.file_path = tmp_path / "nonexistent.TBL"

        with pytest.raises(FileNotFoundError, match="TBL file not found"):
            tbl.read_tbl()

    def test_read_tbl_raises_error_for_non_tbl_file(self, tmp_path):
        """Test that read_tbl() raises ValueError for non-.TBL file."""
        non_tbl = tmp_path / "test.txt"
        non_tbl.touch()

        tbl = MTUTable()
        tbl.file_path = non_tbl

        with pytest.raises(ValueError, match="Not a TBL file"):
            tbl.read_tbl()

    def test_read_tbl_raises_error_for_empty_file(self, tmp_path):
        """Test that read_tbl() raises ValueError for empty file."""
        empty_file = tmp_path / "empty.TBL"
        empty_file.touch()

        tbl = MTUTable()
        tbl.file_path = empty_file

        with pytest.raises(ValueError, match="TBL file is empty"):
            tbl.read_tbl()

    def test_read_tbl_raises_error_for_too_small_file(self, tmp_path):
        """Test that read_tbl() raises ValueError for file smaller than 25 bytes."""
        small_file = tmp_path / "small.TBL"
        small_file.write_bytes(b"X" * 20)  # Less than 25 bytes

        tbl = MTUTable()
        tbl.file_path = small_file

        with pytest.raises(ValueError, match="TBL file is too small"):
            tbl.read_tbl()


# =============================================================================
# Value Decoding Tests - Parameterized
# =============================================================================


class TestMTUTableValueDecoding:
    """Test value decoding for different data types."""

    @pytest.mark.parametrize(
        "data_type,value_bytes,expected",
        [
            ("int", struct.pack("<i", 1690) + b"\x00" * 9, 1690),
            ("int", struct.pack("<i", -500) + b"\x00" * 9, -500),
            ("int", struct.pack("<i", 0) + b"\x00" * 9, 0),
        ],
    )
    def test_decode_int_values(self, empty_tbl_table, data_type, value_bytes, expected):
        """Test decoding integer values."""
        result = empty_tbl_table.decode_tbl_value(value_bytes, data_type)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "data_type,value_bytes,expected",
        [
            ("double", struct.pack("<d", 100.5) + b"\x00" * 5, 100.5),
            ("double", struct.pack("<d", -42.123) + b"\x00" * 5, -42.123),
            ("double", struct.pack("<d", 0.0) + b"\x00" * 5, 0.0),
        ],
    )
    def test_decode_double_values(
        self, empty_tbl_table, data_type, value_bytes, expected
    ):
        """Test decoding double-precision float values."""
        result = empty_tbl_table.decode_tbl_value(value_bytes, data_type)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "data_type,value_bytes,expected",
        [
            ("char", b"TEST\x00\x00\x00\x00\x00\x00\x00\x00\x00", "TEST"),
            ("char", b"SITE_NAME\x00\x00\x00\x00", "SITE_NAME"),
            ("char", b"ABC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "ABC"),
        ],
    )
    def test_decode_char_values(
        self, empty_tbl_table, data_type, value_bytes, expected
    ):
        """Test decoding character string values."""
        result = empty_tbl_table.decode_tbl_value(value_bytes, data_type)
        assert result == expected
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "data_type,value_bytes,expected",
        [
            ("byte", struct.pack("<B", 50) + b"\x00" * 12, 50),
            ("byte", struct.pack("<B", 0) + b"\x00" * 12, 0),
            ("byte", struct.pack("<B", 255) + b"\x00" * 12, 255),
        ],
    )
    def test_decode_byte_values(
        self, empty_tbl_table, data_type, value_bytes, expected
    ):
        """Test decoding byte values."""
        result = empty_tbl_table.decode_tbl_value(value_bytes, data_type)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "value_bytes,expected",
        [
            (
                b"\x00\x00\x00\x01\x01\x09\x00\x00\x00\x00\x00\x00\x00",
                "2009-01-01-T00:00:00",
            ),
            (
                b"\x1e\x1e\x17\x1f\x0c\x14\x00\x00\x00\x00\x00\x00\x00",
                "2020-12-31-T23:30:30",
            ),
        ],
    )
    def test_decode_time_values(self, empty_tbl_table, value_bytes, expected):
        """Test decoding time values."""
        result = empty_tbl_table.decode_tbl_value(value_bytes, "time")
        assert result == expected
        assert isinstance(result, str)

    def test_decode_unknown_type_returns_bytes(self, empty_tbl_table):
        """Test that unknown data types return raw bytes."""
        value_bytes = b"UNKNOWN_VALUE"
        result = empty_tbl_table.decode_tbl_value(value_bytes, "unknown_type")
        assert result == value_bytes
        assert isinstance(result, bytes)

    def test_decode_firmware_version_with_null_bytes(self, empty_tbl_table):
        """Test decoding firmware version (16s type) with null-byte truncation."""
        # Simulate HW tag value with null padding and garbage (13 bytes from TBL)
        value_bytes = b"MTU52\x00\x003100\xecQ\xd2\x00"
        result = empty_tbl_table.decode_tbl_value(value_bytes, "16s")

        # Should be decoded and truncated at first null byte
        assert isinstance(result, str)
        assert result == "MTU52"
        assert "\x00" not in result
        assert "\xec" not in result  # No garbage characters


# =============================================================================
# Dictionary Reading Tests
# =============================================================================


class TestMTUTableDictionaryReading:
    """Test _get_dictionary_from_tbl method."""

    def test_get_dictionary_with_decoding(self, loaded_tbl_table, sample_tbl_file):
        """Test reading dictionary with value decoding enabled."""
        result = loaded_tbl_table._get_dictionary_from_tbl(
            sample_tbl_file, decode_values=True
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "SNUM" in result
        assert isinstance(result["SNUM"], int)  # Should be decoded

    def test_get_dictionary_without_decoding(self, loaded_tbl_table, sample_tbl_file):
        """Test reading dictionary with value decoding disabled."""
        result = loaded_tbl_table._get_dictionary_from_tbl(
            sample_tbl_file, decode_values=False
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        # All values should be bytes
        for value in result.values():
            if value:  # Skip empty keys
                assert isinstance(value, bytes)

    def test_duplicate_keys_handled(self, temp_tbl_file, empty_tbl_table):
        """Test that duplicate keys are handled with numeric suffixes."""
        # Add duplicate SNUM entries
        with open(temp_tbl_file, "ab") as f:
            f.write(b"SNUM\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(struct.pack("<i", 5678) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()

        # Should have SNUM_1 and SNUM_2
        assert "SNUM_1" in empty_tbl_table.tbl_dict
        assert "SNUM_2" in empty_tbl_table.tbl_dict
        assert empty_tbl_table.tbl_dict["SNUM_1"] == 1234
        assert empty_tbl_table.tbl_dict["SNUM_2"] == 5678


# =============================================================================
# Metadata Extraction Tests
# =============================================================================


class TestMTUTableMetadataExtraction:
    """Test metadata extraction properties."""

    def test_has_metadata_returns_true(self, loaded_tbl_table):
        """Test _has_metadata returns True when data is loaded."""
        assert loaded_tbl_table._has_metadata() is True

    def test_has_metadata_returns_false(self, empty_tbl_table):
        """Test _has_metadata returns False when no data loaded."""
        assert empty_tbl_table._has_metadata() is False

    def test_survey_metadata_returns_survey_object(self, loaded_tbl_table):
        """Test that survey_metadata returns a Survey object."""
        survey = loaded_tbl_table.survey_metadata
        assert survey is not None
        assert hasattr(survey, "id")
        assert hasattr(survey, "acquired_by")

    def test_survey_metadata_with_no_data_warns(self, empty_tbl_table):
        """Test that accessing survey_metadata without data returns empty Survey."""
        survey = empty_tbl_table.survey_metadata
        # Note: loguru logger output is not captured by caplog
        # Just verify that an empty Survey is returned
        assert survey is not None
        assert hasattr(survey, "id")

    def test_station_metadata_returns_station_object(self, loaded_tbl_table):
        """Test that station_metadata returns a Station object."""
        station = loaded_tbl_table.station_metadata
        assert station is not None
        assert hasattr(station, "id")
        assert hasattr(station, "location")
        assert hasattr(station, "time_period")

    def test_station_metadata_has_location_data(self, loaded_tbl_table):
        """Test that station metadata includes location information."""
        station = loaded_tbl_table.station_metadata
        assert hasattr(station.location, "latitude")
        assert hasattr(station.location, "longitude")
        assert hasattr(station.location, "elevation")

    def test_run_metadata_returns_run_object(self, loaded_tbl_table):
        """Test that run_metadata returns a Run object."""
        run = loaded_tbl_table.run_metadata
        assert run is not None
        assert hasattr(run, "id")
        assert hasattr(run, "data_logger")

    def test_run_metadata_has_channels(self, loaded_tbl_table):
        """Test that run metadata includes channel information."""
        run = loaded_tbl_table.run_metadata
        # Run should have been populated with channels
        assert hasattr(run, "channels")

    def test_run_metadata_firmware_version(self, loaded_tbl_table):
        """Test that firmware version is properly decoded without null bytes."""
        run = loaded_tbl_table.run_metadata
        firmware_version = run.data_logger.firmware.version

        # Should be properly decoded string without null bytes
        assert isinstance(firmware_version, str)
        assert (
            "\x00" not in firmware_version
        ), "Firmware version should not contain null bytes"

        # Known value from test data
        assert (
            firmware_version == "MTU52"
        ), f"Expected 'MTU52' but got '{firmware_version}'"


# =============================================================================
# Channel Metadata Tests - Parameterized
# =============================================================================


class TestMTUTableChannelMetadata:
    """Test channel metadata extraction properties."""

    @pytest.mark.parametrize(
        "channel_property,component",
        [
            ("ex_metadata", "ex"),
            ("ey_metadata", "ey"),
        ],
    )
    def test_electric_channel_metadata(
        self, loaded_tbl_table, channel_property, component
    ):
        """Test electric channel metadata properties."""
        channel = getattr(loaded_tbl_table, channel_property)
        assert channel is not None
        assert hasattr(channel, "component")
        assert channel.component == component
        assert hasattr(channel, "dipole_length")
        assert hasattr(channel, "measurement_azimuth")

    @pytest.mark.parametrize(
        "channel_property,component",
        [
            ("hx_metadata", "hx"),
            ("hy_metadata", "hy"),
            ("hz_metadata", "hz"),
        ],
    )
    def test_magnetic_channel_metadata(
        self, loaded_tbl_table, channel_property, component
    ):
        """Test magnetic channel metadata properties."""
        channel = getattr(loaded_tbl_table, channel_property)
        assert channel is not None
        assert hasattr(channel, "component")
        assert channel.component == component
        assert hasattr(channel, "sensor")

    def test_ex_metadata_values(self, loaded_tbl_table):
        """Test specific Ex metadata values."""
        ex = loaded_tbl_table.ex_metadata
        if loaded_tbl_table._has_metadata() and "EXLN" in loaded_tbl_table.tbl_dict:
            assert ex.dipole_length == loaded_tbl_table.tbl_dict["EXLN"]

    def test_ey_azimuth_offset(self, loaded_tbl_table):
        """Test that Ey azimuth is Ex azimuth + 90 degrees."""
        ex = loaded_tbl_table.ex_metadata
        ey = loaded_tbl_table.ey_metadata
        if loaded_tbl_table._has_metadata() and "EAZM" in loaded_tbl_table.tbl_dict:
            assert ey.measurement_azimuth == ex.measurement_azimuth + 90.0

    def test_hy_azimuth_offset(self, loaded_tbl_table):
        """Test that Hy azimuth is Hx azimuth + 90 degrees."""
        hx = loaded_tbl_table.hx_metadata
        hy = loaded_tbl_table.hy_metadata
        if loaded_tbl_table._has_metadata() and "HAZM" in loaded_tbl_table.tbl_dict:
            assert hy.measurement_azimuth == hx.measurement_azimuth + 90.0


# =============================================================================
# Coordinate Conversion Tests - Parameterized
# =============================================================================


class TestMTUTableCoordinateConversion:
    """Test latitude and longitude conversion methods."""

    @pytest.mark.parametrize(
        "lat_str,expected",
        [
            ("4100.388,N", 41.00388),
            ("3530.500,N", 35.305),
            ("0000.000,N", 0.0),
            ("4100.388,S", -41.00388),
            ("3530.500,S", -35.305),
        ],
    )
    def test_read_latitude_conversion(self, loaded_tbl_table, lat_str, expected):
        """Test latitude conversion from degree-minute to decimal."""
        result = loaded_tbl_table._read_latitude(lat_str)
        assert result == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "lon_str,expected",
        [
            ("10400.536,E", 104.00536),
            ("12045.200,E", 120.452),
            ("00000.000,E", 0.0),
            ("10400.536,W", -104.00536),
            ("12045.200,W", -120.452),
        ],
    )
    def test_read_longitude_conversion(self, loaded_tbl_table, lon_str, expected):
        """Test longitude conversion from degree-minute to decimal."""
        result = loaded_tbl_table._read_longitude(lon_str)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_read_latitude_handles_invalid_format(self, loaded_tbl_table):
        """Test that invalid latitude format returns 0.0."""
        result = loaded_tbl_table._read_latitude("INVALID")
        assert result == 0.0
        # Note: loguru logger output is not captured by caplog

    def test_read_longitude_handles_invalid_format(self, loaded_tbl_table):
        """Test that invalid longitude format returns 0.0."""
        result = loaded_tbl_table._read_longitude("INVALID")
        assert result == 0.0
        # Note: loguru logger output is not captured by caplog


# =============================================================================
# Channel Keys Tests
# =============================================================================


class TestMTUTableChannelKeys:
    """Test channel_keys property method."""

    def test_channel_keys_returns_dict(self, loaded_tbl_table):
        """Test that channel_keys returns a dictionary."""
        keys = loaded_tbl_table.channel_keys
        assert isinstance(keys, dict)

    def test_channel_keys_with_all_channels(self, temp_tbl_file, empty_tbl_table):
        """Test channel_keys when all channel types are present."""
        # Add all channel types
        with open(temp_tbl_file, "ab") as f:
            for i, ch in enumerate(["CHEX", "CHEY", "CHHX", "CHHY", "CHHZ"], 1):
                f.write(ch.encode() + b"\x00" * (12 - len(ch)))
                f.write(struct.pack("<i", i) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        assert len(keys) == 5
        assert "ex" in keys
        assert "ey" in keys
        assert "hx" in keys
        assert "hy" in keys
        assert "hz" in keys
        assert keys["ex"] == 1
        assert keys["ey"] == 2
        assert keys["hx"] == 3
        assert keys["hy"] == 4
        assert keys["hz"] == 5

    def test_channel_keys_with_partial_channels(self, temp_tbl_file, empty_tbl_table):
        """Test channel_keys when only some channels are present."""
        # Add only electric channels
        with open(temp_tbl_file, "ab") as f:
            f.write(b"CHEX\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(struct.pack("<i", 1) + b"\x00" * 9)
            f.write(b"CHEY\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(struct.pack("<i", 2) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        assert len(keys) == 2
        assert "ex" in keys
        assert "ey" in keys
        assert "hx" not in keys
        assert "hy" not in keys
        assert "hz" not in keys

    def test_channel_keys_with_no_channels(self, temp_tbl_file, empty_tbl_table):
        """Test channel_keys when no channel keys are present."""
        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        assert isinstance(keys, dict)
        assert len(keys) == 0

    def test_channel_keys_without_metadata(self, empty_tbl_table):
        """Test channel_keys returns empty dict without loaded metadata."""
        keys = empty_tbl_table.channel_keys
        assert isinstance(keys, dict)
        assert len(keys) == 0

    def test_channel_keys_values_are_integers(self, temp_tbl_file, empty_tbl_table):
        """Test that channel_keys values are integers."""
        with open(temp_tbl_file, "ab") as f:
            f.write(b"CHHX\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(struct.pack("<i", 10) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        assert "hx" in keys
        assert isinstance(keys["hx"], int)
        assert keys["hx"] == 10

    def test_channel_keys_lowercase_component_names(
        self, temp_tbl_file, empty_tbl_table
    ):
        """Test that channel keys use lowercase component names."""
        with open(temp_tbl_file, "ab") as f:
            f.write(b"CHEX\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(struct.pack("<i", 5) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        # Should be lowercase 'ex', not 'EX'
        assert "ex" in keys
        assert "EX" not in keys

    @pytest.mark.parametrize(
        "ch_tag,expected_key,value",
        [
            ("CHEX", "ex", 1),
            ("CHEY", "ey", 2),
            ("CHHX", "hx", 3),
            ("CHHY", "hy", 4),
            ("CHHZ", "hz", 5),
        ],
    )
    def test_channel_keys_individual_channels(
        self, temp_tbl_file, empty_tbl_table, ch_tag, expected_key, value
    ):
        """Test each channel type individually (parameterized)."""
        with open(temp_tbl_file, "ab") as f:
            f.write(ch_tag.encode() + b"\x00" * (12 - len(ch_tag)))
            f.write(struct.pack("<i", value) + b"\x00" * 9)

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()
        keys = empty_tbl_table.channel_keys

        assert expected_key in keys
        assert keys[expected_key] == value

    def test_channel_keys_real_data(self, loaded_tbl_table):
        """Test channel_keys with real TBL file data."""
        keys = loaded_tbl_table.channel_keys

        # Real file should have channel keys if they're defined
        if keys:
            assert isinstance(keys, dict)
            # All keys should be lowercase component names
            for key in keys.keys():
                assert key in ["ex", "ey", "hx", "hy", "hz"]
                assert key.islower()
            # All values should be integers
            for value in keys.values():
                assert isinstance(value, int)


# =============================================================================
# Calibration Calculation Tests
# =============================================================================


class TestMTUTableCalibration:
    """Test calibration factor calculations."""

    def test_ex_calibration_returns_float(self, loaded_tbl_table):
        """Test that ex_calibration returns a float value."""
        if loaded_tbl_table._has_metadata():
            cal = loaded_tbl_table.ex_calibration
            assert isinstance(cal, float)
            assert cal > 0  # Should be positive

    def test_ey_calibration_returns_float(self, loaded_tbl_table):
        """Test that ey_calibration returns a float value."""
        if loaded_tbl_table._has_metadata():
            cal = loaded_tbl_table.ey_calibration
            assert isinstance(cal, float)
            assert cal > 0

    def test_magnetic_calibration_returns_float(self, loaded_tbl_table):
        """Test that magnetic_calibration returns a float value."""
        if loaded_tbl_table._has_metadata():
            cal = loaded_tbl_table.magnetic_calibration
            assert isinstance(cal, float)
            assert cal > 0

    def test_calibration_without_metadata_returns_none(self, empty_tbl_table):
        """Test that calibration properties return None without metadata."""
        assert empty_tbl_table.ex_calibration is None
        assert empty_tbl_table.ey_calibration is None
        assert empty_tbl_table.magnetic_calibration is None

    def test_ex_ey_calibration_same_if_equal_lengths(self, loaded_tbl_table):
        """Test that Ex and Ey calibrations are equal when dipole lengths are equal."""
        if loaded_tbl_table._has_metadata():
            if loaded_tbl_table.tbl_dict.get("EXLN") == loaded_tbl_table.tbl_dict.get(
                "EYLN"
            ):
                ex_cal = loaded_tbl_table.ex_calibration
                ey_cal = loaded_tbl_table.ey_calibration
                assert ex_cal == pytest.approx(ey_cal)

    def test_calibration_calculation_formula(self, loaded_tbl_table):
        """Test that calibration calculation follows expected formula."""
        if loaded_tbl_table._has_metadata():
            # Manual calculation for Ex
            fscv = float(loaded_tbl_table.tbl_dict.get("FSCV", 1.0))
            egn = float(loaded_tbl_table.tbl_dict.get("EGN", 1.0))
            exln = float(loaded_tbl_table.tbl_dict.get("EXLN", 1.0))

            expected = fscv / 2**23 * 1000 / egn / exln * 1000
            actual = loaded_tbl_table.ex_calibration

            assert actual == pytest.approx(expected, rel=1e-9)


# =============================================================================
# Real Data Validation Tests
# =============================================================================


class TestMTUTableRealDataValidation:
    """Test with real TBL file data to validate expected values."""

    def test_sample_file_has_expected_keys(self, loaded_tbl_table):
        """Test that sample TBL file contains expected keys."""
        expected_keys = [
            "SNUM",
            "SITE",
            "EXLN",
            "EYLN",
            "HXSN",
            "HYSN",
            "HZSN",
            "SRL3",
            "SRL4",
            "FSCV",
            "ELEV",
        ]

        for key in expected_keys:
            assert key in loaded_tbl_table.tbl_dict, f"Expected key '{key}' not found"

    def test_sample_file_serial_number(self, loaded_tbl_table):
        """Test that sample file has expected serial number."""
        if "SNUM" in loaded_tbl_table.tbl_dict:
            snum = loaded_tbl_table.tbl_dict["SNUM"]
            assert isinstance(snum, int)
            assert snum == 1690  # Known value from 1690C16C.TBL

    def test_sample_file_site_name(self, loaded_tbl_table):
        """Test that sample file has expected site name."""
        if "SITE" in loaded_tbl_table.tbl_dict:
            site = loaded_tbl_table.tbl_dict["SITE"]
            assert isinstance(site, str)
            assert site == "10441W10"  # Known value

    def test_sample_file_sample_rates(self, loaded_tbl_table):
        """Test that sample file has expected sample rates."""
        if all(key in loaded_tbl_table.tbl_dict for key in ["SRL3", "SRL4", "SRL5"]):
            assert loaded_tbl_table.tbl_dict["SRL3"] == 2400
            assert loaded_tbl_table.tbl_dict["SRL4"] == 150
            assert loaded_tbl_table.tbl_dict["SRL5"] == 15

    def test_sample_file_dipole_lengths(self, loaded_tbl_table):
        """Test that sample file has expected dipole lengths."""
        if all(key in loaded_tbl_table.tbl_dict for key in ["EXLN", "EYLN"]):
            assert loaded_tbl_table.tbl_dict["EXLN"] == pytest.approx(100.0)
            assert loaded_tbl_table.tbl_dict["EYLN"] == pytest.approx(100.0)

    def test_sample_file_firmware_version(self, loaded_tbl_table):
        """Test that sample file has properly decoded firmware version."""
        if "HW" in loaded_tbl_table.tbl_dict:
            hw_version = loaded_tbl_table.tbl_dict["HW"]
            assert isinstance(hw_version, str)
            assert "\x00" not in hw_version, "HW version should not contain null bytes"
            assert hw_version == "MTU52", f"Expected 'MTU52' but got '{hw_version}'"

        # Also verify through run metadata
        run = loaded_tbl_table.run_metadata
        firmware_version = run.data_logger.firmware.version
        assert (
            firmware_version == "MTU52"
        ), f"Expected 'MTU52' but got '{firmware_version}'"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestMTUTableEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_key_blocks_skipped(self, temp_tbl_file, empty_tbl_table):
        """Test that blocks with empty keys are skipped."""
        # Add a block with empty key
        with open(temp_tbl_file, "ab") as f:
            f.write(b"\x00" * 25)  # All zeros

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()

        # Should not have empty string as key
        assert "" not in empty_tbl_table.tbl_dict

    def test_partial_block_at_end_ignored(self, temp_tbl_file, empty_tbl_table):
        """Test that partial blocks at file end are ignored."""
        # Add partial block (less than 25 bytes)
        with open(temp_tbl_file, "ab") as f:
            f.write(b"PARTIAL_BLOCK")  # Less than 25 bytes

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()

        # Should still load without error
        assert len(empty_tbl_table.tbl_dict) > 0

    def test_unknown_tag_decoded_as_char_by_default(
        self, temp_tbl_file, empty_tbl_table
    ):
        """Test that unknown tags are decoded as char by default."""
        # Add block with unknown tag
        with open(temp_tbl_file, "ab") as f:
            f.write(b"UNKN\x00\x00\x00\x00\x00\x00\x00\x00")
            f.write(b"TESTVALUE\x00\x00\x00\x00")

        empty_tbl_table.file_path = temp_tbl_file
        empty_tbl_table.read_tbl()

        if "UNKN" in empty_tbl_table.tbl_dict:
            assert isinstance(empty_tbl_table.tbl_dict["UNKN"], str)


# =============================================================================
# Parallel Execution Safety Tests
# =============================================================================


class TestMTUTableParallelSafety:
    """Test that operations are safe for parallel execution."""

    def test_multiple_instances_independent(self, sample_tbl_file):
        """Test that multiple MTUTable instances are independent."""
        tbl1 = MTUTable(sample_tbl_file)
        tbl2 = MTUTable(sample_tbl_file)

        # Instances should be independent
        assert tbl1.tbl_dict is not tbl2.tbl_dict
        assert tbl1.TBL_TAG_TYPES is not tbl2.TBL_TAG_TYPES

        # But should have same data
        assert tbl1.tbl_dict == tbl2.tbl_dict

    def test_no_shared_state_between_tests(self, sample_tbl_file):
        """Test that there's no shared state between test invocations."""
        tbl = MTUTable(sample_tbl_file)
        original_dict = tbl.tbl_dict.copy()

        # Create new instance
        tbl2 = MTUTable(sample_tbl_file)

        # Should have identical data
        assert tbl2.tbl_dict == original_dict


# =============================================================================
# Performance and Resource Tests
# =============================================================================


@pytest.mark.benchmark
class TestMTUTablePerformance:
    """Performance tests (marked with benchmark for optional execution)."""

    def test_read_tbl_performance(self, sample_tbl_file, benchmark):
        """Benchmark TBL file reading performance."""

        def read_file():
            tbl = MTUTable()
            tbl.file_path = sample_tbl_file
            tbl.read_tbl()
            return tbl

        result = benchmark(read_file)
        assert len(result.tbl_dict) > 0

    def test_metadata_extraction_performance(self, loaded_tbl_table, benchmark):
        """Benchmark metadata extraction performance."""

        def extract_all_metadata():
            _ = loaded_tbl_table.survey_metadata
            _ = loaded_tbl_table.station_metadata
            _ = loaded_tbl_table.run_metadata
            _ = loaded_tbl_table.ex_metadata
            _ = loaded_tbl_table.ey_metadata
            _ = loaded_tbl_table.hx_metadata
            _ = loaded_tbl_table.hy_metadata
            _ = loaded_tbl_table.hz_metadata

        benchmark(extract_all_metadata)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMTUTableIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_from_file_to_metadata(self, sample_tbl_file):
        """Test complete workflow from file reading to metadata extraction."""
        # Initialize and read
        tbl = MTUTable(sample_tbl_file)
        assert tbl._has_metadata()

        # Extract survey
        survey = tbl.survey_metadata
        assert survey.id is not None

        # Extract station
        station = tbl.station_metadata
        assert station.id is not None
        assert station.location.latitude != 0

        # Extract run
        run = tbl.run_metadata
        assert run.id is not None

        # Verify firmware version is properly decoded
        firmware_version = run.data_logger.firmware.version
        assert isinstance(firmware_version, str)
        assert "\x00" not in firmware_version, "Firmware should not contain null bytes"
        assert (
            firmware_version == "MTU52"
        ), f"Expected firmware 'MTU52' but got '{firmware_version}'"

        # Extract channels
        ex = tbl.ex_metadata
        ey = tbl.ey_metadata
        hx = tbl.hx_metadata
        hy = tbl.hy_metadata
        hz = tbl.hz_metadata

        assert all([ex, ey, hx, hy, hz])

        # Calculate calibrations
        ex_cal = tbl.ex_calibration
        ey_cal = tbl.ey_calibration
        mag_cal = tbl.magnetic_calibration

        assert all([ex_cal, ey_cal, mag_cal])
        assert all([cal > 0 for cal in [ex_cal, ey_cal, mag_cal]])

    def test_workflow_with_empty_initialization(self, sample_tbl_file):
        """Test workflow starting with empty initialization."""
        tbl = MTUTable()
        assert not tbl._has_metadata()

        # Set path and read
        tbl.file_path = sample_tbl_file
        tbl.read_tbl()

        assert tbl._has_metadata()
        assert len(tbl.tbl_dict) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
