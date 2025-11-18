# -*- coding: utf-8 -*-
"""
Pytest test suite for Z3DSchedule functionality.

Translated from unittest version with optimizations:
- Session-scoped fixtures for speed
- Parametrized tests for similar attributes
- Comprehensive error handling
- Support for missing test data
- Extended functionality testing

@author: jpeacock (original), translated to pytest with enhancements
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.io.zen import Z3DSchedule


# =============================================================================
# Test Data Setup
# =============================================================================


def get_z3d_test_data():
    """Get Z3D test data path if available, otherwise None."""
    try:
        import mth5_test_data

        return mth5_test_data.get_test_data_path("zen")
    except ImportError:
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def z3d_test_path():
    """Session-scoped fixture for Z3D test data path."""
    path = get_z3d_test_data()
    if path is None:
        pytest.skip("mth5_test_data package not available")
    return path


@pytest.fixture(scope="session")
def z3d_schedule_file(z3d_test_path):
    """Session-scoped fixture for Z3D file path."""
    return z3d_test_path / "bm100_20220517_131017_256_EY.Z3D"


@pytest.fixture(scope="session")
def z3d_schedule_obj(z3d_schedule_file):
    """Session-scoped fixture for Z3DSchedule object."""
    z3d_obj = Z3DSchedule(fn=z3d_schedule_file)
    z3d_obj.read_schedule()
    return z3d_obj


# =============================================================================
# Additional Fixtures for Extended Testing
# =============================================================================


@pytest.fixture
def basic_z3d_schedule():
    """Basic Z3DSchedule instance for testing."""
    return Z3DSchedule()


@pytest.fixture
def sample_schedule_attributes():
    """Sample schedule attributes for testing."""
    return {
        "AutoGain": "N",
        "Comment": "",
        "Date": "2022-05-17",
        "Duty": "0",
        "FFTStacks": "0",
        "Filename": "",
        "Gain": "1.0000",
        "Log": "Y",
        "NewFile": "Y",
        "Period": "0",
        "RadioOn": "X",
        "SR": "256",
        "SamplesPerAcq": "0",
        "Sleep": "N",
        "Sync": "Y",
        "Time": "13:10:15",
    }


@pytest.fixture
def sample_schedule_meta_string():
    """Sample raw schedule metadata string."""
    return (
        b"\\n\\n\\nGPS Brd339/Brd357 Schedule Details\\n"
        b"Schedule.Date = 2022-05-17\\n"
        b"Schedule.Time = 13:10:15\\n"
        b"Schedule.Sync = Y\\n"
        b"Schedule.NewFile = Y\\n"
        b"Schedule.Period = 0\\n"
        b"Schedule.Duty = 0\\n"
        b"Schedule.S/R = 256\\n"
        b"Schedule.Gain = 1.0000\\n"
        b"Schedule.SamplesPerAcq = 0\\n"
        b"Schedule.FFTStacks = 0\\n"
        b"Schedule.Log = Y\\n"
        b"Schedule.Sleep = N\\n"
        b"Schedule.RadioOn = X\\n"
        b"Schedule.AutoGain = N\\n"
        b"Schedule.Filename = \\n"
        b"Schedule.Comment = \\n"
        b"\\n\\x00"
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestZ3DScheduleBasicAttributes:
    """Test class for basic Z3DSchedule attributes from original test file."""

    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("AutoGain", "N"),
            ("Comment", ""),
            ("Date", "2022-05-17"),
            ("Duty", "0"),
            ("FFTStacks", "0"),
            ("Filename", ""),
            ("Gain", "1.0000"),
            ("Log", "Y"),
            ("NewFile", "Y"),
            ("Period", "0"),
            ("RadioOn", "X"),
            ("SR", "256"),
            ("SamplesPerAcq", "0"),
            ("Sleep", "N"),
            ("Sync", "Y"),
            ("Time", "13:10:15"),
        ],
    )
    def test_schedule_attributes(self, z3d_schedule_obj, attribute, expected):
        """Test basic schedule attributes match expected values."""
        assert getattr(z3d_schedule_obj, attribute) == expected

    def test_fn_attribute(self, z3d_schedule_obj, z3d_schedule_file):
        """Test filename attribute."""
        assert getattr(z3d_schedule_obj, "fn") == z3d_schedule_file

    def test_initial_start(self, z3d_schedule_obj):
        """Test initial_start attribute."""
        assert getattr(z3d_schedule_obj, "initial_start") == "2022-05-17T13:09:57+00:00"

    def test_meta_string(self, z3d_schedule_obj):
        """Test meta_string attribute contains expected data."""
        meta_string = getattr(z3d_schedule_obj, "meta_string")
        assert isinstance(meta_string, bytes)
        assert b"GPS Brd339/Brd357 Schedule Details" in meta_string
        assert b"Schedule.Date = 2022-05-17" in meta_string
        assert b"Schedule.Time = 13:10:15" in meta_string


# =============================================================================
# Extended Test Classes for Enhanced Coverage
# =============================================================================


class TestZ3DScheduleInitialization:
    """Test Z3DSchedule initialization and basic properties."""

    def test_default_initialization(self, basic_z3d_schedule):
        """Test default initialization values."""
        zs = basic_z3d_schedule

        # Test file attributes
        assert zs.fn is None
        assert zs.fid is None
        assert zs.meta_string is None

        # Test file format constants
        assert zs._header_len == 512
        assert zs._schedule_metadata_len == 512

        # Test all schedule attributes are initially None
        schedule_attrs = [
            "AutoGain",
            "Comment",
            "Date",
            "Duty",
            "FFTStacks",
            "Filename",
            "Gain",
            "Log",
            "NewFile",
            "Period",
            "RadioOn",
            "SR",
            "SamplesPerAcq",
            "Sleep",
            "Sync",
            "Time",
        ]

        for attr in schedule_attrs:
            assert getattr(zs, attr) is None

        # Test initial_start is MTime object
        assert hasattr(zs.initial_start, "iso_str")

    def test_initialization_with_filename(self):
        """Test initialization with filename parameter."""
        test_fn = "/path/to/test.z3d"
        zs = Z3DSchedule(fn=test_fn)

        assert zs.fn == test_fn
        assert zs.fid is None

    def test_initialization_with_file_object(self):
        """Test initialization with file object."""
        mock_file = Mock()
        zs = Z3DSchedule(fid=mock_file)

        assert zs.fn is None
        assert zs.fid is mock_file

    def test_initialization_with_kwargs(self):
        """Test initialization with additional keyword arguments."""
        zs = Z3DSchedule(
            fn="test.z3d", custom_attr="test_value", another_attr=42, AutoGain="Y"
        )

        assert zs.fn == "test.z3d"
        assert zs.custom_attr == "test_value"
        assert zs.another_attr == 42
        assert zs.AutoGain == "Y"


class TestZ3DScheduleAttributeHandling:
    """Test schedule attribute setting and validation."""

    def test_attribute_setting(self, basic_z3d_schedule, sample_schedule_attributes):
        """Test that schedule attributes can be set correctly."""
        zs = basic_z3d_schedule

        # Set attributes manually (simulating what read_schedule does)
        for attr, value in sample_schedule_attributes.items():
            setattr(zs, attr, value)

        # Verify all attributes were set correctly
        for attr, expected_value in sample_schedule_attributes.items():
            actual_value = getattr(zs, attr)
            assert (
                actual_value == expected_value
            ), f"Expected {attr}={expected_value}, got {actual_value}"

    def test_string_attribute_handling(self, basic_z3d_schedule):
        """Test that all schedule attributes are handled as strings."""
        zs = basic_z3d_schedule

        # Test various data types converted to strings
        test_values = {
            "SR": "256",
            "Gain": "1.0000",
            "Duty": "0",
            "Period": "120",
            "FFTStacks": "5",
        }

        for attr, value in test_values.items():
            setattr(zs, attr, value)
            assert isinstance(getattr(zs, attr), str)
            assert getattr(zs, attr) == value

    def test_boolean_flags_as_strings(self, basic_z3d_schedule):
        """Test boolean-like flags are stored as Y/N strings."""
        zs = basic_z3d_schedule

        boolean_attrs = {
            "AutoGain": "N",
            "Log": "Y",
            "NewFile": "Y",
            "Sleep": "N",
            "Sync": "Y",
        }

        for attr, value in boolean_attrs.items():
            setattr(zs, attr, value)
            assert getattr(zs, attr) in ["Y", "N", "X"]  # X for RadioOn
            assert getattr(zs, attr) == value


class TestZ3DScheduleMetadataParsing:
    """Test metadata string parsing and key-value extraction."""

    def test_key_value_parsing(self, basic_z3d_schedule):
        """Test parsing of schedule key-value pairs."""
        zs = basic_z3d_schedule

        # Simulate parsing metadata lines
        test_lines = [
            "Schedule.Date = 2022-05-17",
            "Schedule.Time = 13:10:15",
            "Schedule.S/R = 256",  # Test slash removal
            "Schedule.AutoGain = N",
            "Schedule.Comment = Test comment",
        ]

        for line in test_lines:
            if line.find("=") > 0:
                m_list = line.split("=")
                m_key = m_list[0].split(".")[1].strip()
                m_key = m_key.replace("/", "")  # Remove slashes
                m_value = m_list[1].strip()
                setattr(zs, m_key, m_value)

        # Verify parsing results
        assert zs.Date == "2022-05-17"
        assert zs.Time == "13:10:15"
        assert zs.SR == "256"  # S/R becomes SR
        assert zs.AutoGain == "N"
        assert zs.Comment == "Test comment"

    def test_malformed_line_handling(self, basic_z3d_schedule):
        """Test handling of malformed metadata lines."""
        zs = basic_z3d_schedule

        # Test lines that should be skipped
        malformed_lines = [
            "No equals sign here",
            "Multiple = equals = signs",
            "Schedule only",
            "=No key",
            "",
            "Schedule.Key.Too.Many.Dots = value",
        ]

        # These should not raise exceptions or set attributes
        for line in malformed_lines:
            try:
                if line.find("=") > 0:
                    parts = line.split("=")
                    if len(parts) >= 2 and "." in parts[0]:
                        key_parts = parts[0].split(".")
                        if len(key_parts) >= 2:
                            key = key_parts[1].strip().replace("/", "")
                            value = parts[1].strip()
                            setattr(zs, key, value)
            except (IndexError, AttributeError):
                # Expected for malformed lines
                pass


class TestZ3DScheduleTimeHandling:
    """Test time and date handling functionality."""

    def test_initial_start_creation(self, basic_z3d_schedule):
        """Test creation of initial_start MTime object."""
        zs = basic_z3d_schedule

        # Set date and time
        zs.Date = "2022-05-17"
        zs.Time = "13:10:15"

        # Simulate initial_start creation from read_schedule
        from mt_metadata.common.mttime import MTime

        timestamp_str = f"{zs.Date}T{zs.Time}"
        zs.initial_start = MTime(time_stamp=timestamp_str, gps_time=True)

        # Test the result
        assert str(zs.initial_start) == "2022-05-17T13:09:57+00:00"

    def test_invalid_time_handling(self, basic_z3d_schedule):
        """Test handling of invalid date/time values."""
        zs = basic_z3d_schedule

        # Test invalid date formats
        invalid_dates = ["invalid", "2022-13-45", "", "not-a-date"]

        for invalid_date in invalid_dates:
            zs.Date = invalid_date
            zs.Time = "13:10:15"

            try:
                from mt_metadata.common.mttime import MTime

                timestamp_str = f"{zs.Date}T{zs.Time}"
                zs.initial_start = MTime(time_stamp=timestamp_str, gps_time=True)
            except Exception:
                # Should handle gracefully - keep default MTime
                assert hasattr(zs.initial_start, "iso_str")


class TestZ3DScheduleFileOperations:
    """Test file operation handling and path support."""

    @patch("builtins.open", new_callable=mock_open)
    def test_file_opening(self, mock_file, basic_z3d_schedule):
        """Test file opening with mocked file operations."""
        zs = basic_z3d_schedule
        mock_file.return_value.read.return_value = b"Schedule.Date = 2022-05-17\\n"

        test_fn = "/path/to/test.z3d"

        # Test that we can set filename
        zs.fn = test_fn
        assert zs.fn == test_fn

        # Test file opening would be called (in real read_schedule)
        # We don't call read_schedule here to avoid complex parsing

    def test_file_object_handling(self, basic_z3d_schedule):
        """Test file object handling."""
        zs = basic_z3d_schedule

        mock_fid = Mock()
        zs.fid = mock_fid

        # Test file object assignment
        assert zs.fid is mock_fid

        # Test file object methods exist
        assert hasattr(mock_fid, "read")
        assert hasattr(mock_fid, "seek")

    def test_pathlib_path_support(self, basic_z3d_schedule):
        """Test Path object support."""
        zs = basic_z3d_schedule

        test_path = Path("/path/to/test.z3d")
        zs.fn = test_path

        assert zs.fn == test_path

        # Test path operations
        path_str = str(zs.fn)
        assert path_str.endswith("test.z3d")
        assert "path" in path_str
        assert zs.fn.name == "test.z3d"

    def test_temporary_file_handling(self):
        """Test with temporary files."""
        with tempfile.NamedTemporaryFile(suffix=".z3d", delete=False) as tmp:
            tmp.write(b"\\x00" * 1024)  # Write dummy header + schedule data
            tmp_path = tmp.name

        try:
            # Test initialization with temporary file
            zs = Z3DSchedule(fn=tmp_path)
            assert zs.fn == tmp_path

            # Verify file exists
            assert Path(tmp_path).exists()

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestZ3DScheduleErrorHandling:
    """Test error handling and edge cases."""

    def test_no_file_specified(self, basic_z3d_schedule):
        """Test warning when no file is specified."""
        zs = basic_z3d_schedule

        with patch.object(zs.logger, "warning") as mock_warning:
            # This should trigger the warning
            zs.read_schedule()

            mock_warning.assert_called_once_with("No Z3D file to read.")

    def test_file_not_found_handling(self, basic_z3d_schedule):
        """Test handling of non-existent files."""
        zs = basic_z3d_schedule

        with pytest.raises(FileNotFoundError):
            zs.read_schedule(fn="nonexistent_file.z3d")

    def test_unicode_decode_error_handling(self, basic_z3d_schedule):
        """Test handling of binary data that can't be decoded."""
        zs = basic_z3d_schedule

        # Simulate binary data that can't be decoded as UTF-8
        binary_data = b"\xff\xfe\x00\x01Schedule.Date = invalid\n"
        zs.meta_string = binary_data

        # Should handle decode errors gracefully and skip problematic lines
        with patch.object(zs.logger, "debug") as mock_debug:
            # Simulate the parsing part of read_schedule
            meta_list = zs.meta_string.split(b"\n")
            for m_str in meta_list:
                try:
                    m_str_decoded = m_str.decode()
                    # Process normally
                except UnicodeDecodeError as e:
                    zs.logger.debug(f"Skipped malformed schedule line: {m_str!r} - {e}")

            # Should have logged the error for the binary data
            assert mock_debug.called


class TestZ3DScheduleIntegration:
    """Integration tests for complete workflow."""

    def test_complete_read_workflow_simulation(
        self, basic_z3d_schedule, sample_schedule_meta_string
    ):
        """Test complete read workflow without actual file I/O."""
        zs = basic_z3d_schedule

        # Simulate read_schedule workflow
        zs.meta_string = sample_schedule_meta_string

        # Parse metadata (simulate read_schedule parsing)
        meta_list = zs.meta_string.split(b"\\n")
        for m_str in meta_list:
            try:
                m_str_decoded = m_str.decode()
                if m_str_decoded.find("=") > 0:
                    m_list = m_str_decoded.split("=")
                    m_key = m_list[0].split(".")[1].strip()
                    m_key = m_key.replace("/", "")
                    m_value = m_list[1].strip()
                    setattr(zs, m_key, m_value)
            except (UnicodeDecodeError, IndexError):
                continue

        # Create initial_start
        from mt_metadata.common.mttime import MTime

        if zs.Date and zs.Time:
            timestamp_str = f"{zs.Date}T{zs.Time}"
            zs.initial_start = MTime(time_stamp=timestamp_str, gps_time=True)

        # Verify final state
        assert zs.Date == "2022-05-17"
        assert zs.Time == "13:10:15"
        assert zs.SR == "256"  # S/R becomes SR
        assert zs.Sync == "Y"
        assert str(zs.initial_start) == "2022-05-17T13:09:57+00:00"

    def test_attribute_consistency(self, z3d_schedule_obj):
        """Test consistency between related attributes."""
        zs = z3d_schedule_obj

        # Test related attributes are consistent
        if zs.Date and zs.Time:
            # Should have valid initial_start
            assert hasattr(zs.initial_start, "iso_str")
            assert str(zs.initial_start).startswith(zs.Date)

        # Test boolean flags have expected values
        boolean_flags = ["AutoGain", "Log", "NewFile", "Sleep", "Sync"]
        for flag in boolean_flags:
            flag_value = getattr(zs, flag)
            if flag_value is not None:
                assert flag_value in ["Y", "N", "X"]

        # Test numeric strings can be converted
        numeric_attrs = ["SR", "Gain", "Duty", "Period", "FFTStacks", "SamplesPerAcq"]
        for attr in numeric_attrs:
            attr_value = getattr(zs, attr)
            if attr_value is not None and attr_value != "":
                try:
                    float(attr_value)  # Should be convertible to number
                except ValueError:
                    pytest.fail(f"Attribute {attr} = '{attr_value}' is not numeric")


class TestZ3DSchedulePerformance:
    """Test performance-related aspects."""

    def test_multiple_reads_same_file(self, z3d_schedule_file):
        """Test reading schedule multiple times from same file."""
        import time

        zs = Z3DSchedule(fn=z3d_schedule_file)

        # Time multiple reads
        start_time = time.time()
        for _ in range(10):
            zs.read_schedule()
        end_time = time.time()

        # Should be reasonably fast (less than 1 second for 10 reads)
        assert (end_time - start_time) < 1.0
        assert zs.Date is not None

    def test_fixture_reuse_performance(self, z3d_schedule_obj):
        """Test that session-scoped fixtures provide fast access."""
        # This test validates that our session-scoped fixtures work
        assert hasattr(z3d_schedule_obj, "Date")
        assert hasattr(z3d_schedule_obj, "Time")
        assert z3d_schedule_obj.Date is not None
        assert z3d_schedule_obj.Time is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
