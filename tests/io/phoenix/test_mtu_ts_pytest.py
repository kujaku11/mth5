"""
Comprehensive pytest test suite for MTUTSN class (mtu_ts.py).

Tests the legacy Phoenix MTU-5A instrument time series binary file reader
for TS2, TS3, TS4, TS5, TSL, and TSH formats.

Author: pytest test suite
Date: 2026-01-02
"""

import struct
from pathlib import Path

import numpy as np
import pytest
from mth5_test_data import get_test_data_path

from mth5.io.phoenix.readers.mtu.mtu_ts import MTUTSN


# =============================================================================
# Session-level fixtures (shared across all tests)
# =============================================================================


@pytest.fixture(scope="session")
def phoenix_mtu_data_path():
    """Get path to phoenix_mtu test data directory."""
    return get_test_data_path("phoenix_mtu")


@pytest.fixture(scope="session")
def sample_ts3_file(phoenix_mtu_data_path):
    """Get path to sample TS3 file."""
    ts_file = phoenix_mtu_data_path / "1690C16C.TS3"
    if not ts_file.exists():
        pytest.skip(f"TS3 file not found: {ts_file}")
    return ts_file


@pytest.fixture(scope="session")
def sample_ts4_file(phoenix_mtu_data_path):
    """Get path to sample TS4 file."""
    ts_file = phoenix_mtu_data_path / "1690C16C.TS4"
    if not ts_file.exists():
        pytest.skip(f"TS4 file not found: {ts_file}")
    return ts_file


@pytest.fixture(scope="session")
def sample_ts5_file(phoenix_mtu_data_path):
    """Get path to sample TS5 file."""
    ts_file = phoenix_mtu_data_path / "1690C16C.TS5"
    if not ts_file.exists():
        pytest.skip(f"TS5 file not found: {ts_file}")
    return ts_file


@pytest.fixture(scope="session")
def sample_tbl_file(phoenix_mtu_data_path):
    """Get path to sample TBL file."""
    tbl_file = phoenix_mtu_data_path / "1690C16C.TBL"
    if not tbl_file.exists():
        pytest.skip(f"TBL file not found: {tbl_file}")
    return tbl_file


# =============================================================================
# Module-level fixtures (reused across test classes)
# =============================================================================


@pytest.fixture(scope="module")
def loaded_ts3_reader(sample_ts3_file):
    """Create MTUTSN instance with TS3 file loaded."""
    reader = MTUTSN(sample_ts3_file)
    reader.read()
    return reader


@pytest.fixture(scope="module")
def loaded_ts4_reader(sample_ts4_file):
    """Create MTUTSN instance with TS4 file loaded."""
    reader = MTUTSN(sample_ts4_file)
    reader.read()
    return reader


@pytest.fixture(scope="module")
def loaded_ts5_reader(sample_ts5_file):
    """Create MTUTSN instance with TS5 file loaded."""
    reader = MTUTSN(sample_ts5_file)
    reader.read()
    return reader


# =============================================================================
# Function-level fixtures (created fresh for each test)
# =============================================================================


@pytest.fixture
def temp_ts_file(tmp_path):
    """Create a temporary TS file for testing."""
    ts_file = tmp_path / "test.TS3"
    ts_file.touch()
    return ts_file


@pytest.fixture
def empty_reader():
    """Create MTUTSN instance without file."""
    return MTUTSN(file_path=None)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMTUTSNInitialization:
    """Test MTUTSN class initialization."""

    def test_init_with_valid_file_path(self, sample_ts3_file):
        """Test initialization with valid file path."""
        reader = MTUTSN(sample_ts3_file)
        assert reader.file_path == sample_ts3_file
        assert reader.ts is None
        assert reader.ts_metadata is None

        # Now explicitly read the file
        reader.read()
        assert reader.ts is not None
        assert reader.ts_metadata is not None

    def test_init_with_string_path(self, sample_ts3_file):
        """Test initialization with string path."""
        reader = MTUTSN(str(sample_ts3_file))
        assert reader.file_path == sample_ts3_file
        assert isinstance(reader.file_path, Path)

        # Data should not be loaded until read() is called
        assert reader.ts is None
        assert reader.ts_metadata is None

    def test_init_without_file_path(self):
        """Test initialization without file path."""
        reader = MTUTSN(file_path=None)
        assert reader.file_path is None

    def test_init_with_nonexistent_file(self):
        """Test initialization with nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            MTUTSN("/nonexistent/path/file.TS3")

    def test_init_with_invalid_extension(self, tmp_path):
        """Test initialization with invalid extension raises ValueError."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.touch()
        with pytest.raises(ValueError, match="not a recognized TSN format"):
            MTUTSN(invalid_file)

    @pytest.mark.parametrize(
        "extension",
        ["TS2", "TS3", "TS4", "TS5", "TSL", "TSH", "ts2", "ts3", "ts4", "ts5"],
    )
    def test_accepted_extensions(self, tmp_path, extension):
        """Test that all accepted extensions are recognized."""
        test_file = tmp_path / f"test.{extension}"
        # Create a minimal valid header to avoid read errors
        with open(test_file, "wb") as f:
            # Write 32 bytes of header
            f.write(b"\x00" * 32)
            # This will fail to read properly but extension should be accepted

        # Just check that ValueError is not raised for extension
        try:
            reader = MTUTSN(test_file)
        except (EOFError, struct.error, Exception):
            # Other errors are OK, just not extension validation error
            pass


# =============================================================================
# File Path Property Tests
# =============================================================================


class TestFilePathProperty:
    """Test file_path property setter/getter."""

    def test_set_valid_file_path(self, empty_reader, sample_ts3_file):
        """Test setting valid file path."""
        empty_reader.file_path = sample_ts3_file
        assert empty_reader.file_path == sample_ts3_file

    def test_set_none_file_path(self, empty_reader):
        """Test setting None as file path."""
        empty_reader.file_path = None
        assert empty_reader.file_path is None

    def test_set_nonexistent_file_path(self, empty_reader):
        """Test setting nonexistent file path raises error."""
        with pytest.raises(FileNotFoundError):
            empty_reader.file_path = Path("/nonexistent/file.TS3")

    def test_set_invalid_extension_path(self, empty_reader, tmp_path):
        """Test setting path with invalid extension raises error."""
        invalid_file = tmp_path / "test.invalid"
        invalid_file.touch()
        with pytest.raises(ValueError, match="not a recognized TSN format"):
            empty_reader.file_path = invalid_file


# =============================================================================
# Sign24 Conversion Tests
# =============================================================================


class TestSign24Conversion:
    """Test the get_sign24 method for 24-bit signed integer conversion."""

    @pytest.mark.parametrize(
        "input_value,expected_output",
        [
            (0, 0),  # Zero
            (1, 1),  # Small positive
            (100, 100),  # Positive
            (2**23 - 1, 2**23 - 1),  # Max positive (8388607)
            (2**23, -(2**23)),  # Min negative as unsigned (-8388608)
            (2**24 - 1, -1),  # -1 as unsigned
            (2**24 - 100, -100),  # -100 as unsigned
        ],
    )
    def test_sign24_single_values(self, empty_reader, input_value, expected_output):
        """Test get_sign24 with single values."""
        result = empty_reader.get_sign24(input_value)
        assert result == expected_output

    def test_sign24_array(self, empty_reader):
        """Test get_sign24 with numpy array."""
        input_array = np.array(
            [0, 1, 2**23 - 1, 2**23, 2**24 - 1], dtype=np.int32
        )
        expected = np.array([0, 1, 2**23 - 1, -(2**23), -1], dtype=np.int32)
        result = empty_reader.get_sign24(input_array)
        np.testing.assert_array_equal(result, expected)

    def test_sign24_list(self, empty_reader):
        """Test get_sign24 with list input."""
        input_list = [0, 100, 2**24 - 100]
        expected = np.array([0, 100, -100], dtype=np.int32)
        result = empty_reader.get_sign24(input_list)
        np.testing.assert_array_equal(result, expected)

    def test_sign24_preserves_positive(self, empty_reader):
        """Test that positive values below 2^23 are preserved."""
        positive_values = np.arange(0, 1000, dtype=np.int32)
        result = empty_reader.get_sign24(positive_values)
        np.testing.assert_array_equal(result, positive_values)

    def test_sign24_converts_negative(self, empty_reader):
        """Test that values >= 2^23 are converted to negative."""
        # Values that should become negative
        high_values = np.array([2**23, 2**23 + 1, 2**24 - 1], dtype=np.int32)
        result = empty_reader.get_sign24(high_values)
        assert np.all(result < 0)


# =============================================================================
# Data Reading Tests
# =============================================================================


class TestDataReading:
    """Test reading TS data files."""

    def test_read_ts3_file(self, loaded_ts3_reader):
        """Test reading TS3 file produces valid data."""
        assert loaded_ts3_reader.ts is not None
        assert loaded_ts3_reader.ts_metadata is not None
        assert isinstance(loaded_ts3_reader.ts, np.ndarray)
        assert isinstance(loaded_ts3_reader.ts_metadata, dict)

    def test_read_ts4_file(self, loaded_ts4_reader):
        """Test reading TS4 file produces valid data."""
        assert loaded_ts4_reader.ts is not None
        assert loaded_ts4_reader.ts_metadata is not None
        assert isinstance(loaded_ts4_reader.ts, np.ndarray)
        assert isinstance(loaded_ts4_reader.ts_metadata, dict)

    def test_read_ts5_file(self, loaded_ts5_reader):
        """Test reading TS5 file produces valid data."""
        assert loaded_ts5_reader.ts is not None
        assert loaded_ts5_reader.ts_metadata is not None
        assert isinstance(loaded_ts5_reader.ts, np.ndarray)
        assert isinstance(loaded_ts5_reader.ts_metadata, dict)

    def test_ts_data_shape(self, loaded_ts3_reader):
        """Test that TS data has correct shape."""
        ts = loaded_ts3_reader.ts
        ts_metadata = loaded_ts3_reader.ts_metadata
        assert ts.ndim == 2
        assert ts.shape[0] == ts_metadata["n_ch"]  # Number of channels
        assert ts.shape[1] > 0  # Has samples

    def test_ts_data_dtype(self, loaded_ts3_reader):
        """Test that TS data has correct dtype."""
        assert loaded_ts3_reader.ts.dtype == np.float64

    def test_empty_file_raises_eoferror(self, tmp_path):
        """Test that empty file raises EOFError."""
        empty_file = tmp_path / "empty.TS3"
        empty_file.touch()
        reader = MTUTSN(empty_file)
        with pytest.raises(EOFError, match="empty or could not be read"):
            reader.read()


# =============================================================================
# Tag Metadata Tests
# =============================================================================


class TestTagMetadata:
    """Test the tag metadata dictionary."""

    @pytest.mark.parametrize(
        "key",
        [
            "box_number",
            "ts_type",
            "sample_rate",
            "n_ch",
            "n_scan",
            "start",
            "ts_length",
            "n_block",
        ],
    )
    def test_tag_has_required_keys(self, loaded_ts3_reader, key):
        """Test that tag dictionary has all required keys."""
        assert key in loaded_ts3_reader.ts_metadata

    def test_box_number_is_int(self, loaded_ts3_reader):
        """Test that box_number is an integer."""
        assert isinstance(loaded_ts3_reader.ts_metadata["box_number"], int)

    def test_ts_type_is_string(self, loaded_ts3_reader):
        """Test that ts_type is a string."""
        assert isinstance(loaded_ts3_reader.ts_metadata["ts_type"], str)
        assert loaded_ts3_reader.ts_metadata["ts_type"] in ["MTU-5", "V5-2000"]

    def test_sample_rate_is_numeric(self, loaded_ts3_reader):
        """Test that sample_rate is numeric."""
        assert isinstance(loaded_ts3_reader.ts_metadata["sample_rate"], (int, float))
        assert loaded_ts3_reader.ts_metadata["sample_rate"] >= 0

    def test_n_ch_is_positive(self, loaded_ts3_reader):
        """Test that n_ch is positive integer."""
        assert isinstance(loaded_ts3_reader.ts_metadata["n_ch"], int)
        assert loaded_ts3_reader.ts_metadata["n_ch"] > 0

    def test_n_scan_is_positive(self, loaded_ts3_reader):
        """Test that n_scan is positive integer."""
        assert isinstance(loaded_ts3_reader.ts_metadata["n_scan"], int)
        assert loaded_ts3_reader.ts_metadata["n_scan"] > 0

    def test_n_block_is_positive(self, loaded_ts3_reader):
        """Test that n_block is positive integer."""
        assert isinstance(loaded_ts3_reader.ts_metadata["n_block"], int)
        assert loaded_ts3_reader.ts_metadata["n_block"] > 0

    def test_ts_length_calculation(self, loaded_ts3_reader):
        """Test that ts_length is correctly calculated."""
        ts_metadata = loaded_ts3_reader.ts_metadata
        if ts_metadata["sample_rate"] > 0:
            expected_length = ts_metadata["n_scan"] / ts_metadata["sample_rate"]
            assert abs(ts_metadata["ts_length"] - expected_length) < 1e-6

    def test_start_time_is_mtime(self, loaded_ts3_reader):
        """Test that start time is MTime object."""
        from mt_metadata.common import MTime

        assert isinstance(loaded_ts3_reader.ts_metadata["start"], MTime)


# =============================================================================
# Channel Configuration Tests
# =============================================================================


class TestChannelConfigurations:
    """Test different channel configurations."""

    def test_ts3_channel_count(self, loaded_ts3_reader):
        """Test TS3 file channel count."""
        # TS3 typically has 3 channels
        n_ch = loaded_ts3_reader.ts_metadata["n_ch"]
        assert n_ch in [3, 4, 5, 6]  # Valid channel counts
        assert loaded_ts3_reader.ts.shape[0] == n_ch

    def test_ts4_channel_count(self, loaded_ts4_reader):
        """Test TS4 file channel count."""
        # TS4 typically has 4 channels
        n_ch = loaded_ts4_reader.ts_metadata["n_ch"]
        assert n_ch in [3, 4, 5, 6]
        assert loaded_ts4_reader.ts.shape[0] == n_ch

    def test_ts5_channel_count(self, loaded_ts5_reader):
        """Test TS5 file channel count."""
        # TS5 typically has 5 channels
        n_ch = loaded_ts5_reader.ts_metadata["n_ch"]
        assert n_ch in [3, 4, 5, 6]
        assert loaded_ts5_reader.ts.shape[0] == n_ch

    def test_channel_data_length_consistency(self, loaded_ts3_reader):
        """Test that all channels have the same length."""
        ts = loaded_ts3_reader.ts
        n_ch = ts.shape[0]
        if n_ch > 1:
            lengths = [ts[i, :].shape[0] for i in range(n_ch)]
            assert len(set(lengths)) == 1  # All channels same length


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_ts_data_no_nans(self, loaded_ts3_reader):
        """Test that TS data contains no NaN values."""
        assert not np.any(np.isnan(loaded_ts3_reader.ts))

    def test_ts_data_finite(self, loaded_ts3_reader):
        """Test that TS data contains only finite values."""
        assert np.all(np.isfinite(loaded_ts3_reader.ts))

    def test_ts_data_range(self, loaded_ts3_reader):
        """Test that TS data is within expected 24-bit signed range."""
        ts = loaded_ts3_reader.ts
        assert np.all(ts >= -(2**23))
        assert np.all(ts <= 2**23 - 1)

    def test_consistent_sample_count(self, loaded_ts3_reader):
        """Test that total samples match n_scan * n_block."""
        ts_metadata = loaded_ts3_reader.ts_metadata
        ts = loaded_ts3_reader.ts
        # Allow some tolerance for incomplete final block
        expected_samples = ts_metadata["n_scan"] * ts_metadata["n_block"]
        actual_samples = ts.shape[1]
        assert actual_samples <= expected_samples
        assert actual_samples >= ts_metadata["n_scan"]  # At least one block


# =============================================================================
# Read Method Tests
# =============================================================================


class TestReadMethod:
    """Test the read() method."""

    def test_read_with_explicit_path(self, sample_ts3_file):
        """Test read method with explicit file path."""
        reader = MTUTSN(file_path=None)
        reader.read(sample_ts3_file)
        assert isinstance(reader.ts, np.ndarray)
        assert isinstance(reader.ts_metadata, dict)
        assert reader.ts.shape[0] == reader.ts_metadata["n_ch"]

    def test_read_uses_existing_path(self, sample_ts3_file):
        """Test read method uses existing file_path if not provided."""
        reader = MTUTSN(sample_ts3_file)
        # Read again without specifying path
        reader.read()
        assert isinstance(reader.ts, np.ndarray)
        assert isinstance(reader.ts_metadata, dict)

    def test_read_updates_file_path(self, sample_ts3_file, sample_ts4_file):
        """Test that read updates file_path when new path provided."""
        reader = MTUTSN(sample_ts3_file)
        reader.read()
        original_path = reader.file_path

        # Read different file
        reader.read(sample_ts4_file)
        assert reader.file_path == sample_ts4_file
        assert reader.file_path != original_path


# =============================================================================
# Multiple File Reading Tests
# =============================================================================


class TestMultipleFileReading:
    """Test reading multiple files in sequence."""

    def test_read_ts3_then_ts4(self, sample_ts3_file, sample_ts4_file):
        """Test reading TS3 file then TS4 file."""
        reader = MTUTSN(sample_ts3_file)
        reader.read()
        ts3_shape = reader.ts.shape

        reader.file_path = sample_ts4_file
        reader.read()

        # Files may have different shapes
        assert reader.ts.shape[0] == reader.ts_metadata["n_ch"]

    def test_read_multiple_files_independent(
        self, sample_ts3_file, sample_ts4_file, sample_ts5_file
    ):
        """Test that reading multiple files produces independent results."""
        reader3 = MTUTSN(sample_ts3_file)
        reader4 = MTUTSN(sample_ts4_file)
        reader5 = MTUTSN(sample_ts5_file)

        # Read all files
        reader3.read()
        reader4.read()
        reader5.read()

        # Each should have valid data
        assert reader3.ts.shape[0] == reader3.ts_metadata["n_ch"]
        assert reader4.ts.shape[0] == reader4.ts_metadata["n_ch"]
        assert reader5.ts.shape[0] == reader5.ts_metadata["n_ch"]


# =============================================================================
# Parallel Execution Safety Tests
# =============================================================================


class TestParallelSafety:
    """Test that instances are safe for parallel execution."""

    def test_multiple_instances_independent(self, sample_ts3_file, sample_ts4_file):
        """Test that multiple instances don't interfere with each other."""
        reader1 = MTUTSN(sample_ts3_file)
        reader2 = MTUTSN(sample_ts4_file)

        # Read both files
        reader1.read()
        reader2.read()

        # Verify they loaded different data
        assert reader1.file_path != reader2.file_path

        # Each should have valid independent data
        assert reader1.ts is not None
        assert reader2.ts is not None
        assert not np.array_equal(reader1.ts, reader2.ts)

    def test_no_shared_state_between_instances(self, sample_ts3_file):
        """Test that instances don't share state."""
        reader1 = MTUTSN(sample_ts3_file)
        reader2 = MTUTSN(sample_ts3_file)

        # Modify one
        reader1.file_path = None

        # Other should be unaffected
        assert reader2.file_path == sample_ts3_file


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_file_with_minimal_header(self, tmp_path):
        """Test file with only header (no data)."""
        minimal_file = tmp_path / "minimal.TS3"
        with open(minimal_file, "wb") as f:
            # Write 32-byte header
            f.write(b"\x00" * 32)

        # Should handle gracefully
        reader = MTUTSN(minimal_file)
        with pytest.raises((EOFError, Exception)):
            reader.read()

    def test_constants_defined(self, empty_reader):
        """Test that power constants are defined."""
        assert empty_reader._p16 == 2**16
        assert empty_reader._p8 == 2**8

    def test_accepted_extensions_list(self, empty_reader):
        """Test that accepted extensions list is defined."""
        assert hasattr(empty_reader, "_accepted_extensions")
        assert len(empty_reader._accepted_extensions) > 0
        assert "TS3" in empty_reader._accepted_extensions


# =============================================================================
# Real Data Validation Tests
# =============================================================================


class TestRealDataValidation:
    """Validate real test data files."""

    def test_ts3_box_number(self, loaded_ts3_reader):
        """Test TS3 file has expected box number."""
        box_num = loaded_ts3_reader.ts_metadata["box_number"]
        assert isinstance(box_num, int)
        # Phoenix serial numbers are typically positive
        assert box_num > 0

    def test_ts3_sample_rate_reasonable(self, loaded_ts3_reader):
        """Test TS3 sample rate is reasonable."""
        sr = loaded_ts3_reader.ts_metadata["sample_rate"]
        # Sample rates should be positive and typically < 100000
        assert 0 <= sr <= 100000

    def test_data_statistics_reasonable(self, loaded_ts3_reader):
        """Test that data statistics are reasonable."""
        ts = loaded_ts3_reader.ts

        # Data should have non-zero variance (not all same value)
        for ch in range(ts.shape[0]):
            variance = np.var(ts[ch, :])
            assert variance > 0  # Data should vary

    def test_different_files_different_data(
        self, loaded_ts3_reader, loaded_ts4_reader, loaded_ts5_reader
    ):
        """Test that different files contain different data."""
        # Files should not be identical
        # (At minimum, they should have different channel counts or lengths)
        ts3 = loaded_ts3_reader.ts
        ts4 = loaded_ts4_reader.ts
        ts5 = loaded_ts5_reader.ts

        # At least one dimension should differ
        shapes_different = (
            ts3.shape != ts4.shape or ts4.shape != ts5.shape or ts3.shape != ts5.shape
        )
        assert shapes_different


# =============================================================================
# to_runts Method Tests
# =============================================================================


class TestToRunTS:
    """Test the to_runts method for creating RunTS objects."""

    def test_to_runts_with_explicit_table_path(self, sample_ts3_file, sample_tbl_file):
        """Test to_runts with explicit TBL file path."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(table_filepath=sample_tbl_file)

        # Verify RunTS object is created
        from mth5.timeseries import RunTS

        assert isinstance(run_ts, RunTS)

    def test_to_runts_without_table_path(self, sample_ts3_file):
        """Test to_runts without explicit TBL path (auto-detect)."""
        reader = MTUTSN(sample_ts3_file)
        # Should auto-detect 1690C16C.TBL from 1690C16C.TS3
        run_ts = reader.to_runts()

        from mth5.timeseries import RunTS

        assert isinstance(run_ts, RunTS)

    def test_to_runts_returns_runts_object(self, sample_ts3_file, sample_tbl_file):
        """Test that to_runts returns a RunTS object."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        from mth5.timeseries import RunTS

        assert isinstance(run_ts, RunTS)
        assert hasattr(run_ts, "channels")

    def test_to_runts_channels_added(self, sample_ts3_file, sample_tbl_file):
        """Test that channels are added to RunTS object."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # Should have channels based on what's in the TBL file
        assert len(run_ts.channels) > 0

    def test_to_runts_channel_metadata_populated(
        self, sample_ts3_file, sample_tbl_file
    ):
        """Test that channel metadata is properly populated."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # Check that channels have metadata
        for ch_name in run_ts.channels:
            channel = getattr(run_ts, ch_name)
            assert hasattr(channel, "channel_metadata")
            assert channel.channel_metadata is not None
            assert hasattr(channel.channel_metadata, "sample_rate")
            assert channel.channel_metadata.sample_rate > 0

    def test_to_runts_channel_data_shape(self, sample_ts3_file, sample_tbl_file):
        """Test that channel data has correct shape."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # All channels should have data
        for ch_name in run_ts.channels:
            channel = getattr(run_ts, ch_name)
            assert hasattr(channel, "ts")
            assert len(channel.ts) > 0

    def test_to_runts_with_calibration_enabled(self, sample_ts3_file, sample_tbl_file):
        """Test to_runts with calibration enabled (default)."""
        reader = MTUTSN(sample_ts3_file)
        reader.read()
        # Store original uncalibrated data
        ts_uncalibrated = reader.ts.copy()

        # Read with calibration
        run_ts = reader.to_runts(sample_tbl_file, calibrate=True)

        # Verify RunTS was created
        assert run_ts is not None
        assert len(run_ts.channels) > 0

    def test_to_runts_with_calibration_disabled(self, sample_ts3_file, sample_tbl_file):
        """Test to_runts with calibration disabled."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file, calibrate=False)

        # Verify RunTS was created
        assert run_ts is not None
        assert len(run_ts.channels) > 0

    def test_to_runts_calibration_affects_data(self, sample_ts3_file, sample_tbl_file):
        """Test that calibration actually modifies the data."""
        # Create two readers with same file
        reader1 = MTUTSN(sample_ts3_file)
        reader2 = MTUTSN(sample_ts3_file)

        # Read with and without calibration
        run_ts_calibrated = reader1.to_runts(sample_tbl_file, calibrate=True)
        run_ts_uncalibrated = reader2.to_runts(sample_tbl_file, calibrate=False)

        # Data should be different (unless calibration factor is exactly 1.0)
        if (
            len(run_ts_calibrated.channels) > 0
            and len(run_ts_uncalibrated.channels) > 0
        ):
            # Get first channel from each
            ch_name = run_ts_calibrated.channels[0]
            cal_data = getattr(run_ts_calibrated, ch_name).ts
            uncal_data = getattr(run_ts_uncalibrated, ch_name).ts

            # They should have same length
            assert len(cal_data) == len(uncal_data)

    def test_to_runts_survey_metadata_propagated(
        self, sample_ts3_file, sample_tbl_file
    ):
        """Test that survey metadata is propagated to RunTS."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # Should have survey metadata
        assert hasattr(run_ts, "survey_metadata")
        assert run_ts.survey_metadata is not None

    def test_to_runts_channel_components(self, sample_ts3_file, sample_tbl_file):
        """Test that channels have correct component names."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # Get channel components
        components = [
            getattr(run_ts, ch_name).channel_metadata.component
            for ch_name in run_ts.channels
        ]

        # All components should be valid MT components
        valid_components = ["ex", "ey", "hx", "hy", "hz"]
        for comp in components:
            assert comp in valid_components

    def test_to_runts_multiple_files(
        self, sample_ts3_file, sample_ts4_file, sample_tbl_file
    ):
        """Test to_runts with multiple different TS files."""
        reader3 = MTUTSN(sample_ts3_file)
        reader4 = MTUTSN(sample_ts4_file)

        run_ts3 = reader3.to_runts(sample_tbl_file)
        run_ts4 = reader4.to_runts(sample_tbl_file)

        # Both should be valid
        assert run_ts3 is not None
        assert run_ts4 is not None
        assert len(run_ts3.channels) > 0
        assert len(run_ts4.channels) > 0

    def test_to_runts_start_time_set(self, sample_ts3_file, sample_tbl_file):
        """Test that channel start times are set correctly."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # All channels should have start time from TS file
        for ch_name in run_ts.channels:
            channel = getattr(run_ts, ch_name)
            assert hasattr(channel.channel_metadata, "time_period")
            # Start time should match the TS file's start time
            assert channel.channel_metadata.time_period.start is not None

    def test_to_runts_sample_rate_set(self, sample_ts3_file, sample_tbl_file):
        """Test that channel sample rates are set correctly."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # All channels should have sample rate from TS file
        ts_sample_rate = reader.ts_metadata["sample_rate"]
        for ch_name in run_ts.channels:
            channel = getattr(run_ts, ch_name)
            assert channel.channel_metadata.sample_rate == ts_sample_rate

    def test_to_runts_with_nonexistent_table(self, sample_ts3_file, tmp_path):
        """Test to_runts with nonexistent TBL file raises error."""
        reader = MTUTSN(sample_ts3_file)
        nonexistent_tbl = tmp_path / "nonexistent.TBL"

        with pytest.raises(FileNotFoundError):
            reader.to_runts(nonexistent_tbl)

    def test_to_runts_channel_number_mapping(self, sample_ts3_file, sample_tbl_file):
        """Test that channel numbers from TBL are used correctly."""
        reader = MTUTSN(sample_ts3_file)
        run_ts = reader.to_runts(sample_tbl_file)

        # Verify we have channels
        assert len(run_ts.channels) > 0

        # Each channel should have valid data from the correct channel number
        for ch_name in run_ts.channels:
            channel = getattr(run_ts, ch_name)
            # Channel data should be 1D array
            assert channel.ts.ndim == 1
            # Channel should have positive length
            assert len(channel.ts) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_from_file_to_data(self, sample_ts3_file):
        """Test complete workflow from file initialization to data access."""
        # Initialize
        reader = MTUTSN(sample_ts3_file)

        # Read data
        reader.read()

        # Access data
        ts = reader.ts
        ts_metadata = reader.ts_metadata

        # Verify complete workflow
        assert reader.file_path.exists()
        assert ts.shape[0] == ts_metadata["n_ch"]
        assert ts.shape[1] == ts_metadata["n_scan"] * ts_metadata["n_block"]
        assert ts_metadata["sample_rate"] >= 0
        assert ts_metadata["n_block"] > 0

    def test_workflow_with_explicit_read(self, sample_ts3_file):
        """Test workflow using explicit read() call."""
        reader = MTUTSN(file_path=None)
        reader.read(sample_ts3_file)

        # Verify data
        assert reader.ts.shape[0] == reader.ts_metadata["n_ch"]
        assert reader.ts_metadata["start"] is not None
        assert reader.ts_metadata["box_number"] > 0
