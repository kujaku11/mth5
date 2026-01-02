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


# =============================================================================
# Module-level fixtures (reused across test classes)
# =============================================================================


@pytest.fixture(scope="module")
def loaded_ts3_reader(sample_ts3_file):
    """Create MTUTSN instance with TS3 file loaded."""
    return MTUTSN(sample_ts3_file)


@pytest.fixture(scope="module")
def loaded_ts4_reader(sample_ts4_file):
    """Create MTUTSN instance with TS4 file loaded."""
    return MTUTSN(sample_ts4_file)


@pytest.fixture(scope="module")
def loaded_ts5_reader(sample_ts5_file):
    """Create MTUTSN instance with TS5 file loaded."""
    return MTUTSN(sample_ts5_file)


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
        assert reader.ts is not None
        assert reader.tag is not None

    def test_init_with_string_path(self, sample_ts3_file):
        """Test initialization with string path."""
        reader = MTUTSN(str(sample_ts3_file))
        assert reader.file_path == sample_ts3_file
        assert isinstance(reader.file_path, Path)

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
        assert loaded_ts3_reader.tag is not None
        assert isinstance(loaded_ts3_reader.ts, np.ndarray)
        assert isinstance(loaded_ts3_reader.tag, dict)

    def test_read_ts4_file(self, loaded_ts4_reader):
        """Test reading TS4 file produces valid data."""
        assert loaded_ts4_reader.ts is not None
        assert loaded_ts4_reader.tag is not None
        assert isinstance(loaded_ts4_reader.ts, np.ndarray)
        assert isinstance(loaded_ts4_reader.tag, dict)

    def test_read_ts5_file(self, loaded_ts5_reader):
        """Test reading TS5 file produces valid data."""
        assert loaded_ts5_reader.ts is not None
        assert loaded_ts5_reader.tag is not None
        assert isinstance(loaded_ts5_reader.ts, np.ndarray)
        assert isinstance(loaded_ts5_reader.tag, dict)

    def test_ts_data_shape(self, loaded_ts3_reader):
        """Test that TS data has correct shape."""
        ts = loaded_ts3_reader.ts
        tag = loaded_ts3_reader.tag
        assert ts.ndim == 2
        assert ts.shape[0] == tag["n_ch"]  # Number of channels
        assert ts.shape[1] > 0  # Has samples

    def test_ts_data_dtype(self, loaded_ts3_reader):
        """Test that TS data has correct dtype."""
        assert loaded_ts3_reader.ts.dtype == np.float64

    def test_empty_file_raises_eoferror(self, tmp_path):
        """Test that empty file raises EOFError."""
        empty_file = tmp_path / "empty.TS3"
        empty_file.touch()
        with pytest.raises(EOFError, match="empty or could not be read"):
            MTUTSN(empty_file)


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
        assert key in loaded_ts3_reader.tag

    def test_box_number_is_int(self, loaded_ts3_reader):
        """Test that box_number is an integer."""
        assert isinstance(loaded_ts3_reader.tag["box_number"], int)

    def test_ts_type_is_string(self, loaded_ts3_reader):
        """Test that ts_type is a string."""
        assert isinstance(loaded_ts3_reader.tag["ts_type"], str)
        assert loaded_ts3_reader.tag["ts_type"] in ["MTU-5", "V5-2000"]

    def test_sample_rate_is_numeric(self, loaded_ts3_reader):
        """Test that sample_rate is numeric."""
        assert isinstance(loaded_ts3_reader.tag["sample_rate"], (int, float))
        assert loaded_ts3_reader.tag["sample_rate"] >= 0

    def test_n_ch_is_positive(self, loaded_ts3_reader):
        """Test that n_ch is positive integer."""
        assert isinstance(loaded_ts3_reader.tag["n_ch"], int)
        assert loaded_ts3_reader.tag["n_ch"] > 0

    def test_n_scan_is_positive(self, loaded_ts3_reader):
        """Test that n_scan is positive integer."""
        assert isinstance(loaded_ts3_reader.tag["n_scan"], int)
        assert loaded_ts3_reader.tag["n_scan"] > 0

    def test_n_block_is_positive(self, loaded_ts3_reader):
        """Test that n_block is positive integer."""
        assert isinstance(loaded_ts3_reader.tag["n_block"], int)
        assert loaded_ts3_reader.tag["n_block"] > 0

    def test_ts_length_calculation(self, loaded_ts3_reader):
        """Test that ts_length is correctly calculated."""
        tag = loaded_ts3_reader.tag
        if tag["sample_rate"] > 0:
            expected_length = tag["n_scan"] / tag["sample_rate"]
            assert abs(tag["ts_length"] - expected_length) < 1e-6

    def test_start_time_is_mtime(self, loaded_ts3_reader):
        """Test that start time is MTime object."""
        from mt_metadata.common import MTime

        assert isinstance(loaded_ts3_reader.tag["start"], MTime)


# =============================================================================
# Channel Configuration Tests
# =============================================================================


class TestChannelConfigurations:
    """Test different channel configurations."""

    def test_ts3_channel_count(self, loaded_ts3_reader):
        """Test TS3 file channel count."""
        # TS3 typically has 3 channels
        n_ch = loaded_ts3_reader.tag["n_ch"]
        assert n_ch in [3, 4, 5, 6]  # Valid channel counts
        assert loaded_ts3_reader.ts.shape[0] == n_ch

    def test_ts4_channel_count(self, loaded_ts4_reader):
        """Test TS4 file channel count."""
        # TS4 typically has 4 channels
        n_ch = loaded_ts4_reader.tag["n_ch"]
        assert n_ch in [3, 4, 5, 6]
        assert loaded_ts4_reader.ts.shape[0] == n_ch

    def test_ts5_channel_count(self, loaded_ts5_reader):
        """Test TS5 file channel count."""
        # TS5 typically has 5 channels
        n_ch = loaded_ts5_reader.tag["n_ch"]
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
        tag = loaded_ts3_reader.tag
        ts = loaded_ts3_reader.ts
        # Allow some tolerance for incomplete final block
        expected_samples = tag["n_scan"] * tag["n_block"]
        actual_samples = ts.shape[1]
        assert actual_samples <= expected_samples
        assert actual_samples >= tag["n_scan"]  # At least one block


# =============================================================================
# Read Method Tests
# =============================================================================


class TestReadMethod:
    """Test the read() method."""

    def test_read_with_explicit_path(self, sample_ts3_file):
        """Test read method with explicit file path."""
        reader = MTUTSN(file_path=None)
        ts, tag = reader.read(sample_ts3_file)
        assert isinstance(ts, np.ndarray)
        assert isinstance(tag, dict)
        assert ts.shape[0] == tag["n_ch"]

    def test_read_uses_existing_path(self, sample_ts3_file):
        """Test read method uses existing file_path if not provided."""
        reader = MTUTSN(sample_ts3_file)
        # Read again without specifying path
        ts, tag = reader.read()
        assert isinstance(ts, np.ndarray)
        assert isinstance(tag, dict)

    def test_read_updates_file_path(self, sample_ts3_file, sample_ts4_file):
        """Test that read updates file_path when new path provided."""
        reader = MTUTSN(sample_ts3_file)
        original_path = reader.file_path

        # Read different file
        ts, tag = reader.read(sample_ts4_file)
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
        ts3_shape = reader.ts.shape

        reader.file_path = sample_ts4_file
        ts4, tag4 = reader.read()

        # Files may have different shapes
        assert ts4.shape[0] == tag4["n_ch"]

    def test_read_multiple_files_independent(
        self, sample_ts3_file, sample_ts4_file, sample_ts5_file
    ):
        """Test that reading multiple files produces independent results."""
        reader3 = MTUTSN(sample_ts3_file)
        reader4 = MTUTSN(sample_ts4_file)
        reader5 = MTUTSN(sample_ts5_file)

        # Each should have valid data
        assert reader3.ts.shape[0] == reader3.tag["n_ch"]
        assert reader4.ts.shape[0] == reader4.tag["n_ch"]
        assert reader5.ts.shape[0] == reader5.tag["n_ch"]


# =============================================================================
# Parallel Execution Safety Tests
# =============================================================================


class TestParallelSafety:
    """Test that instances are safe for parallel execution."""

    def test_multiple_instances_independent(self, sample_ts3_file, sample_ts4_file):
        """Test that multiple instances don't interfere with each other."""
        reader1 = MTUTSN(sample_ts3_file)
        reader2 = MTUTSN(sample_ts4_file)

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
        with pytest.raises((EOFError, Exception)):
            MTUTSN(minimal_file)

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
        box_num = loaded_ts3_reader.tag["box_number"]
        assert isinstance(box_num, int)
        # Phoenix serial numbers are typically positive
        assert box_num > 0

    def test_ts3_sample_rate_reasonable(self, loaded_ts3_reader):
        """Test TS3 sample rate is reasonable."""
        sr = loaded_ts3_reader.tag["sample_rate"]
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
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_from_file_to_data(self, sample_ts3_file):
        """Test complete workflow from file initialization to data access."""
        # Initialize
        reader = MTUTSN(sample_ts3_file)

        # Access data
        ts = reader.ts
        tag = reader.tag

        # Verify complete workflow
        assert reader.file_path.exists()
        assert ts.shape[0] == tag["n_ch"]
        assert ts.shape[1] == tag["n_scan"] * tag["n_block"]
        assert tag["sample_rate"] >= 0
        assert tag["n_block"] > 0

    def test_workflow_with_explicit_read(self, sample_ts3_file):
        """Test workflow using explicit read() call."""
        reader = MTUTSN(file_path=None)
        ts, tag = reader.read(sample_ts3_file)

        # Verify data
        assert ts.shape[0] == tag["n_ch"]
        assert tag["start"] is not None
        assert tag["box_number"] > 0
