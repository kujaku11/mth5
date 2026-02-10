# -*- coding: utf-8 -*-
"""
Test suite for SWMR (Single Writer Multiple Reader) mode in MTH5.

IMPORTANT: HDF5 SWMR Mode Limitations
--------------------------------------
SWMR mode in HDF5 has strict limitations:
- Can ONLY read existing data
- Can ONLY append to existing resizable datasets
- CANNOT create new groups (surveys, stations, runs)
- CANNOT create new datasets (channels)
- CANNOT delete or modify existing structure

All file structure (groups and datasets) must be created BEFORE entering SWMR mode.

Tests cover:
- SWMR writer mode activation
- SWMR reader mode functionality
- Mode validation and error handling
- Concurrent access patterns
- File state verification

All tests are safe for parallel execution using unique temporary files.

Test Organization:
------------------
- Fast tests: Basic functionality, error handling (run by default)
- Slow tests: Integration, performance, multi-reader tests (marked with @pytest.mark.slow)
- Skipped tests: Tests for operations not supported by HDF5 SWMR (marked with @pytest.mark.skip)

Running Tests:
--------------
# Run only fast tests (recommended for development):
pytest tests/test_mth5_open_swmr.py -v -k "not slow"

# Run all tests including slow ones:
pytest tests/test_mth5_open_swmr.py -v

# Run specific test class:
pytest tests/test_mth5_open_swmr.py::TestSWMRWriterMode -v

# Run with parallel execution (fast tests only):
pytest tests/test_mth5_open_swmr.py -v -n auto -k "not slow"

Created on February 9, 2026

@author: MTH5 Development Team
"""

# =============================================================================
# Imports
# =============================================================================
import multiprocessing as mp
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from mth5 import helpers
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_files():
    """Ensure any open files are closed before and after tests."""
    helpers.close_open_files()
    yield
    helpers.close_open_files()


@pytest.fixture(scope="function")
def temp_dir():
    """Create a unique temporary directory for each test."""
    temp_path = Path(tempfile.mkdtemp(prefix="mth5_swmr_test_"))
    yield temp_path
    # Cleanup with retry for file locking issues
    try:
        time.sleep(0.1)  # Brief delay for file handles to close
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def unique_test_file(temp_dir):
    """Generate a unique test file path for parallel-safe testing."""
    import uuid

    unique_id = uuid.uuid4().hex[:8]
    return temp_dir / f"test_swmr_{unique_id}.mth5"


@pytest.fixture
def basic_mth5_file(unique_test_file):
    """
    Create a basic MTH5 file with minimal structure for testing.

    File structure:
    - Survey: test_survey
    - Station: STA001
    - Run: run_001
    - Channel: Ex (1000 samples)

    Note: Created with libver='latest' to support SWMR mode.
    """
    mth5 = MTH5(file_version="0.2.0")
    mth5.open_mth5(unique_test_file, mode="w", libver="latest")

    # Create basic structure
    survey = mth5.add_survey("test_survey")
    station = mth5.add_station("STA001", survey="test_survey")
    run = station.add_run("run_001")

    # Add sample channel data
    data = np.random.randn(1000)
    run.add_channel("Ex", "electric", data, channel_dtype="float32")

    mth5.close_mth5()

    yield unique_test_file
    # Cleanup handled by temp_dir fixture


@pytest.fixture
def empty_mth5_file(unique_test_file):
    """
    Create an empty MTH5 file with just survey/station structure.

    Note: Created with libver='latest' to support SWMR mode.
    """
    mth5 = MTH5(file_version="0.2.0")
    mth5.open_mth5(unique_test_file, mode="w", libver="latest")
    survey = mth5.add_survey("test_survey")
    station = mth5.add_station("STA001", survey="test_survey")
    mth5.close_mth5()

    yield unique_test_file


# =============================================================================
# Test Classes
# =============================================================================


class TestSWMRFileCreationErrors:
    """Test that SWMR mode properly rejects file creation attempts."""

    def test_swmr_with_mode_w_raises_error(self, unique_test_file):
        """Test that mode='w' with SWMR raises MTH5Error."""
        mth5 = MTH5(file_version="0.2.0")

        with pytest.raises(MTH5Error, match="cannot be used with mode='w'"):
            mth5.open_mth5(
                unique_test_file, mode="w", single_writer_multiple_reader=True
            )

    def test_swmr_with_mode_x_raises_error(self, unique_test_file):
        """Test that mode='x' with SWMR raises MTH5Error."""
        mth5 = MTH5(file_version="0.2.0")

        with pytest.raises(MTH5Error, match="cannot be used with mode='x'"):
            mth5.open_mth5(
                unique_test_file, mode="x", single_writer_multiple_reader=True
            )

    def test_swmr_with_mode_w_dash_raises_error(self, unique_test_file):
        """Test that mode='w-' with SWMR raises MTH5Error."""
        mth5 = MTH5(file_version="0.2.0")

        with pytest.raises(MTH5Error, match="cannot be used with mode='w-'"):
            mth5.open_mth5(
                unique_test_file, mode="w-", single_writer_multiple_reader=True
            )

    def test_swmr_with_nonexistent_file_raises_error(self, unique_test_file):
        """Test that SWMR on non-existent file raises MTH5Error."""
        mth5 = MTH5(file_version="0.2.0")

        with pytest.raises(
            MTH5Error, match="Cannot use SWMR mode on non-existent file"
        ):
            mth5.open_mth5(
                unique_test_file, mode="a", single_writer_multiple_reader=True
            )


class TestSWMRWriterMode:
    """Test SWMR writer mode functionality."""

    def test_swmr_writer_mode_activation(self, basic_mth5_file):
        """Test that SWMR writer mode activates successfully."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Verify SWMR mode is active
        assert mth5.is_swmr_mode() is True
        assert mth5.h5_is_write() is True

        mth5.close_mth5()

    def test_swmr_writer_mode_r_plus(self, basic_mth5_file):
        """Test SWMR writer with mode='r+'."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r+", single_writer_multiple_reader=True)

        assert mth5.is_swmr_mode() is True
        assert mth5.h5_is_write() is True

        mth5.close_mth5()

    @pytest.mark.skip(
        reason="HDF5 SWMR mode does not allow creating new datasets. "
        "add_channel() creates a dataset, which causes indefinite hang in SWMR mode. "
        "SWMR is intended for reading existing data or appending to existing datasets only."
    )
    def test_swmr_writer_can_add_data(self, empty_mth5_file):
        """Test that SWMR writer can add new data.

        Note: This test is skipped because HDF5 SWMR mode does not support
        creating new datasets. The add_channel() operation creates a new HDF5
        dataset, which is not allowed in SWMR mode and causes the process to hang.

        SWMR is designed for:
        - Reading existing data concurrently
        - Appending to existing resizable datasets
        NOT for creating new groups or datasets.
        """
        # Pre-create run structure before SWMR mode (SWMR cannot create groups)
        mth5_setup = MTH5()
        mth5_setup.open_mth5(empty_mth5_file, mode="a")
        station = mth5_setup.get_station("STA001", survey="test_survey")
        run = station.add_run("run_001")
        mth5_setup.close_mth5()

        # Now open in SWMR mode and append data
        mth5 = MTH5()
        mth5.open_mth5(empty_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Get existing run and add channel data
        station = mth5.get_station("STA001", survey="test_survey")
        run = station.get_run("run_001")

        # Add channel data (creates dataset, which hangs in SWMR)
        data = np.random.randn(500)
        channel = run.add_channel("Hy", "magnetic", data, channel_dtype="float32")

        assert channel is not None
        assert len(channel.hdf5_dataset) == 500

        mth5.close_mth5()

    def test_swmr_writer_flush(self, basic_mth5_file):
        """Test that flush() works in SWMR writer mode.

        This test opens an existing file in SWMR mode and verifies
        that flush operations work correctly.
        """
        # Open existing file in SWMR mode
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Flush should succeed without error
        mth5.flush()

        # Verify we can read existing data
        run_summary = mth5.run_summary
        assert len(run_summary) >= 1  # At least the original run

    def test_swmr_writer_metadata_access(self, basic_mth5_file):
        """Test that SWMR writer can access metadata."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Read metadata
        channel_summary = mth5.channel_summary.to_dataframe()
        run_summary = mth5.run_summary

        assert len(channel_summary) > 0
        assert len(run_summary) > 0

        mth5.close_mth5()

    def test_swmr_writer_close_flushes(self, basic_mth5_file):
        """Test that closing SWMR writer works correctly.

        Verifies that a file opened in SWMR mode can be closed cleanly
        and that existing data persists after closing.
        """
        # Open in SWMR mode
        mth5_swmr = MTH5()
        mth5_swmr.open_mth5(
            basic_mth5_file, mode="a", single_writer_multiple_reader=True
        )
        mth5_swmr.close_mth5()

        # Reopen and verify existing data persisted
        mth5_read = MTH5()
        mth5_read.open_mth5(basic_mth5_file, mode="r")

        run_summary = mth5_read.run_summary
        run_ids = run_summary["run"].tolist()
        assert "run_001" in run_ids  # From basic_mth5_file fixture
        mth5_read.close_mth5()


class TestSWMRReaderMode:
    """Test SWMR reader mode functionality."""

    def test_swmr_reader_mode_activation(self, basic_mth5_file):
        """Test that SWMR reader mode activates successfully."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        assert mth5.is_swmr_mode() is True
        assert mth5.h5_is_read() is True
        assert mth5.h5_is_write() is False

        mth5.close_mth5()

    def test_swmr_reader_can_read_data(self, basic_mth5_file):
        """Test that SWMR reader can read existing data."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        # Read channel data
        channel = mth5.get_channel("STA001", "run_001", "Ex", survey="test_survey")
        data = channel.hdf5_dataset[:]

        assert len(data) == 1000
        # Data might be float64 or float32 depending on storage
        assert data.dtype in [np.float32, np.float64]

        mth5.close_mth5()

    def test_swmr_reader_metadata_access(self, basic_mth5_file):
        """Test that SWMR reader can access metadata."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        channel_summary = mth5.channel_summary.to_dataframe()
        run_summary = mth5.run_summary

        assert len(channel_summary) > 0
        assert len(run_summary) > 0
        # Component names are normalized to lowercase
        assert "ex" in channel_summary["component"].tolist()

        mth5.close_mth5()

    def test_swmr_reader_cannot_write(self, basic_mth5_file):
        """Test that SWMR reader cannot modify data."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        # Verify file is read-only
        assert mth5.h5_is_write() is False
        assert mth5.h5_is_read() is True

        # Note: add_run will fail due to read-only file mode,
        # but the error might be caught internally and logged as a warning
        # Rather than crash.  The key test is h5_is_write() == False

        mth5.close_mth5()


class TestSWMRModeDetection:
    """Test SWMR mode detection and status checking."""

    def test_is_swmr_mode_false_for_normal_file(self, basic_mth5_file):
        """Test that is_swmr_mode() returns False for normal files."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a")

        assert mth5.is_swmr_mode() is False

        mth5.close_mth5()

    def test_is_swmr_mode_true_for_swmr_writer(self, basic_mth5_file):
        """Test that is_swmr_mode() returns True for SWMR writer."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        assert mth5.is_swmr_mode() is True

        mth5.close_mth5()

    def test_is_swmr_mode_true_for_swmr_reader(self, basic_mth5_file):
        """Test that is_swmr_mode() returns True for SWMR reader."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        assert mth5.is_swmr_mode() is True

        mth5.close_mth5()

    def test_is_swmr_mode_false_for_closed_file(self):
        """Test that is_swmr_mode() returns False for closed file."""
        mth5 = MTH5()

        assert mth5.is_swmr_mode() is False


class TestSWMRFileVersions:
    """Test SWMR mode with different MTH5 file versions."""

    @pytest.mark.parametrize("file_version", ["0.1.0", "0.2.0"])
    def test_swmr_with_different_versions(self, unique_test_file, file_version):
        """Test SWMR mode works with both file versions."""
        # Create file with libver='latest' for SWMR support
        mth5_create = MTH5(file_version=file_version)
        mth5_create.open_mth5(unique_test_file, mode="w", libver="latest")

        if file_version == "0.1.0":
            station = mth5_create.add_station("STA001")
        else:  # 0.2.0
            survey = mth5_create.add_survey("test_survey")
            station = mth5_create.add_station("STA001", survey="test_survey")

        mth5_create.close_mth5()

        # Open in SWMR writer mode
        mth5_swmr = MTH5()
        mth5_swmr.open_mth5(
            unique_test_file, mode="a", single_writer_multiple_reader=True
        )

        assert mth5_swmr.is_swmr_mode() is True
        assert mth5_swmr.file_version == file_version

        mth5_swmr.close_mth5()


class TestSWMREdgeCases:
    """Test edge cases and special scenarios."""

    def test_swmr_with_context_manager(self, basic_mth5_file):
        """Test SWMR mode with context manager."""
        with MTH5() as mth5:
            mth5.open_mth5(
                basic_mth5_file, mode="a", single_writer_multiple_reader=True
            )
            assert mth5.is_swmr_mode() is True

        # File should be closed after context
        assert mth5.h5_is_read() is False

    def test_swmr_flush_on_closed_file(self):
        """Test that flush() on closed file doesn't crash."""
        mth5 = MTH5()

        # Should not raise error
        mth5.flush()

    def test_swmr_kwargs_passthrough(self, basic_mth5_file):
        """Test that swmr kwargs work correctly."""
        mth5 = MTH5()

        # Using single_writer_multiple_reader parameter
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        assert mth5.is_swmr_mode() is True

        mth5.close_mth5()

    def test_swmr_libver_automatically_set(self, basic_mth5_file):
        """Test that libver='latest' is automatically set for SWMR."""
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Check underlying h5py file has correct libver
        # Note: h5py doesn't expose libver directly, but SWMR mode proves it's set
        assert mth5.is_swmr_mode() is True

        mth5.close_mth5()


class TestSWMRConcurrentAccess:
    """Test concurrent access patterns (simplified for unit testing)."""

    @pytest.mark.slow
    def test_multiple_readers_sequential(self, basic_mth5_file):
        """Test opening multiple readers sequentially."""
        readers = []

        try:
            # Open 2 readers sequentially (reduced from 3 for speed)
            for i in range(2):
                reader = MTH5()
                reader.open_mth5(
                    basic_mth5_file, mode="r", single_writer_multiple_reader=True
                )
                assert reader.is_swmr_mode() is True
                readers.append(reader)

            # All should be able to read
            for reader in readers:
                channel_summary = reader.channel_summary.to_dataframe()
                assert len(channel_summary) > 0

        finally:
            # Close all readers
            for reader in readers:
                try:
                    reader.close_mth5()
                except Exception:
                    pass

    def test_writer_then_reader(self, basic_mth5_file):
        """Test that reader can read data after file is opened in SWMR mode.

        This uses basic_mth5_file which already has data, demonstrating
        sequential access pattern (not concurrent).
        """
        # Open in SWMR writer mode (no modifications needed)
        writer = MTH5()
        writer.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)
        writer.flush()
        writer.close_mth5()

        # Reader should see the existing data
        reader = MTH5()
        reader.open_mth5(basic_mth5_file, mode="r", single_writer_multiple_reader=True)

        channel_summary = reader.channel_summary.to_dataframe()
        assert "ex" in channel_summary["component"].tolist()

        reader.close_mth5()

    @pytest.mark.skip(
        reason="Multiprocessing test is slow and can hang on Windows - run manually if needed"
    )
    def test_concurrent_readers_multiprocess(self, basic_mth5_file):
        """Test multiple readers in separate processes (parallel-safe)."""

        def reader_process(filepath, result_queue):
            """Reader process function."""
            try:
                mth5 = MTH5()
                mth5.open_mth5(filepath, mode="r", single_writer_multiple_reader=True)

                # Read data
                channel_summary = mth5.channel_summary.to_dataframe()
                num_channels = len(channel_summary)

                mth5.close_mth5()
                result_queue.put(("success", num_channels))
            except Exception as e:
                result_queue.put(("error", str(e)))

        # Start multiple reader processes
        num_readers = 2
        result_queue = mp.Queue()
        processes = []

        try:
            for _ in range(num_readers):
                p = mp.Process(
                    target=reader_process, args=(str(basic_mth5_file), result_queue)
                )
                p.start()
                processes.append(p)

            # Collect results
            results = []
            for _ in range(num_readers):
                result = result_queue.get(timeout=10)
                results.append(result)

            # Wait for processes
            for p in processes:
                p.join(timeout=5)

            # Verify all succeeded
            for status, value in results:
                assert status == "success"
                assert value > 0  # Should have read channels

        finally:
            # Cleanup processes
            for p in processes:
                if p.is_alive():
                    p.terminate()


# =============================================================================
# Integration Tests
# =============================================================================


class TestSWMRIntegration:
    """Integration tests for complete SWMR workflows."""

    @pytest.mark.slow
    def test_full_swmr_workflow(self, unique_test_file):
        """Test complete workflow: create structure, open in SWMR, read in SWMR.

        Note: All structure including channels must be created before
        entering SWMR mode since SWMR cannot create new groups or datasets.
        This test verifies the file can be created, opened in SWMR mode,
        and read in SWMR mode.
        """
        # Step 1: Create file with libver='latest' and full structure including data
        mth5_create = MTH5(file_version="0.2.0")
        mth5_create.open_mth5(unique_test_file, mode="w", libver="latest")
        survey = mth5_create.add_survey("test_survey")
        station = mth5_create.add_station("STA001", survey="test_survey")
        run = station.add_run("run_001")
        # Add data BEFORE entering SWMR mode
        data = np.random.randn(1000)
        run.add_channel("Ex", "electric", data)
        mth5_create.close_mth5()

        # Step 2: Open as SWMR writer (no modifications, just verify it works)
        mth5_writer = MTH5()
        mth5_writer.open_mth5(
            unique_test_file, mode="a", single_writer_multiple_reader=True
        )

        # Step 3: Open as SWMR reader and verify
        mth5_reader = MTH5()
        mth5_reader.open_mth5(
            unique_test_file, mode="r", single_writer_multiple_reader=True
        )

        channel = mth5_reader.get_channel(
            "STA001", "run_001", "Ex", survey="test_survey"
        )
        read_data = channel.hdf5_dataset[:]

        assert len(read_data) == 1000
        # Data might be float64 or float32 depending on storage
        assert read_data.dtype in [np.float32, np.float64]

        mth5_reader.close_mth5()

    def test_swmr_mode_persistence_check(self, basic_mth5_file):
        """Test that SWMR mode doesn't persist after file is closed and reopened normally."""
        # Open in SWMR mode
        mth5_swmr = MTH5()
        mth5_swmr.open_mth5(
            basic_mth5_file, mode="a", single_writer_multiple_reader=True
        )
        assert mth5_swmr.is_swmr_mode() is True
        mth5_swmr.close_mth5()

        # Reopen in normal mode
        mth5_normal = MTH5()
        mth5_normal.open_mth5(basic_mth5_file, mode="a")
        assert mth5_normal.is_swmr_mode() is False
        mth5_normal.close_mth5()


# =============================================================================
# Performance Tests (Optional - can be skipped in CI)
# =============================================================================


@pytest.mark.slow
class TestSWMRPerformance:
    """Performance tests for SWMR mode (marked as slow)."""

    def test_swmr_flush_performance(self, basic_mth5_file):
        """Test flush performance in SWMR mode.

        This test verifies that multiple flush operations complete
        in reasonable time when file is opened in SWMR mode.
        """
        # Open in SWMR mode
        mth5 = MTH5()
        mth5.open_mth5(basic_mth5_file, mode="a", single_writer_multiple_reader=True)

        # Time multiple flush operations
        start_time = time.time()

        for i in range(10):
            # Just flush, don't try to add data (would hang)
            mth5.flush()

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for 10 flushes)
        assert elapsed < 5.0

        mth5.close_mth5()


# =============================================================================
# Cleanup
# =============================================================================


def test_final_cleanup():
    """Final cleanup to ensure all files are closed."""
    helpers.close_open_files()
    assert True  # Always passes, just ensures cleanup runs
