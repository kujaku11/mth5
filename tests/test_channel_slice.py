# -*- coding: utf-8 -*-
"""
Pytest suite for MTH5 channel slicing functionality.

Converted from unittest to pytest with:
- Session-scoped fixtures for test data
- Worker-safe file paths for pytest-xdist compatibility
- pytest subtests for grouped assertions
- Proper cleanup and error handling

@author: pytest conversion
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from mt_metadata.timeseries import Electric

from mth5.mth5 import MTH5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def channel_slice_mth5(
    make_worker_safe_path,
) -> Generator[tuple[MTH5, object], None, None]:
    """
    Create MTH5 file with channel data for slice testing.

    Uses worker-safe paths to avoid conflicts in parallel testing.
    Returns MTH5 object and channel dataset for testing.
    """
    # Create worker-safe file path
    fn = make_worker_safe_path("test_channel_slice.h5", Path(__file__).parent)

    # Create MTH5 file with test data
    m = MTH5(file_version="0.1.0")
    m.open_mth5(fn, "w")

    # Add station, run, and channel
    station_group = m.add_station("mt01")
    run_group = station_group.add_run("a")
    channel_metadata = Electric(component="ex", sample_rate=1)
    n_samples = 4096
    ch_ds = run_group.add_channel(
        "ex",
        "electric",
        np.arange(n_samples),
        channel_metadata=channel_metadata,
    )

    yield m, ch_ds

    # Cleanup
    m.close_mth5()
    if fn.exists():
        fn.unlink()


@pytest.fixture(scope="session")
def ch_ds(channel_slice_mth5):
    """Extract channel dataset from MTH5 fixture."""
    _, ch_ds = channel_slice_mth5
    return ch_ds


@pytest.fixture(scope="session")
def n_samples():
    """Number of samples in test channel."""
    return 4096


# =============================================================================
# Test Classes
# =============================================================================


class TestMTH5ChannelSlice:
    """Test channel slicing functionality with various time and sample parameters."""

    def test_get_index_from_time_start(self, ch_ds):
        """Test getting index from start time."""
        assert ch_ds.get_index_from_time(ch_ds.start) == 0

    def test_get_index_from_time_start_too_early(self, ch_ds):
        """Test getting index from time before channel start."""
        assert ch_ds.get_index_from_time(ch_ds.start - 5) == -5

    def test_get_index_from_time_end(self, ch_ds, n_samples):
        """Test getting index from end time."""
        assert ch_ds.get_index_from_end_time(ch_ds.end) == n_samples

    def test_get_slice_index_values_times(self, ch_ds, n_samples, subtests):
        """Test getting slice index values from start and end times."""
        start_index, end_index, npts = ch_ds._get_slice_index_values(
            ch_ds.start, end=ch_ds.end
        )

        with subtests.test("npts"):
            assert npts == n_samples

        with subtests.test("start"):
            assert start_index == 0

        with subtests.test("end"):
            assert end_index == n_samples

    def test_get_slice_index_values_n_samples(self, ch_ds, n_samples, subtests):
        """Test getting slice index values from start time and n_samples."""
        start_index, end_index, npts = ch_ds._get_slice_index_values(
            ch_ds.start, n_samples=n_samples
        )

        with subtests.test("npts"):
            assert npts == n_samples

        with subtests.test("start"):
            assert start_index == 0

        with subtests.test("end"):
            assert end_index == n_samples

    def test_full_slice_from_time(self, ch_ds, n_samples, subtests):
        """Test full slice using start and end times."""
        data = ch_ds.time_slice(ch_ds.start, end=ch_ds.end)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == ch_ds.end

    def test_full_slice_overtime(self, ch_ds, n_samples, subtests):
        """Test full slice with end time beyond channel end."""
        data = ch_ds.time_slice(ch_ds.start, end=ch_ds.end + 5)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == ch_ds.end

    def test_full_slice_from_points(self, ch_ds, n_samples, subtests):
        """Test full slice using start time and n_samples."""
        data = ch_ds.time_slice(ch_ds.start, n_samples=n_samples)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == ch_ds.end

    def test_full_slice_from_too_many_points(self, ch_ds, n_samples, subtests):
        """Test slice with n_samples exceeding available data."""
        data = ch_ds.time_slice(ch_ds.start, n_samples=5096)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == ch_ds.end

    def test_full_slice_too_early(self, ch_ds, n_samples, subtests):
        """Test slice starting before channel start."""
        data = ch_ds.time_slice(ch_ds.start - 5, n_samples=5096)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == ch_ds.end

    def test_small_slice_from_end_time(self, ch_ds, subtests):
        """Test small slice using start and specific end time."""
        end = "1980-01-01T00:00:59+00:00"
        data = ch_ds.time_slice(ch_ds.start, end=end)

        with subtests.test("size"):
            assert data.data_array.size == 60

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == end

    def test_small_slice_from_n_samples(self, ch_ds, subtests):
        """Test small slice using start time and n_samples."""
        n_samples = 60
        data = ch_ds.time_slice(ch_ds.start, n_samples=n_samples)

        with subtests.test("size"):
            assert data.data_array.size == n_samples

        with subtests.test("start"):
            assert data.start == ch_ds.start

        with subtests.test("end"):
            assert data.end == "1980-01-01T00:00:59+00:00"


# =============================================================================
# Run pytest
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
