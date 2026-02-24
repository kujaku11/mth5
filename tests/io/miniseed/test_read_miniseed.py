# -*- coding: utf-8 -*-
"""
Pytest test suite for the `read_miniseed` function.

This test suite validates the functionality of the `read_miniseed` function, ensuring it correctly reads MiniSEED files and converts them into `RunTS` objects. The tests use fixtures to provide test data and ensure safe parallel execution.

Requirements:
- mth5_test_data package must be installed with MiniSEED data.
- Tests will be skipped if MiniSEED test data files are not available.

"""

from pathlib import Path

import pytest
from mth5_test_data import get_test_data_path

from mth5.io.miniseed.miniseed import read_miniseed
from mth5.timeseries import RunTS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def session_miniseed_test_data():
    """Session-scoped fixture providing paths to MiniSEED test data files."""
    miniseed_path = get_test_data_path("miniseed")
    return {
        "path": miniseed_path,
        "streams": miniseed_path / "cas_04_streams.mseed",
    }


@pytest.fixture
def miniseed_test_data():
    """Provide paths to MiniSEED test data files."""
    miniseed_path = get_test_data_path("miniseed")
    return {
        "path": miniseed_path,
        "streams": miniseed_path / "cas_04_streams.mseed",
    }


# =============================================================================
# Tests
# =============================================================================


def test_read_miniseed_valid_file(miniseed_test_data):
    """Test reading a valid MiniSEED file."""
    miniseed_file = miniseed_test_data["streams"]

    # Ensure the file exists
    assert miniseed_file.exists(), f"Test file {miniseed_file} does not exist."

    # Read the MiniSEED file
    run_ts = read_miniseed(miniseed_file)

    # Validate the returned object
    assert isinstance(run_ts, RunTS), "The returned object is not of type RunTS."
    assert len(run_ts.channels) > 0, "The RunTS object has no channels."
    # Validate start and end times
    assert run_ts.start is not None, "The RunTS object has no start time."
    assert run_ts.end is not None, "The RunTS object has no end time."
    assert run_ts.end > run_ts.start, "End time is not after start time."
    assert run_ts.sample_rate > 0, "Sample rate is not positive."
    assert (
        run_ts.start == "2020-06-02T19:00:00+00:00"
    ), "RunTS start time does not match channel start time."
    assert (
        run_ts.end == "2020-07-13T19:00:00+00:00"
    ), "RunTS end time does not match channel end time."
    assert (
        run_ts.sample_rate == 1.0
    ), "RunTS sample rate does not match channel sample rate."
    assert (
        run_ts.run_metadata.id == "sr1_001"
    ), "RunTS metadata ID does not match expected value."


def test_read_miniseed_invalid_file(tmp_path):
    """Test reading an invalid MiniSEED file."""
    invalid_file = tmp_path / "invalid.mseed"
    invalid_file.write_text("This is not a valid MiniSEED file.")

    # Attempt to read the invalid file
    with pytest.raises(Exception, match=".*Unknown format for file.*"):
        read_miniseed(invalid_file)


def test_read_miniseed_missing_file():
    """Test reading a missing MiniSEED file."""
    missing_file = Path("nonexistent_file.mseed")

    # Attempt to read the missing file
    with pytest.raises(FileNotFoundError, match=".*No such file or directory.*"):
        read_miniseed(missing_file)
