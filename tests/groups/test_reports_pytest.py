# -*- coding: utf-8 -*-
"""
Pytest test suite for the ReportsGroup class.

This test suite validates the functionality of the ReportsGroup class, including
adding, retrieving, listing, and removing reports and images. It uses pytest fixtures,
parameterization, and mocking where necessary. The tests are designed to be safe for
parallel execution.
"""

from unittest.mock import ANY, MagicMock, patch

import h5py
import pytest
from PIL import Image

from mth5.groups.reports import ReportsGroup


@pytest.fixture
def mock_hdf5_group():
    """Fixture to create a mock HDF5 group."""
    mock_group = MagicMock(spec=h5py.Group)
    yield mock_group


@pytest.fixture
def reports_group(mock_hdf5_group):
    """Fixture to create a ReportsGroup instance."""
    return ReportsGroup(mock_hdf5_group)


@pytest.fixture
def temp_file(tmp_path):
    """Fixture to create a temporary file for testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
    return file_path


@pytest.mark.parametrize("file_extension", ["pdf", "txt", "md", "png", "jpg"])
def test_add_report(reports_group, temp_file, file_extension):
    """Test adding a report or image to the ReportsGroup."""
    temp_file = temp_file.with_suffix(f".{file_extension}")
    if file_extension in ["png", "jpg"]:
        # Create a valid image file
        img = Image.new("RGB", (10, 10), color="red")
        img.save(temp_file)
    else:
        temp_file.write_text("Test content")

    report_name = "test_report"
    report_metadata = {"author": "test_user"}

    reports_group.add_report(report_name, report_metadata, temp_file)

    # Verify the dataset was created with the correct name and data
    reports_group.hdf5_group.create_dataset.assert_called_once_with(
        report_name, data=ANY
    )

    # Verify metadata was added
    dataset = reports_group.hdf5_group.create_dataset.return_value
    for key, value in report_metadata.items():
        dataset.attrs.__setitem__.assert_any_call(key, value)


def test_get_report(reports_group, mock_hdf5_group):
    """Test retrieving a report from the ReportsGroup."""
    report_name = "test_report"
    mock_dataset = MagicMock()
    mock_dataset.attrs = {
        "file_type": "txt",
        "filename": "test_report.txt",
    }
    mock_dataset.__getitem__.return_value = bytearray(b"Test content")
    mock_hdf5_group.__getitem__.return_value = mock_dataset

    with patch("pathlib.Path.write_bytes", return_value=None) as mock_write_bytes:
        result = reports_group.get_report(report_name)

        # Verify the file was written to disk
        mock_write_bytes.assert_called_once_with(b"Test content")

        # Verify the correct path was returned
        assert result.name == "test_report.txt"


def test_list_reports(reports_group, mock_hdf5_group):
    """Test listing all reports in the ReportsGroup."""
    mock_hdf5_group.keys.return_value = ["report1", "report2", "report3"]

    result = reports_group.list_reports()

    # Verify the correct list of reports was returned
    assert result == ["report1", "report2", "report3"]


def test_remove_report(reports_group, mock_hdf5_group):
    """Test removing a report from the ReportsGroup."""
    report_name = "test_report"
    mock_hdf5_group.__contains__.return_value = True

    with patch.object(reports_group.logger, "info") as mock_logger:
        reports_group.remove_report(report_name)

        # Verify the report was removed
        mock_hdf5_group.__delitem__.assert_called_once_with(report_name)

        # Verify the logger was called
        mock_logger.assert_called_once_with(
            f"Removed report '{report_name}' from the group."
        )


def test_remove_nonexistent_report(reports_group, mock_hdf5_group):
    """Test removing a nonexistent report from the ReportsGroup."""
    report_name = "nonexistent_report"
    mock_hdf5_group.__contains__.return_value = False

    with patch.object(reports_group.logger, "warning") as mock_logger:
        reports_group.remove_report(report_name)

        # Verify the logger was called with a warning
        mock_logger.assert_called_once_with(
            f"Report '{report_name}' not found in the group."
        )
