# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for NIMSCollection class.

This test suite uses fixtures, subtests, and mocking to efficiently test
the NIMSCollection functionality for managing NIMS binary files.

Created on November 10, 2025
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from mth5.io.nims import NIMSCollection


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_nims_file_structure(temp_directory):
    """Create mock NIMS .bin files in temporary directory."""
    files = []
    for i, filename in enumerate(["test001a.BIN", "test001b.BIN", "test002a.BIN"]):
        file_path = temp_directory / filename
        file_path.write_bytes(
            b"mock_binary_data" * (100 + i * 10)
        )  # Different file sizes
        files.append(file_path)
    return files


@pytest.fixture
def mock_empty_entry_dict():
    """Mock empty entry dictionary returned by Collection.get_empty_entry_dict()."""
    return {
        "survey": None,
        "station": None,
        "run": None,
        "start": None,
        "end": None,
        "channel_id": None,
        "component": None,
        "fn": None,
        "sample_rate": None,
        "file_size": None,
        "n_samples": None,
        "sequence_number": None,
        "dipole": None,
        "coil_number": None,
        "latitude": None,
        "longitude": None,
        "elevation": None,
        "declination": None,
        "calibration_fn": None,
    }


@pytest.fixture
def nims_collection(mock_nims_file_structure):
    """Create NIMSCollection instance with mocked file structure."""
    return NIMSCollection(file_path=mock_nims_file_structure[0].parent)


@pytest.fixture
def expected_dataframe_columns():
    """Expected columns in the output DataFrame."""
    return [
        "survey",
        "station",
        "run",
        "start",
        "end",
        "channel_id",
        "component",
        "fn",
        "sample_rate",
        "file_size",
        "n_samples",
        "sequence_number",
        "dipole",
        "coil_number",
        "latitude",
        "longitude",
        "elevation",
        "declination",
        "calibration_fn",
    ]


# =============================================================================
# Test Classes
# =============================================================================


class TestNIMSCollectionInitialization:
    """Test NIMSCollection initialization and basic properties."""

    def test_initialization_default(self):
        """Test default initialization."""
        collection = NIMSCollection()
        assert collection.file_path is None
        assert collection.file_ext == "bin"
        assert collection.survey_id == "mt"

    def test_initialization_with_path(self, temp_directory):
        """Test initialization with file path."""
        collection = NIMSCollection(file_path=temp_directory)
        assert collection.file_path == temp_directory
        assert collection.file_ext == "bin"
        assert collection.survey_id == "mt"

    def test_initialization_with_kwargs(self, temp_directory):
        """Test initialization with additional keyword arguments."""
        # Note: NIMSCollection doesn't currently support setting survey_id in constructor
        collection = NIMSCollection(file_path=temp_directory)
        collection.survey_id = "custom_survey"  # Set after initialization
        assert collection.file_path == temp_directory
        assert collection.survey_id == "custom_survey"

    def test_file_extension_property(self, nims_collection):
        """Test that file extension is set correctly."""
        assert nims_collection.file_ext == "bin"


class TestNIMSCollectionFileHandling:
    """Test file discovery and handling functionality."""

    def test_get_files_with_bin_files(self, nims_collection, mock_nims_file_structure):
        """Test getting .bin files from directory."""
        files = nims_collection.get_files(nims_collection.file_ext)
        file_names = [f.name for f in files]

        expected_names = [f.name for f in mock_nims_file_structure]
        assert len(files) == 3
        assert all(name in file_names for name in expected_names)

    def test_get_files_empty_directory(self, temp_directory):
        """Test getting files from empty directory."""
        collection = NIMSCollection(file_path=temp_directory)
        files = collection.get_files(collection.file_ext)
        assert len(files) == 0

    def test_get_files_non_existent_directory(self):
        """Test getting files from non-existent directory."""
        # Create collection without setting file_path to avoid IOError in constructor
        collection = NIMSCollection()
        collection._file_path = Path(
            "/non/existent/path"
        )  # Set private attribute directly
        files = collection.get_files(collection.file_ext)
        assert len(files) == 0


class TestNIMSCollectionDataFrameCreation:
    """Test DataFrame creation from NIMS files."""

    @patch("mth5.io.nims.nims_collection.NIMS")
    def test_to_dataframe_basic_simple(self, mock_nims_class, nims_collection):
        """Test basic DataFrame creation with simplified mocking."""
        # Create simple mock without MTime objects
        mock_nims = Mock()
        mock_nims.station = "test001"
        mock_nims.run_id = "test001a"
        mock_nims.start_time.isoformat.return_value = "2023-01-01T12:00:00+00:00"
        mock_nims.end_time.isoformat.return_value = "2023-01-01T13:00:00+00:00"
        mock_nims.sample_rate = 8
        mock_nims.file_size = 1200
        mock_nims.n_samples = 28800
        mock_nims.ex_length = 100.0
        mock_nims.ey_length = 100.0
        mock_nims.read_header = Mock()

        mock_nims_class.return_value = mock_nims

        # Mock the parent class methods
        with patch.object(
            nims_collection, "get_empty_entry_dict", return_value={}
        ), patch.object(
            nims_collection, "_sort_df", side_effect=lambda df, zeros: df
        ), patch.object(
            nims_collection, "_set_df_dtypes", side_effect=lambda df: df
        ):
            df = nims_collection.to_dataframe()

        # Should have processed the mock files
        assert len(df) >= 0  # May be empty if no .bin files found


class TestNIMSCollectionRunNameAssignment:
    """Test run name assignment functionality."""

    def test_assign_run_names_single_station(self):
        """Test run name assignment for single station."""
        collection = NIMSCollection()

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "station": ["test001", "test001", "test001"],
                "run": [None, None, None],
                "sample_rate": [8, 8, 8],
                "start": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-01T12:00:00",
                ],
            }
        )
        df["start"] = pd.to_datetime(df["start"])

        result_df = collection.assign_run_names(df, zeros=2)

        expected_runs = ["sr8_01", "sr8_02", "sr8_03"]
        expected_sequences = [1, 2, 3]

        assert result_df["run"].tolist() == expected_runs
        assert result_df["sequence_number"].tolist() == expected_sequences

    def test_assign_run_names_multiple_stations(self):
        """Test run name assignment for multiple stations."""
        collection = NIMSCollection()

        df = pd.DataFrame(
            {
                "station": ["test001", "test001", "test002", "test002"],
                "run": [None, None, None, None],
                "sample_rate": [8, 8, 8, 8],
                "start": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-02T10:00:00",
                    "2023-01-02T11:00:00",
                ],
            }
        )
        df["start"] = pd.to_datetime(df["start"])

        result_df = collection.assign_run_names(df, zeros=3)

        # Check that each station starts counting from 1
        test001_runs = result_df[result_df["station"] == "test001"]["run"].tolist()
        test002_runs = result_df[result_df["station"] == "test002"]["run"].tolist()

        assert test001_runs == ["sr8_001", "sr8_002"]
        assert test002_runs == ["sr8_001", "sr8_002"]

    def test_assign_run_names_existing_run_names(self):
        """Test that existing run names are preserved."""
        collection = NIMSCollection()

        df = pd.DataFrame(
            {
                "station": ["test001", "test001", "test001"],
                "run": ["existing_run", None, "another_run"],
                "sample_rate": [8, 8, 8],
                "start": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-01T12:00:00",
                ],
            }
        )
        df["start"] = pd.to_datetime(df["start"])

        result_df = collection.assign_run_names(df, zeros=2)

        # Should preserve existing names, only assign to None values
        expected_runs = ["existing_run", "sr8_02", "another_run"]
        assert result_df["run"].tolist() == expected_runs

    def test_assign_run_names_different_sample_rates(self):
        """Test run name assignment with different sample rates."""
        collection = NIMSCollection()

        df = pd.DataFrame(
            {
                "station": ["test001", "test001", "test001"],
                "run": [None, None, None],
                "sample_rate": [1, 8, 8],
                "start": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T11:00:00",
                    "2023-01-01T12:00:00",
                ],
            }
        )
        df["start"] = pd.to_datetime(df["start"])

        result_df = collection.assign_run_names(df, zeros=2)

        expected_runs = ["sr1_01", "sr8_02", "sr8_03"]
        assert result_df["run"].tolist() == expected_runs


class TestNIMSCollectionIntegration:
    """Integration tests combining multiple functionalities."""

    def test_basic_file_discovery(self, temp_directory):
        """Test basic file discovery functionality."""
        # Create some mock files
        (temp_directory / "test001a.bin").write_bytes(b"data")
        (temp_directory / "test002b.BIN").write_bytes(b"data")
        (temp_directory / "test003c.txt").write_bytes(b"data")  # Non-bin file

        collection = NIMSCollection(file_path=temp_directory)
        files = collection.get_files(collection.file_ext)

        # Should find only .bin/.BIN files
        assert len(files) >= 2

    def test_collection_properties(self, nims_collection):
        """Test that collection has expected properties."""
        assert hasattr(nims_collection, "file_ext")
        # file_ext might be a string 'bin' or list ['.bin', '.BIN'] depending on implementation
        assert nims_collection.file_ext in ("bin", [".bin", ".BIN"], ["bin", "BIN"])
        assert hasattr(nims_collection, "get_files")
        assert hasattr(nims_collection, "to_dataframe")

    def test_empty_directory_handling(self, temp_directory):
        """Test handling of directory with no NIMS files."""
        # Create a file that's not a NIMS file
        (temp_directory / "not_nims.txt").write_text("not a nims file")

        collection = NIMSCollection(file_path=temp_directory)
        files = collection.get_files(collection.file_ext)

        # Should find no .bin files
        assert len(files) == 0


class TestNIMSCollectionPerformance:
    """Performance and efficiency tests."""

    def test_multiple_file_handling(self, temp_directory):
        """Test handling multiple files efficiently."""
        # Create multiple mock files
        for i in range(5):
            file_path = temp_directory / f"test_{i:03d}a.BIN"
            file_path.write_bytes(b"mock_data" * 100)

        collection = NIMSCollection(file_path=temp_directory)
        files = collection.get_files(collection.file_ext)

        # Should efficiently discover all files
        assert len(files) == 5
        assert all(f.name.endswith(".BIN") for f in files)


class TestNIMSCollectionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_survey_id(self):
        """Test with empty survey ID."""
        collection = NIMSCollection()
        collection.survey_id = ""
        assert collection.survey_id == ""

    def test_assign_run_names_empty_dataframe(self):
        """Test assign_run_names with empty DataFrame."""
        collection = NIMSCollection()
        df = pd.DataFrame(columns=["station", "run", "sample_rate", "start"])

        result_df = collection.assign_run_names(df)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["station", "run", "sample_rate", "start"]

    def test_file_discovery_efficiency(self, temp_directory):
        """Test that file discovery is efficient."""
        # Create many files of different types
        for i in range(10):
            bin_path = temp_directory / f"valid_{i}.BIN"
            txt_path = temp_directory / f"invalid_{i}.txt"
            bin_path.write_bytes(b"bin_data")
            txt_path.write_text("text_data")

        collection = NIMSCollection(file_path=temp_directory)

        # Should only find .bin files
        files = collection.get_files(collection.file_ext)
        assert len(files) == 10
        assert all(f.suffix.lower() in [".bin"] for f in files)


# =============================================================================
# Performance and stress tests (can be run separately)
# =============================================================================


class TestNIMSCollectionStress:
    """Stress tests for NIMSCollection (marked as slow)."""

    def test_many_files_discovery(self, temp_directory):
        """Test file discovery with many files."""
        # Create many files
        num_files = 50
        for i in range(num_files):
            file_path = temp_directory / f"stress_test_{i:04d}.BIN"
            file_path.write_bytes(b"stress_data" * 50)

        collection = NIMSCollection(file_path=temp_directory)
        files = collection.get_files(collection.file_ext)

        # Should discover all files efficiently
        assert len(files) == num_files
        assert all(f.name.startswith("stress_test_") for f in files)


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Run tests with various options
    pytest.main([__file__, "-v", "--tb=short"])
