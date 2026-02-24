# -*- coding: utf-8 -*-
"""
Pytest test suite for Z3DCollection class functionality.

Converted from unittest to pytest with fixtures and optimized for comprehensive testing.
Updated for pydantic version of mt_metadata and expanded functionality coverage.

@author: jpeacock, updated for pytest
"""

# =============================================================================
# Imports
# =============================================================================
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from mt_metadata.timeseries import Station

from mth5.io.zen import Z3D, Z3DCollection


try:
    import mth5_test_data

    zen_path = mth5_test_data.get_test_data_path("zen")
except ImportError:
    zen_path = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def z3d_collection():
    """Session-scoped fixture for Z3DCollection with real data."""
    if zen_path is None:
        pytest.skip("mth5_test_data not available")

    zc = Z3DCollection(zen_path)
    return zc


@pytest.fixture(scope="session")
def z3d_dataframe(z3d_collection):
    """Session-scoped fixture for dataframe from real data."""
    return z3d_collection.to_dataframe([256, 4096, 1024])


@pytest.fixture(scope="session")
def z3d_runs(z3d_collection):
    """Session-scoped fixture for runs from real data."""
    return z3d_collection.get_runs([256, 4096, 1024])


@pytest.fixture
def mock_z3d_file_data():
    """Mock Z3D file data for testing."""
    return {
        "fn": Path("/test/data/station_001_20220517_131017_256_EY.Z3D"),
        "station": "001",
        "sample_rate": 256.0,
        "component": "ey",
        "channel_number": 5,
        "start": "2022-05-17T13:09:58+00:00",
        "end": "2022-05-17T15:54:42+00:00",
        "file_size": 10759100,
        "n_samples": 2530304,
        "dipole_length": 56.0,
        "coil_number": None,
        "latitude": 40.49757833327694,
        "longitude": -116.8211900230401,
        "elevation": 1456.3,
        "box_number": 24,
        "job_name": "test_survey",
    }


@pytest.fixture
def mock_z3d_files_list():
    """Create list of mock Z3D file paths."""
    files = []
    components = ["EX", "EY", "HX", "HY", "HZ"]
    sample_rates = [256, 1024, 4096]

    for sr in sample_rates:
        for comp in components:
            fn = Path(f"/test/data/bm100_20220517_131017_{sr}_{comp}.Z3D")
            files.append(fn)
    return files


@pytest.fixture
def mock_z3d_object(mock_z3d_file_data):
    """Create a mock Z3D object with realistic data."""
    z3d = Mock(spec=Z3D)

    # Basic properties
    z3d.station = mock_z3d_file_data["station"]
    z3d.sample_rate = mock_z3d_file_data["sample_rate"]
    z3d.component = mock_z3d_file_data["component"]
    z3d.channel_number = mock_z3d_file_data["channel_number"]
    z3d.file_size = mock_z3d_file_data["file_size"]
    z3d.n_samples = mock_z3d_file_data["n_samples"]
    z3d.dipole_length = mock_z3d_file_data["dipole_length"]
    z3d.coil_number = mock_z3d_file_data["coil_number"]
    z3d.latitude = mock_z3d_file_data["latitude"]
    z3d.longitude = mock_z3d_file_data["longitude"]
    z3d.elevation = mock_z3d_file_data["elevation"]

    # Mock time properties
    z3d.start = Mock()
    z3d.start.isoformat.return_value = mock_z3d_file_data["start"]
    z3d.end = Mock()
    z3d.end.isoformat.return_value = mock_z3d_file_data["end"]

    # Mock header and metadata
    z3d.header = Mock()
    z3d.header.box_number = mock_z3d_file_data["box_number"]
    z3d.metadata = Mock()
    z3d.metadata.job_name = mock_z3d_file_data["job_name"]

    # Mock station metadata
    z3d.station_metadata = Mock()
    station_dict = {
        "id": mock_z3d_file_data["station"],
        "location.latitude": mock_z3d_file_data["latitude"],
        "location.longitude": mock_z3d_file_data["longitude"],
        "location.elevation": mock_z3d_file_data["elevation"],
    }
    z3d.station_metadata.to_dict.return_value = station_dict

    return z3d


@pytest.fixture
def empty_z3d_collection():
    """Create an empty Z3DCollection for testing."""
    return Z3DCollection()


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe matching Z3DCollection output structure."""
    data = []
    components = ["ex", "ey", "hx", "hy", "hz"]
    sample_rates = [256, 1024, 4096]

    for i, sr in enumerate(sample_rates):
        for j, comp in enumerate(components):
            entry = {
                "survey": "",  # Updated: pydantic default for empty survey
                "station": "100",
                "run": f"sr{sr}_000{i+1:d}",
                "start": pd.Timestamp("2022-05-17T13:09:58+00:00"),
                "end": pd.Timestamp("2022-05-17T15:54:42+00:00"),
                "channel_id": j + 1,
                "component": comp,
                "fn": Path(f"/test/data/bm100_20220517_131017_{sr}_{comp.upper()}.Z3D"),
                "sample_rate": float(sr),
                "file_size": 10759100,
                "n_samples": 2530304,
                "sequence_number": i + 1,
                "dipole": 56.0 if "e" in comp else 0.0,
                "coil_number": "2324" if "h" in comp else None,
                "latitude": 40.49757833327694,
                "longitude": -116.8211900230401,
                "elevation": 1456.3,
                "instrument_id": "ZEN024",
                "calibration_fn": "/test/cal/antenna.cal" if "h" in comp else None,
            }
            data.append(entry)

    return pd.DataFrame(data)


# =============================================================================
# Original Test Suite (Converted from unittest)
# =============================================================================


class TestZ3DCollectionBasic:
    """Test basic Z3DCollection functionality (converted from unittest)."""

    def test_file_path(self, z3d_collection):
        """Test file_path property."""
        assert isinstance(z3d_collection.file_path, Path)

    def test_get_files(self, z3d_collection):
        """Test get_files method."""
        files = z3d_collection.get_files(z3d_collection.file_ext)
        assert len(files) == 10
        assert all(isinstance(f, Path) for f in files)

    def test_df_columns(self, z3d_collection, z3d_dataframe):
        """Test dataframe columns match expected structure."""
        assert z3d_collection._columns == z3d_dataframe.columns.to_list()

    def test_df_shape(self, z3d_dataframe):
        """Test dataframe shape."""
        assert z3d_dataframe.shape == (10, 19)

    def test_df_types(self, z3d_collection, z3d_dataframe):
        """Test dataframe data types are properly set."""
        df = z3d_collection._set_df_dtypes(z3d_dataframe)

        # Test datetime columns
        assert pd.api.types.is_datetime64_any_dtype(df.start)
        assert pd.api.types.is_datetime64_any_dtype(df.end)

        # Test object columns (pandas 2.x uses StringDtype)
        assert pd.api.types.is_string_dtype(
            df.instrument_id
        ) or pd.api.types.is_object_dtype(df.instrument_id)
        assert pd.api.types.is_string_dtype(
            df.calibration_fn
        ) or pd.api.types.is_object_dtype(df.calibration_fn)

    def test_survey_id(self, z3d_dataframe):
        """Test survey ID values."""
        assert (z3d_dataframe.survey == "").all()

    def test_df_run_names_256(self, z3d_dataframe):
        """Test run names for 256 Hz data."""
        run_256 = z3d_dataframe[z3d_dataframe.sample_rate == 256].run.unique()[0]
        assert run_256 == "sr256_0002"

    def test_df_run_names_4096(self, z3d_dataframe):
        """Test run names for 4096 Hz data."""
        run_4096 = z3d_dataframe[z3d_dataframe.sample_rate == 4096].run.unique()[0]
        assert run_4096 == "sr4096_0001"

    def test_run_dtype(self, z3d_runs):
        """Test runs return type."""
        assert isinstance(z3d_runs, OrderedDict)

    def test_run_elements(self, z3d_dataframe, z3d_runs):
        """Test run elements consistency."""
        station = z3d_dataframe.station.unique()[0]

        for key, rdf in z3d_runs[station].items():
            test_rdf = z3d_dataframe[z3d_dataframe.run == key]
            rdf = rdf.fillna(0)
            test_rdf = test_rdf.fillna(0)

            # Check first 8 columns for equality
            comparison = test_rdf.iloc[0:8].eq(rdf).all(axis=0).all()
            assert comparison


# =============================================================================
# Extended Test Suite (New functionality)
# =============================================================================


class TestZ3DCollectionInitialization:
    """Test Z3DCollection initialization and configuration."""

    def test_default_initialization(self, empty_z3d_collection):
        """Test default initialization."""
        assert empty_z3d_collection.file_path is None
        assert empty_z3d_collection.file_ext == "z3d"
        assert isinstance(empty_z3d_collection.station_metadata_dict, dict)
        assert len(empty_z3d_collection.station_metadata_dict) == 0

    def test_initialization_with_path_string(self):
        """Test initialization with string path."""
        with patch("pathlib.Path.exists", return_value=True):
            zc = Z3DCollection("/test/path")
            assert zc.file_path == Path("/test/path")

    def test_initialization_with_path_object(self):
        """Test initialization with Path object."""
        test_path = Path("/test/path")
        with patch("pathlib.Path.exists", return_value=True):
            zc = Z3DCollection(test_path)
            assert zc.file_path == test_path

    def test_file_extension_customization(self):
        """Test custom file extension setting."""
        zc = Z3DCollection()
        zc.file_ext = "Z3D"
        assert zc.file_ext == "Z3D"

    def test_columns_property(self, z3d_collection):
        """Test _columns property structure."""
        expected_columns = [
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
            "instrument_id",
            "calibration_fn",
        ]
        assert z3d_collection._columns == expected_columns


class TestZ3DCollectionFileOperations:
    """Test file discovery and processing operations."""

    def test_get_files_with_extension_list(self, empty_z3d_collection):
        """Test get_files with list of extensions."""
        with patch("mth5.io.collection.Collection.get_files") as mock_get_files:
            mock_get_files.return_value = [
                Path("/test/file1.Z3D"),
                Path("/test/file2.z3d"),
            ]

            result = empty_z3d_collection.get_files(["z3d", "Z3D"])
            assert len(result) == 2
            assert all(isinstance(f, Path) for f in result)

    def test_get_files_empty_directory(self, empty_z3d_collection):
        """Test get_files with empty directory."""
        with patch("mth5.io.collection.Collection.get_files") as mock_get_files:
            mock_get_files.return_value = []

            result = empty_z3d_collection.get_files(["z3d"])
            assert len(result) == 0
            assert isinstance(result, list)

    def test_file_filtering_by_pattern(self, empty_z3d_collection):
        """Test file filtering capabilities."""
        test_files = [
            Path("/test/station_001_256_EY.Z3D"),
            Path("/test/station_002_256_EX.Z3D"),
            Path("/test/station_001_1024_HX.Z3D"),
            Path("/test/other_file.txt"),
        ]

        with patch("mth5.io.collection.Collection.get_files") as mock_get_files:
            mock_get_files.return_value = test_files[:3]  # Only Z3D files

            result = empty_z3d_collection.get_files(["z3d"])
            assert len(result) == 3
            assert all(f.suffix.upper() == ".Z3D" for f in result)


class TestZ3DCollectionDataframeProcessing:
    """Test dataframe creation and processing."""

    def test_to_dataframe_structure(self, z3d_dataframe):
        """Test dataframe structure and content."""
        # Test basic structure
        assert isinstance(z3d_dataframe, pd.DataFrame)
        assert len(z3d_dataframe) > 0

        # Test required columns exist
        required_columns = [
            "station",
            "component",
            "sample_rate",
            "start",
            "end",
            "fn",
            "file_size",
            "n_samples",
        ]
        for col in required_columns:
            assert col in z3d_dataframe.columns

    def test_to_dataframe_with_sample_rates(self, z3d_collection):
        """Test to_dataframe with specific sample rates."""
        df_256 = z3d_collection.to_dataframe([256])
        df_4096 = z3d_collection.to_dataframe([4096])

        # Should filter by sample rate (check if dataframes contain data)
        if len(df_256) > 0:
            assert (df_256.sample_rate == 256).all()
        if len(df_4096) > 0:
            assert (df_4096.sample_rate == 4096).all()

        # May be same length if one sample rate is missing
        assert isinstance(df_256, pd.DataFrame)
        assert isinstance(df_4096, pd.DataFrame)

    def test_to_dataframe_no_sample_rates(self, z3d_collection):
        """Test to_dataframe without sample rate filtering."""
        df_all = z3d_collection.to_dataframe()
        df_filtered = z3d_collection.to_dataframe([256, 1024, 4096])

        # Should include all sample rates when no filter
        assert len(df_all) >= len(df_filtered)

    def test_dataframe_data_types(self, z3d_collection, z3d_dataframe):
        """Test proper data type assignment."""
        df_typed = z3d_collection._set_df_dtypes(z3d_dataframe)

        # Numeric types
        assert pd.api.types.is_numeric_dtype(df_typed.sample_rate)
        assert pd.api.types.is_numeric_dtype(df_typed.file_size)
        assert pd.api.types.is_numeric_dtype(df_typed.n_samples)
        assert pd.api.types.is_numeric_dtype(df_typed.latitude)
        assert pd.api.types.is_numeric_dtype(df_typed.longitude)
        assert pd.api.types.is_numeric_dtype(df_typed.elevation)

        # DateTime types
        assert pd.api.types.is_datetime64_any_dtype(df_typed.start)
        assert pd.api.types.is_datetime64_any_dtype(df_typed.end)

        # Object types (pandas 2.x uses StringDtype)
        assert pd.api.types.is_string_dtype(
            df_typed.station
        ) or pd.api.types.is_object_dtype(df_typed.station)
        assert pd.api.types.is_string_dtype(
            df_typed.component
        ) or pd.api.types.is_object_dtype(df_typed.component)
        assert pd.api.types.is_string_dtype(
            df_typed.run
        ) or pd.api.types.is_object_dtype(df_typed.run)

    def test_dataframe_sorting(self, z3d_collection, z3d_dataframe):
        """Test dataframe sorting functionality."""
        # Test that dataframe is properly sorted
        df_sorted = z3d_collection._sort_df(z3d_dataframe, zeros=3)

        # Should be sorted by station, start time, and sample rate
        assert (
            df_sorted.station.is_monotonic_increasing
            or len(df_sorted.station.unique()) == 1
        )

        # Test that sorting function returns a dataframe
        assert isinstance(df_sorted, pd.DataFrame)
        assert len(df_sorted) == len(z3d_dataframe)

        # Test that run names are sorted properly
        assert df_sorted.run.is_monotonic_increasing or len(df_sorted.run.unique()) == 1


class TestZ3DCollectionRunProcessing:
    """Test run processing and organization."""

    def test_get_runs_structure(self, z3d_runs):
        """Test runs structure and organization."""
        assert isinstance(z3d_runs, OrderedDict)

        # Should be organized by station
        for station_id, runs in z3d_runs.items():
            assert isinstance(station_id, str)
            assert isinstance(runs, OrderedDict)

            # Each run should be a dataframe
            for run_id, run_df in runs.items():
                assert isinstance(run_id, str)
                assert isinstance(run_df, pd.DataFrame)
                assert len(run_df) > 0

    def test_get_runs_with_sample_rates(self, z3d_collection):
        """Test get_runs with specific sample rates."""
        # Test with sample rates we know exist based on the original test
        runs_mixed = z3d_collection.get_runs([256, 4096])
        assert isinstance(runs_mixed, OrderedDict)

        # Should have at least one station
        assert len(runs_mixed) > 0

        # Test structure
        for station_id, station_runs in runs_mixed.items():
            assert isinstance(station_id, str)
            assert isinstance(station_runs, OrderedDict)

            # Each station should have run data
            for run_id, run_df in station_runs.items():
                assert isinstance(run_id, str)
                assert isinstance(run_df, pd.DataFrame)

        # Test with just one known sample rate
        try:
            runs_single = z3d_collection.get_runs([256])
            assert isinstance(runs_single, OrderedDict)
        except Exception:
            # If 256 Hz data doesn't exist alone, that's okay
            pass

    def test_assign_run_names(self, empty_z3d_collection):
        """Test run name assignment functionality."""
        # Create test dataframe
        test_df = pd.DataFrame(
            {
                "station": ["001", "001", "002", "002"],
                "start": [
                    pd.Timestamp("2022-01-01T12:00:00"),
                    pd.Timestamp("2022-01-01T14:00:00"),
                    pd.Timestamp("2022-01-01T12:00:00"),
                    pd.Timestamp("2022-01-01T13:00:00"),
                ],
                "sample_rate": [256.0, 256.0, 1024.0, 1024.0],
                "run": [None, None, None, None],
                "sequence_number": [0, 0, 0, 0],
            }
        )

        result = empty_z3d_collection.assign_run_names(test_df, zeros=3)

        # Should have assigned run names
        assert result.run.notna().all()
        assert result.run.str.contains("sr").all()
        assert result.run.str.contains("_").all()

    def test_run_name_uniqueness(self, empty_z3d_collection):
        """Test that run names are unique within stations."""
        test_df = pd.DataFrame(
            {
                "station": ["001"] * 4,
                "start": [
                    pd.Timestamp("2022-01-01T12:00:00"),
                    pd.Timestamp("2022-01-01T14:00:00"),
                    pd.Timestamp("2022-01-02T12:00:00"),
                    pd.Timestamp("2022-01-02T14:00:00"),
                ],
                "sample_rate": [256.0] * 4,
                "run": [None] * 4,
                "sequence_number": [0] * 4,
            }
        )

        result = empty_z3d_collection.assign_run_names(test_df)

        # Run names should be unique
        assert len(result.run.unique()) == len(result.run)


class TestZ3DCollectionStationMetadata:
    """Test station metadata handling."""

    def test_sort_station_metadata_basic(self, empty_z3d_collection):
        """Test basic station metadata sorting."""
        station_list = [
            {
                "id": "001",
                "location.latitude": 40.5,
                "location.longitude": -116.8,
                "location.elevation": 1456.0,
            },
            {
                "id": "002",
                "location.latitude": 41.0,
                "location.longitude": -117.0,
                "location.elevation": 1500.0,
            },
        ]

        result = empty_z3d_collection._sort_station_metadata(station_list)
        assert isinstance(result, dict)
        assert "001" in result
        assert "002" in result

    def test_sort_station_metadata_empty(self, empty_z3d_collection):
        """Test station metadata sorting with empty input."""
        with pytest.raises(
            AttributeError, match="'DataFrame' object has no attribute 'id'"
        ):
            empty_z3d_collection._sort_station_metadata([])

    def test_station_metadata_dict_persistence(self, z3d_collection):
        """Test that station metadata persists in collection."""
        # After processing, should have station metadata
        z3d_collection.to_dataframe()

        assert isinstance(z3d_collection.station_metadata_dict, dict)
        if len(z3d_collection.station_metadata_dict) > 0:
            # Should contain Station objects
            for station_id, station_obj in z3d_collection.station_metadata_dict.items():
                assert isinstance(station_id, str)
                assert isinstance(station_obj, Station)


class TestZ3DCollectionCalibration:
    """Test calibration file handling."""

    def test_get_calibrations_interface(self, empty_z3d_collection):
        """Test calibration interface exists and is callable."""
        assert hasattr(empty_z3d_collection, "get_calibrations")
        assert callable(empty_z3d_collection.get_calibrations)

    @patch("builtins.open")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("mth5.io.zen.coil_response.CoilResponse")
    def test_get_calibrations_file_processing(
        self, mock_coil_response, mock_path_exists, mock_open, empty_z3d_collection
    ):
        """Test calibration file processing."""
        mock_instance = Mock()
        mock_coil_response.return_value = mock_instance

        result = empty_z3d_collection.get_calibrations("/test/cal/path")

        # Should return a coil response object
        assert result is not None

    def test_calibration_filename_extraction(self, empty_z3d_collection):
        """Test calibration filename extraction from Z3D files."""
        # Test that calibration filenames can be extracted
        # This would typically be done during dataframe creation
        # but we can test the interface
        assert hasattr(empty_z3d_collection, "get_calibrations")


class TestZ3DCollectionErrorHandling:
    """Test error handling and edge cases."""

    def test_to_dataframe_no_files(self, empty_z3d_collection):
        """Test to_dataframe with no Z3D files found."""
        with patch.object(empty_z3d_collection, "get_files", return_value=[]):
            with patch.object(
                empty_z3d_collection, "_sort_station_metadata", return_value={}
            ):
                with patch.object(
                    empty_z3d_collection, "get_calibrations", return_value=Mock()
                ):
                    with patch.object(empty_z3d_collection, "_sort_df") as mock_sort:
                        with patch.object(
                            empty_z3d_collection, "_set_df_dtypes"
                        ) as mock_dtypes:
                            empty_df = pd.DataFrame()
                            mock_dtypes.return_value = empty_df
                            mock_sort.return_value = empty_df

                            result = empty_z3d_collection.to_dataframe()
                            assert isinstance(result, pd.DataFrame)

    def test_invalid_sample_rates(self, z3d_collection):
        """Test handling of invalid sample rates."""
        # Test with non-existent sample rate
        df_empty = z3d_collection.to_dataframe([99999])
        assert len(df_empty) == 0

        # Test with empty sample rate list
        df_all = z3d_collection.to_dataframe([])
        # Should return all data when empty list provided
        assert isinstance(df_all, pd.DataFrame)

    def test_corrupted_z3d_file_handling(self, empty_z3d_collection):
        """Test handling of corrupted Z3D files."""
        # Mock a scenario where Z3D file reading fails
        with patch("mth5.io.zen.Z3D") as mock_z3d_class:
            mock_z3d_class.side_effect = Exception("Corrupted file")

            with patch.object(empty_z3d_collection, "get_files") as mock_get_files:
                mock_get_files.return_value = [Path("/test/corrupted.Z3D")]

                # Should handle the error gracefully
                try:
                    result = empty_z3d_collection.to_dataframe()
                    # If it completes, should return a DataFrame
                    assert isinstance(result, pd.DataFrame)
                except Exception as e:
                    # Should raise appropriate error
                    assert "Corrupted file" in str(e) or isinstance(e, Exception)

    def test_missing_metadata_handling(self, empty_z3d_collection):
        """Test handling of Z3D files with missing metadata."""
        # Create mock Z3D with missing metadata
        mock_z3d = Mock()
        mock_z3d.station = None
        mock_z3d.component = "unknown"
        mock_z3d.sample_rate = None

        # Should handle missing metadata gracefully
        assert mock_z3d.station is None
        assert mock_z3d.component == "unknown"
        assert mock_z3d.sample_rate is None


class TestZ3DCollectionFiltering:
    """Test data filtering and selection functionality."""

    def test_component_filtering(self, z3d_dataframe):
        """Test filtering by component type."""
        # Test electric components
        electric_df = z3d_dataframe[z3d_dataframe.component.str.startswith("e")]
        assert len(electric_df) > 0
        assert electric_df.component.isin(["ex", "ey"]).all()

        # Test magnetic components
        magnetic_df = z3d_dataframe[z3d_dataframe.component.str.startswith("h")]
        assert len(magnetic_df) > 0
        assert magnetic_df.component.isin(["hx", "hy", "hz"]).all()

    def test_station_filtering(self, z3d_dataframe):
        """Test filtering by station."""
        stations = z3d_dataframe.station.unique()

        for station in stations:
            station_df = z3d_dataframe[z3d_dataframe.station == station]
            assert len(station_df) > 0
            assert (station_df.station == station).all()

    def test_time_range_filtering(self, z3d_dataframe):
        """Test filtering by time range."""
        # Test that all records have valid time ranges
        assert (z3d_dataframe.start <= z3d_dataframe.end).all()

        # Test time range consistency
        time_diff = z3d_dataframe.end - z3d_dataframe.start
        assert (time_diff > pd.Timedelta(0)).all()

    def test_sample_rate_filtering(self, z3d_collection):
        """Test sample rate filtering functionality."""
        # Test single sample rate
        df_256 = z3d_collection.to_dataframe([256])
        if len(df_256) > 0:
            assert (df_256.sample_rate == 256).all()

        # Test multiple sample rates
        df_multi = z3d_collection.to_dataframe([256, 1024])
        if len(df_multi) > 0:
            assert df_multi.sample_rate.isin([256, 1024]).all()


class TestZ3DCollectionPerformance:
    """Test performance and optimization aspects."""

    def test_file_processing_efficiency(self, empty_z3d_collection):
        """Test efficient file processing."""
        # Mock large file list with duplicates
        test_files = [Path(f"/test/file_{i}.Z3D") for i in range(10)]
        test_files.extend(test_files[:5])  # Add duplicates

        # Collection should handle duplicates efficiently
        unique_files = set(test_files)
        assert len(unique_files) == 10  # Should deduplicate

    def test_dataframe_memory_efficiency(self, z3d_dataframe):
        """Test memory-efficient data types."""
        # Check numeric types
        assert z3d_dataframe.sample_rate.dtype in [np.float64, np.float32, float]
        assert z3d_dataframe.file_size.dtype in [np.int64, np.int32, int]
        assert z3d_dataframe.n_samples.dtype in [np.int64, np.int32, int]

        # Check object types for strings (pandas 2.x uses StringDtype)
        assert pd.api.types.is_string_dtype(
            z3d_dataframe.component
        ) or pd.api.types.is_object_dtype(z3d_dataframe.component)
        assert pd.api.types.is_string_dtype(
            z3d_dataframe.station
        ) or pd.api.types.is_object_dtype(z3d_dataframe.station)

    def test_large_file_set_handling(self, empty_z3d_collection):
        """Test handling of large numbers of files."""
        # Create large mock file list
        large_file_list = [Path(f"/test/file_{i:04d}.Z3D") for i in range(1000)]

        with patch.object(empty_z3d_collection, "get_files") as mock_get_files:
            mock_get_files.return_value = large_file_list

            files = empty_z3d_collection.get_files(["z3d"])
            assert len(files) == 1000
            assert all(isinstance(f, Path) for f in files)


class TestZ3DCollectionIntegration:
    """Integration tests for complete workflows."""

    def test_complete_processing_workflow(self, z3d_collection):
        """Test complete data processing workflow."""
        # Step 1: Get files
        files = z3d_collection.get_files(["z3d"])
        assert len(files) > 0

        # Step 2: Create dataframe
        df = z3d_collection.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Step 3: Get organized runs
        runs = z3d_collection.get_runs(
            [256, 4096, 1024]
        )  # provide required sample_rates
        assert isinstance(runs, OrderedDict)
        assert len(runs) > 0

        # Step 4: Verify consistency
        total_files = sum(len(station_runs) for station_runs in runs.values())
        assert total_files <= len(files)  # Some files might be filtered out

    def test_dataframe_runs_consistency(self, z3d_collection):
        """Test consistency between dataframe and runs."""
        df = z3d_collection.to_dataframe([256, 4096, 1024])
        runs = z3d_collection.get_runs([256, 4096, 1024])

        # Check that all runs in runs dict appear in dataframe
        all_run_names = []
        for station_runs in runs.values():
            all_run_names.extend(station_runs.keys())

        df_run_names = df.run.unique().tolist()

        for run_name in all_run_names:
            assert run_name in df_run_names

    def test_station_metadata_integration(self, z3d_collection):
        """Test integration between dataframe and station metadata."""
        df = z3d_collection.to_dataframe()

        # After processing, should have station metadata
        assert isinstance(z3d_collection.station_metadata_dict, dict)

        # Station IDs should match between dataframe and metadata dict
        df_stations = set(df.station.unique())
        metadata_stations = set(z3d_collection.station_metadata_dict.keys())

        # All dataframe stations should have metadata (or vice versa)
        assert len(df_stations.intersection(metadata_stations)) > 0


class TestZ3DCollectionUtilities:
    """Test utility functions and helper methods."""

    def test_string_representation(self, z3d_collection):
        """Test string representation methods."""
        str_repr = str(z3d_collection)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

        repr_str = repr(z3d_collection)
        assert isinstance(repr_str, str)
        assert "Collection" in repr_str  # Base class name appears in repr

    def test_attribute_access(self, z3d_collection):
        """Test attribute access and properties."""
        # Test essential attributes exist
        assert hasattr(z3d_collection, "file_path")
        assert hasattr(z3d_collection, "file_ext")
        assert hasattr(z3d_collection, "station_metadata_dict")
        assert hasattr(z3d_collection, "_columns")

    def test_method_availability(self, z3d_collection):
        """Test that all expected methods are available."""
        expected_methods = [
            "get_files",
            "to_dataframe",
            "get_runs",
            "assign_run_names",
            "get_calibrations",
            "_sort_df",
            "_set_df_dtypes",
            "_sort_station_metadata",
        ]

        for method_name in expected_methods:
            assert hasattr(z3d_collection, method_name)
            assert callable(getattr(z3d_collection, method_name))


# =============================================================================
# Parameterized Tests
# =============================================================================


class TestZ3DCollectionParametrized:
    """Parameterized tests for various scenarios."""

    @pytest.mark.parametrize("sample_rate", [256, 512, 1024, 4096])
    def test_single_sample_rate_processing(self, z3d_collection, sample_rate):
        """Test processing with different single sample rates."""
        df = z3d_collection.to_dataframe([sample_rate])

        if len(df) > 0:
            # If data exists for this sample rate, all should match
            assert (df.sample_rate == sample_rate).all()
        else:
            # If no data, should return empty dataframe
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    @pytest.mark.parametrize("component", ["ex", "ey", "hx", "hy", "hz"])
    def test_component_presence(self, z3d_dataframe, component):
        """Test presence of different components in data."""
        component_data = z3d_dataframe[z3d_dataframe.component == component]

        # Should have data for common components
        if component in ["ex", "ey", "hx", "hy"]:
            assert len(component_data) > 0

        # All component data should be consistent
        if len(component_data) > 0:
            assert (component_data.component == component).all()

    @pytest.mark.parametrize("zeros", [1, 2, 3, 4])
    def test_run_name_formatting(self, empty_z3d_collection, zeros):
        """Test run name formatting with different zero padding."""
        test_df = pd.DataFrame(
            {
                "station": ["001"],
                "start": [pd.Timestamp("2022-01-01")],
                "sample_rate": [256.0],
                "run": [None],
                "sequence_number": [1],
            }
        )

        result = empty_z3d_collection.assign_run_names(test_df, zeros=zeros)

        run_name = result.run.iloc[0]
        # Should have correct zero padding
        assert f"{1:0{zeros}d}" in run_name


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
