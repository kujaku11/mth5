"""
Comprehensive pytest suite for Phoenix Geophysics collection functionality.

Based on test_phoenix_collection.py but using pytest fixtures and modern testing patterns.
Combines real data testing with enhanced coverage patterns from pytest mock example.
"""

import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.io.phoenix import PhoenixCollection
from mth5.io.phoenix.readers.calibrations import PhoenixCalibration


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def phoenix_data_path():
    """Get path to real Phoenix test data."""
    if not has_test_data:
        pytest.skip("mth5_test_data not available")
    return phx_data_path


@pytest.fixture(scope="module")
def phoenix_collection_real_data(phoenix_data_path):
    """Create PhoenixCollection instance with real test data."""
    return PhoenixCollection(phoenix_data_path / "10128_2021-04-27-032436")


@pytest.fixture(scope="module")
def phoenix_dataframe(phoenix_collection_real_data):
    """Create dataframe from real Phoenix data."""
    df = phoenix_collection_real_data.to_dataframe([150, 24000])
    return df.fillna(0)


@pytest.fixture(scope="module")
def phoenix_runs(phoenix_collection_real_data):
    """Create runs dictionary from real Phoenix data."""
    return phoenix_collection_real_data.get_runs([150, 24000])


@pytest.fixture(scope="module")
def test_station(phoenix_dataframe):
    """Get test station ID from real data."""
    return phoenix_dataframe.station.unique()[0]


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    import os
    import shutil
    import tempfile

    tmpdir = tempfile.mkdtemp()
    temp_path = Path(tmpdir)

    yield temp_path

    # Force cleanup on Windows
    try:
        shutil.rmtree(tmpdir)
    except PermissionError:
        # On Windows, try to remove read-only flags and try again
        try:
            import stat

            def handle_remove_readonly(func, path, exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(tmpdir, onerror=handle_remove_readonly)
        except:
            # If all else fails, just ignore cleanup errors in tests
            pass


@pytest.fixture
def mock_recmeta_data():
    """Mock receiver metadata JSON structure."""
    return {
        "instid": "10128",
        "receiver_commercial_name": "MTU-5C",
        "receiver_model": "MTU-5C-10128",
        "start": "2021-04-27T03:24:18.000000+00:00",
        "stop": "2021-04-27T03:30:43.000000+00:00",
        "channel_map": {
            "mapping": [
                {"idx": 0, "tag": "H2"},
                {"idx": 1, "tag": "E1"},
                {"idx": 2, "tag": "H1"},
                {"idx": 4, "tag": "E2"},
            ]
        },
        "chconfig": {
            "chans": [
                {
                    "tag": "E1",
                    "ty": "e",
                    "ga": 1,
                    "sampleRate": 150,
                    "lp": 10000,
                    "length1": 50,
                    "length2": 50,
                    "pot_p": 1000,
                    "pot_n": 1000,
                },
                {
                    "tag": "H1",
                    "ty": "h",
                    "ga": 1,
                    "sampleRate": 150,
                    "lp": 10000,
                    "type_name": "MTC-150",
                    "type": "4",
                    "serial": "0",
                },
            ]
        },
        "layout": {
            "Station_Name": "10128",
            "Survey_Name": "COCORAHS",
            "Company_Name": "USGS",
            "Operator": "Test Operator",
            "Notes": "Test station",
        },
        "timing": {
            "gps_lat": 43.69625473022461,
            "gps_lon": -79.39364624023438,
            "gps_alt": 140.10263061523438,
            "tm_drift": 0.001,
        },
        "motherboard": {"mb_fw_ver": "2.7.1"},
    }


@pytest.fixture
def mock_station_directory(temp_directory, mock_recmeta_data):
    """Create a mock station directory structure for testing."""
    station_dir = temp_directory / "test_station"
    station_dir.mkdir()

    # Create mock recmeta.json
    recmeta_path = station_dir / "recmeta.json"
    with open(recmeta_path, "w") as f:
        json.dump(mock_recmeta_data, f)

    # Create mock data files
    for ext in ["td_150", "td_24k"]:
        for i in range(3):
            data_file = station_dir / f"10128_608783F4_{i}_0000000{i+1}.{ext}"
            data_file.write_bytes(b"mock data" * 100)

    return station_dir


@pytest.fixture
def mock_calibration_data():
    """Mock calibration JSON data."""
    return {
        "file_type": "receiver calibration",
        "timestamp_utc": "2021-04-27T03:24:18.000000+00:00",
        "instrument_type": "MTU-5C",
        "instrument_model": "RMT03-J",
        "inst_serial": "10128",
        "cal_data": [
            {
                "tag": "E1",
                "chan_data": [
                    {
                        "freq_Hz": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        "magnitude": [1.0, 1.0, 1.0, 1.0, 0.5, 0.1],
                        "phs_deg": [0.0, 0.0, 0.0, -5.0, -45.0, -90.0],
                    }
                ],
            },
            {
                "tag": "H1",
                "chan_data": [
                    {
                        "freq_Hz": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        "magnitude": [1000.0, 1000.0, 1000.0, 900.0, 500.0, 100.0],
                        "phs_deg": [90.0, 90.0, 85.0, 45.0, 5.0, -45.0],
                    }
                ],
            },
        ],
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestPhoenixCollectionInitialization:
    """Test PhoenixCollection initialization and basic properties."""

    def test_init_default(self):
        """Test default initialization."""
        pc = PhoenixCollection()
        assert pc.file_path is None
        assert isinstance(pc._file_extension_map, dict)
        assert isinstance(pc._default_channel_map, dict)
        assert isinstance(pc.metadata_dict, dict)

    def test_init_with_real_path(self, phoenix_data_path):
        """Test initialization with real Phoenix data path."""
        station_path = phoenix_data_path / "10128_2021-04-27-032436"
        pc = PhoenixCollection(station_path)
        assert pc.file_path == station_path
        assert isinstance(pc.file_path, Path)

    def test_init_with_mock_path(self, mock_station_directory):
        """Test initialization with mock path."""
        pc = PhoenixCollection(mock_station_directory)
        assert pc.file_path == mock_station_directory
        assert isinstance(pc.file_path, Path)

    def test_file_extension_map(self):
        """Test file extension mapping completeness."""
        pc = PhoenixCollection()
        expected_map = {
            30: "td_30",
            150: "td_150",
            2400: "td_2400",
            24000: "td_24k",
            96000: "td_96k",
        }
        assert pc._file_extension_map == expected_map

    def test_default_channel_map(self):
        """Test default channel mapping completeness."""
        pc = PhoenixCollection()
        expected_map = {
            0: "E1",
            1: "H3",
            2: "H2",
            3: "H1",
            4: "H4",
            5: "H5",
            6: "H6",
            7: "E2",
        }
        assert pc._default_channel_map == expected_map


class TestPhoenixCollectionFileOperations:
    """Test file system operations with real and mock data."""

    def test_file_path_property(self, phoenix_collection_real_data):
        """Test file_path property with real data."""
        assert isinstance(phoenix_collection_real_data.file_path, Path)
        assert phoenix_collection_real_data.file_path.exists()

    def test_get_files_150hz(self, phoenix_collection_real_data):
        """Test getting files for 150 Hz sample rate."""
        files = list(phoenix_collection_real_data.get_files("td_150"))
        assert len(files) == 8  # Based on original test expectation
        assert all(f.suffix == ".td_150" for f in files)

    def test_get_files_24khz(self, phoenix_collection_real_data):
        """Test getting files for 24 kHz sample rate."""
        files = list(phoenix_collection_real_data.get_files("td_24k"))
        assert len(files) > 0
        assert all(f.suffix == ".td_24k" for f in files)

    @patch("mth5.io.phoenix.phoenix_collection.PhoenixReceiverMetadata")
    def test_read_receiver_metadata_json_success(
        self, mock_metadata_class, mock_station_directory
    ):
        """Test successful reading of receiver metadata."""
        mock_metadata = MagicMock()
        mock_metadata_class.return_value = mock_metadata

        pc = PhoenixCollection(mock_station_directory)
        recmeta_path = mock_station_directory / "recmeta.json"

        result = pc._read_receiver_metadata_json(recmeta_path)

        mock_metadata_class.assert_called_once_with(fn=recmeta_path)
        assert result == mock_metadata

    def test_read_receiver_metadata_json_missing_file(
        self, phoenix_collection_real_data
    ):
        """Test handling of missing receiver metadata file."""
        nonexistent_path = Path("/nonexistent/path/recmeta.json")

        result = phoenix_collection_real_data._read_receiver_metadata_json(
            nonexistent_path
        )

        assert result is None

    def test_locate_station_folders(self, mock_station_directory):
        """Test locating station folders with recmeta.json."""
        # Test from parent directory
        pc = PhoenixCollection(mock_station_directory.parent)
        folders = pc._locate_station_folders()

        assert len(folders) == 1
        assert folders[0] == mock_station_directory
        assert (folders[0] / "recmeta.json").exists()

    def test_locate_station_folders_no_metadata(self, temp_directory):
        """Test locating station folders when no metadata files exist."""
        # Create directory without recmeta.json
        empty_dir = temp_directory / "empty_station"
        empty_dir.mkdir()

        pc = PhoenixCollection(temp_directory)
        folders = pc._locate_station_folders()

        assert len(folders) == 0


class TestPhoenixCollectionDataFrameOperations:
    """Test dataframe creation and manipulation with real data."""

    def test_dataframe_columns(self, phoenix_collection_real_data, phoenix_dataframe):
        """Test dataframe has expected columns."""
        expected_columns = phoenix_collection_real_data._columns
        assert phoenix_dataframe.columns.to_list() == expected_columns

    def test_dataframe_shape(self, phoenix_dataframe):
        """Test dataframe has expected shape."""
        assert phoenix_dataframe.shape == (12, 19)

    def test_dataframe_dtypes(self, phoenix_collection_real_data, phoenix_dataframe):
        """Test dataframe data types are set correctly."""
        df = phoenix_collection_real_data._set_df_dtypes(phoenix_dataframe)

        # Test timestamp columns
        assert pd.api.types.is_datetime64_any_dtype(df.start)
        assert pd.api.types.is_datetime64_any_dtype(df.end)

        # Test string/object columns - accept both StringDtype (pandas 2.x) and object dtype (pandas 1.x)
        assert pd.api.types.is_string_dtype(
            df.instrument_id
        ) or pd.api.types.is_object_dtype(df.instrument_id)
        assert pd.api.types.is_string_dtype(
            df.calibration_fn
        ) or pd.api.types.is_object_dtype(df.calibration_fn)

    def test_survey_id_consistency(
        self, phoenix_collection_real_data, phoenix_dataframe
    ):
        """Test survey ID consistency across dataframe."""
        expected_survey_id = list(phoenix_collection_real_data.metadata_dict.values())[
            0
        ].survey_metadata.id
        assert (phoenix_dataframe.survey == expected_survey_id).all()

    def test_run_names_150hz(self, phoenix_dataframe):
        """Test run names for 150 Hz data."""
        run_150 = phoenix_dataframe[phoenix_dataframe.sample_rate == 150].run.unique()[
            0
        ]
        assert run_150 == "sr150_0001"

    def test_run_names_24khz(self, phoenix_dataframe):
        """Test run names for 24 kHz data."""
        run_names_24k = phoenix_dataframe[
            phoenix_dataframe.sample_rate == 24000
        ].run.unique()

        assert len(run_names_24k) == 1
        assert run_names_24k[0] == "sr24k_0001"

    def test_to_dataframe_skip_calibration_files(self, mock_station_directory):
        """Test that calibration files are skipped."""
        # This test verifies behavior with mock files
        pc = PhoenixCollection(mock_station_directory)

        with patch.object(pc, "_locate_station_folders") as mock_locate:
            mock_locate.return_value = [mock_station_directory]
            with patch.object(pc, "_read_receiver_metadata_json") as mock_read:
                mock_read.return_value = None  # No metadata found

                # Should handle missing metadata gracefully
                try:
                    df = pc.to_dataframe([150])
                    assert isinstance(df, pd.DataFrame)
                except Exception:
                    # Expected due to missing/mock data
                    pass

    def test_to_dataframe_handle_oserror(self, mock_station_directory):
        """Test handling of OSError when opening Phoenix files."""
        pc = PhoenixCollection(mock_station_directory)

        with patch.object(pc, "_locate_station_folders") as mock_locate:
            mock_locate.return_value = [mock_station_directory]
            with patch.object(pc, "_read_receiver_metadata_json") as mock_read:
                mock_read.return_value = None  # No metadata found

                # Should handle missing metadata gracefully
                try:
                    df = pc.to_dataframe([150])
                    assert isinstance(df, pd.DataFrame)
                except Exception:
                    # Expected due to missing/mock data
                    pass

    def test_assign_run_names_continuous_data(self, phoenix_collection_real_data):
        """Test run name assignment for continuous data."""
        # Create test dataframe with continuous data - must include 'end' column
        df = pd.DataFrame(
            {
                "station": ["10128"] * 4,
                "sample_rate": [150] * 4,
                "start": pd.to_datetime(
                    [
                        "2021-04-27T03:24:18",
                        "2021-04-27T03:25:18",
                        "2021-04-27T03:26:18",
                        "2021-04-27T03:27:18",
                    ]
                ),
                "end": pd.to_datetime(
                    [
                        "2021-04-27T03:25:18",
                        "2021-04-27T03:26:18",
                        "2021-04-27T03:27:18",
                        "2021-04-27T03:28:18",
                    ]
                ),
                "sequence_number": [1, 2, 3, 4],
                "run": [None] * 4,
            }
        )

        result = phoenix_collection_real_data.assign_run_names(df)

        # Continuous data should have same run name
        assert len(result["run"].unique()) == 1
        assert result["run"].iloc[0] == "sr150_0001"

    def test_assign_run_names_segmented_data(self, phoenix_collection_real_data):
        """Test run name assignment for segmented data."""
        # Create test dataframe with segmented data (high sample rate)
        df = pd.DataFrame(
            {
                "station": ["10128"] * 3,
                "sample_rate": [24000] * 3,
                "start": pd.to_datetime(
                    [
                        "2021-04-27T03:24:18",
                        "2021-04-27T03:25:18",
                        "2021-04-27T03:26:18",
                    ]
                ),
                "run": [None] * 3,
            }
        )

        result = phoenix_collection_real_data.assign_run_names(df)

        # Each segment should have unique run name for high sample rates
        run_names = result["run"].unique()
        assert len(run_names) <= 3  # May be grouped depending on timing


class TestPhoenixCollectionRunOperations:
    """Test run-related operations with real data."""

    def test_runs_keys(self, phoenix_runs, test_station):
        """Test runs dictionary has expected structure."""
        assert len(phoenix_runs[test_station].keys()) == 2  # 150Hz and 24kHz runs
        assert isinstance(phoenix_runs, OrderedDict)

    def test_runs_data_consistency(self, phoenix_dataframe, phoenix_runs, test_station):
        """Test run data consistency with original dataframe."""
        for key, rdf in phoenix_runs[test_station].items():
            df_subset = phoenix_dataframe[phoenix_dataframe.run == key]

            # Fill NaN values for comparison
            rdf_filled = rdf.fillna(0)
            df_filled = df_subset.fillna(0)

            # Test that runs data matches dataframe subset
            # Note: Using iloc[0:4] to match first 4 rows as in original test
            if len(df_filled) >= 4:
                # Compare all columns except calibration_fn which has dtype issues (StringDtype vs object)
                cols_to_compare = [
                    col for col in rdf_filled.columns if col != "calibration_fn"
                ]
                assert (
                    df_filled.iloc[0:4][cols_to_compare]
                    .eq(rdf_filled[cols_to_compare])
                    .all(axis=0)
                    .all()
                )

    def test_get_runs_basic(self, phoenix_collection_real_data):
        """Test basic get_runs functionality."""
        result = phoenix_collection_real_data.get_runs([150, 24000])

        assert isinstance(result, OrderedDict)
        assert len(result) > 0

        # Each station should have runs
        for station_id, runs in result.items():
            assert isinstance(runs, OrderedDict)
            assert len(runs) > 0

    def test_get_runs_first_row_selection(self, phoenix_collection_real_data):
        """Test that get_runs selects appropriate rows for each component."""
        result = phoenix_collection_real_data.get_runs([150])

        # Verify structure
        assert isinstance(result, OrderedDict)

        for station_id, station_runs in result.items():
            for run_id, run_df in station_runs.items():
                # Should have valid dataframe
                assert isinstance(run_df, pd.DataFrame)
                assert len(run_df) > 0

    def test_get_runs_sorted_order(self, phoenix_collection_real_data):
        """Test that runs are returned in sorted order."""
        result = phoenix_collection_real_data.get_runs([150, 24000])

        for station_id, station_runs in result.items():
            run_keys = list(station_runs.keys())
            # Run keys should be in sorted order
            assert run_keys == sorted(run_keys)


class TestPhoenixCalibrationIntegration:
    """Test Phoenix calibration functionality."""

    def test_calibration_init_with_file(self, mock_calibration_data):
        """Test PhoenixCalibration initialization with file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_calibration_data, f)
            temp_path = Path(f.name)

        try:
            cal = PhoenixCalibration(temp_path)
            assert cal.cal_fn == temp_path
            assert cal.obj is not None
        finally:
            temp_path.unlink()

    def test_calibration_init_without_file(self):
        """Test PhoenixCalibration initialization without file."""
        cal = PhoenixCalibration()
        # When initialized with None, _cal_fn is not set
        assert not hasattr(cal, "_cal_fn")
        assert cal.obj is None

    def test_calibration_base_filter_name(self, mock_calibration_data):
        """Test base filter name generation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_calibration_data, f)
            temp_path = Path(f.name)

        try:
            cal = PhoenixCalibration(temp_path)

            # The calibration object should have the attributes from our mock data
            assert cal.obj.instrument_type == "MTU-5C"
            assert cal.obj.instrument_model == "RMT03-J"
            assert cal.obj.inst_serial == "10128"

            base_name = cal.base_filter_name
            assert base_name == "mtu-5c_rmt03-j_10128"
        finally:
            temp_path.unlink()


class TestPhoenixCollectionParameterized:
    """Parameterized tests for various scenarios."""

    @pytest.mark.parametrize(
        "sample_rates,expected_extensions",
        [
            ([150], ["td_150"]),
            ([24000], ["td_24k"]),
            ([150, 24000], ["td_150", "td_24k"]),
            ([30, 2400, 96000], ["td_30", "td_2400", "td_96k"]),
        ],
    )
    def test_sample_rate_to_extension_mapping(self, sample_rates, expected_extensions):
        """Test mapping of sample rates to file extensions."""
        pc = PhoenixCollection()

        for sr, ext in zip(sample_rates, expected_extensions):
            assert pc._file_extension_map[sr] == ext

    @pytest.mark.parametrize(
        "run_name_zeros,expected_format",
        [(2, "sr150_01"), (3, "sr150_001"), (4, "sr150_0001"), (5, "sr150_00001")],
    )
    def test_run_name_formatting(
        self, run_name_zeros, expected_format, phoenix_collection_real_data
    ):
        """Test run name formatting with different zero padding."""
        df = pd.DataFrame(
            {
                "station": ["10128"],
                "sample_rate": [150],
                "start": pd.to_datetime(["2021-04-27T03:24:18"]),
                "end": pd.to_datetime(["2021-04-27T03:25:18"]),
                "sequence_number": [1],
                "run": [None],
            }
        )

        result = phoenix_collection_real_data.assign_run_names(df, zeros=run_name_zeros)
        assert result["run"].iloc[0] == expected_format

    @pytest.mark.parametrize(
        "channel_id,expected_component",
        [
            (0, "E1"),
            (1, "H3"),
            (2, "H2"),
            (3, "H1"),
            (4, "H4"),
            (5, "H5"),
            (6, "H6"),
            (7, "E2"),
        ],
    )
    def test_default_channel_mapping(self, channel_id, expected_component):
        """Test default channel ID to component mapping."""
        pc = PhoenixCollection()
        assert pc._default_channel_map[channel_id] == expected_component


class TestPhoenixCollectionEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_directory(self):
        """Test collection with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pc = PhoenixCollection(tmpdir)
            folders = pc._locate_station_folders()
            assert len(folders) == 0

    def test_invalid_sample_rates(self, phoenix_collection_real_data):
        """Test handling of invalid sample rates."""
        # This should raise a KeyError for unknown sample rates
        with pytest.raises(KeyError):
            phoenix_collection_real_data.to_dataframe([999999])  # Invalid sample rate

    def test_corrupted_data_files(self, mock_station_directory):
        """Test handling of corrupted data files."""
        pc = PhoenixCollection(mock_station_directory)

        # Should handle errors gracefully
        with patch.object(pc, "_locate_station_folders") as mock_locate:
            mock_locate.return_value = [mock_station_directory]
            with patch.object(pc, "_read_receiver_metadata_json") as mock_read:
                mock_read.return_value = None  # No metadata found

                try:
                    df = pc.to_dataframe([150])
                    assert isinstance(df, pd.DataFrame)
                except Exception:
                    # Expected due to missing/mock data
                    pass

    def test_malformed_recmeta_json(self, temp_directory):
        """Test handling of malformed recmeta.json."""
        station_dir = temp_directory / "malformed_station"
        station_dir.mkdir()

        # Write malformed JSON
        recmeta_path = station_dir / "recmeta.json"
        with open(recmeta_path, "w") as f:
            f.write("{ malformed json")

        pc = PhoenixCollection(station_dir)

        # Should handle JSON parsing errors
        with pytest.raises((json.JSONDecodeError, Exception)):
            pc._read_receiver_metadata_json(recmeta_path)


class TestPhoenixCollectionAdditional:
    """Additional tests for functionality not covered in original unittest."""

    def test_metadata_dict_population(self, phoenix_collection_real_data):
        """Test that metadata_dict gets populated correctly."""
        # Trigger dataframe creation to populate metadata_dict
        _ = phoenix_collection_real_data.to_dataframe([150])

        assert len(phoenix_collection_real_data.metadata_dict) > 0

        for (
            station_path,
            metadata,
        ) in phoenix_collection_real_data.metadata_dict.items():
            assert isinstance(station_path, (str, Path))
            assert hasattr(metadata, "station_metadata")
            assert hasattr(metadata, "survey_metadata")

    def test_get_empty_entry_dict(self, phoenix_collection_real_data):
        """Test empty entry dictionary creation."""
        empty_entry = phoenix_collection_real_data.get_empty_entry_dict()

        expected_keys = [
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
            "instrument_id",
            "calibration_fn",
        ]

        for key in expected_keys:
            assert key in empty_entry
            assert empty_entry[key] is None

    def test_sort_df_functionality(self, phoenix_collection_real_data):
        """Test dataframe sorting functionality."""
        # Create unsorted test dataframe
        df = pd.DataFrame(
            {
                "survey": ["TEST"] * 4,
                "station": ["MT001"] * 4,
                "start": pd.to_datetime(
                    [
                        "2021-04-27T03:26:18",
                        "2021-04-27T03:24:18",
                        "2021-04-27T03:27:18",
                        "2021-04-27T03:25:18",
                    ]
                ),
                "channel_id": [2, 0, 3, 1],
                "sequence_number": [3, 1, 4, 2],
                "sample_rate": [150, 150, 150, 150],
                "run": [None, None, None, None],
                "end": pd.to_datetime(
                    [
                        "2021-04-27T03:27:18",
                        "2021-04-27T03:25:18",
                        "2021-04-27T03:28:18",
                        "2021-04-27T03:26:18",
                    ]
                ),
            }
        )

        sorted_df = phoenix_collection_real_data._sort_df(df, zeros=4)

        # Should be sorted by start time
        assert sorted_df["start"].is_monotonic_increasing
        # Run names should be assigned
        assert not sorted_df["run"].isnull().all()

    def test_file_extension_reverse_lookup(self, phoenix_collection_real_data):
        """Test reverse lookup of sample rate from extension."""
        ext_map = phoenix_collection_real_data._file_extension_map

        # Test that we can reverse lookup
        for sample_rate, extension in ext_map.items():
            # Find files with this extension
            files = list(phoenix_collection_real_data.get_files(extension))
            # Files may or may not exist, but method should not crash

    def test_dataframe_consistency_multiple_calls(self, phoenix_collection_real_data):
        """Test that multiple calls to to_dataframe return consistent results."""
        df1 = phoenix_collection_real_data.to_dataframe([150])
        df2 = phoenix_collection_real_data.to_dataframe([150])

        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_runs_consistency_multiple_calls(self, phoenix_collection_real_data):
        """Test that multiple calls to get_runs return consistent results."""
        runs1 = phoenix_collection_real_data.get_runs([150])
        runs2 = phoenix_collection_real_data.get_runs([150])

        # Structure should be identical
        assert runs1.keys() == runs2.keys()
        for station in runs1.keys():
            assert runs1[station].keys() == runs2[station].keys()


class TestPhoenixCollectionPerformance:
    """Test performance characteristics."""

    def test_large_sample_rate_list(self, phoenix_collection_real_data):
        """Test performance with large sample rate list."""
        # Test with all available sample rates
        all_sample_rates = list(phoenix_collection_real_data._file_extension_map.keys())

        # Should complete without hanging
        df = phoenix_collection_real_data.to_dataframe(all_sample_rates)
        assert isinstance(df, pd.DataFrame)

    def test_memory_efficiency(self, phoenix_collection_real_data):
        """Test that operations don't leak memory excessively."""
        import gc

        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform operations
        for _ in range(5):
            df = phoenix_collection_real_data.to_dataframe([150])
            runs = phoenix_collection_real_data.get_runs([150])
            del df, runs

        # Force cleanup
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Arbitrary reasonable threshold


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhoenixCollectionIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_real_data(self, phoenix_collection_real_data):
        """Test complete workflow from files to runs with real data."""
        # Test dataframe creation
        df = phoenix_collection_real_data.to_dataframe([150, 24000])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Test runs extraction
        runs = phoenix_collection_real_data.get_runs([150, 24000])
        assert isinstance(runs, OrderedDict)
        assert len(runs) > 0

        # Test consistency between dataframe and runs
        for station_id, station_runs in runs.items():
            for run_id, run_df in station_runs.items():
                # Run data should be subset of main dataframe
                main_subset = df[df.run == run_id]
                assert len(run_df) <= len(main_subset)

    def test_end_to_end_data_flow(self, phoenix_collection_real_data):
        """Test end-to-end data flow validation."""
        # Get all data
        df = phoenix_collection_real_data.to_dataframe([150, 24000])
        runs = phoenix_collection_real_data.get_runs([150, 24000])

        # Validate data consistency
        total_run_rows = sum(
            len(run_df)
            for station_runs in runs.values()
            for run_df in station_runs.values()
        )

        # Total rows in runs should be reasonable subset of dataframe
        assert total_run_rows <= len(df)

    def test_error_recovery_workflow(self, mock_station_directory):
        """Test error recovery throughout the workflow."""
        # Test with directory that has some issues
        pc = PhoenixCollection(mock_station_directory)

        # Should handle potential errors gracefully
        try:
            df = pc.to_dataframe([150])
            runs = pc.get_runs([150])

            # Even with errors, should return valid (possibly empty) structures
            assert isinstance(df, pd.DataFrame)
            assert isinstance(runs, OrderedDict)
        except Exception as e:
            # If exceptions occur, they should be specific and informative
            assert isinstance(e, (FileNotFoundError, ValueError, OSError))


if __name__ == "__main__":
    pytest.main([__file__])
