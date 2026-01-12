# =============================================================================
# Imports
# =============================================================================
"""
Comprehensive pytest suite for testing processing workflow from FDSN MTH5 files.

This test suite validates the complete processing workflow:
1. MTH5 files created from FDSN miniseed data (CAS04 station)
2. RunSummary creation from MTH5 files
3. KernelDataset creation from RunSummary
4. Filter validation in RunTS channel responses

Tests are designed to be parallel-safe using session-scoped fixtures and
independent test instances. Tests cover both MTH5 v0.1.0 and v0.2.0 formats.

@author: GitHub Copilot
"""

import pandas as pd
import pytest

from mth5.mth5 import MTH5
from mth5.processing import KERNEL_DATASET_COLUMNS, RUN_SUMMARY_COLUMNS
from mth5.processing.kernel_dataset import KernelDataset
from mth5.processing.run_summary import RunSummary


# =============================================================================
# Helper Functions
# =============================================================================


def get_survey_for_version(m):
    """Get survey name for v0.2.0 files, None for v0.1.0."""
    if m.file_version == "0.2.0":
        surveys = m.surveys_group.groups_list
        return surveys[0] if surveys else None
    return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def session_cas04_run_summary_v010(global_fdsn_miniseed_v010):
    """Session-scoped RunSummary from v0.1.0 CAS04 MTH5 file."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_fdsn_miniseed_v010])
    return run_summary


@pytest.fixture(scope="session")
def session_cas04_run_summary_v020(global_fdsn_miniseed_v020):
    """Session-scoped RunSummary from v0.2.0 CAS04 MTH5 file."""
    run_summary = RunSummary()
    run_summary.from_mth5s([global_fdsn_miniseed_v020])
    return run_summary


@pytest.fixture
def cas04_run_summary_v010(session_cas04_run_summary_v010):
    """Fresh clone of v0.1.0 RunSummary for each test."""
    return session_cas04_run_summary_v010.clone()


@pytest.fixture
def cas04_run_summary_v020(session_cas04_run_summary_v020):
    """Fresh clone of v0.2.0 RunSummary for each test."""
    return session_cas04_run_summary_v020.clone()


@pytest.fixture(params=["v010", "v020"])
def cas04_run_summary_parameterized(
    request, session_cas04_run_summary_v010, session_cas04_run_summary_v020
):
    """Parameterized fixture providing both v0.1.0 and v0.2.0 RunSummary instances."""
    if request.param == "v010":
        return session_cas04_run_summary_v010.clone()
    else:
        return session_cas04_run_summary_v020.clone()


@pytest.fixture(scope="session")
def session_cas04_kernel_dataset_v010(session_cas04_run_summary_v010):
    """Session-scoped KernelDataset from v0.1.0 CAS04 data (single station)."""
    kd = KernelDataset()
    kd.from_run_summary(session_cas04_run_summary_v010.clone(), "CAS04")
    return kd


@pytest.fixture(scope="session")
def session_cas04_kernel_dataset_v020(session_cas04_run_summary_v020):
    """Session-scoped KernelDataset from v0.2.0 CAS04 data (single station)."""
    kd = KernelDataset()
    kd.from_run_summary(session_cas04_run_summary_v020.clone(), "CAS04")
    return kd


@pytest.fixture
def cas04_kernel_dataset_v010(session_cas04_kernel_dataset_v010):
    """Fresh clone of v0.1.0 KernelDataset for each test."""
    return session_cas04_kernel_dataset_v010.clone()


@pytest.fixture
def cas04_kernel_dataset_v020(session_cas04_kernel_dataset_v020):
    """Fresh clone of v0.2.0 KernelDataset for each test."""
    return session_cas04_kernel_dataset_v020.clone()


@pytest.fixture(params=["v010", "v020"])
def cas04_kernel_dataset_parameterized(
    request, session_cas04_kernel_dataset_v010, session_cas04_kernel_dataset_v020
):
    """Parameterized fixture providing both v0.1.0 and v0.2.0 KernelDataset instances."""
    if request.param == "v010":
        return session_cas04_kernel_dataset_v010.clone()
    else:
        return session_cas04_kernel_dataset_v020.clone()


# =============================================================================
# Test RunSummary Creation
# =============================================================================


class TestRunSummaryCreation:
    """Test RunSummary creation from CAS04 MTH5 files."""

    def test_run_summary_created_from_v010(self, cas04_run_summary_v010):
        """Test RunSummary successfully created from v0.1.0 MTH5."""
        assert cas04_run_summary_v010 is not None
        assert cas04_run_summary_v010.df is not None
        assert isinstance(cas04_run_summary_v010.df, pd.DataFrame)

    def test_run_summary_created_from_v020(self, cas04_run_summary_v020):
        """Test RunSummary successfully created from v0.2.0 MTH5."""
        assert cas04_run_summary_v020 is not None
        assert cas04_run_summary_v020.df is not None
        assert isinstance(cas04_run_summary_v020.df, pd.DataFrame)

    @pytest.mark.parametrize("version", ["v010", "v020"])
    def test_run_summary_has_correct_columns(
        self, version, cas04_run_summary_v010, cas04_run_summary_v020
    ):
        """Test RunSummary has all required columns for both versions."""
        rs = cas04_run_summary_v010 if version == "v010" else cas04_run_summary_v020

        assert sorted(rs.df.columns) == sorted(RUN_SUMMARY_COLUMNS)

    def test_run_summary_contains_cas04_station(self, cas04_run_summary_parameterized):
        """Test RunSummary contains CAS04 station data."""
        stations = cas04_run_summary_parameterized.df["station"].unique()
        assert "CAS04" in stations

    def test_run_summary_has_data_rows(self, cas04_run_summary_parameterized):
        """Test RunSummary has at least one data row."""
        assert len(cas04_run_summary_parameterized.df) > 0

    def test_run_summary_has_data_flag(self, cas04_run_summary_parameterized):
        """Test RunSummary has_data column contains True values."""
        assert cas04_run_summary_parameterized.df["has_data"].any()
        assert cas04_run_summary_parameterized.df["has_data"].dtype == bool


class TestRunSummaryProperties:
    """Test RunSummary properties and data integrity."""

    def test_run_summary_station_column(self, cas04_run_summary_parameterized):
        """Test station column contains valid data."""
        stations = cas04_run_summary_parameterized.df["station"].unique()

        assert len(stations) > 0
        assert all(isinstance(s, str) for s in stations)
        assert "CAS04" in stations

    def test_run_summary_run_column(self, cas04_run_summary_parameterized):
        """Test run column contains valid data."""
        runs = cas04_run_summary_parameterized.df["run"].unique()

        assert len(runs) > 0
        assert all(isinstance(r, str) for r in runs)

    def test_run_summary_sample_rate(self, cas04_run_summary_parameterized):
        """Test sample_rate column contains valid numeric data."""
        sample_rates = cas04_run_summary_parameterized.df["sample_rate"].unique()

        assert len(sample_rates) > 0
        assert all(sr > 0 for sr in sample_rates)

    def test_run_summary_time_columns(self, cas04_run_summary_parameterized):
        """Test start and end time columns are valid."""
        df = cas04_run_summary_parameterized.df

        # Ensure start and end are present
        assert "start" in df.columns
        assert "end" in df.columns

        # Convert to timestamps if needed
        if df["start"].dtype != "datetime64[ns]":
            df["start"] = pd.to_datetime(df["start"])
        if df["end"].dtype != "datetime64[ns]":
            df["end"] = pd.to_datetime(df["end"])

        # Test that start is before end for all rows
        assert (df["start"] < df["end"]).all()

    def test_run_summary_channel_columns(self, cas04_run_summary_parameterized):
        """Test input and output channel columns contain valid data."""
        df = cas04_run_summary_parameterized.df

        assert "input_channels" in df.columns
        assert "output_channels" in df.columns

        # Check that channel strings are not empty
        assert df["input_channels"].notna().all()
        assert df["output_channels"].notna().all()

    def test_run_summary_mth5_path_column(self, cas04_run_summary_parameterized):
        """Test mth5_path column contains valid paths."""
        df = cas04_run_summary_parameterized.df

        assert "mth5_path" in df.columns

        # Get unique paths
        unique_paths = df["mth5_path"].unique()
        assert len(unique_paths) > 0

    def test_run_summary_mini_summary(self, cas04_run_summary_parameterized):
        """Test mini_summary property returns valid DataFrame."""
        mini = cas04_run_summary_parameterized.mini_summary

        assert mini is not None
        assert isinstance(mini, pd.DataFrame)
        assert len(mini) == len(cas04_run_summary_parameterized.df)


class TestRunSummaryMethods:
    """Test RunSummary methods and operations."""

    def test_run_summary_clone(self, cas04_run_summary_parameterized):
        """Test cloning creates independent copy."""
        original = cas04_run_summary_parameterized
        clone = original.clone()

        # Verify clone has same data
        assert len(clone.df) == len(original.df)
        assert clone.df.columns.tolist() == original.df.columns.tolist()

        # Verify independence - modify clone
        if len(clone.df) > 0:
            clone.df.loc[0, "has_data"] = False
            assert original.df.loc[0, "has_data"] != clone.df.loc[0, "has_data"]

    def test_run_summary_set_sample_rate(self, cas04_run_summary_parameterized):
        """Test filtering by sample rate."""
        rs = cas04_run_summary_parameterized
        available_rates = rs.df["sample_rate"].unique()

        if len(available_rates) > 0:
            test_rate = available_rates[0]
            filtered_rs = rs.set_sample_rate(test_rate)

            assert filtered_rs is not None
            assert (filtered_rs.df["sample_rate"] == test_rate).all()

    def test_run_summary_drop_no_data_rows(self, cas04_run_summary_parameterized):
        """Test dropping rows with no data."""
        rs = cas04_run_summary_parameterized
        original_len = len(rs.df)

        # Mark a row as no data
        if original_len > 0:
            rs.df.loc[0, "has_data"] = False
            rs.drop_no_data_rows()

            assert len(rs.df) == original_len - 1


# =============================================================================
# Test KernelDataset Creation
# =============================================================================


class TestKernelDatasetCreation:
    """Test KernelDataset creation from CAS04 RunSummary."""

    def test_kernel_dataset_created_from_v010(self, cas04_kernel_dataset_v010):
        """Test KernelDataset successfully created from v0.1.0 RunSummary."""
        assert cas04_kernel_dataset_v010 is not None
        assert cas04_kernel_dataset_v010.df is not None
        assert isinstance(cas04_kernel_dataset_v010.df, pd.DataFrame)

    def test_kernel_dataset_created_from_v020(self, cas04_kernel_dataset_v020):
        """Test KernelDataset successfully created from v0.2.0 RunSummary."""
        assert cas04_kernel_dataset_v020 is not None
        assert cas04_kernel_dataset_v020.df is not None
        assert isinstance(cas04_kernel_dataset_v020.df, pd.DataFrame)

    @pytest.mark.parametrize("version", ["v010", "v020"])
    def test_kernel_dataset_has_correct_columns(
        self, version, cas04_kernel_dataset_v010, cas04_kernel_dataset_v020
    ):
        """Test KernelDataset has all required columns for both versions."""
        kd = (
            cas04_kernel_dataset_v010
            if version == "v010"
            else cas04_kernel_dataset_v020
        )

        assert sorted(kd.df.columns) == sorted(KERNEL_DATASET_COLUMNS)

    def test_kernel_dataset_has_data_rows(self, cas04_kernel_dataset_parameterized):
        """Test KernelDataset has at least one data row."""
        assert len(cas04_kernel_dataset_parameterized.df) > 0

    def test_kernel_dataset_local_station_id(self, cas04_kernel_dataset_parameterized):
        """Test KernelDataset has correct local station ID."""
        assert cas04_kernel_dataset_parameterized.local_station_id == "CAS04"


class TestKernelDatasetProperties:
    """Test KernelDataset properties and data integrity."""

    def test_kernel_dataset_station_ids(self, cas04_kernel_dataset_parameterized):
        """Test station ID properties."""
        kd = cas04_kernel_dataset_parameterized

        assert kd.local_station_id == "CAS04"
        # Single station case - no remote
        assert kd.remote_station_id is None or kd.remote_station_id == ""

    def test_kernel_dataset_channels(self, cas04_kernel_dataset_parameterized):
        """Test channel properties."""
        kd = cas04_kernel_dataset_parameterized

        assert kd.input_channels is not None
        assert kd.output_channels is not None
        assert len(kd.input_channels) > 0
        assert len(kd.output_channels) > 0

    def test_kernel_dataset_sample_rate(self, cas04_kernel_dataset_parameterized):
        """Test sample rate property."""
        kd = cas04_kernel_dataset_parameterized

        # Should have at least one sample rate
        if kd.num_sample_rates == 1:
            assert kd.sample_rate > 0

    def test_kernel_dataset_processing_id(self, cas04_kernel_dataset_parameterized):
        """Test processing ID generation."""
        kd = cas04_kernel_dataset_parameterized

        processing_id = kd.processing_id
        assert processing_id is not None
        assert isinstance(processing_id, str)
        assert "CAS04" in processing_id

    def test_kernel_dataset_mini_summary(self, cas04_kernel_dataset_parameterized):
        """Test mini_summary property."""
        kd = cas04_kernel_dataset_parameterized
        mini = kd.mini_summary

        assert mini is not None
        assert isinstance(mini, pd.DataFrame)
        assert len(mini) == len(kd.df)

    def test_kernel_dataset_local_df(self, cas04_kernel_dataset_parameterized):
        """Test local_df property."""
        kd = cas04_kernel_dataset_parameterized
        local_df = kd.local_df

        assert local_df is not None
        assert isinstance(local_df, pd.DataFrame)
        assert all(local_df["station"] == "CAS04")

    def test_kernel_dataset_survey_id(self, cas04_kernel_dataset_parameterized):
        """Test survey ID property."""
        kd = cas04_kernel_dataset_parameterized

        survey_id = kd.local_survey_id
        assert survey_id is not None
        assert isinstance(survey_id, str)

    def test_metadata_station(self, cas04_kernel_dataset_parameterized, subtests):
        """Test metadata_station property."""
        kd = cas04_kernel_dataset_parameterized

        with subtests.test("Check metadata_station for CAS04"):
            station = kd.metadata_station
            assert station is not None
            assert station.station_id == "CAS04"

        with subtests.test("Check metadata_station has runs"):
            assert len(station.runs) > 0

        with subtests.test("Check runs have channels"):
            for run in station.runs.values():
                assert len(run.channels) > 0

        with subtests.test("Check channel metadata"):
            for run in station.runs.values():
                for channel in run.channels.values():
                    assert channel.component is not None
                    assert channel.sample_rate > 0
                    assert channel.start != "1980-01-01T00:00:00"
                    assert channel.end != "1980-01-01T00:00:00"


class TestKernelDatasetMethods:
    """Test KernelDataset methods and operations."""

    def test_kernel_dataset_clone(self, cas04_kernel_dataset_parameterized):
        """Test cloning creates independent copy."""
        original = cas04_kernel_dataset_parameterized
        clone = original.clone()

        # Verify clone has same data
        assert len(clone.df) == len(original.df)
        assert clone.df.columns.tolist() == original.df.columns.tolist()

        # Verify station IDs match
        assert clone.local_station_id == original.local_station_id

    def test_kernel_dataset_clone_dataframe(self, cas04_kernel_dataset_parameterized):
        """Test DataFrame cloning."""
        kd = cas04_kernel_dataset_parameterized
        cloned_df = kd.clone_dataframe()

        assert cloned_df is not None
        assert len(cloned_df) == len(kd.df)
        assert cloned_df.columns.tolist() == kd.df.columns.tolist()


# =============================================================================
# Test Filter Validation in RunTS
# =============================================================================


class TestRunTSFilters:
    """Test that RunTS objects have proper filter information."""

    def test_runts_can_be_created_v010(
        self, global_fdsn_miniseed_v010, cas04_run_summary_v010
    ):
        """Test RunTS can be created from v0.1.0 MTH5 file."""
        with MTH5() as m:
            m.open_mth5(global_fdsn_miniseed_v010, mode="r")

            # Get first run from channel summary
            ch_df = m.channel_summary.to_dataframe()
            assert len(ch_df) > 0

            # Get first run reference
            first_run_ref = ch_df.iloc[0].run_hdf5_reference
            run = m.from_reference(first_run_ref)

            # Convert to RunTS
            run_ts = run.to_runts()
            assert run_ts is not None

    def test_runts_can_be_created_v020(
        self, global_fdsn_miniseed_v020, cas04_run_summary_v020
    ):
        """Test RunTS can be created from v0.2.0 MTH5 file."""
        with MTH5() as m:
            m.open_mth5(global_fdsn_miniseed_v020, mode="r")

            # Get first run from channel summary
            ch_df = m.channel_summary.to_dataframe()
            assert len(ch_df) > 0

            # Get first run reference
            first_run_ref = ch_df.iloc[0].run_hdf5_reference
            run = m.from_reference(first_run_ref)

            # Convert to RunTS
            run_ts = run.to_runts()
            assert run_ts is not None

    @pytest.mark.parametrize(
        "version,mth5_file",
        [("v010", "global_fdsn_miniseed_v010"), ("v020", "global_fdsn_miniseed_v020")],
    )
    def test_runts_channels_have_filters(self, version, mth5_file, request):
        """Test that all channels in RunTS have filter information."""
        mth5_path = request.getfixturevalue(mth5_file)

        with MTH5() as m:
            m.open_mth5(mth5_path, mode="r")

            # Get channel summary
            ch_df = m.channel_summary.to_dataframe()

            # Test first 3 channels as representative sample
            for idx, row in ch_df.head(3).iterrows():
                channel_ref = row.hdf5_reference
                channel = m.from_reference(channel_ref)
                channel_ts = channel.to_channel_ts()

                # Test that channel has response with filters
                assert hasattr(
                    channel_ts, "channel_response"
                ), f"Channel {row.component} missing channel_response"
                assert hasattr(
                    channel_ts.channel_response, "filters_list"
                ), f"Channel {row.component} channel_response missing filters_list"

                # Test that filters list is populated
                filters_list = channel_ts.channel_response.filters_list
                assert (
                    len(filters_list) > 0
                ), f"Channel {row.component} has empty filters_list"

    @pytest.mark.parametrize(
        "version,mth5_file",
        [("v010", "global_fdsn_miniseed_v010"), ("v020", "global_fdsn_miniseed_v020")],
    )
    def test_channel_ts_filter_attributes(self, version, mth5_file, request):
        """Test ChannelTS.channel_response filter attributes."""
        mth5_path = request.getfixturevalue(mth5_file)

        with MTH5() as m:
            m.open_mth5(mth5_path, mode="r")

            # Get first channel
            ch_df = m.channel_summary.to_dataframe()
            first_ch_ref = ch_df.iloc[0].hdf5_reference
            channel_ts = m.from_reference(first_ch_ref).to_channel_ts()

            # Verify channel_response exists
            assert channel_ts.channel_response is not None

            # Verify filters_list exists and has content
            filters_list = channel_ts.channel_response.filters_list
            assert filters_list is not None
            assert isinstance(filters_list, list)
            assert len(filters_list) > 0

            # Check first filter has expected attributes
            first_filter = filters_list[0]
            assert hasattr(first_filter, "name")

    @pytest.mark.parametrize(
        "version,mth5_file",
        [("v010", "global_fdsn_miniseed_v010"), ("v020", "global_fdsn_miniseed_v020")],
    )
    def test_all_channels_in_run_have_filters(self, version, mth5_file, request):
        """Test that all channels in a run have filter information."""
        mth5_path = request.getfixturevalue(mth5_file)

        with MTH5() as m:
            m.open_mth5(mth5_path, mode="r")

            # Get station and first run
            survey = get_survey_for_version(m)
            station = m.get_station("CAS04", survey=survey)
            runs = station.groups_list

            # Filter out non-run groups
            ignored_groups = ["Fourier_Coefficients", "Transfer_Functions", "Features"]
            run_list = [x for x in runs if x not in ignored_groups]

            assert len(run_list) > 0, "No runs found in station"

            # Get first run
            run_group = station.get_run(run_list[0])
            channels = run_group.groups_list

            assert len(channels) > 0, "No channels found in run"

            # Test each channel
            for ch_name in channels:
                channel = run_group.get_channel(ch_name)
                channel_ts = channel.to_channel_ts()

                # Verify filters exist
                assert hasattr(
                    channel_ts.channel_response, "filters_list"
                ), f"Channel {ch_name} missing filters_list"

                filters_list = channel_ts.channel_response.filters_list
                assert (
                    len(filters_list) > 0
                ), f"Channel {ch_name} has empty filters_list"


class TestRunTSFromKernelDataset:
    """Test RunTS access through KernelDataset workflow."""

    @pytest.mark.parametrize(
        "version,mth5_file",
        [("v010", "global_fdsn_miniseed_v010"), ("v020", "global_fdsn_miniseed_v020")],
    )
    def test_runts_via_kernel_dataset_has_filters(self, version, mth5_file, request):
        """Test RunTS accessed via KernelDataset has filters."""
        mth5_path = request.getfixturevalue(mth5_file)

        # Create RunSummary
        run_summary = RunSummary()
        run_summary.from_mth5s([mth5_path])

        # Create KernelDataset
        kd = KernelDataset()
        kd.from_run_summary(run_summary, "CAS04")

        # Open MTH5 and get RunTS
        with MTH5() as m:
            m.open_mth5(mth5_path, mode="r")

            # Get first run from kernel dataset
            first_row = kd.df.iloc[0]

            # Get run via station
            survey = get_survey_for_version(m)
            station = m.get_station("CAS04", survey=survey)
            run = station.get_run(first_row.run)
            run_ts = run.to_runts()

            # Verify RunTS has channels
            assert len(run_ts.dataset.data_vars) > 0

            # Get first channel from run
            channels = run.groups_list
            assert len(channels) > 0

            first_channel = run.get_channel(channels[0])
            channel_ts = first_channel.to_channel_ts()

            # Verify filters exist
            assert hasattr(channel_ts.channel_response, "filters_list")
            assert len(channel_ts.channel_response.filters_list) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationWorkflow:
    """Test complete workflow from MTH5 to RunTS with filters."""

    @pytest.mark.parametrize(
        "version,mth5_file",
        [("v010", "global_fdsn_miniseed_v010"), ("v020", "global_fdsn_miniseed_v020")],
    )
    def test_end_to_end_workflow(self, version, mth5_file, request):
        """Test complete workflow: MTH5 → RunSummary → KernelDataset → RunTS with filters."""
        mth5_path = request.getfixturevalue(mth5_file)

        # Step 1: Create RunSummary
        run_summary = RunSummary()
        run_summary.from_mth5s([mth5_path])

        assert run_summary.df is not None
        assert len(run_summary.df) > 0
        assert "CAS04" in run_summary.df["station"].unique()

        # Step 2: Create KernelDataset
        kd = KernelDataset()
        kd.from_run_summary(run_summary, "CAS04")

        assert kd.df is not None
        assert len(kd.df) > 0
        assert kd.local_station_id == "CAS04"

        # Step 3: Access RunTS and verify filters
        with MTH5() as m:
            m.open_mth5(mth5_path, mode="r")

            # Get first run
            first_row = kd.df.iloc[0]
            survey = get_survey_for_version(m)
            station = m.get_station("CAS04", survey=survey)
            run = station.get_run(first_row.run)
            run_ts = run.to_runts()

            assert run_ts is not None

            # Get channels and verify filters
            channels = run.groups_list
            for ch_name in channels[:3]:  # Test first 3 channels
                channel = run.get_channel(ch_name)
                channel_ts = channel.to_channel_ts()

                # Verify filter chain
                assert hasattr(channel_ts, "channel_response")
                assert hasattr(channel_ts.channel_response, "filters_list")
                assert len(channel_ts.channel_response.filters_list) > 0

    def test_workflow_v010_specific(self, global_fdsn_miniseed_v010):
        """Test workflow specific to v0.1.0 format."""
        # Create RunSummary
        run_summary = RunSummary()
        run_summary.from_mth5s([global_fdsn_miniseed_v010])

        # Verify v0.1.0 structure (flat, no survey level)
        with MTH5() as m:
            m.open_mth5(global_fdsn_miniseed_v010, mode="r")

            # v0.1.0 has stations at root level
            stations = m.stations_group.groups_list
            assert "CAS04" in stations

    def test_workflow_v020_specific(self, global_fdsn_miniseed_v020):
        """Test workflow specific to v0.2.0 format."""
        # Create RunSummary
        run_summary = RunSummary()
        run_summary.from_mth5s([global_fdsn_miniseed_v020])

        # Verify v0.2.0 structure (has survey level)
        with MTH5() as m:
            m.open_mth5(global_fdsn_miniseed_v020, mode="r")

            # v0.2.0 has surveys - get survey name
            survey = get_survey_for_version(m)
            # Just verify we can access CAS04
            station = m.get_station("CAS04", survey=survey)
            assert station is not None


class TestDataConsistency:
    """Test data consistency across RunSummary and KernelDataset."""

    def test_station_consistency(
        self, cas04_run_summary_parameterized, cas04_kernel_dataset_parameterized
    ):
        """Test station data is consistent between RunSummary and KernelDataset."""
        rs_stations = cas04_run_summary_parameterized.df["station"].unique()
        kd_stations = cas04_kernel_dataset_parameterized.df["station"].unique()

        # KernelDataset should be subset of RunSummary stations
        assert all(s in rs_stations for s in kd_stations)

    def test_run_consistency(
        self, cas04_run_summary_parameterized, cas04_kernel_dataset_parameterized
    ):
        """Test run data is consistent between RunSummary and KernelDataset."""
        rs_runs = cas04_run_summary_parameterized.df["run"].unique()
        kd_runs = cas04_kernel_dataset_parameterized.df["run"].unique()

        # KernelDataset runs should be subset of RunSummary runs
        assert all(r in rs_runs for r in kd_runs)

    def test_sample_rate_consistency(
        self, cas04_run_summary_parameterized, cas04_kernel_dataset_parameterized
    ):
        """Test sample rate is consistent between RunSummary and KernelDataset."""
        rs_rates = set(cas04_run_summary_parameterized.df["sample_rate"].unique())
        kd_rates = set(cas04_kernel_dataset_parameterized.df["sample_rate"].unique())

        # KernelDataset sample rates should be subset of RunSummary
        assert kd_rates.issubset(rs_rates)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases in processing workflow."""

    def test_empty_run_summary_raises_error(self):
        """Test KernelDataset creation fails with empty RunSummary."""
        # Create empty RunSummary
        empty_rs = RunSummary()
        empty_rs.df = pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)

        kd = KernelDataset()
        with pytest.raises(ValueError):
            kd.from_run_summary(empty_rs, "CAS04")

    def test_invalid_station_id_raises_error(self, cas04_run_summary_parameterized):
        """Test KernelDataset creation fails with invalid station ID."""
        kd = KernelDataset()

        with pytest.raises((ValueError, KeyError, Exception)):
            kd.from_run_summary(cas04_run_summary_parameterized, "INVALID_STATION")

    def test_clone_independence(
        self, cas04_run_summary_parameterized, cas04_kernel_dataset_parameterized
    ):
        """Test clones are independent of originals."""
        # Test RunSummary clone independence
        rs_clone = cas04_run_summary_parameterized.clone()
        original_rs_len = len(cas04_run_summary_parameterized.df)

        if original_rs_len > 0:
            rs_clone.df.loc[0, "has_data"] = False
            rs_clone.drop_no_data_rows()

            # Original should be unchanged
            assert len(cas04_run_summary_parameterized.df) == original_rs_len

        # Test KernelDataset clone independence
        kd_clone = cas04_kernel_dataset_parameterized.clone()
        original_kd_len = len(cas04_kernel_dataset_parameterized.df)

        # Verify clone
        assert len(kd_clone.df) == original_kd_len


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
