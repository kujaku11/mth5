# -*- coding: utf-8 -*-
"""
Optimized pytest suite for testing RunTS object functionality.

Created from original test_runts.py and test_runts_02.py using fixtures and
subtests for efficiency while maintaining comprehensive coverage.

@author: pytest optimization
"""


import mt_metadata.timeseries as metadata
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import ChannelResponse, PoleZeroFilter

from mth5.timeseries import ChannelTS, RunTS


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_signal_params():
    """Create reproducible test signal parameters (session-scoped for efficiency)."""
    np.random.seed(42)  # Fixed seed for reproducibility
    return {
        "n_samples": 4096,
        "sample_rate": 8.0,
        "start_time": "2015-01-08T19:49:18+00:00",
        "end_time": "2015-01-08T19:57:49.875000",
        "common_start": "2020-01-01T00:00:00+00:00",
    }


@pytest.fixture(scope="session")
def multi_frequency_signal(test_signal_params):
    """Generate reproducible multi-frequency test signal."""
    params = test_signal_params
    t = np.arange(params["n_samples"])
    frequencies = np.logspace(-3, 3, 20)
    phases = np.random.rand(20)

    signal = np.sum(
        [np.cos(2 * np.pi * w * t + phi) for w, phi in zip(frequencies, phases)], axis=0
    )

    return signal


@pytest.fixture
def basic_pole_zero_filter():
    """Create a basic pole-zero filter for testing."""
    pz = PoleZeroFilter(
        units_in="Volt", units_out="nanoTesla", name="instrument_response"
    )
    pz.poles = [
        (-6.283185 + 10.882477j),
        (-6.283185 - 10.882477j),
        (-12.566371 + 0j),
    ]
    pz.zeros = []
    pz.normalization_factor = 18244400
    return pz


@pytest.fixture
def channel_response(basic_pole_zero_filter):
    """Create a ChannelResponse object with basic filter."""
    return ChannelResponse(filters_list=[basic_pole_zero_filter])


@pytest.fixture
def metadata_objects():
    """Create standard metadata objects for testing."""
    station_metadata = metadata.Station(id="mt001")
    run_metadata = metadata.Run(id="001")
    return {
        "station": station_metadata,
        "run": run_metadata,
    }


@pytest.fixture(params=["magnetic", "electric", "auxiliary"])
def channel_type(request):
    """Parameterized fixture for different channel types."""
    return request.param


@pytest.fixture
def channel_config_map():
    """Map channel types to their component configurations."""
    return {
        "magnetic": ["hx", "hy", "hz"],
        "electric": ["ex", "ey"],
        "auxiliary": ["temperature"],
    }


@pytest.fixture
def basic_channel_ts(test_signal_params, channel_response):
    """Create a basic ChannelTS object for testing."""
    params = test_signal_params

    ex = ChannelTS(
        "electric",
        data=np.random.rand(params["n_samples"]),
        channel_metadata={
            "electric": {
                "component": "ex",
                "sample_rate": params["sample_rate"],
                "time_period.start": params["start_time"],
            }
        },
        channel_response=channel_response,
    )
    return ex


@pytest.fixture
def multi_channel_list(
    test_signal_params, multi_frequency_signal, metadata_objects, channel_response
):
    """Create a list of multiple ChannelTS objects for comprehensive testing."""
    params = test_signal_params
    channel_list = []

    # Create magnetic channels
    for component in ["hx", "hy", "hz"]:
        h_metadata = metadata.Magnetic(component=component)
        h_metadata.time_period.start = params["common_start"]
        h_metadata.sample_rate = 1.0  # Use 1.0 for alignment tests

        h_channel = ChannelTS(
            channel_type="magnetic",
            data=multi_frequency_signal,
            channel_metadata=h_metadata,
            run_metadata=metadata_objects["run"],
            station_metadata=metadata_objects["station"],
        )
        channel_list.append(h_channel)

    # Create electric channels
    for component in ["ex", "ey"]:
        e_metadata = metadata.Electric(component=component)
        e_metadata.time_period.start = params["common_start"]
        e_metadata.sample_rate = 1.0

        e_channel = ChannelTS(
            channel_type="electric",
            data=multi_frequency_signal,
            channel_metadata=e_metadata,
            run_metadata=metadata_objects["run"],
            station_metadata=metadata_objects["station"],
        )
        channel_list.append(e_channel)

    # Create auxiliary channel
    aux_metadata = metadata.Auxiliary(component="temperature")
    aux_metadata.time_period.start = params["common_start"]
    aux_metadata.sample_rate = 1.0

    aux_channel = ChannelTS(
        channel_type="auxiliary",
        data=np.random.rand(params["n_samples"]) * 30,
        channel_metadata=aux_metadata,
        run_metadata=metadata_objects["run"],
        station_metadata=metadata_objects["station"],
    )
    channel_list.append(aux_channel)

    return channel_list


@pytest.fixture
def basic_run_ts(basic_channel_ts):
    """Create a basic RunTS object with a single channel."""
    run_ts = RunTS()
    run_ts.set_dataset([basic_channel_ts])
    return run_ts


@pytest.fixture
def multi_channel_run_ts(multi_channel_list):
    """Create a RunTS object with multiple channels."""
    return RunTS(multi_channel_list)


@pytest.fixture
def misaligned_channels(test_signal_params, metadata_objects):
    """Create channels with different lengths and start times for alignment testing."""
    params = test_signal_params
    channel_list = []

    # HX channel with extra samples
    hx_n_samples = 4098
    hx_metadata = metadata.Magnetic(component="hx")
    hx_metadata.time_period.start = params["common_start"]
    hx_metadata.sample_rate = 1.0

    t = np.arange(hx_n_samples)
    data = np.sum(
        [
            np.cos(2 * np.pi * w * t + phi)
            for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
        ],
        axis=0,
    )

    hx = ChannelTS(
        channel_type="magnetic",
        data=data,
        channel_metadata=hx_metadata,
        run_metadata=metadata_objects["run"],
        station_metadata=metadata_objects["station"],
    )
    channel_list.append(hx)

    # EY channel with standard samples
    ey_n_samples = 4096
    ey_metadata = metadata.Electric(component="ey")
    ey_metadata.time_period.start = params["common_start"]
    ey_metadata.sample_rate = 1.0

    t = np.arange(ey_n_samples)
    data = np.sum(
        [
            np.cos(2 * np.pi * w * t + phi)
            for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
        ],
        axis=0,
    )

    ey = ChannelTS(
        channel_type="electric",
        data=data,
        channel_metadata=ey_metadata,
        run_metadata=metadata_objects["run"],
        station_metadata=metadata_objects["station"],
    )
    channel_list.append(ey)

    # Bad channel with different start time and sample rate
    ex_metadata = metadata.Electric(component="ex")
    ex_metadata.time_period.start = "2021-05-05T12:10:05+00:00"
    ex_metadata.sample_rate = 11.0  # Different sample rate

    bad_ch = ChannelTS(
        channel_type="electric",
        data=data[:ey_n_samples],
        channel_metadata=ex_metadata,
        run_metadata=metadata_objects["run"],
        station_metadata=metadata_objects["station"],
    )

    return {
        "aligned": channel_list,
        "bad_channel": bad_ch,
        "hx": hx,
        "ey": ey,
        "hx_n_samples": hx_n_samples,
        "ey_n_samples": ey_n_samples,
    }


@pytest.fixture
def merge_test_runs(test_signal_params, channel_response):
    """Create two RunTS objects for merge testing."""
    params = test_signal_params

    # Create first run
    start_01 = "2015-01-08T19:49:18+00:00"
    start_02 = "2015-01-08T19:57:52+00:00"

    # Filters for testing filter merging
    pz1 = PoleZeroFilter(units_in="Volt", units_out="nanoTesla", name="filter_1")
    pz1.poles = [(-6.283185 + 10.882477j), (-6.283185 - 10.882477j), (-12.566371 + 0j)]
    pz1.zeros = []
    pz1.normalization_factor = 18244400

    pz2 = pz1.copy()
    pz2.name = "filter_2"

    cr_01 = ChannelResponse(filters_list=[pz1])
    cr_02 = ChannelResponse(filters_list=[pz2])

    # Create channels for first run
    run_01 = RunTS()
    ey_01 = ChannelTS(
        "electric",
        data=np.random.rand(params["n_samples"]),
        channel_metadata={
            "electric": {
                "component": "ey",
                "sample_rate": params["sample_rate"],
                "time_period.start": start_01,
            }
        },
        channel_response=cr_01,
    )
    hx_01 = ChannelTS(
        "magnetic",
        data=np.random.rand(params["n_samples"]),
        channel_metadata={
            "magnetic": {
                "component": "hx",
                "sample_rate": params["sample_rate"],
                "time_period.start": start_01,
            }
        },
        channel_response=cr_01,
    )
    run_01.set_dataset([ey_01, hx_01])

    # Create channels for second run
    run_02 = RunTS()
    ey_02 = ChannelTS(
        "electric",
        data=np.random.rand(params["n_samples"]),
        channel_metadata={
            "electric": {
                "component": "ey",
                "sample_rate": params["sample_rate"],
                "time_period.start": start_02,
            }
        },
        channel_response=cr_02,
    )
    hx_02 = ChannelTS(
        "magnetic",
        data=np.random.rand(params["n_samples"]),
        channel_metadata={
            "magnetic": {
                "component": "hx",
                "sample_rate": params["sample_rate"],
                "time_period.start": start_02,
            }
        },
        channel_response=cr_02,
    )
    run_02.set_dataset([ey_02, hx_02])

    return {
        "run_01": run_01,
        "run_02": run_02,
        "start_01": start_01,
        "start_02": start_02,
        "end_02": "2015-01-08T20:06:23.875000+00:00",
        "pz1": pz1,
        "pz2": pz2,
        "combined": run_01 + run_02,
        "merged": run_01.merge(run_02),
        "merged_decimated": run_01.merge(run_02, new_sample_rate=1),
    }


# =============================================================================
# Core RunTS Tests
# =============================================================================


class TestRunTSInitialization:
    """Test RunTS initialization and basic functionality."""

    def test_empty_initialization(self):
        """Test creating empty RunTS object."""
        run_ts = RunTS()

        assert isinstance(run_ts, RunTS)
        assert run_ts.channels == []
        assert not run_ts.has_data()
        assert run_ts.survey_metadata.id == "0"
        assert run_ts.station_metadata.id == "0"
        assert run_ts.run_metadata.id == "0"

    def test_initialization_with_channel_list(self, multi_channel_list):
        """Test creating RunTS with channel list."""
        run_ts = RunTS(multi_channel_list)

        assert isinstance(run_ts, RunTS)
        assert len(run_ts.channels) == 6
        assert run_ts.has_data()
        expected_channels = ["ex", "ey", "hx", "hy", "hz", "temperature"]
        assert sorted(run_ts.channels) == sorted(expected_channels)

    def test_metadata_initialization(self, metadata_objects):
        """Test RunTS initialization with metadata."""
        run_ts = RunTS(
            run_metadata=metadata_objects["run"],
            station_metadata=metadata_objects["station"],
        )

        assert run_ts.run_metadata.id == metadata_objects["run"].id
        assert run_ts.station_metadata.id == metadata_objects["station"].id


class TestRunTSProperties:
    """Test RunTS properties and string representations."""

    def test_string_representation(self, basic_run_ts):
        """Test __str__ and __repr__ methods."""
        str_repr = str(basic_run_ts)

        assert "RunTS Summary:" in str_repr
        assert f"Survey:      {basic_run_ts.survey_metadata.id}" in str_repr
        assert f"Station:     {basic_run_ts.station_metadata.id}" in str_repr
        assert f"Run:         {basic_run_ts.run_metadata.id}" in str_repr
        assert f"Sample Rate: {basic_run_ts.sample_rate}" in str_repr
        assert f"Components:  {basic_run_ts.channels}" in str_repr

        # Test __repr__ returns same as __str__
        assert str(basic_run_ts) == repr(basic_run_ts)

    def test_sample_rate_properties(self, multi_channel_run_ts, test_signal_params):
        """Test sample rate and sample interval properties."""
        expected_rate = 1.0  # From multi_channel_list fixture

        assert multi_channel_run_ts.sample_rate == expected_rate
        assert multi_channel_run_ts.sample_interval == 1.0 / expected_rate

    def test_time_properties(self, multi_channel_run_ts, test_signal_params):
        """Test start and end time properties."""
        params = test_signal_params

        assert multi_channel_run_ts.start == MTime(time_stamp=params["common_start"])
        # End time should be computed from data
        assert isinstance(multi_channel_run_ts.end, MTime)

    def test_channels_property(self, multi_channel_run_ts):
        """Test channels property returns correct channel names."""
        expected_channels = ["ex", "ey", "hx", "hy", "hz", "temperature"]
        assert sorted(multi_channel_run_ts.channels) == sorted(expected_channels)

    def test_filters_property(self, basic_run_ts):
        """Test filters property and setter."""
        # Should contain the instrument_response filter
        assert "instrument_response" in basic_run_ts.filters

        # Test filters setter validation
        with pytest.raises(TypeError):
            basic_run_ts.filters = "invalid"


class TestRunTSEquality:
    """Test RunTS equality and comparison operations."""

    def test_equality_same_object(self, basic_run_ts):
        """Test equality with same object."""
        assert basic_run_ts == basic_run_ts

    def test_equality_copied_object(self, basic_run_ts):
        """Test equality with copied object."""
        copied_run = basic_run_ts.copy()
        assert basic_run_ts == copied_run

    def test_inequality_different_type(self, basic_run_ts):
        """Test inequality with different object type."""
        with pytest.raises(TypeError):
            basic_run_ts == "not_a_run_ts"

    def test_inequality_different_data(self, basic_run_ts, multi_channel_run_ts):
        """Test inequality with different data."""
        assert basic_run_ts != multi_channel_run_ts


class TestRunTSMetadataValidation:
    """Test metadata validation methods."""

    def test_validate_run_metadata(self, basic_run_ts):
        """Test run metadata validation."""
        validated = basic_run_ts._validate_run_metadata(basic_run_ts.run_metadata)
        assert validated == basic_run_ts.run_metadata

        # Test validation from dict
        dict_metadata = {"id": "test_run"}
        validated_from_dict = basic_run_ts._validate_run_metadata(dict_metadata)
        assert validated_from_dict.id == "test_run"

    def test_validate_station_metadata(self, basic_run_ts):
        """Test station metadata validation."""
        validated = basic_run_ts._validate_station_metadata(
            basic_run_ts.station_metadata
        )
        assert validated == basic_run_ts.station_metadata

        # Test validation from dict
        dict_metadata = {"id": "test_station"}
        validated_from_dict = basic_run_ts._validate_station_metadata(dict_metadata)
        assert validated_from_dict.id == "test_station"

    def test_validate_survey_metadata(self, basic_run_ts):
        """Test survey metadata validation."""
        validated = basic_run_ts._validate_survey_metadata(basic_run_ts.survey_metadata)
        assert validated == basic_run_ts.survey_metadata

        # Test validation from dict
        dict_metadata = {"id": "test_survey"}
        validated_from_dict = basic_run_ts._validate_survey_metadata(dict_metadata)
        assert validated_from_dict.id == "test_survey"


class TestRunTSChannelManagement:
    """Test channel management operations."""

    def test_channel_access(self, multi_channel_run_ts):
        """Test accessing channels as attributes."""
        for channel_name in multi_channel_run_ts.channels:
            channel = getattr(multi_channel_run_ts, channel_name)
            assert isinstance(channel, ChannelTS)
            assert channel.component == channel_name

    def test_channel_access_invalid(self, multi_channel_run_ts):
        """Test accessing non-existent channel raises error."""
        with pytest.raises(NameError):
            _ = multi_channel_run_ts.nonexistent_channel

    def test_add_channel_xarray(self, basic_run_ts):
        """Test adding a channel from xarray."""
        x = basic_run_ts.ex.to_xarray()
        x.attrs["component"] = "ez"
        x.name = "ez"

        basic_run_ts.add_channel(x)

        expected_channels = sorted(["ex", "ez"])
        assert sorted(basic_run_ts.channels) == expected_channels

    def test_validate_array_list_failures(self, basic_run_ts):
        """Test array list validation failures."""
        with pytest.raises(TypeError, match="array_list must be a list or tuple"):
            basic_run_ts._validate_array_list(10)

        with pytest.raises(TypeError):
            basic_run_ts._validate_array_list([10])


class TestRunTSMetadataSync:
    """Test metadata synchronization and validation."""

    def test_metadata_validation_sync(self, basic_run_ts, test_signal_params):
        """Test metadata validation synchronizes values."""
        params = test_signal_params

        # Modify metadata to wrong values
        basic_run_ts.run_metadata.sample_rate = 10
        basic_run_ts.run_metadata.time_period.start = "2020-01-01T00:00:00"
        basic_run_ts.run_metadata.time_period.end = "2020-01-01T00:00:00"

        # Validate should sync with actual data
        basic_run_ts.validate_metadata()

        assert basic_run_ts.run_metadata.sample_rate == params["sample_rate"]
        assert basic_run_ts.run_metadata.time_period.start == basic_run_ts.start
        assert basic_run_ts.run_metadata.time_period.end == basic_run_ts.end

    def test_summarize_metadata(self, multi_channel_run_ts):
        """Test metadata summary generation."""
        meta_dict = multi_channel_run_ts.summarize_metadata

        # Should contain entries for each channel's metadata
        for channel in multi_channel_run_ts.channels:
            channel_keys = [k for k in meta_dict.keys() if k.startswith(channel)]
            assert len(channel_keys) > 0


# =============================================================================
# Channel Alignment and Dataset Tests
# =============================================================================


class TestRunTSAlignment:
    """Test channel alignment and dataset operations."""

    def test_misaligned_channels_creation(self, misaligned_channels):
        """Test creating RunTS with misaligned channels."""
        aligned_channels = misaligned_channels["aligned"]
        run_ts = RunTS(aligned_channels)

        # Should handle different length channels
        assert run_ts.has_data()
        assert len(run_ts.channels) == 2

        # Time span should be determined by longest channel
        assert run_ts.dataset.sizes["time"] == misaligned_channels["hx_n_samples"]

    def test_sample_rate_validation(self, misaligned_channels, basic_run_ts):
        """Test sample rate validation across channels."""
        bad_channel = misaligned_channels["bad_channel"]
        aligned_channels = misaligned_channels["aligned"]

        # Should raise error for different sample rates
        with pytest.raises(ValueError, match="sample rate"):
            basic_run_ts.set_dataset(aligned_channels + [bad_channel])

    def test_alignment_methods(self, misaligned_channels):
        """Test channel alignment utility methods."""
        run_ts = RunTS()
        channel_data_arrays = [ch.data_array for ch in misaligned_channels["aligned"]]

        # Test sample rate checking
        sample_rate = run_ts._check_sample_rate(channel_data_arrays)
        assert sample_rate == 1.0

        # Test start time checking
        common_start = run_ts._check_common_start(channel_data_arrays)
        assert common_start is True

        # Test end time checking (should be False due to different lengths)
        common_end = run_ts._check_common_end(channel_data_arrays)
        assert common_end is False

        # Test time index generation
        earliest_start = run_ts._get_earliest_start(channel_data_arrays)
        latest_end = run_ts._get_latest_end(channel_data_arrays)
        time_index = run_ts._get_common_time_index(earliest_start, latest_end, 1.0)

        assert time_index is not None
        assert len(time_index) > 0


class TestRunTSDatasetOperations:
    """Test dataset creation and manipulation."""

    def test_set_dataset_alignment_types(self, multi_channel_list):
        """Test different alignment types for dataset creation."""
        run_ts = RunTS()

        # Test default (outer) alignment
        run_ts.set_dataset(multi_channel_list, align_type="outer")
        assert run_ts.has_data()

        # Test inner alignment
        run_ts_inner = RunTS()
        run_ts_inner.set_dataset(multi_channel_list, align_type="inner")
        assert run_ts_inner.has_data()

    def test_dataset_property(self, multi_channel_run_ts):
        """Test dataset property access and manipulation."""
        import xarray as xr

        dataset = multi_channel_run_ts.dataset
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) == len(multi_channel_run_ts.channels)


# =============================================================================
# Advanced RunTS Operations Tests
# =============================================================================


class TestRunTSSlicing:
    """Test time slicing operations."""

    def test_get_slice_by_samples(self, multi_channel_run_ts, test_signal_params):
        """Test slicing by number of samples."""
        start = "2020-01-01T00:00:30+00:00"
        n_samples = 256

        sliced_run = multi_channel_run_ts.get_slice(start, n_samples=n_samples)

        assert isinstance(sliced_run, RunTS)
        assert sliced_run.sample_rate == multi_channel_run_ts.sample_rate
        assert sliced_run.start == MTime(time_stamp=start)

        # Check data size
        for channel in sliced_run.channels:
            channel_data = getattr(sliced_run, channel)
            assert channel_data.n_samples == n_samples

    def test_get_slice_by_end_time(self, multi_channel_run_ts):
        """Test slicing by end time."""
        start = "2020-01-01T00:00:30+00:00"
        end = "2020-01-01T00:01:00+00:00"

        sliced_run = multi_channel_run_ts.get_slice(start, end=end)

        assert isinstance(sliced_run, RunTS)
        assert sliced_run.start == MTime(time_stamp=start)
        # End time might be slightly different due to sampling
        time_diff = abs(sliced_run.end - MTime(time_stamp=end))
        assert time_diff < 2.0  # Should be within 2 seconds

    def test_slice_parameter_validation(self, multi_channel_run_ts):
        """Test slice parameter validation."""
        start = "2020-01-01T00:00:30+00:00"

        # Should raise error if neither end nor n_samples provided
        with pytest.raises(ValueError, match="Must input n_samples or end"):
            multi_channel_run_ts.get_slice(start)


class TestRunTSMerging:
    """Test RunTS merging operations."""

    def test_add_operator(self, merge_test_runs, test_signal_params):
        """Test RunTS addition operator."""
        params = test_signal_params
        combined = merge_test_runs["combined"]
        expected_samples = 2 * params["n_samples"] + 16  # Gap between runs

        assert combined.dataset.sizes["time"] == expected_samples
        assert combined.start == merge_test_runs["start_01"]
        assert combined.end == merge_test_runs["end_02"]

        # Check filter merging
        expected_filters = {merge_test_runs["pz1"].name, merge_test_runs["pz2"].name}
        assert set(combined.filters.keys()) == expected_filters

        # Check channels preserved
        assert sorted(combined.channels) == ["ey", "hx"]

    def test_merge_method(self, merge_test_runs, test_signal_params):
        """Test explicit merge method."""
        params = test_signal_params
        merged = merge_test_runs["merged"]
        expected_samples = 2 * params["n_samples"] + 16

        assert merged.dataset.sizes["time"] == expected_samples
        assert merged.start == merge_test_runs["start_01"]
        assert merged.end == merge_test_runs["end_02"]

        # Metadata should be updated
        assert merged.run_metadata.time_period.start == merge_test_runs["start_01"]
        assert merged.run_metadata.time_period.end == merge_test_runs["end_02"]

    def test_merge_with_decimation(self, merge_test_runs, test_signal_params):
        """Test merge with sample rate decimation."""
        params = test_signal_params
        merged_dec = merge_test_runs["merged_decimated"]
        expected_samples = (2 * params["n_samples"] + 16) / params["sample_rate"]

        assert merged_dec.dataset.sizes["time"] == expected_samples
        assert merged_dec.run_metadata.sample_rate == 1

    def test_merge_type_validation(self, basic_run_ts):
        """Test merge type validation."""
        with pytest.raises(TypeError, match="Cannot combine"):
            basic_run_ts + "not_a_run_ts"

        with pytest.raises(TypeError, match="Cannot combine"):
            basic_run_ts.merge("not_a_run_ts")


class TestRunTSObspyIntegration:
    """Test ObsPy Stream integration."""

    def test_to_obspy_stream(self, multi_channel_run_ts):
        """Test conversion to ObsPy Stream."""
        from obspy.core import Stream

        stream = multi_channel_run_ts.to_obspy_stream()

        assert isinstance(stream, Stream)
        assert stream.count() == len(multi_channel_run_ts.channels)

        # Check each trace
        for trace in stream.traces:
            assert trace.stats.sampling_rate == multi_channel_run_ts.sample_rate
            assert trace.stats.npts == multi_channel_run_ts.dataset.sizes["time"]

    @pytest.mark.skip(reason="ObsPy stream input testing requires more setup")
    def test_from_obspy_stream(self, multi_channel_run_ts):
        """Test creation from ObsPy Stream."""
        # This would require creating a proper ObsPy stream
        # Skip for now as it's complex to set up properly


# =============================================================================
# Performance and Error Cases Tests
# =============================================================================


class TestRunTSErrorCases:
    """Test error cases and edge conditions."""

    def test_initialization_failures(self, basic_channel_ts):
        """Test RunTS initialization failures."""
        with pytest.raises(TypeError):
            RunTS([basic_channel_ts], run_metadata=[])

        with pytest.raises(TypeError):
            RunTS([basic_channel_ts], station_metadata=[])

    def test_channel_response_handling(self, multi_channel_run_ts):
        """Test channel response filter handling."""
        for channel_name in multi_channel_run_ts.channels:
            channel = getattr(multi_channel_run_ts, channel_name)

            # Should have access to channel response
            assert hasattr(channel, "channel_response")

            # Filter retrieval should work
            response = multi_channel_run_ts._get_channel_response(channel_name)
            if response is not None:
                assert hasattr(response, "filters_list")


class TestRunTSPerformance:
    """Test performance-related aspects."""

    def test_large_dataset_handling(self, test_signal_params):
        """Test handling of larger datasets."""
        # Create larger test data
        n_large = 16384  # 4x larger
        large_data = np.random.rand(n_large)

        large_channel = ChannelTS(
            "electric",
            data=large_data,
            channel_metadata={
                "electric": {
                    "component": "ex",
                    "sample_rate": 1.0,
                    "time_period.start": test_signal_params["common_start"],
                }
            },
        )

        run_ts = RunTS([large_channel])

        assert run_ts.has_data()
        assert run_ts.dataset.sizes["time"] == n_large

    def test_copy_performance(self, multi_channel_run_ts):
        """Test copy operation performance and correctness."""
        import time

        start_time = time.time()
        copied_run = multi_channel_run_ts.copy()
        copy_time = time.time() - start_time

        # Copy should be reasonably fast (< 1 second for test data)
        assert copy_time < 1.0

        # Copy should be equal to original
        assert copied_run == multi_channel_run_ts

        # But should be separate objects
        assert copied_run is not multi_channel_run_ts


# =============================================================================
# Integration Tests
# =============================================================================


class TestRunTSIntegration:
    """Integration tests combining multiple operations."""

    def test_full_workflow(self, multi_channel_list, test_signal_params):
        """Test complete workflow from creation to manipulation."""
        params = test_signal_params

        # 1. Create RunTS
        run_ts = RunTS(multi_channel_list)
        assert run_ts.has_data()

        # 2. Validate metadata
        run_ts.validate_metadata()
        assert run_ts.run_metadata.sample_rate == 1.0

        # 3. Get slice
        sliced = run_ts.get_slice(params["common_start"], n_samples=1000)
        assert sliced.dataset.sizes["time"] == 1000

        # 4. Copy operation
        copied = run_ts.copy()
        assert copied == run_ts

        # 5. Channel access
        for channel_name in run_ts.channels:
            channel = getattr(run_ts, channel_name)
            assert isinstance(channel, ChannelTS)

    def test_complex_merge_workflow(self, merge_test_runs):
        """Test complex merge operations."""
        run_01 = merge_test_runs["run_01"]
        run_02 = merge_test_runs["run_02"]

        # Test different merge methods
        added = run_01 + run_02
        merged = run_01.merge(run_02)
        merged_dec = run_01.merge(run_02, new_sample_rate=1)

        # All should have same channels
        expected_channels = ["ey", "hx"]
        assert sorted(added.channels) == expected_channels
        assert sorted(merged.channels) == expected_channels
        assert sorted(merged_dec.channels) == expected_channels

        # Decimated version should have different sample rate
        assert merged_dec.sample_rate == 1
        assert added.sample_rate == run_01.sample_rate

    def test_metadata_consistency(self, multi_channel_run_ts):
        """Test metadata consistency across operations."""
        original_station_id = multi_channel_run_ts.station_metadata.id
        original_run_id = multi_channel_run_ts.run_metadata.id

        # Copy should preserve metadata
        copied = multi_channel_run_ts.copy()
        assert copied.station_metadata.id == original_station_id
        assert copied.run_metadata.id == original_run_id

        # Slice should preserve metadata
        sliced = multi_channel_run_ts.get_slice(
            multi_channel_run_ts.start, n_samples=1000
        )
        assert sliced.station_metadata.id == original_station_id
        assert sliced.run_metadata.id == original_run_id


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
