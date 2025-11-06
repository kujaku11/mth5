# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:05:54 2021

Pytest version of StationXML tests with fixtures and subtests

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
#  Imports
# =============================================================================
import pytest
from pathlib import Path
from mth5.mth5 import MTH5
from mt_metadata.timeseries import stationxml
from mt_metadata import STATIONXML_01


@pytest.fixture(scope="session")
def experiment():
    """Create experiment from StationXML - session scoped for efficiency."""
    translator = stationxml.XMLInventoryMTExperiment()
    return translator.xml_to_mt(stationxml_fn=STATIONXML_01)


@pytest.fixture(scope="session")
def mth5_file(experiment):
    """Create MTH5 file from experiment - session scoped for efficiency."""
    fn_path = Path(__file__).parent
    fn = fn_path.joinpath("from_stationxml_pytest.h5")

    # Clean up any existing file
    if fn.exists():
        fn.unlink()

    m = MTH5(file_version="0.1.0")
    m.open_mth5(fn)
    m.from_experiment(experiment, 0)

    yield m

    # Cleanup
    m.close_mth5()
    if fn.exists():
        fn.unlink()


@pytest.fixture
def station_cas04(mth5_file):
    """Get CAS04 station for tests that need it."""
    return mth5_file.get_station("CAS04")


@pytest.fixture
def run_001(mth5_file):
    """Get run 001 for tests that need it."""
    return mth5_file.get_run("CAS04", "001")


@pytest.fixture
def ey_channel(mth5_file):
    """Get ey channel for tests that need it."""
    return mth5_file.get_channel("CAS04", "001", "ey")


@pytest.fixture
def hy_channel(mth5_file):
    """Get hy channel for tests that need it."""
    return mth5_file.get_channel("CAS04", "001", "hy")


class TestStationXMLStructure:
    """Test MTH5 file structure and group existence."""

    @pytest.mark.parametrize(
        "group_path,expected",
        [
            ("Survey", True),
            ("Survey/Stations", True),
            ("Survey/Stations/CAS04", True),
            ("Survey/Stations/CAS04/001", True),
            ("Survey/Stations/CAS04/001/ey", True),
            ("Survey/Stations/CAS04/001/hy", True),
        ],
    )
    def test_has_groups(self, mth5_file, group_path, expected):
        """Test that expected groups exist in the MTH5 file."""
        assert mth5_file.has_group(group_path) == expected


class TestSurveyMetadata:
    """Test survey-level metadata."""

    def test_survey_fdsn_network(self, mth5_file):
        """Test survey FDSN network."""
        assert mth5_file.survey_group.metadata.fdsn.network == "ZU"

    def test_survey_time_period(self, mth5_file):
        """Test survey time period."""
        metadata = mth5_file.survey_group.metadata
        assert metadata.time_period.start_date == "2020-06-02"
        assert metadata.time_period.end_date == "2020-07-13"

    def test_survey_summary(self, mth5_file):
        """Test survey summary."""
        expected_summary = (
            "USMTArray South Magnetotelluric Time Series (USMTArray CONUS South-USGS)"
        )
        assert mth5_file.survey_group.metadata.summary == expected_summary

    def test_survey_doi(self, mth5_file):
        """Test survey DOI."""
        expected_doi = "https://doi.org/10.7914/SN/ZU_2020"
        assert (
            mth5_file.survey_group.metadata.citation_dataset.doi.unicode_string()
            == expected_doi
        )


class TestStationMetadata:
    """Test station-level metadata."""

    @pytest.fixture
    def station_metadata_dict(self):
        """Expected station metadata dictionary."""
        return {
            "acquired_by.author": "",
            "channels_recorded": ["ey", "hy"],
            "data_type": "BBMT",
            "fdsn.id": "CAS04",
            "geographic_name": "Corral Hollow, CA, USA",
            "hdf5_reference": "<HDF5 object reference>",
            "id": "CAS04",
            "location.declination.model": "IGRF",
            "location.declination.value": 0.0,
            "location.elevation": 329.3875,
            "location.latitude": 37.633351,
            "location.longitude": -121.468382,
            "mth5_type": "Station",
            "orientation.method": "compass",
            "orientation.reference_frame": "geographic",
            "provenance.software.author": "",
            "provenance.software.name": "",
            "provenance.software.version": "",
            "provenance.submitter.author": "",
            "provenance.submitter.email": None,
            "provenance.submitter.organization": None,
            "release_license": None,
            "run_list": ["001"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

    def test_station_metadata_attributes(self, station_cas04, station_metadata_dict):
        """Test station metadata attributes match expected values."""
        m_station = station_cas04.metadata

        for key, expected_value in station_metadata_dict.items():
            actual_value = m_station.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Station metadata '{key}': expected {expected_value}, got {actual_value}"


class TestRunMetadata:
    """Test run-level metadata."""

    @pytest.fixture
    def run_metadata_dict(self):
        """Expected run metadata dictionary."""
        return {
            "id": "001",
            "channels_recorded_electric": ["ey"],
            "channels_recorded_magnetic": ["hy"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

    def test_run_metadata_attributes(self, run_001, run_metadata_dict):
        """Test run metadata attributes match expected values."""
        m_run = run_001.metadata

        for key, expected_value in run_metadata_dict.items():
            actual_value = m_run.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Run metadata '{key}': expected {expected_value}, got {actual_value}"


class TestElectricChannelMetadata:
    """Test electric channel (ey) metadata."""

    @pytest.fixture
    def ey_metadata_dict(self):
        """Expected ey channel metadata dictionary."""
        return {
            "component": "ey",
            "positive.id": "200402F",
            "positive.manufacturer": "Oregon State University",
            "positive.type": "electrode",
            "positive.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
            "negative.id": "2004020",
            "negative.manufacturer": "Oregon State University",
            "negative.type": "electrode",
            "negative.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
            "dipole_length": 92.0,
            "measurement_azimuth": 103.2,
            "type": "electric",
            "units": "digital counts",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

    def test_ey_metadata_attributes(self, ey_channel, ey_metadata_dict):
        """Test ey channel metadata attributes match expected values."""
        m_ch = ey_channel.metadata

        for key, expected_value in ey_metadata_dict.items():
            actual_value = m_ch.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"EY channel metadata '{key}': expected {expected_value}, got {actual_value}"


class TestMagneticChannelMetadata:
    """Test magnetic channel (hy) metadata."""

    @pytest.fixture
    def hy_metadata_dict(self):
        """Expected hy channel metadata dictionary."""
        return {
            "component": "hy",
            "measurement_azimuth": 103.2,
            "type": "magnetic",
            "units": "digital counts",
            "sensor.manufacturer": "Barry Narod",
            "sensor.model": "fluxgate NIMS",
            "sensor.type": "Magnetometer",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

    def test_hy_metadata_attributes(self, hy_channel, hy_metadata_dict):
        """Test hy channel metadata attributes match expected values."""
        m_ch = hy_channel.metadata

        for key, expected_value in hy_metadata_dict.items():
            actual_value = m_ch.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"HY channel metadata '{key}': expected {expected_value}, got {actual_value}"


class TestFilters:
    """Test filter functionality and data integrity."""

    def test_all_filters(self, mth5_file, experiment):
        """Test that all experiment filters are properly transferred to MTH5."""
        experiment_filters = experiment.surveys[0].filters

        for filter_name in experiment_filters.keys():
            # Get the filter from both sources
            exp_filter = experiment_filters[filter_name]
            h5_filter = mth5_file.survey_group.filters_group.to_filter_object(
                filter_name
            )

            # Test that both exist and are truthy
            assert (
                exp_filter
            ), f"Experiment filter '{filter_name}' should exist and be truthy"
            assert h5_filter, f"MTH5 filter '{filter_name}' should exist and be truthy"

            # Additional validation could be added here for specific filter properties
            # if needed, such as comparing filter types, parameters, etc.


# Performance optimizations implemented:
# 1. Session-scoped fixtures for expensive operations (file creation, experiment loading)
# 2. Parametrized tests to reduce code duplication for similar test cases
# 3. Strategic fixture scoping to minimize resource creation
# 4. Clear test organization with descriptive class names
# 5. Meaningful assertion messages for better debugging
# 6. Automatic cleanup with yield fixtures
# 7. Separated concerns into logical test classes
# 8. Focused fixture creation - only create objects when needed
# 9. Efficient metadata testing through dictionary-driven approaches
#
# Performance improvements over unittest version:
# - Single file creation per test session (vs per test class)
# - Shared fixtures reduce redundant object creation
# - Parametrized tests run efficiently with clear test names
# - Better resource management with automatic cleanup
# - More granular test organization for easier maintenance
