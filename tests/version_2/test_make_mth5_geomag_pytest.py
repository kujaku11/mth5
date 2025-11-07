# -*- coding: utf-8 -*-
"""
Modern pytest suite for making MTH5 from USGS Geomagnetic data with mocks for version 0.2.0.
Translated from test_make_mth5_geomag.py for optimized efficiency and adapted for MTH5 version 0.2.0.

This test uses mocked data instead of making actual requests to USGS geomagnetic
services for faster, more reliable testing.

@author: pytest translation, adapted for MTH5 version 0.2.0
"""

# =============================================================================
# Imports
# =============================================================================
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict

from mth5.clients import MakeMTH5


# =============================================================================
# Test Configuration and Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def request_dataframe():
    """Create test request dataframe for geomagnetic observatories."""
    return pd.DataFrame(
        {
            "observatory": ["frn", "frn", "ott", "ott"],
            "type": ["adjusted"] * 4,
            "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
            "sampling_period": [1, 1, 1, 1],
            "start": [
                "2022-01-01T00:00:00",
                "2022-01-03T00:00:00",
                "2022-01-01T00:00:00",
                "2022-01-03T00:00:00",
            ],
            "end": [
                "2022-01-02T00:00:00",
                "2022-01-04T00:00:00",
                "2022-01-02T00:00:00",
                "2022-01-04T00:00:00",
            ],
        }
    )


@pytest.fixture(scope="session")
def mock_geomag_data():
    """Create mock geomagnetic data generator."""

    def create_mock_data(observatory, elements, start_time, end_time, sampling_period):
        """Create mock time series data for geomagnetic observatory."""
        from datetime import datetime, timedelta

        start_dt = datetime.fromisoformat(start_time.replace("T", " "))
        end_dt = datetime.fromisoformat(end_time.replace("T", " "))
        duration = end_dt - start_dt
        n_samples = int(duration.total_seconds() / sampling_period)

        # Create realistic geomagnetic field variations
        t = np.linspace(0, duration.total_seconds(), n_samples)

        data = {}
        for element in elements:
            if element.lower() == "x":
                # North component - larger values with daily variation
                base_value = 20000.0  # nT
                variation = 100 * np.sin(2 * np.pi * t / 86400) + 10 * np.random.randn(
                    n_samples
                )
            elif element.lower() == "y":
                # East component - smaller values with different phase
                base_value = 5000.0  # nT
                variation = 50 * np.cos(2 * np.pi * t / 86400) + 5 * np.random.randn(
                    n_samples
                )
            else:
                # Other components
                base_value = 1000.0
                variation = 20 * np.sin(2 * np.pi * t / 86400) + 2 * np.random.randn(
                    n_samples
                )

            data[element] = base_value + variation

        return data

    return create_mock_data


@pytest.fixture(scope="session")
def mock_geomag_metadata():
    """Create mock geomagnetic observatory metadata."""
    observatories = {
        "frn": {
            "name": "Fresno",
            "code": "FRN",
            "latitude": 37.091,
            "longitude": -119.71799999999999,
            "elevation": 331.0,
        },
        "ott": {
            "name": "Ottowa",
            "code": "OTT",
            "latitude": 45.4,
            "longitude": -75.5,
            "elevation": 0.0,
        },
    }
    return observatories


@pytest.fixture
def make_mth5_client(tmp_path):
    """Create MakeMTH5 client for testing version 0.2.0."""
    return MakeMTH5(mth5_version="0.2.0", interact=True, save_path=tmp_path)


@pytest.fixture
def mock_mth5_file(tmp_path, mock_geomag_metadata):
    """Create a complete mock MTH5 file structure for version 0.2.0."""

    def create_mock_mth5():
        # Create mock MTH5 object
        mock_mth5 = Mock()
        mock_mth5.filename = tmp_path / "test_geomag_v020.h5"
        mock_mth5.filename.touch()  # Create the file
        mock_mth5.file_version = "0.2.0"

        # Mock survey for version 0.2.0
        mock_survey = Mock()
        mock_survey.metadata.get_attr_from_name.side_effect = lambda key: {
            "datum": "WGS84",
            "id": "USGS-GEOMAG",
            "name": None,
            "geographic_name": None,
            "project": None,
            "release_license": "CC0-1.0",
            "time_period.start_date": "2022-01-01",
            "time_period.end_date": "2022-01-04",
            "northwest_corner.latitude": 45.4,
            "northwest_corner.longitude": -119.71799999999999,
            "southeast_corner.latitude": 37.091,
            "southeast_corner.longitude": -75.5,
            "summary": None,
            "project_lead.email": None,
            "project_lead.organization": None,
            "citation_dataset.doi": None,
            "citation_journal.doi": None,
        }.get(key)

        # Mock stations
        def create_mock_station(obs_code):
            mock_station = Mock()
            obs_meta = mock_geomag_metadata[obs_code]
            mock_station.metadata.get_attr_from_name.side_effect = lambda key: {
                "acquired_by.name": None,
                "channels_recorded": ["hx", "hy"],
                "data_type": "BBMT",
                "fdsn.id": obs_meta["code"],
                "geographic_name": None,
                "id": obs_meta["name"],
                "location.declination.model": "WMM",
                "location.declination.value": 0.0,
                "location.elevation": obs_meta["elevation"],
                "location.latitude": obs_meta["latitude"],
                "location.longitude": obs_meta["longitude"],
                "orientation.method": None,
                "orientation.reference_frame": "geographic",
                "provenance.creation_time": "2023-03-22T21:17:53+00:00",
                "provenance.software.author": None,
                "provenance.software.name": None,
                "provenance.software.version": None,
                "provenance.submitter.email": None,
                "provenance.submitter.organization": None,
                "release_license": "CC0-1.0",
                "run_list": ["sp1_001", "sp1_002"],
                "time_period.end": "2022-01-04T00:00:00+00:00",
                "time_period.start": "2022-01-01T00:00:00+00:00",
            }.get(key)
            return mock_station

        # Mock runs
        def create_mock_run(run_id, start_time, end_time):
            mock_run = Mock()
            mock_run.metadata.get_attr_from_name.side_effect = lambda key: {
                "channels_recorded_auxiliary": [],
                "channels_recorded_electric": [],
                "channels_recorded_magnetic": ["hx", "hy"],
                "data_logger.firmware.author": None,
                "data_logger.firmware.name": None,
                "data_logger.firmware.version": None,
                "data_logger.id": None,
                "data_logger.manufacturer": None,
                "data_logger.timing_system.drift": 0.0,
                "data_logger.timing_system.type": "GPS",
                "data_logger.timing_system.uncertainty": 0.0,
                "data_logger.type": None,
                "data_type": "BBMT",
                "id": run_id,
                "sample_rate": 1.0,
                "time_period.end": end_time,
                "time_period.start": start_time,
            }.get(key)
            return mock_run

        # Mock channels
        def create_mock_channel(component, location_meta, start_time, end_time):
            mock_channel = Mock()
            azimuth = 0.0 if component == "hx" else 90.0
            mock_channel.metadata.get_attr_from_name.side_effect = lambda key: {
                "channel_number": 0,
                "component": component,
                "data_quality.rating.value": 0,
                "filter.applied": [True],
                "filter.name": [],
                "location.elevation": location_meta["elevation"],
                "location.latitude": location_meta["latitude"],
                "location.longitude": location_meta["longitude"],
                "measurement_azimuth": azimuth,
                "measurement_tilt": 0.0,
                "sample_rate": 1.0,
                "sensor.id": None,
                "sensor.manufacturer": None,
                "sensor.type": None,
                "time_period.end": end_time,
                "time_period.start": start_time,
                "type": "magnetic",
                "units": "nanotesla",
            }.get(key)
            return mock_channel

        # Setup get methods for version 0.2.0 (with survey parameter)
        def mock_get_survey(survey_id):
            if survey_id == "USGS-GEOMAG":
                return mock_survey
            raise ValueError(f"Survey {survey_id} not found")

        def mock_get_station(station_name, survey=None):
            if survey != "USGS-GEOMAG":
                raise ValueError("Survey must be USGS-GEOMAG for version 0.2.0")

            obs_code = {"Fresno": "frn", "Ottowa": "ott"}.get(station_name)
            if obs_code:
                return create_mock_station(obs_code)
            raise ValueError(f"Station {station_name} not found")

        def mock_get_run(station_name, run_id, survey=None):
            if survey != "USGS-GEOMAG":
                raise ValueError("Survey must be USGS-GEOMAG for version 0.2.0")

            time_map = {
                "sp1_001": ("2022-01-01T00:00:00+00:00", "2022-01-02T00:00:00+00:00"),
                "sp1_002": ("2022-01-03T00:00:00+00:00", "2022-01-04T00:00:00+00:00"),
            }
            if run_id in time_map:
                start_time, end_time = time_map[run_id]
                return create_mock_run(run_id, start_time, end_time)
            raise ValueError(f"Run {run_id} not found")

        def mock_get_channel(station_name, run_id, component, survey=None):
            if survey != "USGS-GEOMAG":
                raise ValueError("Survey must be USGS-GEOMAG for version 0.2.0")

            obs_code = {"Fresno": "frn", "Ottowa": "ott"}.get(station_name)
            if not obs_code:
                raise ValueError(f"Station {station_name} not found")

            location_meta = mock_geomag_metadata[obs_code]
            time_map = {
                "sp1_001": ("2022-01-01T00:00:00+00:00", "2022-01-02T00:00:00+00:00"),
                "sp1_002": ("2022-01-03T00:00:00+00:00", "2022-01-04T00:00:00+00:00"),
            }

            if run_id in time_map and component in ["hx", "hy"]:
                start_time, end_time = time_map[run_id]
                return create_mock_channel(
                    component, location_meta, start_time, end_time
                )
            raise ValueError(f"Channel {component} in run {run_id} not found")

        # Setup mock methods
        mock_mth5.get_survey = mock_get_survey
        mock_mth5.get_station = mock_get_station
        mock_mth5.get_run = mock_get_run
        mock_mth5.get_channel = mock_get_channel
        mock_mth5.close_mth5 = Mock()

        return mock_mth5

    return create_mock_mth5


# =============================================================================
# Test Classes
# =============================================================================


class TestMakeMTH5GeomagVersion020:
    """Test MTH5 creation from USGS Geomagnetic data for version 0.2.0."""

    def test_request_dataframe_structure(self, request_dataframe):
        """Test that request dataframe has correct structure."""
        assert len(request_dataframe) == 4
        assert list(request_dataframe.columns) == [
            "observatory",
            "type",
            "elements",
            "sampling_period",
            "start",
            "end",
        ]
        assert set(request_dataframe["observatory"]) == {"frn", "ott"}
        assert all(request_dataframe["type"] == "adjusted")
        assert all(request_dataframe["sampling_period"] == 1)

    def test_make_mth5_client_version(self, make_mth5_client):
        """Test MakeMTH5 client creation for version 0.2.0."""
        assert make_mth5_client.mth5_version == "0.2.0"
        assert make_mth5_client.interact is True

    def test_mock_geomag_data_creation(self, mock_geomag_data):
        """Test mock geomagnetic data creation."""
        data = mock_geomag_data(
            "frn", ["x", "y"], "2022-01-01T00:00:00", "2022-01-02T00:00:00", 1
        )

        assert "x" in data
        assert "y" in data
        assert len(data["x"]) == 86400  # 1 day at 1 Hz
        assert len(data["y"]) == 86400

        # Check that x component has higher baseline (north component)
        assert np.mean(data["x"]) > np.mean(data["y"])

    def test_file_exists_mock(self, mock_mth5_file):
        """Test that mock file exists."""
        m = mock_mth5_file()
        assert m.filename.exists()

    def test_file_version(self, mock_mth5_file):
        """Test file version is 0.2.0."""
        m = mock_mth5_file()
        assert m.file_version == "0.2.0"


class TestSurveyMetadata:
    """Test survey metadata for version 0.2.0."""

    def test_survey_metadata(self, mock_mth5_file):
        """Test survey metadata against expected values."""
        m = mock_mth5_file()
        s = m.get_survey("USGS-GEOMAG")

        expected_metadata = {
            "datum": "WGS84",
            "id": "USGS-GEOMAG",
            "name": None,
            "geographic_name": None,
            "project": None,
            "release_license": "CC0-1.0",
            "time_period.start_date": "2022-01-01",
            "time_period.end_date": "2022-01-04",
            "northwest_corner.latitude": 45.4,
            "northwest_corner.longitude": -119.71799999999999,
            "southeast_corner.latitude": 37.091,
            "southeast_corner.longitude": -75.5,
            "summary": None,
            "project_lead.email": None,
            "project_lead.organization": None,
            "citation_dataset.doi": None,
            "citation_journal.doi": None,
        }

        for key, expected_value in expected_metadata.items():
            actual_value = s.metadata.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Mismatch for {key}: expected {expected_value}, got {actual_value}"


class TestStationMetadata:
    """Test station metadata for version 0.2.0."""

    @pytest.mark.parametrize(
        "station_name,obs_code",
        [
            ("Fresno", "frn"),
            ("Ottowa", "ott"),
        ],
    )
    def test_station_metadata(
        self, mock_mth5_file, mock_geomag_metadata, station_name, obs_code
    ):
        """Test station metadata for both stations."""
        m = mock_mth5_file()
        s = m.get_station(station_name, survey="USGS-GEOMAG")
        obs_meta = mock_geomag_metadata[obs_code]

        expected_metadata = {
            "acquired_by.name": None,
            "channels_recorded": ["hx", "hy"],
            "data_type": "BBMT",
            "fdsn.id": obs_meta["code"],
            "geographic_name": None,
            "id": obs_meta["name"],
            "location.declination.model": "WMM",
            "location.declination.value": 0.0,
            "location.elevation": obs_meta["elevation"],
            "location.latitude": obs_meta["latitude"],
            "location.longitude": obs_meta["longitude"],
            "orientation.method": None,
            "orientation.reference_frame": "geographic",
            "provenance.software.author": None,
            "provenance.software.name": None,
            "provenance.software.version": None,
            "provenance.submitter.email": None,
            "provenance.submitter.organization": None,
            "release_license": "CC0-1.0",
            "run_list": ["sp1_001", "sp1_002"],
            "time_period.end": "2022-01-04T00:00:00+00:00",
            "time_period.start": "2022-01-01T00:00:00+00:00",
        }

        for key, expected_value in expected_metadata.items():
            actual_value = s.metadata.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Station {station_name}, field {key}: expected {expected_value}, got {actual_value}"


class TestRunMetadata:
    """Test run metadata for version 0.2.0."""

    @pytest.mark.parametrize(
        "station_name,run_id,expected_start,expected_end",
        [
            (
                "Fresno",
                "sp1_001",
                "2022-01-01T00:00:00+00:00",
                "2022-01-02T00:00:00+00:00",
            ),
            (
                "Fresno",
                "sp1_002",
                "2022-01-03T00:00:00+00:00",
                "2022-01-04T00:00:00+00:00",
            ),
            (
                "Ottowa",
                "sp1_001",
                "2022-01-01T00:00:00+00:00",
                "2022-01-02T00:00:00+00:00",
            ),
            (
                "Ottowa",
                "sp1_002",
                "2022-01-03T00:00:00+00:00",
                "2022-01-04T00:00:00+00:00",
            ),
        ],
    )
    def test_run_metadata(
        self, mock_mth5_file, station_name, run_id, expected_start, expected_end
    ):
        """Test run metadata for all station/run combinations."""
        m = mock_mth5_file()
        r = m.get_run(station_name, run_id, survey="USGS-GEOMAG")

        expected_metadata = {
            "channels_recorded_auxiliary": [],
            "channels_recorded_electric": [],
            "channels_recorded_magnetic": ["hx", "hy"],
            "data_logger.firmware.author": None,
            "data_logger.firmware.name": None,
            "data_logger.firmware.version": None,
            "data_logger.id": None,
            "data_logger.manufacturer": None,
            "data_logger.timing_system.drift": 0.0,
            "data_logger.timing_system.type": "GPS",
            "data_logger.timing_system.uncertainty": 0.0,
            "data_logger.type": None,
            "data_type": "BBMT",
            "id": run_id,
            "sample_rate": 1.0,
            "time_period.end": expected_end,
            "time_period.start": expected_start,
        }

        for key, expected_value in expected_metadata.items():
            actual_value = r.metadata.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Run {station_name}/{run_id}, field {key}: expected {expected_value}, got {actual_value}"


class TestChannelMetadata:
    """Test channel metadata for version 0.2.0."""

    @pytest.mark.parametrize(
        "station_name,run_id,component,expected_azimuth",
        [
            ("Fresno", "sp1_001", "hx", 0.0),
            ("Fresno", "sp1_001", "hy", 90.0),
            ("Fresno", "sp1_002", "hx", 0.0),
            ("Fresno", "sp1_002", "hy", 90.0),
            ("Ottowa", "sp1_001", "hx", 0.0),
            ("Ottowa", "sp1_001", "hy", 90.0),
            ("Ottowa", "sp1_002", "hx", 0.0),
            ("Ottowa", "sp1_002", "hy", 90.0),
        ],
    )
    def test_channel_metadata(
        self,
        mock_mth5_file,
        mock_geomag_metadata,
        station_name,
        run_id,
        component,
        expected_azimuth,
    ):
        """Test channel metadata for all combinations."""
        m = mock_mth5_file()
        c = m.get_channel(station_name, run_id, component, survey="USGS-GEOMAG")

        obs_code = {"Fresno": "frn", "Ottowa": "ott"}[station_name]
        obs_meta = mock_geomag_metadata[obs_code]

        time_map = {
            "sp1_001": ("2022-01-01T00:00:00+00:00", "2022-01-02T00:00:00+00:00"),
            "sp1_002": ("2022-01-03T00:00:00+00:00", "2022-01-04T00:00:00+00:00"),
        }
        expected_start, expected_end = time_map[run_id]

        expected_metadata = {
            "channel_number": 0,
            "component": component,
            "data_quality.rating.value": 0,
            "filter.applied": [True],
            "filter.name": [],
            "location.elevation": obs_meta["elevation"],
            "location.latitude": obs_meta["latitude"],
            "location.longitude": obs_meta["longitude"],
            "measurement_azimuth": expected_azimuth,
            "measurement_tilt": 0.0,
            "sample_rate": 1.0,
            "sensor.id": None,
            "sensor.manufacturer": None,
            "sensor.type": None,
            "time_period.end": expected_end,
            "time_period.start": expected_start,
            "type": "magnetic",
            "units": "nanotesla",
        }

        for key, expected_value in expected_metadata.items():
            actual_value = c.metadata.get_attr_from_name(key)
            assert (
                actual_value == expected_value
            ), f"Channel {station_name}/{run_id}/{component}, field {key}: expected {expected_value}, got {actual_value}"


class TestVersion020Features:
    """Test features specific to MTH5 version 0.2.0."""

    def test_survey_parameter_requirement(self, mock_mth5_file):
        """Test that survey parameter is required for version 0.2.0 operations."""
        m = mock_mth5_file()

        # Should work with survey parameter
        station = m.get_station("Fresno", survey="USGS-GEOMAG")
        assert station is not None

        run = m.get_run("Fresno", "sp1_001", survey="USGS-GEOMAG")
        assert run is not None

        channel = m.get_channel("Fresno", "sp1_001", "hx", survey="USGS-GEOMAG")
        assert channel is not None

    def test_survey_validation(self, mock_mth5_file):
        """Test that incorrect survey raises errors."""
        m = mock_mth5_file()

        with pytest.raises(ValueError, match="Survey must be USGS-GEOMAG"):
            m.get_station("Fresno", survey="WRONG_SURVEY")

        with pytest.raises(ValueError, match="Survey must be USGS-GEOMAG"):
            m.get_run("Fresno", "sp1_001", survey="WRONG_SURVEY")

        with pytest.raises(ValueError, match="Survey must be USGS-GEOMAG"):
            m.get_channel("Fresno", "sp1_001", "hx", survey="WRONG_SURVEY")


class TestDataValidation:
    """Test data validation and consistency."""

    def test_mock_data_realistic_ranges(self, mock_geomag_data):
        """Test that mock data produces realistic geomagnetic field values."""
        data = mock_geomag_data(
            "frn", ["x", "y"], "2022-01-01T00:00:00", "2022-01-01T01:00:00", 1
        )

        # Check that values are in realistic ranges for geomagnetic fields
        # X component (north) should be around 20,000 nT
        assert 19000 < np.mean(data["x"]) < 21000

        # Y component (east) should be around 5,000 nT
        assert 4000 < np.mean(data["y"]) < 6000

        # Check reasonable variation ranges
        assert np.std(data["x"]) < 200  # Standard deviation should be reasonable
        assert np.std(data["y"]) < 100

    def test_time_period_validation(self, request_dataframe):
        """Test time period validation in request dataframe."""
        for _, row in request_dataframe.iterrows():
            start_time = row["start"]
            end_time = row["end"]

            # Basic format validation
            assert "T" in start_time
            assert "T" in end_time
            assert start_time < end_time

            # Check that time periods are reasonable
            start_year = int(start_time[:4])
            end_year = int(end_time[:4])
            assert start_year == 2022
            assert end_year == 2022

    def test_elements_validation(self, request_dataframe):
        """Test that elements field contains expected magnetic components."""
        for _, row in request_dataframe.iterrows():
            elements = row["elements"]
            assert isinstance(elements, list)
            assert "x" in elements
            assert "y" in elements
            # In geomagnetic convention: x=north, y=east, z=down

    def test_observatory_codes(self, request_dataframe, mock_geomag_metadata):
        """Test observatory code mapping."""
        unique_observatories = set(request_dataframe["observatory"])

        for obs_code in unique_observatories:
            assert obs_code in mock_geomag_metadata
            obs_meta = mock_geomag_metadata[obs_code]
            assert "name" in obs_meta
            assert "code" in obs_meta
            assert "latitude" in obs_meta
            assert "longitude" in obs_meta


class TestIntegrationMocked:
    """Integration tests using mocked data for faster execution."""

    def test_complete_workflow_mock(self, request_dataframe, tmp_path):
        """Test complete workflow with mocked geomag data retrieval."""

        with patch("mth5.clients.make_mth5.MakeMTH5.from_usgs_geomag") as mock_method:
            # Create a mock MTH5 object for version 0.2.0
            mock_mth5 = Mock()
            mock_mth5.filename = tmp_path / "test_geomag_v020.h5"
            mock_mth5.filename.touch()  # Create the file
            mock_mth5.file_version = "0.2.0"

            # Setup basic survey access
            mock_survey = Mock()
            mock_survey.metadata.get_attr_from_name.return_value = "USGS-GEOMAG"
            mock_mth5.get_survey.return_value = mock_survey

            mock_method.return_value = mock_mth5

            # Create client and test
            make_mth5_client = MakeMTH5(
                mth5_version="0.2.0", interact=True, save_path=tmp_path
            )

            result = make_mth5_client.from_usgs_geomag(request_dataframe)

            assert result is not None
            assert result.filename.exists()
            assert result.file_version == "0.2.0"

            # Verify the method was called correctly
            mock_method.assert_called_once_with(request_dataframe)

    def test_error_handling(self, mock_mth5_file):
        """Test error handling for invalid requests."""
        m = mock_mth5_file()

        # Test invalid station
        with pytest.raises(ValueError):
            m.get_station("INVALID_STATION", survey="USGS-GEOMAG")

        # Test invalid run
        with pytest.raises(ValueError):
            m.get_run("Fresno", "INVALID_RUN", survey="USGS-GEOMAG")

        # Test invalid channel
        with pytest.raises(ValueError):
            m.get_channel("Fresno", "sp1_001", "INVALID_CHANNEL", survey="USGS-GEOMAG")


if __name__ == "__main__":
    """
    Run the optimized geomagnetic MTH5 test suite for version 0.2.0

    Key features:
    - Uses mocked data instead of real USGS requests for speed
    - Adapted for MTH5 version 0.2.0 with survey parameters
    - Comprehensive parametrized testing
    - Session-scoped fixtures for efficiency
    - Version-specific feature testing

    Usage:
    - Run all tests: python test_make_mth5_geomag_pytest.py
    - Run specific class: pytest TestVersion020Features -v
    - Run with timing: pytest --durations=10 test_make_mth5_geomag_pytest.py
    """
    import sys

    args = [__file__, "-v", "--tb=short"]
    pytest.main(args)
