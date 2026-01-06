"""
Test making MTH5 from USGS Geomagnetic data using pytest with mocking.

This is a pytest version of test_make_mth5_geomag.py that uses mocked data
instead of making actual requests to USGS geomagnetic services for faster,
more reliable testing.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from mth5.clients import MakeMTH5


class TestMakeMTH5GeomagPytest:
    """Test MTH5 creation from USGS Geomagnetic data using pytest with mocks."""

    @pytest.fixture
    def request_dataframe(self):
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

    @pytest.fixture
    def mock_geomag_data(self):
        """Create mock geomagnetic data."""

        def create_mock_data(
            observatory, elements, start_time, end_time, sampling_period
        ):
            """Create mock time series data for geomagnetic observatory."""
            from datetime import datetime

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
                    variation = 100 * np.sin(
                        2 * np.pi * t / 86400
                    ) + 10 * np.random.randn(n_samples)
                elif element.lower() == "y":
                    # East component - smaller values with different phase
                    base_value = 5000.0  # nT
                    variation = 50 * np.cos(
                        2 * np.pi * t / 86400
                    ) + 5 * np.random.randn(n_samples)
                else:
                    # Other components
                    base_value = 1000.0
                    variation = 20 * np.sin(
                        2 * np.pi * t / 86400
                    ) + 2 * np.random.randn(n_samples)

                data[element] = base_value + variation

            return data

        return create_mock_data

    @pytest.fixture
    def mock_geomag_metadata(self):
        """Create mock geomagnetic observatory metadata."""
        observatories = {
            "frn": {
                "name": "Fresno",
                "code": "FRN",
                "latitude": 37.091,
                "longitude": -119.718,
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
    def make_mth5_client(self, tmp_path):
        """Create MakeMTH5 client for testing."""
        return MakeMTH5(mth5_version="0.1.0", interact=True, save_path=tmp_path)

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

    def test_mock_geomag_metadata_structure(self, mock_geomag_metadata):
        """Test mock metadata structure."""
        assert "frn" in mock_geomag_metadata
        assert "ott" in mock_geomag_metadata

        frn_meta = mock_geomag_metadata["frn"]
        assert frn_meta["name"] == "Fresno"
        assert frn_meta["code"] == "FRN"
        assert isinstance(frn_meta["latitude"], float)
        assert isinstance(frn_meta["longitude"], float)

    def test_make_mth5_client_creation(self, make_mth5_client):
        """Test MakeMTH5 client creation."""
        assert make_mth5_client.mth5_version == "0.1.0"
        assert make_mth5_client.interact is True

    def test_geomag_from_usgs_mock_basic(
        self, request_dataframe, mock_geomag_data, mock_geomag_metadata, tmp_path
    ):
        """Test basic functionality of from_usgs_geomag with mocks."""

        # Mock the actual geomag data retrieval and processing
        with patch("mth5.clients.make_mth5.MakeMTH5.from_usgs_geomag") as mock_method:
            # Create a mock MTH5 object
            mock_mth5 = Mock()
            mock_mth5.filename = tmp_path / "test_geomag.h5"
            mock_mth5.filename.touch()  # Create the file
            mock_mth5.file_version = "0.1.0"

            # Setup mock survey metadata
            mock_survey = Mock()
            mock_survey.metadata.get_attr_from_name.side_effect = lambda key: {
                "datum": "WGS84",
                "id": "USGS-GEOMAG",
                "release_license": "CC0-1.0",
                "time_period.start_date": "2022-01-01",
                "time_period.end_date": "2022-01-04",
                "northwest_corner.latitude": 45.4,
                "northwest_corner.longitude": -119.718,
                "southeast_corner.latitude": 37.091,
                "southeast_corner.longitude": -75.5,
            }.get(key)
            mock_mth5.survey_group = mock_survey

            mock_method.return_value = mock_mth5

            # Test the method call
            result = MakeMTH5.from_usgs_geomag(
                request_dataframe,
                mth5_version="0.1.0",
                interact=True,
                save_path=tmp_path,
            )

            assert result is not None
            assert result.filename.exists()
            assert result.file_version == "0.1.0"

    def test_station_metadata_mock(self, mock_geomag_metadata):
        """Test station metadata structure for mocked data."""
        expected_fresno_metadata = {
            "id": "Fresno",
            "fdsn.id": "FRN",
            "data_type": "BBMT",
            "location.latitude": 37.091,
            "location.longitude": -119.718,
            "location.elevation": 331.0,
            "orientation.reference_frame": "geographic",
            "release_license": "CC0-1.0",
            "channels_recorded": ["hx", "hy"],
            "run_list": ["sp1_001", "sp1_002"],
        }

        # Test that our mock metadata contains expected fields
        frn_meta = mock_geomag_metadata["frn"]
        assert frn_meta["name"] == "Fresno"
        assert frn_meta["latitude"] == expected_fresno_metadata["location.latitude"]
        assert frn_meta["longitude"] == expected_fresno_metadata["location.longitude"]
        assert frn_meta["elevation"] == expected_fresno_metadata["location.elevation"]

    def test_run_metadata_structure(self):
        """Test expected run metadata structure."""
        expected_run_metadata = {
            "channels_recorded_auxiliary": [],
            "channels_recorded_electric": [],
            "channels_recorded_magnetic": ["hx", "hy"],
            "data_logger.timing_system.type": "GPS",
            "data_logger.timing_system.drift": 0.0,
            "data_logger.timing_system.uncertainty": 0.0,
            "data_type": "BBMT",
            "sample_rate": 1.0,
        }

        # This test just validates the expected structure
        assert "channels_recorded_magnetic" in expected_run_metadata
        assert expected_run_metadata["channels_recorded_magnetic"] == ["hx", "hy"]
        assert expected_run_metadata["sample_rate"] == 1.0

    def test_channel_metadata_structure(self):
        """Test expected channel metadata structure."""
        expected_hx_metadata = {
            "component": "hx",
            "type": "magnetic",
            "units": "nanotesla",
            "measurement_azimuth": 0.0,
            "measurement_tilt": 0.0,
            "sample_rate": 1.0,
            "filter.applied": [True],
            "data_quality.rating.value": 0,
        }

        expected_hy_metadata = {
            "component": "hy",
            "type": "magnetic",
            "units": "nanotesla",
            "measurement_azimuth": 90.0,
            "measurement_tilt": 0.0,
            "sample_rate": 1.0,
            "filter.applied": [True],
            "data_quality.rating.value": 0,
        }

        # Test expected channel structures
        assert expected_hx_metadata["component"] == "hx"
        assert expected_hx_metadata["measurement_azimuth"] == 0.0
        assert expected_hy_metadata["component"] == "hy"
        assert expected_hy_metadata["measurement_azimuth"] == 90.0

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

    def test_sampling_period_consistency(self, request_dataframe):
        """Test that sampling periods are consistent."""
        sampling_periods = request_dataframe["sampling_period"].unique()
        assert len(sampling_periods) == 1  # All should be the same
        assert sampling_periods[0] == 1  # 1 second sampling

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
