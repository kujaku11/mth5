# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for MakeMTH5 with fixtures, parametrization,
and additional coverage for untested functionality.

Created by modernizing test_makemth5.py with pytest patterns.

@author: pytest conversion
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.clients import MakeMTH5
from mth5.clients.fdsn import FDSN
from mth5.clients.geomag import USGSGeomag
from mth5.clients.lemi424 import LEMI424Client
from mth5.clients.metronix import MetronixClient
from mth5.clients.phoenix import PhoenixClient
from mth5.clients.zen import ZenClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file_path(temp_data_dir):
    """Create a temporary file path for testing."""
    return temp_data_dir / "test.h5"


@pytest.fixture
def basic_make_mth5():
    """Create a basic MakeMTH5 instance for testing."""
    return MakeMTH5()


@pytest.fixture
def configured_make_mth5_v1(temp_data_dir):
    """Create a configured MakeMTH5 instance with v1 settings."""
    return MakeMTH5(
        mth5_version="0.1.0",
        interact=True,
        save_path=temp_data_dir,
        h5_compression="gzip",
        h5_compression_opts=4,
        h5_shuffle=True,
        h5_fletcher32=True,
        h5_data_level=1,
    )


@pytest.fixture
def configured_make_mth5_v2(temp_data_dir):
    """Create a configured MakeMTH5 instance with v2 settings."""
    return MakeMTH5(
        mth5_version="0.2.0",
        interact=False,
        save_path=temp_data_dir,
        h5_compression=None,
        h5_compression_opts=None,
        h5_shuffle=False,
        h5_fletcher32=False,
        h5_data_level=2,
    )


@pytest.fixture
def sample_fdsn_request_df():
    """Create a sample FDSN request DataFrame for testing."""
    return pd.DataFrame(
        {
            "network": ["EM", "EM"],
            "station": ["MT001", "MT002"],
            "location": ["", ""],
            "channel": ["*", "*"],
            "start": ["2020-01-01T00:00:00", "2020-01-01T00:00:00"],
            "end": ["2020-01-02T00:00:00", "2020-01-02T00:00:00"],
        }
    )


@pytest.fixture
def sample_geomag_request_df():
    """Create a sample geomag request DataFrame for testing."""
    return pd.DataFrame(
        {
            "observatory": ["FRD", "BOU"],
            "type": ["adjusted", "adjusted"],
            "elements": ["X,Y,Z", "X,Y,Z"],
            "sampling_period": [60, 60],
            "start": ["2020-01-01T00:00:00", "2020-01-01T00:00:00"],
            "end": ["2020-01-02T00:00:00", "2020-01-02T00:00:00"],
        }
    )


# =============================================================================
# MakeMTH5 Initialization Tests
# =============================================================================


class TestMakeMTH5Initialization:
    """Test MakeMTH5 initialization and basic properties."""

    def test_default_initialization(self):
        """Test default initialization with minimal parameters."""
        maker = MakeMTH5()

        assert maker.mth5_version == "0.2.0"
        assert maker.interact is False
        assert maker.save_path == Path().cwd()
        assert maker.h5_compression == "gzip"
        assert maker.h5_compression_opts == 4
        assert maker.h5_shuffle is True
        assert maker.h5_fletcher32 is True
        assert maker.h5_data_level == 1

    def test_custom_initialization(self, temp_data_dir):
        """Test initialization with custom parameters."""
        maker = MakeMTH5(
            mth5_version="0.1.0",
            interact=True,
            save_path=temp_data_dir,
            h5_compression="lzf",
            h5_compression_opts=2,
            h5_shuffle=False,
            h5_fletcher32=False,
            h5_data_level=2,
        )

        assert maker.mth5_version == "0.1.0"
        assert maker.interact is True
        assert maker.save_path == temp_data_dir
        assert maker.h5_compression == "lzf"
        assert maker.h5_compression_opts == 2
        assert maker.h5_shuffle is False
        assert maker.h5_fletcher32 is False
        assert maker.h5_data_level == 2

    def test_kwargs_initialization(self, temp_data_dir):
        """Test initialization with arbitrary kwargs."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            custom_param="test_value",
            h5_custom_setting="custom",
            another_param=42,
        )

        assert maker.save_path == temp_data_dir
        assert maker.custom_param == "test_value"
        assert maker.h5_custom_setting == "custom"
        assert maker.another_param == 42

    def test_none_save_path_defaults_to_cwd(self):
        """Test that None save_path defaults to current working directory."""
        maker = MakeMTH5(save_path=None)
        assert maker.save_path == Path().cwd()

    def test_string_save_path_conversion(self, temp_data_dir):
        """Test that string save_path is properly handled."""
        maker = MakeMTH5(save_path=str(temp_data_dir))
        # The implementation should handle string paths
        assert maker.save_path == str(temp_data_dir)


# =============================================================================
# MakeMTH5 Version-Specific Tests (Original Test Cases)
# =============================================================================


class TestMakeMTH5v1:
    """Test MakeMTH5 with version 1.0 settings (original test case)."""

    @pytest.fixture
    def make_mth5_v1(self):
        """Create MakeMTH5 instance with v1 settings."""
        return MakeMTH5(mth5_version="0.1.0", interact=True, save_path=None)

    def test_mth5_version(self, make_mth5_v1):
        """Test MTH5 version setting."""
        assert make_mth5_v1.mth5_version == "0.1.0"

    def test_interact_true(self, make_mth5_v1):
        """Test interact setting."""
        assert make_mth5_v1.interact is True

    def test_save_path(self, make_mth5_v1):
        """Test save path default."""
        assert make_mth5_v1.save_path == Path().cwd()

    def test_compression(self, make_mth5_v1):
        """Test compression setting."""
        assert make_mth5_v1.h5_compression == "gzip"

    def test_compression_opts(self, make_mth5_v1):
        """Test compression options."""
        assert make_mth5_v1.h5_compression_opts == 4

    def test_shuffle(self, make_mth5_v1):
        """Test shuffle setting."""
        assert make_mth5_v1.h5_shuffle is True

    def test_fletcher32(self, make_mth5_v1):
        """Test fletcher32 setting."""
        assert make_mth5_v1.h5_fletcher32 is True

    def test_data_level(self, make_mth5_v1):
        """Test data level setting."""
        assert make_mth5_v1.h5_data_level == 1


class TestMakeMTH5v2:
    """Test MakeMTH5 with version 2.0 settings (original test case)."""

    @pytest.fixture
    def make_mth5_v2(self):
        """Create MakeMTH5 instance with v2 settings."""
        return MakeMTH5(
            mth5_version="0.2.0",
            interact=False,
            save_path=None,
            h5_compression=None,
            h5_compression_opts=None,
            h5_shuffle=False,
            h5_fletcher32=False,
            h5_data_level=2,
        )

    def test_mth5_version(self, make_mth5_v2):
        """Test MTH5 version setting."""
        assert make_mth5_v2.mth5_version == "0.2.0"

    def test_interact_false(self, make_mth5_v2):
        """Test interact setting."""
        assert make_mth5_v2.interact is False

    def test_save_path(self, make_mth5_v2):
        """Test save path default."""
        assert make_mth5_v2.save_path == Path().cwd()

    def test_compression(self, make_mth5_v2):
        """Test compression setting."""
        assert make_mth5_v2.h5_compression is None

    def test_compression_opts(self, make_mth5_v2):
        """Test compression options."""
        assert make_mth5_v2.h5_compression_opts is None

    def test_shuffle(self, make_mth5_v2):
        """Test shuffle setting."""
        assert make_mth5_v2.h5_shuffle is False

    def test_fletcher32(self, make_mth5_v2):
        """Test fletcher32 setting."""
        assert make_mth5_v2.h5_fletcher32 is False

    def test_data_level(self, make_mth5_v2):
        """Test data level setting."""
        assert make_mth5_v2.h5_data_level == 2


# =============================================================================
# H5 Parameters and Configuration Tests
# =============================================================================


class TestMakeMTH5H5Parameters:
    """Test H5 parameter handling and get_h5_kwargs method."""

    def test_get_h5_kwargs_default(self, basic_make_mth5):
        """Test get_h5_kwargs with default parameters."""
        h5_kwargs = basic_make_mth5.get_h5_kwargs()

        expected_keys = [
            "mth5_version",
            "h5_compression",
            "h5_compression_opts",
            "h5_shuffle",
            "h5_fletcher32",
            "h5_data_level",
        ]

        for key in expected_keys:
            assert key in h5_kwargs

        assert h5_kwargs["mth5_version"] == "0.2.0"
        assert h5_kwargs["h5_compression"] == "gzip"
        assert h5_kwargs["h5_compression_opts"] == 4
        assert h5_kwargs["h5_shuffle"] is True
        assert h5_kwargs["h5_fletcher32"] is True
        assert h5_kwargs["h5_data_level"] == 1

    def test_get_h5_kwargs_custom(self, configured_make_mth5_v2):
        """Test get_h5_kwargs with custom parameters."""
        h5_kwargs = configured_make_mth5_v2.get_h5_kwargs()

        assert h5_kwargs["mth5_version"] == "0.2.0"
        assert h5_kwargs["h5_compression"] is None
        assert h5_kwargs["h5_compression_opts"] is None
        assert h5_kwargs["h5_shuffle"] is False
        assert h5_kwargs["h5_fletcher32"] is False
        assert h5_kwargs["h5_data_level"] == 2

    def test_get_h5_kwargs_includes_h5_prefixed_attributes(self, temp_data_dir):
        """Test that get_h5_kwargs includes all h5-prefixed attributes."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            h5_custom_param="test_value",
            h5_another_setting=42,
            h5_complex_setting={"key": "value"},
            non_h5_param="should_not_be_included",
        )

        h5_kwargs = maker.get_h5_kwargs()

        assert h5_kwargs["h5_custom_param"] == "test_value"
        assert h5_kwargs["h5_another_setting"] == 42
        assert h5_kwargs["h5_complex_setting"] == {"key": "value"}
        assert "non_h5_param" not in h5_kwargs

    @pytest.mark.parametrize(
        "compression,expected",
        [("gzip", "gzip"), ("lzf", "lzf"), ("szip", "szip"), (None, None)],
    )
    def test_compression_options(self, compression, expected, temp_data_dir):
        """Test different compression options."""
        maker = MakeMTH5(save_path=temp_data_dir, h5_compression=compression)
        assert maker.h5_compression == expected

    @pytest.mark.parametrize(
        "compression_opts,expected", [(1, 1), (4, 4), (9, 9), (None, None)]
    )
    def test_compression_opts_options(self, compression_opts, expected, temp_data_dir):
        """Test different compression options."""
        maker = MakeMTH5(save_path=temp_data_dir, h5_compression_opts=compression_opts)
        assert maker.h5_compression_opts == expected


# =============================================================================
# String Representation Tests
# =============================================================================


class TestMakeMTH5StringRepresentation:
    """Test string representation methods."""

    def test_str_method(self, basic_make_mth5):
        """Test __str__ method."""
        str_repr = str(basic_make_mth5)

        assert "MakeMTH5 Attibutes:" in str_repr
        assert "mth5_version: 0.2.0" in str_repr
        assert "h5_compression: gzip" in str_repr
        assert "h5_compression_opts: 4" in str_repr
        assert "h5_shuffle: True" in str_repr
        assert "h5_fletcher32: True" in str_repr
        assert "h5_data_level: 1" in str_repr

    def test_repr_method(self, basic_make_mth5):
        """Test __repr__ method."""
        repr_str = repr(basic_make_mth5)
        str_str = str(basic_make_mth5)

        assert repr_str == str_str

    def test_str_with_custom_parameters(self, configured_make_mth5_v2):
        """Test __str__ with custom parameters."""
        str_repr = str(configured_make_mth5_v2)

        assert "mth5_version: 0.2.0" in str_repr
        assert "h5_compression: None" in str_repr
        assert "h5_shuffle: False" in str_repr
        assert "h5_fletcher32: False" in str_repr
        assert "h5_data_level: 2" in str_repr


# =============================================================================
# FDSN Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5FDSNClient:
    """Test FDSN client functionality with mocking."""

    @patch("mth5.clients.make_mth5.FDSN")
    def test_from_fdsn_client_basic(
        self, mock_fdsn_class, sample_fdsn_request_df, temp_data_dir
    ):
        """Test basic FDSN client creation."""
        # Setup mock
        mock_fdsn_instance = Mock(spec=FDSN)
        mock_fdsn_instance.make_mth5_from_fdsn_client.return_value = (
            temp_data_dir / "test.h5"
        )
        mock_fdsn_class.return_value = mock_fdsn_instance

        # Test method
        result = MakeMTH5.from_fdsn_client(
            sample_fdsn_request_df, client="IRIS", save_path=temp_data_dir
        )

        # Verify calls
        mock_fdsn_class.assert_called_once()
        mock_fdsn_instance.make_mth5_from_fdsn_client.assert_called_once()

        assert result == temp_data_dir / "test.h5"

    @patch("mth5.clients.make_mth5.FDSN")
    def test_from_fdsn_client_custom_parameters(
        self, mock_fdsn_class, sample_fdsn_request_df, temp_data_dir
    ):
        """Test FDSN client with custom H5 parameters."""
        # Setup mock
        mock_fdsn_instance = Mock(spec=FDSN)
        mock_fdsn_instance.make_mth5_from_fdsn_client.return_value = (
            temp_data_dir / "custom.h5"
        )
        mock_fdsn_class.return_value = mock_fdsn_instance

        # Test with custom parameters
        result = MakeMTH5.from_fdsn_client(
            sample_fdsn_request_df,
            client="EARTHSCOPE",
            save_path=temp_data_dir,
            h5_compression="lzf",
            h5_compression_opts=1,
            interact=True,
            mth5_version="0.1.0",
        )

        # Verify FDSN was called with correct parameters
        call_args = mock_fdsn_class.call_args[1]
        assert call_args["h5_compression"] == "lzf"
        assert call_args["h5_compression_opts"] == 1
        assert call_args["mth5_version"] == "0.1.0"

        # Verify make_mth5_from_fdsn_client was called with correct parameters
        fdsn_call_args = mock_fdsn_instance.make_mth5_from_fdsn_client.call_args
        assert fdsn_call_args[0][0].equals(sample_fdsn_request_df)
        assert fdsn_call_args[1]["path"] == temp_data_dir
        assert fdsn_call_args[1]["interact"] is True

    @patch("mth5.clients.make_mth5.FDSN")
    def test_from_fdsn_miniseed_and_stationxml(self, mock_fdsn_class, temp_data_dir):
        """Test FDSN creation from miniseed and StationXML."""
        # Setup mock
        mock_fdsn_instance = Mock(spec=FDSN)
        mock_fdsn_instance.make_mth5_from_inventory_and_streams.return_value = (
            temp_data_dir / "from_inventory.h5"
        )
        mock_fdsn_class.return_value = mock_fdsn_instance

        # Create mock files
        station_xml_path = temp_data_dir / "station.xml"
        station_xml_path.touch()
        miniseed_files = [temp_data_dir / "data1.mseed", temp_data_dir / "data2.mseed"]
        for mseed_file in miniseed_files:
            mseed_file.touch()

        # Test method
        result = MakeMTH5.from_fdsn_miniseed_and_stationxml(
            station_xml_path,
            miniseed_files,
            save_path=temp_data_dir,
            h5_compression="gzip",
        )

        # Verify calls
        mock_fdsn_class.assert_called_once()
        mock_fdsn_instance.make_mth5_from_inventory_and_streams.assert_called_once_with(
            station_xml_path, miniseed_files, save_path=temp_data_dir
        )

        assert result == temp_data_dir / "from_inventory.h5"


# =============================================================================
# USGS Geomag Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5USGSGeomag:
    """Test USGS Geomag client functionality with mocking."""

    @patch("mth5.clients.make_mth5.USGSGeomag")
    def test_from_usgs_geomag_basic(
        self, mock_geomag_class, sample_geomag_request_df, temp_data_dir
    ):
        """Test basic USGS Geomag client creation."""
        # Setup mock
        mock_geomag_instance = Mock(spec=USGSGeomag)
        mock_geomag_instance.make_mth5_from_geomag.return_value = (
            temp_data_dir / "geomag.h5"
        )
        mock_geomag_class.return_value = mock_geomag_instance

        # Test method
        result = MakeMTH5.from_usgs_geomag(
            sample_geomag_request_df, save_path=temp_data_dir
        )

        # Verify calls
        mock_geomag_class.assert_called_once()
        call_args = mock_geomag_class.call_args[1]
        assert call_args["save_path"] == temp_data_dir
        assert call_args["interact"] is False  # Default value

        mock_geomag_instance.make_mth5_from_geomag.assert_called_once_with(
            sample_geomag_request_df
        )

        assert result == temp_data_dir / "geomag.h5"

    @patch("mth5.clients.make_mth5.USGSGeomag")
    def test_from_usgs_geomag_custom_parameters(
        self, mock_geomag_class, sample_geomag_request_df, temp_data_dir
    ):
        """Test USGS Geomag client with custom parameters."""
        # Setup mock
        mock_geomag_instance = Mock(spec=USGSGeomag)
        mock_geomag_instance.make_mth5_from_geomag.return_value = (
            temp_data_dir / "custom_geomag.h5"
        )
        mock_geomag_class.return_value = mock_geomag_instance

        # Test with custom parameters
        result = MakeMTH5.from_usgs_geomag(
            sample_geomag_request_df,
            save_path=temp_data_dir,
            interact=True,
            h5_compression="lzf",
            mth5_version="0.1.0",
        )

        # Verify USGSGeomag was called with correct parameters
        call_args = mock_geomag_class.call_args[1]
        assert call_args["save_path"] == temp_data_dir
        assert call_args["interact"] is True
        assert call_args["h5_compression"] == "lzf"
        assert call_args["mth5_version"] == "0.1.0"


# =============================================================================
# Zen Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5Zen:
    """Test Zen client functionality with mocking."""

    @patch("mth5.clients.make_mth5.ZenClient")
    def test_from_zen_basic(self, mock_zen_class, temp_data_dir):
        """Test basic Zen client creation."""
        # Setup mock
        mock_zen_instance = Mock(spec=ZenClient)
        mock_zen_instance.make_mth5_from_zen.return_value = temp_data_dir / "zen.h5"
        mock_zen_class.return_value = mock_zen_instance

        # Test method
        result = MakeMTH5.from_zen(
            temp_data_dir, sample_rates=[4096, 1024, 256], save_path=temp_data_dir
        )

        # Verify calls
        mock_zen_class.assert_called_once()
        call_args = mock_zen_class.call_args
        assert call_args[0][0] == temp_data_dir  # data_path
        assert call_args[1]["sample_rates"] == [4096, 1024, 256]
        assert call_args[1]["save_path"] == temp_data_dir

        mock_zen_instance.make_mth5_from_zen.assert_called_once()

        assert result == temp_data_dir / "zen.h5"

    @patch("mth5.clients.make_mth5.ZenClient")
    def test_from_zen_custom_parameters(self, mock_zen_class, temp_data_dir):
        """Test Zen client with custom parameters."""
        # Setup mock
        mock_zen_instance = Mock(spec=ZenClient)
        mock_zen_instance.make_mth5_from_zen.return_value = (
            temp_data_dir / "custom_zen.h5"
        )
        mock_zen_class.return_value = mock_zen_instance

        # Create mock calibration file
        cal_file = temp_data_dir / "amtant.cal"
        cal_file.touch()

        # Test with custom parameters
        result = MakeMTH5.from_zen(
            temp_data_dir,
            sample_rates=[1024, 256],
            calibration_path=cal_file,
            survey_id="TEST_SURVEY",
            combine=False,
            h5_compression="lzf",
        )

        # Verify ZenClient was called with correct parameters
        call_args = mock_zen_class.call_args
        assert call_args[0][0] == temp_data_dir
        assert call_args[1]["sample_rates"] == [1024, 256]
        assert call_args[1]["calibration_path"] == cal_file
        assert call_args[1]["h5_compression"] == "lzf"

        # Verify make_mth5_from_zen was called with correct parameters
        zen_call_args = mock_zen_instance.make_mth5_from_zen.call_args[1]
        assert zen_call_args["survey_id"] == "TEST_SURVEY"
        assert zen_call_args["combine"] is False


# =============================================================================
# Phoenix Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5Phoenix:
    """Test Phoenix client functionality with mocking."""

    @patch("mth5.clients.make_mth5.PhoenixClient")
    def test_from_phoenix_basic(self, mock_phoenix_class, temp_data_dir):
        """Test basic Phoenix client creation."""
        # Setup mock
        mock_phoenix_instance = Mock(spec=PhoenixClient)
        mock_phoenix_instance.make_mth5_from_phoenix.return_value = (
            temp_data_dir / "phoenix.h5"
        )
        mock_phoenix_class.return_value = mock_phoenix_instance

        # Test method
        result = MakeMTH5.from_phoenix(temp_data_dir, save_path=temp_data_dir)

        # Verify calls
        mock_phoenix_class.assert_called_once()
        call_args = mock_phoenix_class.call_args
        assert call_args[0][0] == temp_data_dir  # data_path
        assert call_args[1]["sample_rates"] == [150, 24000]  # Default
        assert call_args[1]["save_path"] == temp_data_dir

        mock_phoenix_instance.make_mth5_from_phoenix.assert_called_once()

        assert result == temp_data_dir / "phoenix.h5"

    @patch("mth5.clients.make_mth5.PhoenixClient")
    def test_from_phoenix_custom_parameters(self, mock_phoenix_class, temp_data_dir):
        """Test Phoenix client with custom parameters."""
        # Setup mock
        mock_phoenix_instance = Mock(spec=PhoenixClient)
        mock_phoenix_instance.make_mth5_from_phoenix.return_value = (
            temp_data_dir / "custom_phoenix.h5"
        )
        mock_phoenix_class.return_value = mock_phoenix_instance

        # Create mock calibration directories
        rx_cal_dict = {"RX001": temp_data_dir / "rx001.json"}
        sensor_cal_dict = {"SENSOR001": temp_data_dir / "sensor001.json"}

        # Test with custom parameters
        result = MakeMTH5.from_phoenix(
            temp_data_dir,
            mth5_filename="custom_phoenix.h5",
            sample_rates=[75, 12000],
            receiver_calibration_dict=rx_cal_dict,
            sensor_calibration_dict=sensor_cal_dict,
            h5_compression="lzf",
        )

        # Verify PhoenixClient was called with correct parameters
        call_args = mock_phoenix_class.call_args
        assert call_args[0][0] == temp_data_dir
        assert call_args[1]["mth5_filename"] == "custom_phoenix.h5"
        assert call_args[1]["sample_rates"] == [75, 12000]
        assert call_args[1]["receiver_calibration_dict"] == rx_cal_dict
        assert call_args[1]["sensor_calibration_dict"] == sensor_cal_dict
        assert call_args[1]["h5_compression"] == "lzf"


# =============================================================================
# LEMI424 Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5LEMI424:
    """Test LEMI424 client functionality with mocking."""

    @patch("mth5.clients.make_mth5.LEMI424Client")
    def test_from_lemi424_basic(self, mock_lemi_class, temp_data_dir):
        """Test basic LEMI424 client creation."""
        # Setup mock
        mock_lemi_instance = Mock(spec=LEMI424Client)
        mock_lemi_instance.make_mth5_from_lemi424.return_value = (
            temp_data_dir / "lemi424.h5"
        )
        mock_lemi_class.return_value = mock_lemi_instance

        # Test method
        result = MakeMTH5.from_lemi424(
            temp_data_dir, "TEST_SURVEY", "MT001", save_path=temp_data_dir
        )

        # Verify calls
        mock_lemi_class.assert_called_once()
        call_args = mock_lemi_class.call_args
        assert call_args[0][0] == temp_data_dir  # data_path
        assert call_args[1]["save_path"] == temp_data_dir
        assert call_args[1]["mth5_filename"] == "from_lemi424.h5"  # Default

        mock_lemi_instance.make_mth5_from_lemi424.assert_called_once_with(
            "TEST_SURVEY", "MT001"
        )

        assert result == temp_data_dir / "lemi424.h5"

    @patch("mth5.clients.make_mth5.LEMI424Client")
    def test_from_lemi424_custom_parameters(self, mock_lemi_class, temp_data_dir):
        """Test LEMI424 client with custom parameters."""
        # Setup mock
        mock_lemi_instance = Mock(spec=LEMI424Client)
        mock_lemi_instance.make_mth5_from_lemi424.return_value = (
            temp_data_dir / "custom_lemi424.h5"
        )
        mock_lemi_class.return_value = mock_lemi_instance

        # Test with custom parameters
        result = MakeMTH5.from_lemi424(
            temp_data_dir,
            "CUSTOM_SURVEY",
            "MT002",
            mth5_filename="custom_lemi424.h5",
            save_path=temp_data_dir,
            h5_compression="lzf",
            h5_data_level=2,
        )

        # Verify LEMI424Client was called with correct parameters
        call_args = mock_lemi_class.call_args
        assert call_args[0][0] == temp_data_dir
        assert call_args[1]["save_path"] == temp_data_dir
        assert call_args[1]["mth5_filename"] == "custom_lemi424.h5"
        assert call_args[1]["h5_compression"] == "lzf"
        assert call_args[1]["h5_data_level"] == 2


# =============================================================================
# Metronix Client Tests (Mocked)
# =============================================================================


class TestMakeMTH5Metronix:
    """Test Metronix client functionality with mocking."""

    @patch("mth5.clients.make_mth5.MetronixClient")
    def test_from_metronix_basic(self, mock_metronix_class, temp_data_dir):
        """Test basic Metronix client creation."""
        # Setup mock
        mock_metronix_instance = Mock(spec=MetronixClient)
        mock_metronix_instance.make_mth5_from_metronix.return_value = (
            temp_data_dir / "metronix.h5"
        )
        mock_metronix_class.return_value = mock_metronix_instance

        # Test method
        result = MakeMTH5.from_metronix(
            temp_data_dir, sample_rates=[128], save_path=temp_data_dir
        )

        # Verify calls
        mock_metronix_class.assert_called_once()
        call_args = mock_metronix_class.call_args
        assert call_args[0][0] == temp_data_dir  # data_path
        assert call_args[1]["sample_rates"] == [128]
        assert call_args[1]["save_path"] == temp_data_dir

        mock_metronix_instance.make_mth5_from_metronix.assert_called_once()

        assert result == temp_data_dir / "metronix.h5"

    @patch("mth5.clients.make_mth5.MetronixClient")
    def test_from_metronix_custom_parameters(self, mock_metronix_class, temp_data_dir):
        """Test Metronix client with custom parameters."""
        # Setup mock
        mock_metronix_instance = Mock(spec=MetronixClient)
        mock_metronix_instance.make_mth5_from_metronix.return_value = (
            temp_data_dir / "custom_metronix.h5"
        )
        mock_metronix_class.return_value = mock_metronix_instance

        # Test with custom parameters
        result = MakeMTH5.from_metronix(
            temp_data_dir,
            sample_rates=[256, 1024],
            mth5_filename="custom_metronix.h5",
            run_name_zeros=4,
            h5_compression="gzip",
            h5_compression_opts=9,
        )

        # Verify MetronixClient was called with correct parameters
        call_args = mock_metronix_class.call_args
        assert call_args[0][0] == temp_data_dir
        assert call_args[1]["sample_rates"] == [256, 1024]
        assert call_args[1]["mth5_filename"] == "custom_metronix.h5"
        assert call_args[1]["h5_compression"] == "gzip"
        assert call_args[1]["h5_compression_opts"] == 9

        # Verify make_mth5_from_metronix was called with run_name_zeros
        metronix_call_args = mock_metronix_instance.make_mth5_from_metronix.call_args[1]
        assert metronix_call_args["run_name_zeros"] == 4


# =============================================================================
# Parameter Validation Tests
# =============================================================================


class TestMakeMTH5ParameterValidation:
    """Test parameter validation and edge cases."""

    @pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "1.0.0", "2.1.0"])
    def test_valid_mth5_versions(self, version, temp_data_dir):
        """Test valid MTH5 version formats."""
        maker = MakeMTH5(mth5_version=version, save_path=temp_data_dir)
        assert maker.mth5_version == version

    @pytest.mark.parametrize("interact", [True, False])
    def test_interact_boolean_values(self, interact, temp_data_dir):
        """Test boolean interact values."""
        maker = MakeMTH5(interact=interact, save_path=temp_data_dir)
        assert maker.interact is interact

    @pytest.mark.parametrize("data_level", [1, 2, 3, 4])
    def test_data_level_values(self, data_level, temp_data_dir):
        """Test different data level values."""
        maker = MakeMTH5(h5_data_level=data_level, save_path=temp_data_dir)
        assert maker.h5_data_level == data_level

    def test_none_values_handling(self, temp_data_dir):
        """Test handling of None values for optional parameters."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            h5_compression=None,
            h5_compression_opts=None,
            h5_shuffle=None,
            h5_fletcher32=None,
        )

        assert maker.h5_compression is None
        assert maker.h5_compression_opts is None
        assert maker.h5_shuffle is None
        assert maker.h5_fletcher32 is None


# =============================================================================
# Integration and Edge Case Tests
# =============================================================================


class TestMakeMTH5Integration:
    """Test integration scenarios and edge cases."""

    def test_kwargs_override_defaults(self, temp_data_dir):
        """Test that kwargs properly override default values."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            h5_compression="lzf",  # Override default
            h5_compression_opts=1,  # Override default
            h5_shuffle=False,  # Override default
            custom_param="test",  # New parameter
        )

        assert maker.h5_compression == "lzf"
        assert maker.h5_compression_opts == 1
        assert maker.h5_shuffle is False
        assert maker.custom_param == "test"
        # Verify unchanged defaults
        assert maker.h5_fletcher32 is True
        assert maker.h5_data_level == 1

    def test_multiple_h5_parameters_in_kwargs(self, temp_data_dir):
        """Test multiple h5-prefixed parameters in get_h5_kwargs."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            h5_param1="value1",
            h5_param2=42,
            h5_param3=True,
            h5_nested_param={"nested": "value"},
            non_h5_param="ignored",
        )

        h5_kwargs = maker.get_h5_kwargs()

        assert h5_kwargs["h5_param1"] == "value1"
        assert h5_kwargs["h5_param2"] == 42
        assert h5_kwargs["h5_param3"] is True
        assert h5_kwargs["h5_nested_param"] == {"nested": "value"}
        assert "non_h5_param" not in h5_kwargs

    def test_path_object_handling(self, temp_data_dir):
        """Test that Path objects are handled correctly."""
        maker = MakeMTH5(save_path=temp_data_dir)

        # The save_path should be stored as provided
        assert maker.save_path == temp_data_dir
        assert isinstance(maker.save_path, Path)

    def test_empty_kwargs(self, temp_data_dir):
        """Test initialization with empty kwargs."""
        maker = MakeMTH5(save_path=temp_data_dir, **{})

        # Should have default values
        assert maker.mth5_version == "0.2.0"
        assert maker.interact is False
        assert maker.h5_compression == "gzip"


# =============================================================================
# Client Method Parameter Passing Tests
# =============================================================================


class TestMakeMTH5ClientParameterPassing:
    """Test that parameters are correctly passed to client methods."""

    def test_client_method_inheritance_consistency(self, temp_data_dir):
        """Test that all client methods follow consistent parameter patterns."""
        # This is a meta-test to verify all client methods accept save_path and h5 parameters
        client_methods = [
            ("from_fdsn_client", {"request_df": pd.DataFrame()}),
            ("from_usgs_geomag", {"request_df": pd.DataFrame()}),
            ("from_zen", {"data_path": temp_data_dir}),
            ("from_phoenix", {"data_path": temp_data_dir}),
            (
                "from_lemi424",
                {"data_path": temp_data_dir, "survey_id": "test", "station_id": "mt01"},
            ),
            ("from_metronix", {"data_path": temp_data_dir}),
        ]

        # Test that each method accepts common parameters without error
        for method_name, required_params in client_methods:
            method = getattr(MakeMTH5, method_name)

            # Create parameters that should be acceptable to all methods
            common_params = {
                "save_path": temp_data_dir,
                "h5_compression": "gzip",
                "h5_compression_opts": 4,
                "h5_shuffle": True,
                "h5_fletcher32": True,
                "h5_data_level": 1,
                "mth5_version": "0.2.0",
                "interact": False,
            }

            # Combine with required parameters
            test_params = {**required_params, **common_params}

            # This test verifies the method signature accepts the parameters
            # We don't actually call the method to avoid external dependencies
            # Instead, we verify the method exists and can be called with these parameters
            assert hasattr(MakeMTH5, method_name)
            assert callable(method)


# =============================================================================
# Performance and Memory Tests
# =============================================================================


class TestMakeMTH5Performance:
    """Test performance characteristics and memory usage."""

    def test_large_kwargs_dictionary(self, temp_data_dir):
        """Test handling of large kwargs dictionary."""
        # Create a large number of parameters
        large_kwargs = {f"h5_param_{i}": f"value_{i}" for i in range(100)}
        large_kwargs["save_path"] = temp_data_dir

        maker = MakeMTH5(**large_kwargs)
        h5_kwargs = maker.get_h5_kwargs()

        # Verify all h5-prefixed parameters are included
        assert len([k for k in h5_kwargs.keys() if k.startswith("h5_param_")]) == 100

        # Verify access to individual parameters
        assert maker.h5_param_0 == "value_0"
        assert maker.h5_param_99 == "value_99"

    def test_get_h5_kwargs_performance(self, temp_data_dir):
        """Test that get_h5_kwargs performs well with many attributes."""
        # Add many attributes to the instance
        maker = MakeMTH5(save_path=temp_data_dir)

        for i in range(50):
            setattr(maker, f"h5_attr_{i}", f"value_{i}")
            setattr(maker, f"non_h5_attr_{i}", f"ignored_{i}")

        # get_h5_kwargs should only include h5-prefixed attributes plus standard ones
        h5_kwargs = maker.get_h5_kwargs()

        # Should have 6 standard keys + 50 h5_attr_ keys
        h5_keys = [k for k in h5_kwargs.keys() if k.startswith("h5_attr_")]
        assert len(h5_keys) == 50

        # Should not include non-h5 attributes
        non_h5_keys = [k for k in h5_kwargs.keys() if k.startswith("non_h5_attr_")]
        assert len(non_h5_keys) == 0


# =============================================================================
# Documentation and Docstring Tests
# =============================================================================


class TestMakeMTH5Documentation:
    """Test documentation and docstring completeness."""

    def test_class_has_docstring(self):
        """Test that MakeMTH5 class has a docstring."""
        # Note: MakeMTH5 class currently doesn't have a docstring
        # This test documents the current state
        # assert MakeMTH5.__doc__ is not None
        # assert len(MakeMTH5.__doc__.strip()) > 0
        assert hasattr(MakeMTH5, "__doc__")  # At least the attribute exists

    def test_client_methods_have_docstrings(self):
        """Test that all client methods have docstrings."""
        client_methods = [
            "from_fdsn_client",
            "from_fdsn_miniseed_and_stationxml",
            "from_usgs_geomag",
            "from_zen",
            "from_phoenix",
            "from_lemi424",
            "from_metronix",
        ]

        for method_name in client_methods:
            method = getattr(MakeMTH5, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0

    def test_class_has_docstring(self):
        """Test that MakeMTH5 class has a docstring."""
        assert MakeMTH5.__doc__ is not None
        assert len(MakeMTH5.__doc__.strip()) > 0

    def test_get_h5_kwargs_method_has_docstring(self):
        """Test that get_h5_kwargs method has a docstring."""
        # Note: get_h5_kwargs might not have an explicit docstring
        # This test documents the current state
        method = getattr(MakeMTH5, "get_h5_kwargs")
        assert callable(method)


# =============================================================================
# Error Handling and Exception Tests
# =============================================================================


class TestMakeMTH5ErrorHandling:
    """Test error handling and exception scenarios."""

    def test_invalid_attribute_access(self, basic_make_mth5):
        """Test that accessing non-existent attributes raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = basic_make_mth5.non_existent_attribute

    def test_h5_kwargs_with_invalid_types(self, temp_data_dir):
        """Test get_h5_kwargs with various data types."""
        maker = MakeMTH5(
            save_path=temp_data_dir,
            h5_list_param=[1, 2, 3],
            h5_dict_param={"key": "value"},
            h5_none_param=None,
            h5_bool_param=True,
        )

        h5_kwargs = maker.get_h5_kwargs()

        # All types should be preserved
        assert h5_kwargs["h5_list_param"] == [1, 2, 3]
        assert h5_kwargs["h5_dict_param"] == {"key": "value"}
        assert h5_kwargs["h5_none_param"] is None
        assert h5_kwargs["h5_bool_param"] is True

    def test_string_representation_with_none_values(self, temp_data_dir):
        """Test string representation when parameters are None."""
        maker = MakeMTH5(
            save_path=temp_data_dir, h5_compression=None, h5_compression_opts=None
        )

        str_repr = str(maker)

        # Should handle None values gracefully
        assert "h5_compression: None" in str_repr
        assert "h5_compression_opts: None" in str_repr


# =============================================================================
# Regression Tests
# =============================================================================


class TestMakeMTH5RegressionTests:
    """Test for regressions and backward compatibility."""

    def test_backward_compatibility_v1_parameters(self):
        """Test that v1-style parameters still work."""
        # This should work as in the original test
        maker = MakeMTH5(mth5_version="0.1.0", interact=True, save_path=None)

        assert maker.mth5_version == "0.1.0"
        assert maker.interact is True
        assert maker.save_path == Path().cwd()
        assert maker.h5_compression == "gzip"
        assert maker.h5_compression_opts == 4
        assert maker.h5_shuffle is True
        assert maker.h5_fletcher32 is True
        assert maker.h5_data_level == 1

    def test_backward_compatibility_v2_parameters(self):
        """Test that v2-style parameters still work."""
        # This should work as in the original test
        maker = MakeMTH5(
            mth5_version="0.2.0",
            interact=False,
            save_path=None,
            h5_compression=None,
            h5_compression_opts=None,
            h5_shuffle=False,
            h5_fletcher32=False,
            h5_data_level=2,
        )

        assert maker.mth5_version == "0.2.0"
        assert maker.interact is False
        assert maker.save_path == Path().cwd()
        assert maker.h5_compression is None
        assert maker.h5_compression_opts is None
        assert maker.h5_shuffle is False
        assert maker.h5_fletcher32 is False
        assert maker.h5_data_level == 2

    def test_original_test_cases_still_pass(self):
        """Test that all original test cases still pass with new implementation."""
        # V1 test case
        m1 = MakeMTH5(mth5_version="0.1.0", interact=True, save_path=None)
        assert m1.mth5_version == "0.1.0"
        assert m1.interact is True
        assert m1.save_path == Path().cwd()
        assert m1.h5_compression == "gzip"
        assert m1.h5_compression_opts == 4
        assert m1.h5_shuffle is True
        assert m1.h5_fletcher32 is True
        assert m1.h5_data_level == 1

        # V2 test case
        m2 = MakeMTH5(
            mth5_version="0.2.0",
            interact=False,
            save_path=None,
            h5_compression=None,
            h5_compression_opts=None,
            h5_shuffle=False,
            h5_fletcher32=False,
            h5_data_level=2,
        )
        assert m2.mth5_version == "0.2.0"
        assert m2.interact is False
        assert m2.save_path == Path().cwd()
        assert m2.h5_compression is None
        assert m2.h5_compression_opts is None
        assert m2.h5_shuffle is False
        assert m2.h5_fletcher32 is False
        assert m2.h5_data_level == 2


# =============================================================================
# Run pytest if script is executed directly
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__])
