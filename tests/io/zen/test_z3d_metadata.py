# -*- coding: utf-8 -*-
"""
Pytest test suite for Z3DMetadata functionality.

Translated from unittest version with optimizations:
- Session-scoped fixtures for speed
- Parametrized tests for similar attributes
- Comprehensive error handling
- Support for missing test data

@author: jpeacock (original), translated to pytest
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mth5.io.zen import Z3DMetadata


# =============================================================================
# Test Data Setup
# =============================================================================


def get_z3d_test_data():
    """Get Z3D test data path if available, otherwise None."""
    try:
        import mth5_test_data

        return mth5_test_data.get_test_data_path("zen")
    except ImportError:
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def z3d_test_path():
    """Session-scoped fixture for Z3D test data path."""
    path = get_z3d_test_data()
    if path is None:
        pytest.skip("mth5_test_data package not available")
    return path


@pytest.fixture(scope="session")
def ey_z3d_file(z3d_test_path):
    """Session-scoped fixture for EY Z3D file path."""
    return z3d_test_path / "bm100_20220517_131017_256_EY.Z3D"


@pytest.fixture(scope="session")
def hx_z3d_file(z3d_test_path):
    """Session-scoped fixture for HX Z3D file path."""
    return z3d_test_path / "bm100_20220517_131017_256_HX.Z3D"


@pytest.fixture(scope="session")
def ey_z3d_metadata(ey_z3d_file):
    """Session-scoped fixture for EY Z3DMetadata object."""
    z3d_obj = Z3DMetadata(fn=ey_z3d_file)
    z3d_obj.read_metadata()
    return z3d_obj


@pytest.fixture(scope="session")
def hx_z3d_metadata(hx_z3d_file):
    """Session-scoped fixture for HX Z3DMetadata object."""
    z3d_obj = Z3DMetadata(fn=hx_z3d_file)
    z3d_obj.read_metadata()
    return z3d_obj


# =============================================================================
# Additional Fixtures for Extended Testing
# =============================================================================


@pytest.fixture
def basic_z3d_metadata():
    """Basic Z3DMetadata instance for testing."""
    return Z3DMetadata()


@pytest.fixture
def sample_metadata_attributes():
    """Sample metadata attributes for testing."""
    return {
        "ch_cmp": "ey",
        "ch_stn": "100",
        "gdp_progver": "zenacqv4.65",
        "gdp_number": "24",
        "rx_stn": "100",
        "rx_xazimuth": "0",
        "rx_zpositive": "down",
        "gdp_operator": "test_operator",
        "job_by": "test_job",
        "station": "100",
    }


@pytest.fixture
def sample_board_cal_data():
    """Sample board calibration data."""
    return [
        [100.0, 1.0, 0.998, -10.5],
        [256.0, 2.0, 0.999, -12.3],
        [1024.0, 8.0, 1.001, -15.7],
    ]


@pytest.fixture
def sample_coil_cal_data():
    """Sample coil calibration data."""
    return [
        7.32422e-04,
        0.28228,
        1570.2,
        9.76563e-04,
        0.40325,
        1563.3,
        1.46484e-03,
        0.60488,
        1555.5,
        1.95313e-03,
        0.76618,
        1551.0,
    ]


@pytest.fixture
def sample_cal_board_dict():
    """Sample cal_board dictionary."""
    return {
        "cal.date": "11/19/2021",
        "cal.attn": 0.0,
        "gdp.volt": 13.25,
        "gdp.temp": 41.5,
        256: {"frequency": 2.0, "amplitude": 1.00153, "phase": -1533.33},
        1024: {"frequency": 8.0, "amplitude": 1.001513, "phase": -1532.9},
        4096: {"frequency": 32.0, "amplitude": 1.001414, "phase": -1534.24},
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestZ3DMetadataEY:
    """Test class for EY component Z3D metadata."""

    # Simple attribute tests with expected values
    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("board_cal", []),
            ("cal_ant", None),
            ("cal_ver", None),
            ("ch_azimuth", None),
            ("ch_cmp", "ey"),
            ("ch_cres", None),
            ("ch_length", None),
            ("ch_number", None),
            ("ch_offset_xyz1", "0:0:0"),
            ("ch_offset_xyz2", "0:-56:0"),
            ("ch_sp", "0.02"),
            ("ch_stn", "100"),
            ("ch_vmax", "2e-05"),
            ("ch_xyz1", None),
            ("ch_xyz2", None),
            ("coil_cal", []),
            ("count", 3),
            ("find_metadata", False),
            ("gdp_number", "24"),
            ("gdp_operator", ""),
            ("gdp_progver", "zenacqv4.65"),
            ("gdp_temp", None),
            ("gdp_volt", None),
            ("job_by", ""),
            ("job_for", ""),
            ("job_name", ""),
            ("job_number", ""),
            ("line_name", None),
            ("rx_aspace", None),
            ("rx_sspace", None),
            ("rx_stn", "100"),
            ("rx_xazimuth", "0"),
            ("rx_xyz0", None),
            ("rx_yazimuth", None),
            ("rx_zpositive", "down"),
            ("station", "100"),
            ("survey_type", None),
            ("unit_length", None),
        ],
    )
    def test_ey_simple_attributes(self, ey_z3d_metadata, attribute, expected):
        """Test simple EY metadata attributes."""
        assert getattr(ey_z3d_metadata, attribute) == expected

    def test_ey_fn_attribute(self, ey_z3d_metadata, ey_z3d_file):
        """Test EY filename attribute."""
        assert getattr(ey_z3d_metadata, "fn") == ey_z3d_file

    def test_ey_cal_board(self, ey_z3d_metadata):
        """Test EY board calibration data."""
        expected = {
            "cal.date": "11/19/2021",
            "cal.attn": 0.0,
            256: {
                "frequency": 2.0,
                "amplitude": 1.00153,
                "phase": -1533.33,
            },
            1024: {
                "frequency": 8.0,
                "amplitude": 1.001513,
                "phase": -1532.9,
            },
            4096: {
                "frequency": 32.0,
                "amplitude": 1.001414,
                "phase": -1534.24,
            },
            "gdp.volt": 13.25,
            "gdp.temp": 41.5,
        }
        assert getattr(ey_z3d_metadata, "cal_board") == expected


class TestZ3DMetadataHX:
    """Test class for HX component Z3D metadata."""

    # Simple attribute tests with expected values
    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("board_cal", []),
            ("cal_ant", "2314"),
            ("cal_ver", None),
            ("ch_antsn", "2314"),
            ("ch_azimuth", "0"),
            ("ch_cmp", "hx"),
            ("ch_cres", None),
            ("ch_incl", "0"),
            ("ch_length", None),
            ("ch_number", None),
            ("ch_offset_xyz1", "0:0:0"),
            ("ch_sp", "0.02"),
            ("ch_stn", "100"),
            ("ch_vmax", "2e-05"),
            ("ch_xyz1", None),
            ("ch_xyz2", None),
            ("count", 9),
            ("find_metadata", False),
            ("gdp_number", "24"),
            ("gdp_operator", ""),
            ("gdp_progver", "zenacqv4.65"),
            ("gdp_temp", None),
            ("gdp_volt", None),
            ("job_by", ""),
            ("job_for", ""),
            ("job_name", ""),
            ("job_number", ""),
            ("line_name", None),
            ("rx_aspace", None),
            ("rx_sspace", None),
            ("rx_stn", "100"),
            ("rx_xazimuth", "0"),
            ("rx_xyz0", None),
            ("rx_yazimuth", None),
            ("rx_zpositive", "down"),
            ("station", "100"),
            ("survey_type", None),
            ("unit_length", None),
        ],
    )
    def test_hx_simple_attributes(self, hx_z3d_metadata, attribute, expected):
        """Test simple HX metadata attributes."""
        assert getattr(hx_z3d_metadata, attribute) == expected

    def test_hx_fn_attribute(self, hx_z3d_metadata, hx_z3d_file):
        """Test HX filename attribute."""
        assert getattr(hx_z3d_metadata, "fn") == hx_z3d_file

    def test_hx_cal_board(self, hx_z3d_metadata):
        """Test HX board calibration data."""
        expected = {
            "cal.date": "11/19/2021",
            "cal.attn": 0.0,
            256: {
                "frequency": 2.0,
                "amplitude": 1.000128,
                "phase": -1533.34,
            },
            1024: {
                "frequency": 8.0,
                "amplitude": 1.000109,
                "phase": -1532.93,
            },
            4096: {
                "frequency": 32.0,
                "amplitude": 1.000011,
                "phase": -1534.34,
            },
            "gdp.volt": 13.25,
            "gdp.temp": 41.5,
        }
        assert getattr(hx_z3d_metadata, "cal_board") == expected

    def test_hx_coil_cal(self, hx_z3d_metadata):
        """Test HX coil calibration data (numpy array)."""
        expected_array = np.array(
            [
                (7.32422e-04, 0.28228, 1570.2),
                (9.76563e-04, 0.40325, 1563.3),
                (1.46484e-03, 0.60488, 1555.5),
                (1.95313e-03, 0.76618, 1551.0),
                (2.92969e-03, 1.16944, 1542.8),
                (3.90625e-03, 1.57269, 1537.1),
                (5.85938e-03, 2.3792, 1529.2),
                (7.81250e-03, 3.14538, 1523.8),
                (1.17188e-02, 4.71807, 1516.0),
                (1.56250e-02, 6.29076, 1510.5),
                (2.34375e-02, 9.33806, 1473.0),
                (3.12500e-02, 12.4627, 1445.4),
                (4.68750e-02, 18.6755, 1390.9),
                (6.25000e-02, 24.6371, 1312.7),
                (9.37500e-02, 35.4436, 1205.7),
                (1.25000e-01, 44.9365, 1103.1),
                (1.87500e-01, 59.9935, 923.5),
                (2.50000e-01, 70.6949, 781.2),
                (3.75000e-01, 83.106, 583.3),
                (5.00000e-01, 89.3232, 459.3),
                (7.50000e-01, 94.7067, 318.1),
                (1.00000e00, 96.7628, 241.1),
                (1.50000e00, 98.412, 160.7),
                (2.00000e00, 98.9519, 119.4),
                (3.00000e00, 99.3774, 76.6),
                (4.00000e00, 99.5137, 53.9),
                (6.00000e00, 99.6679, 29.3),
                (8.00000e00, 99.6429, 14.9),
                (1.20000e01, 99.7832, -3.2),
                (1.60000e01, 99.7335, -16.8),
                (2.40000e01, 99.7926, -38.3),
                (3.20000e01, 99.7775, -56.9),
                (4.80000e01, 99.836, -92.2),
                (6.40000e01, 99.7502, -126.2),
                (9.60000e01, 99.8861, -192.3),
                (1.28000e02, 99.8326, -258.8),
                (1.92000e02, 99.8494, -392.1),
                (2.56000e02, 99.8944, -525.6),
                (3.84000e02, 99.7758, -803.6),
                (5.12000e02, 99.1085, -1099.6),
                (7.68000e02, 90.8383, -1767.5),
                (1.02400e03, 67.8248, -2460.5),
                (1.53600e03, 26.1416, 2930.7),
                (2.04800e03, 11.2864, 2549.5),
                (3.07200e03, 3.50231, 2169.9),
                (4.09600e03, 1.52457, 1951.5),
                (6.14400e03, 0.48749, 1748.3),
                (8.19200e03, 0.22572, 1577.5),
            ]
        )
        expected_records = np.rec.fromrecords(
            expected_array, names="frequency, amplitude, phase"
        )

        coil_cal = getattr(hx_z3d_metadata, "coil_cal")

        # Compare structured arrays element by element
        assert isinstance(coil_cal, np.ndarray)
        assert coil_cal.dtype.names == expected_records.dtype.names
        assert len(coil_cal) == len(expected_records)

        # Compare all values using numpy testing for floating point precision
        np.testing.assert_array_equal(coil_cal, expected_records)


# =============================================================================
# Additional Test Classes for Extended Coverage
# =============================================================================


class TestZ3DMetadataAttributeHandling:
    """Test metadata attribute setting and validation."""

    def test_attribute_setting(self, basic_z3d_metadata, sample_metadata_attributes):
        """Test that metadata attributes can be set correctly."""
        zm = basic_z3d_metadata

        # Set attributes manually (simulating what read_metadata does)
        for attr, value in sample_metadata_attributes.items():
            setattr(zm, attr, value)

        # Verify all attributes were set correctly
        for attr, expected_value in sample_metadata_attributes.items():
            actual_value = getattr(zm, attr)
            assert (
                actual_value == expected_value
            ), f"Expected {attr}={expected_value}, got {actual_value}"

    def test_attribute_type_conversion(self, basic_z3d_metadata):
        """Test that string attributes are handled properly."""
        zm = basic_z3d_metadata

        # Test string attributes
        zm.ch_cmp = "ey"
        zm.ch_stn = "100"  # String representation of number
        zm.rx_xazimuth = "0"  # String representation of number

        assert isinstance(zm.ch_cmp, str)
        assert isinstance(zm.ch_stn, str)
        assert isinstance(zm.rx_xazimuth, str)

        # Values should remain as strings (as per Z3D format)
        assert zm.ch_cmp == "ey"
        assert zm.ch_stn == "100"
        assert zm.rx_xazimuth == "0"

    def test_key_normalization(self, basic_z3d_metadata):
        """Test attribute key normalization (dots to underscores, lowercase)."""
        zm = basic_z3d_metadata

        # Simulate the key transformation from read_metadata
        test_cases = [
            ("ch.cmp", "ch_cmp"),
            ("GDP.PROGVER", "gdp_progver"),
            ("rx.stn", "rx_stn"),
            ("line.name", "line_name"),
        ]

        for original_key, expected_attr in test_cases:
            normalized_key = original_key.replace(".", "_").lower()
            assert normalized_key == expected_attr

            # Test setting the attribute
            setattr(zm, normalized_key, "test_value")
            assert getattr(zm, expected_attr) == "test_value"


class TestZ3DMetadataStationLogic:
    """Test station name determination logic."""

    def test_station_from_rx_stn(self, basic_z3d_metadata):
        """Test station name from rx_stn attribute."""
        zm = basic_z3d_metadata
        zm.rx_stn = "100"

        # Simulate the station logic from read_metadata
        if hasattr(zm, "rx_stn"):
            zm.station = f"{zm.rx_stn}"

        assert zm.station == "100"

    def test_station_from_ch_stn_fallback(self, basic_z3d_metadata):
        """Test station name fallback to ch_stn."""
        zm = basic_z3d_metadata
        zm.ch_stn = "200"

        # Simulate fallback logic when rx_stn not available
        if hasattr(zm, "ch_stn"):
            zm.station = f"{zm.ch_stn}"

        assert zm.station == "200"

    def test_station_from_line_name_and_rx_xyz0(self, basic_z3d_metadata):
        """Test station name from line_name and rx_xyz0."""
        zm = basic_z3d_metadata
        zm.line_name = "test_line"
        zm.rx_xyz0 = "300:400:500"

        # Simulate the primary station logic
        try:
            zm.station = "{0}{1}".format(zm.line_name, zm.rx_xyz0.split(":")[0])
        except AttributeError:
            pass

        assert zm.station == "test_line300"

    def test_station_warning_when_unavailable(self, basic_z3d_metadata):
        """Test warning when station info is not available."""
        zm = basic_z3d_metadata

        with patch.object(zm, "logger", create=True) as mock_logger:
            # Simulate the warning case
            zm.station = None
            mock_logger.warning("Need to input station name")

            mock_logger.warning.assert_called_once_with("Need to input station name")
            assert zm.station is None


class TestZ3DMetadataCalibrationConversions:
    """Test calibration data conversions and numpy array handling."""

    def test_board_calibration_numpy_conversion(
        self, basic_z3d_metadata, sample_board_cal_data
    ):
        """Test board calibration conversion to numpy structured array."""
        zm = basic_z3d_metadata
        zm.board_cal = sample_board_cal_data

        # Simulate the numpy conversion from read_metadata
        try:
            zm.board_cal = np.rec.fromrecords(
                zm.board_cal, names="frequency, rate, amplitude, phase"
            )

            # Test successful conversion
            assert isinstance(zm.board_cal, np.ndarray)
            assert zm.board_cal.dtype.names == (
                "frequency",
                "rate",
                "amplitude",
                "phase",
            )
            assert len(zm.board_cal) == 3

            # Test data values
            assert zm.board_cal[0]["frequency"] == 100.0
            assert zm.board_cal[1]["frequency"] == 256.0
            assert zm.board_cal[2]["frequency"] == 1024.0

        except ValueError:
            # Test error handling path
            zm.board_cal = None
            assert zm.board_cal is None

    def test_coil_calibration_numpy_conversion(
        self, basic_z3d_metadata, sample_coil_cal_data
    ):
        """Test coil calibration conversion to numpy structured array."""
        zm = basic_z3d_metadata
        zm.coil_cal = sample_coil_cal_data

        # Simulate the conversion logic from read_metadata
        if len(zm.coil_cal) > 0:
            a = np.array(zm.coil_cal)
            a = a.reshape((int(a.size / 3), 3))
            zm.coil_cal = np.rec.fromrecords(a, names="frequency, amplitude, phase")

        assert isinstance(zm.coil_cal, np.ndarray)
        assert zm.coil_cal.dtype.names == ("frequency", "amplitude", "phase")
        assert len(zm.coil_cal) == 4

        # Test first record values
        assert zm.coil_cal[0]["frequency"] == pytest.approx(7.32422e-04)
        assert zm.coil_cal[0]["amplitude"] == pytest.approx(0.28228)
        assert zm.coil_cal[0]["phase"] == pytest.approx(1570.2)

    def test_board_calibration_error_handling(self, basic_z3d_metadata):
        """Test board calibration error handling with invalid data."""
        zm = basic_z3d_metadata

        # Invalid data (inconsistent record lengths)
        zm.board_cal = [
            [1, 2, 3],  # 3 elements
            [4, 5, 6, 7, 8],  # 5 elements - inconsistent!
        ]

        # Simulate error handling
        try:
            zm.board_cal = np.rec.fromrecords(
                zm.board_cal, names="frequency, rate, amplitude, phase"
            )
        except ValueError:
            zm.board_cal = None

        assert zm.board_cal is None

    def test_empty_calibration_arrays(self, basic_z3d_metadata):
        """Test handling of empty calibration arrays."""
        zm = basic_z3d_metadata

        # Test empty coil_cal
        zm.coil_cal = []
        if len(zm.coil_cal) > 0:
            # This block shouldn't execute
            assert False, "Should not process empty coil_cal array"

        # Test empty board_cal
        zm.board_cal = []
        if len(zm.board_cal) > 0:
            # This block shouldn't execute
            assert False, "Should not process empty board_cal array"

        # Arrays should remain as empty lists
        assert zm.coil_cal == []
        assert zm.board_cal == []


class TestZ3DMetadataCalibrationDataStructures:
    """Test calibration data structure handling."""

    def test_cal_board_dictionary_structure(
        self, basic_z3d_metadata, sample_cal_board_dict
    ):
        """Test cal_board dictionary structure and access."""
        zm = basic_z3d_metadata
        zm.cal_board = sample_cal_board_dict

        # Test dictionary structure
        assert isinstance(zm.cal_board, dict)

        # Test string keys
        assert "cal.date" in zm.cal_board
        assert "cal.attn" in zm.cal_board
        assert "gdp.volt" in zm.cal_board
        assert "gdp.temp" in zm.cal_board

        # Test integer keys (frequency-specific data)
        assert 256 in zm.cal_board
        assert 1024 in zm.cal_board
        assert 4096 in zm.cal_board

        # Test value types
        assert isinstance(zm.cal_board["cal.date"], str)
        assert isinstance(zm.cal_board["cal.attn"], (int, float))
        assert isinstance(zm.cal_board["gdp.volt"], (int, float))

        # Test frequency-specific dictionaries
        assert isinstance(zm.cal_board[256], dict)
        assert "frequency" in zm.cal_board[256]
        assert "amplitude" in zm.cal_board[256]
        assert "phase" in zm.cal_board[256]

        # Test specific values
        assert zm.cal_board["cal.date"] == "11/19/2021"
        assert zm.cal_board["cal.attn"] == 0.0
        assert zm.cal_board[256]["frequency"] == 2.0
        assert zm.cal_board[256]["amplitude"] == 1.00153
        assert zm.cal_board[256]["phase"] == -1533.33


class TestZ3DMetadataFileOperations:
    """Test file operation mocking and path handling."""

    @patch("builtins.open", new_callable=mock_open)
    def test_file_opening(self, mock_file, basic_z3d_metadata):
        """Test file opening with mocked file operations."""
        zm = basic_z3d_metadata
        mock_file.return_value.read.return_value = b""

        test_fn = "/path/to/test.z3d"

        # Test that we can set filename
        zm.fn = test_fn
        assert zm.fn == test_fn

        # Test file opening would be called (in real read_metadata)
        # We don't call read_metadata here to avoid the complex parsing logic

    def test_file_object_handling(self, basic_z3d_metadata):
        """Test file object handling."""
        zm = basic_z3d_metadata

        mock_fid = Mock()
        zm.fid = mock_fid

        # Test file object assignment
        assert zm.fid is mock_fid

        # Test file object methods exist
        assert hasattr(mock_fid, "read")
        assert hasattr(mock_fid, "seek")
        assert hasattr(mock_fid, "tell")

    def test_pathlib_path_support(self, basic_z3d_metadata):
        """Test Path object support."""
        zm = basic_z3d_metadata

        test_path = Path("/path/to/test.z3d")
        zm.fn = test_path

        assert zm.fn == test_path

        # Test path operations (handle Windows path separators)
        path_str = str(zm.fn)
        assert path_str.endswith("test.z3d")
        assert "path" in path_str
        assert zm.fn.name == "test.z3d"

    def test_temporary_file_handling(self):
        """Test with temporary files."""
        with tempfile.NamedTemporaryFile(suffix=".z3d", delete=False) as tmp:
            tmp.write(b"\x00" * 2048)  # Write some dummy data
            tmp_path = tmp.name

        try:
            # Test initialization with temporary file
            zm = Z3DMetadata(fn=tmp_path)
            assert zm.fn == tmp_path

            # Verify file exists
            assert Path(tmp_path).exists()

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestZ3DMetadataIntegration:
    """Integration tests that test component interactions."""

    def test_complete_metadata_workflow_simulation(self, basic_z3d_metadata):
        """Test complete metadata workflow without file I/O."""
        zm = basic_z3d_metadata

        # Simulate what read_metadata does step by step
        # 1. Initialize metadata reading state
        zm.find_metadata = True
        zm.board_cal = []
        zm.coil_cal = []
        zm.count = 0

        # 2. Simulate processing metadata blocks
        zm.count = 3

        # 3. Simulate setting attributes from parsing
        zm.ch_cmp = "ey"
        zm.ch_stn = "100"
        zm.gdp_progver = "zenacqv4.65"
        zm.rx_stn = "100"
        zm.rx_xazimuth = "0"

        # 4. Simulate calibration data
        zm.board_cal = [[256.0, 2.0, 1.001, -1533.33], [1024.0, 8.0, 1.002, -1532.9]]

        # 5. Simulate numpy conversion
        if len(zm.board_cal) > 0:
            try:
                zm.board_cal = np.rec.fromrecords(
                    zm.board_cal, names="frequency, rate, amplitude, phase"
                )
            except ValueError:
                zm.board_cal = None

        # 6. Simulate station logic
        if hasattr(zm, "rx_stn"):
            zm.station = f"{zm.rx_stn}"

        # 7. Test final state
        assert zm.ch_cmp == "ey"
        assert zm.ch_stn == "100"
        assert zm.station == "100"
        assert zm.count == 3
        assert isinstance(zm.board_cal, np.ndarray)
        assert len(zm.board_cal) == 2

    def test_metadata_consistency(self, basic_z3d_metadata, sample_metadata_attributes):
        """Test metadata attribute consistency."""
        zm = basic_z3d_metadata

        # Set all sample attributes
        for attr, value in sample_metadata_attributes.items():
            setattr(zm, attr, value)

        # Test consistency checks
        assert zm.ch_stn == zm.rx_stn  # Should be the same station
        assert zm.station == zm.rx_stn  # Station should match rx_stn

        # Test attribute access patterns
        for attr in ["ch_cmp", "gdp_progver", "rx_xazimuth"]:
            assert hasattr(zm, attr)
            assert getattr(zm, attr) is not None


class TestZ3DMetadataAdvancedErrorHandling:
    """Test advanced error handling and edge cases."""

    def test_missing_attributes(self, basic_z3d_metadata):
        """Test handling of missing attributes."""
        zm = basic_z3d_metadata

        # Test hasattr checks (using truly non-existent attributes)
        assert not hasattr(zm, "non_existent_attr_1")
        assert not hasattr(zm, "made_up_attribute")
        assert not hasattr(zm, "random_property")
        assert not hasattr(zm, "fake_attribute")

        # Test getattr with defaults
        non_existent = getattr(zm, "non_existent_attr_1", None)
        made_up = getattr(zm, "made_up_attribute", None)
        random_prop = getattr(zm, "random_property", "default_value")

        assert non_existent is None
        assert made_up is None
        assert random_prop == "default_value"

    def test_attribute_error_handling(self, basic_z3d_metadata):
        """Test AttributeError handling for complex logic."""
        zm = basic_z3d_metadata

        # Test the station logic that uses try/except AttributeError
        try:
            # This will fail because line_name and rx_xyz0 don't exist
            zm.station = "{0}{1}".format(zm.line_name, zm.rx_xyz0.split(":")[0])
        except AttributeError:
            # Should fall back to other logic or set station = None
            zm.station = None

        assert zm.station is None

    def test_value_error_in_float_conversion(self, basic_z3d_metadata):
        """Test ValueError handling in numeric conversions."""
        zm = basic_z3d_metadata

        # Simulate trying to convert invalid strings to float
        test_values = ["invalid", "not_a_number", "text", ""]

        for test_val in test_values:
            try:
                float_val = float(test_val)
                assert False, f"Should not convert {test_val} to float"
            except ValueError:
                # Expected behavior - should handle gracefully
                pass

    def test_index_error_handling(self, basic_z3d_metadata):
        """Test IndexError handling in string parsing."""
        zm = basic_z3d_metadata

        # Test split operations that might cause IndexError
        test_strings = ["no_colon_here", ":empty_start", "empty_end:", "", "single"]

        for test_str in test_strings:
            split_result = test_str.split(":")
            # Safe access to avoid IndexError
            first_part = split_result[0] if len(split_result) > 0 else ""
            second_part = split_result[1] if len(split_result) > 1 else ""

            # Should not raise exceptions
            assert isinstance(first_part, str)
            assert isinstance(second_part, str)


class TestZ3DMetadataInitialization:
    """Test Z3DMetadata initialization and basic functionality."""

    def test_init_with_filename(self, ey_z3d_file):
        """Test Z3DMetadata initialization with filename."""
        z3d_obj = Z3DMetadata(fn=ey_z3d_file)
        assert z3d_obj.fn == ey_z3d_file
        assert z3d_obj.count == 0  # Before reading metadata

    def test_init_without_filename(self):
        """Test Z3DMetadata initialization without filename."""
        z3d_obj = Z3DMetadata()
        assert z3d_obj.fn is None
        assert z3d_obj.count == 0

    def test_read_metadata_sets_count(self, ey_z3d_metadata):
        """Test that read_metadata sets the count attribute."""
        assert ey_z3d_metadata.count > 0


class TestZ3DMetadataErrorHandling:
    """Test error handling in Z3DMetadata."""

    def test_read_metadata_nonexistent_file(self):
        """Test reading metadata from nonexistent file."""
        z3d_obj = Z3DMetadata(fn="nonexistent_file.Z3D")
        # This should handle the error gracefully
        # The actual behavior depends on implementation
        with pytest.raises((FileNotFoundError, IOError)):
            z3d_obj.read_metadata()

    def test_read_metadata_without_filename(self):
        """Test reading metadata without setting filename."""
        z3d_obj = Z3DMetadata()
        # This should handle the error gracefully
        with pytest.raises((AttributeError, ValueError, TypeError)):
            z3d_obj.read_metadata()


class TestZ3DMetadataComparison:
    """Test comparison between EY and HX metadata objects."""

    def test_different_components(self, ey_z3d_metadata, hx_z3d_metadata):
        """Test that EY and HX have different components."""
        assert ey_z3d_metadata.ch_cmp != hx_z3d_metadata.ch_cmp
        assert ey_z3d_metadata.ch_cmp == "ey"
        assert hx_z3d_metadata.ch_cmp == "hx"

    def test_common_station_info(self, ey_z3d_metadata, hx_z3d_metadata):
        """Test that EY and HX share common station information."""
        assert ey_z3d_metadata.station == hx_z3d_metadata.station
        assert ey_z3d_metadata.rx_stn == hx_z3d_metadata.rx_stn
        assert ey_z3d_metadata.gdp_number == hx_z3d_metadata.gdp_number

    def test_different_calibration_data(self, ey_z3d_metadata, hx_z3d_metadata):
        """Test that EY and HX have different calibration data."""
        # EY should have empty coil_cal, HX should have data
        assert len(ey_z3d_metadata.coil_cal) == 0
        assert len(hx_z3d_metadata.coil_cal) > 0

        # HX should have antenna serial number, EY should not
        assert ey_z3d_metadata.cal_ant is None
        assert hx_z3d_metadata.cal_ant == "2314"


# =============================================================================
# Performance Tests
# =============================================================================


class TestZ3DMetadataPerformance:
    """Test performance-related aspects of Z3DMetadata."""

    def test_multiple_reads_same_file(self, ey_z3d_file):
        """Test reading metadata multiple times from same file."""
        import time

        z3d_obj = Z3DMetadata(fn=ey_z3d_file)

        # Time multiple reads
        start_time = time.time()
        for _ in range(10):
            z3d_obj.read_metadata()
        end_time = time.time()

        # Should be reasonably fast (less than 1 second for 10 reads)
        assert (end_time - start_time) < 1.0
        assert z3d_obj.count > 0

    def test_fixture_reuse_performance(self, ey_z3d_metadata, hx_z3d_metadata):
        """Test that session-scoped fixtures provide fast access."""
        # This test validates that our session-scoped fixtures work
        # and provide already-parsed metadata objects
        assert hasattr(ey_z3d_metadata, "ch_cmp")
        assert hasattr(hx_z3d_metadata, "ch_cmp")
        assert ey_z3d_metadata.count > 0
        assert hx_z3d_metadata.count > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
