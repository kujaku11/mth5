# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 21:58:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.timeseries import Auxiliary, Electric, Magnetic

from mth5.utils import fdsn_tools


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="session")
def electric_channel():
    """Electric channel metadata for testing."""
    return Electric(
        component="ex",
        channel_number=1,
        sample_rate=50,
        measurement_azimuth=5,
    )


@pytest.fixture(scope="session")
def magnetic_channel():
    """Magnetic channel metadata for testing."""
    return Magnetic(
        component="hx",
        channel_number=3,
        sample_rate=500,
        measurement_azimuth=95,
    )


@pytest.fixture(scope="session")
def auxiliary_channel():
    """Auxiliary temperature channel metadata for testing."""
    return Auxiliary(
        component="temperature",
        channel_number=6,
        sample_rate=1200,
        measurement_azimuth=105,
        type="temperature",
    )


@pytest.fixture(scope="session")
def period_code_mapping():
    """Period code to sample rate mapping for testing."""
    return {
        "F": 2000,
        "C": 500,
        "E": 100,
        "B": 50,
        "M": 5,
        "L": 1,
        "V": 0.1,
        "U": 0.01,
        "R": 0.0005,
        "P": 0.00005,
        "T": 0.000005,
        "Q": 0.0000005,
    }


@pytest.fixture(scope="session")
def measurement_code_mapping():
    """Measurement type to code mapping for testing."""
    return {
        "tilt": "A",
        "creep": "B",
        "calibration": "C",
        "pressure": "D",
        "magnetics": "F",
        "gravity": "G",
        "humidity": "I",
        "temperature": "K",
        "water_current": "O",
        "electric": "Q",
        "rain_fall": "R",
        "linear_strain": "S",
        "tide": "T",
        "wind": "W",
    }


@pytest.fixture(scope="session")
def horizontal_orientation_mapping():
    """Horizontal orientation angle to code mapping for testing."""
    return {0: "N", 5: "N", 20: "1", 90: "E", 55: "2"}


@pytest.fixture(scope="session")
def vertical_orientation_mapping():
    """Vertical orientation angle to code mapping for testing."""
    return {0: "Z", 5: "Z", 20: "3"}


@pytest.fixture(scope="session")
def channel_code_data():
    """Channel code test data for testing."""
    return {
        "BQN": {
            "period": {"min": 10, "max": 80},
            "component": "electric",
            "orientation": {"min": 0, "max": 15},
            "vertical": False,
        },
        "FQZ": {
            "period": {"min": 1000, "max": 5000},
            "component": "electric",
            "orientation": {"min": 0, "max": 15},
            "vertical": True,
        },
    }


@pytest.fixture(scope="session")
def mt_channel_mapping():
    """MT channel mapping test data."""
    return {
        "BQN": "ex",
        "FFN": "hx",
        "KKZ": "temperaturez",
    }


# =============================================================================
# Test Functions
# =============================================================================
def test_location_code(electric_channel, magnetic_channel, auxiliary_channel, subtests):
    """Test location code generation for different channel types."""
    test_cases = [
        (electric_channel, "E1", "ex"),
        (magnetic_channel, "H3", "hx"),
        (auxiliary_channel, "T6", "temperature"),
    ]

    for channel, expected, description in test_cases:
        with subtests.test(description):
            assert fdsn_tools.get_location_code(channel) == expected


@pytest.mark.parametrize(
    "code,sample_rate",
    [
        ("F", 2000),
        ("C", 500),
        ("E", 100),
        ("B", 50),
        ("M", 5),
        ("L", 1),
        ("V", 0.1),
        ("U", 0.01),
        ("R", 0.0005),
        ("P", 0.00005),
        ("T", 0.000005),
        ("Q", 0.0000005),
    ],
)
def test_period_code(code, sample_rate):
    """Test period code generation for different sample rates."""
    assert fdsn_tools.get_period_code(sample_rate) == code


def test_period_code_all_mappings(period_code_mapping, subtests):
    """Test all period code mappings using subtests."""
    for expected_code, sample_rate in period_code_mapping.items():
        with subtests.test(code=expected_code, rate=sample_rate):
            assert fdsn_tools.get_period_code(sample_rate) == expected_code


@pytest.mark.parametrize(
    "measurement_type,expected_code",
    [
        ("tilt", "A"),
        ("creep", "B"),
        ("calibration", "C"),
        ("pressure", "D"),
        ("magnetics", "F"),
        ("gravity", "G"),
        ("humidity", "I"),
        ("temperature", "K"),
        ("water_current", "O"),
        ("electric", "Q"),
        ("rain_fall", "R"),
        ("linear_strain", "S"),
        ("tide", "T"),
        ("wind", "W"),
    ],
)
def test_measurement_code(measurement_type, expected_code):
    """Test measurement code generation for different types."""
    assert fdsn_tools.get_measurement_code(measurement_type) == expected_code


def test_measurement_code_all_mappings(measurement_code_mapping, subtests):
    """Test all measurement code mappings using subtests."""
    for measurement_type, expected_code in measurement_code_mapping.items():
        with subtests.test(type=measurement_type):
            assert fdsn_tools.get_measurement_code(measurement_type) == expected_code


@pytest.mark.parametrize(
    "angle,expected_code",
    [
        (0, "N"),
        (5, "N"),
        (20, "1"),
        (90, "E"),
        (55, "2"),
    ],
)
def test_orientation_code_horizontal(angle, expected_code):
    """Test horizontal orientation code generation."""
    assert fdsn_tools.get_orientation_code(angle) == expected_code


def test_orientation_code_horizontal_all_mappings(
    horizontal_orientation_mapping, subtests
):
    """Test all horizontal orientation code mappings using subtests."""
    for angle, expected_code in horizontal_orientation_mapping.items():
        with subtests.test(angle=angle):
            assert fdsn_tools.get_orientation_code(angle) == expected_code


@pytest.mark.parametrize(
    "angle,expected_code",
    [
        (0, "Z"),
        (5, "Z"),
        (20, "3"),
    ],
)
def test_orientation_code_vertical(angle, expected_code):
    """Test vertical orientation code generation."""
    assert (
        fdsn_tools.get_orientation_code(angle, orientation="vertical") == expected_code
    )


def test_orientation_code_vertical_all_mappings(vertical_orientation_mapping, subtests):
    """Test all vertical orientation code mappings using subtests."""
    for angle, expected_code in vertical_orientation_mapping.items():
        with subtests.test(angle=angle):
            assert (
                fdsn_tools.get_orientation_code(angle, orientation="vertical")
                == expected_code
            )


def test_orientation_fail():
    """Test that invalid orientation raises ValueError."""
    with pytest.raises(ValueError):
        fdsn_tools.get_orientation_code(0, "juxtapose")


def test_make_channel_code(
    electric_channel, magnetic_channel, auxiliary_channel, subtests
):
    """Test channel code creation for different channel types."""
    test_cases = [
        (electric_channel, "BQN", "ex"),
        (magnetic_channel, "CFN", "hx"),
        (auxiliary_channel, "FKN", "temperature"),
    ]

    for channel, expected, description in test_cases:
        with subtests.test(description):
            assert fdsn_tools.make_channel_code(channel) == expected


@pytest.mark.parametrize(
    "channel_code,expected_data",
    [
        (
            "BQN",
            {
                "period": {"min": 10, "max": 80},
                "component": "electric",
                "orientation": {"min": 0, "max": 15},
                "vertical": False,
            },
        ),
        (
            "FQZ",
            {
                "period": {"min": 1000, "max": 5000},
                "component": "electric",
                "orientation": {"min": 0, "max": 15},
                "vertical": True,
            },
        ),
    ],
)
def test_read_channel_code(channel_code, expected_data):
    """Test reading channel code data."""
    assert fdsn_tools.read_channel_code(channel_code) == expected_data


def test_read_channel_code_all_mappings(channel_code_data, subtests):
    """Test all channel code reading using subtests."""
    for channel_code, expected_data in channel_code_data.items():
        with subtests.test(code=channel_code):
            assert fdsn_tools.read_channel_code(channel_code) == expected_data


@pytest.mark.parametrize(
    "channel_code,expected_mt_channel",
    [
        ("BQN", "ex"),
        ("FFN", "hx"),
        ("KKZ", "temperaturez"),
    ],
)
def test_make_mt_channel(channel_code, expected_mt_channel):
    """Test MT channel creation from channel codes."""
    channel_data = fdsn_tools.read_channel_code(channel_code)
    assert fdsn_tools.make_mt_channel(channel_data) == expected_mt_channel


def test_make_mt_channel_all_mappings(mt_channel_mapping, subtests):
    """Test all MT channel mappings using subtests."""
    for channel_code, expected_mt_channel in mt_channel_mapping.items():
        with subtests.test(code=channel_code):
            channel_data = fdsn_tools.read_channel_code(channel_code)
            assert fdsn_tools.make_mt_channel(channel_data) == expected_mt_channel


# =============================================================================
# Integration Tests
# =============================================================================
def test_full_workflow_electric(electric_channel):
    """Test complete FDSN workflow for electric channel."""
    # Generate location code
    location_code = fdsn_tools.get_location_code(electric_channel)
    assert location_code == "E1"

    # Generate channel code
    channel_code = fdsn_tools.make_channel_code(electric_channel)
    assert channel_code == "BQN"

    # Read channel code back
    channel_data = fdsn_tools.read_channel_code(channel_code)
    assert channel_data["component"] == "electric"

    # Generate MT channel name
    mt_channel = fdsn_tools.make_mt_channel(channel_data)
    assert mt_channel == "ex"


def test_full_workflow_magnetic(magnetic_channel):
    """Test complete FDSN workflow for magnetic channel."""
    # Generate location code
    location_code = fdsn_tools.get_location_code(magnetic_channel)
    assert location_code == "H3"

    # Generate channel code
    channel_code = fdsn_tools.make_channel_code(magnetic_channel)
    assert channel_code == "CFN"

    # Read channel code back
    channel_data = fdsn_tools.read_channel_code(channel_code)
    assert channel_data["component"] == "magnetics"

    # Generate MT channel name
    mt_channel = fdsn_tools.make_mt_channel(channel_data)
    assert mt_channel == "hx"


def test_full_workflow_auxiliary(auxiliary_channel):
    """Test complete FDSN workflow for auxiliary channel."""
    # Generate location code
    location_code = fdsn_tools.get_location_code(auxiliary_channel)
    assert location_code == "T6"

    # Generate channel code
    channel_code = fdsn_tools.make_channel_code(auxiliary_channel)
    assert channel_code == "FKN"

    # Read channel code back
    channel_data = fdsn_tools.read_channel_code(channel_code)
    assert channel_data["component"] == "temperature"

    # Generate MT channel name
    mt_channel = fdsn_tools.make_mt_channel(channel_data)
    assert mt_channel == "temperaturex"
