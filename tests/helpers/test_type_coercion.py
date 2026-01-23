# -*- coding: utf-8 -*-
"""
Tests for helper functions related to type coercion and metadata handling.

This module tests:
- get_data_type: Converting string representations to Python types
- get_metadata_type_dict: Extracting expected types from metadata classes
- coerce_value_to_expected_type: Type coercion for backwards compatibility

Created on January 22, 2026
"""

import numpy as np
import pytest
from loguru import logger
from mt_metadata import timeseries as metadata

from mth5.helpers import (
    coerce_value_to_expected_type,
    get_data_type,
    get_metadata_type_dict,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def survey_metadata():
    """Fixture for Survey metadata object."""
    return metadata.Survey()


@pytest.fixture(scope="module")
def station_metadata():
    """Fixture for Station metadata object."""
    return metadata.Station()


@pytest.fixture(scope="module")
def run_metadata():
    """Fixture for Run metadata object."""
    return metadata.Run()


@pytest.fixture(scope="module")
def electric_metadata():
    """Fixture for Electric channel metadata object."""
    return metadata.Electric()


@pytest.fixture(scope="module")
def magnetic_metadata():
    """Fixture for Magnetic channel metadata object."""
    return metadata.Magnetic()


@pytest.fixture(scope="module")
def auxiliary_metadata():
    """Fixture for Auxiliary channel metadata object."""
    return metadata.Auxiliary()


@pytest.fixture(scope="module")
def channel_metadata():
    """Fixture for base Channel metadata object."""
    return metadata.Channel()


@pytest.fixture
def mock_logger():
    """Create a mock object with a logger for testing coerce methods."""

    class MockObject:
        def __init__(self):
            self.logger = logger

    return MockObject()


# =============================================================================
# Test get_data_type
# =============================================================================


class TestGetDataType:
    """Test the get_data_type function."""

    @pytest.mark.parametrize(
        "input_str,expected_type",
        [
            ("int", int),
            ("float", float),
            ("str", str),
            ("bool", bool),
            ("list", list),
            ("dict", dict),
            ("complex", complex),
            ("Int", int),  # Test case insensitivity
            ("FLOAT", float),
            ("STR", str),
        ],
    )
    def test_basic_types(self, input_str, expected_type):
        """Test basic type string to type conversions."""
        result = get_data_type(input_str)
        assert result == expected_type

    @pytest.mark.parametrize(
        "input_str,expected_type",
        [
            ("'<class 'int'>'", int),
            ("'<class 'float'>'", float),
            ("'<class 'str'>'", str),
            ("<class 'int'>", int),
            ("<class 'float'>", float),
        ],
    )
    def test_class_string_representations(self, input_str, expected_type):
        """Test type strings from class representations."""
        result = get_data_type(input_str)
        assert result == expected_type

    @pytest.mark.parametrize(
        "input_str,expected_type",
        [
            ("str | None", str),
            ("float | None", float),
            ("int | None", int),
            ("str|None", str),
        ],
    )
    def test_union_types(self, input_str, expected_type):
        """Test Union type strings (e.g., str | None)."""
        result = get_data_type(input_str)
        assert result == expected_type

    @pytest.mark.parametrize(
        "input_str,expected_type",
        [
            ("list[str]", list),
            ("list[int]", list),
            ("dict[str, int]", dict),
            ("list[AppliedFilter]", list),
        ],
    )
    def test_generic_types(self, input_str, expected_type):
        """Test generic type strings with parameters."""
        result = get_data_type(input_str)
        assert result == expected_type

    def test_invalid_type_raises_error(self):
        """Test that invalid type string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown data type string representation"):
            get_data_type("not_a_real_type")


# =============================================================================
# Test get_metadata_type_dict
# =============================================================================


class TestGetMetadataTypeDict:
    """Test the get_metadata_type_dict function."""

    def test_survey_metadata_types(self, survey_metadata, subtests):
        """Test extracting type dict from Survey metadata."""
        type_dict = get_metadata_type_dict(survey_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "id" in type_dict
            assert "citation_dataset.doi" in type_dict

        with subtests.test("correct types"):
            assert type_dict["id"] == str
            # Check a few known fields
            if "acquired_by.author" in type_dict:
                assert type_dict["acquired_by.author"] == str

    def test_station_metadata_types(self, station_metadata, subtests):
        """Test extracting type dict from Station metadata."""
        type_dict = get_metadata_type_dict(station_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "id" in type_dict
            assert "location.latitude" in type_dict
            assert "location.longitude" in type_dict

        with subtests.test("correct types"):
            assert type_dict["id"] == str
            assert type_dict["location.latitude"] == float
            assert type_dict["location.longitude"] == float

    def test_run_metadata_types(self, run_metadata, subtests):
        """Test extracting type dict from Run metadata."""
        type_dict = get_metadata_type_dict(run_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "id" in type_dict
            assert "sample_rate" in type_dict

        with subtests.test("correct types"):
            assert type_dict["id"] == str
            assert type_dict["sample_rate"] == float

    def test_electric_metadata_types(self, electric_metadata, subtests):
        """Test extracting type dict from Electric metadata."""
        type_dict = get_metadata_type_dict(electric_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "component" in type_dict
            assert "sample_rate" in type_dict
            assert "measurement_azimuth" in type_dict

        with subtests.test("correct types"):
            assert type_dict["component"] == str
            assert type_dict["sample_rate"] == float
            assert type_dict["measurement_azimuth"] == float

    def test_magnetic_metadata_types(self, magnetic_metadata, subtests):
        """Test extracting type dict from Magnetic metadata."""
        type_dict = get_metadata_type_dict(magnetic_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "component" in type_dict
            assert "sample_rate" in type_dict
            assert "measurement_azimuth" in type_dict

        with subtests.test("correct types"):
            assert type_dict["component"] == str
            assert type_dict["sample_rate"] == float

    def test_auxiliary_metadata_types(self, auxiliary_metadata, subtests):
        """Test extracting type dict from Auxiliary metadata."""
        type_dict = get_metadata_type_dict(auxiliary_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "component" in type_dict
            assert "sample_rate" in type_dict
            assert "type" in type_dict

        with subtests.test("correct types"):
            assert type_dict["component"] == str
            assert type_dict["sample_rate"] == float
            assert type_dict["type"] == str

    def test_channel_metadata_types(self, channel_metadata, subtests):
        """Test extracting type dict from base Channel metadata."""
        type_dict = get_metadata_type_dict(channel_metadata)

        with subtests.test("returns dict"):
            assert isinstance(type_dict, dict)

        with subtests.test("has expected fields"):
            assert "component" in type_dict
            assert "sample_rate" in type_dict
            assert "channel_number" in type_dict

        with subtests.test("correct types"):
            assert type_dict["component"] == str
            assert type_dict["sample_rate"] == float
            assert type_dict["channel_number"] == int


# =============================================================================
# Test coerce_value_to_expected_type
# =============================================================================


class TestCoerceValueToExpectedType:
    """Test the coerce_value_to_expected_type function."""

    def test_none_returns_none(self, mock_logger):
        """Test that None values are returned as-is."""
        result = coerce_value_to_expected_type(mock_logger, "test_key", None, float)
        assert result is None

    def test_correct_type_returns_as_is(self, mock_logger, subtests):
        """Test that values of correct type are returned unchanged."""
        with subtests.test("float"):
            result = coerce_value_to_expected_type(mock_logger, "key", 256.0, float)
            assert result == 256.0
            assert isinstance(result, float)

        with subtests.test("int"):
            result = coerce_value_to_expected_type(mock_logger, "key", 5, int)
            assert result == 5
            assert isinstance(result, int)

        with subtests.test("str"):
            result = coerce_value_to_expected_type(mock_logger, "key", "test", str)
            assert result == "test"
            assert isinstance(result, str)

    def test_coerce_to_float(self, mock_logger, subtests):
        """Test coercing various types to float."""
        with subtests.test("int to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", 256, float)
            assert result == 256.0
            assert isinstance(result, float)

        with subtests.test("str to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", "256.5", float)
            assert result == 256.5
            assert isinstance(result, float)

        with subtests.test("numpy int to float"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", np.int32(100), float
            )
            assert result == 100.0
            assert isinstance(result, float)

        with subtests.test("numpy float to float"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", np.float64(99.9), float
            )
            assert result == 99.9
            assert isinstance(result, float)

        with subtests.test("single element list to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", [256.0], float)
            assert result == 256.0
            assert isinstance(result, float)

    def test_coerce_to_int(self, mock_logger, subtests):
        """Test coercing various types to int."""
        with subtests.test("float to int"):
            result = coerce_value_to_expected_type(mock_logger, "key", 5.0, int)
            assert result == 5
            assert isinstance(result, int)

        with subtests.test("str to int"):
            result = coerce_value_to_expected_type(mock_logger, "key", "42", int)
            assert result == 42
            assert isinstance(result, int)

        with subtests.test("numpy float to int"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", np.float64(10.0), int
            )
            assert result == 10
            assert isinstance(result, int)

        with subtests.test("single element list to int"):
            result = coerce_value_to_expected_type(mock_logger, "key", [7], int)
            assert result == 7
            assert isinstance(result, int)

    def test_coerce_to_str(self, mock_logger, subtests):
        """Test coercing various types to str."""
        with subtests.test("int to str"):
            result = coerce_value_to_expected_type(mock_logger, "key", 123, str)
            assert result == "123"
            assert isinstance(result, str)

        with subtests.test("float to str"):
            result = coerce_value_to_expected_type(mock_logger, "key", 45.6, str)
            assert result == "45.6"
            assert isinstance(result, str)

        with subtests.test("single element list to str"):
            result = coerce_value_to_expected_type(mock_logger, "key", ["test"], str)
            assert result == "test"
            assert isinstance(result, str)

    def test_coerce_to_bool(self, mock_logger, subtests):
        """Test coercing various types to bool."""
        with subtests.test("int 1 to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", 1, bool)
            assert result is True

        with subtests.test("int 0 to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", 0, bool)
            assert result is False

        with subtests.test("str 'true' to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", "true", bool)
            assert result is True

        with subtests.test("str 'false' to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", "false", bool)
            assert result is False

        with subtests.test("str '1' to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", "1", bool)
            assert result is True

        with subtests.test("str '0' to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", "0", bool)
            assert result is False

        with subtests.test("single element list to bool"):
            result = coerce_value_to_expected_type(mock_logger, "key", [1], bool)
            assert result is True

    def test_coerce_to_list(self, mock_logger, subtests):
        """Test coercing various types to list."""
        with subtests.test("json string to list"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", '["a", "b", "c"]', list
            )
            assert result == ["a", "b", "c"]
            assert isinstance(result, list)

        with subtests.test("comma-separated string to list"):
            result = coerce_value_to_expected_type(mock_logger, "key", "a, b, c", list)
            assert result == ["a", "b", "c"]
            assert isinstance(result, list)

        with subtests.test("tuple to list"):
            result = coerce_value_to_expected_type(mock_logger, "key", (1, 2, 3), list)
            assert result == [1, 2, 3]
            assert isinstance(result, list)

    def test_invalid_coercion_returns_original(self, mock_logger, subtests):
        """Test that invalid coercions return the original value."""
        with subtests.test("invalid string to float"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", "not_a_number", float
            )
            assert result == "not_a_number"

        with subtests.test("invalid string to int"):
            result = coerce_value_to_expected_type(
                mock_logger, "key", "not_an_int", int
            )
            assert result == "not_an_int"

    def test_expected_type_none_returns_original(self, mock_logger):
        """Test that None expected type returns original value."""
        result = coerce_value_to_expected_type(mock_logger, "key", 123, None)
        assert result == 123


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationWithMetadata:
    """Test type coercion integration with actual metadata objects."""

    @pytest.mark.parametrize(
        "metadata_obj,field,old_value,expected_type,expected_value",
        [
            # Electric metadata
            (metadata.Electric(), "sample_rate", "256.0", float, 256.0),
            (metadata.Electric(), "channel_number", 1.0, int, 1),
            (metadata.Electric(), "component", 123, str, "123"),
            # Magnetic metadata
            (metadata.Magnetic(), "sample_rate", "512.0", float, 512.0),
            (metadata.Magnetic(), "measurement_azimuth", "90", float, 90.0),
            # Auxiliary metadata
            (metadata.Auxiliary(), "sample_rate", "1.0", float, 1.0),
            (metadata.Auxiliary(), "channel_number", "5", int, 5),
            # Station metadata
            (metadata.Station(), "location.latitude", "45.5", float, 45.5),
            (metadata.Station(), "location.longitude", "-120.3", float, -120.3),
            (metadata.Station(), "location.elevation", "1500", float, 1500.0),
            # Run metadata
            (metadata.Run(), "sample_rate", "256", float, 256.0),
        ],
    )
    def test_realistic_coercion_scenarios(
        self, mock_logger, metadata_obj, field, old_value, expected_type, expected_value
    ):
        """Test realistic scenarios of coercing old file values."""
        # Get the expected type from metadata
        type_dict = get_metadata_type_dict(metadata_obj)
        field_type = type_dict.get(field, expected_type)

        # Coerce the value
        result = coerce_value_to_expected_type(
            mock_logger, field, old_value, field_type
        )

        assert result == expected_value
        assert isinstance(result, expected_type)

    def test_complete_workflow(self, mock_logger, electric_metadata, subtests):
        """Test complete workflow of getting types and coercing values."""
        # Step 1: Get type dictionary
        type_dict = get_metadata_type_dict(electric_metadata)

        with subtests.test("type dict created"):
            assert isinstance(type_dict, dict)
            assert "sample_rate" in type_dict

        # Step 2: Simulate old file data (all strings)
        old_file_data = {
            "sample_rate": "256.0",
            "channel_number": "1",
            "measurement_azimuth": "0.0",
            "component": "ex",
        }

        # Step 3: Coerce all values
        coerced_data = {}
        for key, value in old_file_data.items():
            expected_type = type_dict.get(key)
            coerced_data[key] = coerce_value_to_expected_type(
                mock_logger, key, value, expected_type
            )

        # Step 4: Verify coercions
        with subtests.test("sample_rate coerced to float"):
            assert isinstance(coerced_data["sample_rate"], float)
            assert coerced_data["sample_rate"] == 256.0

        with subtests.test("channel_number coerced to int"):
            assert isinstance(coerced_data["channel_number"], int)
            assert coerced_data["channel_number"] == 1

        with subtests.test("measurement_azimuth coerced to float"):
            assert isinstance(coerced_data["measurement_azimuth"], float)
            assert coerced_data["measurement_azimuth"] == 0.0

        with subtests.test("component remains str"):
            assert isinstance(coerced_data["component"], str)
            assert coerced_data["component"] == "ex"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_coercion(self, mock_logger, subtests):
        """Test coercion of empty strings."""
        with subtests.test("empty string to float returns original"):
            result = coerce_value_to_expected_type(mock_logger, "key", "", float)
            assert result == ""

        with subtests.test("empty string to int returns original"):
            result = coerce_value_to_expected_type(mock_logger, "key", "", int)
            assert result == ""

    def test_special_float_values(self, mock_logger, subtests):
        """Test coercion of special float values."""
        with subtests.test("inf string to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", "inf", float)
            assert np.isinf(result)

        with subtests.test("nan string to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", "nan", float)
            assert np.isnan(result)

    def test_numpy_array_coercion(self, mock_logger, subtests):
        """Test coercion of numpy arrays."""
        with subtests.test("single element array to float"):
            arr = np.array([256.0])
            result = coerce_value_to_expected_type(mock_logger, "key", arr, list)
            assert isinstance(result, list)

    def test_unicode_strings(self, mock_logger):
        """Test coercion with unicode strings."""
        unicode_str = "测试"
        result = coerce_value_to_expected_type(mock_logger, "key", unicode_str, str)
        assert result == unicode_str
        assert isinstance(result, str)

    def test_large_numbers(self, mock_logger, subtests):
        """Test coercion of large numbers."""
        with subtests.test("large int"):
            large_int = 2**63 - 1
            result = coerce_value_to_expected_type(mock_logger, "key", large_int, int)
            assert result == large_int
            assert isinstance(result, int)

        with subtests.test("large float"):
            large_float = 1e308
            result = coerce_value_to_expected_type(
                mock_logger, "key", large_float, float
            )
            assert result == large_float
            assert isinstance(result, float)

    def test_negative_numbers(self, mock_logger, subtests):
        """Test coercion of negative numbers."""
        with subtests.test("negative int string to int"):
            result = coerce_value_to_expected_type(mock_logger, "key", "-42", int)
            assert result == -42
            assert isinstance(result, int)

        with subtests.test("negative float string to float"):
            result = coerce_value_to_expected_type(mock_logger, "key", "-3.14", float)
            assert result == -3.14
            assert isinstance(result, float)
