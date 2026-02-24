# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for FIRGroup (FIR filter group) functionality.

This test suite covers:
- FIR filter group initialization
- Adding filters with coefficients
- Retrieving filters
- Converting to/from FIRFilter objects
- Filter dictionary management
- Edge cases and error handling

Created on January 13, 2026

@author: pytest suite
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from mt_metadata.timeseries.filters import FIRFilter

from mth5.groups.filter_groups.fir_filter_group import FIRGroup
from mth5.mth5 import MTH5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test_fir_group.h5"
    yield test_file
    # Cleanup
    if test_file.exists():
        test_file.unlink()
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mth5_object(temp_h5_file):
    """Create MTH5 object for testing."""
    m = MTH5(file_version="0.2.0")
    m.open_mth5(temp_h5_file, "w")
    yield m
    m.close_mth5()


@pytest.fixture
def survey_group(mth5_object):
    """Create survey group for filter testing."""
    return mth5_object.add_survey("test_survey")


@pytest.fixture
def fir_group(survey_group):
    """Create FIR filter group."""
    filters_group = survey_group.filters_group
    return filters_group.fir_group


@pytest.fixture
def sample_coefficients():
    """Provide sample FIR coefficients."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])


@pytest.fixture
def sample_fir_metadata():
    """Provide sample FIR filter metadata."""
    return {
        "type": "fir",
        "name": "test_fir",
        "units_in": "digital counts",
        "units_out": "milliVolt",
        "gain": 1.0,
    }


@pytest.fixture
def fir_filter_object():
    """Create a FIRFilter object for testing."""
    fir = FIRFilter()
    fir.name = "test_fir_filter"
    fir.units_in = "digital counts"
    fir.units_out = "milliVolt"
    fir.gain = 1.0
    fir.coefficients = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])
    return fir


@pytest.fixture
def added_fir_group(fir_group, sample_coefficients, sample_fir_metadata):
    """Create a FIR group with one filter already added."""
    fir_group.add_filter("test_filter", sample_coefficients, sample_fir_metadata)
    return fir_group


# =============================================================================
# Test Classes
# =============================================================================


class TestFIRGroupInitialization:
    """Test FIRGroup initialization and basic properties."""

    def test_fir_group_initialization_complete(self, fir_group, subtests):
        """Test that FIR group is properly initialized with correct state."""
        with subtests.test("instance_type"):
            assert isinstance(fir_group, FIRGroup)

        with subtests.test("has_hdf5_group"):
            assert hasattr(fir_group, "hdf5_group")

        with subtests.test("empty_initially"):
            assert len(fir_group.hdf5_group) == 0

        with subtests.test("filter_dict_empty"):
            assert fir_group.filter_dict == {}


class TestFIRGroupAddFilter:
    """Test adding FIR filters to the group."""

    def test_add_filter_basic(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test adding a basic FIR filter."""
        filter_name = "test_filter_01"

        result = fir_group.add_filter(
            filter_name, sample_coefficients, sample_fir_metadata
        )

        assert result is not None
        assert filter_name in fir_group.hdf5_group

    def test_add_filter_stores_coefficients(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test that coefficients are correctly stored."""
        filter_name = "coef_test"

        fir_group.add_filter(filter_name, sample_coefficients, sample_fir_metadata)

        stored_coefficients = fir_group.hdf5_group[filter_name]["coefficients"][:]
        np.testing.assert_array_almost_equal(stored_coefficients, sample_coefficients)

    def test_add_filter_stores_metadata(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test that metadata attributes are correctly stored."""
        filter_name = "meta_test"

        fir_group.add_filter(filter_name, sample_coefficients, sample_fir_metadata)

        filter_h5 = fir_group.hdf5_group[filter_name]
        stored_metadata = {k: filter_h5.attrs[k] for k in sample_fir_metadata.keys()}
        assert stored_metadata == sample_fir_metadata

    @pytest.mark.parametrize(
        "coef_array",
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.linspace(-1, 1, 20),
            np.array(
                [0.23, -0.45, 0.67, -0.12, 0.89, -0.34, 0.56, -0.78, 0.01, -0.99]
            ),  # Fixed values instead of random
        ],
    )
    def test_add_filter_various_coefficient_arrays(
        self, fir_group, sample_fir_metadata, coef_array
    ):
        """Test adding filters with various coefficient arrays."""
        filter_name = f"filter_{len(coef_array)}"

        fir_group.add_filter(filter_name, coef_array, sample_fir_metadata)

        stored = fir_group.hdf5_group[filter_name]["coefficients"][:]
        np.testing.assert_array_almost_equal(stored, coef_array)

    def test_add_multiple_filters(self, fir_group, sample_coefficients):
        """Test adding multiple filters to the same group."""
        num_filters = 5

        for i in range(num_filters):
            metadata = {"name": f"filter_{i}", "type": "fir"}
            fir_group.add_filter(f"filter_{i}", sample_coefficients, metadata)

        assert len(fir_group.hdf5_group) == num_filters

    def test_add_filter_returns_group(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test that add_filter returns the HDF5 group."""
        result = fir_group.add_filter("test", sample_coefficients, sample_fir_metadata)

        # Check it's an HDF5 group
        assert hasattr(result, "attrs")
        assert hasattr(result, "keys")


class TestFIRGroupFilterDict:
    """Test filter_dict property functionality."""

    def test_filter_dict_contains_added_filter(self, added_fir_group):
        """Test that filter_dict includes added filters."""
        filter_dict = added_fir_group.filter_dict

        assert "test_filter" in filter_dict

    def test_filter_dict_structure(self, added_fir_group):
        """Test the structure of filter_dict entries."""
        filter_dict = added_fir_group.filter_dict

        filter_info = filter_dict["test_filter"]
        assert "type" in filter_info
        assert "hdf5_ref" in filter_info

    def test_filter_dict_type_field(self, added_fir_group, sample_fir_metadata):
        """Test that type field is correctly stored in filter_dict."""
        filter_dict = added_fir_group.filter_dict

        assert filter_dict["test_filter"]["type"] == sample_fir_metadata["type"]

    def test_filter_dict_with_multiple_filters(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test filter_dict with multiple filters."""
        filter_names = ["filter_a", "filter_b", "filter_c"]

        for name in filter_names:
            fir_group.add_filter(name, sample_coefficients, sample_fir_metadata)

        filter_dict = fir_group.filter_dict

        assert len(filter_dict) == len(filter_names)
        for name in filter_names:
            assert name in filter_dict


class TestFIRGroupGetFilter:
    """Test retrieving filters from the group."""

    def test_get_filter_returns_group(self, added_fir_group):
        """Test that get_filter returns an HDF5 group."""
        result = added_fir_group.get_filter("test_filter")

        assert hasattr(result, "attrs")
        assert "coefficients" in result

    def test_get_filter_with_valid_name(self, added_fir_group, sample_coefficients):
        """Test retrieving filter with valid name."""
        filter_h5 = added_fir_group.get_filter("test_filter")

        stored_coef = filter_h5["coefficients"][:]
        np.testing.assert_array_almost_equal(stored_coef, sample_coefficients)

    def test_get_filter_nonexistent_raises_error(self, fir_group):
        """Test that getting non-existent filter raises KeyError."""
        with pytest.raises(KeyError):
            fir_group.get_filter("nonexistent_filter")

    @pytest.mark.parametrize("filter_name", ["filter_1", "my_fir", "lowpass_128"])
    def test_get_filter_various_names(
        self, fir_group, sample_coefficients, sample_fir_metadata, filter_name
    ):
        """Test getting filters with various names."""
        fir_group.add_filter(filter_name, sample_coefficients, sample_fir_metadata)

        result = fir_group.get_filter(filter_name)

        assert result is not None
        assert "coefficients" in result


class TestFIRGroupFromObject:
    """Test creating filters from FIRFilter objects."""

    def test_from_object_creates_filter(self, fir_group, fir_filter_object):
        """Test that from_object creates a filter in the group."""
        result = fir_group.from_object(fir_filter_object)

        assert fir_filter_object.name in fir_group.hdf5_group
        assert result is not None

    def test_from_object_stores_coefficients(self, fir_group, fir_filter_object):
        """Test that coefficients are correctly stored from object."""
        fir_group.from_object(fir_filter_object)

        stored = fir_group.hdf5_group[fir_filter_object.name]["coefficients"][:]
        np.testing.assert_array_almost_equal(stored, fir_filter_object.coefficients)

    def test_from_object_stores_metadata(self, fir_group, fir_filter_object, subtests):
        """Test that metadata is correctly stored from object."""
        fir_group.from_object(fir_filter_object)

        filter_h5 = fir_group.hdf5_group[fir_filter_object.name]

        with subtests.test("name"):
            assert filter_h5.attrs["name"] == fir_filter_object.name

        with subtests.test("units_in"):
            assert filter_h5.attrs["units_in"] == fir_filter_object.units_in

        with subtests.test("units_out"):
            assert filter_h5.attrs["units_out"] == fir_filter_object.units_out

    def test_from_object_rejects_wrong_type(self, fir_group):
        """Test that from_object raises TypeError for wrong filter type."""
        wrong_filter = {"name": "test", "type": "wrong"}

        with pytest.raises(
            TypeError, match="Filter must be a FrequencyResponseTableFilter"
        ):
            fir_group.from_object(wrong_filter)

    def test_from_object_handles_none_values(self, fir_group):
        """Test that from_object handles None values in metadata."""
        fir = FIRFilter()
        fir.name = "test_none"
        fir.coefficients = np.array([1.0, 2.0, 3.0])
        # Leave some attributes as None
        fir.units_in = None
        fir.units_out = None

        result = fir_group.from_object(fir)

        assert result is not None
        # None values should be converted to strings
        filter_h5 = fir_group.hdf5_group["test_none"]
        assert filter_h5 is not None
        assert "units_in" in filter_h5.attrs

    @pytest.mark.parametrize(
        "coefficients",
        [
            np.array([1.0]),
            np.array([0.5, 0.5]),
            np.ones(100),
        ],
    )
    def test_from_object_various_coefficient_sizes(self, fir_group, coefficients):
        """Test from_object with various coefficient array sizes."""
        fir = FIRFilter()
        fir.name = f"test_{len(coefficients)}"
        fir.coefficients = coefficients

        fir_group.from_object(fir)

        stored = fir_group.hdf5_group[fir.name]["coefficients"][:]
        np.testing.assert_array_equal(stored, coefficients)


class TestFIRGroupToObject:
    """Test converting filters to FIRFilter objects."""

    def test_to_object_creates_fir_filter(self, added_fir_group):
        """Test that to_object creates a FIRFilter object."""
        result = added_fir_group.to_object("test_filter")

        assert isinstance(result, FIRFilter)

    def test_to_object_preserves_coefficients(
        self, added_fir_group, sample_coefficients
    ):
        """Test that coefficients are correctly retrieved."""
        fir_obj = added_fir_group.to_object("test_filter")

        np.testing.assert_array_almost_equal(fir_obj.coefficients, sample_coefficients)

    def test_to_object_preserves_metadata(
        self, added_fir_group, sample_fir_metadata, subtests
    ):
        """Test that metadata is correctly retrieved."""
        fir_obj = added_fir_group.to_object("test_filter")

        with subtests.test("name"):
            assert fir_obj.name == sample_fir_metadata["name"]

        with subtests.test("units_in"):
            assert fir_obj.units_in == sample_fir_metadata["units_in"]

        with subtests.test("units_out"):
            assert fir_obj.units_out == sample_fir_metadata["units_out"]

    def test_to_object_round_trip(self, fir_group, fir_filter_object):
        """Test round-trip: object -> HDF5 -> object."""
        # Add from object
        fir_group.from_object(fir_filter_object)

        # Retrieve back to object
        retrieved = fir_group.to_object(fir_filter_object.name)

        # Compare
        assert retrieved.name == fir_filter_object.name
        np.testing.assert_array_almost_equal(
            retrieved.coefficients, fir_filter_object.coefficients
        )

    def test_to_object_nonexistent_raises_error(self, fir_group):
        """Test that to_object raises error for non-existent filter."""
        with pytest.raises(KeyError):
            fir_group.to_object("nonexistent")

    def test_to_object_handles_missing_coefficients(
        self, fir_group, sample_fir_metadata
    ):
        """Test handling of filter with no coefficients dataset."""
        # Add filter without proper coefficients setup
        filter_name = "no_coef"
        filter_h5 = fir_group.hdf5_group.create_group(filter_name)
        filter_h5.attrs.update(sample_fir_metadata)
        # Don't create coefficients dataset

        # Implementation raises KeyError for missing coefficients
        with pytest.raises(KeyError, match="coefficients"):
            fir_group.to_object(filter_name)


class TestFIRGroupEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_coefficients_array(self, fir_group, sample_fir_metadata):
        """Test adding filter with empty coefficients."""
        empty_coef = np.array([])

        result = fir_group.add_filter("empty", empty_coef, sample_fir_metadata)

        assert result is not None

    def test_single_coefficient(self, fir_group, sample_fir_metadata):
        """Test adding filter with single coefficient."""
        single_coef = np.array([1.0])

        fir_group.add_filter("single", single_coef, sample_fir_metadata)

        stored = fir_group.hdf5_group["single"]["coefficients"][:]
        np.testing.assert_array_equal(stored, single_coef)

    def test_large_coefficient_array(self, fir_group, sample_fir_metadata):
        """Test adding filter with large coefficient array."""
        np.random.seed(42)  # Use fixed seed for reproducibility
        large_coef = np.random.randn(10000)

        fir_group.add_filter("large", large_coef, sample_fir_metadata)

        stored = fir_group.hdf5_group["large"]["coefficients"][:]
        np.testing.assert_array_almost_equal(stored, large_coef, decimal=5)

    def test_special_float_values(self, fir_group, sample_fir_metadata):
        """Test coefficients with special float values."""
        special_coef = np.array([0.0, -0.0, 1e-10, 1e10, -1e10])

        fir_group.add_filter("special", special_coef, sample_fir_metadata)

        stored = fir_group.hdf5_group["special"]["coefficients"][:]
        np.testing.assert_array_almost_equal(stored, special_coef, decimal=5)

    def test_metadata_with_empty_strings(self, fir_group, sample_coefficients):
        """Test metadata with empty string values."""
        metadata = {
            "name": "",
            "type": "fir",
            "units_in": "",
            "units_out": "",
        }

        result = fir_group.add_filter("empty_strings", sample_coefficients, metadata)

        assert result is not None

    def test_unicode_filter_names(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test filter names with unicode characters."""
        unicode_names = ["filter_α", "filter_β", "测试"]

        for name in unicode_names:
            try:
                fir_group.add_filter(name, sample_coefficients, sample_fir_metadata)
                # If it works, verify it's stored
                assert name in fir_group.hdf5_group
            except (ValueError, TypeError):
                # Some HDF5 versions may not support unicode
                pytest.skip("HDF5 version doesn't support unicode names")


class TestFIRGroupIntegration:
    """Integration tests combining multiple operations."""

    def test_add_retrieve_convert_workflow(
        self, fir_group, sample_coefficients, sample_fir_metadata
    ):
        """Test complete workflow: add, retrieve, convert."""
        filter_name = "workflow_test"

        # Add filter
        fir_group.add_filter(filter_name, sample_coefficients, sample_fir_metadata)

        # Retrieve as HDF5 group
        filter_h5 = fir_group.get_filter(filter_name)
        assert filter_h5 is not None

        # Convert to object
        fir_obj = fir_group.to_object(filter_name)
        assert isinstance(fir_obj, FIRFilter)
        np.testing.assert_array_almost_equal(fir_obj.coefficients, sample_coefficients)

    def test_multiple_filters_workflow(self, fir_group):
        """Test working with multiple filters simultaneously."""
        np.random.seed(123)  # Fixed seed for reproducibility
        filters_data = [
            ("lpf_128", np.random.randn(51), {"type": "fir", "name": "lpf_128"}),
            ("hpf_128", np.random.randn(51), {"type": "fir", "name": "hpf_128"}),
            ("bpf_128", np.random.randn(51), {"type": "fir", "name": "bpf_128"}),
        ]

        # Add all filters
        for name, coef, meta in filters_data:
            fir_group.add_filter(name, coef, meta)

        # Verify all in filter_dict
        filter_dict = fir_group.filter_dict
        assert len(filter_dict) == 3

        # Convert all to objects
        for name, coef, _ in filters_data:
            fir_obj = fir_group.to_object(name)
            np.testing.assert_array_almost_equal(fir_obj.coefficients, coef)

    def test_filter_persistence_across_close_reopen(
        self, temp_h5_file, sample_coefficients, sample_fir_metadata
    ):
        """Test that filters persist when file is closed and reopened."""
        filter_name = "persistent_filter"

        # Create and add filter
        m1 = MTH5(file_version="0.2.0")
        m1.open_mth5(temp_h5_file, "w")
        survey = m1.add_survey("test")
        fir_group = survey.filters_group.fir_group
        fir_group.add_filter(filter_name, sample_coefficients, sample_fir_metadata)
        m1.close_mth5()

        # Reopen and verify
        m2 = MTH5(file_version="0.2.0")
        m2.open_mth5(temp_h5_file, "r")
        survey2 = m2.surveys_group.get_survey("test")
        fir_group2 = survey2.filters_group.fir_group

        assert filter_name in fir_group2.hdf5_group
        retrieved = fir_group2.to_object(filter_name)
        np.testing.assert_array_almost_equal(
            retrieved.coefficients, sample_coefficients
        )

        m2.close_mth5()


class TestFIRGroupParametrized:
    """Parametrized tests covering multiple scenarios."""

    @pytest.mark.parametrize(
        "filter_config",
        [
            {
                "name": "lowpass_1",
                "units_in": "digital counts",
                "units_out": "milliVolt",
                "gain": 1.0,
            },
            {
                "name": "highpass_1",
                "units_in": "Volt",
                "units_out": "nanoTesla",
                "gain": 2.5,
            },
            {
                "name": "bandpass_1",
                "units_in": "digital counts",
                "units_out": "milliVolt",
                "gain": 0.1,
            },
        ],
    )
    def test_various_filter_configurations(
        self, fir_group, sample_coefficients, filter_config
    ):
        """Test adding filters with various configurations."""
        config = filter_config.copy()  # Don't mutate parametrized data
        config["type"] = "fir"

        fir_group.add_filter(config["name"], sample_coefficients, config)

        fir_obj = fir_group.to_object(config["name"])

        assert fir_obj.name == config["name"]
        assert fir_obj.units_in == config["units_in"]
        assert fir_obj.units_out == config["units_out"]

    @pytest.mark.parametrize("num_coefficients", [1, 10, 51, 101, 501])
    def test_various_coefficient_lengths(
        self, fir_group, sample_fir_metadata, num_coefficients
    ):
        """Test filters with various coefficient array lengths."""
        np.random.seed(
            num_coefficients
        )  # Unique seed per parameter for reproducibility
        coefficients = np.random.randn(num_coefficients)
        filter_name = f"fir_{num_coefficients}"

        fir_group.add_filter(filter_name, coefficients, sample_fir_metadata)

        retrieved = fir_group.to_object(filter_name)
        np.testing.assert_array_almost_equal(retrieved.coefficients, coefficients)
