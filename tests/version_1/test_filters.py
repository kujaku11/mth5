# -*- coding: utf-8 -*-
"""
Modern pytest test suite for MTH5 filters functionality.

Translated from unittest to pytest with fixtures and subtests.

Created on Tue May 11 10:35:30 2021
Converted to pytest on November 4, 2025

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
from pathlib import Path
from typing import Generator, Tuple

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.timeseries.filters import CoefficientFilter, PoleZeroFilter

from mth5 import helpers
from mth5.mth5 import MTH5


# =============================================================================
# Constants and Utilities
# =============================================================================

fn_path = Path(__file__).parent
helpers.close_open_files()

# Test data constants
ZPK_TEST_DATA = {
    "units_in": "counts",
    "units_out": "mV",
    "name": "zpk_test",
    "poles": np.array([1 + 2j, 0, 1 - 2j]),
    "zeros": np.array([10 - 1j, 10 + 1j]),
}

COEFFICIENT_TEST_DATA = {
    "units_in": "volt",
    "units_out": "milliVolt per meter",
    "name": "coefficient_test",
    "gain": 10.0,
}

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_h5_file(make_worker_safe_path) -> Generator[Path, None, None]:
    """Session-scoped fixture providing the test HDF5 file path."""
    test_file = make_worker_safe_path("filter_test_pytest.h5", Path(__file__).parent)
    yield test_file
    # Cleanup
    if test_file.exists():
        test_file.unlink()


@pytest.fixture(scope="session")
def mth5_object(test_h5_file: Path) -> Generator[MTH5, None, None]:
    """Session-scoped fixture providing the MTH5 object."""
    helpers.close_open_files()
    m_obj = MTH5(file_version="0.1.0")
    m_obj.open_mth5(test_h5_file, "w")
    yield m_obj
    m_obj.close_mth5()


@pytest.fixture(scope="session")
def filter_group(mth5_object: MTH5):
    """Session-scoped fixture providing the filters group."""
    return mth5_object.filters_group


@pytest.fixture(scope="session")
def zpk_filter() -> PoleZeroFilter:
    """Session-scoped fixture providing a configured ZPK filter."""
    zpk = PoleZeroFilter()
    zpk.units_in = ZPK_TEST_DATA["units_in"]
    zpk.units_out = ZPK_TEST_DATA["units_out"]
    zpk.name = ZPK_TEST_DATA["name"]
    zpk.poles = ZPK_TEST_DATA["poles"]
    zpk.zeros = ZPK_TEST_DATA["zeros"]
    return zpk


@pytest.fixture(scope="session")
def coefficient_filter() -> CoefficientFilter:
    """Session-scoped fixture providing a configured coefficient filter."""
    coefficient = CoefficientFilter()
    coefficient.units_in = COEFFICIENT_TEST_DATA["units_in"]
    coefficient.units_out = COEFFICIENT_TEST_DATA["units_out"]
    coefficient.name = COEFFICIENT_TEST_DATA["name"]
    coefficient.gain = COEFFICIENT_TEST_DATA["gain"]
    return coefficient


@pytest.fixture(scope="session")
def zpk_group(filter_group, zpk_filter):
    """Session-scoped fixture providing the ZPK filter group."""
    return filter_group.add_filter(zpk_filter)


@pytest.fixture(scope="session")
def coefficient_group(filter_group, coefficient_filter):
    """Session-scoped fixture providing the coefficient filter group."""
    return filter_group.add_filter(coefficient_filter)


@pytest.fixture(scope="session")
def all_filters(zpk_group, coefficient_group) -> Tuple:
    """Session-scoped fixture ensuring both filters are created."""
    return zpk_group, coefficient_group


# =============================================================================
# Test Classes
# =============================================================================


class TestZPKFilter:
    """Test class for ZPK (Pole-Zero) filter functionality."""

    def test_zpk_in_groups_list(self, filter_group, zpk_filter, all_filters):
        """Test that ZPK filter is added to groups list."""
        assert zpk_filter.name in filter_group.zpk_group.groups_list

    def test_zpk_attributes(self, zpk_group, zpk_filter, all_filters, subtests):
        """Test ZPK filter attributes using subtests."""

        with subtests.test("name"):
            assert zpk_group.attrs["name"] == zpk_filter.name

        with subtests.test("units_in"):
            assert zpk_group.attrs["units_in"] == zpk_filter.units_in

        with subtests.test("units_out"):
            assert zpk_group.attrs["units_out"] == zpk_filter.units_out

    def test_zpk_complex_arrays(self, zpk_group, zpk_filter, all_filters, subtests):
        """Test ZPK filter complex array data using subtests."""

        with subtests.test("poles"):
            assert np.allclose(zpk_group["poles"][()], zpk_filter.poles)

        with subtests.test("zeros"):
            assert np.allclose(zpk_group["zeros"], zpk_filter.zeros)

    def test_zpk_roundtrip(self, filter_group, zpk_filter, all_filters):
        """Test ZPK filter roundtrip serialization."""
        new_zpk = filter_group.to_filter_object(zpk_filter.name)
        assert new_zpk == zpk_filter


class TestCoefficientFilter:
    """Test class for coefficient filter functionality."""

    def test_coefficient_in_groups_list(
        self, filter_group, coefficient_filter, all_filters
    ):
        """Test that coefficient filter is added to groups list."""
        assert coefficient_filter.name in filter_group.coefficient_group.groups_list

    def test_coefficient_attributes(
        self, coefficient_group, coefficient_filter, all_filters, subtests
    ):
        """Test coefficient filter attributes using subtests."""

        with subtests.test("name"):
            assert coefficient_group.attrs["name"] == coefficient_filter.name

        with subtests.test("units_in"):
            assert coefficient_group.attrs["units_in"] == coefficient_filter.units_in

        with subtests.test("units_out"):
            assert coefficient_group.attrs["units_out"] == coefficient_filter.units_out

    def test_coefficient_roundtrip(self, filter_group, coefficient_filter, all_filters):
        """Test coefficient filter roundtrip serialization."""
        new_coefficient = filter_group.to_filter_object(coefficient_filter.name)
        assert new_coefficient == coefficient_filter


class TestFilterIntegration:
    """Integration tests for filter functionality."""

    @pytest.mark.parametrize(
        "filter_type,expected_name",
        [
            ("zpk", ZPK_TEST_DATA["name"]),
            ("coefficient", COEFFICIENT_TEST_DATA["name"]),
        ],
    )
    def test_filter_retrieval_by_type(
        self, filter_group, filter_type, expected_name, all_filters
    ):
        """Test that filters can be retrieved by type."""
        if filter_type == "zpk":
            groups_list = filter_group.zpk_group.groups_list
        else:
            groups_list = filter_group.coefficient_group.groups_list

        assert expected_name in groups_list

    def test_multiple_filters_coexist(
        self, filter_group, zpk_filter, coefficient_filter, all_filters
    ):
        """Test that multiple filter types can coexist."""
        zpk_retrieved = filter_group.to_filter_object(zpk_filter.name)
        coeff_retrieved = filter_group.to_filter_object(coefficient_filter.name)

        assert zpk_retrieved == zpk_filter
        assert coeff_retrieved == coefficient_filter

    def test_filter_independence(
        self, filter_group, zpk_filter, coefficient_filter, all_filters
    ):
        """Test that filters maintain independence."""
        # Verify different filter types have different properties
        assert zpk_filter.name != coefficient_filter.name
        assert zpk_filter.units_in != coefficient_filter.units_in
        assert zpk_filter.units_out != coefficient_filter.units_out


class TestFilterPerformance:
    """Performance and efficiency tests for filter operations."""

    @pytest.mark.performance
    def test_filter_creation_efficiency(self, filter_group):
        """Test that filter creation is efficient."""
        import time

        # Create multiple filters and measure performance
        start_time = time.time()

        for i in range(10):
            zpk = PoleZeroFilter()
            zpk.name = f"perf_test_zpk_{i}"
            zpk.poles = np.array([1 + i * 1j, 0, 1 - i * 1j])
            zpk.zeros = np.array([10 - i * 1j, 10 + i * 1j])
            filter_group.add_filter(zpk)

        creation_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert (
            creation_time < 5.0
        ), f"Filter creation took too long: {creation_time:.2f}s"

    @pytest.mark.performance
    def test_filter_retrieval_efficiency(
        self, filter_group, zpk_filter, coefficient_filter, all_filters
    ):
        """Test that filter retrieval is efficient."""
        import time

        start_time = time.time()

        # Retrieve filters multiple times
        for _ in range(100):
            zpk_retrieved = filter_group.to_filter_object(zpk_filter.name)
            coeff_retrieved = filter_group.to_filter_object(coefficient_filter.name)

        retrieval_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            retrieval_time < 3.0
        ), f"Filter retrieval took too long: {retrieval_time:.2f}s"


# =============================================================================
# Standalone Test Functions
# =============================================================================


def test_filter_constants():
    """Test that filter test constants are properly defined."""
    assert "name" in ZPK_TEST_DATA
    assert "name" in COEFFICIENT_TEST_DATA
    assert ZPK_TEST_DATA["name"] != COEFFICIENT_TEST_DATA["name"]


def test_numpy_array_properties():
    """Test numpy array properties used in filters."""
    poles = ZPK_TEST_DATA["poles"]
    zeros = ZPK_TEST_DATA["zeros"]

    # Test complex number properties
    assert np.iscomplexobj(poles)
    assert np.iscomplexobj(zeros)

    # Test array shapes
    assert poles.shape == (3,)
    assert zeros.shape == (2,)


@pytest.mark.parametrize(
    "filter_class,expected_type",
    [(PoleZeroFilter, PoleZeroFilter), (CoefficientFilter, CoefficientFilter)],
)
def test_filter_instantiation(filter_class, expected_type):
    """Test that filter classes can be instantiated correctly."""
    filter_instance = filter_class()
    assert isinstance(filter_instance, expected_type)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestFilterErrorHandling:
    """Test error handling in filter operations."""

    def test_invalid_filter_name_retrieval(self, filter_group, all_filters):
        """Test handling of invalid filter name during retrieval."""
        with pytest.raises((KeyError, AttributeError)):
            filter_group.to_filter_object("nonexistent_filter")

    def test_duplicate_filter_name(self, filter_group, zpk_filter, all_filters):
        """Test behavior when adding filters with duplicate names."""
        # Create another filter with the same name
        duplicate_zpk = PoleZeroFilter()
        duplicate_zpk.name = zpk_filter.name
        duplicate_zpk.poles = np.array([2 + 3j, 1, 2 - 3j])
        duplicate_zpk.zeros = np.array([5 - 2j, 5 + 2j])

        # The behavior may vary - either replace or raise error
        # This test documents the expected behavior
        try:
            filter_group.add_filter(duplicate_zpk)
            # If no error, verify the filter was updated or kept original
            retrieved = filter_group.to_filter_object(zpk_filter.name)
            assert retrieved.name == zpk_filter.name
        except Exception as e:
            # If error is expected, verify it's the right type
            assert isinstance(e, (ValueError, KeyError))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
