"""
Tests for xarray helper functions.
"""
import numpy as np
import xarray as xr
import pytest

from mth5.timeseries.xarray_helpers import (
    covariance_xr,
    initialize_xrda_1d,
    initialize_xrda_2d,
    initialize_xrda_2d_cov,
    initialize_xrds_2d,
)


def test_covariance_xr():
    """Test covariance computation with xarray."""
    # Create sample data
    channels = ["ex", "ey", "hx", "hy"]
    times = np.arange(100)

    # Test with real data
    data_real = np.random.randn(len(times), len(channels))
    X_real = xr.DataArray(
        data_real,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )

    # Compute covariance for real data
    S_real = covariance_xr(X_real)

    # Check dimensions and coordinates
    assert S_real.dims == ("channel_1", "channel_2")
    assert list(S_real.coords["channel_1"].values) == channels
    assert list(S_real.coords["channel_2"].values) == channels

    # Check symmetry
    assert np.allclose(S_real.values, S_real.values.T)

    # Check against numpy covariance
    np_cov_real = np.cov(X_real.values.T)
    assert np.allclose(S_real.values, np_cov_real)

    # Test with complex data
    data_complex = np.random.randn(len(times), len(channels)) + 1j * np.random.randn(len(times), len(channels))
    X_complex = xr.DataArray(
        data_complex,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )

    # Compute covariance for complex data
    S_complex = covariance_xr(X_complex)

    # Check dimensions and coordinates
    assert S_complex.dims == ("channel_1", "channel_2")
    assert list(S_complex.coords["channel_1"].values) == channels
    assert list(S_complex.coords["channel_2"].values) == channels

    # Check Hermitian symmetry (conjugate transpose)
    assert np.allclose(S_complex.values, S_complex.values.conj().T)

    # Check against numpy covariance
    np_cov_complex = np.cov(X_complex.values.T)
    assert np.allclose(S_complex.values, np_cov_complex)


def test_covariance_xr_with_weights():
    """Test weighted covariance computation with xarray."""
    # Create sample data
    channels = ["ex", "ey", "hx", "hy"]
    times = np.arange(100)
    weights = np.random.rand(len(times))

    # Test with real data
    data_real = np.random.randn(len(times), len(channels))
    X_real = xr.DataArray(
        data_real,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )

    # Compute weighted covariance for real data
    S_real = covariance_xr(X_real, aweights=weights)

    # Check dimensions and coordinates
    assert S_real.dims == ("channel_1", "channel_2")
    assert list(S_real.coords["channel_1"].values) == channels
    assert list(S_real.coords["channel_2"].values) == channels

    # Check symmetry
    assert np.allclose(S_real.values, S_real.values.T)

    # Check against numpy weighted covariance
    np_cov_real = np.cov(X_real.values.T, aweights=weights)
    assert np.allclose(S_real.values, np_cov_real)

    # Test with complex data
    data_complex = np.random.randn(len(times), len(channels)) + 1j * np.random.randn(len(times), len(channels))
    X_complex = xr.DataArray(
        data_complex,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )

    # Compute weighted covariance for complex data
    S_complex = covariance_xr(X_complex, aweights=weights)

    # Check dimensions and coordinates
    assert S_complex.dims == ("channel_1", "channel_2")
    assert list(S_complex.coords["channel_1"].values) == channels
    assert list(S_complex.coords["channel_2"].values) == channels

    # Check Hermitian symmetry (conjugate transpose)
    assert np.allclose(S_complex.values, S_complex.values.conj().T)

    # Check against numpy weighted covariance
    np_cov_complex = np.cov(X_complex.values.T, aweights=weights)
    assert np.allclose(S_complex.values, np_cov_complex)


def test_initialize_xrda_1d():
    """Test initialization of 1D xarray DataArray."""
    channels = ["ex", "ey", "hx", "hy"]

    # Test with default parameters (real)
    xrda = initialize_xrda_1d(channels)
    assert xrda.dims == ("variable",)
    assert list(xrda.coords["variable"].values) == channels
    assert np.allclose(xrda.values, np.zeros(len(channels)))

    # Test with real value
    xrda = initialize_xrda_1d(channels, dtype=float, value=1.5)
    assert xrda.dtype == float
    assert np.allclose(xrda.values, np.ones(len(channels)) * 1.5)

    # Test with complex value
    xrda = initialize_xrda_1d(channels, dtype=complex, value=1+1j)
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.ones(len(channels)) * (1+1j))

    # Test with boolean type
    xrda = initialize_xrda_1d(channels, dtype=bool, value=True)
    assert xrda.dtype == bool
    assert np.all(xrda.values == True)


def test_initialize_xrds_2d():
    """Test initialization of 2D xarray Dataset with frequency and time dimensions"""
    # Create test data
    frequencies = np.array([0.1, 1.0])  # Two frequencies
    times = np.linspace(0, 100, 100)    # 100 time points
    variables = ['var1', 'var2']        # Two variables

    coords = {
        'frequency': frequencies,
        'time': times
    }

    # Initialize dataset
    xrds = initialize_xrds_2d(variables, coords)

    # Test that it's a Dataset
    assert isinstance(xrds, xr.Dataset)

    # Test variables
    assert list(xrds.data_vars.keys()) == variables

    # Test each variable's dimensions and coordinates
    for var in variables:
        assert xrds[var].dims == ('frequency', 'time')
        assert np.array_equal(xrds[var].frequency.values, frequencies)
        assert np.array_equal(xrds[var].time.values, times)
        assert np.iscomplexobj(xrds[var].values)
        assert np.all(xrds[var].values == 0)

    # Test with non-zero value
    xrds_ones = initialize_xrds_2d(variables, coords, value=1+1j)
    for var in variables:
        assert np.all(xrds_ones[var].values == 1+1j)


def test_initialize_xrda_2d():
    """Test initialization of 3D xarray DataArray with frequency, time, and variable dimensions"""
    # Create test data
    frequencies = np.array([0.1, 1.0, 10.0])  # Two frequencies
    times = np.linspace(0, 100, 100)    # 100 time points
    variables = ['var1', 'var2']        # Two variables

    coords = {
        'time': times,
        'frequency': frequencies,
    }

    # Initialize array
    xrda = initialize_xrda_2d(variables, coords)

    # Test dimensions and coordinates
    assert xrda.dims == ('time', 'frequency', 'variable')
    assert np.array_equal(xrda.frequency.values, frequencies)
    assert np.array_equal(xrda.time.values, times)
    assert np.array_equal(xrda.coords['variable'].values, variables)

    # Test data type and initialization
    assert np.iscomplexobj(xrda.values)
    assert np.all(xrda.values == 0)

    # Test with non-zero value
    xrda_ones = initialize_xrda_2d(variables, coords, value=1+1j)
    assert np.all(xrda_ones.values == 1+1j)

    # Test conversion from Dataset maintains data
    xrds = initialize_xrds_2d(variables, coords, value=2+3j)
    xrda_from_ds = xrds.to_array(dim='variable')
    assert np.array_equal(xrda_from_ds.coords['variable'].values, variables)
    assert np.all(xrda_from_ds.values == 2+3j)


def test_initialize_xrda_2d_cov():
    """Test initialization of 2D covariance xarray DataArray."""
    channels = ["ex", "ey", "hx", "hy"]

    # Test with default parameters (complex)
    xrda = initialize_xrda_2d_cov(channels)
    assert xrda.dims == ('channel_1', 'channel_2')
    assert list(xrda.coords['channel_1'].values) == channels
    assert list(xrda.coords['channel_2'].values) == channels
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.zeros((len(channels), len(channels))))

    # Test with real value
    xrda = initialize_xrda_2d_cov(channels, dtype=float, value=1.5)
    assert xrda.dtype == float
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels))) * 1.5)

    # Test with complex value
    xrda = initialize_xrda_2d_cov(channels, dtype=complex, value=2+3j)
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels))) * (2+3j))
