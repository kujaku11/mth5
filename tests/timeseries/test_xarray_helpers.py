"""
Tests for xarray helper functions.
"""
import numpy as np
import xarray as xr
import pytest

from mth5.timeseries.xarray_helpers import covariance_xr, initialize_xrda_1d, initialize_xrda_2d


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


def test_initialize_xrda_2d():
    """Test initialization of 2D xarray DataArray."""
    channels = ["ex", "ey", "hx", "hy"]
    
    # Test with default parameters (complex)
    xrda = initialize_xrda_2d(channels)
    assert xrda.dims == ("channel_1", "channel_2")
    assert list(xrda.coords["channel_1"].values) == channels
    assert list(xrda.coords["channel_2"].values) == channels
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.zeros((len(channels), len(channels))))
    
    # Test with real value
    xrda = initialize_xrda_2d(channels, dtype=float, value=1.5)
    assert xrda.dtype == float
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels))) * 1.5)
    
    # Test with complex value
    xrda = initialize_xrda_2d(channels, dtype=complex, value=2+3j)
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels))) * (2+3j))
    
    # Test with different dimensions
    channels2 = ["hx", "hy"]
    xrda = initialize_xrda_2d(channels, dims=[channels, channels2], dtype=complex, value=1-1j)
    assert xrda.dims == ("channel_1", "channel_2")
    assert list(xrda.coords["channel_1"].values) == channels
    assert list(xrda.coords["channel_2"].values) == channels2
    assert xrda.shape == (len(channels), len(channels2))
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels2))) * (1-1j)) 