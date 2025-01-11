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
    data = np.random.randn(len(times), len(channels))
    
    # Create xarray DataArray
    X = xr.DataArray(
        data,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )
    
    # Compute covariance
    S = covariance_xr(X)
    
    # Check dimensions and coordinates
    assert S.dims == ("channel_1", "channel_2")
    assert list(S.coords["channel_1"].values) == channels
    assert list(S.coords["channel_2"].values) == channels
    
    # Check symmetry
    assert np.allclose(S.values, S.values.T)
    
    # Check against numpy covariance
    np_cov = np.cov(X.values.T)
    assert np.allclose(S.values, np_cov)


def test_covariance_xr_with_weights():
    """Test weighted covariance computation with xarray."""
    # Create sample data
    channels = ["ex", "ey", "hx", "hy"]
    times = np.arange(100)
    data = np.random.randn(len(times), len(channels))
    weights = np.random.rand(len(times))
    
    # Create xarray DataArray
    X = xr.DataArray(
        data,
        dims=["time", "variable"],
        coords={
            "time": times,
            "variable": channels
        }
    )
    
    # Compute weighted covariance
    S = covariance_xr(X, aweights=weights)
    
    # Check dimensions and coordinates
    assert S.dims == ("channel_1", "channel_2")
    assert list(S.coords["channel_1"].values) == channels
    assert list(S.coords["channel_2"].values) == channels
    
    # Check symmetry
    assert np.allclose(S.values, S.values.T)
    
    # Check against numpy weighted covariance
    np_cov = np.cov(X.values.T, aweights=weights)
    assert np.allclose(S.values, np_cov)


def test_initialize_xrda_1d():
    """Test initialization of 1D xarray DataArray."""
    # Test with default parameters
    channels = ["ex", "ey", "hx", "hy"]
    xrda = initialize_xrda_1d(channels)
    
    assert xrda.dims == ("variable",)
    assert list(xrda.coords["variable"].values) == channels
    assert np.allclose(xrda.values, np.zeros(len(channels)))
    
    # Test with custom dtype and value
    xrda = initialize_xrda_1d(channels, dtype=complex, value=1+1j)
    
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.ones(len(channels)) * (1+1j))
    
    # Test with boolean type
    xrda = initialize_xrda_1d(channels, dtype=bool, value=True)
    
    assert xrda.dtype == bool
    assert np.all(xrda.values == True)


def test_initialize_xrda_2d():
    """Test initialization of 2D xarray DataArray."""
    # Test with default parameters
    channels = ["ex", "ey", "hx", "hy"]
    xrda = initialize_xrda_2d(channels)
    
    assert xrda.dims == ("channel_1", "channel_2")
    assert list(xrda.coords["channel_1"].values) == channels
    assert list(xrda.coords["channel_2"].values) == channels
    assert xrda.dtype == complex
    assert np.allclose(xrda.values, np.zeros((len(channels), len(channels))))
    
    # Test with custom dtype and value
    xrda = initialize_xrda_2d(channels, dtype=float, value=1.5)
    
    assert xrda.dtype == float
    assert np.allclose(xrda.values, np.ones((len(channels), len(channels))) * 1.5)
    
    # Test with different dimensions
    channels2 = ["hx", "hy"]
    xrda = initialize_xrda_2d(channels, dims=[channels, channels2])
    
    assert xrda.dims == ("channel_1", "channel_2")
    assert list(xrda.coords["channel_1"].values) == channels
    assert list(xrda.coords["channel_2"].values) == channels2
    assert xrda.shape == (len(channels), len(channels2)) 