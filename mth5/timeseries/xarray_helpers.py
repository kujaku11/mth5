"""
Module containing helper functions for working with xarray objects.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union


def covariance_xr(
    X: xr.DataArray, aweights: Optional[Union[np.ndarray, None]] = None
) -> xr.DataArray:
    """
    Compute the covariance matrix with numpy.cov.

    Parameters
    ----------
    X: xarray.core.dataarray.DataArray
        Multivariate time series as an xarray
    aweights: array_like, optional
        Doc taken from numpy cov follows:
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    Returns
    -------
    S: xarray.DataArray
        The covariance matrix of the data in xarray form.
    """

    channels = list(X.coords["variable"].values)

    S = xr.DataArray(
        np.cov(X.values, rowvar=False, aweights=aweights),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    return S


def initialize_xrda_1d(
    channels: list,
    dtype: Optional[type] = None,
    value: Optional[Union[complex, float, bool]] = 0,
) -> xr.DataArray:
    """
    Returns a 1D xr.DataArray with variable "channel", having values channels named by the input list.

    Parameters
    ----------
    channels: list
        The channels in the multivariate array
    dtype: type, optional
        The datatype to initialize the array.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool], optional
        The default value to assign the array

    Returns
    -------
    xrda: xarray.core.dataarray.DataArray
        An xarray container for the channels, initialized to zeros.
    """
    k = len(channels)
    xrda = xr.DataArray(
        np.zeros(k, dtype=dtype),
        dims=[
            "variable",
        ],
        coords={
            "variable": channels,
        },
    )
    if value != 0:
        data = value * np.ones(k, dtype=dtype)
        xrda.data = data
    return xrda 


def initialize_xrda_2d(
    channels: list,
    dtype: Optional[type] = complex,
    value: Optional[Union[complex, float, bool]] = 0,
    dims: Optional[list] = None,
) -> xr.DataArray:
    """
    Returns a 2D xr.DataArray with dimensions channel_1 and channel_2.

    Parameters
    ----------
    channels: list
        The channels in the multivariate array
    dtype: type, optional
        The datatype to initialize the array.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool], optional
        The default value to assign the array
    dims: list, optional
        List of two channel lists for the two dimensions. If None, uses the same channels for both dimensions.

    Returns
    -------
    xrda: xarray.core.dataarray.DataArray
        An xarray container for the channel variances etc., initialized to zeros.
    """
    if dims is None:
        dims = [channels, channels]

    K1 = len(dims[0])
    K2 = len(dims[1])
    xrda = xr.DataArray(
        np.zeros((K1, K2), dtype=dtype),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": dims[0], "channel_2": dims[1]},
    )
    if value != 0:
        data = value * np.ones(xrda.shape, dtype=dtype)
        xrda.data = data

    return xrda 
