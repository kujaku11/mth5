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
    coords: dict,
    dtype: Optional[type] = complex,
    value: Optional[Union[complex, float, bool]] = 0,
) -> xr.DataArray:
    """
    Returns a 2D xr.DataArray with dimensions channel_1 and channel_2.

    Parameters
    ----------
    channels: list
        The channels in the multivariate array
    coords: dict
        the coordinates of the data array to return.
    dtype: type, optional
        The datatype to initialize the array.
        Common cases are complex, float, and bool
    value: Union[complex, float, bool], optional
        The default value to assign the array

    Returns
    -------
    xrda: xarray.core.dataarray.DataArray
        An xarray container for the channel variances etc., initialized to zeros.
    """
    dims = list(coords.keys())
    K1 = len(coords[dims[0]])
    K2 = len(coords[dims[1]])
    xrda = xr.DataArray(
        np.zeros((K1, K2), dtype=dtype),
        dims=dims,
        coords=coords
    )
    if value != 0:
        data = value * np.ones(xrda.shape, dtype=dtype)
        xrda.data = data

    return xrda


def initialize_xrda_2d_cov(
    channels, dtype=complex, value: Optional[Union[complex, float, bool]] = 0
):
    """
     TODO: consider changing nomenclature from dims=["channel_1", "channel_2"],
     to dims=["variable_1", "variable_2"], to be consistent with initialize_xrda_1d

    Parameters
     ----------
     channels: list
         The channels in the multivariate array. The covariance matrix will be square
         with dimensions len(channels) x len(channels).
     dtype: type
         The datatype to initialize the array.
         Common cases are complex, float, and bool
     value: Union[complex, float, bool]
         The default value to assign the array

    Returns
     -------
     xrda: xarray.core.dataarray.DataArray
         An xarray container for the channel covariances, initialized to zeros.
         The matrix is square with dimensions len(channels) x len(channels).
    """
    K = len(channels)
    xrda = xr.DataArray(
        np.zeros((K, K), dtype=dtype),
        dims=["channel_1", "channel_2"],
        coords={"channel_1": channels, "channel_2": channels},
    )
    if value != 0:
        data = value * np.ones(xrda.shape, dtype=dtype)
        xrda.data = data

    return xrda
