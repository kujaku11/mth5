mth5.timeseries.xarray_helpers
==============================

.. py:module:: mth5.timeseries.xarray_helpers

.. autoapi-nested-parse::

   Module containing helper functions for working with xarray objects.



Functions
---------

.. autoapisummary::

   mth5.timeseries.xarray_helpers.covariance_xr
   mth5.timeseries.xarray_helpers.initialize_xrda_1d
   mth5.timeseries.xarray_helpers.initialize_xrds_2d
   mth5.timeseries.xarray_helpers.initialize_xrda_2d
   mth5.timeseries.xarray_helpers.initialize_xrda_2d_cov


Module Contents
---------------

.. py:function:: covariance_xr(X: xarray.DataArray, aweights: Optional[Union[numpy.ndarray, None]] = None, bias: Optional[bool] = False, rowvar: Optional[bool] = False) -> xarray.DataArray

   Compute the covariance matrix with numpy.cov.

   :param X: Multivariate time series as an xarray
   :type X: xarray.core.dataarray.DataArray
   :param aweights: Passthrough param for np.cov.
                    1-D array of observation vector weights. These relative weights are
                    typically large for observations considered "important" and smaller for
                    observations considered less "important". If ``ddof=0`` the array of
                    weights can be used to assign probabilities to observation vectors.
   :type aweights: array_like, optional
   :param bias: Passthrough param for np.cov.
                Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                number of observations given (unbiased estimate). If `bias` is True,
                then normalization is by ``N``. These values can be overridden by using
                the keyword ``ddof`` in numpy versions >= 1.5.
   :type bias: bool
   :param rowvar: Passthrough param for np.cov.
                  If `rowvar` is True (default), then each row represents a
                  variable, with observations in the columns. Otherwise, the relationship
                  is transposed: each column represents a variable, while the rows
                  contain observations.
   :type rowvar: bool

   :returns: * **S** (*xarray.DataArray*) -- The covariance matrix of the data in xarray form.
             * *Development Notes* -- In case of ValueError: conflicting sizes for dimension 'channel_1', this likely means the bool for rowvar
               should be flipped.


.. py:function:: initialize_xrda_1d(channels: list, dtype: Optional[type] = None, value: Optional[Union[complex, float, bool]] = 0.0) -> xarray.DataArray

   Returns a 1D xr.DataArray with variable "channel", having values channels named by the input list.

   :param channels: The channels in the multivariate array
   :type channels: list
   :param dtype: The datatype to initialize the array.
                 Common cases are complex, float, and bool
   :type dtype: type, optional
   :param value: The default value to assign the array
   :type value: Union[complex, float, bool], optional

   :returns: **xrda** -- An xarray container for the channels, initialized to zeros.
   :rtype: xarray.core.dataarray.DataArray


.. py:function:: initialize_xrds_2d(variables: list, coords: dict, dtype: Optional[type] = complex, value: Optional[Union[complex, float, bool]] = 0) -> xarray.Dataset

   Returns a 2D xr.Dataset with the given variables and coordinates.

   :param variables: List of variable names to create in the dataset
   :type variables: list
   :param coords: Dictionary of coordinates for the dataset dimensions
   :type coords: dict
   :param dtype: The datatype to initialize the arrays.
                 Common cases are complex, float, and bool
   :type dtype: type, optional
   :param value: The default value to assign the arrays
   :type value: Union[complex, float, bool], optional

   :returns: **xrds** -- A 2D xarray Dataset with dimensions from coords
   :rtype: xr.Dataset


.. py:function:: initialize_xrda_2d(variables: list, coords: dict, dtype: type = complex, value: Union[int, float] = 0.0) -> xarray.Dataset

   Initialize a 3D xarray DataArray with dimensions from coords plus 'variable'.

   :param variables: List of variable names for the additional dimension.
   :type variables: list
   :param coords: Dictionary of coordinates for the dataset dimensions.
   :type coords: dict
   :param dtype: Data type for the array, by default complex.
   :type dtype: type, optional
   :param value: Value to initialize the array with, by default 0.
   :type value: int or float, optional

   :returns: A 3D DataArray with dimensions from coords plus 'variable'.
   :rtype: xr.DataArray


.. py:function:: initialize_xrda_2d_cov(channels, dtype=complex, value: Optional[Union[complex, float, bool]] = 0)

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


