mth5.timeseries.scipy_filters
=============================

.. py:module:: mth5.timeseries.scipy_filters

.. autoapi-nested-parse::

   Scipy filter wrappers for xarray.

   This module provides xarray-compatible wrappers for scipy.signal filtering
   functions, enabling efficient filtering operations on labeled N-dimensional
   arrays with automatic dimension handling.

   .. rubric:: Notes

   - Adapted from xr-scipy: https://github.com/fujiisoup/xr-scipy
   - Filters can be applied along any dimension with automatic sampling rate detection.
   - Supports both IIR and FIR filter types with forward/backward filtering.

   .. rubric:: Examples

   Apply a bandpass filter to an xarray DataArray::

       >>> import xarray as xr
       >>> import numpy as np
       >>> data = xr.DataArray(np.random.randn(1000), dims=['time'])
       >>> data['time'] = np.arange(1000) / 100.0  # 100 Hz
       >>> filtered = data.sps_filters.bandpass(10, 40)  # 10-40 Hz



Attributes
----------

.. autoapisummary::

   mth5.timeseries.scipy_filters.sosfiltfilt


Exceptions
----------

.. autoapisummary::

   mth5.timeseries.scipy_filters.UnevenSamplingWarning
   mth5.timeseries.scipy_filters.FilteringNaNWarning
   mth5.timeseries.scipy_filters.DecimationWarning


Classes
-------

.. autoapisummary::

   mth5.timeseries.scipy_filters.FilterAccessor


Functions
---------

.. autoapisummary::

   mth5.timeseries.scipy_filters.get_maybe_only_dim
   mth5.timeseries.scipy_filters.get_maybe_last_dim_axis
   mth5.timeseries.scipy_filters.get_sampling_step
   mth5.timeseries.scipy_filters.frequency_filter
   mth5.timeseries.scipy_filters.lowpass
   mth5.timeseries.scipy_filters.highpass
   mth5.timeseries.scipy_filters.bandpass
   mth5.timeseries.scipy_filters.bandstop
   mth5.timeseries.scipy_filters.decimate
   mth5.timeseries.scipy_filters.resample_poly
   mth5.timeseries.scipy_filters.savgol_filter
   mth5.timeseries.scipy_filters.detrend


Module Contents
---------------

.. py:data:: sosfiltfilt
   :value: None


.. py:exception:: UnevenSamplingWarning

   Bases: :py:obj:`Warning`


   Base class for warning categories.


.. py:exception:: FilteringNaNWarning

   Bases: :py:obj:`Warning`


   Base class for warning categories.


.. py:exception:: DecimationWarning

   Bases: :py:obj:`Warning`


   Base class for warning categories.


.. py:function:: get_maybe_only_dim(darray: xarray.DataArray | xarray.Dataset, dim: str | None) -> str

   Determine the dimension along which to operate.

   If `dim` is None and the array is 1-D, returns the single dimension.
   Otherwise, returns the provided `dim`.

   :param darray: An xarray DataArray or Dataset.
   :type darray: xarray.DataArray | xarray.Dataset
   :param dim: Dimension name, or None to auto-detect for 1-D arrays.
   :type dim: str | None

   :returns: The dimension name.
   :rtype: str

   :raises ValueError: If `dim` is None and the array is not 1-D.


.. py:function:: get_maybe_last_dim_axis(darray: xarray.DataArray | xarray.Dataset, dim: str | None = None) -> tuple[str, int]

   Get dimension name and axis index.

   :param darray: Input array.
   :type darray: xarray.DataArray | xarray.Dataset
   :param dim: Dimension name. If None, uses the last dimension.
   :type dim: str | None, default None

   :returns: Dimension name and corresponding axis index.
   :rtype: tuple[str, int]


.. py:function:: get_sampling_step(darray: xarray.DataArray | xarray.Dataset, dim: str | None = None, rtol: float = 0.001) -> float

   Compute the sampling step along a dimension.

   Automatically detects time unit (ns, us, ms, s) and scales accordingly.
   Issues a warning if sampling is not uniform (average vs first step mismatch).

   :param darray: Input array with coordinates.
   :type darray: xarray.DataArray | xarray.Dataset
   :param dim: Dimension name. Auto-detected for 1-D arrays.
   :type dim: str | None, default None
   :param rtol: Relative tolerance for detecting uneven sampling.
   :type rtol: float, default 1e-3

   :returns: Sampling step in appropriate units (seconds for time coordinates).
   :rtype: float

   :raises ValueError: If the coordinate has fewer than 2 samples.

   .. warning::

      UnevenSamplingWarning
          If average sampling step differs from first step by more than rtol.

   .. rubric:: Examples

   Get sampling step from a time series::

       >>> dt = get_sampling_step(data, dim='time')
       >>> sample_rate = 1.0 / dt


.. py:function:: frequency_filter(darray: xarray.DataArray | xarray.Dataset, f_crit: float | list[float] | tuple[float, float], order: int | None = None, irtype: str = 'iir', filtfilt: bool = True, apply_kwargs: dict[str, Any] | None = None, in_nyq: bool = False, dim: str | None = None, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

   Apply a frequency filter to an xarray.

   Supports IIR (infinite impulse response) and FIR (finite impulse response)
   filters with optional forward-backward filtering for zero-phase response.

   :param darray: Data to be filtered.
   :type darray: xarray.DataArray | xarray.Dataset
   :param f_crit: Critical frequency or frequencies (in Hz).
                  Scalar for lowpass/highpass, 2-element for bandpass/bandstop.
   :type f_crit: float | list[float] | tuple[float, float]
   :param order: Filter order. If None, uses default (8 for IIR, 29 for FIR).
   :type order: int | None, default None
   :param irtype: Impulse response type: 'iir' or 'fir'.
   :type irtype: {'iir', 'fir'}, default 'iir'
   :param filtfilt: Apply filter forwards and backwards for zero-phase response.
   :type filtfilt: bool, default True
   :param apply_kwargs: Additional kwargs passed to the filter function.
   :type apply_kwargs: dict | None, default None
   :param in_nyq: If True, `f_crit` values are already normalized by Nyquist frequency.
   :type in_nyq: bool, default False
   :param dim: Dimension along which to filter. Auto-detected for 1-D arrays.
   :type dim: str | None, default None
   :param \*\*kwargs: Passed to filter design function (scipy.signal.iirfilter or firwin).

   :returns: Filtered data with same structure as input.
   :rtype: xarray.DataArray | xarray.Dataset

   :raises ValueError: If `irtype` is not 'iir' or 'fir', or if `dim` is ambiguous.

   .. warning::

      FilteringNaNWarning
          If input contains NaN values.

   .. rubric:: Examples

   Apply a 4th-order IIR lowpass at 10 Hz::

       >>> filtered = frequency_filter(data, 10, order=4, btype='low')

   FIR bandpass filter from 5 to 15 Hz::

       >>> filtered = frequency_filter(data, [5, 15], irtype='fir', btype='band')


.. py:function:: lowpass(darray: xarray.DataArray | xarray.Dataset, f_cutoff: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

   Apply a lowpass filter.

   :param darray: Data to filter.
   :type darray: xarray.DataArray | xarray.Dataset
   :param f_cutoff: Cutoff frequency in Hz.
   :type f_cutoff: float
   :param \*args: Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
   :param \*\*kwargs: Passed to filter design (see `frequency_filter`).

   :returns: Lowpass-filtered data.
   :rtype: xarray.DataArray | xarray.Dataset

   .. rubric:: Examples

   Remove components above 50 Hz::

       >>> filtered = lowpass(data, 50)


.. py:function:: highpass(darray: xarray.DataArray | xarray.Dataset, f_cutoff: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

   Apply a highpass filter.

   :param darray: Data to filter.
   :type darray: xarray.DataArray | xarray.Dataset
   :param f_cutoff: Cutoff frequency in Hz.
   :type f_cutoff: float
   :param \*args: Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
   :param \*\*kwargs: Passed to filter design (see `frequency_filter`).

   :returns: Highpass-filtered data.
   :rtype: xarray.DataArray | xarray.Dataset

   .. rubric:: Examples

   Remove components below 1 Hz::

       >>> filtered = highpass(data, 1.0)


.. py:function:: bandpass(darray: xarray.DataArray | xarray.Dataset, f_low: float, f_high: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

   Apply a bandpass filter.

   :param darray: Data to filter.
   :type darray: xarray.DataArray | xarray.Dataset
   :param f_low: Lower cutoff frequency in Hz.
   :type f_low: float
   :param f_high: Upper cutoff frequency in Hz.
   :type f_high: float
   :param \*args: Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
   :param \*\*kwargs: Passed to filter design (see `frequency_filter`).

   :returns: Bandpass-filtered data.
   :rtype: xarray.DataArray | xarray.Dataset

   .. rubric:: Examples

   Keep components between 10 and 50 Hz::

       >>> filtered = bandpass(data, 10, 50)


.. py:function:: bandstop(darray: xarray.DataArray | xarray.Dataset, f_low: float, f_high: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

   Apply a bandstop (notch) filter.

   :param darray: Data to filter.
   :type darray: xarray.DataArray | xarray.Dataset
   :param f_low: Lower cutoff frequency in Hz.
   :type f_low: float
   :param f_high: Upper cutoff frequency in Hz.
   :type f_high: float
   :param \*args: Passed to `frequency_filter`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
   :param \*\*kwargs: Passed to filter design (see `frequency_filter`).

   :returns: Bandstop-filtered data (removes frequencies between f_low and f_high).
   :rtype: xarray.DataArray | xarray.Dataset

   .. rubric:: Examples

   Remove 50-60 Hz powerline noise::

       >>> filtered = bandstop(data, 50, 60)


.. py:function:: decimate(darray: xarray.Dataset | xarray.DataArray, target_sample_rate: float, n_order: int = 8, dim: str | None = None) -> xarray.DataArray | xarray.Dataset

   Decimate data using Chebyshev filter and downsampling.

   Applies an 8th-order Chebyshev type I filter with zero-phase filtering
   (sosfiltfilt) before downsampling.

   :param darray: Data to decimate.
   :type darray: xarray.DataArray | xarray.Dataset
   :param target_sample_rate: Target sample rate in samples per second (or per space unit).
   :type target_sample_rate: float
   :param n_order: Order of the Chebyshev type I filter.
   :type n_order: int, default 8
   :param dim: Dimension to decimate along. Auto-detected for 1-D arrays.
   :type dim: str | None, default None

   :returns: Decimated data with adjusted coordinates.
   :rtype: xarray.DataArray | xarray.Dataset

   :raises ValueError: If `dim` is None and array is not 1-D.

   .. warning::

      UserWarning
          If decimation factor > 13, suggest calling decimate multiple times.

   .. rubric:: Notes

   If sample_rate / target_sample_rate > 13, call decimate multiple times
   to avoid aliasing artifacts.

   .. rubric:: Examples

   Decimate from 100 Hz to 10 Hz::

       >>> decimated = decimate(data, target_sample_rate=10.0)


.. py:function:: resample_poly(darray: xarray.DataArray | xarray.Dataset, new_sample_rate: float, dim: str | None = None, pad_type: str = 'mean') -> xarray.DataArray | xarray.Dataset

   Resample using polyphase filtering.

   Computes rational resampling ratio (up/down) and applies
   scipy.signal.resample_poly. Automatically handles coordinate updates.

   :param darray: Data to resample.
   :type darray: xarray.DataArray | xarray.Dataset
   :param new_sample_rate: Target sample rate.
   :type new_sample_rate: float
   :param dim: Dimension to resample along. Auto-detected for 1-D arrays.
   :type dim: str | None, default None
   :param pad_type: Padding type passed to scipy.signal.resample_poly.
                    Options: 'constant', 'line', 'mean', 'median', etc.
   :type pad_type: str, default 'mean'

   :returns: Resampled data with updated coordinates.
   :rtype: xarray.DataArray | xarray.Dataset

   :raises ValueError: If `dim` is None and array is not 1-D.

   .. warning::

      UserWarning
          If new sample rate is not an integer multiple of original rate.

   .. rubric:: Notes

   In newer scipy versions, data is cast to float and returns float dtype.

   .. rubric:: Examples

   Resample to 50 Hz::

       >>> resampled = resample_poly(data, new_sample_rate=50.0)


.. py:function:: savgol_filter(darray: xarray.DataArray | xarray.Dataset, window_length: int, polyorder: int, deriv: int = 0, delta: float | None = None, dim: str | None = None, mode: str = 'interp', cval: float = 0.0) -> xarray.DataArray | xarray.Dataset

   Apply a Savitzky-Golay filter.

   Smooths data using least-squares polynomial fit over a sliding window.
   Can also compute derivatives.

   :param darray: Data to filter. Converted to float64 if not already float.
   :type darray: xarray.DataArray | xarray.Dataset
   :param window_length: Length of the filter window (number of coefficients).
                         Must be positive odd integer.
   :type window_length: int
   :param polyorder: Order of polynomial for fitting. Must be < window_length.
   :type polyorder: int
   :param deriv: Order of derivative to compute (0 = no differentiation).
   :type deriv: int, default 0
   :param delta: Sample spacing for derivative computation. Only used if deriv > 0.
   :type delta: float | None, default None
   :param dim: Dimension along which to filter. Auto-detected for 1-D arrays.
   :type dim: str | None, default None
   :param mode: Extension mode: 'mirror', 'constant', 'nearest', 'wrap', or 'interp'.
   :type mode: str, default 'interp'
   :param cval: Fill value when mode='constant'.
   :type cval: float, default 0.0

   :returns: Filtered data.
   :rtype: xarray.DataArray | xarray.Dataset

   :raises ValueError: If window_length is not positive odd, polyorder >= window_length,
       or `dim` is None for multi-dimensional arrays.

   .. rubric:: Examples

   Smooth with 11-point window, 2nd-order polynomial::

       >>> smoothed = savgol_filter(data, window_length=11, polyorder=2)

   Compute first derivative::

       >>> deriv1 = savgol_filter(data, 11, 2, deriv=1, delta=0.01)


.. py:function:: detrend(darray: xarray.DataArray | xarray.Dataset, dim: str | None = None, trend_type: str = 'linear') -> xarray.DataArray | xarray.Dataset

   Remove linear or constant trend from data.

   :param darray: Data to detrend.
   :type darray: xarray.DataArray | xarray.Dataset
   :param dim: Dimension along which to detrend. Auto-detected for 1-D arrays.
   :type dim: str | None, default None
   :param trend_type: Type of detrending: 'linear' removes linear trend via least-squares,
                      'constant' removes mean.
   :type trend_type: {'linear', 'constant'}, default 'linear'

   :returns: Detrended data.
   :rtype: xarray.DataArray | xarray.Dataset

   :raises ValueError: If `dim` is None and array is not 1-D.

   .. rubric:: Examples

   Remove linear trend::

       >>> detrended = detrend(data, trend_type='linear')

   Remove mean (DC component)::

       >>> demeaned = detrend(data, trend_type='constant')


.. py:class:: FilterAccessor(darray: xarray.DataArray | xarray.Dataset)

   Accessor exposing common frequency and filtering methods.

   Registered as xarray accessor under `.sps_filters` for both DataArray
   and Dataset objects.

   .. attribute:: darray

      The wrapped xarray object.

      :type: xarray.DataArray | xarray.Dataset

   .. rubric:: Examples

   Apply filters via accessor::

       >>> data.sps_filters.low(10)  # lowpass at 10 Hz
       >>> data.sps_filters.bandpass(5, 15)  # bandpass 5-15 Hz


   .. py:attribute:: darray
      :type:  xarray.DataArray | xarray.Dataset


   .. py:property:: dt
      :type: float


      Sampling step of last axis.


   .. py:property:: fs
      :type: float


      Sampling frequency in inverse units of self.dt.


   .. py:property:: dx
      :type: numpy.ndarray


      Sampling steps for all axes as array.


   .. py:method:: low(f_cutoff: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

      Apply lowpass filter.

      :param f_cutoff: Cutoff frequency in Hz.
      :type f_cutoff: float
      :param \*args: Passed to `lowpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
      :param \*\*kwargs: Passed to filter design.

      :returns: Lowpass-filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: high(f_cutoff: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

      Apply highpass filter.

      :param f_cutoff: Cutoff frequency in Hz.
      :type f_cutoff: float
      :param \*args: Passed to `highpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
      :param \*\*kwargs: Passed to filter design.

      :returns: Highpass-filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: bandpass(f_low: float, f_high: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

      Apply bandpass filter.

      :param f_low: Lower cutoff frequency in Hz.
      :type f_low: float
      :param f_high: Upper cutoff frequency in Hz.
      :type f_high: float
      :param \*args: Passed to `bandpass`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
      :param \*\*kwargs: Passed to filter design.

      :returns: Bandpass-filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: bandstop(f_low: float, f_high: float, *args: Any, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

      Apply bandstop filter.

      :param f_low: Lower cutoff frequency in Hz.
      :type f_low: float
      :param f_high: Upper cutoff frequency in Hz.
      :type f_high: float
      :param \*args: Passed to `bandstop`: (order, irtype, filtfilt, apply_kwargs, in_nyq, dim).
      :param \*\*kwargs: Passed to filter design.

      :returns: Bandstop-filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: freq(f_crit: float | list[float] | tuple[float, float], order: int | None = None, irtype: str = 'iir', filtfilt: bool = True, apply_kwargs: dict[str, Any] | None = None, in_nyq: bool = False, dim: str | None = None, **kwargs: Any) -> xarray.DataArray | xarray.Dataset

      Apply general frequency filter.

      :param f_crit: Critical frequency or frequencies in Hz.
      :type f_crit: float | list[float] | tuple[float, float]
      :param order: Filter order.
      :type order: int | None, default None
      :param irtype: Impulse response type.
      :type irtype: {'iir', 'fir'}, default 'iir'
      :param filtfilt: Apply forward-backward filtering.
      :type filtfilt: bool, default True
      :param apply_kwargs: Additional filter function kwargs.
      :type apply_kwargs: dict | None, default None
      :param in_nyq: If True, f_crit is normalized by Nyquist frequency.
      :type in_nyq: bool, default False
      :param dim: Dimension along which to filter.
      :type dim: str | None, default None
      :param \*\*kwargs: Passed to filter design.

      :returns: Filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: savgol(window_length: int, polyorder: int, deriv: int = 0, delta: float | None = None, dim: str | None = None, mode: str = 'interp', cval: float = 0.0) -> xarray.DataArray | xarray.Dataset

      Apply Savitzky-Golay filter.

      :param window_length: Filter window length (positive odd integer).
      :type window_length: int
      :param polyorder: Polynomial order (< window_length).
      :type polyorder: int
      :param deriv: Derivative order.
      :type deriv: int, default 0
      :param delta: Sample spacing.
      :type delta: float | None, default None
      :param dim: Dimension to filter along.
      :type dim: str | None, default None
      :param mode: Extension mode.
      :type mode: str, default 'interp'
      :param cval: Constant fill value.
      :type cval: float, default 0.0

      :returns: Filtered data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: decimate(target_sample_rate: float, n_order: int = 8, dim: str | None = None) -> xarray.DataArray | xarray.Dataset

      Decimate signal.

      :param target_sample_rate: Target sample rate.
      :type target_sample_rate: float
      :param n_order: Chebyshev filter order.
      :type n_order: int, default 8
      :param dim: Dimension to decimate along.
      :type dim: str | None, default None

      :returns: Decimated data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: detrend(trend_type: str = 'linear', dim: str | None = None) -> xarray.DataArray | xarray.Dataset

      Remove trend from data.

      :param trend_type: Type of trend to remove.
      :type trend_type: {'linear', 'constant'}, default 'linear'
      :param dim: Dimension to detrend along.
      :type dim: str | None, default None

      :returns: Detrended data.
      :rtype: xarray.DataArray | xarray.Dataset



   .. py:method:: resample_poly(target_sample_rate: float, pad_type: str = 'mean', dim: str | None = None) -> xarray.DataArray | xarray.Dataset

      Resample using polyphase filtering.

      :param target_sample_rate: Target sample rate.
      :type target_sample_rate: float
      :param pad_type: Padding type for resampling.
      :type pad_type: str, default 'mean'
      :param dim: Dimension to resample along.
      :type dim: str | None, default None

      :returns: Resampled data.
      :rtype: xarray.DataArray | xarray.Dataset



