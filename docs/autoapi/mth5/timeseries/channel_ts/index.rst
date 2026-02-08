mth5.timeseries.channel_ts
==========================

.. py:module:: mth5.timeseries.channel_ts

.. autoapi-nested-parse::

   Channel time series module for MT data.

   This module provides the `ChannelTS` class for handling magnetotelluric (MT)
   time series data with comprehensive metadata management, calibration,
   and signal processing capabilities.

   .. rubric:: Notes

   - Time series are stored in `xarray.DataArray` for efficient operations.
   - Metadata follows the mt_metadata standard with Survey/Station/Run/Channel hierarchy.
   - Supports instrument response removal, resampling, merging, and Obspy integration.



Attributes
----------

.. autoapisummary::

   mth5.timeseries.channel_ts.meta_classes


Classes
-------

.. autoapisummary::

   mth5.timeseries.channel_ts.ChannelTS


Module Contents
---------------

.. py:data:: meta_classes

.. py:class:: ChannelTS(channel_type: str = 'auxiliary', data: numpy.ndarray | pandas.DataFrame | pandas.Series | xarray.DataArray | list | tuple | None = None, channel_metadata: mt_metadata.timeseries.Electric | mt_metadata.timeseries.Magnetic | mt_metadata.timeseries.Auxiliary | dict | None = None, station_metadata: mt_metadata.timeseries.Station | dict | None = None, run_metadata: mt_metadata.timeseries.Run | dict | None = None, survey_metadata: mt_metadata.timeseries.Survey | dict | None = None, **kwargs: Any)

   Time series container for a single MT channel with full metadata.

   Stores equally-spaced time series data in an `xarray.DataArray` with
   a time coordinate index. Integrates comprehensive metadata from
   Survey/Station/Run/Channel hierarchy and supports calibration,
   resampling, merging, and format conversions.

   :param channel_type: Type of the channel.
   :type channel_type: {'electric', 'magnetic', 'auxiliary'}, default 'auxiliary'
   :param data: Time series data (numpy array, pandas DataFrame/Series, xarray.DataArray).
   :type data: array-like, optional
   :param channel_metadata: Channel-specific metadata.
   :type channel_metadata: mt_metadata.timeseries.Electric | Magnetic | Auxiliary | dict, optional
   :param station_metadata: Station metadata.
   :type station_metadata: mt_metadata.timeseries.Station | dict, optional
   :param run_metadata: Run metadata.
   :type run_metadata: mt_metadata.timeseries.Run | dict, optional
   :param survey_metadata: Survey metadata.
   :type survey_metadata: mt_metadata.timeseries.Survey | dict, optional
   :param \*\*kwargs: Additional attributes to set on the object.

   .. attribute:: ts

      The time series data array.

      :type: numpy.ndarray

   .. attribute:: sample_rate

      Sample rate in samples per second.

      :type: float

   .. attribute:: start

      Start time (UTC).

      :type: MTime

   .. attribute:: end

      End time (UTC), derived from start + duration.

      :type: MTime

   .. attribute:: n_samples

      Number of samples.

      :type: int

   .. attribute:: component

      Component name (e.g., 'ex', 'hy', 'temperature').

      :type: str

   .. attribute:: channel_response

      Full instrument response filter chain.

      :type: ChannelResponse

   .. rubric:: Notes

   - End time is a derived property and cannot be set directly.
   - Leverages xarray for efficient interpolation, resampling, and groupby operations.
   - Metadata follows mt_metadata standards with automatic time period updates.

   .. rubric:: Examples

   Create an auxiliary channel with synthetic data::

       >>> from mth5.timeseries import ChannelTS
       >>> import numpy as np
       >>> ts_obj = ChannelTS('auxiliary')
       >>> ts_obj.sample_rate = 8
       >>> ts_obj.start = '2020-01-01T12:00:00+00:00'
       >>> ts_obj.ts = np.random.randn(4096)
       >>> ts_obj.station_metadata.id = 'MT001'
       >>> ts_obj.run_metadata.id = 'MT001a'
       >>> ts_obj.component = 'temperature'
       >>> print(ts_obj)

   Calibrate and remove instrument response::

       >>> calibrated = ts_obj.remove_instrument_response()
       >>> calibrated.channel_metadata.units


   .. py:attribute:: logger


   .. py:attribute:: data_array


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Survey metadata.

      :returns: Survey metadata with updated keys.
      :rtype: mt_metadata.timeseries.Survey


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Station metadata.

      :returns: Station metadata from the first station in the survey.
      :rtype: mt_metadata.timeseries.Station


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Run metadata.

      :returns: Run metadata from the first run in the station.
      :rtype: mt_metadata.timeseries.Run


   .. py:property:: channel_metadata
      :type: mt_metadata.timeseries.Electric | mt_metadata.timeseries.Magnetic | mt_metadata.timeseries.Auxiliary


      Channel metadata.

      :returns: Channel metadata from the first channel in the run.
      :rtype: mt_metadata.timeseries.Electric | Magnetic | Auxiliary


   .. py:method:: get_sample_rate_supplied_at_init(channel_metadata: mt_metadata.timeseries.Electric | mt_metadata.timeseries.Magnetic | mt_metadata.timeseries.Auxiliary | dict | None) -> float | None

      Extract sample_rate from channel_metadata if available.

      :param channel_metadata: Metadata that may contain a sample_rate field.
      :type channel_metadata: mt_metadata.timeseries.Electric | Magnetic | Auxiliary | dict | None

      :returns: Sample rate if found, otherwise None.
      :rtype: float | None

      .. rubric:: Notes

      Supports nested dict structures like ``{"electric": {"sample_rate": 8.0}}``.



   .. py:method:: copy(data: bool = True) -> ChannelTS

      Create a copy of the ChannelTS object.

      :param data: Include data in the copy (True) or only metadata (False).
      :type data: bool, default True

      :returns: Copy of the channel.
      :rtype: ChannelTS

      .. rubric:: Examples

      Copy metadata structure without data::

          >>> ch_copy = ts_obj.copy(data=False)



   .. py:property:: ts
      :type: numpy.ndarray


      Time series data as a numpy array.

      :returns: The time series data.
      :rtype: numpy.ndarray


   .. py:property:: time_index
      :type: numpy.ndarray


      Time index as a numpy array.

      :returns: Array of datetime64[ns] timestamps.
      :rtype: numpy.ndarray


   .. py:property:: channel_type
      :type: str


      Channel type.

      :returns: Channel type: 'Electric', 'Magnetic', or 'Auxiliary'.
      :rtype: str


   .. py:property:: component

      component


   .. py:property:: n_samples

      number of samples


   .. py:method:: has_data()

      check to see if there is an index in the time series



   .. py:method:: is_high_frequency(threshold_dt=0.0001)

      Quasi hard-coded condition to check if data are logged at more than 10kHz
      can be parameterized in future



   .. py:method:: compute_sample_rate()

      Two cases, high_frequency (HF) data and not HF data.

      # Original comment about the HF case:
      Taking the median(diff(timestamps)) is more accurate for high sample rates, the way pandas.date_range
      rounds nanoseconds is not consistent between samples, therefore taking the median provides better results
      if the time series is long this can be inefficient so test first




   .. py:property:: sample_rate

      sample rate in samples/second


   .. py:property:: sample_interval

      Sample interval = 1 / sample_rate

      :return: sample interval as time distance between time samples
      :rtype: float


   .. py:property:: start

      MTime object


   .. py:property:: end

      MTime object


   .. py:property:: channel_response

      Full channel response filter

      :return: full channel response filter
      :rtype: :class:`mt_metadata.timeseries.filters.ChannelResponse`


   .. py:method:: get_calibration_operation()


   .. py:method:: get_calibrated_units()

      Follows the FDSN standard which has the filter stages starting with physical units to digital counts.

      The channel_response is expected to have a list of filter "stages" of which the first stage
      has input units corresponding to the the physical quantity that the instrument measures, and the last is
      normally counts.

      channel_response can be viewed as the chaining together of all of these filters.

      Thus it is normal for channel_response.units_out will be in the same units as the archived raw
      time series, and for the units after the response is corrected for will be the units_in of

      The units of the channel metadata are compared to the input and output units of the channel_response.


      :return: tuple, calibration_operation, either "mulitply" or divide", and a string for calibrated units
      :rtype: tuple (of two strings_



   .. py:method:: remove_instrument_response(include_decimation=False, include_delay=False, **kwargs)

      Remove instrument response from the given channel response filter

      The order of operations is important (if applied):

          1) detrend
          2) zero mean
          3) zero pad
          4) time window
          5) frequency window
          6) remove response
          7) undo time window
          8) bandpass

      :param include_decimation: Include decimation in response,
       defaults to True
      :type include_decimation: bool, optional
      :param include_delay: include delay in complex response,
       defaults to False
      :type include_delay: bool, optional

      **kwargs**

      :param plot: to plot the calibration process [ False | True ]
      :type plot: boolean, default True
      :param detrend: Remove linar trend of the time series
      :type detrend: boolean, default True
      :param zero_mean: Remove the mean of the time series
      :type zero_mean: boolean, default True
      :param zero_pad: pad the time series to the next power of 2 for efficiency
      :type zero_pad: boolean, default True
      :param t_window: Time domain windown name see `scipy.signal.windows` for options
      :type t_window: string, default None
      :param t_window_params: Time domain window parameters, parameters can be
      found in `scipy.signal.windows`
      :type t_window_params: dictionary
      :param f_window: Frequency domain windown name see `scipy.signal.windows` for options
      :type f_window: string, defualt None
      :param f_window_params: Frequency window parameters, parameters can be
      found in `scipy.signal.windows`
      :type f_window_params: dictionary
      :param bandpass: bandpass freequency and order {"low":, "high":, "order":,}
      :type bandpass: dictionary




   .. py:method:: get_slice(start, end=None, n_samples=None)

      Get a slice from the time series given a start and end time.

      Looks for >= start & <= end

      Uses loc to be exact with milliseconds

      :param start: start time of the slice
      :type start: string, MTime
      :param end: end time of the slice
      :type end: string, MTime
      :param n_samples: number of sample to get after start time
      :type n_samples: integer
      :return: slice of the channel requested
      :rtype: ChannelTS




   .. py:method:: decimate(new_sample_rate, inplace=False, max_decimation=8)

      decimate the data by using scipy.signal.decimate

      :param dec_factor: decimation factor
      :type dec_factor: int

      * refills ts.data with decimated data and replaces sample_rate




   .. py:method:: resample_poly(new_sample_rate, pad_type='mean', inplace=False)

      Use scipy.signal.resample_poly to resample data while using an FIR
      filter to remove aliasing.

      :param new_sample_rate: DESCRIPTION
      :type new_sample_rate: TYPE
      :param pad_type: DESCRIPTION, defaults to "mean"
      :type pad_type: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: merge(other, gap_method='slinear', new_sample_rate=None, resample_method='poly')

      merg two channels or list of channels together in the following steps

      1. xr.combine_by_coords([original, other])
      2. compute monotonic time index
      3. reindex(new_time_index, method=gap_method)

      If you want a different method or more control use merge

      :param other: Another channel
      :type other: :class:`mth5.timeseries.ChannelTS`
      :raises TypeError: If input is not a ChannelTS
      :raises ValueError: if the components are different
      :return: Combined channel with monotonic time index and same metadata
      :rtype: :class:`mth5.timeseries.ChannelTS`




   .. py:method:: to_xarray()

      Returns a :class:`xarray.DataArray` object of the channel timeseries
      this way metadata from the metadata class is updated upon return.

      :return: Returns a :class:`xarray.DataArray` object of the channel timeseries
      this way metadata from the metadata class is updated upon return.
      :rtype: :class:`xarray.DataArray`


      >>> import numpy as np
      >>> from mth5.timeseries import ChannelTS
      >>> ex = ChannelTS("electric")
      >>> ex.start = "2020-01-01T12:00:00"
      >>> ex.sample_rate = 16
      >>> ex.ts = np.random.rand(4096)





   .. py:method:: to_obspy_trace(network_code=None, encoding=None)

      Convert the time series to an :class:`obspy.core.trace.Trace` object.  This
      will be helpful for converting between data pulled from IRIS and data going
      into IRIS.

      :param network_code: two letter code provided by FDSN DMC
      :type network_code: string
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: from_obspy_trace(obspy_trace)

      Fill data from an :class:`obspy.core.Trace`

      :param obspy.core.trace obspy_trace: Obspy trace object




   .. py:method:: plot()

      Simple plot of the data

      :return: figure object
      :rtype: matplotlib.figure




   .. py:method:: welch_spectra(window_length=2**12, **kwargs)

      get welch spectra

      :param window_length: DESCRIPTION
      :type window_length: TYPE
      :param **kwargs: DESCRIPTION
      :type **kwargs: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: plot_spectra(spectra_type='welch', window_length=2**12, **kwargs)

      :param spectra_type: spectra type, defaults to "welch"
      :type spectra_type: string, optional
      :param window_length: window length of the welch method should be a
       power of 2, defaults to 2 ** 12
      :type window_length: int, optional
      :param **kwargs: DESCRIPTION
      :type **kwargs: TYPE




