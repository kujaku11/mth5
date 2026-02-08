mth5.timeseries
===============

.. py:module:: mth5.timeseries


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/timeseries/channel_ts/index
   /autoapi/mth5/timeseries/run_ts/index
   /autoapi/mth5/timeseries/scipy_filters/index
   /autoapi/mth5/timeseries/spectre/index
   /autoapi/mth5/timeseries/ts_filters/index
   /autoapi/mth5/timeseries/ts_helpers/index
   /autoapi/mth5/timeseries/xarray_helpers/index


Classes
-------

.. autoapisummary::

   mth5.timeseries.ChannelTS
   mth5.timeseries.RunTS


Package Contents
----------------

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




.. py:class:: RunTS(array_list: list[mth5.timeseries.channel_ts.ChannelTS] | list[xarray.DataArray] | xarray.Dataset | None = None, run_metadata: mt_metadata.timeseries.Run | dict | None = None, station_metadata: mt_metadata.timeseries.Station | dict | None = None, survey_metadata: mt_metadata.timeseries.Survey | dict | None = None)

   Container for MT time series data from a single run.

   Holds all run time series in one aligned xarray Dataset with channels as
   data variables and time as the coordinate. Manages metadata for survey,
   station, and run levels.

   :param array_list: List of ChannelTS objects, xarray DataArrays, or an xarray Dataset
                      containing the time series data. All channels will be aligned to a
                      common time index.
   :type array_list: list[ChannelTS] | list[xr.DataArray] | xr.Dataset | None, optional
   :param run_metadata: Metadata for the run. Can be a Run object or dictionary.
   :type run_metadata: timeseries.Run | dict | None, optional
   :param station_metadata: Metadata for the station. Can be a Station object or dictionary.
   :type station_metadata: timeseries.Station | dict | None, optional
   :param survey_metadata: Metadata for the survey. Can be a Survey object or dictionary.
   :type survey_metadata: timeseries.Survey | dict | None, optional

   .. attribute:: dataset

      xarray Dataset containing all channel data with time coordinate

      :type: xr.Dataset

   .. attribute:: survey_metadata

      Survey-level metadata

      :type: timeseries.Survey

   .. attribute:: station_metadata

      Station-level metadata

      :type: timeseries.Station

   .. attribute:: run_metadata

      Run-level metadata

      :type: timeseries.Run

   .. attribute:: filters

      Dictionary of channel response filters keyed by filter name

      :type: dict[str, Filter]

   .. attribute:: sample_rate

      Sample rate in samples per second

      :type: float

   .. attribute:: channels

      List of channel names in the dataset

      :type: list[str]

   .. rubric:: Examples

   Create an empty RunTS:

   >>> from mth5.timeseries import RunTS
   >>> run = RunTS()

   Create RunTS from ChannelTS objects:

   >>> from mth5.timeseries import ChannelTS, RunTS
   >>> ex = ChannelTS('electric', data=ex_data,
   ...                channel_metadata={'component': 'ex'})
   >>> ey = ChannelTS('electric', data=ey_data,
   ...                channel_metadata={'component': 'ey'})
   >>> run = RunTS(array_list=[ex, ey])
   >>> print(run.channels)
   ['ex', 'ey']

   Access individual channels:

   >>> ex_channel = run.ex  # Returns ChannelTS object
   >>> print(ex_channel.sample_rate)
   256.0

   .. seealso::

      :obj:`ChannelTS`
          Individual channel time series container

   .. rubric:: Notes

   When multiple channels are provided with different start/end times,
   they will be automatically aligned using the earliest start and latest
   end times, with NaN values filling gaps.


   .. py:attribute:: logger


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Survey timeseries.

      :returns: Survey-level metadata object.
      :rtype: timeseries.Survey


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Station timeseries.

      :returns: Station-level metadata object (first station in survey).
      :rtype: timeseries.Station


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Run timeseries.

      :returns: Run-level metadata object (first run in first station).
      :rtype: timeseries.Run


   .. py:method:: copy(data: bool = True) -> RunTS

      Create a copy of the RunTS object.

      :param data: If True, copy the data along with timeseries. If False, only
                   copy the metadata (default is True).
      :type data: bool, optional

      :returns: A copy of the RunTS object.
      :rtype: RunTS

      .. rubric:: Examples

      Create a copy with data:

      >>> run_copy = run.copy()

      Create a metadata-only copy:

      >>> run_meta = run.copy(data=False)
      >>> print(run_meta.has_data())
      False



   .. py:method:: has_data() -> bool

      Check if the RunTS contains any data.

      :returns: True if channels with data exist, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> run = RunTS()
      >>> print(run.has_data())
      False
      >>> run.add_channel(ex_channel)
      >>> print(run.has_data())
      True



   .. py:property:: summarize_metadata
      :type: dict[str, any]


      Get a summary of all channel timeseries.

      Flattens the metadata from all channels into a single dictionary
      with keys in the format 'channel.attribute'.

      :returns: Dictionary with flattened metadata from all channels.
      :rtype: dict[str, any]

      .. rubric:: Examples

      >>> meta_summary = run.summarize_metadata
      >>> print(meta_summary.keys())
      dict_keys(['ex.time_period.start', 'ex.sample_rate', ...])


   .. py:method:: validate_metadata() -> None

      Check to make sure that the metadata matches what is in the data set.

      updates metadata from the data.

      Check the start and end times, channels recorded




   .. py:method:: set_dataset(array_list: list[mth5.timeseries.channel_ts.ChannelTS] | list[xarray.DataArray] | xarray.Dataset, align_type: str = 'outer') -> None

      Set the dataset from a list of channels or existing dataset.

      Creates an xarray Dataset from the input channels or dataset, validates
      metadata consistency, and updates dataset attributes with run metadata.

      :param array_list: Input data as a list of ChannelTS objects, list of xarray DataArrays,
                         or an existing xarray Dataset.
      :type array_list: list[ChannelTS] | list[xr.DataArray] | xr.Dataset
      :param align_type: Method for aligning channels with different time indexes:

                         * 'outer' - use union of all time indexes (default)
                         * 'inner' - use intersection of time indexes
                         * 'left' - use time index from first channel
                         * 'right' - use time index from last channel
                         * 'exact' - raise ValueError if indexes don't match exactly
                         * 'override' - rewrite indexes to match first channel (requires same size)
      :type align_type: str, optional

      .. rubric:: Notes

      This method performs the following operations:

      1. Validates the input array_list
      2. Converts ChannelTS objects to xarray format
      3. Combines channels into a single Dataset
      4. Validates metadata consistency
      5. Updates dataset attributes with run metadata

      When providing ChannelTS objects, their metadata and filters are
      automatically extracted and merged into the run's metadata structure.

      .. rubric:: Examples

      Set dataset from ChannelTS objects:

      >>> ex = ChannelTS('electric', data=ex_data,
      ...                channel_metadata={'component': 'ex'})
      >>> ey = ChannelTS('electric', data=ey_data,
      ...                channel_metadata={'component': 'ey'})
      >>> run.set_dataset([ex, ey])

      Set dataset with custom alignment:

      >>> run.set_dataset([ex, ey, hx], align_type='inner')

      Set dataset from existing xarray Dataset:

      >>> import xarray as xr
      >>> ds = xr.Dataset({'ex': ex_da, 'ey': ey_da})
      >>> run.set_dataset(ds)

      .. seealso::

         :obj:`dataset`
             Property for setting dataset with default alignment

         :obj:`add_channel`
             Add a single channel to existing dataset

         :obj:`_validate_array_list`
             Validation and conversion of array list



   .. py:method:: add_channel(channel: xarray.DataArray | mth5.timeseries.channel_ts.ChannelTS) -> None

      Add a channel to the dataset.

      The channel must have the same sample rate and time coordinates that
      are compatible with the existing dataset. If start times don't match,
      NaN values will be placed where timing doesn't align.

      :param channel: A channel as an xarray DataArray or ChannelTS object to add.
      :type channel: xr.DataArray | ChannelTS

      :raises ValueError: If the channel has a different sample rate than the run, or if
          the input is not a DataArray or ChannelTS.

      .. rubric:: Examples

      Add a ChannelTS:

      >>> hz = ChannelTS('magnetic', data=hz_data,
      ...                channel_metadata={'component': 'hz'})
      >>> run.add_channel(hz)
      >>> print(run.channels)
      ['ex', 'ey', 'hx', 'hy', 'hz']

      Add an xarray DataArray:

      >>> import xarray as xr
      >>> data_array = xr.DataArray(data, coords={'time': times})
      >>> run.add_channel(data_array)



   .. py:property:: dataset
      :type: xarray.Dataset


      The xarray Dataset containing all channel data.

      :returns: Dataset with channels as data variables and time as coordinate.
      :rtype: xr.Dataset

      .. rubric:: Examples

      >>> print(run.dataset)
      <xarray.Dataset>
      Dimensions:  (time: 4096)
      Coordinates:
        * time     (time) datetime64[ns] ...
      Data variables:
          ex       (time) float64 ...
          ey       (time) float64 ...


   .. py:property:: start
      :type: mt_metadata.common.mttime.MTime


      Start time of the run in UTC.

      :returns: Start time from the dataset if data exists, otherwise from
                run_metadata.
      :rtype: MTime

      .. rubric:: Examples

      >>> print(run.start)
      2020-01-01T00:00:00+00:00


   .. py:property:: end
      :type: mt_metadata.common.mttime.MTime


      End time of the run in UTC.

      :returns: End time from the dataset if data exists, otherwise from
                run_metadata.
      :rtype: MTime

      .. rubric:: Examples

      >>> print(run.end)
      2020-01-01T01:00:00+00:00


   .. py:property:: sample_rate
      :type: float


      Sample rate in samples per second.

      :returns: Sample rate estimated from time differences if data exists,
                otherwise from timeseries.
      :rtype: float

      .. rubric:: Examples

      >>> print(run.sample_rate)
      256.0


   .. py:property:: sample_interval
      :type: float


      Sample interval in seconds (inverse of sample_rate).

      :returns: Sample interval = 1 / sample_rate, or 0.0 if sample_rate is 0.
      :rtype: float

      .. rubric:: Examples

      >>> print(run.sample_interval)
      0.00390625  # for 256 Hz


   .. py:property:: channels
      :type: list[str]


      List of channel names in the dataset.

      :returns: List of channel component names (e.g., ['ex', 'ey', 'hx']).
      :rtype: list[str]

      .. rubric:: Examples

      >>> print(run.channels)
      ['ex', 'ey', 'hx', 'hy', 'hz']


   .. py:property:: filters
      :type: dict[str, mt_metadata.timeseries.filters.ChannelResponse]


      Dictionary of channel response filters.

      :returns: Dictionary keyed by filter name containing ChannelResponse objects.
      :rtype: dict[str, ChannelResponse]

      .. rubric:: Examples

      >>> print(run.filters.keys())
      dict_keys(['v_to_counts', 'dipole_100m'])


   .. py:method:: to_obspy_stream(network_code: str | None = None, encoding: str | None = None) -> obspy.core.Stream

      Convert time series to an ObsPy Stream object.

      Creates an ObsPy Stream containing a Trace for each channel in the run.

      :param network_code: Two-letter network code provided by FDSN DMC. If None, uses
                           station timeseries.
      :type network_code: str | None, optional
      :param encoding: Data encoding format (e.g., 'STEIM2', 'FLOAT64'). If None, uses
                       default encoding.
      :type encoding: str | None, optional

      :returns: Stream object containing Trace objects for all channels.
      :rtype: obspy.core.Stream

      .. rubric:: Examples

      >>> stream = run.to_obspy_stream(network_code='MT')
      >>> print(stream)
      3 Trace(s) in Stream:
      MT.MT001..EX | 2020-01-01T00:00:00 - ... | 256.0 Hz, 4096 samples

      .. seealso::

         :obj:`from_obspy_stream`
             Create RunTS from ObsPy Stream

         :obj:`ChannelTS.to_obspy_trace`
             Convert single channel



   .. py:method:: wrangle_leap_seconds_from_obspy(array_list: list[mth5.timeseries.channel_ts.ChannelTS]) -> list[mth5.timeseries.channel_ts.ChannelTS]

      Handle potential leap second issues from ObsPy streams.

      Removes runs with only one sample that are numerically identical to
      adjacent samples, which may be artifacts of leap second handling.

      :param array_list: List of ChannelTS objects from ObsPy conversion.
      :type array_list: list[ChannelTS]

      :returns: Filtered list with single-sample runs removed.
      :rtype: list[ChannelTS]

      .. rubric:: Notes

      This is experimental handling for issue #169. The exact behavior of
      ObsPy's leap second handling is not fully documented.



   .. py:method:: from_obspy_stream(obspy_stream: obspy.core.Stream, run_metadata: mt_metadata.timeseries.Run | None = None) -> None

      Get a run from an :class:`obspy.core.stream` which is a list of
      :class:`obspy.core.Trace` objects.

      :param obspy_stream: Obspy Stream object
      :type obspy_stream: :class:`obspy.core.Stream`

      Development Notes:
       - There is a baked in assumption here that the channel nomenclature
         in obspy is e1,e2,h1,h2,h3 and we want to convert to mth5 conventions
         ex,ey,hx,hy,hz.  This should be made more flexible in the future.
       - A bug was found that was creating channels e1, ex, ey in the same run
         when reading from obspy -- this is fixed here by renaming the components and a workaround
         to reset the station's channels_recorded list.





   .. py:method:: get_slice(start: str | mt_metadata.common.mttime.MTime, end: str | mt_metadata.common.mttime.MTime | None = None, n_samples: int | None = None) -> RunTS

      Extract a time slice from the run.

      Gets a chunk of data from the run, finding the closest points to the
      given parameters. Uses pandas slice_indexer for robust slicing.

      :param start: Start time of the slice (ISO format string or MTime object).
      :type start: str | MTime
      :param end: End time of the slice. Required if n_samples not provided.
      :type end: str | MTime | None, optional
      :param n_samples: Number of samples to get. Required if end not provided.
      :type n_samples: int | None, optional

      :returns: New RunTS object containing the requested slice with copies of
                metadata and filters.
      :rtype: RunTS

      :raises ValueError: If neither end nor n_samples is provided.

      .. rubric:: Examples

      Get slice by start and end times:

      >>> slice1 = run.get_slice('2020-01-01T00:00:00',
      ...                         '2020-01-01T00:01:00')
      >>> print(slice1.start, slice1.end)

      Get slice by start time and number of samples:

      >>> slice2 = run.get_slice('2020-01-01T00:00:00', n_samples=1024)
      >>> print(len(slice2.dataset.time))
      1024

      .. rubric:: Notes

      Uses pandas slice_indexer which handles near-matches better than
      xarray's native slicing. The actual slice may be slightly adjusted
      to match available data points.



   .. py:method:: calibrate(**kwargs) -> RunTS

      Remove instrument response from all channels.

      Applies the channel response filters to calibrate each channel,
      creating a new run with calibrated data.

      :param \*\*kwargs: Additional keyword arguments passed to each channel's
                         remove_instrument_response method.

      :returns: New RunTS object with calibrated channels.
      :rtype: RunTS

      .. rubric:: Examples

      >>> calibrated_run = run.calibrate()
      >>> # Calibration typically converts from counts to physical units

      .. seealso::

         :obj:`ChannelTS.remove_instrument_response`
             Calibrate single channel



   .. py:method:: decimate(new_sample_rate: float, inplace: bool = False, max_decimation: int = 8) -> RunTS | None

      Decimate data to a new sample rate using multi-stage decimation.

      Applies FIR filtering and downsampling in multiple stages to achieve
      the target sample rate while preventing aliasing.

      :param new_sample_rate: Target sample rate in samples per second.
      :type new_sample_rate: float
      :param inplace: If True, modify the current run. If False, return a new run
                      (default is False).
      :type inplace: bool, optional
      :param max_decimation: Maximum decimation factor for each stage (default is 8).
      :type max_decimation: int, optional

      :returns: If inplace is False, returns new decimated RunTS. Otherwise None.
      :rtype: RunTS | None

      .. rubric:: Examples

      Decimate from 256 Hz to 1 Hz:

      >>> decimated_run = run.decimate(1.0)
      >>> print(decimated_run.sample_rate)
      1.0

      Decimate in place:

      >>> run.decimate(16.0, inplace=True)
      >>> print(run.sample_rate)
      16.0

      .. rubric:: Notes

      NaN values are filled with 0 before decimation to prevent NaN
      propagation. Multi-stage decimation is used to maintain signal
      quality and prevent aliasing.

      .. seealso::

         :obj:`resample_poly`
             Alternative resampling using polyphase filtering

         :obj:`resample`
             Simple resampling without anti-aliasing



   .. py:method:: resample_poly(new_sample_rate: float, pad_type: str = 'mean', inplace: bool = False) -> RunTS | None

      Resample data using polyphase filtering.

      Uses scipy.signal.resample_poly to resample while applying an FIR
      filter to remove aliasing. Generally more accurate than simple
      resampling but slower than decimation.

      :param new_sample_rate: Target sample rate in samples per second.
      :type new_sample_rate: float
      :param pad_type: Padding method for edge effects: 'mean', 'median', 'zero'
                       (default is 'mean').
      :type pad_type: str, optional
      :param inplace: If True, modify current run. If False, return new run
                      (default is False).
      :type inplace: bool, optional

      :returns: If inplace is False, returns new resampled RunTS. Otherwise None.
      :rtype: RunTS | None

      .. rubric:: Examples

      Resample from 256 Hz to 100 Hz:

      >>> resampled_run = run.resample_poly(100.0)
      >>> print(resampled_run.sample_rate)
      100.0

      .. rubric:: Notes

      NaN values are filled with 0 before resampling. The polyphase method
      is particularly good for arbitrary sample rate ratios.

      .. seealso::

         :obj:`decimate`
             Multi-stage decimation for downsampling

         :obj:`resample`
             Simple nearest-neighbor resampling



   .. py:method:: resample(new_sample_rate: float, inplace: bool = False) -> RunTS | None

      Resample data to a new sample rate using nearest-neighbor method.

      Simple resampling without anti-aliasing filtering. Use decimate or
      resample_poly for better quality when downsampling.

      :param new_sample_rate: Target sample rate in samples per second.
      :type new_sample_rate: float
      :param inplace: If True, modify current run. If False, return new run
                      (default is False).
      :type inplace: bool, optional

      :returns: If inplace is False, returns new resampled RunTS. Otherwise None.
      :rtype: RunTS | None

      .. rubric:: Examples

      >>> resampled_run = run.resample(128.0)
      >>> print(resampled_run.sample_rate)
      128.0

      .. warning::

         This method does not apply anti-aliasing filtering. For downsampling,
         consider using decimate() or resample_poly() instead.

      .. seealso::

         :obj:`decimate`
             Proper downsampling with anti-aliasing

         :obj:`resample_poly`
             High-quality resampling with polyphase filtering



   .. py:method:: merge(other: RunTS | list[RunTS], gap_method: str = 'slinear', new_sample_rate: float | None = None, resample_method: str = 'poly') -> RunTS

      Merge multiple runs into a single run.

      Combines this run with one or more other runs, optionally resampling
      to a common sample rate and filling gaps with interpolation.

      :param other: Another RunTS object or list of RunTS objects to merge.
      :type other: RunTS | list[RunTS]
      :param gap_method: Interpolation method for filling gaps: 'linear', 'nearest',
                         'zero', 'slinear', 'quadratic', 'cubic' (default is 'slinear').
      :type gap_method: str, optional
      :param new_sample_rate: If provided, all runs will be resampled to this rate before
                              merging. If None, uses the sample rate of the first run.
      :type new_sample_rate: float | None, optional
      :param resample_method: Resampling method if new_sample_rate is provided: 'decimate'
                              or 'poly' (default is 'poly').
      :type resample_method: str, optional

      :returns: New merged RunTS object with monotonic time index.
      :rtype: RunTS

      :raises TypeError: If other is not a RunTS or list of RunTS objects.

      .. rubric:: Examples

      Merge two runs:

      >>> run1 = RunTS(array_list=[ex1, ey1])
      >>> run2 = RunTS(array_list=[ex2, ey2])
      >>> merged = run1.merge(run2)

      Merge multiple runs with resampling:

      >>> runs = [run1, run2, run3]
      >>> merged = run1.merge(runs, new_sample_rate=1.0,
      ...                     resample_method='poly')

      .. rubric:: Notes

      The merge process:

      1. Optionally resample all runs to common sample rate
      2. Combine datasets using xr.combine_by_coords
      3. Create monotonic time index spanning full range
      4. Interpolate to new index filling gaps
      5. Merge all filter dictionaries

      Metadata is taken from the first run (self).

      .. seealso::

         :obj:`__add__`
             Simple merging with + operator



   .. py:method:: plot(color_map: dict[str, tuple[float, float, float]] | None = None, channel_order: list[str] | None = None) -> matplotlib.figure.Figure

      Plot all channels as time series.

      Creates a multi-panel figure with each channel in its own subplot,
      sharing a common time axis.

      :param color_map: Dictionary mapping channel names to RGB color tuples (values 0-1).
                        Default colors:

                        - ex: (1, 0.2, 0.2) - red
                        - ey: (1, 0.5, 0) - orange
                        - hx: (0, 0.5, 1) - blue
                        - hy: (0.5, 0.2, 1) - purple
                        - hz: (0.2, 1, 1) - cyan
      :type color_map: dict[str, tuple[float, float, float]] | None, optional
      :param channel_order: Order of channels from top to bottom. If None, uses order from
                            self.channels.
      :type channel_order: list[str] | None, optional

      :returns: Figure object containing the plot.
      :rtype: matplotlib.figure.Figure

      .. rubric:: Examples

      Plot with default settings:

      >>> fig = run.plot()
      >>> fig.savefig('timeseries.png')

      Plot with custom colors and order:

      >>> colors = {'ex': (1, 0, 0), 'ey': (0, 1, 0)}
      >>> fig = run.plot(color_map=colors, channel_order=['ey', 'ex'])

      .. warning::

         May be slow for large datasets (millions of points). Consider
         using get_slice() first to plot a subset.



   .. py:method:: plot_spectra(spectra_type: str = 'welch', color_map: dict[str, tuple[float, float, float]] | None = None, **kwargs) -> matplotlib.figure.Figure

      Plot power spectral density for all channels.

      Computes and plots the power spectrum of each channel on a single
      log-log plot with period on x-axis.

      :param spectra_type: Type of spectral estimate to compute. Currently only 'welch'
                           is supported (default is 'welch').
      :type spectra_type: str, optional
      :param color_map: Dictionary mapping channel names to RGB color tuples (values 0-1).
                        Uses same defaults as plot().
      :type color_map: dict[str, tuple[float, float, float]] | None, optional
      :param \*\*kwargs: Additional keyword arguments passed to the spectra computation
                         method (e.g., nperseg, window for Welch's method).

      :returns: Figure object containing the spectra plot.
      :rtype: matplotlib.figure.Figure

      .. rubric:: Examples

      Plot spectra with default settings:

      >>> fig = run.plot_spectra()

      Plot with custom Welch parameters:

      >>> fig = run.plot_spectra(nperseg=1024, window='hann')

      .. rubric:: Notes

      The plot shows:

      - Period (seconds) on bottom x-axis
      - Frequency (Hz) on top x-axis
      - Power (dB) on y-axis

      .. seealso::

         :obj:`ChannelTS.welch_spectra`
             Compute Welch power spectrum



