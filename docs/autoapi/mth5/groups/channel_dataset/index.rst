mth5.groups.channel_dataset
===========================

.. py:module:: mth5.groups.channel_dataset

.. autoapi-nested-parse::

   Created on Sat May 27 10:03:23 2023

   @author: jpeacock



Attributes
----------

.. autoapisummary::

   mth5.groups.channel_dataset.meta_classes


Classes
-------

.. autoapisummary::

   mth5.groups.channel_dataset.ChannelDataset
   mth5.groups.channel_dataset.ElectricDataset
   mth5.groups.channel_dataset.MagneticDataset
   mth5.groups.channel_dataset.AuxiliaryDataset


Module Contents
---------------

.. py:data:: meta_classes

.. py:class:: ChannelDataset(dataset: h5py.Dataset | None, dataset_metadata: mt_metadata.base.MetadataBase | None = None, write_metadata: bool = True, **kwargs: Any)

   A container for channel time series data stored in HDF5 format.

   This class provides a flexible interface to work with magnetotelluric channel data,
   allowing conversion to various formats (xarray, pandas, numpy) while maintaining
   metadata integrity.

   :param dataset: HDF5 dataset object containing the channel time series data.
   :type dataset: h5py.Dataset or None
   :param dataset_metadata: Metadata container for Electric, Magnetic, or Auxiliary channel types.
                            Default is None.
   :type dataset_metadata: MetadataBase, optional
   :param write_metadata: Whether to write metadata to the HDF5 dataset on initialization.
                          Default is True.
   :type write_metadata: bool, optional
   :param \*\*kwargs: Additional keyword arguments to set as instance attributes.
   :type \*\*kwargs: dict

   .. attribute:: hdf5_dataset

      Weak reference to the underlying HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Channel metadata object with validation.

      :type: MetadataBase

   .. attribute:: logger

      Logger instance for tracking operations.

      :type: loguru.Logger

   :raises MTH5Error: If the dataset is not of the correct type or metadata validation fails.

   .. seealso::

      :obj:`ElectricDataset`
          Specialized container for electric field channels.

      :obj:`MagneticDataset`
          Specialized container for magnetic field channels.

      :obj:`AuxiliaryDataset`
          Specialized container for auxiliary channels.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
   >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
   >>> channel = run.get_channel('Ex')
   >>> channel
   Channel Electric:
   -------------------
     component:        Ex
     data type:        electric
     data format:      float32
     data shape:       (4096,)
     start:            1980-01-01T00:00:00+00:00
     end:              1980-01-01T00:00:01+00:00
     sample rate:      4096

   Access time series data

   >>> ts_data = channel.to_channel_ts()
   >>> print(f"Mean: {ts_data.ts.mean():.2f}, Std: {ts_data.ts.std():.2f}")

   Convert to xarray for time-based indexing

   >>> xr_data = channel.to_xarray()
   >>> subset = xr_data.sel(time=slice('1980-01-01T00:00:00', '1980-01-01T00:00:10'))


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Get the run-level metadata containing this channel.

      :returns: Run metadata object with channel information included.
      :rtype: metadata.Run

      .. rubric:: Examples

      >>> run_meta = channel.run_metadata
      >>> print(run_meta.id)
      'MT001a'
      >>> print(run_meta.channels_recorded_electric)
      ['Ex', 'Ey']


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Get the station-level metadata containing this channel.

      :returns: Station metadata object with run and channel information.
      :rtype: metadata.Station

      .. rubric:: Examples

      >>> station_meta = channel.station_metadata
      >>> print(f"{station_meta.id}: {station_meta.location.latitude}, {station_meta.location.longitude}")
      'MT001: 40.5, -112.3'


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Get the survey-level metadata containing this channel.

      :returns: Complete survey metadata hierarchy including this channel.
      :rtype: metadata.Survey

      .. rubric:: Examples

      >>> survey_meta = channel.survey_metadata
      >>> print(survey_meta.id)
      'MT Survey 2023'
      >>> print(f"Stations: {len(survey_meta.stations)}")
      Stations: 15


   .. py:property:: survey_id
      :type: str


      Get the survey identifier.

      :returns: Survey ID string.
      :rtype: str

      .. rubric:: Examples

      >>> print(channel.survey_id)
      'MT_Survey_2023'


   .. py:property:: channel_response
      :type: mt_metadata.timeseries.filters.ChannelResponse


      Get the complete channel response from applied filters.

      Constructs a ChannelResponse object by retrieving all filters referenced
      in the channel metadata from the survey's Filters group.

      :returns: Channel response object containing all applied filters in sequence.
      :rtype: ChannelResponse

      .. rubric:: Notes

      Filters are applied in the order specified by their sequence_number.
      Filter names are normalized by replacing '/' with ' per ' and converting
      to lowercase.

      .. rubric:: Examples

      >>> response = channel.channel_response
      >>> print(f"Number of filters: {len(response.filters_list)}")
      Number of filters: 3
      >>> for filt in response.filters_list:
      ...     print(f"{filt.name}: {filt.type}")
      zpk: zpk
      coefficient: coefficient
      time delay: time_delay


   .. py:property:: start
      :type: mt_metadata.common.mttime.MTime


      Get the start time of the channel data.

      :returns: Start time from metadata.time_period.start.
      :rtype: MTime

      .. rubric:: Examples

      >>> print(channel.start)
      1980-01-01T00:00:00+00:00
      >>> print(channel.start.iso_str)
      '1980-01-01T00:00:00.000000+00:00'


   .. py:property:: end
      :type: mt_metadata.common.mttime.MTime


      Calculate the end time based on start time, sample rate, and number of samples.

      :returns: Calculated end time of the data.
      :rtype: MTime

      .. rubric:: Notes

      End time is calculated as: start + (n_samples - 1) / sample_rate
      The -1 ensures the last sample falls exactly at the end time.

      .. rubric:: Examples

      >>> print(f"Duration: {channel.end - channel.start} seconds")
      Duration: 3600.0 seconds
      >>> print(channel.end.iso_str)
      '1980-01-01T01:00:00.000000+00:00'


   .. py:property:: sample_rate
      :type: float


      Get the sample rate in samples per second.

      :returns: Sample rate in Hz.
      :rtype: float

      .. rubric:: Examples

      >>> print(f"Sample rate: {channel.sample_rate} Hz")
      Sample rate: 256.0 Hz


   .. py:property:: n_samples
      :type: int


      Get the total number of samples in the dataset.

      :returns: Number of data points in the time series.
      :rtype: int

      .. rubric:: Examples

      >>> print(f"Total samples: {channel.n_samples:,}")
      Total samples: 921,600
      >>> duration = channel.n_samples / channel.sample_rate
      >>> print(f"Duration: {duration/3600:.1f} hours")
      Duration: 1.0 hours


   .. py:property:: time_index
      :type: pandas.DatetimeIndex


      Create a time index for the dataset based on metadata.

      :returns: Pandas datetime index spanning the entire dataset.
      :rtype: pd.DatetimeIndex

      .. rubric:: Notes

      The time index is useful for time-based queries and slicing operations.
      It is generated dynamically from start time, sample rate, and number of samples.

      .. rubric:: Examples

      >>> time_idx = channel.time_index
      >>> print(time_idx[0], time_idx[-1])
      1980-01-01 00:00:00 1980-01-01 00:59:59.996093750
      >>> print(f"Index length: {len(time_idx)}")
      Index length: 921600


   .. py:method:: read_metadata() -> None

      Read metadata from HDF5 attributes into the metadata container.

      Loads all HDF5 attributes from the dataset and converts them to the
      appropriate Python types before populating the metadata object.

      For older MTH5 files, this method attempts to coerce values to the
      expected types based on the metadata schema to maintain backwards
      compatibility.

      .. rubric:: Notes

      This method automatically validates metadata through the metadata
      container's validators. Type coercion is applied to handle older
      file formats that may have stored metadata with different types.

      .. rubric:: Examples

      >>> channel.read_metadata()
      >>> print(channel.metadata.component)
      'Ex'
      >>> print(channel.metadata.sample_rate)
      256.0

      Handles type coercion for older files

      >>> # If sample_rate was stored as string '256.0' in old file
      >>> channel.read_metadata()
      >>> print(type(channel.metadata.sample_rate))
      <class 'float'>



   .. py:method:: write_metadata() -> None

      Write metadata from the container to HDF5 dataset attributes.

      Converts all metadata values to numpy-compatible types before writing
      to HDF5 attributes. Falls back to string conversion if direct conversion fails.

      .. rubric:: Notes

      This method is automatically called during initialization and when
      metadata is updated.

      .. rubric:: Examples

      >>> channel.metadata.component = 'Ey'
      >>> channel.metadata.measurement_azimuth = 90.0
      >>> channel.write_metadata()



   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace the entire dataset with new data.

      :param new_data_array: New data array with shape (npts,). Must be 1-dimensional.
      :type new_data_array: np.ndarray

      :raises TypeError: If new_data_array cannot be converted to numpy array.

      .. rubric:: Notes

      The HDF5 dataset will be resized if the new array has a different shape.
      All existing data will be overwritten.

      .. rubric:: Examples

      Replace with synthetic data

      >>> import numpy as np
      >>> new_data = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2560))
      >>> channel.replace_dataset(new_data)
      >>> print(f"New shape: {channel.hdf5_dataset.shape}")
      New shape: (2560,)

      Replace with processed data

      >>> original = channel.hdf5_dataset[:]
      >>> filtered = np.convolve(original, np.ones(5)/5, mode='same')
      >>> channel.replace_dataset(filtered)



   .. py:method:: extend_dataset(new_data_array: numpy.ndarray, start_time: str | mt_metadata.common.mttime.MTime, sample_rate: float, fill: str | float | int | None = None, max_gap_seconds: float | int = 1, fill_window: int = 10) -> None

      Extend or prepend data to the existing dataset with gap handling.

      Intelligently adds new data before, after, or within the existing time series.
      Handles time alignment, overlaps, and gaps with configurable fill strategies.

      :param new_data_array: New data array with shape (npts,).
      :type new_data_array: np.ndarray
      :param start_time: Start time of the new data array in UTC.
      :type start_time: str or MTime
      :param sample_rate: Sample rate of the new data array in Hz. Must match existing sample rate.
      :type sample_rate: float
      :param fill: Strategy for filling data gaps:

                   - None : Raise MTH5Error if gap exists (default)
                   - 'mean' : Fill with mean of both datasets within fill_window
                   - 'median' : Fill with median of both datasets within fill_window
                   - 'nan' : Fill with NaN values
                   - numeric value : Fill with specified constant
      :type fill: str, float, int, or None, optional
      :param max_gap_seconds: Maximum allowed gap in seconds. Exceeding this raises MTH5Error.
                              Default is 1 second.
      :type max_gap_seconds: float or int, optional
      :param fill_window: Number of points from each dataset edge to estimate fill values.
                          Default is 10 points.
      :type fill_window: int, optional

      :raises MTH5Error: If sample rates don't match, gap exceeds max_gap_seconds, or
          fill strategy is invalid.
      :raises TypeError: If new_data_array cannot be converted to numpy array.

      .. rubric:: Notes

      - **Prepend**: New data start < existing start
      - **Append**: New data start > existing end
      - **Overwrite**: New data overlaps existing data

      The dataset is automatically resized to accommodate new data.

      .. rubric:: Examples

      Append data with a small gap

      >>> ex = mth5_obj.get_channel('MT001', 'MT001a', 'Ex')
      >>> print(f"Original: {ex.n_samples} samples, ends {ex.end}")
      Original: 4096 samples, ends 2015-01-08T19:32:09.500000+00:00
      >>> new_data = np.random.randn(4096)
      >>> new_start = (ex.end + 0.5).isoformat()  # 0.5s gap
      >>> ex.extend_dataset(new_data, new_start, ex.sample_rate,
      ...                   fill='median', max_gap_seconds=2)
      >>> print(f"Extended: {ex.n_samples} samples, ends {ex.end}")
      Extended: 8200 samples, ends 2015-01-08T19:40:42.500000+00:00

      Prepend data seamlessly

      >>> prepend_data = np.random.randn(2048)
      >>> prepend_start = (ex.start - 2048/ex.sample_rate).isoformat()
      >>> ex.extend_dataset(prepend_data, prepend_start, ex.sample_rate)
      >>> print(f"New start: {ex.start}")

      Overwrite section of existing data

      >>> replacement_data = np.zeros(1024)
      >>> replace_start = (ex.start + 1.0).isoformat()  # 1s after start
      >>> ex.extend_dataset(replacement_data, replace_start, ex.sample_rate)



   .. py:method:: has_data() -> bool

      Check if the channel contains non-zero data.

      :returns: True if dataset has non-zero values, False if all zeros or empty.
      :rtype: bool

      .. rubric:: Examples

      >>> if channel.has_data():
      ...     print("Channel has valid data")
      ... else:
      ...     print("Channel is empty or all zeros")
      Channel has valid data

      >>> empty_channel.has_data()
      False



   .. py:method:: to_channel_ts() -> mth5.timeseries.ChannelTS

      Convert the dataset to a ChannelTS object with full metadata.

      :returns: Time series object with data, metadata, and channel response.
      :rtype: ChannelTS

      .. rubric:: Notes

      Data is loaded into memory. The resulting ChannelTS object is independent
      of the HDF5 file and can be modified without affecting the original dataset.

      .. rubric:: Examples

      >>> ts = channel.to_channel_ts()
      >>> print(f"Type: {type(ts)}")
      Type: <class 'mth5.timeseries.channel_ts.ChannelTS'>
      >>> print(f"Shape: {ts.ts.shape}, Mean: {ts.ts.mean():.2f}")
      Shape: (4096,), Mean: 0.15

      Process the time series

      >>> filtered_ts = ts.low_pass_filter(cutoff=10.0)
      >>> detrended_ts = ts.detrend('linear')
      >>> ts.plot()



   .. py:method:: to_xarray() -> xarray.DataArray

      Convert the dataset to an xarray DataArray with time coordinates.

      :returns: DataArray with time index and metadata as attributes.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Data is loaded into memory. Metadata is stored in the attrs dictionary
      and will not be validated if modified.

      .. rubric:: Examples

      >>> xr_data = channel.to_xarray()
      >>> print(xr_data)
      <xarray.DataArray (time: 4096)>
      array([0.931, 0.142, ..., 0.882])
      Coordinates:
        * time     (time) datetime64[ns] 1980-01-01 ... 1980-01-01T00:00:15.996
      .. attribute:: component

         Ex

      .. attribute:: sample_rate

         256.0

      .. attribute:: ...

         

      Use xarray's powerful selection

      >>> morning = xr_data.sel(time=slice('1980-01-01T06:00', '1980-01-01T12:00'))
      >>> daily_mean = xr_data.resample(time='1D').mean()
      >>> xr_data.plot()



   .. py:method:: to_dataframe() -> pandas.DataFrame

      Convert the dataset to a pandas DataFrame with time index.

      :returns: DataFrame with 'data' column and time index. Metadata stored in attrs.
      :rtype: pd.DataFrame

      .. rubric:: Notes

      Data is loaded into memory. Metadata is stored in the experimental
      attrs attribute and will not be validated if modified.

      .. rubric:: Examples

      >>> df = channel.to_dataframe()
      >>> print(df.head())
                           data
      time
      1980-01-01 00:00:00  0.931
      1980-01-01 00:00:00  0.142
      ...

      Use pandas operations

      >>> df['data'].describe()
      >>> df.resample('1H').mean()
      >>> df.plot(y='data', figsize=(12, 4))

      Access metadata

      >>> print(df.attrs['component'])
      'Ex'
      >>> print(df.attrs['sample_rate'])
      256.0



   .. py:method:: to_numpy() -> numpy.recarray

      Convert the dataset to a numpy structured array with time and data columns.

      :returns: Record array with 'time' and 'channel_data' fields.
      :rtype: np.recarray

      .. rubric:: Notes

      Data is loaded into memory. The 'data' name is avoided as it's a
      builtin to numpy.

      .. rubric:: Examples

      >>> arr = channel.to_numpy()
      >>> print(arr.dtype.names)
      ('time', 'channel_data')
      >>> print(arr['time'][0])
      1980-01-01T00:00:00.000000000
      >>> print(arr['channel_data'].mean())
      0.152

      Access fields

      >>> times = arr['time']
      >>> data = arr['channel_data']
      >>> import matplotlib.pyplot as plt
      >>> plt.plot(times, data)



   .. py:method:: from_channel_ts(channel_ts_obj: mth5.timeseries.ChannelTS, how: str = 'replace', fill: str | float | int | None = None, max_gap_seconds: float | int = 1, fill_window: int = 10) -> None

      Populate the dataset from a ChannelTS object.

      :param channel_ts_obj: Time series object containing data and metadata.
      :type channel_ts_obj: ChannelTS
      :param how: Method for adding data:

                  - 'replace' : Replace entire dataset (default)
                  - 'extend' : Append/prepend to existing data with gap handling
      :type how: {'replace', 'extend'}, optional
      :param fill: Gap filling strategy (only used with how='extend'):

                   - None : Raise error on gaps (default)
                   - 'mean' : Fill with mean of both datasets
                   - 'median' : Fill with median of both datasets
                   - 'nan' : Fill with NaN
                   - numeric : Fill with constant value
      :type fill: str, float, int, or None, optional
      :param max_gap_seconds: Maximum allowed gap in seconds. Default is 1.
      :type max_gap_seconds: float or int, optional
      :param fill_window: Points to use for estimating fill values. Default is 10.
      :type fill_window: int, optional

      :raises TypeError: If channel_ts_obj is not a ChannelTS instance.
      :raises MTH5Error: If time alignment or metadata validation fails.

      .. rubric:: Examples

      Replace entire dataset

      >>> from mth5.timeseries import ChannelTS
      >>> import numpy as np
      >>> ts = ChannelTS(
      ...     channel_type='electric',
      ...     data=np.random.randn(1000),
      ...     channel_metadata={'electric': {
      ...         'component': 'ex',
      ...         'sample_rate': 256.0
      ...     }}
      ... )
      >>> channel.from_channel_ts(ts, how='replace')
      >>> print(channel.n_samples)
      1000

      Extend existing dataset

      >>> new_ts = ChannelTS(
      ...     channel_type='electric',
      ...     data=np.random.randn(500),
      ...     channel_metadata={'electric': {
      ...         'component': 'ex',
      ...         'sample_rate': 256.0,
      ...         'time_period.start': channel.end.isoformat()
      ...     }}
      ... )
      >>> channel.from_channel_ts(new_ts, how='extend', fill='median')
      >>> print(channel.n_samples)
      1500



   .. py:method:: from_xarray(data_array: xarray.DataArray, how: str = 'replace', fill: str | float | int | None = None, max_gap_seconds: float | int = 1, fill_window: int = 10) -> None

      Populate the dataset from an xarray DataArray.

      :param data_array: DataArray with time coordinate and metadata in attrs.
      :type data_array: xr.DataArray
      :param how: Method for adding data:

                  - 'replace' : Replace entire dataset (default)
                  - 'extend' : Append/prepend to existing data with gap handling
      :type how: {'replace', 'extend'}, optional
      :param fill: Gap filling strategy (only used with how='extend'):

                   - None : Raise error on gaps (default)
                   - 'mean' : Fill with mean of both datasets
                   - 'median' : Fill with median of both datasets
                   - 'nan' : Fill with NaN
                   - numeric : Fill with constant value
      :type fill: str, float, int, or None, optional
      :param max_gap_seconds: Maximum allowed gap in seconds. Default is 1.
      :type max_gap_seconds: float or int, optional
      :param fill_window: Points to use for estimating fill values. Default is 10.
      :type fill_window: int, optional

      :raises TypeError: If data_array is not an xarray.DataArray.
      :raises MTH5Error: If time alignment fails.

      .. rubric:: Examples

      Replace from xarray

      >>> import xarray as xr
      >>> import numpy as np
      >>> import pandas as pd
      >>> time = pd.date_range('2020-01-01', periods=1000, freq='0.004S')
      >>> data = xr.DataArray(
      ...     np.random.randn(1000),
      ...     coords=[('time', time)],
      ...     attrs={'component': 'ex', 'sample_rate': 256.0}
      ... )
      >>> channel.from_xarray(data, how='replace')
      >>> print(channel.n_samples)
      1000

      Extend from xarray with gap

      >>> time2 = pd.date_range('2020-01-01T00:00:05', periods=500, freq='0.004S')
      >>> data2 = xr.DataArray(np.random.randn(500), coords=[('time', time2)])
      >>> channel.from_xarray(data2, how='extend', fill='mean')



   .. py:property:: channel_entry
      :type: numpy.ndarray


      Create a structured array entry for channel summary tables.

      :returns: Structured array with dtype=CHANNEL_DTYPE containing channel metadata
                and HDF5 references for survey-wide summaries.
      :rtype: np.ndarray

      .. rubric:: Notes

      This entry includes survey ID, station ID, run ID, location, component,
      time period, sample rate, and HDF5 references for navigation.

      .. rubric:: Examples

      >>> entry = channel.channel_entry
      >>> print(entry['component'][0])
      'Ex'
      >>> print(entry['sample_rate'][0])
      256.0
      >>> print(entry['station'][0])
      'MT001'


   .. py:method:: time_slice(start: str | mt_metadata.common.mttime.MTime, end: str | mt_metadata.common.mttime.MTime | None = None, n_samples: int | None = None, return_type: str = 'channel_ts') -> mth5.timeseries.ChannelTS | xarray.DataArray | pandas.DataFrame | numpy.ndarray

      Extract a time slice from the channel dataset.

      :param start: Start time of the slice in UTC.
      :type start: str or MTime
      :param end: End time of the slice. Mutually exclusive with n_samples.
      :type end: str or MTime, optional
      :param n_samples: Number of samples to extract. Mutually exclusive with end.
      :type n_samples: int, optional
      :param return_type: Format for returned data. Default is 'channel_ts'.
      :type return_type: {'channel_ts', 'xarray', 'pandas', 'numpy'}, optional

      :returns: Time slice in the requested format with appropriate metadata.
      :rtype: ChannelTS or xr.DataArray or pd.DataFrame or np.ndarray

      :raises ValueError: If both end and n_samples are provided or neither is provided.

      .. rubric:: Notes

      - If the requested slice extends beyond available data, it will be
        automatically truncated with a warning.
      - Regional HDF5 references are used when possible for efficiency.

      .. rubric:: Examples

      Extract by number of samples

      >>> ex = mth5_obj.get_channel('FL001', 'FL001a', 'Ex')
      >>> ex_slice = ex.time_slice("2015-01-08T19:49:15", n_samples=4096)
      >>> print(type(ex_slice))
      <class 'mth5.timeseries.channel_ts.ChannelTS'>
      >>> print(f"Slice shape: {ex_slice.ts.shape}")
      Slice shape: (4096,)
      >>> ex_slice.plot()

      Extract by time range

      >>> ex_slice = ex.time_slice(
      ...     "2015-01-08T19:49:15",
      ...     end="2015-01-08T20:49:15"
      ... )
      >>> print(f"Duration: {ex_slice.end - ex_slice.start} seconds")
      Duration: 3600.0 seconds

      Return as xarray for analysis

      >>> xr_slice = ex.time_slice(
      ...     "2015-01-08T19:49:15",
      ...     n_samples=1000,
      ...     return_type='xarray'
      ... )
      >>> print(xr_slice.mean().values)
      0.152
      >>> xr_slice.plot()

      Return as pandas for tabular ops

      >>> df_slice = ex.time_slice(
      ...     "2015-01-08T19:49:15",
      ...     n_samples=500,
      ...     return_type='pandas'
      ... )
      >>> df_slice['data'].describe()
      >>> df_slice.resample('10S').mean()

      Return as numpy for computation

      >>> np_slice = ex.time_slice(
      ...     "2015-01-08T19:49:15",
      ...     n_samples=100,
      ...     return_type='numpy'
      ... )
      >>> np.fft.fft(np_slice)



   .. py:method:: get_index_from_time(given_time: str | mt_metadata.common.mttime.MTime) -> int

      Calculate the array index for a given time.

      :param given_time: Time to convert to index.
      :type given_time: str or MTime

      :returns: Array index corresponding to the given time.
      :rtype: int

      .. rubric:: Notes

      Index is calculated as: (time - start_time) * sample_rate
      and rounded to nearest integer.

      .. rubric:: Examples

      >>> idx = channel.get_index_from_time('1980-01-01T00:00:10')
      >>> print(f"Index for 10 seconds: {idx}")
      Index for 10 seconds: 2560
      >>> # With 256 Hz sample rate: 10 * 256 = 2560

      >>> start_idx = channel.get_index_from_time(channel.start)
      >>> print(start_idx)
      0



   .. py:method:: get_index_from_end_time(given_time: str | mt_metadata.common.mttime.MTime) -> int

      Get the end index value (inclusive) for a given time.

      :param given_time: Time to convert to end index.
      :type given_time: str or MTime

      :returns: Array index + 1 for inclusive slicing.
      :rtype: int

      .. rubric:: Notes

      Adds 1 to the calculated index to make it suitable for
      inclusive end slicing (e.g., array[start:end]).

      .. rubric:: Examples

      >>> end_idx = channel.get_index_from_end_time('1980-01-01T00:00:10')
      >>> data_slice = channel.hdf5_dataset[0:end_idx]
      >>> # Includes sample at exactly 10 seconds



.. py:class:: ElectricDataset(group: h5py.Dataset, **kwargs: Any)

   Bases: :py:obj:`ChannelDataset`


   Specialized container for electric field channel data.

   Inherits all functionality from ChannelDataset with electric field
   specific metadata handling.

   :param group: HDF5 dataset containing electric field data.
   :type group: h5py.Dataset
   :param \*\*kwargs: Additional keyword arguments passed to ChannelDataset.
   :type \*\*kwargs: dict

   .. rubric:: Examples

   >>> ex_dataset = run_group.get_channel('Ex')
   >>> print(type(ex_dataset))
   <class 'mth5.groups.channel_dataset.ElectricDataset'>
   >>> print(ex_dataset.metadata.type)
   'electric'
   >>> print(ex_dataset.metadata.units)
   'mV/km'


.. py:class:: MagneticDataset(group: h5py.Dataset, **kwargs: Any)

   Bases: :py:obj:`ChannelDataset`


   Specialized container for magnetic field channel data.

   Inherits all functionality from ChannelDataset with magnetic field
   specific metadata handling.

   :param group: HDF5 dataset containing magnetic field data.
   :type group: h5py.Dataset
   :param \*\*kwargs: Additional keyword arguments passed to ChannelDataset.
   :type \*\*kwargs: dict

   .. rubric:: Examples

   >>> hx_dataset = run_group.get_channel('Hx')
   >>> print(type(hx_dataset))
   <class 'mth5.groups.channel_dataset.MagneticDataset'>
   >>> print(hx_dataset.metadata.type)
   'magnetic'
   >>> print(hx_dataset.metadata.units)
   'nT'


.. py:class:: AuxiliaryDataset(group: h5py.Dataset, **kwargs: Any)

   Bases: :py:obj:`ChannelDataset`


   Specialized container for auxiliary channel data.

   Inherits all functionality from ChannelDataset with auxiliary channel
   specific metadata handling. Used for temperature, battery voltage, etc.

   :param group: HDF5 dataset containing auxiliary data.
   :type group: h5py.Dataset
   :param \*\*kwargs: Additional keyword arguments passed to ChannelDataset.
   :type \*\*kwargs: dict

   .. rubric:: Examples

   >>> temp_dataset = run_group.get_channel('Temperature')
   >>> print(type(temp_dataset))
   <class 'mth5.groups.channel_dataset.AuxiliaryDataset'>
   >>> print(temp_dataset.metadata.type)
   'auxiliary'
   >>> print(temp_dataset.metadata.units)
   'celsius'


