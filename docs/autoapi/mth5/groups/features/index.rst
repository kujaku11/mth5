mth5.groups.features
====================

.. py:module:: mth5.groups.features

.. autoapi-nested-parse::

   Created on Fri Dec 13 12:40:34 2024

   @author: jpeacock



Attributes
----------

.. autoapisummary::

   mth5.groups.features.TIME_DOMAIN
   mth5.groups.features.FREQUENCY_DOMAIN


Classes
-------

.. autoapisummary::

   mth5.groups.features.MasterFeaturesGroup
   mth5.groups.features.FeatureGroup
   mth5.groups.features.FeatureTSRunGroup
   mth5.groups.features.FeatureFCRunGroup
   mth5.groups.features.FeatureDecimationGroup


Module Contents
---------------

.. py:data:: TIME_DOMAIN
   :value: ['ts', 'time', 'time series', 'time_series']


.. py:data:: FREQUENCY_DOMAIN
   :value: ['fc', 'frequency', 'fourier', 'fourier_domain']


.. py:class:: MasterFeaturesGroup(group: h5py.Group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Master group container for features associated with Fourier Coefficients or time series.

   This class manages the top-level organization of geophysical feature data,
   organizing it into feature-specific groups. Features can include various
   frequency or time-domain analyses.

   Hierarchy
   ---------
   MasterFeatureGroup -> FeatureGroup -> FeatureRunGroup ->

   - FC: FeatureDecimationGroup -> FeatureChannelDataset
   - Time Series: FeatureChannelDataset

   :param group: HDF5 group object for this MasterFeaturesGroup.
   :type group: h5py.Group
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> import h5py
   >>> from mth5.groups.features import MasterFeaturesGroup
   >>> with h5py.File('data.h5', 'r') as f:
   ...     master = MasterFeaturesGroup(f['features'])
   ...     feature_list = master.groups_list


   .. py:method:: add_feature_group(feature_name: str, feature_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None) -> FeatureGroup

      Add a feature group to the master features container.

      Creates a new FeatureGroup with the specified name and optional metadata.
      Feature groups organize all runs and decimation levels for a particular feature.

      :param feature_name: Name for the feature group. Will be validated and formatted.
      :type feature_name: str
      :param feature_metadata: Metadata describing the feature. Default is None.
      :type feature_metadata: FeatureDecimationChannel, optional

      :returns: Newly created feature group object.
      :rtype: FeatureGroup

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> feature = master.add_feature_group('coherency')
      >>> print(feature.name)
      'coherency'



   .. py:method:: get_feature_group(feature_name: str) -> FeatureGroup

      Retrieve a feature group by name.

      :param feature_name: Name of the feature group to retrieve.
      :type feature_name: str

      :returns: The requested feature group.
      :rtype: FeatureGroup

      :raises MTH5Error: If the feature group does not exist.

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> feature = master.get_feature_group('coherency')
      >>> print(feature.name)
      'coherency'



   .. py:method:: remove_feature_group(feature_name: str) -> None

      Remove a feature group from the master container.

      Deletes the specified feature group and its associated data from the
      HDF5 file. Note that this operation removes the reference but does not
      reduce the file size; copy desired data to a new file for size reduction.

      :param feature_name: Name of the feature group to remove.
      :type feature_name: str

      :raises MTH5Error: If the feature group does not exist.

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> master.remove_feature_group('coherency')



.. py:class:: FeatureGroup(group: h5py.Group, feature_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single feature set with all associated runs and decimation levels.

   This class manages feature-specific data including all processing runs and
   decimation levels. Features can include both Fourier Coefficient and time series data.

   Hierarchy
   ---------
   FeatureGroup -> FeatureRunGroup ->

   - FC: FeatureDecimationLevel -> FeatureChannelDataset
   - TS: FeatureChannelDataset

   :param group: HDF5 group object for this FeatureGroup.
   :type group: h5py.Group
   :param feature_metadata: Metadata specific to this feature. Should include description and parameters.
   :type feature_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Notes

   Feature metadata should be specific to the feature and include descriptions
   of the feature and any parameters used in its computation.

   .. rubric:: Examples

   >>> feature = FeatureGroup(h5_group, feature_metadata=metadata)
   >>> run_group = feature.add_feature_run_group('run_1', domain='fc')


   .. py:method:: add_feature_run_group(feature_name: str, feature_run_metadata: Optional[object] = None, domain: str = 'fc') -> object

      Add a feature run group for a single feature.

      Creates either a Fourier Coefficient run group or a time series run group
      based on the specified domain. The domain can be determined from the metadata
      or explicitly provided.

      :param feature_name: Name for the feature run group.
      :type feature_name: str
      :param feature_run_metadata: Metadata for the feature run. If provided, domain is extracted from
                                   metadata.domain attribute. Default is None.
      :type feature_run_metadata: optional
      :param domain: Domain type for the data. Must be one of:

                     - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
                     - 'ts', 'time', 'time series', 'time_series': Time series
      :type domain: str, default='fc'

      :returns: Newly created feature run group.
      :rtype: FeatureFCRunGroup or FeatureTSRunGroup

      :raises ValueError: If domain is not recognized.
      :raises AttributeError: If metadata does not have a domain attribute when metadata is provided.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> fc_run = feature.add_feature_run_group('processing_run_1', domain='fc')
      >>> ts_run = feature.add_feature_run_group('ts_analysis', domain='ts')



   .. py:method:: get_feature_run_group(feature_name: str, domain: str = 'frequency') -> object

      Retrieve a feature run group by name and domain type.

      :param feature_name: Name of the feature run group to retrieve.
      :type feature_name: str
      :param domain: Domain type. Must be one of:

                     - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
                     - 'ts', 'time', 'time series', 'time_series': Time series
      :type domain: str, default='frequency'

      :returns: The requested feature run group.
      :rtype: FeatureFCRunGroup or FeatureTSRunGroup

      :raises ValueError: If domain is not recognized.
      :raises MTH5Error: If the feature run group does not exist.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> fc_run = feature.get_feature_run_group('processing_run_1', domain='fc')



   .. py:method:: remove_feature_run_group(feature_name: str) -> None

      Remove a feature run group.

      Deletes the specified feature run group and all its associated data.
      Note that deletion removes the reference but does not reduce HDF5 file size.

      :param feature_name: Name of the feature run group to remove.
      :type feature_name: str

      :raises MTH5Error: If the feature run group does not exist.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> feature.remove_feature_run_group('processing_run_1')



.. py:class:: FeatureTSRunGroup(group: h5py.Group, feature_run_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for time series features from a processing or analysis run.

   This class wraps a RunGroup to manage time series data features while
   maintaining compatibility with the feature hierarchy structure.

   :param group: HDF5 group object for this FeatureTSRunGroup.
   :type group: h5py.Group
   :param feature_run_metadata: Metadata for the feature run (same type as timeseries.Run).
   :type feature_run_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Notes

   This class uses methods from RunGroup for channel management, which may
   have performance implications due to multiple RunGroup instantiations.

   .. rubric:: Examples

   >>> ts_run = FeatureTSRunGroup(h5_group, feature_run_metadata=metadata)
   >>> channel = ts_run.add_feature_channel('Ex', 'electric', data)


   .. py:method:: add_feature_channel(channel_name: str, channel_type: str, data: Optional[numpy.ndarray] = None, channel_dtype: str = 'int32', shape: Optional[tuple] = None, max_shape: tuple = (None, ), chunks: bool = True, channel_metadata: Optional[object] = None, **kwargs) -> object

      Add a time series channel to the feature run group.

      Creates a new channel for time series data with the specified properties
      and optional metadata. Channel metadata should be a timeseries.Channel object.

      :param channel_name: Name for the channel.
      :type channel_name: str
      :param channel_type: Type of channel (e.g., 'electric', 'magnetic').
      :type channel_type: str
      :param data: Initial data for the channel. Default is None.
      :type data: np.ndarray, optional
      :param channel_dtype: Data type for the channel.
      :type channel_dtype: str, default='int32'
      :param shape: Shape of the channel data. Default is None.
      :type shape: tuple, optional
      :param max_shape: Maximum shape for expandable dimensions.
      :type max_shape: tuple, default=(None,)
      :param chunks: Whether to use chunking for the dataset.
      :type chunks: bool, default=True
      :param channel_metadata: Metadata object (timeseries.Channel type). Default is None.
      :type channel_metadata: optional
      :param \*\*kwargs: Additional keyword arguments for dataset creation.

      :returns: Channel object from RunGroup.
      :rtype: object

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> channel = ts_run.add_feature_channel(
      ...     'Ex', 'electric', data=np.arange(1000))



   .. py:method:: get_feature_channel(channel_name: str) -> object

      Retrieve a feature channel by name.

      :param channel_name: Name of the channel to retrieve.
      :type channel_name: str

      :returns: Channel object from RunGroup.
      :rtype: object

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> channel = ts_run.get_feature_channel('Ex')



   .. py:method:: remove_feature_channel(channel_name: str) -> None

      Remove a feature channel from the run group.

      :param channel_name: Name of the channel to remove.
      :type channel_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> ts_run.remove_feature_channel('Ex')



.. py:class:: FeatureFCRunGroup(group: h5py.Group, feature_run_metadata: Optional[mt_metadata.processing.fourier_coefficients.decimation.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for Fourier Coefficient features from a processing run.

   This class manages Fourier Coefficient data organized by decimation levels,
   each containing multiple frequency channels with time-frequency data.

   Hierarchy
   ---------
   FeatureFCRunGroup -> FeatureDecimationGroup -> FeatureChannelDataset

   .. attribute:: metadata

      Metadata including:

      - list of decimation levels
      - start time (earliest)
      - end time (latest)
      - method (fft, wavelet, ...)
      - list of channels used
      - starting sample rate
      - bands used
      - type (TS or FC)

      :type: Decimation

   :param group: HDF5 group object for this FeatureFCRunGroup.
   :type group: h5py.Group
   :param feature_run_metadata: Decimation metadata for the feature run. Default is None.
   :type feature_run_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> fc_run = FeatureFCRunGroup(h5_group, feature_run_metadata=metadata)
   >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)


   .. py:method:: metadata() -> mt_metadata.processing.fourier_coefficients.decimation.Decimation

      Overwrite get metadata to include channel information in the runs



   .. py:property:: decimation_level_summary
      :type: pandas.DataFrame


      Get a summary of all decimation levels in the run.

      Returns a pandas DataFrame with information about each decimation level
      including decimation factor, time range, and HDF5 reference.

      :returns: DataFrame with columns:

                - name : str
                    Decimation level name
                - start : datetime64[ns]
                    Start time of the decimation level
                - end : datetime64[ns]
                    End time of the decimation level
                - hdf5_reference : h5py.ref_dtype
                    HDF5 reference to the decimation level group
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> summary = fc_run.decimation_level_summary
      >>> print(summary[['name', 'start', 'end']])


   .. py:method:: add_decimation_level(decimation_level_name: str, feature_decimation_level_metadata: Optional[object] = None) -> FeatureDecimationGroup

      Add a decimation level group to the feature run.

      :param decimation_level_name: Name for the decimation level.
      :type decimation_level_name: str
      :param feature_decimation_level_metadata: Metadata for the decimation level. Default is None.
      :type feature_decimation_level_metadata: optional

      :returns: Newly created decimation level group.
      :rtype: FeatureDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)
      >>> print(decimation.name)
      'level_0'



   .. py:method:: get_decimation_level(decimation_level_name: str) -> FeatureDecimationGroup

      Retrieve a decimation level group by name.

      :param decimation_level_name: Name of the decimation level to retrieve.
      :type decimation_level_name: str

      :returns: The requested decimation level group.
      :rtype: FeatureDecimationGroup

      :raises MTH5Error: If the decimation level does not exist.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> decimation = fc_run.get_decimation_level('level_0')



   .. py:method:: remove_decimation_level(decimation_level_name: str) -> None

      Remove a decimation level from the feature run.

      :param decimation_level_name: Name of the decimation level to remove.
      :type decimation_level_name: str

      :raises MTH5Error: If the decimation level does not exist.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> fc_run.remove_decimation_level('level_0')



   .. py:method:: update_metadata() -> None

      Update metadata from all decimation levels.

      Scans all decimation levels and updates the run-level metadata with
      aggregated information including time ranges.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> fc_run.update_metadata()



.. py:class:: FeatureDecimationGroup(group: h5py.Group, decimation_level_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single decimation level with multiple Fourier Coefficient channels.

   This class manages Fourier Coefficient data organized by frequency, time, and channel.
   Data is assumed to be uniformly sampled in both frequency and time domains.

   Hierarchy
   ---------
   FeatureDecimationGroup -> FeatureChannelDataset (multiple channels)

   Data Assumptions
   ----------------
   1. Data are uniformly sampled in frequency domain
   2. Data are uniformly sampled in time domain
   3. FFT moving window has uniform step size

   .. attribute:: start time

      Start time of the decimation level

      :type: datetime

   .. attribute:: end time

      End time of the decimation level

      :type: datetime

   .. attribute:: channels

      List of channel names in this decimation level

      :type: list

   .. attribute:: decimation_factor

      Factor by which data was decimated

      :type: int

   .. attribute:: decimation_level

      Level index in decimation hierarchy

      :type: int

   .. attribute:: decimation_sample_rate

      Sample rate after decimation (Hz)

      :type: float

   .. attribute:: method

      Method used (FFT, wavelet, etc.)

      :type: str

   .. attribute:: anti_alias_filter

      Anti-aliasing filter used

      :type: optional

   .. attribute:: prewhitening_type

      Type of prewhitening applied

      :type: optional

   .. attribute:: harmonics_kept

      Harmonic indices kept in the data

      :type: list or 'all'

   .. attribute:: window

      Window parameters (length, overlap, type, sample rate)

      :type: dict

   .. attribute:: bands

      Frequency bands in the data

      :type: list

   :param group: HDF5 group object for this FeatureDecimationGroup.
   :type group: h5py.Group
   :param decimation_level_metadata: Metadata for the decimation level. Default is None.
   :type decimation_level_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> decimation = FeatureDecimationGroup(h5_group, metadata)
   >>> channel = decimation.add_channel('Ex', fc_data=fc_array, fc_metadata=ch_metadata)


   .. py:method:: metadata()

      Overwrite get metadata to include channel information in the runs



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get a summary of all channels in this decimation level.

      Returns a pandas DataFrame with detailed information about each Fourier
      Coefficient channel including time ranges, dimensions, and sampling rates.

      :returns: DataFrame with columns:

                - name : str
                    Channel name
                - start : datetime64[ns]
                    Start time of the channel data
                - end : datetime64[ns]
                    End time of the channel data
                - n_frequency : int64
                    Number of frequency bins
                - n_windows : int64
                    Number of time windows
                - sample_rate_decimation_level : float64
                    Decimation level sample rate (Hz)
                - sample_rate_window_step : float64
                    Sample rate of window stepping (Hz)
                - units : str
                    Physical units of the data
                - hdf5_reference : h5py.ref_dtype
                    HDF5 reference to the channel dataset
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> summary = decimation.channel_summary
      >>> print(summary[['name', 'n_frequency', 'n_windows']])


   .. py:method:: from_dataframe(df: pandas.DataFrame, channel_key: str, time_key: str = 'time', frequency_key: str = 'frequency') -> None

      Load Fourier Coefficient data from a pandas DataFrame.

      Assumes the channel_key column contains complex coefficient values
      organized with time and frequency dimensions.

      :param df: Input DataFrame containing the coefficient data.
      :type df: pd.DataFrame
      :param channel_key: Name of the column containing coefficient values.
      :type channel_key: str
      :param time_key: Name of the time coordinate column.
      :type time_key: str, default='time'
      :param frequency_key: Name of the frequency coordinate column.
      :type frequency_key: str, default='frequency'

      :raises TypeError: If df is not a pandas DataFrame.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.from_dataframe(df, channel_key='Ex', time_key='time')



   .. py:method:: from_xarray(data_array: xarray.DataArray | xarray.Dataset, sample_rate_decimation_level: float) -> None

      Load Fourier Coefficient data from an xarray DataArray or Dataset.

      Automatically extracts metadata (time, frequency, units) from the xarray
      object and creates appropriate FeatureChannelDataset instances for each
      variable or the single DataArray.

      :param data_array: Input xarray object with 'time' and 'frequency' coordinates and
                         dimensions ['time', 'frequency'] (or transposed variant).
      :type data_array: xr.DataArray or xr.Dataset
      :param sample_rate_decimation_level: Sample rate of the decimation level (Hz).
      :type sample_rate_decimation_level: float

      :raises TypeError: If data_array is not an xarray Dataset or DataArray.

      .. rubric:: Notes

      Automatically handles both (time, frequency) and (frequency, time) dimension ordering.
      Units are extracted from xarray attributes if available.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np
      >>> decimation = FeatureDecimationGroup(h5_group)

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex',
      ...     attrs={'units': 'mV/km'}
      ... )

      Load into decimation group:

      >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)



   .. py:method:: to_xarray(channels: Optional[list] = None) -> xarray.Dataset

      Create an xarray Dataset from Fourier Coefficient channels.

      If no channels are specified, all channels in the decimation level
      are included. Each channel becomes a data variable in the resulting Dataset.

      :param channels: List of channel names to include. If None, all channels are used.
                       Default is None.
      :type channels: list, optional

      :returns: xarray Dataset with channels as data variables and 'time' and
                'frequency' as shared coordinates.
      :rtype: xr.Dataset

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> xr_data = decimation.to_xarray()
      >>> print(xr_data.data_vars)
      Data variables:
          Ex  (time, frequency) complex128
          Ey  (time, frequency) complex128

      Get specific channels:

      >>> subset = decimation.to_xarray(channels=['Ex', 'Ey'])



   .. py:method:: from_numpy_array(nd_array: numpy.ndarray, ch_name: str | list) -> None

      Load Fourier Coefficient data from a numpy array.

      Assumes array shape is either (n_frequencies, n_windows) for a single
      channel or (n_channels, n_frequencies, n_windows) for multiple channels.

      :param nd_array: Input numpy array containing coefficient data.
      :type nd_array: np.ndarray
      :param ch_name: Channel name (for 2D array) or list of channel names
                      (for 3D array).
      :type ch_name: str or list

      :raises TypeError: If nd_array is not a numpy ndarray.
      :raises ValueError: If array shape is not (n_frequencies, n_windows) or
          (n_channels, n_frequencies, n_windows).

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)

      Load single channel:

      >>> data_2d = np.random.randn(256, 100) + 1j * np.random.randn(256, 100)
      >>> decimation.from_numpy_array(data_2d, ch_name='Ex')

      Load multiple channels:

      >>> data_3d = np.random.randn(2, 256, 100) + 1j * np.random.randn(2, 256, 100)
      >>> decimation.from_numpy_array(data_3d, ch_name=['Ex', 'Ey'])



   .. py:method:: add_channel(fc_name: str, fc_data: Optional[numpy.ndarray | xarray.DataArray | xarray.Dataset | pandas.DataFrame] = None, fc_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None, max_shape: tuple = (None, None), chunks: bool = True, dtype: type = complex, **kwargs) -> mth5.groups.FeatureChannelDataset

      Add a Fourier Coefficient channel to the decimation level.

      Creates a new FeatureChannelDataset for a single channel at a single
      decimation level. Input data can be provided as numpy array, xarray,
      DataFrame, or created empty.

      :param fc_name: Name for the Fourier Coefficient channel.
      :type fc_name: str
      :param fc_data: Input data. Can be numpy array (time, frequency) or xarray/DataFrame
                      format. Default is None (creates empty dataset).
      :type fc_data: np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, optional
      :param fc_metadata: Metadata for the channel. Default is None.
      :type fc_metadata: FeatureDecimationChannel, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param dtype: Data type for the dataset (e.g., complex, float, int).
      :type dtype: type, default=complex
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      :returns: Newly created FeatureChannelDataset object.
      :rtype: FeatureChannelDataset

      :raises TypeError: If fc_data type is not supported or metadata type mismatch.
      :raises RuntimeError or OSError: If channel already exists (will return existing channel).

      .. rubric:: Notes

      Data layout assumes (time, frequency) organization:

      - time index: window start times
      - frequency index: harmonic indices or float values
      - data: complex Fourier coefficients

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> metadata = FeatureDecimationChannel(name='Ex')

      Create from numpy array:

      >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

      Create empty channel (expandable):

      >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)



   .. py:method:: get_channel(fc_name: str) -> mth5.groups.FeatureChannelDataset

      Retrieve a Fourier Coefficient channel by name.

      :param fc_name: Name of the channel to retrieve.
      :type fc_name: str

      :returns: The requested FeatureChannelDataset object.
      :rtype: FeatureChannelDataset

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> channel = decimation.get_channel('Ex')
      >>> data = channel.to_numpy()



   .. py:method:: remove_channel(fc_name: str) -> None

      Remove a Fourier Coefficient channel from the decimation level.

      Deletes the channel from the HDF5 file. Note that this removes the
      reference but does not reduce file size.

      :param fc_name: Name of the channel to remove.
      :type fc_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Notes

      To reduce HDF5 file size, copy desired data to a new file.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.remove_channel('Ex')



   .. py:method:: update_metadata() -> None

      Update metadata from all channels in the decimation level.

      Scans all channels and updates the decimation-level metadata with
      aggregated information including time ranges and sampling rates.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.update_metadata()



   .. py:method:: add_weights(weight_name: str, weight_data: Optional[numpy.ndarray] = None, weight_metadata: Optional[object] = None, max_shape: tuple = (None, None, None), chunks: bool = True, **kwargs) -> None

      Add weight or masking data for Fourier Coefficients.

      Creates a dataset to store weights or masks for quality control,
      frequency band selection, or time window filtering.

      :param weight_name: Name for the weight dataset.
      :type weight_name: str
      :param weight_data: Weight values. Default is None.
      :type weight_data: np.ndarray, optional
      :param weight_metadata: Metadata for the weight dataset. Default is None.
      :type weight_metadata: optional
      :param max_shape: Maximum shape for expandable dimensions.
      :type max_shape: tuple, default=(None, None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      .. rubric:: Notes

      Weight datasets can track:

      - weight_channel: Per-channel weights
      - weight_band: Per-frequency-band weights
      - weight_time: Per-time-window weights

      This method is a placeholder for future implementation.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.add_weights('coherency_weights', weight_data=weights)



