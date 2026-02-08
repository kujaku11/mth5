mth5.groups.fourier_coefficients
================================

.. py:module:: mth5.groups.fourier_coefficients

.. autoapi-nested-parse::

   Fourier Coefficient group management for MTH5 format.

   This module provides classes for organizing and managing Fourier Coefficient
   data at multiple decimation levels, including utilities for data import/export
   with different formats (numpy, xarray, pandas).

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)



Classes
-------

.. autoapisummary::

   mth5.groups.fourier_coefficients.MasterFCGroup
   mth5.groups.fourier_coefficients.FCDecimationGroup
   mth5.groups.fourier_coefficients.FCGroup


Module Contents
---------------

.. py:class:: MasterFCGroup(group: h5py.Group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Master container for all Fourier Coefficient estimations of time series data.

   This class manages multiple Fourier Coefficient processing runs, each containing
   different decimation levels. No metadata is required at the master level.

   Hierarchy
   ---------
   MasterFCGroup -> FCGroup (processing runs) -> FCDecimationGroup (decimation levels)
   -> FCChannelDataset (individual channels)

   :param group: HDF5 group object for the master FC container.
   :type group: h5py.Group
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> import h5py
   >>> from mth5.groups.fourier_coefficients import MasterFCGroup
   >>> with h5py.File('data.h5', 'r') as f:
   ...     master = MasterFCGroup(f['FC'])
   ...     fc_group = master.add_fc_group('processing_run_1')


   .. py:property:: fc_summary
      :type: pandas.DataFrame


      Get a summary of all Fourier Coefficient processing runs.

      :returns: Summary information for all FC groups including names and metadata.
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> summary = master.fc_summary


   .. py:method:: add_fc_group(fc_name: str, fc_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None) -> FCGroup

      Add a Fourier Coefficient processing run group.

      :param fc_name: Name for the FC group (usually identifies the processing run).
      :type fc_name: str
      :param fc_metadata: Metadata for the FC group. Default is None.
      :type fc_metadata: fc.Decimation, optional

      :returns: Newly created Fourier Coefficient group.
      :rtype: FCGroup

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> fc_group = master.add_fc_group('processing_run_1')
      >>> print(fc_group.name)
      'processing_run_1'



   .. py:method:: get_fc_group(fc_name: str) -> FCGroup

      Retrieve a Fourier Coefficient group by name.

      :param fc_name: Name of the FC group to retrieve.
      :type fc_name: str

      :returns: The requested Fourier Coefficient group.
      :rtype: FCGroup

      :raises MTH5Error: If the FC group does not exist.

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> fc_group = master.get_fc_group('processing_run_1')



   .. py:method:: remove_fc_group(fc_name: str) -> None

      Remove a Fourier Coefficient group.

      Deletes the specified FC group and all associated decimation levels and channels.

      :param fc_name: Name of the FC group to remove.
      :type fc_name: str

      :raises MTH5Error: If the FC group does not exist.

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> master.remove_fc_group('processing_run_1')



.. py:class:: FCDecimationGroup(group: h5py.Group, decimation_level_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single decimation level of Fourier Coefficient data.

   This class manages all channels at a specific decimation level, assuming
   uniform sampling in both frequency and time domains.

   Data Assumptions
   ----------------
   1. Data uniformly sampled in frequency domain
   2. Data uniformly sampled in time domain
   3. FFT moving window has uniform step size

   .. attribute:: start_time

      Start time of the decimation level

      :type: datetime

   .. attribute:: end_time

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

   .. attribute:: sample_rate

      Sample rate after decimation (Hz)

      :type: float

   .. attribute:: method

      Method used (FFT, wavelet, etc.)

      :type: str

   .. attribute:: window

      Window parameters (length, overlap, type, sample rate)

      :type: dict

   :param group: HDF5 group object for this decimation level.
   :type group: h5py.Group
   :param decimation_level_metadata: Metadata for the decimation level. Default is None.
   :type decimation_level_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> decimation = FCDecimationGroup(h5_group, decimation_level_metadata=metadata)
   >>> channel = decimation.add_channel('Ex', fc_data=fc_array)


   .. py:method:: metadata()

      Overwrite get metadata to include channel information in the runs



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get a summary of all channels in this decimation level.

      Returns a pandas DataFrame with detailed information about each Fourier
      Coefficient channel including time ranges, dimensions, and sampling rates.

      :returns: DataFrame with columns:

                - component : str
                    Channel component name (e.g., 'Ex', 'Hy')
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

      >>> decimation = FCDecimationGroup(h5_group)
      >>> summary = decimation.channel_summary
      >>> print(summary[['component', 'n_frequency', 'n_windows']])


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

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.from_dataframe(df, channel_key='Ex', time_key='time')



   .. py:method:: from_xarray(data_array: xarray.Dataset | xarray.DataArray, sample_rate_decimation_level: float) -> None

      Load Fourier Coefficient data from an xarray DataArray or Dataset.

      Automatically extracts metadata (time, frequency, units) from the xarray
      object and creates appropriate FCChannelDataset instances for each
      variable or the single DataArray.

      :param data_array: Input xarray object with 'time' and 'frequency' coordinates and
                         dimensions ['time', 'frequency'] (or transposed variant).
      :type data_array: xr.DataArray or xr.Dataset
      :param sample_rate_decimation_level: Sample rate of the decimation level (Hz).
      :type sample_rate_decimation_level: float

      :raises TypeError: If data_array is not an xarray Dataset or DataArray.

      .. rubric:: Notes

      Automatically handles both (time, frequency) and (frequency, time)
      dimension ordering. Units are extracted from xarray attributes if available.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np
      >>> decimation = FCDecimationGroup(h5_group)

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex'
      ... )

      Load into decimation group:

      >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)



   .. py:method:: to_xarray(channels: Optional[list[str]] = None) -> xarray.Dataset

      Create an xarray Dataset from Fourier Coefficient channels.

      If no channels are specified, all channels in the decimation level
      are included. Each channel becomes a data variable in the resulting Dataset.

      :param channels: List of channel names to include. If None, all channels are used.
                       Default is None.
      :type channels: list[str], optional

      :returns: xarray Dataset with channels as data variables and 'time' and
                'frequency' as shared coordinates.
      :rtype: xr.Dataset

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> xr_data = decimation.to_xarray()
      >>> print(xr_data.data_vars)
      Data variables:
          Ex  (time, frequency) complex128
          Ey  (time, frequency) complex128

      Get specific channels:

      >>> subset = decimation.to_xarray(channels=['Ex', 'Ey'])



   .. py:method:: from_numpy_array(nd_array: numpy.ndarray, ch_name: str | list[str]) -> None

      Load Fourier Coefficient data from a numpy array.

      Assumes array shape is either (n_frequencies, n_windows) for a single
      channel or (n_channels, n_frequencies, n_windows) for multiple channels.

      :param nd_array: Input numpy array containing coefficient data.
      :type nd_array: np.ndarray
      :param ch_name: Channel name (for 2D array) or list of channel names
                      (for 3D array).
      :type ch_name: str or list[str]

      :raises TypeError: If nd_array is not a numpy ndarray.
      :raises ValueError: If array shape is not (n_frequencies, n_windows) or
          (n_channels, n_frequencies, n_windows).

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)

      Load single channel:

      >>> data_2d = np.random.randn(256, 100) + 1j * np.random.randn(256, 100)
      >>> decimation.from_numpy_array(data_2d, ch_name='Ex')

      Load multiple channels:

      >>> data_3d = np.random.randn(2, 256, 100) + 1j * np.random.randn(2, 256, 100)
      >>> decimation.from_numpy_array(data_3d, ch_name=['Ex', 'Ey'])



   .. py:method:: add_channel(fc_name: str, fc_data: Optional[numpy.ndarray] = None, fc_metadata: Optional[mt_metadata.processing.fourier_coefficients.FCChannel] = None, max_shape: tuple = (None, None), chunks: bool = True, dtype: type = complex, **kwargs) -> mth5.groups.FCChannelDataset

      Add a Fourier Coefficient channel to the decimation level.

      Creates a new FCChannelDataset for a single channel at a single
      decimation level. Input data can be provided as numpy array or created empty.

      :param fc_name: Name for the Fourier Coefficient channel (usually component name like 'Ex').
      :type fc_name: str
      :param fc_data: Input data with shape (n_frequencies, n_windows). Default is None (creates empty).
      :type fc_data: np.ndarray, optional
      :param fc_metadata: Metadata for the channel. Default is None.
      :type fc_metadata: fc.FCChannel, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param dtype: Data type for the dataset.
      :type dtype: type, default=complex
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      :returns: Newly created FCChannelDataset object.
      :rtype: FCChannelDataset

      :raises TypeError: If fc_data type is not supported.

      .. rubric:: Notes

      Data layout assumes (time, frequency) organization:

      - time index: window start times
      - frequency index: harmonic indices or float values
      - data: complex Fourier coefficients

      If a channel with the same name already exists, the existing channel
      is returned instead of creating a duplicate.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> metadata = fc.FCChannel(component='Ex')

      Create from numpy array:

      >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

      Create empty channel (expandable):

      >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)



   .. py:method:: get_channel(fc_name: str) -> mth5.groups.FCChannelDataset

      Retrieve a Fourier Coefficient channel by name.

      :param fc_name: Name of the Fourier Coefficient channel to retrieve.
      :type fc_name: str

      :returns: The requested Fourier Coefficient channel dataset.
      :rtype: FCChannelDataset

      :raises KeyError: If the channel does not exist in this decimation level.
      :raises MTH5Error: If unable to retrieve the channel from HDF5.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> channel = decimation.get_channel('Ex')
      >>> print(channel.shape)
      (100, 256)



   .. py:method:: remove_channel(fc_name: str) -> None

      Remove a Fourier Coefficient channel from the decimation level.

      Deletes the HDF5 dataset associated with the channel. Note that this
      removes the reference but does not reduce the HDF5 file size.

      :param fc_name: Name of the Fourier Coefficient channel to remove.
      :type fc_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Notes

      Deleting a channel does not reduce the HDF5 file size; it simply
      removes the reference to the data. To truly reduce file size, copy
      the desired data to a new file.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.remove_channel('Ex')



   .. py:method:: update_metadata() -> None

      Update decimation level metadata from all channels.

      Aggregates metadata from all FC channels in the decimation level
      including time period, sample rates, and window step information.
      Updates the internal metadata object and writes to HDF5.

      .. rubric:: Notes

      Collects the following information from channels:

      - Time period start/end from channel data
      - Sample rate decimation level
      - Sample rate window step

      Should be called after adding or modifying channels to keep
      metadata synchronized.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.add_channel('Ex', fc_data=data_ex)
      >>> decimation.add_channel('Ey', fc_data=data_ey)
      >>> decimation.update_metadata()



   .. py:method:: add_feature(feature_name: str, feature_data: Optional[numpy.ndarray] = None, feature_metadata: Optional[dict] = None, max_shape: tuple = (None, None, None), chunks: bool = True, **kwargs) -> None

      Add a feature dataset to the decimation level.

      Creates a new dataset for auxiliary features or derived quantities
      related to Fourier Coefficients (e.g., SNR, coherency, power, etc.).

      :param feature_name: Name for the feature dataset.
      :type feature_name: str
      :param feature_data: Input data for the feature. Default is None (creates empty).
      :type feature_data: np.ndarray, optional
      :param feature_metadata: Metadata dictionary for the feature. Default is None.
      :type feature_metadata: dict, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      .. rubric:: Notes

      Feature types may include:

      - Power: Total power in Fourier coefficients
      - SNR: Signal-to-noise ratio
      - Coherency: Cross-component coherence
      - Weights: Channel-specific weights
      - Flags: Data quality or processing flags

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> snr_data = np.random.randn(100, 256)
      >>> decimation.add_feature('snr', feature_data=snr_data)

      Or create empty feature for later population:

      >>> decimation.add_feature('power_Ex')



.. py:class:: FCGroup(group: h5py.Group, decimation_level_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Manage a set of Fourier Coefficients from a single processing run.

   Holds Fourier Coefficient estimations organized by decimation level.
   Each decimation level contains channels (Ex, Ey, Hz, etc.) with complex
   frequency or time-frequency representations of the input signal.

   All channels must use the same calibration. Recalibration requires
   rerunning the Fourier Coefficient estimation.

   .. attribute:: hdf5_group

      The HDF5 group containing decimation levels

      :type: h5py.Group

   .. attribute:: metadata

      Decimation metadata including time period, sample rates, and channels

      :type: fc.Decimation

   .. rubric:: Notes

   Processing run structure:

   - Multiple decimation levels at different sample rates
   - Each decimation level contains multiple channels
   - Each channel contains complex Fourier coefficients
   - Time period and sample rates define the estimation window

   .. rubric:: Examples

   >>> with h5py.File('data.h5', 'r') as f:
   ...     fc_run = FCGroup(f['Fourier_Coefficients/run_1'])
   ...     print(fc_run.decimation_level_summary)


   .. py:method:: metadata() -> mt_metadata.processing.fourier_coefficients.Decimation

      Get processing run metadata including all decimation levels.

      Collects metadata from all decimation level groups and aggregates
      into a single Decimation metadata object.

      :returns: Metadata containing time period, sample rates, and all decimation
                level information.
      :rtype: fc.Decimation

      .. rubric:: Notes

      This getter automatically populates:

      - Time period (start and end)
      - List of all decimation levels and their metadata
      - HDF5 reference to this group

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> metadata = fc_run.metadata
      >>> print(metadata.time_period.start)
      2023-01-01T00:00:00



   .. py:property:: decimation_level_summary
      :type: pandas.DataFrame


      Get a summary of all decimation levels in this processing run.

      Returns information about each decimation level including sample rate,
      decimation level value, and time span.

      :returns: Summary with columns:

                - decimation_level: Integer decimation level identifier
                - start: ISO format start time of this decimation level
                - end: ISO format end time of this decimation level
                - hdf5_reference: Reference to the HDF5 group
      :rtype: pd.DataFrame

      .. rubric:: Notes

      Each row represents a single decimation level containing multiple
      channels with Fourier coefficients at different sample rates.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> summary = fc_run.decimation_level_summary
      >>> print(summary[['decimation_level', 'start', 'end']])
         decimation_level                start                  end
      0              0     2023-01-01T00:00:00.000000  2023-01-01T01:00:00.000000
      1              1     2023-01-01T00:00:00.000000  2023-01-01T02:00:00.000000


   .. py:method:: add_decimation_level(decimation_level_name: str, decimation_level_metadata: Optional[dict | mt_metadata.processing.fourier_coefficients.Decimation] = None) -> FCDecimationGroup

      Add a new decimation level to the processing run.

      Creates a new FCDecimationGroup for a single decimation level containing
      Fourier Coefficient channels at a specific sample rate.

      :param decimation_level_name: Identifier for the decimation level.
      :type decimation_level_name: str
      :param decimation_level_metadata: Metadata for the decimation level. Can be a dictionary or
                                        fc.Decimation object. Default is None.
      :type decimation_level_metadata: dict | fc.Decimation, optional

      :returns: Newly created decimation level group.
      :rtype: FCDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> metadata = fc.Decimation(decimation_level=0)
      >>> decimation = fc_run.add_decimation_level('0', metadata)



   .. py:method:: get_decimation_level(decimation_level_name: str) -> FCDecimationGroup

      Retrieve a decimation level by name.

      :param decimation_level_name: Name or identifier of the decimation level.
      :type decimation_level_name: str

      :returns: The requested decimation level group.
      :rtype: FCDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> decimation = fc_run.get_decimation_level('0')
      >>> channels = decimation.groups_list



   .. py:method:: remove_decimation_level(decimation_level_name: str) -> None

      Remove a decimation level from the processing run.

      Deletes the HDF5 group and all its channels (FCChannelDataset objects).

      :param decimation_level_name: Name or identifier of the decimation level to remove.
      :type decimation_level_name: str

      .. rubric:: Notes

      This removes the entire decimation level and all channels within it.
      To remove individual channels, use FCDecimationGroup.remove_channel()
      instead.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> fc_run.remove_decimation_level('0')



   .. py:method:: update_metadata() -> None

      Update processing run metadata from all decimation levels.

      Aggregates time period information from all decimation levels
      and writes updated metadata to HDF5.

      .. rubric:: Notes

      Collects:

      - Earliest start time across all decimation levels
      - Latest end time across all decimation levels

      Should be called after adding or removing decimation levels.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> fc_run.add_decimation_level('0', metadata0)
      >>> fc_run.add_decimation_level('1', metadata1)
      >>> fc_run.update_metadata()



   .. py:method:: supports_aurora_processing_config(processing_config: aurora.config.metadata.processing.Processing, remote: bool) -> bool

      Check if all required decimation levels exist for Aurora processing.

      Performs an all-or-nothing check: returns True only if every decimation
      level required by the processing config is available in this FCGroup.

      Uses sequential logic to short-circuit: if any required decimation level
      is missing, immediately returns False without checking remaining levels.

      :param processing_config: Aurora processing configuration containing required decimation levels.
      :type processing_config: aurora.config.metadata.processing.Processing
      :param remote: Whether to check for remote processing compatibility.
      :type remote: bool

      :returns: True if all required decimation levels are available and consistent,
                False otherwise.
      :rtype: bool

      .. rubric:: Notes

      Validation logic:

      1. Extract list of decimation levels from processing config
      2. Iterate through each required level in sequence
      3. For each level, find a matching FCDecimation in this group
      4. Check consistency using Aurora's validation method
      5. If any level is missing or inconsistent, return False immediately
      6. Return True only if all levels pass validation

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> config = aurora.config.metadata.processing.Processing(...)
      >>> if fc_run.supports_aurora_processing_config(config, remote=False):
      ...     # All decimation levels are available
      ...     pass



