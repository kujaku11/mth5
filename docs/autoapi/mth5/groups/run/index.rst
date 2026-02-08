mth5.groups.run
===============

.. py:module:: mth5.groups.run

.. autoapi-nested-parse::

   Created on Sat May 27 09:59:03 2023

   @author: jpeacock



Attributes
----------

.. autoapisummary::

   mth5.groups.run.meta_classes


Classes
-------

.. autoapisummary::

   mth5.groups.run.RunGroup


Module Contents
---------------

.. py:data:: meta_classes

.. py:class:: RunGroup(group: h5py.Group, run_metadata: Optional[mt_metadata.timeseries.Run] = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single MT measurement run with multiple channels.

   Manages time series data and metadata for one measurement run within a station.
   A run can contain multiple channels of electric, magnetic, and auxiliary data.
   This class provides methods to add, retrieve, and manage individual channels,
   along with convenient access to station and survey metadata.

   The run group is located at ``/Survey/Stations/{station_name}/{run_name}`` in
   the HDF5 file hierarchy.

   .. attribute:: metadata

      Run metadata including sample rate, time period, and channel information.

      :type: mt_metadata.timeseries.Run

   .. attribute:: channel_summary

      Summary table of all channels in the run.

      :type: pd.DataFrame

   .. attribute:: groups_list

      List of channel names in the run.

      :type: list[str]

   :param group: HDF5 group for the run, should have path like
                 ``/Survey/Stations/{station_name}/{run_name}``
   :type group: h5py.Group
   :param run_metadata: Metadata container for the run. Default is None.
   :type run_metadata: mt_metadata.timeseries.Run, optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.
   :type \*\*kwargs: Any

   .. rubric:: Notes

   Key behaviors:

   - Channels can be of type: electric, magnetic, or auxiliary
   - All metadata updates should use the metadata object for validation
   - Call write_metadata() after modifying metadata to persist changes
   - Channel metadata is cached for performance during repeated access
   - Deleting a channel removes the reference but doesn't reduce file size

   .. rubric:: Examples

   Access run from an open MTH5 file:

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
   >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

   Check available channels:

   >>> run.groups_list
   ['Ex', 'Ey', 'Hx', 'Hy']

   Access HDF5 group directly:

   >>> run.hdf5_group.ref
   <HDF5 Group Reference>

   Update metadata and persist to file:

   >>> run.metadata.sample_rate = 512.0
   >>> run.write_metadata()

   Add a channel:

   >>> import numpy as np
   >>> data = np.random.rand(4096)
   >>> ex = run.add_channel('Ex', 'electric', data=data)

   This class provides methods to add and get channels.  A summary table of
   all existing channels in the run is also provided as a convenience look up
   table to make searching easier.

   :param group: HDF5 group for a station, should have a path
                 ``/Survey/Stations/station_name/run_name``
   :type group: :class:`h5py.Group`
   :param station_metadata: metadata container, defaults to None
   :type station_metadata: :class:`mth5.metadata.Station`, optional

   :Access RunGroup from an open MTH5 file:

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
   >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

   :Check what channels exist:

   >>> station.groups_list
   ['Ex', 'Ey', 'Hx', 'Hy']

   To access the hdf5 group directly use `RunGroup.hdf5_group`

   >>> station.hdf5_group.ref
   <HDF5 Group Reference>

   .. note:: All attributes should be input into the metadata object, that
            way all input will be validated against the metadata standards.
            If you change attributes in metadata object, you should run the
            `SurveyGroup.write_metadata()` method.  This is a temporary
            solution, working on an automatic updater if metadata is changed.

   >>> run.metadata.existing_attribute = 'update_existing_attribute'
   >>> run.write_metadata()

   If you want to add a new attribute this should be done using the
   `metadata.add_base_attribute` method.

   >>> station.metadata.add_base_attribute('new_attribute',
   >>> ...                                 'new_attribute_value',
   >>> ...                                 {'type':str,
   >>> ...                                  'required':True,
   >>> ...                                  'style':'free form',
   >>> ...                                  'description': 'new attribute desc.',
   >>> ...                                  'units':None,
   >>> ...                                  'options':[],
   >>> ...                                  'alias':[],
   >>> ...                                  'example':'new attribute

   :Add a channel:

   >>> new_channel = run.add_channel('Ex', 'electric',
   >>> ...                            data=numpy.random.rand(4096))
   >>> new_run
   /Survey/Stations/MT001/MT001a:
   =======================================
       --> Dataset: summary
       ......................
       --> Dataset: Ex
       ......................
       --> Dataset: Ey
       ......................
       --> Dataset: Hx
       ......................
       --> Dataset: Hy
       ......................

   :Add a channel with metadata:

   >>> from mth5.metadata import Electric
   >>> ex_metadata = Electric()
   >>> ex_metadata.time_period.start = '2020-01-01T12:30:00'
   >>> ex_metadata.time_period.end = '2020-01-03T16:30:00'
   >>> new_ex = run.add_channel('Ex', 'electric',
   >>> ...                       channel_metadata=ex_metadata)
   >>> # to look at the metadata
   >>> new_ex.metadata
   {
        "electric": {
           "ac.end": 1.2,
           "ac.start": 2.3,
           ...
           }
   }


   .. seealso:: `mth5.metadata` for details on how to add metadata from
                various files and python objects.

   :Remove a channel:

   >>> run.remove_channel('Ex')
   >>> station
   /Survey/Stations/MT001/MT001a:
   =======================================
       --> Dataset: summary
       ......................
       --> Dataset: Ey
       ......................
       --> Dataset: Hx
       ......................
       --> Dataset: Hy
       ......................

   .. note:: Deleting a station is not as simple as del(station).  In HDF5
             this does not free up memory, it simply removes the reference
             to that station.  The common way to get around this is to
             copy what you want into a new file, or overwrite the station.

   :Get a channel:

   >>> existing_ex = stations.get_channel('Ex')
   >>> existing_ex
   Channel Electric:
   -------------------
       data type:        Ex
       data type:        electric
       data format:      float32
       data shape:       (4096,)
       start:            1980-01-01T00:00:00+00:00
       end:              1980-01-01T00:32:+08:00
       sample rate:      8


   :summary Table:

   A summary table is provided to make searching easier.  The table
   summarized all stations within a survey. To see what names are in the
   summary table:

   >>> run.summary_table.dtype.descr
   [('component', ('|S5', {'h5py_encoding': 'ascii'})),
    ('start', ('|S32', {'h5py_encoding': 'ascii'})),
    ('end', ('|S32', {'h5py_encoding': 'ascii'})),
    ('n_samples', '<i4'),
    ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
    ('units', ('|S25', {'h5py_encoding': 'ascii'})),
    ('hdf5_reference', ('|O', {'ref': h5py.h5r.Reference}))]


   .. note:: When a run is added an entry is added to the summary table,
             where the information is pulled from the metadata.

   >>> new_run.summary_table
   index | component | start | end | n_samples | measurement_type | units |
   hdf5_reference
   --------------------------------------------------------------------------
   -------------


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Get station metadata with current run included.

      :returns: Station metadata object containing this run's information.
      :rtype: metadata.Station

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> station_meta = run.station_metadata
      >>> print(station_meta.id)
      MT001


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Get survey metadata with current station and run included.

      :returns: Survey metadata object containing the full hierarchy.
      :rtype: metadata.Survey

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> survey_meta = run.survey_metadata
      >>> print(survey_meta.id)
      CONUS_South


   .. py:method:: recache_channel_metadata() -> None

      Clear and rebuild the channel metadata cache from current HDF5 data.

      This method reads all channel metadata from HDF5 storage and updates
      the internal cache. Useful when channel metadata has been modified
      externally or needs to be synchronized.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.recache_channel_metadata()
      >>> # Cache is now synchronized with HDF5 storage



   .. py:method:: metadata() -> mt_metadata.timeseries.Run

      Get run metadata including all channel information.

      This property dynamically reads and caches channel metadata from HDF5,
      ensuring the run metadata always reflects the current state of channels.

      :returns: Run metadata object with all channels included.
      :rtype: metadata.Run

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run_meta = run.metadata
      >>> print(run_meta.channels_recorded_electric)
      ['ex', 'ey']
      >>> print(run_meta.sample_rate)
      256.0



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get summary of all channels in the run as a DataFrame.

      :returns: DataFrame with columns: component, start, end, n_samples,
                sample_rate, measurement_type, units, hdf5_reference.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> summary = run.channel_summary
      >>> print(summary[['component', 'sample_rate', 'n_samples']])
        component  sample_rate  n_samples
      0        ex        256.0      65536
      1        ey        256.0      65536
      2        hx        256.0      65536
      3        hy        256.0      65536


   .. py:method:: write_metadata() -> None

      Write run metadata to HDF5 attributes.

      Converts metadata object to dictionary and writes all attributes
      to the HDF5 group.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.metadata.sample_rate = 512.0
      >>> run.write_metadata()
      >>> # Metadata is now persisted to HDF5 file



   .. py:method:: add_channel(channel_name, channel_type, data, channel_dtype='int32', shape=None, max_shape=(None, ), chunks=True, channel_metadata=None, **kwargs)

      Add a channel to the run.

      :param channel_name: Name of the channel (e.g., 'ex', 'ey', 'hx', 'hy', 'hz').
      :type channel_name: str
      :param channel_type: Type of channel: 'electric', 'magnetic', or 'auxiliary'.
      :type channel_type: str
      :param data: Time series data for the channel. If None, an empty resizable
                   dataset will be created.
      :type data: numpy.ndarray or None
      :param channel_dtype: Data type for the channel if data is None, by default "int32".
      :type channel_dtype: str, optional
      :param shape: Initial shape of the dataset. If None and data is None, shape
                    is estimated from metadata or set to (1,), by default None.
      :type shape: tuple of int, optional
      :param max_shape: Maximum shape the dataset can be resized to. Use None for
                        unlimited growth in that dimension, by default (None,).
      :type max_shape: tuple of int or None, optional
      :param chunks: Enable chunked storage. If True, uses automatic chunking.
                     If int, uses that chunk size, by default True.
      :type chunks: bool or int, optional
      :param channel_metadata: Metadata object for the channel, by default None.
      :type channel_metadata: mt_metadata.timeseries.Electric, Magnetic, or Auxiliary, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The created channel dataset object.
      :rtype: ElectricDataset or MagneticDataset or AuxiliaryDataset

      :raises MTH5Error: If channel_type is not one of: electric, magnetic, auxiliary.

      .. rubric:: Examples

      Add a channel with data:

      >>> import numpy as np
      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='a')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> data = np.random.rand(4096)
      >>> ex = run.add_channel('ex', 'electric', data)
      >>> print(ex.metadata.component)
      ex

      Add a channel with metadata:

      >>> from mt_metadata.timeseries import Electric
      >>> ex_meta = Electric()
      >>> ex_meta.time_period.start = '2020-01-01T12:30:00'
      >>> ex_meta.sample_rate = 256.0
      >>> ex = run.add_channel('ex', 'electric', None,
      ...                      channel_metadata=ex_meta)
      >>> print(ex.metadata.sample_rate)
      256.0

      Add a channel with custom shape:

      >>> ex = run.add_channel('ex', 'electric', None,
      ...                      shape=(8192,), channel_dtype='float32')
      >>> print(ex.hdf5_dataset.shape)
      (8192,)



   .. py:method:: get_channel(channel_name: str) -> mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset | mth5.groups.ChannelDataset

      Get a channel from an existing name.

      Returns the appropriate channel dataset container based on the
      channel type (electric, magnetic, or auxiliary).

      :param channel_name: Name of the channel to retrieve (e.g., 'ex', 'ey', 'hx').
      :type channel_name: str

      :returns: Channel dataset object containing the channel data and metadata.
      :rtype: ElectricDataset or MagneticDataset or AuxiliaryDataset or ChannelDataset

      :raises MTH5Error: If the channel does not exist in the run.

      .. rubric:: Examples

      Attempting to get a non-existent channel:

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> ex = run.get_channel('ex')
      MTH5Error: ex does not exist, check groups_list for existing names

      Check available channels first:

      >>> run.groups_list
      ['ey', 'hx', 'hz']

      Get an existing channel:

      >>> ey = run.get_channel('ey')
      >>> print(ey)
      Channel Electric:
      -------------------
              component:        ey
              data type:        electric
              data format:      float32
              data shape:       (4096,)
              start:            1980-01-01T00:00:00+00:00
              end:              1980-01-01T00:00:01+00:00
              sample rate:      4096



   .. py:method:: remove_channel(channel_name: str) -> None

      Remove a channel from the run.

      Deleting a channel is not as simple as del(channel). In HDF5,
      this does not free up memory; it simply removes the reference
      to that channel. The common way to get around this is to
      copy what you want into a new file, or overwrite the channel.

      :param channel_name: Name of the existing channel to remove.
      :type channel_name: str

      .. rubric:: Notes

      Deleting a channel does not reduce the HDF5 file size. It simply
      removes the reference. If file size reduction is your goal, copy
      what you want into another file.

      Todo: Need to remove summary table entry as well.

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
      >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
      >>> run.remove_channel('ex')



   .. py:method:: has_data() -> bool

      Check if the run contains any non-empty, non-zero data.

      Verifies that all channels in the run have valid data (non-zero and
      non-empty arrays). Returns False if any channel lacks data.

      :returns: True if all channels have data, False if any channel is empty
                or all zeros.
      :rtype: bool

      .. rubric:: Notes

      A channel is considered to have data if its has_data() method
      returns True, meaning it contains non-zero values.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> if run.has_data():
      ...     print("Run contains valid data")
      ...     runts = run.to_runts()



   .. py:method:: to_runts(start: Optional[str] = None, end: Optional[str] = None, n_samples: Optional[int] = None) -> mth5.timeseries.RunTS

      Convert run to a RunTS timeseries object.

      Combines all channels in the run into a RunTS object which handles
      multi-channel time series data with associated metadata.

      :param start: Start time for time slice in ISO format (e.g., '2023-01-01T12:00:00').
                    If None, uses entire channel data. Default is None.
      :type start: str, optional
      :param end: End time for time slice in ISO format. Only used if start is specified.
                  Default is None.
      :type end: str, optional
      :param n_samples: Number of samples to extract from start. If both end and n_samples
                        are specified, end takes precedence. Default is None.
      :type n_samples: int, optional

      :returns: RunTS object containing all channels with full run and station metadata.
      :rtype: RunTS

      .. rubric:: Notes

      - Includes run, station, and survey metadata in the output
      - Skips the 'summary' group which is not a channel
      - If start is specified, performs time slicing; otherwise returns full data

      .. rubric:: Examples

      Convert entire run to RunTS:

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> runts = run.to_runts()
      >>> print(runts.channels)
      ['ex', 'ey', 'hx', 'hy']

      Time slice the run:

      >>> runts = run.to_runts(start='2023-01-01T12:00:00',
      ...                       end='2023-01-01T13:00:00')
      >>> print(runts.ex.ts.shape)
      (1024,)



   .. py:method:: from_runts(run_ts_obj: mth5.timeseries.RunTS, **kwargs: Any) -> list[mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset]

      Create channel datasets from a RunTS timeseries object.

      Converts a RunTS object with multiple channels and metadata into
      HDF5 channel datasets and updates run metadata accordingly.

      :param run_ts_obj: RunTS object containing multiple channels and metadata.
      :type run_ts_obj: RunTS
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: Any

      :returns: List of created channel dataset objects.
      :rtype: list[ElectricDataset | MagneticDataset | AuxiliaryDataset]

      :raises MTH5Error: If input is not a RunTS object.

      .. rubric:: Notes

      - Updates run metadata from input object
      - Validates station and run IDs match current context
      - Creates appropriate channel type based on channel metadata
      - Automatically registers recorded channels in run metadata

      .. rubric:: Examples

      >>> from mth5.timeseries import RunTS
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> runts = RunTS.from_file("timeseries_data.txt")
      >>> channels = run.from_runts(runts)
      >>> print(f"Created {len(channels)} channels")
      Created 4 channels



   .. py:method:: from_channel_ts(channel_ts_obj: mth5.timeseries.ChannelTS) -> mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset

      Create a channel dataset from a ChannelTS timeseries object.

      Converts a single ChannelTS object with time series data and metadata
      into an HDF5 channel dataset. Handles filter registration and updates
      run metadata with channel information.

      :param channel_ts_obj: ChannelTS object containing time series data and metadata.
      :type channel_ts_obj: ChannelTS

      :returns: Created channel dataset object.
      :rtype: ElectricDataset | MagneticDataset | AuxiliaryDataset

      :raises MTH5Error: If input is not a ChannelTS object.

      .. rubric:: Notes

      - Registers filters from channel response if present
      - Validates and corrects station/run ID mismatches
      - Updates run metadata recorded channel lists
      - Automatically determines channel type from metadata

      .. rubric:: Examples

      >>> from mth5.timeseries import ChannelTS
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> channel = ChannelTS.from_file("ex_timeseries.txt")
      >>> ex = run.from_channel_ts(channel)
      >>> print(ex.metadata.component)
      ex



   .. py:method:: update_run_metadata() -> None

      Update metadata and table entries (Deprecated).
      .. deprecated::
          Use update_metadata() instead.
      :raises DeprecationWarning: Always raised to indicate this method should not be used.



   .. py:method:: update_metadata() -> None

      Update run metadata from all channels and persist to HDF5.

      Aggregates metadata from all channels including time period and
      sample rate, then writes updated metadata to HDF5 attributes.

      :raises Exception: May raise exceptions if no channels exist (logs warning).

      .. rubric:: Notes

      Updates:

      - Time period start from minimum of all channels
      - Time period end from maximum of all channels
      - Sample rate from first channel (assumes uniform across channels)

      Should be called after adding or removing channels to maintain
      consistency between channel and run metadata.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.add_channel('ex', 'electric', data=ex_data)
      >>> run.add_channel('ey', 'electric', data=ey_data)
      >>> run.update_metadata()  # Updates time period and sample rate



   .. py:method:: plot(start: Optional[str] = None, end: Optional[str] = None, n_samples: Optional[int] = None) -> Any

      Create a matplotlib plot of all channels in the run.

      Generates a multi-panel plot showing all channels in the run using
      the RunTS plotting functionality.

      :param start: Start time for time slice in ISO format. If None, plots entire
                    channel data. Default is None.
      :type start: str, optional
      :param end: End time for time slice in ISO format. Only used if start is
                  specified. Default is None.
      :type end: str, optional
      :param n_samples: Number of samples to extract from start. If both end and n_samples
                        are specified, end takes precedence. Default is None.
      :type n_samples: int, optional

      :returns: Matplotlib figure or axes object (depends on RunTS.plot() implementation).
      :rtype: Any

      .. rubric:: Notes

      - Creates separate subplots for each channel type (electric, magnetic, auxiliary)
      - Time slice parameters work the same as to_runts()
      - Requires matplotlib to be installed

      .. rubric:: Examples

      Plot entire run:

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> fig = run.plot()
      >>> fig.show()

      Plot time slice:

      >>> fig = run.plot(start='2023-01-01T12:00:00',
      ...                end='2023-01-01T13:00:00')



