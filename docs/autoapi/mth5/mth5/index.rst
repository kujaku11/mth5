mth5.mth5
=========

.. py:module:: mth5.mth5

.. autoapi-nested-parse::

   ==================
   MTH5
   ==================

   MTH5 deals with reading and writing an MTH5 file, which are HDF5 files
   developed for magnetotelluric (MT) data.  The code is based on h5py and
   numpy. The main purpose is to provide an object-oriented interface for
   managing MT data in the HDF5 format.

   This module implements the MTH5 class which provides a container for the
   hierarchical structure of MT data collection:

   - Version 0.1.0: Survey → Stations → Runs → Channels
   - Version 0.2.0: Experiment → Surveys → Stations → Runs → Channels

   All timeseries data are stored as individual channels with appropriate
   metadata for electric, magnetic, and auxiliary data.

   Created on Sun Dec  9 20:50:41 2018

   :copyright: Jared Peacock (jpeacock@usgs.gov)

   :license: MIT

   .. rubric:: Notes

   For detailed information about the MTH5 format and metadata standards,
   see https://github.com/kujaku11/MTarchive/

   .. rubric:: Examples

   Create a new MTH5 file and add a station:

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5(file_version='0.2.0')
   >>> mth5_obj.open_mth5('test.mth5', 'w')
   >>> survey = mth5_obj.add_survey('survey_001')
   >>> station = mth5_obj.add_station('MT001', survey='survey_001')

   .. seealso::

      :obj:`h5py`
          HDF5 library used for file I/O

      :obj:`mt_metadata`
          Metadata standards for MT data



Classes
-------

.. autoapisummary::

   mth5.mth5.MTH5


Module Contents
---------------

.. py:class:: MTH5(filename=None, compression='gzip', compression_opts=4, shuffle=True, fletcher32=True, data_level=1, file_version='0.2.0')

   MTH5 is the main container for the HDF5 file format developed for MT data

   It uses the metadata standards developled by the
   `IRIS PASSCAL software group
   <https://www.iris.edu/hq/about_iris/governance/mt_soft>`_
   and defined in the
   `metadata documentation
   <https://github.com/kujaku11/MTarchive/blob/tables/docs/mt_metadata_guide.pdf>`_.

   MTH5 is built with h5py and therefore numpy.  The structure follows the
   different levels of MT data collection:

   For version 0.1.0:

       - Survey

          - Reports
          - Standards
          - Filters
          - Stations

              - Run

                  - Channel

   For version 0.2.0:

       - Experiment

           - Reports
           - Standards
           - Surveys

              - Reports
              - Standards
              - Filters
              - Stations

                  - Run

                      -Channel


   All timeseries data are stored as individual channels with the appropriate
   metadata defined for the given channel, i.e. electric, magnetic, auxiliary.

   Each level is represented as a mth5 group class object which has methods
   to add, remove, and get a group from the level below.  Each group has a
   metadata attribute that is the approprate metadata class object.  For
   instance the SurveyGroup has an attribute metadata that is a
   :class:`mth5.metadata.Survey` object.  Metadata is stored in the HDF5 group
   attributes as (key, value) pairs.

   All groups are represented by their structure tree and can be shown
   at any time from the command line.

   Each level has a summary array of the contents of the levels below to
   hopefully make searching easier.

   :param filename: name of the to be or existing file
   :type filename: string or :class:`pathlib.Path`
   :param compression: compression type.  Supported lossless compressions are

       * 'lzf' - Available with every installation of h5py
                (C source code also available). Low to
                moderate compression, very fast. No options.
       * 'gzip' - Available with every installation of HDF5,
                 so it’s best where portability is required.
                 Good compression, moderate speed.
                 compression_opts sets the compression level
                 and may be an integer from 0 to 9,
                 default is 3.
       * 'szip' - Patent-encumbered filter used in the NASA
                  community. Not available with all
                  installations of HDF5 due to legal reasons.
                  Consult the HDF5 docs for filter options.

   :param compression_opts: compression options, see above
   :type compression_opts: string or int depending on compression type
   :param shuffle: Block-oriented compressors like GZIP or LZF work better
                   when presented with runs of similar values. Enabling the
                   shuffle filter rearranges the bytes in the chunk and may
                   improve compression ratio. No significant speed penalty,
                   lossless.
   :type shuffle: boolean
   :param fletcher32: Adds a checksum to each chunk to detect data corruption.
                      Attempts to read corrupted chunks will fail with an
                      error. No significant speed penalty. Obviously
                      shouldn’t be used with lossy compression filters.
   :type fletcher32: boolean
   :param data_level: level the data are stored following levels defined by
      `NASA ESDS <https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels>`_

        * 0 - Raw data
        * 1 - Raw data with response information and full metadata
        * 2 - Derived product, raw data has been manipulated

   :type data_level: integer, defaults to 1
   :param file_version: Version of the file [ '0.1.0' | '0.2.0' ], defaults to "0.2.0"
   :type file_version: string, optional

   :Usage:

   * Open a new file and show initialized file

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5(file_version='0.1.0')
   >>> # Have a look at the dataset options
   >>> mth5.dataset_options
   {'compression': 'gzip',
    'compression_opts': 3,
    'shuffle': True,
    'fletcher32': True}
   >>> mth5_obj.open_mth5(r"/home/mtdata/mt01.mth5", 'w')
   >>> mth5_obj
   /:
   ====================
       |- Group: Survey
       ----------------
           |- Group: Filters
           -----------------
               --> Dataset: summary
               ......................
           |- Group: Reports
           -----------------
               --> Dataset: summary
               ......................
           |- Group: Standards
           -------------------
               --> Dataset: summary
               ......................
           |- Group: Stations
           ------------------
               --> Dataset: summary
               ......................


   * Add metadata for survey from a dictionary

   >>> survey_dict = {'survey':{'acquired_by': 'me', 'archive_id': 'MTCND'}}
   >>> survey = mth5_obj.survey_group
   >>> survey.metadata.from_dict(survey_dict)
   >>> survey.metadata
   {
   "survey": {
       "acquired_by.author": "me",
       "acquired_by.comments": null,
       "archive_id": "MTCND"
       ...}
   }

   * Add a station from the convenience function

   >>> station = mth5_obj.add_station('MT001')
   >>> mth5_obj
   /:
   ====================
       |- Group: Survey
       ----------------
           |- Group: Filters
           -----------------
               --> Dataset: summary
               ......................
           |- Group: Reports
           -----------------
               --> Dataset: summary
               ......................
           |- Group: Standards
           -------------------
               --> Dataset: summary
               ......................
           |- Group: Stations
           ------------------
               |- Group: MT001
               ---------------
                   --> Dataset: summary
                   ......................
               --> Dataset: summary
               ......................
   >>> station
   /Survey/Stations/MT001:
   ====================
       --> Dataset: summary
       ......................

   >>> data.schedule_01.ex[0:10] = np.nan
   >>> data.calibration_hx[...] = np.logspace(-4, 4, 20)

   .. note:: if replacing an entire array with a new one you need to use [...]
             otherwise the data will not be updated.

   .. warning:: You can only replace entire arrays with arrays of the same
                size.  Otherwise you need to delete the existing data and
                make a new dataset.

   .. seealso:: https://www.hdfgroup.org/ and https://www.h5py.org/



   .. py:attribute:: logger


   .. py:property:: data_level

      data level


   .. py:property:: filename

      file name of the hdf5 file


   .. py:property:: file_version

      mth5 file version


   .. py:property:: file_type

      File Type should be MTH5


   .. py:property:: dataset_options
      :type: dict[str, str | int | bool]


      Get HDF5 dataset compression and storage options.

      :returns: Dictionary containing compression, compression_opts, shuffle, and fletcher32.
      :rtype: dict[str, str | int | bool]

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> opts = mth5_obj.dataset_options
      >>> print(opts['compression'])
      'gzip'


   .. py:property:: file_attributes


   .. py:property:: software_name

      software name that wrote the file


   .. py:property:: experiment_group

      Convenience property for /Experiment group


   .. py:property:: survey_group

      Convenience property for /Survey group


   .. py:property:: surveys_group

      Convenience property for /Surveys group


   .. py:property:: reports_group

      Convenience property for /Survey/Reports group


   .. py:property:: filters_group

      Convenience property for /Survey/Filters group


   .. py:property:: standards_group

      Convenience property for /Standards group


   .. py:property:: stations_group

      Convenience property for /Survey/Stations group


   .. py:property:: station_list

      list of existing stations names


   .. py:method:: open_mth5(filename: str | pathlib.Path | None = None, mode: str = 'a', **kwargs) -> MTH5

      Open an MTH5 file.

      Opens an existing MTH5 file or creates a new one. Validates file structure
      and initializes summary datasets if needed.

      :param filename: Path to MTH5 file. If None, uses stored filename.
      :type filename: str | Path, optional
      :param mode: File opening mode:

                   * 'r' : Read-only
                   * 'a' : Read/write, create if doesn't exist
                   * 'w' : Write, overwrite if exists
                   * 'x' : Write, fail if exists
                   * 'w-' : Write, fail if exists (same as 'x')
                   * 'r+' : Read/write, file must exist
      :type mode: str, default 'a'
      :param \*\*kwargs: Additional arguments passed to h5py.File()

      :returns: Returns self for method chaining.
      :rtype: MTH5

      :raises MTH5Error: If file is invalid or mode is not understood.

      .. rubric:: Examples

      Open an existing file for reading:

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('data.mth5', 'r')

      Create a new file:

      >>> mth5_obj = MTH5(file_version='0.2.0')
      >>> mth5_obj.open_mth5('new_file.mth5', 'w')

      .. seealso::

         :obj:`close_mth5`
             Close the MTH5 file



   .. py:method:: validate_file() -> bool

      Validate an open MTH5 file.

      Checks file attributes, version, data level, and group structure
      for compliance with MTH5 format specifications.

      :returns: True if file is valid, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'r')
      >>> is_valid = mth5_obj.validate_file()



   .. py:method:: close_mth5() -> None

      Close MTH5 file.

      Flushes all data to disk, updates summary tables, and closes the file.
      Safe to call on already-closed files.

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'w')
      >>> mth5_obj.close_mth5()

      .. rubric:: Notes

      Can be called automatically using context manager:

      >>> with MTH5().open_mth5('test.mth5', 'w') as m:
      ...     # do work
      ...     pass  # file closed automatically



   .. py:method:: h5_is_write() -> bool

      Check if HDF5 file is open in write mode.

      :returns: True if file is open and writable, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'w')
      >>> mth5_obj.h5_is_write()
      True



   .. py:method:: h5_is_read() -> bool

      Check if HDF5 file is open and readable.

      :returns: True if file is open and readable, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'r')
      >>> mth5_obj.h5_is_read()
      True



   .. py:method:: has_group(group_name)

      Check to see if the group name exists



   .. py:method:: get_reference_path(h5_reference)

      Get the HDF5 path from a reference

      :param h5_reference: DESCRIPTION
      :type h5_reference: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: from_reference(h5_reference)

      Get an HDF5 group, dataset, etc from a reference

      :param h5_reference: DESCRIPTION
      :type h5_reference: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: to_experiment(has_data=True)

      Create an :class:`mt_metadata.timeseries.Experiment` object from the
      metadata contained in the MTH5 file.

      :returns: :class:`mt_metadata.timeseries.Experiment`




   .. py:method:: from_experiment(experiment, survey_index=0, update=False)

      Fill out an MTH5 from a :class:`mt_metadata.timeseries.Experiment` object
      given a survey_id

      :param experiment: Experiment metadata
      :type experiment: :class:`mt_metadata.timeseries.Experiment`
      :param survey_index: Index of the survey to write
      :type survey_index: int, defaults to 0




   .. py:property:: channel_summary
      :type: mth5.tables.ChannelSummaryTable


      Get channel summary table.

      :returns: Summary of all channels in the file with metadata.
      :rtype: ChannelSummaryTable

      .. rubric:: Examples

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'r')
      >>> summary = mth5_obj.channel_summary


   .. py:property:: fc_summary
      :type: mth5.tables.FCSummaryTable


      Get Fourier coefficient summary table.

      :returns: Summary of all Fourier coefficients in the file.
      :rtype: FCSummaryTable


   .. py:property:: run_summary

      Get run summary with MTH5 file path.

      :returns: Summary of runs with mth5_path column added.
      :rtype: pandas.DataFrame


   .. py:property:: tf_summary
      :type: mth5.tables.TFSummaryTable


      Get transfer function summary table.

      :returns: Summary of all transfer functions in the file.
      :rtype: TFSummaryTable


   .. py:method:: add_survey(survey_name, survey_metadata=None)

      Add a survey with metadata if given with the path:
          ``/Experiment/Surveys/survey_name``

      If the survey already exists, will return that survey and nothing
      is added.

      :param survey_name: Name of the survey, should be the same as
                           metadata.id
      :type survey_name: string
      :param survey_metadata: survey metadata container, defaults to None
      :type survey_metadata: :class:`mth5.metadata.survey`, optional
      :return: A convenience class for the added survey
      :rtype: :class:`mth5_groups.SurveyGroup`

      :Example: ::

          >>> from mth5 import mth5
          >>> mth5_obj = mth5.MTH5()
          >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
          >>> # one option
          >>> new_survey = mth5_obj.add_survey('MT001')
          >>> # another option
          >>> new_station = mth5_obj.experiment_group.surveys_group.add_survey('MT001')




   .. py:method:: get_survey(survey_name)

      Get a survey with the same name as survey_name

      :param survey_name: existing survey name
      :type survey_name: string
      :return: convenience survey class
      :rtype: :class:`mth5.mth5_groups.surveyGroup`
      :raises MTH5Error:  if the survey name is not found.

      :Example:

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
      >>> # one option
      >>> existing_survey = mth5_obj.get_survey('MT001')
      >>> # another option
      >>> existing_staiton = mth5_obj.experiment_group.surveys_group.get_survey('MT001')
      MTH5Error: MT001 does not exist, check groups_list for existing names




   .. py:method:: remove_survey(survey_name)

      Remove a survey from the file.

      .. note:: Deleting a survey is not as simple as del(survey).  In HDF5
            this does not free up memory, it simply removes the reference
            to that survey.  The common way to get around this is to
            copy what you want into a new file, or overwrite the survey.

      :param survey_name: existing survey name
      :type survey_name: string

      :Example: ::

          >>> from mth5 import mth5
          >>> mth5_obj = mth5.MTH5()
          >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
          >>> # one option
          >>> mth5_obj.remove_survey('MT001')
          >>> # another option
          >>> mth5_obj.experiment_group.surveys_group.remove_survey('MT001')




   .. py:method:: add_station(station_name: str, station_metadata=None, survey: str | None = None) -> mth5.groups.StationGroup

      Convenience function to add a station.

      Adds a new station with optional metadata. For v0.2.0 files, a survey
      must be specified.

      :param station_name: Name of the station (should match metadata.archive_id).
      :type station_name: str
      :param station_metadata: Station metadata container. Default is None.
      :type station_metadata: mt_metadata.timeseries.Station, optional
      :param survey: Survey ID. Required for file version 0.2.0. Default is None.
      :type survey: str, optional

      :returns: The added or existing station group object.
      :rtype: groups.StationGroup

      :raises ValueError: If survey is required (v0.2.0) but not provided.

      .. rubric:: Examples

      Add a station to v0.2.0 file:

      >>> mth5_obj = MTH5(file_version='0.2.0')
      >>> mth5_obj.open_mth5('test.mth5', 'w')
      >>> station = mth5_obj.add_station('MT001', survey='survey_001')

      .. seealso::

         :obj:`get_station`
             Retrieve existing station

         :obj:`remove_station`
             Delete a station



   .. py:method:: get_station(station_name: str, survey: str | None = None) -> mth5.groups.StationGroup

      Get an existing station from the MTH5 file.

      :param station_name: Name of the station to retrieve.
      :type station_name: str
      :param survey: Survey ID. Required for file version 0.2.0. Default is None.
      :type survey: str, optional

      :returns: The requested station group object.
      :rtype: groups.StationGroup

      :raises MTH5Error: If the station cannot be found.

      .. rubric:: Examples

      Get a station:

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'r')
      >>> station = mth5_obj.get_station('MT001', survey='survey_001')

      .. seealso::

         :obj:`add_station`
             Create a new station

         :obj:`remove_station`
             Delete a station



   .. py:method:: remove_station(station_name, survey=None)

      Convenience function to remove a station using

      Remove a station from the file.

      .. note:: Deleting a station is not as simple as del(station).  In HDF5
            this does not free up memory, it simply removes the reference
            to that station.  The common way to get around this is to
            copy what you want into a new file, or overwrite the station.

      :param station_name: existing station name
      :type station_name: string
      :param survey: existing survey name, needed for file version >= 0.2.0
      :type survey: string

      :Example:

      >>> mth5_obj.remove_station('MT001')




   .. py:method:: add_run(station_name: str, run_name: str, run_metadata=None, survey: str | None = None) -> mth5.groups.RunGroup

      Add a run to a given station.

      :param station_name: Existing station name.
      :type station_name: str
      :param run_name: Name of the run (typically archive_id followed by a-z).
      :type run_name: str
      :param run_metadata: Run metadata container. Default is None.
      :type run_metadata: mt_metadata.timeseries.Run, optional
      :param survey: Survey ID. Required for file version 0.2.0. Default is None.
      :type survey: str, optional

      :returns: The added or existing run group object.
      :rtype: groups.RunGroup

      .. rubric:: Examples

      Add a run to a station:

      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'w')
      >>> run = mth5_obj.add_run('MT001', 'MT001a', survey='survey_001')

      .. seealso::

         :obj:`get_run`
             Retrieve existing run

         :obj:`remove_run`
             Delete a run



   .. py:method:: get_run(station_name, run_name, survey=None)

      Convenience function to get a run using
      ``mth5.stations_group.get_station(station_name).get_run()``

      get a run from run name for a given station

      :param station_name: existing station name
      :type station_name: string
      :param run_name: existing run name
      :type run_name: string
      :param survey: existing survey name, needed for file version >= 0.2.0
      :type survey: string
      :return: Run object
      :rtype: :class:`mth5.mth5_groups.RunGroup`

      :Example:

      >>> existing_run = mth5_obj.get_run('MT001', 'MT001a')




   .. py:method:: remove_run(station_name, run_name, survey=None)

      Remove a run from the station.

      .. note:: Deleting a run is not as simple as del(run).  In HDF5
            this does not free up memory, it simply removes the reference
            to that station.  The common way to get around this is to
            copy what you want into a new file, or overwrite the run.

      :param station_name: existing station name
      :type station_name: string
      :param run_name: existing run name
      :type run_name: string
      :param survey: existing survey name, needed for file version >= 0.2.0
      :type survey: string

      :Example:

      >>> mth5_obj.remove_station('MT001', 'MT001a')




   .. py:method:: add_channel(station_name: str, run_name: str, channel_name: str, channel_type: str, data, channel_dtype: str = 'int32', max_shape: tuple[int | None, Ellipsis] = (None, ), chunks: bool = True, channel_metadata=None, survey: str | None = None) -> mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset

      Add a channel to a given run and station.

      :param station_name: Existing station name.
      :type station_name: str
      :param run_name: Existing run name.
      :type run_name: str
      :param channel_name: Name of the channel (component, e.g., 'Ex', 'Hy').
      :type channel_name: str
      :param channel_type: Type of channel: 'electric', 'magnetic', or 'auxiliary'.
      :type channel_type: str
      :param data: Channel data array.
      :type data: ndarray
      :param channel_dtype: NumPy data type for storage.
      :type channel_dtype: str, default 'int32'
      :param max_shape: Maximum shape (allows resizing). None allows unlimited growth.
      :type max_shape: tuple[int | None, ...], default (None,)
      :param chunks: Enable HDF5 chunking for better performance.
      :type chunks: bool, default True
      :param channel_metadata: Channel metadata container. Default is None.
      :type channel_metadata: mt_metadata.timeseries.Electric | Magnetic | Auxiliary, optional
      :param survey: Survey ID. Required for file version 0.2.0. Default is None.
      :type survey: str, optional

      :returns: The added channel dataset object.
      :rtype: groups.ElectricDataset | groups.MagneticDataset | groups.AuxiliaryDataset

      :raises MTH5Error: If channel type is not valid.

      .. rubric:: Examples

      Add an electric field channel:

      >>> import numpy as np
      >>> mth5_obj = MTH5()
      >>> mth5_obj.open_mth5('test.mth5', 'w')
      >>> data = np.random.random(1000)
      >>> ch = mth5_obj.add_channel('MT001', 'MT001a', 'Ex', 'electric',
      ...                            data, survey='survey_001')

      .. seealso::

         :obj:`get_channel`
             Retrieve existing channel

         :obj:`remove_channel`
             Delete a channel



   .. py:method:: get_channel(station_name, run_name, channel_name, survey=None)

      Convenience function to get a channel using
      ``mth5.stations_group.get_station().get_run().get_channel()``

      Get a channel from an existing name.  Returns the appropriate
      container.

      :param station_name: existing station name
      :type station_name: string
      :param run_name: existing run name
      :type run_name: string
      :param channel_name: name of the channel
      :type channel_name: string
      :return: Channel container
      :rtype: [ :class:`mth5.mth5_groups.ElectricDatset` |
                :class:`mth5.mth5_groups.MagneticDatset` |
                :class:`mth5.mth5_groups.AuxiliaryDatset` ]
      :param survey: existing survey name, needed for file version >= 0.2.0
      :type survey: string
      :raises MTH5Error:  If no channel is found

      :Example:

      >>> existing_channel = mth5_obj.get_channel(station_name,
      >>> ...                                     run_name,
      >>> ...                                     channel_name)
      >>> existing_channel
      Channel Electric:
      -------------------
                      component:        Ex
              data type:        electric
              data format:      float32
              data shape:       (4096,)
              start:            1980-01-01T00:00:00+00:00
              end:              1980-01-01T00:00:01+00:00
              sample rate:      4096




   .. py:method:: remove_channel(station_name, run_name, channel_name, survey=None)

      Convenience function to remove a channel using
      ``mth5.stations_group.get_station().get_run().remove_channel()``

      Remove a channel from a given run and station.

      .. note:: Deleting a channel is not as simple as del(channel).  In HDF5
            this does not free up memory, it simply removes the reference
            to that channel.  The common way to get around this is to
            copy what you want into a new file, or overwrite the channel.

      :param station_name: existing station name
      :type station_name: string
      :param run_name: existing run name
      :type run_name: string
      :param channel_name: existing station name
      :type channel_name: string
      :param survey: existing survey name, needed for file version >= 0.2.0
      :type survey: string

      :Example:

      >>> mth5_obj.remove_channel('MT001', 'MT001a', 'Ex')




   .. py:method:: add_transfer_function(tf_object, update_metadata=True)

      Add a transfer function
      :param tf_object: DESCRIPTION
      :type tf_object: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_transfer_function(station_id, tf_id, survey=None)

      Get a transfer function

      :param survey_id: DESCRIPTION
      :type survey_id: TYPE
      :param station_id: DESCRIPTION
      :type station_id: TYPE
      :param tf_id: DESCRIPTION
      :type tf_id: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: remove_transfer_function(station_id, tf_id, survey=None)

      remove a transfer function

      :param survey_id: DESCRIPTION
      :type survey_id: TYPE
      :param station_id: DESCRIPTION
      :type station_id: TYPE
      :param tf_id: DESCRIPTION
      :type tf_id: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




