mth5.io.metronix.metronix_atss
==============================

.. py:module:: mth5.io.metronix.metronix_atss

.. autoapi-nested-parse::

   ATSS (Audio Time Series System) file reader for Metronix data.

   This module provides functionality to read and process Metronix ATSS binary
   time series files and their associated JSON metadata files. ATSS files contain
   double precision floating point time series data equivalent to numpy arrays
   of type np.float64.

   The ATSS format consists of two files:
   - .atss file: Binary time series data (np.float64 values)
   - .json file: Metadata in JSON format

   This implementation is translated from:
   https://github.com/bfrmtx/MTHotel/blob/main/python/include/atss_file.py

   Classes
   -------
   ATSS : MetronixFileNameMetadata
       Main class for reading ATSS files and converting to ChannelTS objects.

   Functions
   ---------
   read_atss : function
       Convenience function to read ATSS file and return ChannelTS object.

   .. rubric:: Notes

   ATSS files store time series data as consecutive double precision floating
   point numbers in binary format, making them efficient for large datasets.

   .. rubric:: Examples

   >>> from mth5.io.metronix.metronix_atss import ATSS, read_atss
   >>>
   >>> # Using the ATSS class directly
   >>> atss = ATSS('data/station001.atss')
   >>> data = atss.read_atss()
   >>> channel_ts = atss.to_channel_ts()
   >>>
   >>> # Using the convenience function
   >>> channel_ts = read_atss('data/station001.atss')

   Author
   ------
   jpeacock

   Created
   -------
   Tue Nov 26 15:54:12 2024



Classes
-------

.. autoapisummary::

   mth5.io.metronix.metronix_atss.ATSS


Functions
---------

.. autoapisummary::

   mth5.io.metronix.metronix_atss.read_atss


Module Contents
---------------

.. py:class:: ATSS(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.metronix.MetronixFileNameMetadata`


   ATSS (Audio Time Series System) file reader for Metronix data.

   Handles reading and processing of Metronix ATSS binary time series files
   and their associated JSON metadata files. ATSS files contain double precision
   floating point time series data equivalent to numpy arrays of type np.float64.

   :param fn: Path to the ATSS file. If provided, metadata will be automatically
              loaded if the corresponding JSON file exists.
   :type fn: str or Path, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent class.

   .. attribute:: header

      Metadata handler for the associated JSON file.

      :type: MetronixChannelJSON

   .. rubric:: Notes

   ATSS files come in pairs:
   - .atss file: Binary time series data (np.float64)
   - .json file: Metadata in JSON format

   .. rubric:: Examples

   >>> atss = ATSS('data/station001_run001_ch001.atss')
   >>> data = atss.read_atss()
   >>> channel_ts = atss.to_channel_ts()


   .. py:attribute:: header


   .. py:property:: metadata_fn
      :type: pathlib.Path | None


      Path to the metadata JSON file.

      Returns the path to the JSON metadata file that corresponds to this
      ATSS file. The JSON file has the same base name as the ATSS file
      but with a .json extension.

      :returns: Path to the JSON metadata file, or None if no ATSS file is set.
      :rtype: Path or None

      .. rubric:: Examples

      >>> atss = ATSS('data/station001.atss')
      >>> atss.metadata_fn
      PosixPath('data/station001.json')


   .. py:method:: has_metadata_file() -> bool

      Check if metadata JSON file exists.

      :returns: True if the metadata JSON file exists, False otherwise.
      :rtype: bool

      .. rubric:: Examples

      >>> atss = ATSS('data/station001.atss')
      >>> atss.has_metadata_file()
      True



   .. py:method:: read_atss(fn: str | pathlib.Path | None = None, start: int = 0, stop: int = 0) -> numpy.ndarray

      Read binary ATSS time series data.

      Reads double precision floating point time series data from the ATSS
      binary file. Data is stored as consecutive np.float64 values.

      :param fn: Path to ATSS file. If None, uses the current file path.
      :type fn: str or Path, optional
      :param start: Starting sample index (0-based).
      :type start: int, default 0
      :param stop: Ending sample index. If 0, reads to end of file.
      :type stop: int, default 0

      :returns: Time series data as 1D array of np.float64 values.
      :rtype: np.ndarray

      :raises ValueError: If stop index exceeds the number of samples in the file.

      .. rubric:: Examples

      >>> atss = ATSS('data/station001.atss')
      >>> data = atss.read_atss()  # Read entire file
      >>> data_slice = atss.read_atss(start=1000, stop=2000)  # Read subset



   .. py:method:: write_atss(data_array: numpy.ndarray, filename: str | pathlib.Path) -> None

      Write time series data to ATSS binary file.

      Writes numpy array data as double precision floating point values
      to a binary ATSS file.

      :param data_array: Time series data to write. Will be converted to np.float64.
      :type data_array: np.ndarray
      :param filename: Output file path for the ATSS binary file.
      :type filename: str or Path

      .. rubric:: Examples

      >>> import numpy as np
      >>> atss = ATSS()
      >>> data = np.random.randn(10000)
      >>> atss.write_atss(data, 'output.atss')



   .. py:property:: channel_metadata
      :type: mt_metadata.timeseries.electric.Electric | mt_metadata.timeseries.magnetic.Magnetic | mt_metadata.timeseries.auxiliary.Auxiliary


      Channel metadata from the JSON header file.

      :returns: Channel metadata object based on the channel type.
      :rtype: Electric or Magnetic or Auxiliary


   .. py:property:: channel_response
      :type: mt_metadata.timeseries.filters.ChannelResponse


      Channel response information from the JSON header file.

      :returns: Channel response/calibration information.
      :rtype: ChannelResponse


   .. py:property:: channel_type
      :type: str


      Determine channel type from component name.

      Channel type is determined from the component identifier in the filename:
      - Components starting with 'e': electric
      - Components starting with 'h': magnetic
      - All others: auxiliary

      :returns: Channel type: 'electric', 'magnetic', or 'auxiliary'.
      :rtype: str


   .. py:property:: run_id
      :type: str | None


      Extract run ID from file path.

      Expects file path structure: .../station/run/timeseries.atss
      The run ID is extracted from the parent directory name.

      :returns: Run identifier, or None if file doesn't exist.
      :rtype: str or None


   .. py:property:: station_id
      :type: str | None


      Extract station ID from file path.

      Expects file path structure: .../station/run/timeseries.atss
      The station ID is extracted from the grandparent directory name.

      :returns: Station identifier, or None if file doesn't exist.
      :rtype: str or None


   .. py:property:: survey_id
      :type: str | None


      Extract survey ID from file path.

      Expects file path structure: .../survey/stations/station/run/timeseries.atss
      The survey ID is extracted from the great-great-grandparent directory name.

      :returns: Survey identifier, or None if file doesn't exist.
      :rtype: str or None


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Generate run-level metadata.

      Creates a Run metadata object populated with information from the
      ATSS file and its associated JSON metadata.

      :returns: Run metadata object with data logger info, sample rate,
                and channel metadata.
      :rtype: Run


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Generate station-level metadata.

      Creates a Station metadata object populated with location information
      from the JSON metadata and run information.

      :returns: Station metadata object with location coordinates and run metadata.
      :rtype: Station


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Generate survey-level metadata.

      Creates a Survey metadata object that includes station metadata
      and overall time period information.

      :returns: Survey metadata object containing station information.
      :rtype: Survey


   .. py:method:: to_channel_ts(fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Create a ChannelTS object from ATSS data.

      Converts the ATSS time series data and metadata into a ChannelTS
      object suitable for use with MTH5 workflows.

      :param fn: Path to ATSS file. If None, uses current file path.
      :type fn: str or Path, optional

      :returns: Time series object with data, metadata, and response information.
      :rtype: ChannelTS

      .. warning::

         Can be slow due to pandas datetime index creation for large datasets.
         A warning is logged if the metadata JSON file is missing.

      .. rubric:: Examples

      >>> atss = ATSS('data/station001.atss')
      >>> channel_ts = atss.to_channel_ts()
      >>> print(channel_ts.sample_rate)
      1024.0



.. py:function:: read_atss(fn: str | pathlib.Path, calibration_fn: str | pathlib.Path | None = None, logger_file_handler: Any = None) -> mth5.timeseries.ChannelTS

   Generic tool to read ATSS file and return ChannelTS object.

   Convenience function that creates an ATSS object and converts it
   to a ChannelTS in a single call.

   :param fn: Path to the ATSS file to read.
   :type fn: str or Path
   :param calibration_fn: Path to calibration file (currently unused).
   :type calibration_fn: str or Path, optional
   :param logger_file_handler: Logger file handler (currently unused).
   :type logger_file_handler: Any, optional

   :returns: Time series object with data and metadata from the ATSS file.
   :rtype: ChannelTS

   .. rubric:: Examples

   >>> channel_ts = read_atss('data/station001.atss')
   >>> print(f"Loaded {len(channel_ts.ts)} samples")


