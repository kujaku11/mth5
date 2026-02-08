mth5.io.conversion
==================

.. py:module:: mth5.io.conversion

.. autoapi-nested-parse::

   Convert MTH5 to other formats

   - MTH5 -> miniSEED + StationXML



Classes
-------

.. autoapisummary::

   mth5.io.conversion.MTH5ToMiniSEEDStationXML


Functions
---------

.. autoapisummary::

   mth5.io.conversion.get_encoding
   mth5.io.conversion.split_miniseed_by_day


Module Contents
---------------

.. py:class:: MTH5ToMiniSEEDStationXML(mth5_path: str | pathlib.Path | None = None, save_path: str | pathlib.Path | None = None, network_code: str = 'ZU', use_runs_with_data_only: bool = True, **kwargs: Any)

   Convert MTH5 files to miniSEED and StationXML formats.

   This class provides functionality to convert magnetotelluric data stored
   in MTH5 format to industry-standard miniSEED time series files and
   StationXML metadata files for data exchange and archival purposes.

   :param mth5_path: Path to the input MTH5 file to be converted
   :type mth5_path: str, Path, or None, default None
   :param save_path: Directory path where output files will be saved. If None, uses
                     the parent directory of mth5_path
   :type save_path: str, Path, or None, default None
   :param network_code: Two-character FDSN network code for the output files
   :type network_code: str, default "ZU"
   :param use_runs_with_data_only: If True, only process runs that contain actual time series data
   :type use_runs_with_data_only: bool, default True
   :param \*\*kwargs: Additional keyword arguments to set as instance attributes
   :type \*\*kwargs: dict

   .. attribute:: mth5_path

      Path to the MTH5 input file

      :type: Path or None

   .. attribute:: save_path

      Directory where output files are saved

      :type: Path

   .. attribute:: network_code

      FDSN network code for output files

      :type: str

   .. attribute:: use_runs_with_data_only

      Flag to process only runs with data

      :type: bool

   .. attribute:: encoding

      Encoding format for miniSEED files

      :type: str or None

   .. rubric:: Examples

   >>> converter = MTH5ToMiniSEEDStationXML(
   ...     mth5_path="/path/to/data.h5",
   ...     network_code="MT",
   ...     save_path="/path/to/output"
   ... )
   >>> xml_file, mseed_files = converter.convert_mth5_to_ms_stationxml()


   .. py:property:: mth5_path
      :type: pathlib.Path | None


      Path to the MTH5 input file.

      :returns: Path to the MTH5 file to be converted, or None if not set.
      :rtype: Path or None


   .. py:property:: save_path
      :type: pathlib.Path


      Directory path where output files will be saved.

      :returns: Directory path for saving miniSEED and StationXML files.
      :rtype: Path


   .. py:property:: network_code
      :type: str


      Two-character FDSN network code.

      :returns: Alphanumeric string of exactly 2 characters as required by FDSN DMC.
      :rtype: str


   .. py:attribute:: use_runs_with_data_only
      :value: True



   .. py:attribute:: encoding
      :value: None



   .. py:method:: convert_mth5_to_ms_stationxml(mth5_path: str | pathlib.Path, save_path: str | pathlib.Path | None = None, network_code: str = 'ZU', use_runs_with_data_only: bool = True, **kwargs: Any) -> tuple[pathlib.Path, list[pathlib.Path]]
      :classmethod:


      Convert an MTH5 file to miniSEED and StationXML formats.

      Class method that provides a convenient interface to convert MTH5 data
      to standard seismological formats for data exchange and archival.

      :param mth5_path: Path to the input MTH5 file to be converted
      :type mth5_path: str or Path
      :param save_path: Directory where output files will be saved. If None, uses the
                        parent directory of mth5_path
      :type save_path: str, Path, or None, default None
      :param network_code: Two-character FDSN network code for output files
      :type network_code: str, default "ZU"
      :param use_runs_with_data_only: If True, only process runs containing actual time series data
      :type use_runs_with_data_only: bool, default True
      :param \*\*kwargs: Additional keyword arguments passed to converter initialization
      :type \*\*kwargs: dict

      :returns: Tuple containing:
                - Path to the generated StationXML file
                - List of paths to generated miniSEED files (one per day per channel)
      :rtype: tuple[Path, list[Path]]

      .. rubric:: Examples

      >>> xml_file, mseed_files = MTH5ToMiniSEEDStationXML.convert_mth5_to_ms_stationxml(
      ...     "/path/to/data.h5",
      ...     network_code="MT",
      ...     save_path="/output/directory"
      ... )
      >>> print(f"Created {len(mseed_files)} miniSEED files and {xml_file}")



   .. py:method:: split_ms_to_days(streams, save_path: pathlib.Path, encoding: str) -> list[pathlib.Path]

      Split miniSEED traces into daily files.

      Splits continuous time series traces into separate files for each day
      to conform with standard seismological data archiving practices.

      :param streams: Stream object containing traces to be split by day
      :type streams: obspy.Stream
      :param save_path: Directory where daily miniSEED files will be saved
      :type save_path: Path
      :param encoding: Data encoding format for miniSEED files (e.g., 'INT32', 'FLOAT64')
      :type encoding: str

      :returns: List of paths to the generated daily miniSEED files
      :rtype: list[Path]

      .. rubric:: Notes

      Files are named using the pattern:
      {network}_{station}_{location}_{channel}_{YYYY_MM_DDTHH_MM_SS}.mseed



.. py:function:: get_encoding(run_ts) -> str

   Determine consistent data encoding for miniSEED files across channels.

   Analyzes data types across all channels in a run and selects a median
   encoding to ensure compatibility in miniSEED file generation.

   :param run_ts: Run time series object containing multiple channels of data
   :type run_ts: RunTS

   :returns: String identifier for miniSEED encoding format (e.g., 'INT32', 'FLOAT64')
   :rtype: str

   .. rubric:: Notes

   Uses median data type to handle mixed precision datasets. Automatically
   converts INT64 to INT32 for miniSEED compatibility since some readers
   don't support 64-bit integers.

   .. rubric:: Examples

   >>> encoding = get_encoding(run_timeseries)
   >>> print(f"Selected encoding: {encoding}")


.. py:function:: split_miniseed_by_day(input_file: str | pathlib.Path) -> list[pathlib.Path]

   Split an existing miniSEED file into daily files.

   Utility function to split a multi-day miniSEED file into separate files
   for each calendar day, following standard seismological archiving practices.

   :param input_file: Path to the input miniSEED file to be split
   :type input_file: str or Path

   :returns: List of paths to the generated daily miniSEED files
   :rtype: list[Path]

   .. rubric:: Notes

   Output files are named using the pattern:
   {network}.{station}.{location}.{channel}.{YYYY-MM-DD}.mseed

   Files are saved in the same directory as the input file.

   .. rubric:: Examples

   >>> daily_files = split_miniseed_by_day("/path/to/continuous.mseed")
   >>> print(f"Created {len(daily_files)} daily files")


