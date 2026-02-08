mth5.io.metronix
================

.. py:module:: mth5.io.metronix

.. autoapi-nested-parse::

   Created on Fri Nov 22 13:55:28 2024

   @author: jpeacock



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/metronix/metronix_atss/index
   /autoapi/mth5/io/metronix/metronix_collection/index
   /autoapi/mth5/io/metronix/metronix_metadata/index


Classes
-------

.. autoapisummary::

   mth5.io.metronix.MetronixFileNameMetadata
   mth5.io.metronix.MetronixChannelJSON
   mth5.io.metronix.ATSS
   mth5.io.metronix.MetronixCollection


Functions
---------

.. autoapisummary::

   mth5.io.metronix.read_atss


Package Contents
----------------

.. py:class:: MetronixFileNameMetadata(fn: Union[str, pathlib.Path, None] = None, **kwargs: Any)

   Parse and manage metadata from Metronix filename conventions.

   This class extracts metadata information from Metronix ATSS filenames
   including system information, channel details, and file properties.

   :param fn: Path to Metronix file, by default None
   :type fn: Union[str, Path, None], optional
   :param \*\*kwargs: Additional keyword arguments (currently unused)

   .. attribute:: system_number

      System identification number

      :type: str or None

   .. attribute:: system_name

      Name of the system

      :type: str or None

   .. attribute:: channel_number

      Channel number (parsed from C## format)

      :type: int or None

   .. attribute:: component

      Component designation (e.g., 'ex', 'ey', 'hx', 'hy', 'hz')

      :type: str or None

   .. attribute:: sample_rate

      Sampling rate in Hz

      :type: float or None

   .. attribute:: file_type

      Type of file ('metadata' or 'timeseries')

      :type: str or None


   .. py:attribute:: system_number
      :type:  str | None
      :value: None



   .. py:attribute:: system_name
      :type:  str | None
      :value: None



   .. py:attribute:: channel_number
      :type:  int | None
      :value: None



   .. py:attribute:: component
      :type:  str | None
      :value: None



   .. py:attribute:: sample_rate
      :type:  float | None
      :value: None



   .. py:attribute:: file_type
      :type:  str | None
      :value: None



   .. py:property:: fn
      :type: pathlib.Path | None


      Get the file path.

      :returns: File path object or None if not set
      :rtype: Path or None


   .. py:property:: fn_exists
      :type: bool


      Check if the file exists.

      :returns: True if file exists, False otherwise
      :rtype: bool


   .. py:property:: file_size
      :type: int


      Get file size in bytes.

      :returns: File size in bytes, 0 if file is None
      :rtype: int


   .. py:property:: n_samples
      :type: float


      Get estimated number of samples in file.

      Assumes 8 bytes per sample (double precision).

      :returns: Estimated number of samples
      :rtype: float


   .. py:property:: duration
      :type: float


      Get estimated duration of the file in seconds.

      :returns: Duration in seconds
      :rtype: float


.. py:class:: MetronixChannelJSON(fn: Union[str, pathlib.Path, None] = None, **kwargs: Any)

   Bases: :py:obj:`MetronixFileNameMetadata`


   Read and parse Metronix JSON metadata files.

   This class extends MetronixFileNameMetadata to handle JSON metadata
   files containing channel configuration and calibration information.

   :param fn: Path to Metronix JSON file, by default None
   :type fn: Union[str, Path, None], optional
   :param \*\*kwargs: Additional keyword arguments passed to parent class

   .. attribute:: metadata

      Parsed JSON metadata as a SimpleNamespace object

      :type: SimpleNamespace or None


   .. py:attribute:: metadata
      :type:  types.SimpleNamespace | None
      :value: None



   .. py:method:: read(fn: Union[str, pathlib.Path, None] = None) -> None

      Read JSON metadata from file.

      :param fn: Path to JSON file, by default None (uses self.fn)
      :type fn: Union[str, Path, None], optional

      :raises IOError: If JSON file cannot be found



   .. py:method:: get_channel_metadata() -> Union[mt_metadata.timeseries.Electric, mt_metadata.timeseries.Magnetic, None]

      Translate to mt_metadata.timeseries.Channel object.

      Creates either Electric or Magnetic metadata objects based on the
      component type and applies calibration filters.

      :returns: mt_metadata object based on component type, or None if no metadata
      :rtype: Union[Electric, Magnetic, None]

      :raises ValueError: If component type is not recognized



   .. py:method:: get_sensor_response_filter() -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter | None

      Get the sensor response frequency-amplitude-phase filter.

      Creates a FrequencyResponseTableFilter from the sensor calibration
      data stored in the JSON metadata.

      :returns: Sensor response filter if calibration data exists, None otherwise
      :rtype: FrequencyResponseTableFilter or None



   .. py:method:: get_channel_response() -> mt_metadata.timeseries.filters.ChannelResponse

      Get all filters needed to calibrate the data.

      :returns: Channel response object containing all calibration filters
      :rtype: ChannelResponse



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


.. py:class:: MetronixCollection(file_path: Union[str, pathlib.Path, None] = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection class for managing Metronix ATSS files.

   This class extends the base Collection class to handle Metronix ATSS
   (Audio Time Series System) files and their associated JSON metadata files.
   It provides functionality to create pandas DataFrames with comprehensive
   metadata for processing workflows.

   :param file_path: Path to directory containing Metronix ATSS files, by default None
   :type file_path: Union[str, Path, None], optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Collection class

   .. attribute:: file_ext

      List of file extensions to search for (["atss"])

      :type: list[str]

   .. rubric:: Examples

   >>> from mth5.io.metronix import MetronixCollection
   >>> collection = MetronixCollection("/path/to/metronix/files")
   >>> df = collection.to_dataframe(sample_rates=[128, 256])


   .. py:attribute:: file_ext
      :type:  list[str]
      :value: ['atss']



   .. py:method:: to_dataframe(sample_rates: list[int] = [128], run_name_zeros: int = 0, calibration_path: Union[str, pathlib.Path, None] = None) -> pandas.DataFrame

      Create DataFrame for Metronix timeseries ATSS + JSON file sets.

      Processes all ATSS files in the collection directory, extracts metadata,
      and creates a comprehensive pandas DataFrame with information about each
      channel including timing, location, and instrument details.

      :param sample_rates: List of sample rates to include in Hz, by default [128]
      :type sample_rates: list[int], optional
      :param run_name_zeros: Number of zeros for zero-padding run names. If 0, run names
                             are unchanged. If > 0, run names are formatted as
                             'sr{sample_rate}_{run_number:0{zeros}d}', by default 0
      :type run_name_zeros: int, optional
      :param calibration_path: Path to calibration files (currently unused), by default None
      :type calibration_path: Union[str, Path, None], optional

      :returns: DataFrame with columns:
                - survey: Survey ID
                - station: Station ID
                - run: Run ID
                - start: Start time (datetime)
                - end: End time (datetime)
                - channel_id: Channel number
                - component: Component name (ex, ey, hx, hy, hz)
                - fn: File path
                - sample_rate: Sample rate in Hz
                - file_size: File size in bytes
                - n_samples: Number of samples
                - sequence_number: Sequence number (always 0)
                - dipole: Dipole length (always 0)
                - coil_number: Coil serial number (magnetic channels only)
                - latitude: Latitude in decimal degrees
                - longitude: Longitude in decimal degrees
                - elevation: Elevation in meters
                - instrument_id: Instrument/system number
                - calibration_fn: Calibration file path (always None)
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> collection = MetronixCollection("/path/to/files")
      >>> df = collection.to_dataframe(sample_rates=[128, 256])
      >>> df = collection.to_dataframe(run_name_zeros=4)  # Zero-pad run names



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 0) -> pandas.DataFrame

      Assign formatted run names based on sample rate and run number.

      If zeros is 0, run names are unchanged. Otherwise, run names are
      formatted as 'sr{sample_rate}_{run_number:0{zeros}d}' where the
      run number is extracted from the original run name after the first
      underscore.

      :param df: DataFrame containing run information with 'run' and 'sample_rate' columns
      :type df: pd.DataFrame
      :param zeros: Number of zeros for zero-padding run numbers. If 0, run names
                    are unchanged, by default 0
      :type zeros: int, optional

      :returns: DataFrame with updated run names
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> df = pd.DataFrame({
      ...     'run': ['run_1', 'run_2'],
      ...     'sample_rate': [128, 256]
      ... })
      >>> collection = MetronixCollection()
      >>> result = collection.assign_run_names(df, zeros=3)
      >>> print(result['run'].tolist())
      ['sr128_001', 'sr256_002']

      .. rubric:: Notes

      The method expects run names to be in format 'prefix_number' where
      'number' can be extracted and converted to an integer for formatting.



