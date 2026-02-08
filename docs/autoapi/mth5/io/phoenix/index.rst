mth5.io.phoenix
===============

.. py:module:: mth5.io.phoenix


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/phoenix/phoenix_collection/index
   /autoapi/mth5/io/phoenix/read/index
   /autoapi/mth5/io/phoenix/readers/index


Classes
-------

.. autoapisummary::

   mth5.io.phoenix.PhoenixReceiverMetadata
   mth5.io.phoenix.PhoenixConfig
   mth5.io.phoenix.PhoenixCollection


Functions
---------

.. autoapisummary::

   mth5.io.phoenix.read_phoenix
   mth5.io.phoenix.open_phoenix


Package Contents
----------------

.. py:class:: PhoenixReceiverMetadata(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Container for Phoenix Geophysics recmeta.json metadata files.

   This class reads and parses receiver metadata from JSON configuration files
   used to control Phoenix Geophysics MTU-5C data recording systems. It provides
   methods to extract channel configurations, instrument settings, and convert
   them to standardized metadata objects.

   :param fn: Path to the recmeta.json file. If provided, the file will be read
              automatically during initialization.
   :type fn: str, Path, or None, optional
   :param \*\*kwargs: Additional keyword arguments (currently unused).

   .. attribute:: fn

      Path to the metadata file.

      :type: Path or None

   .. attribute:: obj

      Parsed JSON content as a SimpleNamespace object.

      :type: SimpleNamespace or None

   .. attribute:: logger

      Logger instance for error reporting.

      :type: loguru.Logger

   :raises IOError: If the specified file does not exist.

   .. rubric:: Examples

   >>> metadata = PhoenixReceiverMetadata("recmeta.json")
   >>> channel_map = metadata.channel_map
   >>> e1_config = metadata.e1_metadata

   .. rubric:: Notes

   The class supports both electric and magnetic channel configurations
   with automatic mapping from Phoenix-specific parameter names to
   standardized metadata attributes.


   .. py:property:: fn
      :type: pathlib.Path | None


      Path to the metadata file.

      :returns: Path to the recmeta.json file, or None if not set.
      :rtype: Path or None


   .. py:attribute:: obj
      :type:  types.SimpleNamespace | None
      :value: None



   .. py:attribute:: logger


   .. py:property:: instrument_id
      :type: str | None


      Instrument identifier from metadata.

      :returns: Instrument ID if available, None otherwise.
      :rtype: str or None


   .. py:method:: read(fn: str | pathlib.Path | None = None) -> None

      Read a recmeta.json file in Phoenix format.

      :param fn: Path to the JSON file. If None, uses the current fn property.
      :type fn: str, Path, or None, optional

      :raises IOError: If no file path is specified or file doesn't exist.
      :raises ValueError: If the file cannot be parsed as JSON.



   .. py:method:: has_obj() -> bool

      Check if metadata object is loaded.

      :returns: True if metadata object exists, False otherwise.
      :rtype: bool



   .. py:property:: channel_map
      :type: dict[int, str]


      Channel mapping from index to component tag.

      :returns: Dictionary mapping channel indices to component tags (lowercase).
      :rtype: dict[int, str]

      :raises AttributeError: If metadata object is not loaded or missing channel_map.


   .. py:property:: lp_filter_base_name
      :type: str | None


      Base name for low-pass filter identifiers.

      :returns: Filter base name combining receiver info, or None if not available.
      :rtype: str or None


   .. py:method:: get_ch_index(tag: str) -> int

      Get channel index from component tag.

      :param tag: Component tag (e.g., 'e1', 'h1', etc.).
      :type tag: str

      :returns: Channel index corresponding to the tag.
      :rtype: int

      :raises ValueError: If the tag is not found in the channel map.
      :raises AttributeError: If metadata object is not loaded.



   .. py:method:: get_ch_tag(index: int) -> str

      Get component tag from channel index.

      :param index: Channel index.
      :type index: int

      :returns: Component tag corresponding to the index.
      :rtype: str

      :raises ValueError: If the index is not found in the channel map.
      :raises AttributeError: If metadata object is not loaded.



   .. py:property:: e1_metadata
      :type: mt_metadata.timeseries.Electric


      Electric channel 1 metadata.


   .. py:property:: e2_metadata
      :type: mt_metadata.timeseries.Electric


      Electric channel 2 metadata.


   .. py:property:: h1_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 1 metadata.


   .. py:property:: h2_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 2 metadata.


   .. py:property:: h3_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 3 metadata.


   .. py:property:: h4_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 4 metadata.


   .. py:property:: h5_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 5 metadata.


   .. py:property:: h6_metadata
      :type: mt_metadata.timeseries.Magnetic


      Magnetic channel 6 metadata.


   .. py:method:: get_ch_metadata(index: int) -> mt_metadata.timeseries.Electric | mt_metadata.timeseries.Magnetic

      Get channel metadata from index.

      :param index: Channel index.
      :type index: int

      :returns: Channel metadata object corresponding to the index.
      :rtype: Electric or Magnetic

      :raises ValueError: If index is not found in channel map.
      :raises AttributeError: If the corresponding metadata property doesn't exist.



   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Run metadata from receiver configuration.

      :returns: Run metadata object with data logger and timing information.
      :rtype: Run


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Station metadata from receiver configuration.

      :returns: Station metadata object with location and acquisition information.
      :rtype: Station


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Survey metadata from receiver configuration.

      :returns: Survey metadata object with survey information.
      :rtype: Survey


.. py:class:: PhoenixConfig(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Phoenix Geophysics configuration file reader and metadata container.

   This class reads and provides access to Phoenix MTU-5C instrument
   configuration data stored in JSON format. The configuration file contains
   recording parameters, instrument settings, and metadata used to control
   data acquisition.

   :param fn: Path to the Phoenix configuration file (typically config.json).
              If provided, the file will be validated for existence.
   :type fn: str, pathlib.Path, or None, optional
   :param \*\*kwargs: Additional keyword arguments (currently unused).
   :type \*\*kwargs: Any

   .. attribute:: fn

      Path to the configuration file.

      :type: pathlib.Path or None

   .. attribute:: obj

      Parsed configuration object containing all settings.

      :type: Any or None

   .. attribute:: logger

      Logger instance for debugging and error reporting.

      :type: loguru.Logger

   .. rubric:: Examples

   >>> config = PhoenixConfig("config.json")
   >>> config.read()
   >>> station = config.station_metadata()
   >>> print(f"Station ID: {station.id}")


   .. py:attribute:: obj
      :type:  Any
      :value: None



   .. py:attribute:: logger
      :type:  loguru.Logger


   .. py:property:: fn
      :type: pathlib.Path | None


      Path to the Phoenix configuration file.

      :returns: The path to the configuration file, or None if not set.
      :rtype: pathlib.Path or None


   .. py:method:: read(fn: str | pathlib.Path | None = None) -> None

      Read and parse a Phoenix configuration file.

      Loads and parses a Phoenix MTU-5C configuration file in JSON format.
      The parsed configuration is stored in the obj attribute and provides
      access to all recording parameters and instrument settings.

      :param fn: Path to the configuration file to read. If None, uses the
                 previously set file path from the fn property.
      :type fn: str, pathlib.Path, or None, optional

      :raises ValueError: If no file path is provided and none was previously set.
      :raises IOError: If the configuration file cannot be read or parsed.

      .. rubric:: Notes

      The configuration file should be in Phoenix JSON format containing
      recording parameters, instrument settings, and metadata.



   .. py:method:: has_obj() -> bool

      Check if configuration data has been loaded.

      :returns: True if configuration data is loaded, False otherwise.
      :rtype: bool



   .. py:property:: auto_power_enabled
      :type: Any | None


      Auto power enabled setting from configuration.

      :returns: The auto power enabled setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: config
      :type: Any | None


      Main configuration section from the configuration file.

      :returns: The first configuration object containing recording parameters,
                or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: empower_version
      :type: Any | None


      EMPower software version from configuration.

      :returns: The EMPower software version, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: mtc150_reset
      :type: Any | None


      MTC150 reset setting from configuration.

      :returns: The MTC150 reset setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: network
      :type: Any | None


      Network configuration from configuration file.

      :returns: The network configuration settings, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: receiver
      :type: Any | None


      Receiver configuration from configuration file.

      :returns: The receiver configuration settings, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: schedule
      :type: Any | None


      Recording schedule from configuration file.

      :returns: The recording schedule configuration, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: surveyTechnique
      :type: Any | None


      Survey technique setting from configuration file.

      :returns: The survey technique setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: timezone
      :type: Any | None


      Timezone setting from configuration file.

      :returns: The timezone setting, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: timezone_offset
      :type: Any | None


      Timezone offset from configuration file.

      :returns: The timezone offset in hours, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:property:: version
      :type: Any | None


      Configuration file version from configuration file.

      :returns: The configuration file version, or None if no configuration is loaded.
      :rtype: Any or None


   .. py:method:: station_metadata() -> mt_metadata.timeseries.Station

      Create a Station metadata object from configuration data.

      Extracts station information from the loaded configuration and creates
      a standardized Station metadata object with basic station parameters.

      :returns: A Station metadata object populated with configuration data including
                station ID, operator information, company name, and notes.
      :rtype: Station

      :raises AttributeError: If no configuration is loaded or required fields are missing.

      .. rubric:: Notes

      The method extracts the following information from config.layout:
      - Station_Name -> station.id
      - Operator -> station.acquired_by.name
      - Company_Name -> station.acquired_by.organization
      - Notes -> station.comments

      .. rubric:: Examples

      >>> config = PhoenixConfig("config.json")
      >>> config.read()
      >>> station = config.station_metadata()
      >>> print(f"Station: {station.id}")



.. py:function:: read_phoenix(file_name: str | pathlib.Path, **kwargs: Any) -> mth5.timeseries.ChannelTS | mth5.timeseries.RunTS | mth5.io.phoenix.readers.MTUTable

   Read a Phoenix Geophysics data file into a ChannelTS or RunTS object
   depending on the file type.  Newer files that end in .td_XX or .bin will be
   read into ChannelTS objects.  Older MTU files that end in .TS3, .TS4, .TS5,
   .TSL, or .TSH will be read into RunTS objects.

   :param file_name: Path to the Phoenix data file to read.
   :type file_name: str or pathlib.Path
   :param \*\*kwargs: Additional keyword arguments. May include:

                      - rxcal_fn : str or pathlib.Path, optional
                          Path to receiver calibration file.
                      - scal_fn : str or pathlib.Path, optional
                          Path to sensor calibration file.
                      - table_filepath : str or pathlib.Path, optional
                          Path to the MTU TBL file for use with MTUTSN files.
                      - Other arguments passed to the Phoenix reader constructor.
   :type \*\*kwargs: Any

   :returns: * **channel_ts** (*ChannelTS*) -- Time series data object containing the Phoenix file data
               with calibration applied if calibration files were provided.
             * **run_ts** (*RunTS*) -- Time series data object containing the MTU data from the Phoenix MTU
               files with calibration applied if specified.
             * **mtu_table** (*MTUTable*) -- Metadata table object containing the MTU table data.

   :raises KeyError: If the file extension is not supported by any Phoenix reader.
   :raises ValueError: If the file cannot be read or converted to ChannelTS or RunTS format.


.. py:function:: open_phoenix(file_name: str | pathlib.Path, **kwargs: Any) -> mth5.io.phoenix.readers.DecimatedContinuousReader | mth5.io.phoenix.readers.DecimatedSegmentedReader | mth5.io.phoenix.readers.NativeReader | mth5.io.phoenix.readers.MTUTSN | mth5.io.phoenix.readers.MTUTable

   Open a Phoenix Geophysics data file in the appropriate container.

   :param file_name: Full path to the Phoenix data file to open.
   :type file_name: str or pathlib.Path
   :param \*\*kwargs: Additional keyword arguments to pass to the reader constructor.
   :type \*\*kwargs: Any

   :returns: **reader** -- The appropriate Phoenix reader container based on file extension:
             - .bin files: NativeReader
             - .td_24k files: DecimatedSegmentedReader
             - .td_150/.td_30 files: DecimatedContinuousReader
   :rtype: DecimatedContinuousReader | DecimatedSegmentedReader | NativeReader

   :raises KeyError: If the file extension is not supported by any Phoenix reader.


.. py:class:: PhoenixCollection(file_path: str | pathlib.Path | None = None, **kwargs)

   Bases: :py:obj:`mth5.io.Collection`


   Collection manager for Phoenix MTU data files.

   Organizes Phoenix magnetotelluric receiver files into runs based on
   timing and sample rates. Handles multiple sample rates (30, 150, 2400,
   24000, 96000 Hz) and manages receiver metadata.

   :param file_path: Path to the directory containing Phoenix data files. Can be the
                     station folder or a parent folder containing multiple stations.
   :type file_path: str | Path | None, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Collection class.

   .. attribute:: metadata_dict

      Dictionary mapping station IDs to their receiver metadata.

      :type: dict[str, PhoenixReceiverMetadata]

   .. rubric:: Examples

   Create a collection from a station directory:

   >>> from mth5.io.phoenix import PhoenixCollection
   >>> collection = PhoenixCollection(r"/path/to/station")
   >>> runs = collection.get_runs(sample_rates=[150, 24000])
   >>> print(runs.keys())
   dict_keys(['MT001'])

   Process multiple sample rates:

   >>> df = collection.to_dataframe(sample_rates=[150, 2400, 24000])
   >>> print(df.columns)
   Index(['survey', 'station', 'run', 'start', 'end', ...])

   .. rubric:: Notes

   The class automatically discovers station folders by locating
   'recmeta.json' files and organizes time series files by sample rate.

   File extensions are mapped as:

   - 30 Hz: td_30
   - 150 Hz: td_150
   - 2400 Hz: td_2400
   - 24000 Hz: td_24k
   - 96000 Hz: td_96k

   .. seealso::

      :obj:`mth5.io.Collection`
          Base collection class

      :obj:`mth5.io.phoenix.PhoenixReceiverMetadata`
          Receiver metadata handler


   .. py:attribute:: metadata_dict


   .. py:method:: to_dataframe(sample_rates: list[int] | int = [150, 24000], run_name_zeros: int = 4, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Create a DataFrame cataloging all Phoenix files in the collection.

      Scans all station folders for time series files at specified sample
      rates and creates a comprehensive inventory with metadata for each file.

      :param sample_rates: Sample rate(s) to include in Hz. Valid values are 30, 150, 2400,
                           24000, 96000. Can be a single integer or list (default is [150, 24000]).
      :type sample_rates: list[int] | int, optional
      :param run_name_zeros: Number of zeros for zero-padding run names (default is 4).
                             For example, 4 produces 'sr150_0001'.
      :type run_name_zeros: int, optional
      :param calibration_path: Path to calibration files. Currently unused but reserved for
                               future functionality.
      :type calibration_path: str | Path | None, optional

      :returns: DataFrame with one row per file containing columns:

                - survey: Survey ID from metadata
                - station: Station ID from metadata
                - run: Run ID (assigned by assign_run_names)
                - start: File start time (ISO format)
                - end: File end time (ISO format)
                - channel_id: Numeric channel identifier
                - component: Channel component name (e.g., 'Ex', 'Hy')
                - fn: Full file path
                - sample_rate: Sample rate in Hz
                - file_size: File size in bytes
                - n_samples: Number of samples in file
                - sequence_number: File sequence number for continuous data
                - instrument_id: Recording/receiver ID
                - calibration_fn: Path to calibration file (currently None)
      :rtype: pd.DataFrame

      .. rubric:: Examples

      Get DataFrame for standard sample rates:

      >>> df = collection.to_dataframe(sample_rates=[150, 24000])
      >>> print(df.shape)
      (245, 14)
      >>> print(df.station.unique())
      ['MT001']

      Process single sample rate:

      >>> df_150 = collection.to_dataframe(sample_rates=150)
      >>> print(df_150.sample_rate.unique())
      [150.]

      Check file coverage:

      >>> for comp in df.component.unique():
      ...     comp_df = df[df.component == comp]
      ...     print(f"{comp}: {len(comp_df)} files")
      Ex: 35 files
      Ey: 35 files
      Hx: 35 files

      .. rubric:: Notes

      - Calibration files (identified by 'calibration' in filename) are
        automatically skipped
      - Files that cannot be opened are logged and skipped
      - The DataFrame is sorted by station, sample_rate, and start time
      - Run names must be assigned separately using assign_run_names()

      .. seealso::

         :obj:`assign_run_names`
             Assign run identifiers based on timing

         :obj:`get_runs`
             Get organized runs directly



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 4) -> pandas.DataFrame

      Assign run names based on temporal continuity.

      Analyzes file timing to group files into runs. For continuous data
      (< 1000 Hz), maintains a single run as long as files are contiguous.
      For segmented data (≥ 1000 Hz), assigns a unique run to each segment.

      :param df: DataFrame returned by `to_dataframe` method with file inventory.
      :type df: pd.DataFrame
      :param zeros: Number of zeros for zero-padding run names (default is 4).
      :type zeros: int, optional

      :returns: DataFrame with 'run' column populated. Run names follow the
                format 'sr{rate}_{number:0{zeros}}', e.g., 'sr150_0001'.
      :rtype: pd.DataFrame

      .. rubric:: Examples

      Assign run names to a DataFrame:

      >>> df = collection.to_dataframe(sample_rates=[150, 24000])
      >>> df_with_runs = collection.assign_run_names(df, zeros=4)
      >>> print(df_with_runs.run.unique())
      ['sr150_0001', 'sr24k_0001', 'sr24k_0002', ...]

      Check for data gaps in continuous data:

      >>> df_150 = df_with_runs[df_with_runs.sample_rate == 150]
      >>> print(df_150.run.unique())
      ['sr150_0001', 'sr150_0002']  # Gap detected between runs

      Count segments in high-rate data:

      >>> df_24k = df_with_runs[df_with_runs.sample_rate == 24000]
      >>> n_segments = len(df_24k.run.unique())
      >>> print(f"Found {n_segments} segments at 24 kHz")
      Found 43 segments at 24 kHz

      .. rubric:: Notes

      **Continuous Data (< 1000 Hz):**

      - Maintains single run ID while files are temporally contiguous
      - Detects gaps by comparing end time of file N with start time of
        file N+1
      - Increments run counter when gap > 0 seconds detected

      **Segmented Data (≥ 1000 Hz):**

      - Each unique start time receives a new run ID
      - Typically results in one run per segment/file

      The run naming scheme uses the sample rate in the identifier:

      - 30 Hz → 'sr30_NNNN'
      - 150 Hz → 'sr150_NNNN'
      - 2400 Hz → 'sr2400_NNNN'
      - 24000 Hz → 'sr24k_NNNN'
      - 96000 Hz → 'sr96k_NNNN'



   .. py:method:: get_runs(sample_rates: list[int] | int, run_name_zeros: int = 4, calibration_path: str | pathlib.Path | None = None) -> collections.OrderedDict[str, collections.OrderedDict[str, pandas.DataFrame]]

      Organize Phoenix files into runs ready for reading.

      Creates a nested dictionary structure organizing files by station and
      run. For each run, returns only the first file(s) needed to initialize
      reading, as continuous readers will automatically load sequences.

      :param sample_rates: Sample rate(s) to include in Hz. Valid values are 30, 150, 2400,
                           24000, 96000. Can be a single integer or list.
      :type sample_rates: list[int] | int
      :param run_name_zeros: Number of zeros for zero-padding run names (default is 4).
      :type run_name_zeros: int, optional
      :param calibration_path: Path to calibration files. Currently unused but reserved for
                               future functionality.
      :type calibration_path: str | Path | None, optional

      :returns: Nested OrderedDict with structure:

                - Keys: station IDs
                - Values: OrderedDict of runs

                  - Keys: run IDs (e.g., 'sr150_0001')
                  - Values: DataFrame with first file(s) for each channel
      :rtype: OrderedDict[str, OrderedDict[str, pd.DataFrame]]

      .. rubric:: Examples

      Get runs for standard sample rates:

      >>> from mth5.io.phoenix import PhoenixCollection
      >>> collection = PhoenixCollection(r"/path/to/station")
      >>> runs = collection.get_runs(sample_rates=[150, 24000])
      >>> print(runs.keys())
      odict_keys(['MT001'])

      Access specific station's runs:

      >>> station_runs = runs['MT001']
      >>> print(list(station_runs.keys()))
      ['sr150_0001', 'sr24k_0001', 'sr24k_0002', ...]

      Get first file for a specific run:

      >>> run_df = runs['MT001']['sr150_0001']
      >>> print(run_df[['component', 'fn', 'start']])
        component                           fn                 start
      0        Ex  /path/to/8441_2020...td_150  2020-06-02T19:00:00
      1        Ey  /path/to/8441_2020...td_150  2020-06-02T19:00:00

      Iterate over all runs:

      >>> for station_id, station_runs in runs.items():
      ...     for run_id, run_df in station_runs.items():
      ...         print(f"{station_id}/{run_id}: {len(run_df)} channels")
      MT001/sr150_0001: 5 channels
      MT001/sr24k_0001: 5 channels

      Get single sample rate:

      >>> runs_150 = collection.get_runs(sample_rates=150)
      >>> run_ids = list(runs_150['MT001'].keys())
      >>> print([r for r in run_ids if 'sr150' in r])
      ['sr150_0001']

      .. rubric:: Notes

      **For Continuous Data (< 1000 Hz):**

      Returns only the first file in each sequence per channel. The Phoenix
      reader will automatically load the complete sequence when reading.

      **For Segmented Data (≥ 1000 Hz):**

      Returns the first file for each segment. Each segment must be read
      separately.

      **DataFrame Content:**

      Each DataFrame contains one row per channel component with the earliest
      file for that component in the run. This ensures all channels start from
      the same time.

      The method internally:

      1. Calls to_dataframe() to inventory all files
      2. Calls assign_run_names() to group files into runs
      3. Selects first file(s) for each run and component
      4. Returns organized structure for easy iteration

      .. seealso::

         :obj:`to_dataframe`
             Create complete file inventory

         :obj:`assign_run_names`
             Group files into runs

         :obj:`mth5.io.phoenix.read_phoenix`
             Read Phoenix files



