mth5.io.phoenix.phoenix_collection
==================================

.. py:module:: mth5.io.phoenix.phoenix_collection

.. autoapi-nested-parse::

   Phoenix file collection module for organizing and processing Phoenix MTU data files.

   This module provides the PhoenixCollection class for discovering, organizing,
   and managing Phoenix magnetotelluric receiver files within a directory structure.

   Created on Thu Aug  4 16:48:47 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.phoenix_collection.PhoenixCollection


Module Contents
---------------

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



