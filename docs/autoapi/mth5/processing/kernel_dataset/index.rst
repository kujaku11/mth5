mth5.processing.kernel_dataset
==============================

.. py:module:: mth5.processing.kernel_dataset

.. autoapi-nested-parse::

   Magnetotelluric kernel dataset processing module.

   This module contains a class for representing a dataset that can be processed.

   The module provides functionality for:
   - Managing magnetotelluric time series intervals
   - Supporting single station and remote reference processing
   - Handling run combination and time interval restrictions
   - Interfacing with MTH5 data structures

   Development Notes
   -----------------
   Players on the stage: One or more mth5s.

   Each mth5 has a "run_summary" dataframe available. Run_summary provides options for
   the local and possibly remote reference stations. Candidates for local station are
   the unique values in the station column.

   For any candidate station, there are some integer n runs available.
   This yields 2^n - 1 possible combinations that can be processed, neglecting any
   flagging of time intervals within any run, or any joining of runs.
   (There are actually 2**n, but we ignore the empty set, so -1)

   Intuition suggests default ought to be to process n runs in n+1 configurations:
   {all runs} + each run individually. This will give a bulk answer, and bad runs can
   be flagged by comparing them. After an initial processing, the tfs can be reviewed
   and the problematic runs can be addressed.

   The user can interact with the run_summary_df, selecting sub dataframes via querying,
   and in future maybe via some GUI (or a spreadsheet).

   The intended usage process is as follows:
    0. Start with a list of mth5s
    1. Extract a run_summary
    2. Stare at the run_summary_df, and select a station "S" to process
    3. Select a non-empty set of runs for station "S"
    4. Select a remote reference "RR", (this is allowed to be None)
    5. Extract the sub-dataframe corresponding to acquisition_runs from "S" and "RR"
    7. If the remote is not None:
     - Drop the runs (rows) associated with RR that do not intersect with S
     - Restrict start/end times of RR runs that intersect with S so overlap is complete.
     - Restrict start/end times of S runs so that they intersect with remote
    8. This is now a TFKernel Dataset Definition (ish). Initialize a default processing
    object and pass it this df.

   .. rubric:: Examples

   >>> cc = ConfigCreator()
   >>> p = cc.create_from_kernel_dataset(kernel_dataset)
   - Optionally pass emtf_band_file=emtf_band_setup_file
    9. Edit the Processing Config appropriately,

   TODO: Consider supporting a default value for 'channel_scale_factors' that is None,
   TODO: Might need to groupby survey & station, for now consider station_id unique.



Classes
-------

.. autoapisummary::

   mth5.processing.kernel_dataset.KernelDataset


Functions
---------

.. autoapisummary::

   mth5.processing.kernel_dataset.restrict_to_station_list
   mth5.processing.kernel_dataset.intervals_overlap
   mth5.processing.kernel_dataset.overlap


Module Contents
---------------

.. py:class:: KernelDataset(df: pandas.DataFrame | None = None, local_station_id: str = '', remote_station_id: str | None = None, **kwargs: Any)

   Magnetotelluric kernel dataset for time series processing.

   This class works with mth5-derived channel_summary or run_summary dataframes
   that specify time series intervals. It manages acquisition "runs" that can be
   merged into processing runs, with support for both single station and remote
   reference processing configurations.

   :param df: Pre-formed dataframe with dataset configuration. Normally built from a
              run_summary, by default None
   :type df: pd.DataFrame | None, optional
   :param local_station_id: Local station identifier for the dataset. Normally passed via
                            from_run_summary method, by default ""
   :type local_station_id: str, optional
   :param remote_station_id: Remote reference station identifier. Normally passed via from_run_summary
                             method, by default None
   :type remote_station_id: str | None, optional
   :param \*\*kwargs: Additional keyword arguments to set as attributes
   :type \*\*kwargs: dict

   .. attribute:: df

      Main dataset dataframe with time series intervals

      :type: pd.DataFrame | None

   .. attribute:: local_station_id

      Local station identifier

      :type: str | None

   .. attribute:: remote_station_id

      Remote reference station identifier

      :type: str | None

   .. attribute:: survey_metadata

      Survey metadata container

      :type: dict

   .. attribute:: initialized

      Whether MTH5 objects have been initialized

      :type: bool

   .. attribute:: local_mth5_obj

      Local station MTH5 object

      :type: Any | None

   .. attribute:: remote_mth5_obj

      Remote station MTH5 object

      :type: Any | None

   .. rubric:: Notes

   The class is closely related to (may actually be an extension of) RunSummary.
   The main idea is to specify one or two stations, and a list of acquisition "runs"
   that can be merged into a "processing run". Each acquisition run can be further
   divided into non-overlapping chunks by specifying time-intervals associated with
   that acquisition run.

   The time intervals can be used for several purposes but primarily:
   - STFT processing for merged FC data structures
   - Binding together into xarray time series for gap filling
   - Managing and analyzing availability of reference time series

   .. rubric:: Examples

   Create a kernel dataset from run summary:

   >>> from mth5.processing.run_summary import RunSummary
   >>> run_summary = RunSummary()
   >>> dataset = KernelDataset()
   >>> dataset.from_run_summary(run_summary, "station01", "station02")

   Process single station data:

   >>> single_dataset = KernelDataset()
   >>> single_dataset.from_run_summary(run_summary, "station01")

   .. seealso::

      :obj:`RunSummary`
          Data summary for processing configuration


   .. py:property:: df
      :type: pandas.DataFrame | None


      Main dataset dataframe.

      :returns: Dataset dataframe with time series intervals, or None if not set
      :rtype: pd.DataFrame | None


   .. py:property:: local_station_id
      :type: str | None


      Local station identifier.

      :returns: Local station identifier
      :rtype: str | None


   .. py:property:: remote_station_id
      :type: str | None


      Remote reference station identifier.

      :returns: Remote station identifier
      :rtype: str | None


   .. py:attribute:: survey_metadata
      :type:  dict[str, Any]


   .. py:attribute:: initialized
      :type:  bool
      :value: False



   .. py:attribute:: local_mth5_obj
      :type:  Any
      :value: None



   .. py:attribute:: remote_mth5_obj
      :type:  Any
      :value: None



   .. py:method:: clone() -> KernelDataset

      Create a deep copy of the dataset.

      :returns: Deep copy of this instance
      :rtype: KernelDataset



   .. py:method:: clone_dataframe() -> pandas.DataFrame | None

      Create a deep copy of the dataframe.

      :returns: Deep copy of the dataframe, or None if dataframe is not set
      :rtype: pd.DataFrame | None



   .. py:property:: local_mth5_path
      :type: pathlib.Path | None


      Local station MTH5 file path.

      :returns: Path to local station MTH5 file, extracted from dataframe or
                stored path, or None if not available
      :rtype: Path | None


   .. py:method:: has_local_mth5() -> bool

      Check if local MTH5 file exists.

      :returns: True if local MTH5 file exists on filesystem
      :rtype: bool



   .. py:property:: remote_mth5_path
      :type: pathlib.Path


      Remote mth5 path.
      :return: Remote station MTH5 path, a property extracted from the dataframe.
      :rtype: Path


   .. py:method:: has_remote_mth5() -> bool

      Test if remote mth5 exists.



   .. py:property:: processing_id
      :type: str


      Its difficult to come up with unique ids without crazy long names
      so this is a generic id of local-remote, the station metadata
      will have run information and the config parameters.


   .. py:property:: input_channels
      :type: list[str]


      Get input channels from dataframe.

      :returns: Input channel identifiers (sources)
      :rtype: list[str]

      :raises AttributeError: If dataframe is not available or local_df has no input_channels


   .. py:property:: output_channels
      :type: list[str]


      Get output channels from dataframe.

      :returns: Output channel identifiers
      :rtype: list[str]

      :raises AttributeError: If dataframe is not available or local_df has no output_channels


   .. py:property:: remote_channels
      :type: list[str]


      Get remote reference channels from dataframe.

      :returns: Remote reference channel identifiers
      :rtype: list[str]

      :raises AttributeError: If dataframe is not available or remote_df has no remote_channels


   .. py:property:: local_df
      :type: pandas.DataFrame | None


      Get dataframe subset for local station runs.

      :returns: Local station runs data, or None if dataframe not available
      :rtype: pd.DataFrame | None


   .. py:property:: remote_df
      :type: pandas.DataFrame | None


      Get dataframe subset for remote station runs.

      :returns: Remote station runs data, or None if dataframe not available
                or no remote station configured
      :rtype: pd.DataFrame | None


   .. py:method:: set_path(value: str | pathlib.Path | None) -> pathlib.Path | None
      :classmethod:


      Set and validate a file path.

      :param value: Path value to set and validate
      :type value: str | Path | None

      :returns: Validated Path object, or None if input is None
      :rtype: Path | None

      :raises IOError: If path does not exist on filesystem
      :raises ValueError: If value cannot be converted to Path



   .. py:method:: from_run_summary(run_summary: mth5.processing.run_summary.RunSummary, local_station_id: str | None = None, remote_station_id: str | None = None, sample_rate: float | int | None = None) -> None

      Initialize the dataframe from a run summary.

      :param run_summary: Summary of available data for processing from one or more stations
      :type run_summary: RunSummary
      :param local_station_id: Label of the station for which an estimate will be computed,
                               by default None
      :type local_station_id: str | None, optional
      :param remote_station_id: Label of the remote reference station, by default None
      :type remote_station_id: str | None, optional
      :param sample_rate: Sample rate to filter data by, by default None
      :type sample_rate: float | int | None, optional

      :raises ValueError: If restricting to specified stations yields empty dataset or
          if local and remote stations do not overlap for remote reference



   .. py:method:: get_metadata_from_df(df: pandas.DataFrame) -> mt_metadata.timeseries.Survey

      Extract metadata from the dataframe.  The data frame should only include one
      station.  So use self.local_df or self.remote_df. (Run Summary)

      :param df: Dataframe to extract metadata from
      :type df: pd.DataFrame

      :returns: Dictionary containing survey metadata
      :rtype: dict[str, Any]



   .. py:property:: mini_summary
      :type: pandas.DataFrame


      Return a dataframe that fits in terminal display.

      :returns: Subset of the main dataframe with key columns for summary display
      :rtype: pd.DataFrame


   .. py:property:: local_survey_id
      :type: str


      Return string label for local survey id.

      :returns: Survey ID for the local station
      :rtype: str


   .. py:property:: local_survey_metadata
      :type: mt_metadata.timeseries.Survey


      Return survey metadata for local station.


   .. py:method:: drop_runs_shorter_than(minimum_duration: float, units: str = 's', inplace: bool = True) -> pandas.DataFrame | None

      Drop runs from dataframe that are shorter than minimum duration.

      :param minimum_duration: The minimum allowed duration for a run (in units of units)
      :type minimum_duration: float
      :param units: Time units, by default "s". Currently only seconds are supported
      :type units: str, optional
      :param inplace: Whether to modify dataframe in place, by default True
      :type inplace: bool, optional

      :returns: Modified dataframe if inplace=False, None if inplace=True
      :rtype: pd.DataFrame | None

      :raises NotImplementedError: If units other than seconds are specified

      .. rubric:: Notes

      This method needs to have duration refreshed beforehand.



   .. py:method:: select_station_runs(station_runs_dict: dict, keep_or_drop: bool, inplace: bool = True) -> pandas.DataFrame | None

      Partition dataframe based on station_runs_dict and return one partition.

      :param station_runs_dict: Keys are string IDs of stations to keep/drop.
                                Values are lists of string labels for run_ids to keep/drop.
                                Example: {"mt01": ["0001", "0003"]}
      :type station_runs_dict: dict
      :param keep_or_drop: If True: returns df with only the station-runs specified
                           If False: returns df with station_runs_dict entries removed
      :type keep_or_drop: bool
      :param inplace: If True, modifies dataframe in place, by default True
      :type inplace: bool, optional

      :returns: Modified dataframe if inplace=False, None if inplace=True
      :rtype: pd.DataFrame | None



   .. py:method:: set_run_times(run_time_dict: dict, inplace: bool = True)

      Set run times from a dictionary.

      :param run_time_dict: Dictionary formatted as {run_id: {start, end}}
      :type run_time_dict: dict
      :param inplace: Whether to modify dataframe in place, by default True
      :type inplace: bool, optional

      :returns: Modified dataframe if inplace=False, None if inplace=True
      :rtype: pd.DataFrame | None



   .. py:property:: is_single_station
      :type: bool


      Returns True if no RR station.


   .. py:method:: restrict_run_intervals_to_simultaneous(df: pandas.DataFrame) -> None

      For each run in local_station_id check if it has overlap with other runs

      There is room for optimization here

      Note that you can wind up splitting runs here.  For example, in that case where
      local is running continuously, but remote is intermittent.  Then the local
      run may break into several chunks.
      :rtype: None



   .. py:method:: get_station_metadata(local_station_id: str) -> mt_metadata.timeseries.Station

      Returns the station metadata.

      Development Notes:
      TODO: This appears to be unused.  Was probably a precursor to the
        update_survey_metadata() method. Delete if unused. If used fill out doc:
      "Helper function for archiving the TF -- returns an object we can use to populate
      station metadata in the _____"
      :param local_station_id: The name of the local station.
      :type local_station_id: str
      :rtype: mt_metadata.timeseries.Station



   .. py:method:: get_run_object(index_or_row: int | pandas.Series) -> mt_metadata.timeseries.Run

      Get the run object associated with a row of the dataframe.

      :param index_or_row: Row index or row Series from the dataframe
      :type index_or_row: int | pd.Series

      :returns: The run object associated with the row
      :rtype: mt_metadata.timeseries.Run

      .. rubric:: Notes

      This method may be deprecated in favor of direct calls to
      run_obj = row.mth5_obj.from_reference(row.run_hdf5_reference) in pipelines.



   .. py:property:: num_sample_rates
      :type: int


      Returns the number of unique sample rates in the dataframe.


   .. py:property:: sample_rate
      :type: float


      Returns the sample rate that of the data in the dataframe.


   .. py:method:: update_survey_metadata(i: int, row: pandas.Series, run_ts: mth5.timeseries.run_ts.RunTS) -> None

      Wrangle survey_metadata into kernel_dataset.

      Development Notes:
      - The survey metadata needs to be passed to TF before exporting data.
      - This was factored out of initialize_dataframe_for_processing
      - TODO: It looks like we don't need to pass the whole run_ts, just its metadata
         There may be some performance implications to passing the whole object.
         Consider passing run_ts.survey_metadata, run_ts.run_metadata,
         run_ts.station_metadata only
      :param i: This would be the index of row, if we were sure that the dataframe was cleanly indexed.
      :type i: int
      :param row:
      :type row: pd.Series
      :param run_ts: Mth5 object having the survey_metadata.
      :type run_ts: mth5.timeseries.run_ts.RunTS
      :rtype: None



   .. py:property:: mth5_objs

      Mth5 objs.
      :return: Dictionary [station_id: mth5_obj].
      :rtype: dict


   .. py:method:: initialize_mth5s(mode: str = 'r')

      Return a dictionary of open mth5 objects, keyed by station_id.

      :param mode: File opening mode, by default "r" (read-only)
      :type mode: str, optional

      :returns: Dictionary keyed by station IDs containing MTH5 objects:
                - local station id: mth5.mth5.MTH5
                - remote station id: mth5.mth5.MTH5 (if present)
      :rtype: dict

      .. rubric:: Notes

      Future versions for multiple station processing may need
      nested dict structure with [survey_id][station].



   .. py:method:: initialize_dataframe_for_processing() -> None

      Adds extra columns needed for processing to the dataframe.

      Populates them with mth5 objects, run_hdf5_reference, and xr.Datasets.

      Development Notes:
      Note #1: When assigning xarrays to dataframe cells, df dislikes xr.Dataset,
      so we convert to xr.DataArray before packing df

      Note #2: [OPTIMIZATION] By accessing the run_ts and packing the "run_dataarray" column of the df, we
       perform a non-lazy operation, and essentially forcing the entire decimation_level=0 dataset to be
       loaded into memory.  Seeking a lazy method to handle this maybe worthwhile.  For example, using
       a df.apply() approach to initialize only one row at a time would allow us to generate the FCs one
       row at a time and never ingest more than one run of data at a time ...

      Note #3: Uncommenting the continue statement here is desireable, will speed things up, but
       is not yet tested.  A nice test would be to have two stations, some runs having FCs built
       and others not having FCs built.  What goes wrong is in update_survey_metadata.
       Need a way to get the survey metadata from a run, not a run_ts if possible



   .. py:method:: add_columns_for_processing() -> None

      Add columns to the dataframe used during processing.

      Development Notes:
      - This was originally in pipelines.
      - Q: Should mth5_objs be keyed by survey-station?
      - A: Yes, and ...
      since the KernelDataset dataframe will be iterated over, should probably
      write an iterator method.  This can iterate over survey-station tuples
      for multiple station processing.
      - Currently the model of keeping all these data objects "live" in the df
      seems to work OK, but is not well suited to HPC or lazy processing.
      :param mth5_objs: Keys are station_id, values are MTH5 objects.
      :type mth5_objs: dict,



   .. py:method:: close_mth5s() -> None

      Loop over all unique mth5_objs in dataset df and make sure they are closed.+.



.. py:function:: restrict_to_station_list(df: pandas.DataFrame, station_ids: str | list[str], inplace: bool = True) -> pandas.DataFrame

   Drop all rows where station_ids are NOT in the provided list.

   Operates on a deepcopy of dataframe if inplace=False.

   :param df: A run summary dataframe
   :type df: pd.DataFrame
   :param station_ids: Station ids to keep, normally local and remote
   :type station_ids: str | list[str]
   :param inplace: If True, modifies dataframe in place, by default True
   :type inplace: bool, optional

   :returns: Filtered dataframe with only specified stations
   :rtype: pd.DataFrame


.. py:function:: intervals_overlap(start1: pandas.Timestamp, end1: pandas.Timestamp, start2: pandas.Timestamp, end2: pandas.Timestamp) -> bool

   Checks if intervals 1, and 2 overlap.

   Interval 1 is (start1, end1), Interval 2 is (start2, end2),

   Development Notes:
   This may work vectorized out of the box but has not been tested.
   Also, it is intended to work with pd.Timestamp objects, but should work
   for many objects that have an ordering associated.
   This website was used as a reference when writing the method:
   https://stackoverflow.com/questions/3721249/python-date-interval-intersection
   :param start1: Start of interval 1.
   :type start1: pd.Timestamp
   :param end1: End of interval 1.
   :type end1: pd.Timestamp
   :param start2: Start of interval 2.
   :type start2: pd.Timestamp
   :param end2: End of interval 2.
   :type end2: pd.Timestamp
   :return cond: True of the intervals overlap, False if they do now.
   :rtype cond: bool


.. py:function:: overlap(t1_start: pandas.Timestamp, t1_end: pandas.Timestamp, t2_start: pandas.Timestamp, t2_end: pandas.Timestamp) -> tuple

   Get the start and end times of the overlap between two intervals.

   Interval 1 is (start1, end1), Interval 2 is (start2, end2),

   Development Notes:
    Possibly some nicer syntax in this discussion:
    https://stackoverflow.com/questions/3721249/python-date-interval-intersection
    - Intended to work with pd.Timestamp objects, but should work for many objects
     that have an ordering associated.
   :param t1_start: The start of interval 1.
   :type t1_start: pd.Timestamp
   :param t1_end: The end of interval 1.
   :type t1_end: pd.Timestamp
   :param t2_start: The start of interval 2.
   :type t2_start: pd.Timestamp
   :param t2_end: The end of interval 2.
   :type t2_end: pd.Timestamp
   :return start, end: Start, end are either same type as input, or they are None,None.
   :rtype start, end: tuple


