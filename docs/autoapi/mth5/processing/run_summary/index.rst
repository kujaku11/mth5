mth5.processing.run_summary
===========================

.. py:module:: mth5.processing.run_summary

.. autoapi-nested-parse::

   This module contains the RunSummary class.

   This is a helper class that summarizes the Runs in an mth5.

   TODO: This class and methods could be replaced by methods in MTH5.

   Functionality of RunSummary()
   1. User can get a list of local_station options, which correspond to unique pairs
   of values: (survey,  station)

   2. User can see all possible ways of processing the data:
   - one list per (survey,  station) pair in the run_summary

   Some of the following functionalities may end up in KernelDataset:
   3. User can select local_station
   -this can trigger a reduction of runs to only those that are from the local staion
   and simultaneous runs at other stations
   4. Given a local station, a list of possible reference stations can be generated
   5. Given a remote reference station, a list of all relevent runs, truncated to
   maximize coverage of the local station runs is generated
   6. Given such a "restricted run list", runs can be dropped
   7. Time interval endpoints can be changed


   Development Notes:
       TODO: consider adding methods:
        - drop_runs_shorter_than": removes short runs from summary
        - fill_gaps_by_time_interval": allows runs to be merged if gaps between
          are short
        - fill_gaps_by_run_names": allows runs to be merged if gaps between are
          short
       TODO: Consider whether this should return a copy or modify in-place when
       querying the df.



Classes
-------

.. autoapisummary::

   mth5.processing.run_summary.RunSummary


Functions
---------

.. autoapisummary::

   mth5.processing.run_summary.extract_run_summaries_from_mth5s


Module Contents
---------------

.. py:class:: RunSummary(input_dict: Optional[Union[dict, None]] = None, df: Optional[Union[pandas.DataFrame, None]] = None)

   Class to contain a run-summary table from one or more mth5s.

   WIP: For the full MMT case this may need modification to a channel based
   summary.


   .. py:attribute:: column_dtypes


   .. py:property:: df
      :type: pandas.DataFrame


      Df function.


   .. py:method:: clone()

      2022-10-20:
      Cloning may be causing issues with extra instances of open h5 files ...



   .. py:method:: from_mth5s(mth5_list) -> list

      Iterates over mth5s in list and creates one big dataframe
      summarizing the runs



   .. py:property:: mini_summary
      :type: pandas.DataFrame


      Shows the dataframe with only a few columns for readbility.


   .. py:property:: print_mini_summary
      :type: str


      Calls minisummary through logger so it is formatted.


   .. py:method:: drop_no_data_rows() -> bool

      Drops rows marked `has_data` = False and resets the index of self.df.



   .. py:method:: set_sample_rate(sample_rate: float, inplace: bool = False)

      Set the sample rate so that the run summary represents all runs for
      a single sample rate.

      :param sample_rate:
      :type sample_rate: float
      :param inplace: DESCRIPTION. By default, False.
      :type inplace: bool, optional

      :returns: DESCRIPTION.
      :rtype: TYPE



.. py:function:: extract_run_summaries_from_mth5s(mth5_list, summary_type='run', deduplicate=True)

   Given a list of mth5's, iterate over them, extracting run_summaries and
   merging into one big table.

   Development Notes:
   ToDo: Move this method into mth5? or mth5_helpers?
   ToDo: Make this a class so that the __repr__ is a nice visual representation
   of the
   df, like what channel summary does in mth5
   - 2022-05-28 Modified to allow this method to accept mth5 objects as well
   as the
   already supported types of pathlib.Path or str


   In order to drop duplicates I used the solution here:
   https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists

   :param deduplicate: By default, True.
   :param mth5_list:
   :param mth5_paths: Paths or strings that point to mth5s.
   :type mth5_paths: list
   :param summary_type: One of ["channel", "run"]
                        "channel" returns concatenated channel summary,
                        "run" returns concatenated run summary,. By default, "run".
   :type summary_type: string, optional
   :param deduplicate:
   :type deduplicate: , defaults to True. : bool, optional

   :returns: **super_summary** -- Given a list of mth5s, a dataframe of all available runs.
   :rtype: pd.DataFrame


