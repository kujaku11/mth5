mth5.processing
===============

.. py:module:: mth5.processing


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/processing/kernel_dataset/index
   /autoapi/mth5/processing/run_summary/index
   /autoapi/mth5/processing/spectre/index


Attributes
----------

.. autoapisummary::

   mth5.processing.RUN_SUMMARY_LIST
   mth5.processing.RUN_SUMMARY_COLUMNS
   mth5.processing.RUN_SUMMARY_DTYPE
   mth5.processing.ADDED_KERNEL_DATASET_DTYPE
   mth5.processing.ADDED_KERNEL_DATASET_COLUMNS
   mth5.processing.KERNEL_DATASET_DTYPE
   mth5.processing.KERNEL_DATASET_COLUMNS
   mth5.processing.MINI_SUMMARY_COLUMNS


Classes
-------

.. autoapisummary::

   mth5.processing.RunSummary


Package Contents
----------------

.. py:data:: RUN_SUMMARY_LIST

.. py:data:: RUN_SUMMARY_COLUMNS

.. py:data:: RUN_SUMMARY_DTYPE

.. py:data:: ADDED_KERNEL_DATASET_DTYPE

.. py:data:: ADDED_KERNEL_DATASET_COLUMNS

.. py:data:: KERNEL_DATASET_DTYPE

.. py:data:: KERNEL_DATASET_COLUMNS

.. py:data:: MINI_SUMMARY_COLUMNS
   :value: ['survey', 'station', 'run', 'start', 'end', 'duration']


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



