mth5.io.usgs_ascii.usgs_ascii_collection
========================================

.. py:module:: mth5.io.usgs_ascii.usgs_ascii_collection

.. autoapi-nested-parse::

   LEMI 424 Collection
   ====================

   Collection of TXT files combined into runs

   Created on Wed Aug 31 10:32:44 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.usgs_ascii.usgs_ascii_collection.USGSasciiCollection


Module Contents
---------------

.. py:class:: USGSasciiCollection(file_path=None, **kwargs)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection of USGS ASCII files.

   .. code-block:: python

       >>> from mth5.io.usgs_ascii import USGSasciiCollection
       >>> lc = USGSasciiCollection(r"/path/to/ascii/files")
       >>> run_dict = lc.get_runs(1)




   .. py:attribute:: file_ext
      :value: 'asc'



   .. py:method:: to_dataframe(sample_rates=[4], run_name_zeros=4, calibration_path=None)

      Create a data frame of each TXT file in a given directory.

      .. note:: If a run name is already present it will not be overwritten

      :param sample_rates: sample rate to get, defaults to [4]
      :type sample_rates: int or list, optional
      :param run_name_zeros: number of zeros to assing to the run name,
       defaults to 4
      :type run_name_zeros: int, optional
      :param calibration_path: path to calibration files, defaults to None
      :type calibration_path: string or Path, optional
      :return: Dataframe with information of each TXT file in the given
       directory.
      :rtype: :class:`pandas.DataFrame`

      :Example:

          >>> from mth5.io.usgs_ascii import USGSasciiCollection
          >>> lc = USGSasciiCollection("/path/to/ascii/files")
          >>> ascii_df = lc.to_dataframe()




   .. py:method:: assign_run_names(df, zeros=4)

      Assign run names based on start and end times, checks if a file has
      the same start time as the last end time.

      Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}. Only
      if the run name is not assigned already.

      :param df: Dataframe with the appropriate columns
      :type df: :class:`pandas.DataFrame`
      :param zeros: number of zeros in run name, defaults to 4
      :type zeros: int, optional
      :return: Dataframe with run names
      :rtype: :class:`pandas.DataFrame`




