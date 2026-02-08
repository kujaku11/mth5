mth5.io.lemi.lemi_collection
============================

.. py:module:: mth5.io.lemi.lemi_collection

.. autoapi-nested-parse::

   LEMI 424 Collection
   ====================

   Collection of TXT files combined into runs

   Created on Wed Aug 31 10:32:44 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.lemi.lemi_collection.LEMICollection


Module Contents
---------------

.. py:class:: LEMICollection(file_path: str | pathlib.Path | None = None, file_ext: List[str] | None = None, **kwargs)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection of LEMI 424 files into runs based on start and end times.

   Will assign the run name as 'sr1_{index:0{zeros}}' --> 'sr1_0001' for
   `zeros` = 4.

   .. rubric:: Notes

   This class assumes that the given file path contains a single
   LEMI station. If you want to do multiple stations merge the returned
   data frames.

   LEMI data comes with little metadata about the station or survey,
   therefore you should assign `station_id` and `survey_id`.

   :param file_path: Full path to single station LEMI424 directory, by default None
   :type file_path: str or pathlib.Path, optional
   :param file_ext: Extension of LEMI424 files, by default ["txt", "TXT"]
   :type file_ext: list of str, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Collection class

   .. attribute:: station_id

      Station identification string, defaults to "mt001"

      :type: str

   .. attribute:: survey_id

      Survey identification string, defaults to "mt"

      :type: str

   .. rubric:: Examples

   >>> from mth5.io.lemi import LEMICollection
   >>> lc = LEMICollection(r"/path/to/single/lemi/station")
   >>> lc.station_id = "mt001"
   >>> lc.survey_id = "test_survey"
   >>> run_dict = lc.get_runs(1)


   .. py:attribute:: station_id
      :value: 'mt001'



   .. py:attribute:: survey_id
      :value: 'mt'



   .. py:attribute:: calibration_dict


   .. py:method:: get_calibrations(calibration_path: str | pathlib.Path) -> dict

      Get calibration dictionary for LEMI424 files.  This assumes that the
      calibrations files are in JSON format and named as
      'LEMI-424-<component>.json'

      :param calibration_path: Path to calibration files
      :type calibration_path: str or pathlib.Path

      :returns: Calibration dictionary for LEMI424 files
      :rtype: dict

      .. rubric:: Examples

      >>> from mth5.io.lemi import LEMICollection
      >>> lc = LEMICollection("/path/to/single/lemi/station")
      >>> cal_dict = lc.get_calibrations(Path("/path/to/calibrations"))



   .. py:method:: to_dataframe(sample_rates: int | List[int] | None = None, run_name_zeros: int = 4, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Create a data frame of each TXT file in a given directory.

      .. rubric:: Notes

      This assumes the given directory contains a single station

      :param sample_rates: Sample rate to get, will always be 1 for LEMI data, by default [1]
      :type sample_rates: int or list of int, optional
      :param run_name_zeros: Number of zeros to assign to the run name, by default 4
      :type run_name_zeros: int, optional
      :param calibration_path: Path to calibration files, by default None
      :type calibration_path: str or pathlib.Path, optional

      :returns: DataFrame with information of each TXT file in the given directory
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> from mth5.io.lemi import LEMICollection
      >>> lc = LEMICollection("/path/to/single/lemi/station")
      >>> lemi_df = lc.to_dataframe()



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 4) -> pandas.DataFrame

      Assign run names based on start and end times.

      Checks if a file has the same start time as the last end time.
      Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

      :param df: DataFrame with the appropriate columns
      :type df: pd.DataFrame
      :param zeros: Number of zeros in run name, by default 4
      :type zeros: int, optional

      :returns: DataFrame with run names assigned
      :rtype: pd.DataFrame



