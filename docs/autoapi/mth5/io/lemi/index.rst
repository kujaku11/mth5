mth5.io.lemi
============

.. py:module:: mth5.io.lemi


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/lemi/lemi424/index
   /autoapi/mth5/io/lemi/lemi_collection/index


Classes
-------

.. autoapisummary::

   mth5.io.lemi.LEMI424
   mth5.io.lemi.LEMICollection


Functions
---------

.. autoapisummary::

   mth5.io.lemi.read_lemi424


Package Contents
----------------

.. py:class:: LEMI424(fn: str | pathlib.Path | None = None, **kwargs: Any)

   Read and process LEMI424 magnetotelluric data files.

   This is a placeholder until IRIS finalizes their reader.

   :param fn: Full path to LEMI424 file, by default None.
   :type fn: str or pathlib.Path, optional
   :param \*\*kwargs: Additional keyword arguments for configuration.
   :type \*\*kwargs: dict

   .. attribute:: sample_rate

      Sample rate of the file, default is 1.0.

      :type: float

   .. attribute:: chunk_size

      Chunk size for pandas to use, default is 8640.

      :type: int

   .. attribute:: file_column_names

      Column names of the LEMI424 file.

      :type: list of str

   .. attribute:: dtypes

      Data types for each column.

      :type: dict

   .. attribute:: data_column_names

      Same as file_column_names with an added column for date.

      :type: list of str

   .. attribute:: data

      The loaded data.

      :type: pd.DataFrame or None

   .. rubric:: Notes

   LEMI424 File Column Names:
       year, month, day, hour, minute, second, bx, by, bz,
       temperature_e, temperature_h, e1, e2, e3, e4, battery,
       elevation, latitude, lat_hemisphere, longitude,
       lon_hemisphere, n_satellites, gps_fix, time_diff

   Data Column Names:
       date, bx, by, bz, temperature_e, temperature_h, e1, e2,
       e3, e4, battery, elevation, latitude, lat_hemisphere,
       longitude, lon_hemisphere, n_satellites, gps_fix, time_diff


   .. py:attribute:: logger


   .. py:property:: fn
      :type: pathlib.Path | None


      Full path to LEMI424 file.

      :returns: Path to the file or None if not set.
      :rtype: pathlib.Path or None


   .. py:attribute:: sample_rate
      :value: 1.0



   .. py:attribute:: chunk_size
      :value: 8640



   .. py:property:: data
      :type: pandas.DataFrame | None


      Data represented as a pandas DataFrame with data column names.

      :returns: The loaded data or None if no data is loaded.
      :rtype: pd.DataFrame or None


   .. py:attribute:: file_column_names
      :value: ['year', 'month', 'day', 'hour', 'minute', 'second', 'bx', 'by', 'bz', 'temperature_e',...



   .. py:attribute:: dtypes


   .. py:attribute:: data_column_names
      :value: ['date', 'bx', 'by', 'bz', 'temperature_e', 'temperature_h', 'e1', 'e2', 'e3', 'e4', 'battery',...



   .. py:property:: file_size
      :type: int | None


      Size of file in bytes.

      :returns: File size in bytes or None if no file is set.
      :rtype: int or None


   .. py:property:: start
      :type: mt_metadata.common.mttime.MTime | None


      Start time of data collection in the LEMI424 file.

      :returns: Start time or None if no data is loaded.
      :rtype: MTime or None


   .. py:property:: end
      :type: mt_metadata.common.mttime.MTime | None


      End time of data collection in the LEMI424 file.

      :returns: End time or None if no data is loaded.
      :rtype: MTime or None


   .. py:property:: latitude
      :type: float | None


      Median latitude where data have been collected.

      :returns: Median latitude in degrees or None if no data is loaded.
      :rtype: float or None


   .. py:property:: longitude
      :type: float | None


      Median longitude where data have been collected.

      :returns: Median longitude in degrees or None if no data is loaded.
      :rtype: float or None


   .. py:property:: elevation
      :type: float | None


      Median elevation where data have been collected.

      :returns: Median elevation in meters or None if no data is loaded.
      :rtype: float or None


   .. py:property:: n_samples
      :type: int | None


      Number of samples in the file.

      :returns: Number of samples or None if no data/file available.
      :rtype: int or None


   .. py:property:: gps_lock
      :type: Any | None


      GPS lock status array.

      :returns: GPS fix values or None if no data is loaded.
      :rtype: numpy.ndarray or None


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Station metadata as mt_metadata.timeseries.Station object.

      :returns: Station metadata object.
      :rtype: mt_metadata.timeseries.Station


   .. py:property:: run_metadata
      :type: mt_metadata.timeseries.Run


      Run metadata as mt_metadata.timeseries.Run object.

      :returns: Run metadata object.
      :rtype: mt_metadata.timeseries.Run


   .. py:method:: read(fn: str | pathlib.Path | None = None, fast: bool = True) -> None

      Read a LEMI424 file using pandas.

      The `fast` way will read in the first and last line to get the start
      and end time to make a time index. Then it will read in the data
      skipping parsing the date time columns. It will check to make sure
      the expected amount of points are correct. If not then it will read
      in the slower way which uses the date time parser to ensure any
      time gaps are respected.

      :param fn: Full path to file. Uses LEMI424.fn if not provided, by default None.
      :type fn: str, pathlib.Path, or None, optional
      :param fast: Read the fast way (True) or not (False), by default True.
      :type fast: bool, optional

      :raises IOError: If file cannot be found.



   .. py:method:: read_metadata() -> None

      Read only first and last rows to get important metadata.

      This method is used to extract essential metadata from the collection
      without loading the entire dataset.




   .. py:method:: read_calibration(fn: str | pathlib.Path) -> mt_metadata.timeseries.filters.FrequencyResponseTableFilter

      Read a LEMI424 calibration file.

      Calibration files are assumed to be JSON files with the following format:
      {
          "Calibration": {
              "gain": float,
              "Freq": [float],
              "Re": [float],
              "Im": [float]
          }
      }


      :param fn: Full path to calibration file.
      :type fn: str or pathlib.Path

      :returns: Calibration filter object.
      :rtype: mt_metadata.timeseries.filters.FrequencyResponseTableFilter



   .. py:method:: to_run_ts(fn: str | pathlib.Path | None = None, e_channels: list[str] = ['e1', 'e2'], calibration_dict: dict | None = None) -> mth5.timeseries.RunTS

      Create a RunTS object from the data.

      :param fn: Full path to file. Will use LEMI424.fn if None, by default None.
      :type fn: str, pathlib.Path, or None, optional
      :param e_channels: Column names for the electric channels to use, by default ["e1", "e2"].
      :type e_channels: list of str, optional
      :param calibration_dict: Calibration dictionary to apply to the data, by default {}.  Keys are
                               the channel names and values are the calibration file path. The file
                               path is assumed to be in the format `lemi-{component}.sr.json`.
      :type calibration_dict: dict, optional

      :returns: RunTS object containing the data.
      :rtype: mth5.timeseries.RunTS



.. py:function:: read_lemi424(fn: str | pathlib.Path | list[str | pathlib.Path], e_channels: list[str] = ['e1', 'e2'], fast: bool = True, calibration_dict: dict | None = None) -> mth5.timeseries.RunTS

   Read a LEMI 424 TXT file.

   :param fn: Input file name.
   :type fn: str or pathlib.Path
   :param e_channels: A list of electric channels to read, by default ["e1", "e2"].
   :type e_channels: list of str, optional
   :param fast: Use fast reading method, by default True.
   :type fast: bool, optional
   :param calibration_dict: Calibration dictionary to apply to the data, by default None.  Keys are
                            the channel names and values are the calibration file path.
   :type calibration_dict: dict, optional

   :returns: A RunTS object with appropriate metadata.
   :rtype: mth5.timeseries.RunTS


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



