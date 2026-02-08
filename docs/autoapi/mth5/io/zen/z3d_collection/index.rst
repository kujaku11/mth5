mth5.io.zen.z3d_collection
==========================

.. py:module:: mth5.io.zen.z3d_collection

.. autoapi-nested-parse::

   Z3DCollection
   =================

   An object to hold Z3D file information to make processing easier.


   Created on Sat Apr  4 12:40:40 2020

   @author: peacock



Classes
-------

.. autoapisummary::

   mth5.io.zen.z3d_collection.Z3DCollection


Module Contents
---------------

.. py:class:: Z3DCollection(file_path: str | pathlib.Path | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection manager for Z3D file operations and metadata processing.

   This class provides functionality to handle collections of Z3D files,
   including metadata extraction, station information management, and
   dataframe creation for analysis workflows.

   :param file_path: Path to directory containing Z3D files, by default None
   :type file_path: str or Path, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Collection class
   :type \*\*kwargs: dict

   .. attribute:: station_metadata_dict

      Dictionary mapping station IDs to Station metadata objects

      :type: dict[str, Station]

   .. attribute:: file_ext

      File extension for Z3D files ("z3d")

      :type: str

   .. rubric:: Examples

   >>> zc = Z3DCollection("/path/to/z3d/files")
   >>> df = zc.to_dataframe(sample_rates=[256, 4096])
   >>> print(df.head())


   .. py:attribute:: station_metadata_dict
      :type:  dict[str, mt_metadata.timeseries.Station]


   .. py:attribute:: file_ext
      :type:  str
      :value: 'z3d'



   .. py:method:: get_calibrations(antenna_calibration_file: str | pathlib.Path) -> mth5.io.zen.coil_response.CoilResponse

      Load coil calibration data from antenna calibration file.

      :param antenna_calibration_file: Path to the antenna.cal file containing coil calibration data
      :type antenna_calibration_file: str or Path

      :returns: CoilResponse object containing calibration information for
                various coil serial numbers
      :rtype: CoilResponse

      .. rubric:: Examples

      >>> zc = Z3DCollection("/path/to/z3d/files")
      >>> cal_obj = zc.get_calibrations("/path/to/antenna.cal")
      >>> print(cal_obj.has_coil_number("2324"))



   .. py:method:: to_dataframe(sample_rates: list[int] = [256, 4096], run_name_zeros: int = 4, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Extract Z3D file information and create analysis-ready dataframe.

      Processes all Z3D files in the collection, extracting metadata and
      file information to create a comprehensive dataframe suitable for
      magnetotelluric data analysis workflows.

      :param sample_rates: Allowed sampling rates in Hz. Files with sample rates not in
                           this list will trigger a warning and early return
      :type sample_rates: list of int, default [256, 4096]
      :param run_name_zeros: Number of zero-padding digits for run names in dataframe sorting
      :type run_name_zeros: int, default 4
      :param calibration_path: Path to antenna calibration file. If None, calibration information
                               will not be included, by default None
      :type calibration_path: str or Path, optional

      :returns: Dataframe containing Z3D file information with columns:
                - survey: Survey/job name from Z3D metadata
                - station: Station identifier
                - run: Automatically assigned run names based on start times
                - start/end: ISO format timestamps for data recording period
                - channel_id: Channel number from Z3D file
                - component: Measurement component (ex, ey, hx, hy, hz)
                - fn: Path to Z3D file
                - sample_rate: Sampling frequency in Hz
                - file_size: Size of Z3D file in bytes
                - n_samples: Number of data samples in file
                - sequence_number: Sequential numbering within station
                - dipole: Dipole length in meters (for electric channels)
                - coil_number: Coil serial number (for magnetic channels)
                - latitude/longitude/elevation: Station coordinates
                - instrument_id: ZEN box identifier
                - calibration_fn: Path to calibration file if available
      :rtype: pd.DataFrame

      :raises AttributeError: If Z3D files contain invalid or missing required metadata
      :raises FileNotFoundError: If calibration_path is specified but file doesn't exist

      .. rubric:: Examples

      >>> zc = Z3DCollection("/path/to/z3d/files")
      >>> df = zc.to_dataframe(sample_rates=[256, 4096],
      ...                      calibration_path="/path/to/antenna.cal")
      >>> print(df[['station', 'component', 'sample_rate']].head())
      >>> df.to_csv("/path/output/z3d_inventory.csv")

      .. rubric:: Notes

      This method also populates the `station_metadata_dict` attribute
      with consolidated station metadata derived from all processed files.



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 3) -> pandas.DataFrame

      Assign standardized run names to dataframe based on start times.

      Creates run names using the pattern 'sr{sample_rate}_{block_number}'
      where block_number is assigned sequentially based on unique start
      times within each station.

      :param df: Input dataframe containing Z3D file information with at least
                 'station', 'start', and 'sample_rate' columns
      :type df: pd.DataFrame
      :param zeros: Number of zero-padding digits for block numbers in run names
      :type zeros: int, default 3

      :returns: Modified dataframe with updated 'run' and 'sequence_number'
                columns assigned based on temporal ordering within each station
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> zc = Z3DCollection()
      >>> df = pd.DataFrame({
      ...     'station': ['001', '001', '002'],
      ...     'start': ['2022-01-01T10:00:00', '2022-01-01T12:00:00', '2022-01-01T10:00:00'],
      ...     'sample_rate': [256, 256, 4096]
      ... })
      >>> df_with_runs = zc.assign_run_names(df, zeros=3)
      >>> print(df_with_runs['run'].tolist())
      ['sr256_001', 'sr256_002', 'sr4096_001']

      .. rubric:: Notes

      This method modifies the input dataframe in-place by updating the
      'run' and 'sequence_number' columns. Start times are used to
      determine temporal ordering within each station.



