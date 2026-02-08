mth5.io.metronix.metronix_collection
====================================

.. py:module:: mth5.io.metronix.metronix_collection

.. autoapi-nested-parse::

   Metronix collection utilities for managing ATSS files.

   This module provides classes for collecting and managing Metronix ATSS
   (Audio Time Series System) files and creating pandas DataFrames with
   metadata for processing workflows.

   Classes
   -------
   MetronixCollection
       Collection class for managing Metronix ATSS files

   Created on Fri Nov 22 13:22:44 2024

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.metronix.metronix_collection.MetronixCollection


Module Contents
---------------

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



