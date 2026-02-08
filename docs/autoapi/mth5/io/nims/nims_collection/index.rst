mth5.io.nims.nims_collection
============================

.. py:module:: mth5.io.nims.nims_collection

.. autoapi-nested-parse::

   NIMS Collection
   ===============

   Collection of NIMS binary files combined into runs for magnetotelluric data processing.

   Created on Wed Aug 31 10:32:44 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.nims.nims_collection.NIMSCollection


Module Contents
---------------

.. py:class:: NIMSCollection(file_path: str | pathlib.Path | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection of NIMS binary files into runs.

   This class provides functionality for organizing and processing multiple NIMS
   binary files into a structured format for magnetotelluric data analysis.

   :param file_path: Path to the directory containing NIMS binary files.
   :type file_path: str | Path | None, optional
   :param \*\*kwargs: Additional keyword arguments passed to the parent Collection class.
   :type \*\*kwargs: dict

   .. attribute:: file_ext

      File extension for NIMS binary files ('bin').

      :type: str

   .. attribute:: survey_id

      Survey identifier, defaults to 'mt'.

      :type: str

   .. rubric:: Examples

   >>> from mth5.io.nims import NIMSCollection
   >>> nc = NIMSCollection(r"/path/to/nims/station")
   >>> nc.survey_id = "mt001"
   >>> df = nc.to_dataframe()

   .. seealso::

      :obj:`mth5.io.collection.Collection`
          Base collection class

      :obj:`mth5.io.nims.NIMS`
          NIMS file reader


   .. py:attribute:: file_ext
      :type:  str
      :value: 'bin'



   .. py:attribute:: survey_id
      :type:  str
      :value: 'mt'



   .. py:method:: to_dataframe(sample_rates: int | list[int] = [1], run_name_zeros: int = 2, calibration_path: str | pathlib.Path | None = None) -> pandas.DataFrame

      Create a DataFrame of each NIMS binary file in the collection directory.

      This method processes all NIMS binary files in the specified directory and
      extracts metadata to create a structured DataFrame suitable for further
      magnetotelluric data processing.

      :param sample_rates: Sample rates to include in the DataFrame. Note that for NIMS data,
                           this parameter is present for interface consistency but all files
                           will be processed regardless of their sample rate.
      :type sample_rates: int | list[int], default [1]
      :param run_name_zeros: Number of zeros to use when formatting run names in the output.
      :type run_name_zeros: int, default 2
      :param calibration_path: Path to calibration files. Currently not used in NIMS processing
                               but included for interface consistency.
      :type calibration_path: str | Path | None, optional

      :returns: DataFrame containing metadata for each NIMS file with columns:
                - survey : Survey identifier
                - station : Station name from NIMS file
                - run : Run identifier from NIMS file
                - start : Start time in ISO format
                - end : End time in ISO format
                - fn : File path
                - sample_rate : Sampling rate
                - file_size : File size in bytes
                - n_samples : Number of samples
                - dipole : Electric dipole lengths [Ex, Ey]
                - channel_id : Channel identifier (always 1)
                - sequence_number : Sequence number (always 0)
                - component : Comma-separated component list
                - instrument_id : Instrument identifier (always 'NIMS')
      :rtype: pd.DataFrame

      .. rubric:: Notes

      This method assumes the directory contains files from a single station.
      Each NIMS file is read to extract header information including timing,
      station identification, and measurement parameters.

      .. rubric:: Examples

      >>> from mth5.io.nims import NIMSCollection
      >>> nc = NIMSCollection("/path/to/nims/station")
      >>> df = nc.to_dataframe(run_name_zeros=3)
      >>> print(df[['station', 'run', 'start', 'sample_rate']])



   .. py:method:: assign_run_names(df: pandas.DataFrame, zeros: int = 2) -> pandas.DataFrame

      Assign standardized run names to DataFrame entries by station.

      This method assigns run names following the pattern 'sr{sample_rate}_{run_number}'
      where run_number is zero-padded according to the zeros parameter. Run names
      are assigned sequentially within each station, ordered by start time.

      :param df: DataFrame containing NIMS file metadata with required columns:
                 'station', 'start', 'run', 'sample_rate'. The DataFrame will be
                 modified in-place.
      :type df: pd.DataFrame
      :param zeros: Number of zeros to use for zero-padding the run number in the
                    generated run names (e.g., zeros=2 gives '01', '02', etc.).
      :type zeros: int, default 2

      :returns: The input DataFrame with updated 'run' and 'sequence_number' columns.
                Run names follow the format 'sr{sample_rate}_{run_number:0{zeros}}'.
      :rtype: pd.DataFrame

      .. rubric:: Notes

      - Existing run names (non-None values) are preserved
      - Files are processed in chronological order within each station
      - Sequence numbers are assigned incrementally starting from 1
      - Only files with None run names receive new assignments

      .. rubric:: Examples

      >>> import pandas as pd
      >>> from mth5.io.nims import NIMSCollection
      >>> # Assuming df has columns: station, start, run, sample_rate
      >>> nc = NIMSCollection()
      >>> df_updated = nc.assign_run_names(df, zeros=3)
      >>> print(df_updated['run'].tolist())
      ['sr8_001', 'sr8_002', 'sr1_001']



