mth5.io
=======

.. py:module:: mth5.io


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/collection/index
   /autoapi/mth5/io/conversion/index
   /autoapi/mth5/io/lemi/index
   /autoapi/mth5/io/metronix/index
   /autoapi/mth5/io/miniseed/index
   /autoapi/mth5/io/nims/index
   /autoapi/mth5/io/phoenix/index
   /autoapi/mth5/io/reader/index
   /autoapi/mth5/io/scripps/index
   /autoapi/mth5/io/usgs_ascii/index
   /autoapi/mth5/io/zen/index


Classes
-------

.. autoapisummary::

   mth5.io.Collection


Functions
---------

.. autoapisummary::

   mth5.io.read_file


Package Contents
----------------

.. py:class:: Collection(file_path=None, **kwargs)

   A general collection class to keep track of files with methods to create
   runs and run ids.



   .. py:attribute:: logger


   .. py:property:: file_path

      Path object to file directory


   .. py:attribute:: file_ext
      :value: '*'



   .. py:method:: get_empty_entry_dict()

      :return: an empty dictionary with the proper keys for an entry into
       a dataframe
      :rtype: dict




   .. py:method:: get_files(extension)

      Get files with given extension. Uses Pathlib.Path.rglob, so it finds
      all files within the `file_path` by searching all sub-directories.

      :param extension: file extension(s)
      :type extension: string or list
      :return: list of files in the `file_path` with the given extensions
      :rtype: list of Path objects




   .. py:method:: to_dataframe(sample_rates=None, run_name_zeros=4, calibration_path=None)

      Get a data frame of the file summary with column names:

          - **survey**: survey id
          - **station**: station id
          - **run**: run id
          - **start**: start time UTC
          - **end**: end time UTC
          - **channel_id**: channel id or list of channel id's in file
          - **component**: channel component or list of components in file
          - **fn**: path to file
          - **sample_rate**: sample rate in samples per second
          - **file_size**: file size in bytes
          - **n_samples**: number of samples in file
          - **sequence_number**: sequence number of the file
          - **instrument_id**: instrument id
          - **calibration_fn**: calibration file

      :param sample_rates: list of sample rates to process, defaults to None
      :type sample_rates: list, optional
      :param run_name_zeros: number of zeros in run name, defaults to 4
      :type run_name_zeros: int, optional
      :param calibration_path: path to calibration files, defaults to None
      :type calibration_path: str or Path, optional
      :return: summary table of file names,
      :rtype: pandas.DataFrame




   .. py:method:: assign_run_names(df, zeros=4)

      Assign run names to a dataframe. This is a base method that should
      be overridden by subclasses.

      :param df: dataframe with file information
      :type df: pandas.DataFrame
      :param zeros: number of zeros in run name, defaults to 4
      :type zeros: int, optional
      :return: dataframe with run names assigned
      :rtype: pandas.DataFrame



   .. py:method:: get_runs(sample_rates, run_name_zeros=4, calibration_path=None)

      Get a list of runs contained within the given folder.  First the
      dataframe will be developed from which the runs are extracted.

      For continous data all you need is the first file in the sequence. The
      reader will read in the entire sequence.

      For segmented data it will only read in the given segment, which is
      slightly different from the original reader.

      :param sample_rates: list of sample rates to read, defaults to [150, 24000]
      :param run_name_zeros: Number of zeros in the run name, defaults to 4
      :type run_name_zeros: integer, optional
      :return: List of run dataframes with only the first block of files
      :rtype: :class:`collections.OrderedDict`

      :Example:

          >>> from mth5.io.phoenix import PhoenixCollection
          >>> phx_collection = PhoenixCollection(r"/path/to/station")
          >>> run_dict = phx_collection.get_runs(sample_rates=[150, 24000])




   .. py:method:: get_remote_reference_list(df, max_hours=6, min_hours=1.5)

      get remote reference pairs

      :param max_hours: DESCRIPTION, defaults to 6
      :type max_hours: TYPE, optional
      :param min_hours: DESCRIPTION, defaults to 1.5
      :type min_hours: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




.. py:function:: read_file(fn: str | pathlib.Path | list[str | pathlib.Path], file_type: str | None = None, **kwargs: Any) -> Any

   Universal reader for magnetotelluric time series data files.

   Automatically detects the file type based on extension and dispatches
   to the appropriate reader function. Supports both single files and
   lists of files for multi-file formats.

   :param fn: Full path(s) to data file(s) to be read. For multi-file formats,
              pass a list of file paths.
   :type fn: str, Path, or list of str/Path
   :param file_type: Specific reader type to use if file extension is ambiguous.
                     Must be one of the keys in the readers registry, by default None
   :type file_type: str, optional
   :param \*\*kwargs: Additional keyword arguments passed to the specific reader function.
                      Supported arguments depend on the file format and reader.
   :type \*\*kwargs: dict

   :returns: Time series object containing the data:
             - :class:`mth5.timeseries.MTTS` for single channel data
             - :class:`mth5.timeseries.RunTS` for multi-channel run data
   :rtype: MTTS or RunTS

   :raises IOError: If any specified file does not exist
   :raises KeyError: If the specified file_type is not supported
   :raises ValueError: If no reader can be found for the file extension

   .. rubric:: Examples

   Read a single Z3D file (auto-detected)

   >>> data = read_file("/path/to/station_001.z3d")
   >>> print(type(data))  # <class 'mth5.timeseries.ChannelTS'>

   Read with explicit file type for ambiguous extensions

   >>> data = read_file("/path/to/data.bin", file_type="nims")
   >>> print(data.n_channels)

   Read multiple files for a multi-file format

   >>> files = ["/path/to/file1.asc", "/path/to/file2.asc"]
   >>> run_data = read_file(files, sample_rate=1.0)

   .. rubric:: Notes

   Supported file types and extensions:
   - zen: .z3d (Zonge Z3D files)
   - nims: .bin, .bnn (USGS NIMS files)
   - usgs_ascii: .asc, .zip (USGS ASCII format)
   - miniseed: .miniseed, .ms, .mseed (miniSEED format)
   - lemi424: .txt (LEMI-424 format)
   - phoenix: .bin, .td_30, .td_150, .td_24k (Phoenix formats)
   - metronix: .atss (Metronix ADU format)

   For ambiguous extensions like .bin, specify file_type explicitly.


