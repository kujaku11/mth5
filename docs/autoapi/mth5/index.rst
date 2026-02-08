mth5
====

.. py:module:: mth5

.. autoapi-nested-parse::

   Top-level package for MTH5.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/clients/index
   /autoapi/mth5/data/index
   /autoapi/mth5/groups/index
   /autoapi/mth5/helpers/index
   /autoapi/mth5/io/index
   /autoapi/mth5/mth5/index
   /autoapi/mth5/processing/index
   /autoapi/mth5/tables/index
   /autoapi/mth5/timeseries/index
   /autoapi/mth5/utils/index


Attributes
----------

.. autoapisummary::

   mth5.config
   mth5.CHUNK_SIZE
   mth5.ACCEPTABLE_FILE_TYPES
   mth5.ACCEPTABLE_FILE_SUFFIXES
   mth5.ACCEPTABLE_FILE_VERSIONS
   mth5.ACCEPTABLE_DATA_LEVELS
   mth5.TF_DTYPE_LIST
   mth5.TF_DTYPE
   mth5.CHANNEL_DTYPE_LIST
   mth5.CHANNEL_DTYPE
   mth5.FC_DTYPE_LIST
   mth5.FC_DTYPE
   mth5.RUN_SUMMARY_LIST
   mth5.RUN_SUMMARY_DTYPE
   mth5.RUN_SUMMARY_COLUMNS
   mth5.STANDARDS_DTYPE_LIST
   mth5.STANDARDS_DTYPE


Functions
---------

.. autoapisummary::

   mth5.read_file


Package Contents
----------------

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


.. py:data:: config

.. py:data:: CHUNK_SIZE
   :value: 8196


.. py:data:: ACCEPTABLE_FILE_TYPES
   :value: ['mth5', 'MTH5', 'h5', 'H5']


.. py:data:: ACCEPTABLE_FILE_SUFFIXES

.. py:data:: ACCEPTABLE_FILE_VERSIONS
   :value: ['0.1.0', '0.2.0']


.. py:data:: ACCEPTABLE_DATA_LEVELS
   :value: [0, 1, 2, 3]


.. py:data:: TF_DTYPE_LIST

.. py:data:: TF_DTYPE

.. py:data:: CHANNEL_DTYPE_LIST

.. py:data:: CHANNEL_DTYPE

.. py:data:: FC_DTYPE_LIST

.. py:data:: FC_DTYPE

.. py:data:: RUN_SUMMARY_LIST

.. py:data:: RUN_SUMMARY_DTYPE

.. py:data:: RUN_SUMMARY_COLUMNS

.. py:data:: STANDARDS_DTYPE_LIST

.. py:data:: STANDARDS_DTYPE

