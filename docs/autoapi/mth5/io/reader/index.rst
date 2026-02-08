mth5.io.reader
==============

.. py:module:: mth5.io.reader

.. autoapi-nested-parse::

   Universal reader for magnetotelluric time series data files.

   This module provides a plugin-like system for reading various MT data formats
   and returning appropriate :class:`mth5.timeseries` objects. The reader
   automatically detects file types and dispatches to the correct parser.

   Plugin Structure
   ----------------
   If you are writing your own reader, implement the following structure:

       * Class object that reads the given file format
       * A reader function named read_{file_type} (e.g., read_nims)
       * Return value should be a :class:`mth5.timeseries.MTTS` or
         :class:`mth5.timeseries.RunTS` object plus extra metadata as a
         dictionary with keys formatted as {level.attribute}

   Example Implementation
   ----------------------
   .. code-block:: python

       class NewFile:
           def __init__(self, fn):
               self.fn = fn

           def read_header(self):
               return header_information

           def read_newfile(self):
               ex, ey, hx, hy, hz = read_in_channels_as_MTTS
               return RunTS([ex, ey, hx, hy, hz])

       def read_newfile(fn):
           new_file_obj = NewFile(fn)
           run_obj = new_file_obj.read_newfile()
           return run_obj, extra_metadata

   Then add your reader to the readers dictionary for automatic detection.

   .. seealso::

      Existing readers in `mth5.io` for implementation guidance.

      Created on Wed Aug 26 10:32:45 2020

      :author: Jared Peacock
      :license: MIT



Attributes
----------

.. autoapisummary::

   mth5.io.reader.readers


Functions
---------

.. autoapisummary::

   mth5.io.reader.get_reader
   mth5.io.reader.read_file


Module Contents
---------------

.. py:data:: readers
   :type:  dict[str, dict[str, Any]]

.. py:function:: get_reader(extension: str) -> tuple[str, Callable]

   Get the appropriate reader function for a file extension.

   Searches the reader registry to find the correct parser function
   for the given file extension. Handles ambiguous extensions by
   issuing warnings when multiple readers might apply.

   :param extension: File extension (without the dot) to find a reader for
   :type extension: str

   :returns: Tuple containing:
             - Reader name (str): Identifier for the reader type
             - Reader function (Callable): Function to parse files of this type
   :rtype: tuple[str, Callable]

   :raises ValueError: If no reader is found for the given file extension

   .. rubric:: Examples

   >>> reader_name, reader_func = get_reader("z3d")
   >>> print(reader_name)  # "zen"
   >>> data = reader_func("/path/to/file.z3d")

   .. rubric:: Notes

   Some extensions like "bin" are ambiguous and could match multiple
   readers (NIMS or Phoenix). A warning is issued in such cases.


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


