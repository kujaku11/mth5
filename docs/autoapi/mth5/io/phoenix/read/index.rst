mth5.io.phoenix.read
====================

.. py:module:: mth5.io.phoenix.read

.. autoapi-nested-parse::

   Created on Fri May  6 12:39:34 2022

   @author: jpeacock



Attributes
----------

.. autoapisummary::

   mth5.io.phoenix.read.READERS


Functions
---------

.. autoapisummary::

   mth5.io.phoenix.read.get_file_extenstion
   mth5.io.phoenix.read.open_phoenix
   mth5.io.phoenix.read.read_phoenix


Module Contents
---------------

.. py:data:: READERS
   :type:  dict[str, type]

.. py:function:: get_file_extenstion(file_name: str | pathlib.Path) -> str

   Get the file extension from a file name.

   :param file_name: The file name to extract the extension from.
   :type file_name: str or pathlib.Path

   :returns: The file extension without the leading dot.
   :rtype: str


.. py:function:: open_phoenix(file_name: str | pathlib.Path, **kwargs: Any) -> mth5.io.phoenix.readers.DecimatedContinuousReader | mth5.io.phoenix.readers.DecimatedSegmentedReader | mth5.io.phoenix.readers.NativeReader | mth5.io.phoenix.readers.MTUTSN | mth5.io.phoenix.readers.MTUTable

   Open a Phoenix Geophysics data file in the appropriate container.

   :param file_name: Full path to the Phoenix data file to open.
   :type file_name: str or pathlib.Path
   :param \*\*kwargs: Additional keyword arguments to pass to the reader constructor.
   :type \*\*kwargs: Any

   :returns: **reader** -- The appropriate Phoenix reader container based on file extension:
             - .bin files: NativeReader
             - .td_24k files: DecimatedSegmentedReader
             - .td_150/.td_30 files: DecimatedContinuousReader
   :rtype: DecimatedContinuousReader | DecimatedSegmentedReader | NativeReader

   :raises KeyError: If the file extension is not supported by any Phoenix reader.


.. py:function:: read_phoenix(file_name: str | pathlib.Path, **kwargs: Any) -> mth5.timeseries.ChannelTS | mth5.timeseries.RunTS | mth5.io.phoenix.readers.MTUTable

   Read a Phoenix Geophysics data file into a ChannelTS or RunTS object
   depending on the file type.  Newer files that end in .td_XX or .bin will be
   read into ChannelTS objects.  Older MTU files that end in .TS3, .TS4, .TS5,
   .TSL, or .TSH will be read into RunTS objects.

   :param file_name: Path to the Phoenix data file to read.
   :type file_name: str or pathlib.Path
   :param \*\*kwargs: Additional keyword arguments. May include:

                      - rxcal_fn : str or pathlib.Path, optional
                          Path to receiver calibration file.
                      - scal_fn : str or pathlib.Path, optional
                          Path to sensor calibration file.
                      - table_filepath : str or pathlib.Path, optional
                          Path to the MTU TBL file for use with MTUTSN files.
                      - Other arguments passed to the Phoenix reader constructor.
   :type \*\*kwargs: Any

   :returns: * **channel_ts** (*ChannelTS*) -- Time series data object containing the Phoenix file data
               with calibration applied if calibration files were provided.
             * **run_ts** (*RunTS*) -- Time series data object containing the MTU data from the Phoenix MTU
               files with calibration applied if specified.
             * **mtu_table** (*MTUTable*) -- Metadata table object containing the MTU table data.

   :raises KeyError: If the file extension is not supported by any Phoenix reader.
   :raises ValueError: If the file cannot be read or converted to ChannelTS or RunTS format.


