mth5.io.phoenix.readers.mtu.mtu_ts
==================================

.. py:module:: mth5.io.phoenix.readers.mtu.mtu_ts

.. autoapi-nested-parse::

   =======================================================================
   original comments from MATLAB script:

   read_tsn - reads a (binary) TS file of the legacy Phoenix MTU-5A instrument
   (TS2, TS3, TS4, TS5) and the even older V5-2000 system (TSL, TSH), and
   output the "ts" array and "tag" metadata dictionary.

   =======================================================================
   :param fpath: path to the TS file
   :param fname: name of the TS file (including extensions)

   :returns: output array of the TS data
             tag:   output dict of the TSn metadata
   :rtype: ts

   =======================================================================
   definition of the TS tag (or what I guessed after reading the user manual
   and fiddling with their files)
   0-7   UTC time of first scan in the record.
   8-9   instrument serial number (16-bit integer)
   10-11 number of scans in the record (16-bit integer)
   12    number of channels per scan
   13    tag length (TSn) or tag length code (TSH, TSL)
   14    status code
   15    bit-wise saturation flags (please note that the older TSH/L tag
         ends here )
   16    reserved for future indication of different tag and/or sample
         formats
   17    sample length in bytes
   18-19 sample rate (in units defined by byte 20)
   20    units of sample rate
   21    clock status
   22-25 clock error in seconds
   26-32 reserved; must be 0

   =======================================================================
   notes on the TS format of TSn files:
   The binary TS file consists of several data blocks, each contains a data
   tag and a number of records in it.
   Each time record consists of three bytes (24 bit), let's name them byte1,
   byte2, and byte3:
   the ts record (int) should be (+/-) (byte3*65536 + byte2*256 + byte1)

   Hao
   2012.07.04
   Beijing
   =======================================================================



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.mtu.mtu_ts.MTUTSN


Module Contents
---------------

.. py:class:: MTUTSN(file_path: str | pathlib.Path | None = None, **kwargs)

   Reader for legacy Phoenix MTU-5A instrument time series binary files.

   Reads time series data from Phoenix MTU-5A (.TS2, .TS3, .TS4, .TS5) and
   V5-2000 system (.TSL, .TSH) binary files. The data consists of 24-bit
   signed integers organized in data blocks with headers.

   :param file_path: Path to the TSN file to read. If None, the reader is created without
                     loading data. Default is None.
   :type file_path: str or Path or None, optional

   .. attribute:: file_path

      Path to the currently loaded TSN file.

      :type: Path or None

   .. attribute:: ts

      Time series data array with shape (n_channels, n_samples).

      :type: ndarray or None

   .. attribute:: tag

      Metadata dictionary containing file information.

      :type: dict

   .. rubric:: Examples

   Read a TS3 file:

   >>> from pathlib import Path
   >>> reader = MTUTSN('data/1690C16C.TS3')
   >>> print(reader.ts.shape)
   (3, 86400)
   >>> print(reader.tag['sample_rate'])
   24

   Create reader without loading data:

   >>> reader = MTUTSN()
   >>> reader.read('data/1690C16C.TS3')

   Access metadata:

   >>> reader = MTUTSN('data/1690C16C.TS4')
   >>> reader.read()
   >>> print(f"Channels: {reader.tag['n_ch']}")
   Channels: 4
   >>> print(f"Blocks: {reader.tag['n_block']}")
   Blocks: 48


   .. py:property:: file_path
      :type: pathlib.Path | None


      Get the TSN file path.


   .. py:attribute:: ts
      :value: None



   .. py:attribute:: ts_metadata
      :value: None



   .. py:method:: get_sign24(x: numpy.ndarray | list | int) -> numpy.ndarray

      Convert unsigned 24-bit integers to signed integers.

      Converts unsigned 24-bit values (0 to 16777215) to their signed
      equivalents (-8388608 to 8388607) by applying two's complement.

      :param x: Unsigned 24-bit integer value(s) to convert.
      :type x: ndarray or list or int

      :returns: Signed 24-bit integer value(s) as int32 array.
      :rtype: ndarray

      .. rubric:: Examples

      Convert a single positive value:

      >>> reader = MTUTSN()
      >>> reader.get_sign24(100)
      array([100], dtype=int32)

      Convert a single negative value (unsigned representation):

      >>> reader.get_sign24(16777215)  # -1 in 24-bit signed
      array([-1], dtype=int32)

      Convert an array:

      >>> values = np.array([0, 8388607, 8388608, 16777215])
      >>> reader.get_sign24(values)
      array([       0,  8388607, -8388608,       -1], dtype=int32)



   .. py:method:: read(file_path: str | pathlib.Path | None = None) -> None

      Read and parse a Phoenix MTU time series binary file.

      Reads complete time series data from legacy Phoenix MTU-5A instrument
      files (.TS2, .TS3, .TS4, .TS5) or V5-2000 system files (.TSL, .TSH).
      Each file contains multiple data blocks with 24-bit signed integer
      samples organized by channel.

      :param file_path: Path to the TSN file to read. If None, uses the current file_path
                        attribute. Default is None.
      :type file_path: str or Path or None, optional

      :returns: * **ts** (*ndarray*) -- Time series data array with shape (n_channels, total_samples).
                  Data type is float64. Each row represents one channel, and each
                  column is a time sample.
                * **tag** (*dict*) -- Metadata dictionary containing file information with keys:

                  - 'box_number' (int): Instrument serial number
                  - 'ts_type' (str): Instrument type ('MTU-5' or 'V5-2000')
                  - 'sample_rate' (int): Sampling frequency in Hz
                  - 'n_ch' (int): Number of channels
                  - 'n_scan' (int): Number of scans per data block
                  - 'start' (MTime): UTC timestamp of first sample
                  - 'ts_length' (float): Duration of each block in seconds
                  - 'n_block' (int): Total number of data blocks in file

      :raises EOFError: If the file is empty or cannot be read.
      :raises ValueError: If the file has an unsupported extension or channel count.
      :raises FileNotFoundError: If the specified file does not exist.

      .. rubric:: Examples

      Read a 3-channel TS3 file:

      >>> reader = MTUTSN()
      >>> ts, tag = reader.read('data/1690C16C.TS3')
      >>> print(f"Shape: {ts.shape}")
      Shape: (3, 86400)
      >>> print(f"Sample rate: {tag['sample_rate']} Hz")
      Sample rate: 24 Hz
      >>> print(f"Duration: {ts.shape[1] / tag['sample_rate']:.1f} seconds")
      Duration: 3600.0 seconds

      Read a 4-channel TS4 file:

      >>> reader = MTUTSN('data/1690C16C.TS4')
      >>> print(f"Channels: {reader.tag['n_ch']}")
      Channels: 4
      >>> print(f"Start time: {reader.tag['start'].isoformat()}")
      Start time: 2016-07-16T00:00:00+00:00

      Read and process data:

      >>> ts, tag = MTUTSN().read('data/station.TS5')
      >>> # Calculate statistics for each channel
      >>> for i in range(tag['n_ch']):
      ...     print(f"Ch{i} mean: {ts[i].mean():.2f}, std: {ts[i].std():.2f}")
      Ch0 mean: 123.45, std: 456.78
      Ch1 mean: -234.56, std: 567.89
      ...



   .. py:method:: to_runts(table_filepath: str | pathlib.Path | None = None, calibrate=True) -> mth5.timeseries.RunTS

      Create an MTUTable object from the TSN file and associated TBL file.

      :param table_filepath: Path to the corresponding TBL file.
      :type table_filepath: str or Path

      :returns: An MTUTable object containing metadata from the TBL file.
      :rtype: MTUTable

      .. rubric:: Examples

      >>> reader = MTUTSN('data/1690C16C.TS3')
      >>> mtu_table = reader.to_runts('data/1690C16C.TBL')
      >>> print(mtu_table.metadata)
      {...}



