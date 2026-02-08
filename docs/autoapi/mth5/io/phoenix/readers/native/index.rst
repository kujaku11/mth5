mth5.io.phoenix.readers.native
==============================

.. py:module:: mth5.io.phoenix.readers.native


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/phoenix/readers/native/native_reader/index


Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.native.NativeReader


Package Contents
----------------

.. py:class:: NativeReader(path: str | pathlib.Path, num_files: int = 1, scale_to: int = AD_INPUT_VOLTS, header_length: int = 128, last_frame: int = 0, ad_plus_minus_range: float = 5.0, channel_type: str = 'E', report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Native sampling rate 'Raw' time series reader class.

   This class reads native binary (.bin) files from Phoenix Geophysics MTU-5C
   instruments. The files are formatted with a header of 128 bytes followed by
   frames of 64 bytes each. Each frame contains 20 x 3-byte (24-bit) data
   points plus a 4-byte footer.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param scale_to: Data scaling mode (AD_IN_AD_UNITS, AD_INPUT_VOLTS, or
                    INSTRUMENT_INPUT_VOLTS), by default AD_INPUT_VOLTS
   :type scale_to: int, optional
   :param header_length: Length of file header in bytes, by default 128
   :type header_length: int, optional
   :param last_frame: Last frame number seen by the streamer, by default 0
   :type last_frame: int, optional
   :param ad_plus_minus_range: ADC plus/minus range in volts, by default 5.0
   :type ad_plus_minus_range: float, optional
   :param channel_type: Channel type identifier, by default "E"
   :type channel_type: str, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: last_frame

      Last frame number processed

      :type: int

   .. attribute:: data_scaling

      Current data scaling mode

      :type: int

   .. attribute:: ad_plus_minus_range

      ADC voltage range

      :type: float

   .. attribute:: input_plusminus_range

      Input voltage range after gain correction

      :type: float

   .. attribute:: scale_factor

      Calculated scaling factor for data conversion

      :type: float

   .. attribute:: footer_idx_samp_mask

      Bit mask for frame index in footer

      :type: int

   .. attribute:: footer_sat_mask

      Bit mask for saturation count in footer

      :type: int


   .. py:attribute:: last_frame
      :value: 0



   .. py:attribute:: header_length
      :value: 128


      Length of the header in bytes.

      :returns: Header length in bytes.
      :rtype: int


   .. py:attribute:: data_scaling
      :value: 1



   .. py:attribute:: ad_plus_minus_range
      :value: 5.0



   .. py:attribute:: input_plusminus_range


   .. py:attribute:: scale_factor
      :value: 256



   .. py:attribute:: footer_idx_samp_mask
      :value: 0



   .. py:attribute:: footer_sat_mask
      :value: 0



   .. py:method:: read_frames(num_frames: int) -> numpy.ndarray

      Read the given number of frames from the data stream.

      .. note::

         The seek position is not reset, so iterating this method will read
         from the last position in the stream.

      :param num_frames: Number of frames to read
      :type num_frames: int

      :returns: Scaled data from the given number of frames with dtype float64
      :rtype: np.ndarray



   .. py:property:: npts_per_frame
      :type: int


      Get the number of data points per frame.

      :returns: Number of data points per frame (frame size - 4 footer bytes) / 3 bytes per sample
      :rtype: int


   .. py:method:: read() -> tuple[numpy.ndarray, numpy.ndarray]

      Read the full data file using memory mapping and stride tricks.

      .. note::

         This uses numpy.lib.stride_tricks.as_strided which can be unstable
         if the bytes are not the correct length. See notes by numpy.

         The solution is adapted from:
         https://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers

      :returns: Scaled time series data and footer data as (data, footer)
      :rtype: tuple[np.ndarray, np.ndarray]



   .. py:method:: read_sequence(start: int = 0, end: int | None = None) -> tuple[numpy.ndarray, numpy.ndarray]

      Read a sequence of files into a single array.

      :param start: Sequence start index, by default 0
      :type start: int, optional
      :param end: Sequence end index, by default None
      :type end: int or None, optional

      :returns: Scaled time series data and footer data as (data, footer)
                - data: np.ndarray with dtype float32
                - footer: np.ndarray with dtype int32
      :rtype: tuple[np.ndarray, np.ndarray]



   .. py:method:: skip_frames(num_frames: int) -> bool

      Skip frames in the data stream.

      :param num_frames: Number of frames to skip
      :type num_frames: int

      :returns: True if skip completed successfully, False if end of file reached
      :rtype: bool



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



