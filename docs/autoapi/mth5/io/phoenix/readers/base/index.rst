mth5.io.phoenix.readers.base
============================

.. py:module:: mth5.io.phoenix.readers.base

.. autoapi-nested-parse::

   Module to read and parse native Phoenix Geophysics data formats of the
   MTU-5C Family.

   This module implements Streamed readers for segmented-decimated continuus-decimated
   and native sampling rate time series formats of the MTU-5C family.

   :author: Jorge Torres-Solis

   Revised 2022 by J. Peacock



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.base.TSReaderBase


Module Contents
---------------

.. py:class:: TSReaderBase(path: str | pathlib.Path, num_files: int = 1, header_length: int = 128, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.header.Header`


   Generic reader that all other readers will inherit.

   This base class provides common functionality for reading Phoenix Geophysics
   time series data files, including header parsing, file sequence management,
   and metadata handling.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param header_length: Length of file header in bytes, by default 128
   :type header_length: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent Header class

   .. attribute:: stream

      File stream for reading binary data

      :type: BinaryIO or None

   .. attribute:: base_path

      Path to the current file

      :type: Path

   .. attribute:: last_seq

      Last sequence number in the file sequence

      :type: int

   .. attribute:: rx_metadata

      Receiver metadata object

      :type: PhoenixReceiverMetadata or None


   .. py:attribute:: logger


   .. py:property:: base_path
      :type: pathlib.Path


      Full path of the file.

      :returns: Full path to the file
      :rtype: Path


   .. py:attribute:: last_seq


   .. py:attribute:: stream
      :value: None



   .. py:attribute:: rx_metadata
      :value: None



   .. py:property:: base_dir
      :type: pathlib.Path


      Parent directory of the file.

      :returns: Parent directory of the file
      :rtype: Path


   .. py:property:: file_name
      :type: str


      Name of the file.

      :returns: Name of the file
      :rtype: str


   .. py:property:: file_extension
      :type: str


      File extension.

      :returns: File extension including the dot
      :rtype: str


   .. py:property:: instrument_id
      :type: str


      Instrument ID extracted from filename.

      :returns: Instrument identifier
      :rtype: str


   .. py:property:: seq
      :type: int


      Sequence number of the file.

      :returns: Sequence number extracted from filename or set value
      :rtype: int


   .. py:property:: file_size
      :type: int


      File size in bytes.

      :returns: Size of the file in bytes
      :rtype: int


   .. py:property:: max_samples
      :type: int


      Maximum number of samples in a file.

      Calculated as: (total number of bytes - header length) / frame size * n samples per frame

      :returns: Maximum number of samples in the file
      :rtype: int


   .. py:property:: sequence_list
      :type: list[pathlib.Path]


      Get all the files in the sequence sorted by sequence number.

      :returns: List of Path objects for all files in the sequence
      :rtype: list[Path]


   .. py:property:: config_file_path
      :type: pathlib.Path | None


      Path to the config.json file.

      :returns: Path to config file if it exists, None otherwise
      :rtype: Path or None


   .. py:property:: recmeta_file_path
      :type: pathlib.Path | None


      Path to the recmeta.json file.

      :returns: Path to recmeta file if it exists, None otherwise
      :rtype: Path or None


   .. py:method:: open_next() -> bool

      Open the next file in the sequence.

      :returns: True if next file is now open, False if it is not
      :rtype: bool



   .. py:method:: open_file_seq(file_seq_num: int | None = None) -> bool

      Open a file in the sequence given the sequence number.

      :param file_seq_num: Sequence number to open, by default None
      :type file_seq_num: int, optional

      :returns: True if file is now open, False if it is not
      :rtype: bool



   .. py:method:: close() -> None

      Close the file stream.



   .. py:method:: get_config_object() -> mth5.io.phoenix.readers.config.PhoenixConfig | None

      Read a config file into an object.

      :returns: Configuration object if config file exists, None otherwise
      :rtype: PhoenixConfig or None



   .. py:method:: get_receiver_metadata_object() -> None

      Read recmeta.json into an object and store in rx_metadata attribute.



   .. py:method:: get_lowpass_filter_name() -> str | None

      Get the lowpass filter used by the receiver pre-decimation.

      :returns: Name of the lowpass filter if available, None otherwise
      :rtype: str or None



   .. py:method:: update_channel_map_from_recmeta() -> None

      Update channel map from recmeta.json file.



   .. py:property:: channel_metadata
      :type: Any


      Channel metadata updated from recmeta.

      :returns: Channel metadata object
      :rtype: Any


   .. py:property:: run_metadata
      :type: Any


      Run metadata updated from recmeta.

      :returns: Run metadata object
      :rtype: Any


   .. py:property:: station_metadata
      :type: Any


      Station metadata updated from recmeta.

      :returns: Station metadata object
      :rtype: Any


   .. py:method:: get_receiver_lowpass_filter(rxcal_fn: str | pathlib.Path) -> Any

      Get receiver lowpass filter from the rxcal.json file.

      :param rxcal_fn: Path to the receiver calibration file
      :type rxcal_fn: str or Path

      :returns: Filter object from calibration file
      :rtype: Any

      :raises ValueError: If the lowpass filter name cannot be found



   .. py:method:: get_dipole_filter() -> mt_metadata.timeseries.filters.CoefficientFilter | None

      Get dipole filter for electric field channels.

      :returns: Dipole filter if channel has dipole length, None otherwise
      :rtype: CoefficientFilter or None



   .. py:method:: get_sensor_filter(scal_fn: str | pathlib.Path) -> Any

      Get sensor filter from calibration file.

      :param scal_fn: Path to sensor calibration file
      :type scal_fn: str or Path

      :returns: Sensor filter object
      :rtype: Any

      .. rubric:: Notes

      This method is not implemented yet.



   .. py:method:: get_v_to_mv_filter() -> mt_metadata.timeseries.filters.CoefficientFilter

      Create a filter to convert units from volts to millivolts.

      :returns: Filter that converts volts to millivolts with gain of 1000
      :rtype: CoefficientFilter



   .. py:method:: get_channel_response(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mt_metadata.timeseries.filters.ChannelResponse

      Get the channel response filter.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Complete channel response filter chain
      :rtype: ChannelResponse



