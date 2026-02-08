mth5.io.phoenix.readers.segmented.decimated_segmented_reader
============================================================

.. py:module:: mth5.io.phoenix.readers.segmented.decimated_segmented_reader

.. autoapi-nested-parse::

   Module to read and parse native Phoenix Geophysics data formats of the
   MTU-5C Family

   This module implements Streamed readers for segmented-decimated  time series
    formats of the MTU-5C family.

   :author: Jorge Torres-Solis

   Revised 2022 by J. Peacock



Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.segmented.decimated_segmented_reader.SubHeader
   mth5.io.phoenix.readers.segmented.decimated_segmented_reader.Segment
   mth5.io.phoenix.readers.segmented.decimated_segmented_reader.DecimatedSegmentedReader
   mth5.io.phoenix.readers.segmented.decimated_segmented_reader.DecimatedSegmentCollection


Module Contents
---------------

.. py:class:: SubHeader(**kwargs)

   Class for subheader of segmented files.

   This class handles the parsing and access to subheader information in
   Phoenix Geophysics segmented time series files. The subheader contains
   metadata about each segment including timing, sample counts, and statistics.

   :param \*\*kwargs: Arbitrary keyword arguments that are set as attributes

   .. attribute:: header_length

      Length of the subheader in bytes (32 bytes)

      :type: int

   .. attribute:: _header

      Raw header bytes from the file

      :type: bytes or None

   .. attribute:: _unpack_dict

      Dictionary defining how to unpack different header fields

      :type: dict


   .. py:attribute:: header_length
      :value: 32



   .. py:property:: gps_time_stamp
      :type: mt_metadata.common.mttime.MTime | None


      GPS time stamp in UTC.

      :returns: GPS timestamp if header is available, None otherwise
      :rtype: MTime or None


   .. py:property:: n_samples
      :type: int | None


      Number of samples in the segment.

      :returns: Number of samples if header is available, None otherwise
      :rtype: int or None


   .. py:property:: saturation_count
      :type: int | None


      Number of saturated samples.

      :returns: Saturation count if header is available, None otherwise
      :rtype: int or None


   .. py:property:: missing_count
      :type: int | None


      Number of missing samples.

      :returns: Missing sample count if header is available, None otherwise
      :rtype: int or None


   .. py:property:: value_min
      :type: float | None


      Minimum value in the segment.

      :returns: Minimum value if header is available, None otherwise
      :rtype: float or None


   .. py:property:: value_max
      :type: float | None


      Maximum value in the segment.

      :returns: Maximum value if header is available, None otherwise
      :rtype: float or None


   .. py:property:: value_mean
      :type: float | None


      Mean value in the segment.

      :returns: Mean value if header is available, None otherwise
      :rtype: float or None


   .. py:method:: unpack_header(stream: BinaryIO) -> None

      Unpack the header from a binary stream.

      :param stream: Binary stream to read header from
      :type stream: BinaryIO



.. py:class:: Segment(stream: BinaryIO, **kwargs)

   Bases: :py:obj:`SubHeader`


   A segment class to hold a single segment.

   This class represents a single time series segment with its associated
   metadata and data. It inherits from SubHeader to provide access to
   segment-specific header information.

   :param stream: Binary file stream to read from
   :type stream: BinaryIO
   :param \*\*kwargs: Additional keyword arguments passed to SubHeader

   .. attribute:: stream

      The file stream for reading data

      :type: BinaryIO

   .. attribute:: data

      Time series data for this segment

      :type: np.ndarray or None


   .. py:attribute:: stream


   .. py:attribute:: data
      :type:  numpy.ndarray | None
      :value: None



   .. py:method:: read_segment(metadata_only: bool = False) -> None

      Read the segment data from the file stream.

      :param metadata_only: If True, only read metadata without loading data, by default False
      :type metadata_only: bool, optional



   .. py:property:: segment_start_time
      :type: mt_metadata.common.mttime.MTime | None


      Get the segment start time.

      :returns: GPS timestamp of segment start, or None if not available
      :rtype: MTime or None


   .. py:property:: segment_end_time
      :type: mt_metadata.common.mttime.MTime | None


      Calculate the segment end time.

      :returns: Estimated end time based on start time, sample count and sample rate,
                or None if required information is not available
      :rtype: MTime or None


.. py:class:: DecimatedSegmentedReader(path: str | pathlib.Path, num_files: int = 1, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Class to create a streamer for segmented decimated time series.

   This reader handles segmented decimated time series files such as 'td_24k'.
   These files have sub headers containing metadata for each segment.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: sub_header

      SubHeader instance for parsing segment headers

      :type: SubHeader

   .. attribute:: subheader

      Dictionary for additional subheader information

      :type: dict


   .. py:attribute:: sub_header


   .. py:attribute:: subheader


   .. py:method:: read_segment(metadata_only: bool = False) -> Segment

      Read a single segment from the file.

      :param metadata_only: If True, only read metadata without loading data, by default False
      :type metadata_only: bool, optional

      :returns: Segment object containing data and metadata
      :rtype: Segment

      :raises ValueError: If stream is not available



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



.. py:class:: DecimatedSegmentCollection(path: str | pathlib.Path, num_files: int = 1, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Class to read multiple segments from a segmented decimated time series file.

   This reader handles files containing multiple segments of decimated time series
   data such as 'td_24k'. Each segment has its own sub header with metadata.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: sub_header

      SubHeader instance for parsing segment headers

      :type: SubHeader

   .. attribute:: subheader

      Dictionary for additional subheader information

      :type: dict


   .. py:attribute:: sub_header


   .. py:attribute:: subheader


   .. py:method:: read_segments(metadata_only: bool = False) -> list[Segment]

      Read all segments from the file.

      :param metadata_only: If True, only read metadata without loading data, by default False
      :type metadata_only: bool, optional

      :returns: List of Segment objects containing data and metadata
      :rtype: list[Segment]

      :raises ValueError: If stream is not available



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> list[mth5.timeseries.ChannelTS]

      Convert all segments to ChannelTS objects.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: List of ChannelTS objects, one for each segment
      :rtype: list[ChannelTS]



