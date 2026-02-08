mth5.io.phoenix.readers.segmented
=================================

.. py:module:: mth5.io.phoenix.readers.segmented


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/phoenix/readers/segmented/decimated_segmented_reader/index


Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.segmented.DecimatedSegmentedReader


Package Contents
----------------

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



