mth5.io.phoenix.readers.contiguous
==================================

.. py:module:: mth5.io.phoenix.readers.contiguous


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/phoenix/readers/contiguous/decimated_continuous_reader/index


Classes
-------

.. autoapisummary::

   mth5.io.phoenix.readers.contiguous.DecimatedContinuousReader


Package Contents
----------------

.. py:class:: DecimatedContinuousReader(path: str | pathlib.Path, num_files: int = 1, report_hw_sat: bool = False, **kwargs)

   Bases: :py:obj:`mth5.io.phoenix.readers.TSReaderBase`


   Class to create a streamer for continuous decimated time series.

   This reader handles continuous decimated time series files such as 'td_150',
   'td_30'. These files have no sub header information.

   :param path: Path to the time series file
   :type path: str or Path
   :param num_files: Number of files in the sequence, by default 1
   :type num_files: int, optional
   :param report_hw_sat: Whether to report hardware saturation, by default False
   :type report_hw_sat: bool, optional
   :param \*\*kwargs: Additional keyword arguments passed to parent TSReaderBase class

   .. attribute:: subheader

      Empty dictionary as these files have no sub header information

      :type: dict

   .. attribute:: data_size

      Size of the data sequence when read

      :type: int or None


   .. py:attribute:: subheader


   .. py:attribute:: data_size
      :type:  int | None
      :value: None



   .. py:property:: segment_start_time
      :type: mt_metadata.common.mttime.MTime


      Estimate the segment start time based on sequence number.

      The first sequence starts 1 second later than the set start time due
      to initiation within the data logger.

      :returns: Start time of the recording segment
      :rtype: MTime


   .. py:property:: segment_end_time
      :type: mt_metadata.common.mttime.MTime


      Estimate end time of the segment.

      The first sequence starts 1 second later than the set start time due
      to initiation within the data logger.

      :returns: Estimated end time from number of samples
      :rtype: MTime


   .. py:property:: sequence_start
      :type: mt_metadata.common.mttime.MTime


      Get the sequence start time.

      :returns: Start time of the sequence
      :rtype: MTime


   .. py:property:: sequence_end
      :type: mt_metadata.common.mttime.MTime


      Get the sequence end time.

      :returns: End time of the sequence based on data size or max samples
      :rtype: MTime


   .. py:method:: read() -> numpy.ndarray

      Read in the full data from the current file.

      :returns: Single channel data array with dtype float32
      :rtype: np.ndarray



   .. py:method:: read_sequence(start: int = 0, end: int | None = None) -> numpy.ndarray

      Read a sequence of files.

      :param start: Starting index in the sequence, by default 0
      :type start: int, optional
      :param end: Ending index in the sequence to read, by default None
      :type end: int or None, optional

      :returns: Data within the given sequence range as float32 array
      :rtype: np.ndarray



   .. py:method:: to_channel_ts(rxcal_fn: str | pathlib.Path | None = None, scal_fn: str | pathlib.Path | None = None) -> mth5.timeseries.ChannelTS

      Convert to a ChannelTS object.

      :param rxcal_fn: Path to receiver calibration file, by default None
      :type rxcal_fn: str, Path or None, optional
      :param scal_fn: Path to sensor calibration file, by default None
      :type scal_fn: str, Path or None, optional

      :returns: Channel time series object with data, metadata, and calibration
      :rtype: ChannelTS



