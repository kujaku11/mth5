# -*- coding: utf-8 -*-
"""
Module to read and parse native Phoenix Geophysics data formats of the 
MTU-5C Family.

This module implements Streamed readers for decimated continuos time series
formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock 
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np

from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS

# =============================================================================


class DecimatedContinuousReader(TSReaderBase):
    """
    Class to create a streamer for continuous decimated time series,
    i.e. 'td_150', 'td_30'

    These files have no sub header information.

    """

    def __init__(self, path, num_files=1, report_hw_sat=False, **kwargs):
        # Init the base class
        super().__init__(
            path,
            num_files=num_files,
            header_length=128,
            report_hw_sat=report_hw_sat,
            **kwargs,
        )

        self.unpack_header(self.stream)
        self.subheader = {}

    @property
    def segment_start_time(self):
        """
        estimate the segment start time based on sequence number

        The first sequence starts 1 second later than the set start time due
        to initiation within the data logger

        :return: start time of the recording
        :rtype: :class:`mt_metadata.utils.mttime.MTime`

        """

        if self.file_sequence == 1:
            return self.recording_start_time + 1
        else:
            return self.recording_start_time + (
                self.frag_period * (self.file_sequence - 1)
            )

    @property
    def segment_end_time(self):
        """
        estimate end time

        The first sequence starts 1 second later than the set start time due
        to initiation within the data logger

        :return: estimated end time from number of samples
        :rtype: :class:`mt_metadata.utils.mttime.MTime`

        """

        return self.segment_start_time + (self.max_samples / self.sample_rate)

    # need a read and read sequence
    def read(self):
        """
        Read in the full data from the file given.

        :return: single channel data array
        :rtype: :class:`numpy.ndarray`

        """

        self.stream.seek(self.header_length)
        return np.fromfile(self.stream, dtype=np.float32)

    def read_sequence(self, start=0, end=None):
        """
        Read a sequence of files

        :param start: starting index in the sequence, defaults to 0
        :type start: integer, optional
        :param end: eneding index in the sequence to read, defaults to None
        :type end: integer, optional
        :return: data within the given sequence range
        :rtype: :class:`numpy.ndarray`

        """
        data = np.array([], dtype=np.float32)
        for ii, fn in enumerate(self.sequence_list[slice(start, end)], start):
            self._open_file(fn)
            self.unpack_header(self.stream)
            ts = self.read()
            data = np.append(data, ts)
            self.seq = ii
        self.logger.debug("Read {self.seq + 1} sequences")
        return data

    def to_channel_ts(self):
        """
        convert to a ChannelTS object

        :return: DESCRIPTION
        :rtype: TYPE

        """
        data = self.read_sequence()
        ch_metadata = self.channel_metadata()
        return ChannelTS(
            channel_type=ch_metadata.type,
            data=data,
            channel_metadata=ch_metadata,
            run_metadata=self.run_metadata(),
            station_metadata=self.station_metadata(),
        )
