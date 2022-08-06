# -*- coding: utf-8 -*-
"""Module to read and parse native Phoenix Geophysics data formats of the MTU-5C Family

This module implements Streamed readers for segmented-decimated continuus-decimated
and native sampling rate time series formats of the MTU-5C family.
"""

__author__ = "Jorge Torres-Solis"

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
    i.e. *.td_150, *.td_30

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
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.recording_start_time + (
            self.frag_period * (self.file_sequence - 1)
            - (self.file_sequence - 1)
        )

    @property
    def segment_end_time(self):
        """
        estimate end time

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.segment_start_time + (self.max_samples / self.sample_rate)

    @property
    def table_entry(self):
        """
        data frame entry of important file metadata

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return {
            "recording_id": self.recording_id,
            "start": self.segment_start_time,
            "end": self.segment_end_time,
            "run": None,
            "channel_id": self.channel_id,
            "channel": self.channel_map[str(self.channel_id)],
            "sample_rate": self.sample_rate,
            "file": self.file_name,
            "max_samples": self.max_samples,
        }

    # need a read and read sequence
    def read(self):
        """
        Read in the full data from the file given

        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.stream.seek(self.header_length)
        return np.fromfile(self.stream, dtype=np.float32)

    def read_sequence(self, start=0, end=None):
        """
        Read a sequence of files

        :param start: DESCRIPTION, defaults to 0
        :type start: TYPE, optional
        :param end: DESCRIPTION, defaults to None
        :type end: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        data = np.array([], dtype=np.float32)
        for ii, fn in enumerate(self.sequence_list[slice(start, end)], start):
            self._open_file(fn)
            self.unpack_header(self.stream)
            ts = self.read()
            data = np.append(data, ts)
            self.seq = ii

        self.logger.debug("Read %s sequences", self.seq + 1)
        return data

    def read_data(self, numSamples):
        ret_array = np.empty([0])
        if self.stream is not None:
            ret_array = np.fromfile(
                self.stream, dtype=np.float32, count=numSamples
            )
            while ret_array.size < numSamples:
                if not self.open_next():
                    return np.empty([0])
                # Array below will contain the data, or will be an np.empty array if end of series as desired
                ret_array = np.append(
                    ret_array,
                    np.fromfile(
                        self.stream,
                        dtype=np.float32,
                        count=(numSamples - ret_array.size),
                    ),
                )
        return ret_array

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
