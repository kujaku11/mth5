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
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mt_metadata.common.mttime import MTime

from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS


# =============================================================================


class DecimatedContinuousReader(TSReaderBase):
    """
    Class to create a streamer for continuous decimated time series.

    This reader handles continuous decimated time series files such as 'td_150',
    'td_30'. These files have no sub header information.

    Parameters
    ----------
    path : str or Path
        Path to the time series file
    num_files : int, optional
        Number of files in the sequence, by default 1
    report_hw_sat : bool, optional
        Whether to report hardware saturation, by default False
    **kwargs
        Additional keyword arguments passed to parent TSReaderBase class

    Attributes
    ----------
    subheader : dict
        Empty dictionary as these files have no sub header information
    data_size : int or None
        Size of the data sequence when read
    """

    def __init__(
        self,
        path: str | Path,
        num_files: int = 1,
        report_hw_sat: bool = False,
        **kwargs,
    ) -> None:
        # Init the base class
        super().__init__(
            path,
            num_files=num_files,
            header_length=128,
            report_hw_sat=report_hw_sat,
            **kwargs,
        )

        self._channel_metadata = self._update_channel_metadata_from_recmeta()
        self.subheader = {}
        self._sequence_start = self.segment_start_time
        self.data_size: int | None = None

    @property
    def segment_start_time(self) -> MTime:
        """
        Estimate the segment start time based on sequence number.

        The first sequence starts 1 second later than the set start time due
        to initiation within the data logger.

        Returns
        -------
        MTime
            Start time of the recording segment
        """
        if self.file_sequence == 1:
            return self.recording_start_time + 1
        else:
            return self.recording_start_time + (
                self.frag_period * (self.file_sequence - 1)
            )

    @property
    def segment_end_time(self) -> MTime:
        """
        Estimate end time of the segment.

        The first sequence starts 1 second later than the set start time due
        to initiation within the data logger.

        Returns
        -------
        MTime
            Estimated end time from number of samples
        """
        return self.segment_start_time + (self.max_samples / self.sample_rate)

    @property
    def sequence_start(self) -> MTime:
        """
        Get the sequence start time.

        Returns
        -------
        MTime
            Start time of the sequence
        """
        return self._sequence_start

    @sequence_start.setter
    def sequence_start(self, value: Any) -> None:
        """
        Set the sequence start time.

        Parameters
        ----------
        value : Any
            Time stamp value that can be converted to MTime
        """
        self._sequence_start = MTime(time_stamp=value)

    @property
    def sequence_end(self) -> MTime:
        """
        Get the sequence end time.

        Returns
        -------
        MTime
            End time of the sequence based on data size or max samples
        """
        if self.data_size is not None:
            return self.sequence_start + self.data_size / self.sample_rate
        else:
            return self.sequence_start + self.max_samples / self.sample_rate

    # need a read and read sequence
    def read(self) -> np.ndarray:
        """
        Read in the full data from the current file.

        Returns
        -------
        np.ndarray
            Single channel data array with dtype float32
        """
        if self.stream is not None:
            self.stream.seek(self.header_length)
            return np.fromfile(self.stream, dtype=np.float32)
        return np.array([], dtype=np.float32)

    def read_sequence(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """
        Read a sequence of files.

        Parameters
        ----------
        start : int, optional
            Starting index in the sequence, by default 0
        end : int or None, optional
            Ending index in the sequence to read, by default None

        Returns
        -------
        np.ndarray
            Data within the given sequence range as float32 array
        """
        data = np.array([], dtype=np.float32)
        for ii, fn in enumerate(self.sequence_list[slice(start, end)], start):
            self._open_file(fn)
            if self.stream is not None:
                self.unpack_header(self.stream)
            if ii == start:
                self.sequence_start = self.segment_start_time
            ts = self.read()
            data = np.append(data, ts)
            self.seq = ii
        self.logger.debug(f"Read {self.seq + 1} sequences")
        self.data_size = data.size
        return data

    def to_channel_ts(
        self, rxcal_fn: str | Path | None = None, scal_fn: str | Path | None = None
    ) -> ChannelTS:
        """
        Convert to a ChannelTS object.

        Parameters
        ----------
        rxcal_fn : str, Path or None, optional
            Path to receiver calibration file, by default None
        scal_fn : str, Path or None, optional
            Path to sensor calibration file, by default None

        Returns
        -------
        ChannelTS
            Channel time series object with data, metadata, and calibration
        """
        data = self.read_sequence()

        # need to update the start and end times here
        self._channel_metadata.time_period.start = self.sequence_start
        self._channel_metadata.time_period.end = self.sequence_end

        return ChannelTS(
            channel_type=self.channel_metadata.type,
            data=data,
            channel_metadata=self.channel_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            channel_response=self.get_channel_response(
                rxcal_fn=rxcal_fn, scal_fn=scal_fn
            ),
        )
