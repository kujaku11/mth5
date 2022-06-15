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
from numpy.lib.stride_tricks import as_strided

from struct import unpack_from, unpack
from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS

AD_IN_AD_UNITS = 0
AD_INPUT_VOLTS = 1
INSTRUMENT_INPUT_VOLTS = 2

# =============================================================================


class NativeReader(TSReaderBase):
    """
    Native sampling rate 'Raw' time series reader class, these are the .bin
    files.  They are formatted with a header of 128 bytes then frames of 64.

    Each frame is 20 x 3 byte (24-bit) data point then a 4 byte footer.


    """

    def __init__(
        self,
        path,
        num_files=1,
        scale_to=AD_INPUT_VOLTS,
        header_length=128,
        last_frame=0,
        ad_plus_minus_range=5.0,
        channel_type="E",
        report_hw_sat=False,
        **kwargs,
    ):
        # Init the base class
        super().__init__(path, num_files, header_length, report_hw_sat, **kwargs)

        self._chunk_size = 4096

        # Track the last frame seen by the streamer, to report missing frames
        self.last_frame = last_frame
        self.header_length = header_length
        self.data_scaling = scale_to
        self.ad_plus_minus_range = ad_plus_minus_range

        if header_length == 128:
            self.unpack_header(self.stream)
        # Now that we have the channel circuit-based gain (either form init
        # or from the header)
        # We can calculate the voltage range at the input of the board.
        self.input_plusminus_range = self._calculate_input_plusminus_range()

        self.scale_factor = self._calculate_data_scaling()

        # optimization variables
        self.footer_idx_samp_mask = int("0x0fffffff", 16)
        self.footer_sat_mask = int("0x70000000", 16)

    def _calculate_input_plusminus_range(self):
        """
        set the correct input plusminus range from metadata
        """
        return self.ad_plus_minus_range / self.total_circuitry_gain

    def _calculate_data_scaling(self):
        """
        Get the correct data scaling for the AD converter
        """
        if self.data_scaling == AD_IN_AD_UNITS:
            return 256
        elif self.data_scaling == AD_INPUT_VOLTS:
            return self.ad_plus_minus_range / (2 ** 31)
        elif self.data_scaling == INSTRUMENT_INPUT_VOLTS:
            return self.input_plusminus_range / (2 ** 31)
        else:
            raise LookupError("Invalid scaling requested")

    def read_frames(self, num_frames):
        """
        Read the given amount of frames from the data.

        .. note:: that seek is not reset so if you iterate this the stream
        reads from the last tell.

        :param num_frames: Number of frames to read
        :type num_frames: integer
        :return: Scaled data from the given number of frames
        :rtype: np.ndarray(dtype=float)

        """

        frames_in_buf = 0
        _idx_buf = 0
        _data_buf = np.empty([num_frames * 20])  # 20 samples packed in a frame

        while frames_in_buf < num_frames:

            dataFrame = self.stream.read(64)
            if not dataFrame:
                if not self.open_next():
                    return np.empty([0])
                dataFrame = self.stream.read(64)
                if not dataFrame:
                    return np.empty([0])
            dataFooter = unpack_from("I", dataFrame, 60)

            # Check that there are no skipped frames
            frameCount = dataFooter[0] & self.footer_idx_samp_mask
            difCount = frameCount - self.last_frame
            if difCount != 1:
                self.logger.warning(
                    "Ch [%s] Missing frames at %d [%d]\n"
                    % (self.channel_id, frameCount, difCount)
                )
            self.last_frame = frameCount

            for ptrSamp in range(0, 60, 3):
                # unpack expects 4 bytes, but the frames are only 3?
                value = unpack(">i", dataFrame[ptrSamp : ptrSamp + 3] + b"\x00")[0]
                _data_buf[_idx_buf] = value * self.scale_factor
                _idx_buf += 1
            frames_in_buf += 1

            if self.report_hw_sat:
                satCount = (dataFooter[0] & self.footer_sat_mask) >> 24
                if satCount:
                    self.logger.warning(
                        "Ch [%s] Frame %d has %d saturations"
                        % (self.ch_id, frameCount, satCount)
                    )
        return _data_buf

    @property
    def npts_per_frame(self):
        return int((self.frame_size_bytes - 4) / 3)

    def read(self):
        """
        Read the full data file.

        .. note:: This uses :class:`numpy.lib.stride_tricks.as_strided` which
        can be unstable if the bytes are not the correct length.  See notes by
        numpy.

        Got this solution from:
        https://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers?msclkid=3398046ecd6511ec9a37394f28c5aaba

        :return: scaled data and footer
        :rtype: tuple (data, footer)

        """

        # should do a memory map otherwise things can go badly with as_strided
        raw_data = np.memmap(self.base_path, ">i1", mode="r")
        raw_data = raw_data[self.header_length :]

        # trim of any extra bytes
        extra_bytes = raw_data.size % 64
        if extra_bytes != 0:
            self.warning(f"found {extra_bytes} extra bits in file.")
        useable_bytes = raw_data.size - extra_bytes
        raw_data = raw_data[:useable_bytes]
        # This should now be the exact number of frames after trimming off
        # extra bytes.
        n_frames = int(raw_data.size / self.frame_size_bytes)

        # reshape to (nframes, 64)
        raw_data = raw_data.reshape((n_frames, self.frame_size_bytes))

        # split the data from the footer
        ts = raw_data[:, 0 : self.npts_per_frame * 3].flatten()
        footer = raw_data[:, self.npts_per_frame * 3 :].flatten()

        # get the number of raw byte frames
        raw_frames = int(ts.size / 12)

        # stride over bytes making new 4 bytes for a 32bit integer and scale
        ts_data = (
            as_strided(
                ts.view(np.int32),
                strides=(12, 3),
                shape=(raw_frames, 4),
            )
            .flatten()
            .byteswap()
            * self.scale_factor
        )
        # somehow the number is off by just a bit ~1E-7 V

        # view the footer as an int32
        footer = footer.view(np.int32)

        return ts_data, footer

    def read_sequence(self, start=0, end=None):
        """
        Read sequence of files into a single array

        :param start: sequence start, defaults to 0
        :type start: integer, optional
        :param end: sequence end, defaults to None
        :type end: integer, optional
        :return: scaled data
        :rtype: np.ndarray(dtype=float32)
        :return: footer
        rtype: np.ndarray(dtype=int32)

        """

        data = np.array([])
        footer = np.array([])
        for fn in self.sequence_list[slice(start, end)]:
            self._open_file(fn)
            self.unpack_header(self.stream)
            ts, foot = self.read()
            data = np.append(data, ts)
            footer = np.append(footer, foot)
        return data, footer

    def skip_frames(self, num_frames):
        """
        Skip frames of the stream

        :param num_frames: number of frames to skip
        :type num_frames: integer
        :return: end of file
        :rtype: boolean

        """
        bytes_to_skip = int(num_frames * 64)
        # Python is dumb for seek and tell, it cannot tell us if a seek goes
        # past EOF so instead we need to do inefficient reads to skip bytes
        while bytes_to_skip > 0:
            foo = self.stream.read(bytes_to_skip)
            local_read_size = len(foo)

            # If we ran out of data in this file before finishing the skip,
            # open the next file and return false if there is no next file
            # to indicate that the skip ran out of
            # data before completion
            if local_read_size == 0:
                more_data = self.open_next()
                if not more_data:
                    return False
            else:
                bytes_to_skip -= local_read_size
        # If we reached here we managed to skip all the data requested
        # return true
        self.last_frame += num_frames
        return True

    def to_channel_ts(self):
        """
        convert to a ChannelTS object

        :return: DESCRIPTION
        :rtype: TYPE

        """
        data, footer = self.read()
        ch_metadata = self.channel_metadata()
        return ChannelTS(
            channel_type=ch_metadata.type,
            data=data,
            channel_metadata=ch_metadata,
            run_metadata=self.run_metadata(),
            station_metadata=self.station_metadata(),
        )
