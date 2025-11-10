# -*- coding: utf-8 -*-
"""
Module to read and parse native Phoenix Geophysics data formats of the MTU-5C
Family.

This module implements Streamed readers for segmented-decimated time series
formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from struct import unpack, unpack_from

import numpy as np
from numpy.lib.stride_tricks import as_strided

from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS


AD_IN_AD_UNITS = 0
AD_INPUT_VOLTS = 1
INSTRUMENT_INPUT_VOLTS = 2

# =============================================================================


class NativeReader(TSReaderBase):
    """
    Native sampling rate 'Raw' time series reader class.

    This class reads native binary (.bin) files from Phoenix Geophysics MTU-5C
    instruments. The files are formatted with a header of 128 bytes followed by
    frames of 64 bytes each. Each frame contains 20 x 3-byte (24-bit) data
    points plus a 4-byte footer.

    Parameters
    ----------
    path : str or Path
        Path to the time series file
    num_files : int, optional
        Number of files in the sequence, by default 1
    scale_to : int, optional
        Data scaling mode (AD_IN_AD_UNITS, AD_INPUT_VOLTS, or
        INSTRUMENT_INPUT_VOLTS), by default AD_INPUT_VOLTS
    header_length : int, optional
        Length of file header in bytes, by default 128
    last_frame : int, optional
        Last frame number seen by the streamer, by default 0
    ad_plus_minus_range : float, optional
        ADC plus/minus range in volts, by default 5.0
    channel_type : str, optional
        Channel type identifier, by default "E"
    report_hw_sat : bool, optional
        Whether to report hardware saturation, by default False
    **kwargs
        Additional keyword arguments passed to parent TSReaderBase class

    Attributes
    ----------
    last_frame : int
        Last frame number processed
    data_scaling : int
        Current data scaling mode
    ad_plus_minus_range : float
        ADC voltage range
    input_plusminus_range : float
        Input voltage range after gain correction
    scale_factor : float
        Calculated scaling factor for data conversion
    footer_idx_samp_mask : int
        Bit mask for frame index in footer
    footer_sat_mask : int
        Bit mask for saturation count in footer
    """

    def __init__(
        self,
        path: str | Path,
        num_files: int = 1,
        scale_to: int = AD_INPUT_VOLTS,
        header_length: int = 128,
        last_frame: int = 0,
        ad_plus_minus_range: float = 5.0,
        channel_type: str = "E",
        report_hw_sat: bool = False,
        **kwargs,
    ) -> None:
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

    def _calculate_input_plusminus_range(self) -> float:
        """
        Calculate the correct input plusminus range from metadata.

        Returns
        -------
        float
            Input voltage range after accounting for total circuit gain
        """
        return self.ad_plus_minus_range / self.total_circuitry_gain

    def _calculate_data_scaling(self) -> float:
        """
        Calculate the correct data scaling factor for the AD converter.

        Returns
        -------
        float
            Scaling factor for converting raw ADC values to physical units

        Raises
        ------
        LookupError
            If an invalid scaling mode is requested
        """
        if self.data_scaling == AD_IN_AD_UNITS:
            return 256
        elif self.data_scaling == AD_INPUT_VOLTS:
            return self.ad_plus_minus_range / (2**31)
        elif self.data_scaling == INSTRUMENT_INPUT_VOLTS:
            return self.input_plusminus_range / (2**31)
        else:
            raise LookupError("Invalid scaling requested")

    def read_frames(self, num_frames: int) -> np.ndarray:
        """
        Read the given number of frames from the data stream.

        Note
        ----
        The seek position is not reset, so iterating this method will read
        from the last position in the stream.

        Parameters
        ----------
        num_frames : int
            Number of frames to read

        Returns
        -------
        np.ndarray
            Scaled data from the given number of frames with dtype float64
        """
        frames_in_buf = 0
        _idx_buf = 0
        _data_buf = np.empty([num_frames * 20])  # 20 samples packed in a frame

        while frames_in_buf < num_frames:
            if self.stream is not None:
                dataFrame = self.stream.read(64)
            else:
                return np.empty([0])

            if not dataFrame:
                if not self.open_next():
                    return np.empty([0])
                if self.stream is not None:
                    dataFrame = self.stream.read(64)
                if not dataFrame:
                    return np.empty([0])
            dataFooter = unpack_from("I", dataFrame, 60)

            # Check that there are no skipped frames
            frameCount = dataFooter[0] & self.footer_idx_samp_mask
            difCount = frameCount - self.last_frame
            if difCount != 1:
                self.logger.warning(
                    f"Ch [{self.channel_id}] Missing frames at {frameCount} "
                    f"[{difCount}]"
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
                        f"Ch [{self.channel_id}] Frame {frameCount} has {satCount} "
                        "saturations"
                    )
        return _data_buf

    @property
    def npts_per_frame(self) -> int:
        """
        Get the number of data points per frame.

        Returns
        -------
        int
            Number of data points per frame (frame size - 4 footer bytes) / 3 bytes per sample
        """
        if self.frame_size_bytes is not None:
            return int((self.frame_size_bytes - 4) / 3)
        return 20  # Default to 20 if frame_size_bytes is not available

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read the full data file using memory mapping and stride tricks.

        Note
        ----
        This uses numpy.lib.stride_tricks.as_strided which can be unstable
        if the bytes are not the correct length. See notes by numpy.

        The solution is adapted from:
        https://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Scaled time series data and footer data as (data, footer)
        """
        # should do a memory map otherwise things can go badly with as_strided
        raw_data = np.memmap(self.base_path, ">i1", mode="r")
        raw_data = raw_data[self.header_length :]

        # trim of any extra bytes
        extra_bytes = raw_data.size % 64
        if extra_bytes != 0:
            self.logger.warning(f"found {extra_bytes} extra bits in file.")
        useable_bytes = raw_data.size - extra_bytes
        raw_data = raw_data[:useable_bytes]
        # This should now be the exact number of frames after trimming off
        # extra bytes.
        if self.frame_size_bytes is not None:
            n_frames = int(raw_data.size / self.frame_size_bytes)
        else:
            n_frames = int(raw_data.size / 64)  # Default frame size

        # reshape to (nframes, 64)
        frame_size = self.frame_size_bytes if self.frame_size_bytes is not None else 64
        raw_data = raw_data.reshape((n_frames, frame_size))

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

    def read_sequence(
        self, start: int = 0, end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a sequence of files into a single array.

        Parameters
        ----------
        start : int, optional
            Sequence start index, by default 0
        end : int or None, optional
            Sequence end index, by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Scaled time series data and footer data as (data, footer)
            - data: np.ndarray with dtype float32
            - footer: np.ndarray with dtype int32
        """
        data = np.array([])
        footer = np.array([])
        for fn in self.sequence_list[slice(start, end)]:
            self._open_file(fn)
            if self.stream is not None:
                self.unpack_header(self.stream)
            ts, foot = self.read()
            data = np.append(data, ts)
            footer = np.append(footer, foot)
        return data, footer

    def skip_frames(self, num_frames: int) -> bool:
        """
        Skip frames in the data stream.

        Parameters
        ----------
        num_frames : int
            Number of frames to skip

        Returns
        -------
        bool
            True if skip completed successfully, False if end of file reached
        """
        bytes_to_skip = int(num_frames * 64)
        # Python is dumb for seek and tell, it cannot tell us if a seek goes
        # past EOF so instead we need to do inefficient reads to skip bytes
        while bytes_to_skip > 0:
            if self.stream is not None:
                foo = self.stream.read(bytes_to_skip)
                local_read_size = len(foo)
            else:
                return False

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
        data, footer = self.read_sequence()
        ch_metadata = self.channel_metadata
        return ChannelTS(
            channel_type=ch_metadata.type,
            data=data,
            channel_metadata=ch_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            channel_response=self.get_channel_response(
                rxcal_fn=rxcal_fn, scal_fn=scal_fn
            ),
        )
