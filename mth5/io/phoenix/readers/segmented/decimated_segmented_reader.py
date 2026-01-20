# -*- coding: utf-8 -*-
"""
Module to read and parse native Phoenix Geophysics data formats of the
MTU-5C Family

This module implements Streamed readers for segmented-decimated  time series
 formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock
"""


from __future__ import annotations

from pathlib import Path
from struct import unpack_from
from typing import Any, BinaryIO

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from mt_metadata.common.mttime import MTime

from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS


# =============================================================================
class SubHeader:
    """
    Class for subheader of segmented files.

    This class handles the parsing and access to subheader information in
    Phoenix Geophysics segmented time series files. The subheader contains
    metadata about each segment including timing, sample counts, and statistics.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments that are set as attributes

    Attributes
    ----------
    header_length : int
        Length of the subheader in bytes (32 bytes)
    _header : bytes or None
        Raw header bytes from the file
    _unpack_dict : dict
        Dictionary defining how to unpack different header fields
    """

    def __init__(self, **kwargs) -> None:
        self.header_length = 32
        self._header = None

        for key, value in kwargs.items():
            setattr(self, key, value)
        self._unpack_dict = {
            "gps_time_stamp": {"dtype": "I", "index": 0},
            "n_samples": {"dtype": "I", "index": 4},
            "saturation_count": {"dtype": "H", "index": 8},
            "missing_count": {"dtype": "H", "index": 10},
            "value_min": {"dtype": "f", "index": 12},
            "value_max": {"dtype": "f", "index": 16},
            "value_mean": {"dtype": "f", "index": 20},
        }

    def __str__(self) -> str:
        """String representation of the subheader information."""
        lines = ["subheader information:", "-" * 30]
        for key in [
            "gps_time_stamp",
            "n_samples",
            "saturation_count",
            "missing_count",
            "value_min",
            "value_max",
            "value_mean",
        ]:
            lines.append(f"\t{key:<25}: {getattr(self, key)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the subheader."""
        return self.__str__()

    def _has_header(self) -> bool:
        """
        Check if header data has been loaded.

        Returns
        -------
        bool
            True if header is loaded, False otherwise
        """
        if self._header is not None:
            return True
        return False

    def _unpack_value(self, key: str) -> tuple[Any, ...] | None:
        """
        Unpack a value from the header bytes.

        Parameters
        ----------
        key : str
            Key name for the value to unpack

        Returns
        -------
        tuple or None
            Unpacked value tuple, or None if header not available

        Raises
        ------
        IOError
            If unpacking fails
        """
        if self._has_header() and self._header is not None:
            try:
                return unpack_from(
                    self._unpack_dict[key]["dtype"],
                    self._header,
                    self._unpack_dict[key]["index"],
                )
            except Exception as error:
                raise IOError(error)
        return None

    @property
    def gps_time_stamp(self) -> MTime | None:
        """
        GPS time stamp in UTC.

        Returns
        -------
        MTime or None
            GPS timestamp if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("gps_time_stamp")
            if value is not None:
                return MTime(time_stamp=value[0], gps_time=True)
        return None

    @property
    def n_samples(self) -> int | None:
        """
        Number of samples in the segment.

        Returns
        -------
        int or None
            Number of samples if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("n_samples")
            if value is not None:
                return value[0]
        return None

    @property
    def saturation_count(self) -> int | None:
        """
        Number of saturated samples.

        Returns
        -------
        int or None
            Saturation count if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("saturation_count")
            if value is not None:
                return value[0]
        return None

    @property
    def missing_count(self) -> int | None:
        """
        Number of missing samples.

        Returns
        -------
        int or None
            Missing sample count if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("missing_count")
            if value is not None:
                return value[0]
        return None

    @property
    def value_min(self) -> float | None:
        """
        Minimum value in the segment.

        Returns
        -------
        float or None
            Minimum value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_min")
            if value is not None:
                return value[0]
        return None

    @property
    def value_max(self) -> float | None:
        """
        Maximum value in the segment.

        Returns
        -------
        float or None
            Maximum value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_max")
            if value is not None:
                return value[0]
        return None

    @property
    def value_mean(self) -> float | None:
        """
        Mean value in the segment.

        Returns
        -------
        float or None
            Mean value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_mean")
            if value is not None:
                return value[0]
        return None

    def unpack_header(self, stream: BinaryIO) -> None:
        """
        Unpack the header from a binary stream.

        Parameters
        ----------
        stream : BinaryIO
            Binary stream to read header from
        """
        if self.header_length > 0:
            # be sure to read from the beginning of the file
            self._header = stream.read(self.header_length)
        else:
            return


class Segment(SubHeader):
    """
    A segment class to hold a single segment.

    This class represents a single time series segment with its associated
    metadata and data. It inherits from SubHeader to provide access to
    segment-specific header information.

    Parameters
    ----------
    stream : BinaryIO
        Binary file stream to read from
    **kwargs
        Additional keyword arguments passed to SubHeader

    Attributes
    ----------
    stream : BinaryIO
        The file stream for reading data
    data : np.ndarray or None
        Time series data for this segment
    """

    def __init__(self, stream: BinaryIO, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stream = stream
        self.data: np.ndarray | None = None

    def read_segment(self, metadata_only: bool = False) -> None:
        """
        Read the segment data from the file stream.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only read metadata without loading data, by default False
        """
        self.unpack_header(self.stream)
        if not metadata_only and self.n_samples is not None:
            self.data = np.fromfile(self.stream, dtype=np.float32, count=self.n_samples)

    @property
    def segment_start_time(self) -> MTime | None:
        """
        Get the segment start time.

        Returns
        -------
        MTime or None
            GPS timestamp of segment start, or None if not available
        """
        return self.gps_time_stamp

    @property
    def segment_end_time(self) -> MTime | None:
        """
        Calculate the segment end time.

        Returns
        -------
        MTime or None
            Estimated end time based on start time, sample count and sample rate,
            or None if required information is not available
        """
        start_time = self.segment_start_time
        if (
            start_time is not None
            and self.n_samples is not None
            and hasattr(self, "sample_rate")
        ):
            return start_time + (self.n_samples / self.sample_rate)
        return None


class DecimatedSegmentedReader(TSReaderBase):
    """
    Class to create a streamer for segmented decimated time series.

    This reader handles segmented decimated time series files such as 'td_24k'.
    These files have sub headers containing metadata for each segment.

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
    sub_header : SubHeader
        SubHeader instance for parsing segment headers
    subheader : dict
        Dictionary for additional subheader information
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
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segment(self, metadata_only: bool = False) -> Segment:
        """
        Read a single segment from the file.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only read metadata without loading data, by default False

        Returns
        -------
        Segment
            Segment object containing data and metadata

        Raises
        ------
        ValueError
            If stream is not available
        """
        kwargs = {
            "instrument_type": self.instrument_type,
            "instrument_serial_number": self.instrument_serial_number,
            "latitude": self.gps_lat,
            "longitude": self.gps_long,
            "elevation": self.gps_elevation,
            "sample_rate": self.sample_rate,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "segment": 0,
        }

        if self.stream is None:
            raise ValueError("Stream is not available")

        segment = Segment(self.stream, **kwargs)
        segment.read_segment(metadata_only=metadata_only)

        return segment

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
        segment = self.read_segment()
        ch_metadata = self.channel_metadata

        # Set timing information if available
        if segment.segment_start_time is not None:
            ch_metadata.time_period.start = segment.segment_start_time.isoformat()
        if segment.segment_end_time is not None:
            ch_metadata.time_period.end = segment.segment_end_time.isoformat()

        return ChannelTS(
            channel_type=ch_metadata.type,
            data=segment.data,
            channel_metadata=ch_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            channel_response=self.get_channel_response(
                rxcal_fn=rxcal_fn, scal_fn=scal_fn
            ),
        )


class DecimatedSegmentCollection(TSReaderBase):
    """
    Class to read multiple segments from a segmented decimated time series file.

    This reader handles files containing multiple segments of decimated time series
    data such as 'td_24k'. Each segment has its own sub header with metadata.

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
    sub_header : SubHeader
        SubHeader instance for parsing segment headers
    subheader : dict
        Dictionary for additional subheader information
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

        if self.stream is not None:
            self.unpack_header(self.stream)
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segments(self, metadata_only: bool = False) -> list[Segment]:
        """
        Read all segments from the file.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only read metadata without loading data, by default False

        Returns
        -------
        list[Segment]
            List of Segment objects containing data and metadata

        Raises
        ------
        ValueError
            If stream is not available
        """
        kwargs = {
            "instrument_type": self.instrument_type,
            "instrument_serial_number": self.instrument_serial_number,
            "latitude": self.gps_lat,
            "longitude": self.gps_long,
            "elevation": self.gps_elevation,
            "sample_rate": self.sample_rate,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "segment": 0,
        }

        if self.stream is None:
            raise ValueError("Stream is not available")

        segments = []
        count = 1
        while True:
            try:
                kwargs["segment"] = count
                segment = Segment(self.stream, **kwargs)
                segment.read_segment(metadata_only=metadata_only)
                segments.append(segment)
                count += 1
            except Exception:
                break
        self.logger.info(f"Read {count - 1} segments")

        return segments

    def to_channel_ts(
        self, rxcal_fn: str | Path | None = None, scal_fn: str | Path | None = None
    ) -> list[ChannelTS]:
        """
        Convert all segments to ChannelTS objects.

        Parameters
        ----------
        rxcal_fn : str, Path or None, optional
            Path to receiver calibration file, by default None
        scal_fn : str, Path or None, optional
            Path to sensor calibration file, by default None

        Returns
        -------
        list[ChannelTS]
            List of ChannelTS objects, one for each segment
        """
        seq_list = []
        for seq in self.read_segments():
            ch_metadata = self.channel_metadata
            if seq.gps_time_stamp is not None:
                ch_metadata.time_period.start = seq.gps_time_stamp.isoformat()

            seq_list.append(
                ChannelTS(
                    channel_type=ch_metadata.type,
                    data=seq.data,
                    channel_metadata=ch_metadata,
                    run_metadata=self.run_metadata,
                    station_metadata=self.station_metadata,
                    channel_response=self.get_channel_response(
                        rxcal_fn=rxcal_fn, scal_fn=scal_fn
                    ),
                )
            )
        return seq_list
