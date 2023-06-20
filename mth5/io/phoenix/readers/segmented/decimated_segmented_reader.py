# -*- coding: utf-8 -*-
"""
Module to read and parse native Phoenix Geophysics data formats of the 
MTU-5C Family

This module implements Streamed readers for segmented-decimated  time series
 formats of the MTU-5C family.
 
:author: Jorge Torres-Solis

Revised 2022 by J. Peacock 
"""


# =============================================================================
# Imports
# =============================================================================
import numpy as np

from struct import unpack_from

from mt_metadata.utils.mttime import MTime
from mth5.io.phoenix.readers import TSReaderBase
from mth5.timeseries import ChannelTS

# =============================================================================
class SubHeader:
    """
    Class for subheader of segmented files
    """

    def __init__(self, **kwargs):
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

    def __str__(self):
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

    def __repr__(self):
        return self.__str__()

    def _has_header(self):
        if self._header is not None:
            return True
        return False

    def _unpack_value(self, key):
        if self._has_header():
            try:
                return unpack_from(
                    self._unpack_dict[key]["dtype"],
                    self._header,
                    self._unpack_dict[key]["index"],
                )
            except Exception as error:
                raise IOError(error)

    @property
    def gps_time_stamp(self):
        if self._has_header():
            return MTime(self._unpack_value("gps_time_stamp")[0])

    @property
    def n_samples(self):
        if self._has_header():
            return self._unpack_value("n_samples")[0]

    @property
    def saturation_count(self):
        if self._has_header():
            return self._unpack_value("saturation_count")[0]

    @property
    def missing_count(self):
        if self._has_header():
            return self._unpack_value("missing_count")[0]

    @property
    def value_min(self):
        if self._has_header():
            return self._unpack_value("value_min")[0]

    @property
    def value_max(self):
        if self._has_header():
            return self._unpack_value("value_max")[0]

    @property
    def value_mean(self):
        if self._has_header():
            return self._unpack_value("value_mean")[0]

    def unpack_header(self, stream):
        if self.header_length > 0:
            # be sure to read from the beginning of the file
            self._header = stream.read(self.header_length)
        else:
            return


class Segment(SubHeader):
    """
    A segment class to hold a single segment
    """

    def __init__(self, stream, **kwargs):

        super().__init__(**kwargs)
        self.stream = stream
        self.data = None

    def read_segment(self, metadata_only=False):
        """
        Read the whole file in

        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.unpack_header(self.stream)
        if not metadata_only:
            self.data = np.fromfile(
                self.stream, dtype=np.float32, count=self.n_samples
            )

    @property
    def segment_start_time(self):
        """
        estimate the segment start time based on sequence number

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.gps_time_stamp

    @property
    def segment_end_time(self):
        """
        estimate end time

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.segment_start_time + ((self.n_samples) / self.sample_rate)


class DecimatedSegmentedReader(TSReaderBase):
    """
    Class to create a streamer for segmented decimated time series,
    i.e. 'td_24k'

    These files have a sub header

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
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segment(self, metadata_only=False):
        """
        Read in a single segment

        :param metadata_only: DESCRIPTION, defaults to False
        :type metadata_only: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

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

        segment = Segment(self.stream, **kwargs)
        segment.read_segment(metadata_only=metadata_only)

        return segment

    def to_channel_ts(self, rxcal_fn=None, scal_fn=None):
        """
        convert to a ChannelTS object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        segment = self.read_segment()
        ch_metadata = self.channel_metadata
        ch_metadata.time_period.start = segment.segment_start_time.isoformat()

        return ChannelTS(
            channel_type=ch_metadata.type,
            data=segment.data,
            channel_metadata=ch_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            channel_response_filter=self.get_channel_response_filter(
                rxcal_fn=rxcal_fn, scal_fn=scal_fn
            ),
        )


class DecimatedSegmentCollection(TSReaderBase):
    """
    Class to create a streamer for segmented decimated time series,
    i.e. 'td_24k'

    These files have a sub header

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
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segments(self, metadata_only=False):
        """
        Read the whole file in

        :return: DESCRIPTION
        :rtype: TYPE

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
        segments = []
        count = 1
        while True:
            try:
                kwargs["segment"] = count
                segment = Segment(self.stream, **kwargs)
                segment.read_segment(metadata_only=metadata_only)
                segments.append(segment)

                count += 1
            except:
                break
        self.logger.info(f"Read {count - 1} segments")

        return segments

    def to_channel_ts(self, rxcal_fn=None, scal_fn=None):
        """
        convert to a ChannelTS object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        seq_list = []
        for seq in self.read_segments():
            ch_metadata = self.channel_metadata
            ch_metadata.time_period.start = seq.gps_time_stamp.isoformat()

            seq_list.append(
                ChannelTS(
                    channel_type=ch_metadata.type,
                    data=seq.data,
                    channel_metadata=ch_metadata,
                    run_metadata=self.run_metadata,
                    station_metadata=self.station_metadata,
                    channel_response_filter=self.get_channel_response_filter(
                        rxcal_fn=rxcal_fn, scal_fn=scal_fn
                    ),
                )
            )
        return seq_list
