# -*- coding: utf-8 -*-
"""Module to read and parse native Phoenix Geophysics data formats of the MTU-5C Family

This module implements Streamed readers for segmented-decimated continuus-decimated
and native sampling rate time series formats of the MTU-5C family.
"""

__author__ = "Jorge Torres-Solis"

# =============================================================================
# Imports
# =============================================================================

from datetime import datetime
import numpy as np

from struct import unpack_from
from PhoenixGeoPy.readers import TSReaderBase

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
            return unpack_from(
                self._unpack_dict[key]["dtype"],
                self._header,
                self._unpack_dict[key]["index"],
            )

    @property
    def gps_time_stamp(self):
        if self._has_header():
            return self._unpack_value("gps_time_stamp")[0]

    @property
    def gps_time_stamp_isoformat(self):
        return datetime.fromtimestamp(
            self._unpack_value("gps_time_stamp")[0]
        ).isoformat()

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

    def read_segment(self):
        """
        Read the whole file in

        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.unpack_header(self.stream)
        self.data = np.fromfile(self.stream, dtype=np.float32, count=self.n_samples)


class DecimatedSegmentedReader(TSReaderBase):
    """
    Class to create a streamer for segmented decimated time series,
    i.e. *.td_24k

    These files have a sub header

    """

    def __init__(self, path, num_files=1, report_hw_sat=False):
        # Init the base class
        super().__init__(
            path, num_files=num_files, header_length=128, report_hw_sat=report_hw_sat
        )

        self.unpack_header(self.stream)
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segments(self):
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
                segment.read_segment()
                segments.append(segment)

                count += 1
            except:
                break
        print(f"INFO: Read {count - 1} segments")

        return segments

    def read_subheader(self):
        subheaderBytes = self.stream.read(32)
        if not subheaderBytes:
            if self.open_next():
                subheaderBytes = self.stream.read(32)
        if not subheaderBytes or len(subheaderBytes) < 32:
            self.subheader["timestamp"] = 0
            self.subheader["samplesInRecord"] = 0
        else:
            self.subheader["timestamp"] = unpack_from("I", subheaderBytes, 0)[0]
            self.subheader["samplesInRecord"] = unpack_from("I", subheaderBytes, 4)[0]
            self.subheader["satCount"] = unpack_from("H", subheaderBytes, 8)[0]
            self.subheader["missCount"] = unpack_from("H", subheaderBytes, 10)[0]
            self.subheader["minVal"] = unpack_from("f", subheaderBytes, 12)[0]
            self.subheader["maxVal"] = unpack_from("f", subheaderBytes, 16)[0]
            self.subheader["avgVal"] = unpack_from("f", subheaderBytes, 20)[0]

    def read_record_data(self):
        ret_array = np.empty([0])
        self.stream.seek(self.header_length)
        self.read_subheader()
        if (
            self.stream is not None
            and self.subheader["samplesInRecord"] is not None
            and self.subheader["samplesInRecord"] != 0
        ):
            ret_array = np.fromfile(
                self.stream, dtype=np.float32, count=self.subheader["samplesInRecord"]
            )
            if ret_array.size == 0:
                if not self.open_next():
                    return np.empty([0])
                # Array below will contain the data, or will be an np.empty array if end of series as desired
                ret_array = np.fromfile(
                    self.stream,
                    dtype=np.float32,
                    count=self.subheader["samplesInRecord"],
                )
        return ret_array
