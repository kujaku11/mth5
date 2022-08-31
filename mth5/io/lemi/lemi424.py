# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:31:31 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from io import StringIO
import warnings
from copy import deepcopy

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.timeseries import Station, Run, Electric, Magnetic, Auxiliary
from mt_metadata.utils.mttime import MTime

# =============================================================================
def lemi_date_parser(year, month, day, hour, minute, second):
    """
    convenience function to parse the date-time columns that are output by
    lemi

    :param year: DESCRIPTION
    :type year: TYPE
    :param month: DESCRIPTION
    :type month: TYPE
    :param day: DESCRIPTION
    :type day: TYPE
    :param hour: DESCRIPTION
    :type hour: TYPE
    :param minute: DESCRIPTION
    :type minute: TYPE
    :param second: DESCRIPTION
    :type second: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    return pd.to_datetime(
        datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
        )
    )


def lemi_position_parser(position):
    """
    convenience function to parse the location strings
    :param position: DESCRIPTION
    :type position: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    pos = f"{float(position) / 100}".split(".")
    degrees = int(pos[0])
    decimals = float(f"{pos[1][0:2]}.{pos[1][2:]}") / 60

    location = degrees + decimals
    return location


def lemi_hemisphere_parser(hemisphere):
    """
    convert hemisphere into a value [-1, 1]

    :param hemisphere: DESCRIPTION
    :type hemisphere: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if hemisphere in ["S", "W"]:
        return -1
    return 1


class LEMI424:
    """
    Read in a LEMI424 file, this is a place holder until IRIS finalizes
    their reader.

    """

    def __init__(self, fn=None):
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.fn = fn
        self.sample_rate = 1.0
        self.chunk_size = 8640
        self.data = None
        self.file_column_names = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "bx",
            "by",
            "bz",
            "temperature_e",
            "temperature_h",
            "e1",
            "e2",
            "e3",
            "e4",
            "battery",
            "elevation",
            "latitude",
            "lat_hemisphere",
            "longitude",
            "lon_hemisphere",
            "n_satellites",
            "gps_fix",
            "time_diff",
        ]

        self.dtypes = dict(
            [
                ("year", int),
                ("month", int),
                ("day", int),
                ("hour", int),
                ("minute", int),
                ("second", int),
                ("bx", float),
                ("by", float),
                ("bz", float),
                ("temperature_e", float),
                ("temperature_h", float),
                ("e1", float),
                ("e2", float),
                ("e3", float),
                ("e4", float),
                ("battery", float),
                ("elevation", float),
                ("n_satellites", int),
                ("gps_fix", int),
                ("time_diff", float),
            ]
        )

        self.data_column_names = ["date"] + self.file_column_names[6:]

    def __str__(self):
        lines = ["LEMI 424 data", "-" * 20]
        lines.append(f"start:      {self.start.isoformat()}")
        lines.append(f"end:        {self.end.isoformat()}")
        lines.append(f"N samples:  {self.n_samples}")
        lines.append(f"latitude:   {self.latitude} (degrees)")
        lines.append(f"longitude:  {self.longitude} (degrees)")
        lines.append(f"elevation:  {self.elevation} m")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if not self._has_data():
            raise ValueError("Data is None cannot append to. Read file first")

        if isinstance(other, LEMI424):
            new = LEMI424()
            new.__dict__.update(self.__dict__)
            new.data = pd.concat([new.data, other.data])
            return new

        elif isinstance(other, pd.DataFrame):

            if not other.columns != self.data.columns:
                raise ValueError("DataFrame columns are not the same.")

            new = LEMI424()
            new.__dict__.update(self.__dict__)
            new.data = pd.concat([new.data, other])
            return new

        else:
            raise ValueError(f"Cannot add {type(other)} to pd.DataFrame.")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Make sure if data is set that it is a pandas data frame with the proper
        column names
        """

        if data is None:
            self._data = None

        elif isinstance(data, pd.DataFrame):
            if len(data.columns) == len(self.file_column_names):
                if data.columns != self.file_column_names:
                    raise ValueError(
                        "Column names are not the same. "
                        "Check LEMI424.column_names for accepted names"
                    )
            elif len(data.columns) == len(self.data_column_names):
                if not data.columns == self.data_column_names:
                    raise ValueError(
                        "Column names are not the same. "
                        "Check LEMI424.column_names for accepted names"
                    )
            else:
                self._data = data

    def _has_data(self):
        if self.data is not None:
            return True
        return False

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        if value is not None:
            value = Path(value)
            if not value.exists():
                raise IOError(f"Could not find {value}")
        self._fn = value

    @property
    def file_size(self):
        if self.fn is not None:
            return self.fn.stat().st_size

    @property
    def start(self):
        if self._has_data():
            return MTime(self.data.index[0])

    @property
    def end(self):
        if self._has_data():
            return MTime(self.data.index[-1])

    @property
    def latitude(self):
        if self._has_data():

            return (
                self.data.latitude.median() * self.data.lat_hemisphere.median()
            )

    @property
    def longitude(self):
        if self._has_data():
            return (
                self.data.longitude.median()
                * self.data.lon_hemisphere.median()
            )

    @property
    def elevation(self):
        if self._has_data():
            return self.data.elevation.median()

    @property
    def n_samples(self):
        if self._has_data():
            return self.data.shape[0]
        elif self.fn is not None and self.fn.exists():
            return round(self.fn.stat().st_size / 152.9667)

    @property
    def gps_lock(self):
        if self._has_data():
            return self.data.gps_fix.values

    @property
    def station_metadata(self):
        s = Station()
        if self._has_data():
            s.location.latitude = self.latitude
            s.location.longitude = self.longitude
            s.location.elevation = self.elevation
            s.time_period.start = self.start
            s.time_period.end = self.end
            s.add_run(self.run_metadata)
        return s

    @property
    def run_metadata(self):
        r = Run()
        r.id = "a"
        r.sample_rate = self.sample_rate
        r.data_logger.model = "LEMI424"
        r.data_logger.manufacturer = "LEMI"
        if self._has_data():
            r.data_logger.power_source.voltage.start = self.data.battery.max()
            r.data_logger.power_source.voltage.end = self.data.battery.min()
            r.time_period.start = self.start
            r.time_period.end = self.end

            for ch_aux in ["temperature_e", "temperature_h"]:
                r.add_channel(Auxiliary(component=ch_aux))
            for ch_e in ["e1", "e2"]:
                r.add_channel(Electric(component=ch_e))
            for ch_h in ["bx", "by", "bz"]:
                r.add_channel(Magnetic(component=ch_h))

        return r

    def read(self, fn=None):
        """
        Read a LEMI424 file using pandas

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is not None:
            self.fn = fn

        if not self.fn.exists():
            msg = "Could not find file %s"
            self.logger.error(msg, self.fn)
            raise IOError(msg % self.fn)

        # tried reading in chunks and got Nan's and was took just as long
        # maybe someone smarter can figure it out.
        if self.n_samples > self.chunk_size:
            st = MTime().now()
            dfs = list(
                pd.read_csv(
                    self.fn,
                    delimiter="\s+",
                    names=self.file_column_names,
                    dtype=self.dtypes,
                    parse_dates={
                        "date": [
                            "year",
                            "month",
                            "day",
                            "hour",
                            "minute",
                            "second",
                        ]
                    },
                    date_parser=lemi_date_parser,
                    converters={
                        "latitude": lemi_position_parser,
                        "longitude": lemi_position_parser,
                        "lat_hemisphere": lemi_hemisphere_parser,
                        "lon_hemisphere": lemi_hemisphere_parser,
                    },
                    index_col="date",
                    chunksize=self.chunk_size,
                )
            )

            self.data = pd.concat(dfs)
            et = MTime().now()
            self.logger.info(
                f"Reading {self.fn.name} took {et - st:.2f} seconds"
            )

        else:
            st = MTime().now()
            self.data = pd.read_csv(
                self.fn,
                delimiter="\s+",
                names=self.file_column_names,
                dtype=self.dtypes,
                parse_dates={
                    "date": [
                        "year",
                        "month",
                        "day",
                        "hour",
                        "minute",
                        "second",
                    ]
                },
                date_parser=lemi_date_parser,
                converters={
                    "latitude": lemi_position_parser,
                    "longitude": lemi_position_parser,
                    "lat_hemisphere": lemi_hemisphere_parser,
                    "lon_hemisphere": lemi_hemisphere_parser,
                },
                index_col="date",
            )
            et = MTime().now()
            self.logger.info(
                f"Reading {self.fn.name} took {et - st:.2f} seconds"
            )

    def read_metadata(self):
        """
        Read only first and last rows

        :return: DESCRIPTION
        :rtype: TYPE

        """

        with open(self.fn) as fid:
            first_line = fid.readline()
            for line in fid:
                pass  # iterate to the end
            last_line = line

        lines = StringIO(f"{first_line}\n{last_line}")

        self.data = pd.read_csv(
            lines,
            delimiter="\s+",
            names=self.file_column_names,
            dtype=self.dtypes,
            parse_dates={
                "date": [
                    "year",
                    "month",
                    "day",
                    "hour",
                    "minute",
                    "second",
                ]
            },
            date_parser=lemi_date_parser,
            converters={
                "latitude": lemi_position_parser,
                "longitude": lemi_position_parser,
                "lat_hemisphere": lemi_hemisphere_parser,
                "lon_hemisphere": lemi_hemisphere_parser,
            },
            index_col="date",
        )

    def to_run_ts(self, fn=None, e_channels=["e1", "e2"]):
        """
        Return a RunTS object from the data

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_list = []
        for comp in (
            ["bx", "by", "bz"]
            + e_channels
            + ["temperature_e", "temperature_h"]
        ):
            if comp[0] in ["h", "b"]:
                ch = ChannelTS("magnetic")
            elif comp[0] in ["e"]:
                ch = ChannelTS("electric")
            else:
                ch = ChannelTS("auxiliary")

            ch.sample_rate = self.sample_rate
            ch.start = self.start
            ch.ts = self.data[comp].values
            ch.component = comp
            ch_list.append(ch)

        run_metadata = deepcopy(self.run_metadata)
        run_metadata.channels = []

        station_metadata = deepcopy(self.station_metadata)
        station_metadata.runs = []
        return RunTS(
            array_list=ch_list,
            station_metadata=station_metadata,
            run_metadata=run_metadata,
        )


# =============================================================================
# define the reader
# =============================================================================
def read_lemi424(fn, e_channels=["e1", "e2"], logger_file_handler=None):
    """
    Read a LEMI 424 TXT file.

    :param fn: input file name
    :type fn: string or Path
    :param e_channels: A list of electric channels to read,
    defaults to ["e1", "e2"]
    :type e_channels: list of strings, optional
    :return: A RunTS object with appropriate metadata
    :rtype: :class:`mth5.timeseries.RunTS`

    """

    txt_obj = LEMI424()
    if logger_file_handler:
        txt_obj.logger.addHandler(logger_file_handler)
    txt_obj.read(fn)
    return txt_obj.to_run_ts(e_channels=e_channels)
