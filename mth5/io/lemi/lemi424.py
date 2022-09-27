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

# supress the future warning from pandas about using datetime parser.
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import logging
from datetime import datetime

from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.timeseries import Station, Run, Electric, Magnetic, Auxiliary
from mt_metadata.utils.mttime import MTime

# =============================================================================
def lemi_date_parser(year, month, day, hour, minute, second):
    """
    convenience function to combine the date-time columns that are output by
    lemi into a single column

    Assumes UTC

    :param year: year
    :type year: int
    :param month: month
    :type month: int
    :param day: day of the month
    :type day: int
    :param hour: hour in 24 hr format
    :type hour: int
    :param minute: minutes in the hour
    :type minute: int
    :param second: seconds in the minute
    :type second: int
    :return: date time as a single column
    :rtype: :class:`pandas.DateTime`

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
    convenience function to parse the location strings into a decimal float
    Uses the hemisphere for the sign.

    .. note:: the format of the location is odd in that it is multiplied by
     100 within the LEMI to provide a single floating point value that
     includes the degrees and decimal degrees --> {degrees}{degrees[mm.ss]}.
     For example 40.50166 would be represented as 4030.1.

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
    convert hemisphere into a value [-1, 1].  Assumes the prime meridian is 0.

    :param hemisphere: hemisphere string [ 'N' | 'S' | 'E' | 'W']
    :type hemisphere: string
    :return: unity with a sign for the given hemisphere
    :rtype: signed integer

    """
    if hemisphere in ["S", "W"]:
        return -1
    return 1


class LEMI424:
    """
    Read in a LEMI424 file, this is a place holder until IRIS finalizes
    their reader.

    :param fn: full path to LEMI424 file
    :type fn: :class:`pathlib.Path` or string
    :param sample_rate: sample rate of the file, default is 1.0
    :type sample_rate: float
    :param chunk_size: chunk size for pandas to use, does not change reading
     time much for a single day file. default is 8640
    :type chunk_size: integer
    :param file_column_names: column names of the LEMI424 file
    :type file_column_names: list of strings
    :param dtypes: data types for each column
    :type dtypes: dictionary with keys of column names and values of data types
    :param data_column_names: same as file_column names with and added column
     for date, which is the combined date and time columns.
    :type data_column_names: dictionary with keys of column names and values
     of data types

    :LEMI424 File Column Names:

        - **year**
        - **month**
        - **day**
        - **hour**
        - **minute**
        - **second**
        - **bx**
        - **by**
        - **bz**
        - **temperature_e**
        - **temperature_h**
        - **e1**
        - **e2**
        - **e3**
        - **e4**
        - **battery**
        - **elevation**
        - **latitude**
        - **lat_hemisphere**
        - **longitude**
        - **lon_hemisphere**
        - **n_satellites**
        - **gps_fix**
        - **time_diff**

    :Data Column Names:

        - **date**
        - **bx**
        - **by**
        - **bz**
        - **temperature_e**
        - **temperature_h**
        - **e1**
        - **e2**
        - **e3**
        - **e4**
        - **battery**
        - **elevation**
        - **latitude**
        - **lat_hemisphere**
        - **longitude**
        - **lon_hemisphere**
        - **n_satellites**
        - **gps_fix**
        - **time_diff**

    """

    def __init__(self, fn=None, **kwargs):
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
        """
        Can append other LEMI424 objects together as long as the start and
        end times match up.

        """
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
        """
        Data represented as a :class:`pandas.DataFrame` with data_column names

        """
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
        """check to see if has data or not"""
        if self.data is not None:
            return True
        return False

    @property
    def fn(self):
        """full path to LEMI424 file"""
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
        """size of file in bytes"""
        if self.fn is not None:
            return self.fn.stat().st_size

    @property
    def start(self):
        """start time of data collection in the LEMI424 file"""
        if self._has_data():
            return MTime(self.data.index[0])

    @property
    def end(self):
        """end time of data collection in the LEMI424 file"""
        if self._has_data():
            return MTime(self.data.index[-1])

    @property
    def latitude(self):
        """median latitude where data have been collected in the LEMI424 file"""
        if self._has_data():

            return (
                self.data.latitude.median() * self.data.lat_hemisphere.median()
            )

    @property
    def longitude(self):
        """median longitude where data have been collected in the LEMI424 file"""
        if self._has_data():
            return (
                self.data.longitude.median()
                * self.data.lon_hemisphere.median()
            )

    @property
    def elevation(self):
        """median elevation where data have been collected in the LEMI424 file"""
        if self._has_data():
            return self.data.elevation.median()

    @property
    def n_samples(self):
        """number of samples in the file"""
        if self._has_data():
            return self.data.shape[0]
        elif self.fn is not None and self.fn.exists():
            return round(self.fn.stat().st_size / 152.0)

    @property
    def gps_lock(self):
        """has GPS lock"""
        if self._has_data():
            return self.data.gps_fix.values

    @property
    def station_metadata(self):
        """station metadata as :class:`mt_metadata.timeseries.Station`"""
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
        """run metadata as :class:`mt_metadata.timeseries.Run`"""
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

    def read(self, fn=None, fast=True):
        """
        Read a LEMI424 file using pandas.  The `fast` way will read in the
        first and last line to get the start and end time to make a time index.
        Then it will read in the data skipping parsing the date time columns.
        It will check to make sure the expected amount of points are correct.
        If not then it will read in the slower way which used the date time
        parser to ensure any time gaps are respected.

        :param fn: full path to file, defaults to None.  Uses LEMI424.fn if
         not provided
        :type fn: string or :class:`pathlib.Path`, optional
        :param fast: read the fast way (True) or not (False)
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is not None:
            self.fn = fn

        if not self.fn.exists():
            msg = "Could not find file %s"
            self.logger.error(msg, self.fn)
            raise IOError(msg % self.fn)

        if fast:
            try:
                self.read_metadata()
                data = pd.read_csv(
                    self.fn,
                    delimiter="\s+",
                    names=self.file_column_names,
                    dtype=self.dtypes,
                    usecols=(
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
                    ),
                    converters={
                        "latitude": lemi_position_parser,
                        "longitude": lemi_position_parser,
                        "lat_hemisphere": lemi_hemisphere_parser,
                        "lon_hemisphere": lemi_hemisphere_parser,
                    },
                )
                time_index = pd.date_range(
                    start=self.start.iso_no_tz,
                    end=self.end.iso_no_tz,
                    freq="1000000000N",
                )
                if time_index.size != data.shape[0]:
                    raise ValueError(
                        "Missing a time stamp use read with fast=False"
                    )

                data.index = time_index
                self.data = data
                return
            except ValueError:
                self.logger.warning(
                    "Data is missing a time stamp, reading in slow mode"
                )

        # read in chunks, this doesnt really speed up much as most of the
        # compute time is used in the date time parsing.
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
            self.logger.debug(
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
            self.logger.debug(
                f"Reading {self.fn.name} took {et - st:.2f} seconds"
            )

    def read_metadata(self):
        """
        Read only first and last rows to get important metadata to use in
        the collection.

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
        Create a :class:`mth5.timeseries.RunTS` object from the data

        :param fn: full path to file, defaults to None.  Will use LEMI424.fn
         if None.
        :type fn: string or :class:`pathlib.Path`, optional
        :param e_channels: columns for the electric channels to use,
         defaults to ["e1", "e2"]
        :type e_channels: list of strings, optional
        :return: RunTS object
        :rtype: :class:`mth5.timeseries.RunTS`

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
def read_lemi424(
    fn, e_channels=["e1", "e2"], fast=True, logger_file_handler=None
):
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

    if not isinstance(fn, (list, tuple)):
        fn = [fn]

    txt_obj = LEMI424(fn[0])
    txt_obj.read(fast=fast)

    if logger_file_handler:
        txt_obj.logger.addHandler(logger_file_handler)

    # read a list of files into a single run
    if len(fn) > 1:
        for txt_file in fn:
            other = LEMI424(txt_file)
            other.read()
            txt_obj += other

    return txt_obj.to_run_ts(e_channels=e_channels)
