# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:31:31 2021

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.timeseries import Station, Run


class LEMI424:
    """
    Read in a LEMI424 file, this is a place holder until IRIS finalizes
    their reader.

    """

    def __init__(self, fn=[]):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fn = fn
        self._has_data = False
        self.sample_rate = 1.0
        self.chunk_size = 10000
        self.column_names = [
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
            "tdiff",
        ]

        if self.fn:
            self.read()

    @property
    def num_source_files(self):
        return len(self.fn)

    @property
    def fn(self):
        return self._fn

    @property
    def validate_fn(self):
        """
        Need to check that the filenames are sequential
        """
        return True

    @fn.setter
    def fn(self, value, sort=True):
        """

        Parameters
        ----------
        value:string or pathlib.Path, or list of these

        """
        if isinstance(value, list):
            value = [Path(x) for x in value]
            exists = [x.exists() for x in value]
            for i_fn, cond in enumerate(exists):
                if not cond:
                    raise IOError(f"Could not find {value[i_fn]}")
        elif value is not None:
            value = [
                Path(value),
            ]
            if not value[0].exists():
                raise IOError(f"Could not find {value[0]}")
        if sort:
            value.sort()
        self._fn = value

    @property
    def start(self):
        if self._has_data:
            return "T".join(
                [
                    "-".join(
                        [
                            f"{self._df.iloc[0].year}",
                            f"{self._df.iloc[0].month:02d}",
                            f"{self._df.iloc[0].day:02d}",
                        ]
                    ),
                    ":".join(
                        [
                            f"{self._df.iloc[0].hour:02d}",
                            f"{self._df.iloc[0].minute:02d}",
                            f"{self._df.iloc[0].second:02d}",
                        ]
                    ),
                ]
            )

    @property
    def end(self):
        if self._has_data:
            return "T".join(
                [
                    "-".join(
                        [
                            f"{self._df.iloc[-1].year}",
                            f"{self._df.iloc[-1].month:02d}",
                            f"{self._df.iloc[-1].day:02d}",
                        ]
                    ),
                    ":".join(
                        [
                            f"{self._df.iloc[-1].hour:02d}",
                            f"{self._df.iloc[-1].minute:02d}",
                            f"{self._df.iloc[-1].second:02d}",
                        ]
                    ),
                ]
            )

    @property
    def latitude(self):
        if self._has_data:
            return np.rad2deg(self._df.latitude.median() / 3600)

    @property
    def longitude(self):
        if self._has_data:
            return np.rad2deg(self._df.longitude.median() / 3600)

    @property
    def elevation(self):
        if self._has_data:
            return self._df.elevation.median()

    @property
    def gps_lock(self):
        if self._has_data:
            return self._df.gps_fix.values

    @property
    def station_metadata(self):
        s = Station()
        if self._has_data:
            s.location.latitude = self.latitude
            s.location.longitude = self.longitude
            s.location.elevation = self.elevation
            s.time_period.start = self.start
            s.time_period.end = self.end
        return s

    @property
    def run_metadata(self):
        r = Run()
        r.sample_rate = self.sample_rate
        r.data_logger.model = "LEMI424"
        r.data_logger.manufacturer = "LEMI"
        if self._has_data:
            r.data_logger.power_source.voltage.start = self._df.battery.max()
            r.data_logger.power_source.voltage.end = self._df.battery.min()
            r.time_period.start = self.start
            r.time_period.end = self.end

    def read(self, fn=[]):
        """
        Read a LEMI424 file using pandas

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn:
            self.fn = fn

        exists = [x.exists() for x in self.fn]
        if all(x for x in exists):
            pass
        else:
            msg = "Could not find file %s"
            for i_fn, cond in enumerate(exists):
                if not cond:
                    self.logger.error(msg, self.fn[i_fn])
                    raise IOError(msg % self.fn[i_fn])

        dfs = self.num_source_files * [None]
        for i, fn in enumerate(self.fn):
            dfs[i] = pd.read_csv(fn, delimiter="\s+", names=self.column_names)

        self._df = pd.concat(dfs)
        self._has_data = True

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
            ["bx", "by", "bz"] + e_channels + ["temperature_e", "temperature_h"]
        ):
            if comp[0] in ["h", "b"]:
                ch = ChannelTS("magnetic")
            elif comp[0] in ["e"]:
                ch = ChannelTS("electric")
            else:
                ch = ChannelTS("auxiliary")

            ch.sample_rate = self.sample_rate
            ch.start = self.start
            ch.ts = self._df[comp].values
            ch.component = comp
            ch_list.append(ch)

        return RunTS(
            array_list=ch_list,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
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
