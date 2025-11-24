# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:54:09 2020

:author: Jared Peacock

:license: MIT

"""

import json
from collections import OrderedDict

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from loguru import logger
from mt_metadata.timeseries import Electric, Magnetic, Run, Station, Survey


# =============================================================================
#  Metadata for usgs ascii file
# =============================================================================


class AsciiMetadata:
    """
    Container for all the important metadata in a USGS ascii file.

    ========================= =================================================
    Attributes                Description
    ========================= =================================================
    survey_id                  Survey name
    site_id                    Site name
    run_id                     Run number
    site_latitude              Site latitude in decimal degrees WGS84
    site_longitude             Site longitude in decimal degrees WGS84
    site_elevation             Site elevation according to national map meters
    start                      Start time of station YYYY-MM-DDThh:mm:ss UTC
    end                        Stop time of station YYYY-MM-DDThh:mm:ss UTC
    sample_rate                Sampling rate samples/second
    n_samples                  Number of samples
    n_channels                 Number of channels
    coordinate_system          [ Geographic North | Geomagnetic North ]
    chn_settings               Channel settings, see below
    missing_data_flag          Missing data value
    ========================= =================================================

    :chn_settings:

    ========================= =================================================
    Keys                      Description
    ========================= =================================================
    ChnNum                    site_id+channel number
    ChnID                     Component [ ex | ey | hx | hy | hz ]
    InstrumentID              Data logger + sensor number
    Azimuth                   Setup angle of componet in degrees relative to
                              coordinate_system
    Dipole_Length             Dipole length in meters
    ========================= =================================================


    """

    def __init__(self, fn=None, **kwargs):
        self.logger = logger

        self.fn = fn
        self.missing_data_flag = np.nan
        self.coordinate_system = None
        self._metadata_len = 30
        from mt_metadata.common import DataTypeEnum, TimePeriodDate

        # Survey, Station, Run: instantiate and set required fields directly
        self._survey_metadata = Survey()
        self._survey_metadata.id = ""
        self._survey_metadata.datum = "WGS 84"
        self._survey_metadata.geographic_name = ""
        self._survey_metadata.name = ""
        self._survey_metadata.project = ""
        self._survey_metadata.summary = ""
        self._survey_metadata.time_period = TimePeriodDate(
            start_date="1970-01-01", end_date="1970-01-02"
        )

        self._station_metadata = Station()
        self._station_metadata.id = ""
        self._station_metadata.channels_recorded = []
        self._station_metadata.geographic_name = ""
        self._station_metadata.run_list = []
        self._station_metadata.data_type = DataTypeEnum.BBMT

        self._run_metadata = Run()
        self._run_metadata.id = ""
        self._run_metadata.sample_rate = 0.0
        self._run_metadata.channels_recorded_auxiliary = []
        self._run_metadata.channels_recorded_electric = []
        self._run_metadata.channels_recorded_magnetic = []
        self._run_metadata.data_type = DataTypeEnum.BBMT

        # Electric required fields (all base fields)
        self.ex_metadata = Electric(component="ex")  # type: ignore
        self.ey_metadata = Electric(component="ey")  # type: ignore

        self.hx_metadata = Magnetic(component="hx")  # type: ignore
        self.hy_metadata = Magnetic(component="hy")  # type: ignore
        self.hz_metadata = Magnetic(component="hz")  # type: ignore

        self.channel_order = ["hx", "ex", "hy", "ey", "hz"]

        self.national_map_url = r"https://epqs.nationalmap.gov/v1/json?x={0:.5f}&y={1:.5f}&units=Meters&wkid=4326&includeDate=False"

        self._key_dict = OrderedDict(
            **{
                "SurveyID": "survey_id",
                "SiteID": "site_id",
                "RunID": "run_id",
                "SiteLatitude": "latitude",
                "SiteLongitude": "longitude",
                "SiteElevation": "elevation",
                "AcqStartTime": "start",
                "AcqStopTime": "end",
                "AcqSmpFreq": "sample_rate",
                "AcqNumSmp": "n_samples",
                "Nchan": "n_channels",
                "Channel coordinates relative to geographic north": "",
                "ChnSettings": "",
                "MissingDataFlag": "missing_data_flag",
                "DataSet": "data_set",
            }
        )

        self._chn_dict = OrderedDict(
            **{
                "ChnNum": "channel_number",
                "ChnID": "component",
                "InstrumentID": "sensor.id",
                "Azimuth": "measurement_azimuth",
                "Dipole_Length": "dipole_length",
            }
        )
        self._chn_fmt = {
            "ChnNum": "<8",
            "ChnID": "<6",
            "InstrumentID": "<12",
            "Azimuth": ">7.1f",
            "Dipole_Length": ">14.1f",
        }

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __str__(self):
        return "\n".join(self.write_metadata())

    def __repr__(self):
        return self.__str__()

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        if value is not None:
            self._fn = Path(value)
        else:
            self._fn = None

    @property
    def file_size(self):
        if self.fn is not None:
            if self.fn.exists():
                return self.fn.stat().st_size

    @property
    def survey_id(self):
        return self._survey_metadata.id

    @survey_id.setter
    def survey_id(self, value):
        self._survey_metadata.id = value

    @property
    def run_id(self):
        return self._run_metadata.id

    @run_id.setter
    def run_id(self, value):
        self._run_metadata.id = value

    @property
    def site_id(self):
        return self._station_metadata.id

    @site_id.setter
    def site_id(self, station):
        self._station_metadata.id = station

    @property
    def latitude(self):
        return self._station_metadata.location.latitude

    @latitude.setter
    def latitude(self, lat):
        self._station_metadata.location.latitude = lat

    @property
    def longitude(self):
        return self._station_metadata.location.longitude

    @longitude.setter
    def longitude(self, lon):
        self._station_metadata.location.longitude = lon

    @property
    def elevation(self):
        """
        get elevation from national map
        """
        # the url for national map elevation query
        nm_url = Request(self.national_map_url.format(self.longitude, self.latitude))

        # call the url and get the response
        try:
            response = urlopen(nm_url)
        except HTTPError:
            self.logger.error("could not connect to get elevation from national map.")
            self.logger.debug(nm_url.format(self.longitude, self.latitude))
            return self.station_metadata.location.elevation
        # read the xml response and convert to a float
        info = json.loads(response.read().decode())
        try:
            nm_elev = round(float(info["value"]), 1)
        except ValueError:
            self.logger.warning(
                "could not read elevation from national map url. Setting to 0"
            )
            nm_elev = 0
        return nm_elev

    @elevation.setter
    def elevation(self, value):
        self.station_metadata.location.elevation = value

    @property
    def start(self):
        return self._station_metadata.time_period.start

    @start.setter
    def start(self, time_string):
        # Always convert to UTC ISO string
        import pandas as pd

        ts = pd.Timestamp(time_string)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self.station_metadata.time_period.start = ts.isoformat()

    @property
    def end(self):
        return self._station_metadata.time_period.end

    @end.setter
    def end(self, time_string):
        # Always convert to UTC ISO string
        import pandas as pd

        ts = pd.Timestamp(time_string)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self._station_metadata.time_period.end = ts.isoformat()

    @property
    def n_channels(self):
        return self._chn_num

    @n_channels.setter
    def n_channels(self, n_channel):
        try:
            self._chn_num = int(n_channel)
        except ValueError:
            self.logger.warning(f"{n_channel} is not a number, setting n_channels to 0")

    @property
    def sample_rate(self):
        return self._run_metadata.sample_rate

    @sample_rate.setter
    def sample_rate(self, df):
        self._run_metadata.sample_rate = float(df)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        self._n_samples = int(n_samples)

    @property
    def survey_metadata(self):
        return self._survey_metadata

    @survey_metadata.setter
    def survey_metadata(self, value):
        if isinstance(value, Survey):
            self._survey_metadata.update(value)
        else:
            raise TypeError("Input must be a mt_metadata.timeseries.Survey instance")

    @property
    def station_metadata(self):
        return self._station_metadata

    @station_metadata.setter
    def station_metadata(self, value):
        if isinstance(value, Station):
            self._station_metadata.update(value)
        else:
            raise TypeError("Input must be a mt_metadata.timeseries.Station instance")

    @property
    def run_metadata(self):
        return self._run_metadata

    @run_metadata.setter
    def run_metadata(self, value):
        if isinstance(value, Run):
            self._run_metadata.update(value)
        else:
            raise TypeError("Input must be a mt_metadata.timeseries.Run instance")

    def get_component_info(self, comp):
        """

        :param comp: DESCRIPTION
        :type comp: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, kdict in self.channel_dict.items():
            if kdict["ChnID"].lower() == comp.lower():
                return kdict
        return None

    def read_metadata(self, fn=None, meta_lines=None):
        """
        Read in a meta from the raw string or file.  Populate all metadata
        as attributes.

        :param fn: full path to USGS ascii file
        :type fn: string

        :param meta_lines: lines of metadata to read
        :type meta_lines: list
        """
        chn_find = False

        if fn is not None:
            self.fn = fn
        if self.fn is not None:
            with self.fn.open("r") as fid:
                meta_lines = [fid.readline() for ii in range(self._metadata_len)]
        for ii, line in enumerate(meta_lines):
            if "DataSet" in line:
                break
            if line.find(":") > 0:
                key, value = line.strip().split(":", 1)
                value = value.strip()

                if len(value) < 1:
                    chn_find = True
                    continue
                attr = self._key_dict[key]
                setattr(self, attr, value)
            elif "coordinate" in line:
                self.coordinate_system = " ".join(line.strip().split()[-2:])
            else:
                if chn_find is True:
                    if "chnnum" in line.lower():
                        ch_keys = dict(
                            [(kk, ii) for ii, kk in enumerate(line.strip().split())]
                        )
                    else:
                        line_list = line.strip().split()
                        if len(line_list) == len(ch_keys.keys()):
                            ch = getattr(
                                self,
                                f"{line_list[ch_keys['ChnID']].lower()}_metadata",
                            )

                            ch.channel_number = line_list[ch_keys["ChnNum"]]
                            ch.measurement_azimuth = line_list[ch_keys["Azimuth"]]
                            if ch.type in ["electric"]:
                                ch.dipole_length = line_list[ch_keys["Dipole_Length"]]
                            else:
                                ch.sensor.id = line_list[ch_keys["InstrumentID"]]
                            self.run_metadata.data_logger.id = line_list[
                                ch_keys["InstrumentID"]
                            ]

                            ch.time_period.start = self.start
                            ch.time_period.end = self.end
                            ch.sample_rate = self.sample_rate
                            self._run_metadata.add_channel(ch)
                        else:
                            self.logger.warning("Not sure what line this is")
        self._run_metadata.time_period.start = self.start
        self._run_metadata.time_period.end = self.end
        self._run_metadata.sample_rate = self.sample_rate

        self._station_metadata.add_run(self.run_metadata)

        return ii + 1

    def write_metadata(self, chn_list=["Ex", "Ey", "Hx", "Hy", "Hz"]):
        """
        Write out metadata in the format of USGS ascii.

        :return: list of metadate lines.

        .. note:: meant to use '\n'.join(lines) to write out in a file.

        """

        lines = []
        for key, attr in self._key_dict.items():
            if key in ["ChnSettings"]:
                lines.append("{0}:".format(key))
                lines.append(" ".join(self._chn_dict.keys()))
                for chn_key in self.channel_order:
                    chn_line = []
                    ch = getattr(self, f"{chn_key}_metadata")
                    for comp_key, comp_attr in self._chn_dict.items():
                        try:
                            value = ch.get_attr_from_name(comp_attr)
                            if value is None:
                                value = self.run_metadata.data_logger.id
                        except AttributeError:
                            value = 0
                        chn_line.append(f"{value:{self._chn_fmt[comp_key]}}")
                    lines.append("".join(chn_line))
            elif key in ["DataSet"]:
                lines.append("{0}:".format(key))
                return lines
            else:
                try:
                    value = getattr(self, attr)
                    lines.append(f"{key}: {value}")
                except AttributeError:
                    lines.append(f"{key}")
        return lines
