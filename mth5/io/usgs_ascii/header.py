# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:54:09 2020

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import urllib as url
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

from mth5.utils.mth5_logger import setup_logger
from mt_metadata.timeseries import Magnetic, Electric, Run, Station, Survey

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
    start              Start time of station YYYY-MM-DDThh:mm:ss UTC
    end               Stop time of station YYYY-MM-DDThh:mm:ss UTC
    sample_rate                Sampling rate samples/second
    n_samples                 Number of samples
    n_channels                     Number of channels
    coordinate_system          [ Geographic North | Geomagnetic North ]
    chn_settings               Channel settings, see below
    missing_data_flag           Missing data value
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

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        self.fn = fn
        self.missing_data_flag = np.NaN
        self.coordinate_system = None
        self._metadata_len = 30
        self._survey_metadata = Survey()
        self._station_metadata = Station()
        self._run_metadata = Run()
        self.ex_metadata = Electric(component="ex")
        self.ey_metadata = Electric(component="ey")
        self.hx_metadata = Magnetic(component="hx")
        self.hy_metadata = Magnetic(component="hy")
        self.hz_metadata = Magnetic(component="hz")

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

        self._chn_settings = [
            "ChnNum",
            "ChnID",
            "InstrumentID",
            "Azimuth",
            "Dipole_Length",
        ]
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
        return self.write_metadata()

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
        nm_url = r"https://nationalmap.gov/epqs/pqs.php?x={0:.5f}&y={1:.5f}&units=Meters&output=xml"

        # call the url and get the response
        try:
            response = url.request.urlopen(
                nm_url.format(self.longitude, self.latitude)
            )
        except url.error.HTTPError:
            self.logger.error(
                "could not connect to get elevation from national map."
            )
            self.logger.debug(nm_url.format(self.longitude, self.latitude))
            return -666

        # read the xml response and convert to a float
        info = ET.ElementTree(ET.fromstring(response.read()))
        info = info.getroot()
        for elev in info.iter("Elevation"):
            nm_elev = float(elev.text)
        return nm_elev

    @property
    def start(self):
        return self._station_metadata.time_period.start

    @start.setter
    def start(self, time_string):
        self.station_metadata.time_period.start = time_string

    @property
    def end(self):
        return self._station_metadata.time_period.end

    @end.setter
    def end(self, time_string):
        self._station_metadata.time_period.end = time_string

    @property
    def n_channels(self):
        return self._chn_num

    @n_channels.setter
    def n_channels(self, n_channel):
        try:
            self._chn_num = int(n_channel)
        except ValueError:
            self.logger.warning(
                f"{n_channel} is not a number, setting n_channels to 0"
            )

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
            raise TypeError(
                "Input must be a mt_metadata.timeseries.Survey instance"
            )

    @property
    def station_metadata(self):
        return self._station_metadata

    @station_metadata.setter
    def station_metadata(self, value):
        if isinstance(value, Station):
            self._station_metadata.update(value)
        else:
            raise TypeError(
                "Input must be a mt_metadata.timeseries.Station instance"
            )

    @property
    def run_metadata(self):
        return self._run_metadata

    @run_metadata.setter
    def run_metadata(self, value):
        if isinstance(value, Run):
            self._run_metadata.update(value)
        else:
            raise TypeError(
                "Input must be a mt_metadata.timeseries.Run instance"
            )

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
                meta_lines = [
                    fid.readline() for ii in range(self._metadata_len)
                ]
        for ii, line in enumerate(meta_lines):
            if "DataSet" in line:
                break

            if line.find(":") > 0:
                key, value = line.strip().split(":", 1)
                value = value.strip()

                if len(value) < 1:
                    chn_find = True
                    continue

                if "elev" in key.lower():
                    pass
                else:
                    attr = self._key_dict[key]
                    setattr(self, attr, value)
            elif "coordinate" in line:
                self.coordinate_system = " ".join(line.strip().split()[-2:])
            else:
                if chn_find is True:
                    if "chnnum" in line.lower():
                        ch_keys = dict(
                            [
                                (kk, ii)
                                for ii, kk in enumerate(line.strip().split())
                            ]
                        )
                    else:
                        line_list = line.strip().split()
                        if len(line_list) == len(ch_keys.keys()):
                            ch = getattr(
                                self,
                                f"{line_list[ch_keys['ChnID']].lower()}_metadata",
                            )

                            ch.channel_number = line_list[ch_keys["ChnNum"]]
                            ch.measurement_azimuth = line_list[
                                ch_keys["Azimuth"]
                            ]
                            if ch.type in ["electric"]:
                                ch.dipole_length = line_list[
                                    ch_keys["Dipole_Length"]
                                ]
                            else:
                                ch.sensor.id = line_list[
                                    ch_keys["InstrumentID"]
                                ]
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
        for key in self._key_list:
            if key in ["chn_settings"]:
                lines.append("{0}:".format(key))
                lines.append(" ".join(self._chn_settings))
                for chn_key in chn_list:
                    chn_line = []
                    try:
                        for comp_key in self._chn_settings:
                            chn_line.append(
                                "{0:{1}}".format(
                                    self.channel_dict[chn_key][comp_key],
                                    self._chn_fmt[comp_key],
                                )
                            )
                        lines.append("".join(chn_line))
                    except KeyError:
                        pass
            elif key in ["data_set"]:
                lines.append("{0}:".format(key))
                return lines
            else:
                if key in ["site_latitude", "site_longitude"]:
                    lines.append(
                        "{0}: {1:.5f}".format(key, getattr(self, key))
                    )
                else:
                    lines.append("{0}: {1}".format(key, getattr(self, key)))

        return lines
