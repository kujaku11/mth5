# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:58:44 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import requests
import json
import sys
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from mth5 import __version__ as mth5_version
from mth5.timeseries import ChannelTS, RunTS
from mth5.mth5 import MTH5

from mt_metadata.utils.mttime import MTime
from mt_metadata.timeseries import Survey, Station, Run, Magnetic

# =============================================================================

"https://geomag.usgs.gov/ws/data/?id=FRN&type=adjusted&elements=H&sampling_period=1&format=json&starttime=2020-06-02T19:00:00Z&endtime=2020-06-02T22:07:46Z"


class GeomagClient:
    """
    Get geomagnetic data from observatories.

    key words

    - **observatory**: Geogmangetic observatory ID
    - **type**: type of data to get 'adjusted'
    - **start**: start date time to request UTC
    - **end**: end date time to request UTC
    - **elements**: components to get
    - **sampling_period**: samples between measurements in seconds
    - **format**: JSON or IAGA2002

    .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data

    """

    def __init__(self, **kwargs):

        self._base_url = r"https://geomag.usgs.gov/ws/data/"
        self._timeout = 120
        self._max_length = 172800

        self._valid_observatories = [
            "BDT",
            "BOU",
            "TST",
            "BRW",
            "BRT",
            "BSL",
            "CMO",
            "CMT",
            "DED",
            "DHT",
            "FRD",
            "FRN",
            "GUA",
            "HON",
            "NEW",
            "SHU",
            "SIT",
            "SJG",
            "TUC",
            "USGS",
            "BLC",
            "BRD",
            "CBB",
            "EUA",
            "FCC",
            "IQA",
            "MEA",
            "OTT",
            "RES",
            "SNK",
            "STJ",
            "VIC",
            "YKC",
            "HAD",
            "HER",
            "KAK",
        ]

        self._valid_elements = [
            "D",
            "DIST",
            "DST",
            "E",
            "E-E",
            "E-N",
            "F",
            "G",
            "H",
            "SQ",
            "SV",
            "UK1",
            "UK2",
            "UK3",
            "UK4",
            "X",
            "Y",
            "Z",
        ]

        self._ch_map = {"x": "hx", "y": "hy", "z": "hz"}

        self._valid_sampling_periods = [1, 60, 3600]
        self._valid_output_formats = ["json", "iaga2002"]

        self.type = "adjusted"
        self.sampling_period = 1
        self.elements = ["x", "y"]
        self.format = "json"
        self._timeout = 120
        self.observatory = "FRN"
        self._max_length = 172800
        self.start = None
        self.end = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def user_agent(self):
        """
        User agent for the URL request

        :return: DESCRIPTION
        :rtype: TYPE

        """
        encoding = sys.getdefaultencoding() or "UTF-8"
        platform_ = (
            platform.platform().encode(encoding).decode("ascii", "ignore")
        )

        return f"MTH5 v{mth5_version} ({platform_}, Python {platform.python_version()})"

    @property
    def observatory(self):
        return self._id

    @observatory.setter
    def observatory(self, value):
        """
        make sure value is in accepted list of observatories

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if not isinstance(value, str):
            raise TypeError("input must be a string")

        value = value.upper()
        if value not in self._valid_observatories:
            raise ValueError(
                f"{value} not in accepted observatories see "
                "https://www.usgs.gov/tools/web-service-geomagnetism-data "
                "for more information."
            )

        self._id = value

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value):
        """
        make sure elements are in accepted elements
        """

        if isinstance(value, str):
            if value.count(",") > 0:
                value = [item.strip() for item in value.split(",")]

        if not isinstance(value, list):
            value = [value]

        elements = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"{item} in element list must be a string")
            item = item.upper()
            if item not in self._valid_elements:
                raise ValueError(
                    f"{item} is not an accepted element see "
                    "https://www.usgs.gov/tools/web-service-geomagnetism-data "
                    "for more information."
                )
            elements.append(item)

        self._elements = elements

    @property
    def sampling_period(self):
        return self._sampling_period

    @sampling_period.setter
    def sampling_period(self, value):
        """
        validate sample period value

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    f"{value} must be able to convert to an integer."
                )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{value} must be an integer or float not type({type(value)}"
            )

        if value not in self._valid_sampling_periods:
            raise ValueError(f"{value} must be in [1, 60, 3600]")

        self._sampling_period = value

    @property
    def start(self):
        return f"{self._start.iso_no_tz}Z"

    @start.setter
    def start(self, value):
        if value is None:
            self._start = None
        else:
            self._start = MTime(value)

    @property
    def end(self):
        return f"{self._end.iso_no_tz}Z"

    @end.setter
    def end(self, value):
        if value is None:
            self._end = None
        else:
            self._end = MTime(value)

    def get_chunks(self):
        """
        Get the number of chunks of allowable sized to request

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.start is not None and self.end is not None:
            dt = np.arange(
                np.datetime64(self._start.iso_no_tz),
                np.datetime64(self._end.iso_no_tz),
                np.timedelta64(self._max_length * self.sampling_period, "s"),
            )
            dt = np.append(dt, np.array([np.datetime64(self._end.iso_no_tz)]))

            dt_request = [
                (
                    f"{MTime(dt[ii]).iso_no_tz}Z",
                    f"{MTime(dt[ii + 1]).iso_no_tz}Z",
                )
                for ii in range(0, len(dt), 2)
            ]

            return dt_request

    def _get_request_params(self, start, end):
        """
        Get request parameters

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return {
            "id": self.observatory,
            "type": self.type,
            "elements": ",".join(self.elements),
            "sampling_period": self.sampling_period,
            "format": "json",
            "starttime": start,
            "endtime": end,
        }

    def _get_request_dictionary(self, start, end):
        """
        get the request dictionary

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return {
            "url": self._base_url,
            "headers": {"User-Agent": self.user_agent},
            "params": self._get_request_params(start, end),
            "timeout": self._timeout,
        }

    def _request_data(self, request_dictionary):
        """
        request data from geomag for start and end times using
        `request.get(**request_dictionary)

        :param request_dictionary: DESCRIPTION
        :type request_dictionary: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return requests.get(**request_dictionary)

    def _to_station_metadata(self, request_metadata):
        """

        :param request_metadata: DESCRIPTION
        :type request_metadata: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        sm = Station()
        sm.id = request_metadata["intermagnet"]["imo"]["name"]
        sm.fdsn.id = request_metadata["intermagnet"]["imo"]["iaga_code"]

        coords = request_metadata["intermagnet"]["imo"]["coordinates"]

        if coords[0] > 180:
            sm.location.longitude = coords[0] - 360
        else:
            sm.location.longitude = coords[0]
        sm.location.latitude = coords[1]
        sm.location.elevation = coords[2]
        sm.provenance.creation_time = request_metadata["generated"]

        return sm

    def get_data(self, run_id="001"):
        """
        Get data from geomag client at USGS based on the request.  This might
        have to be done in chunks depending on the request size.  The returned
        output is a json object, which we should turn into a ChannelTS object

        For now read into a pandas dataframe and then into a ChannelTS

        In the future, if the request is large, think about writing
        directly to an MTH5 for better efficiency.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch = dict([(c.lower(), []) for c in self.elements])

        for interval in self.get_chunks():
            request_obj = self._request_data(
                self._get_request_dictionary(interval[0], interval[1])
            )
            if request_obj.status_code == 200:
                request_json = json.loads(request_obj.content)
                for element in request_json["values"]:
                    ch[element["metadata"]["element"].lower()].append(
                        pd.DataFrame(
                            {
                                "data": element["values"],
                            },
                            index=request_json["times"],
                        )
                    )
            else:
                raise IOError(
                    "Could not connect to server. Error code: "
                    f"{request_obj.status_code}"
                )

        survey_metadata = Survey(id="USGS-GEOMAG")
        station_metadata = self._to_station_metadata(request_json["metadata"])
        run_metadata = Run(id=run_id)

        ch_list = []
        for key, df_list in ch.items():
            df = pd.concat(df_list).astype(float)
            ch_metadata = Magnetic()
            ch_metadata.component = self._ch_map[key]
            ch_metadata.sample_rate = 1.0 / self.sampling_period
            ch_metadata.units = "nanotesla"
            if "y" in ch_metadata.component:
                ch_metadata.measurement_azimuth = 90
            ch_metadata.location.latitude = station_metadata.location.latitude
            ch_metadata.location.longitude = (
                station_metadata.location.longitude
            )
            ch_metadata.location.elevation = (
                station_metadata.location.elevation
            )
            ch_metadata.time_period.start = df.index[0]
            ch_metadata.time_period.end = df.index[-1]
            run_metadata.time_period.start = df.index[0]
            run_metadata.time_period.end = df.index[-1]
            station_metadata.time_period.start = df.index[0]
            station_metadata.time_period.end = df.index[-1]
            survey_metadata.time_period.start = df.index[0]
            survey_metadata.time_period.end = df.index[-1]
            ch_list.append(
                ChannelTS(
                    channel_type="magnetic",
                    data=df,
                    channel_metadata=ch_metadata,
                    run_metadata=run_metadata,
                    station_metadata=station_metadata,
                    survey_metadata=survey_metadata,
                )
            )

        return RunTS(
            ch_list,
            run_metadata=run_metadata,
            station_metadata=station_metadata,
            survey_metadata=survey_metadata,
        )


class USGSGeomag:
    def __init__(self, **kwargs):
        self.save_path = Path()
        self.filename = None
        self.mth5_version = "0.2.0"
        self.request_columns = [
            "observatory",
            "type",
            "elements",
            "sampling_period",
            "start",
            "end",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def make_request_df(
        self, observatories, elements, sampling_periods, starts, ends
    ):
        """

        create a request dataframe from information given. This is for
        relatively simple requests, for more complicated ones, think about
        making your own data frame in the form of

        row -> observatory, elements, sampling_period, start, end

        :param observatories: DESCRIPTION
        :type observatories: TYPE
        :param elements: DESCRIPTION
        :type elements: TYPE
        :param sampling_periods: DESCRIPTION
        :type sampling_periods: TYPE
        :param starts: DESCRIPTION
        :type starts: TYPE
        :param ends: DESCRIPTION
        :type ends: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(observatories, str):
            observatories = [observatories]

        if isinstance(sampling_periods, (float, str, int)):
            sampling_periods = [float(sampling_periods)]
        elif isinstance(sampling_periods, (list, tuple)):
            sampling_periods = [float(item) for item in sampling_periods]

        if isinstance(starts, str):
            starts = [starts]

        if isinstance(ends, str):
            ends = [ends]

        if len(starts) != len(ends):
            raise ValueError(
                "starts and stops must have the same number of entries"
            )

        request_list = []

        for observatory in observatories:
            for sampling_period in sampling_periods:
                for start, end in zip(starts, ends):
                    request_list.append(
                        {
                            "observatory": observatory,
                            "elements": elements,
                            "sampling_period": sampling_period,
                            "start": start,
                            "end": end,
                        }
                    )

        return pd.DataFrame(request_list)

    def validate_request_df(self, request_df):
        """

        :param request_df: DESCRIPTION
        :type request_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(request_df, pd.DataFrame):
            raise TypeError(
                f"Request input must be a pandas.DataFrame, not {type(request_df)}."
            )

        if "run" in request_df.columns:
            if sorted(request_df.columns.tolist()) != sorted(
                self.request_columns + ["run"]
            ):
                raise ValueError(
                    f"Request must have columns {', '.join(self.request_columns)}"
                )
        else:
            if sorted(request_df.columns.tolist()) != sorted(
                self.request_columns
            ):
                raise ValueError(
                    f"Request must have columns {', '.join(self.request_columns)}"
                )

        request_df = self.add_run_id(request_df)

        return request_df

    def add_run_id(self, request_df):
        """
        Add run id to request df

        :param request_df: DESCRIPTION
        :type request_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        request_df.start = pd.to_datetime(request_df.start)
        request_df.end = pd.to_datetime(request_df.end)
        request_df["run"] = ""

        for obs in request_df.observatory.unique():
            for sr in request_df.loc[
                request_df.observatory == obs, "sampling_period"
            ].unique():
                sr_df = request_df.loc[
                    (request_df.observatory == obs)
                    & (request_df.sampling_period == sr)
                ].sort_values("start")
                request_df.loc[
                    (request_df.observatory == obs)
                    & (request_df.sampling_period == sr),
                    "run",
                ] = [f"{ii+1:03}" for ii in range(len(sr_df))]

        return request_df

    def _make_filename(self, save_path, request_df):
        """

        :param request_df: DESCRIPTION
        :type request_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        elements = "".join(request_df.elements.explode().unique().tolist())
        obs = "_".join(sorted(request_df.observatory.unique().tolist()))

        save_path = Path(save_path)
        if save_path.is_dir():
            fn = f"usgs_geomag_{obs}_{elements}.h5"
            save_path = save_path.joinpath(fn)

        return save_path

    def make_mth5_from_geomag(
        self, request_df, save_path, interact=False, **kwargs
    ):
        """
        write a mth5 to the path given

        todo: make observatory be a list

        :param save_path: DESCRIPTION
        :type save_path: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        request_df = self.validate_request_df(request_df)

        save_path = self._make_filename(save_path, request_df)

        m = MTH5(file_version=self.mth5_version)
        m.open_mth5(save_path)

        if self.mth5_version in ["0.1.0"]:
            survey_group = m.survey_group
        elif self.mth5_version in ["0.2.0"]:
            survey_group = m.add_survey("USGS-GEOMAG")
        else:
            raise ValueError(
                f"MTH5 version must be [ '0.1.0' | '0.2.0' ] not {self.mth5_version}"
            )

        for row in request_df.itertuples():
            geomag_client = GeomagClient(
                observatory=row.observatory,
                type=row.type,
                elements=row.elements,
                start=row.start,
                end=row.end,
                sampling_period=row.sampling_period,
            )

            run = geomag_client.get_data(run_id=row.run)
            station_group = survey_group.stations_group.add_station(
                run.station_metadata.id, station_metadata=run.station_metadata
            )
            run_group = station_group.add_run(
                run.run_metadata.id, run_metadata=run.run_metadata
            )
            run_group.from_runts(run)

        if interact:
            return m
        else:
            m.close_mth5()
            return m.filename
