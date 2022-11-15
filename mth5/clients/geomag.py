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
import numpy as np

from mth5 import __version__ as mth5_version

from mt_metadata.utils.mttime import MTime

# =============================================================================

"https://geomag.usgs.gov/ws/data/?id=FRN&type=adjusted&elements=H&sampling_period=1&format=json&starttime=2020-06-02T19:00:00Z&endtime=2020-06-02T22:07:46Z"


class GeomagClient:
    """
    Get geomagnetic data from observatories.

    key words

    - **observatory**: Geogmangetic observatory ID
    - **type**: type of data to get 'adjusted'
    - **start_time**: start date time to request UTC
    - **end_time**: end date time to request UTC
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

        self._valid_sample_periods = [1, 60, 3600]
        self._valid_output_formats = ["json", "iaga2002"]

        self.type = "adjusted"
        self.sampling_period = 1
        self.elements = ["x", "y"]
        self.format = "json"
        self._timeout = 120
        self.observatory = "FRN"
        self._max_length = 172800

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

        return f"MTH5 v{mth5_version} ({platform_}, Python {platform.python_version})"

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
    def sample_period(self):
        return self._sample_period

    @sample_period.setter
    def sample_period(self, value):
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

        if value not in self._valid_sample_periods:
            raise ValueError(f"{value} must be in [1, 60, 3600]")

        self._sample_period = value

    @property
    def params(self):
        """parameters for request"""
        return {
            "id": self.observatory,
            "type": self.type,
            "elements": "X,Y",
            "sampling_period": "1",
            "format": "json",
            "starttime": "2020-06-02T19:00:00Z",
            "endtime": "2020-06-02T22:07:46Z",
        }
