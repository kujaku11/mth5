# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:23:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from types import SimpleNamespace
from loguru import logger
import numpy as np

from mt_metadata.timeseries import Magnetic, Electric
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

# =============================================================================


class MetronixFileNameMetadata:
    def __init__(self, fn=None, **kwargs):
        self.fn = fn

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, value):
        if value is None:
            self._fn = None
        else:
            self._fn = Path(value)
            self._parse_fn(self._fn)

    def _parse_fn(self, fn):
        """
        need to parse metadata from the filename

        :param fn: DESCRIPTION
        :type fn: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is None:
            return

        fn_list = fn.stem.split("_")
        self.system_number = fn_list[0]
        self.system_name = fn_list[1]
        self.channel_number = self._parse_channel_number(fn_list[2])
        self.component = self._parse_component(fn_list[3])
        self.sample_rate = self._parse_sample_rate(fn_list[4])
        self.file_type = self._get_file_type(fn)

    def _parse_channel_number(self, value):
        """
        channel number is C## > 0

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return int(value.replace("C", "0"))

    def _parse_component(self, value):
        """
        component is T{comp}

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return value.replace("T", "").lower()

    def _parse_sample_rate(self, value):
        """
        sample rate is {sr}Hz or {sr}s

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if "hz" in value.lower():
            return float(value.lower().replace("hz", ""))
        elif "s" in value.lower():
            return 1.0 / float(value.lower().replace("s", ""))

    def _get_file_type(self, value):
        if value.suffix in [".json"]:
            return "metadata"
        elif value.suffix in [".atss"]:
            return "timeseries"
        else:
            raise ValueError(f"Metronix file type {value} not supported.")

    @property
    def file_size(self):
        """
        file size in bytes

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.fn is not None:
            return self.fn.stat().st_size
        return 0

    @property
    def n_samples(self):
        """estimated number of samples in file"""
        return self.file_size / 8


class MetronixChannelJSON(MetronixFileNameMetadata):
    def __init__(self, fn=None, **kwargs):
        super().__init__(fn=fn, **kwargs)
        if self.fn is not None:
            self.read(self.fn)

    def _has_metadata(self):
        if self.metadata is None:
            raise ValueError("Metronix JSON file has not been read in yet.")
        return True

    def read(self, fn=None):
        """

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if fn is not None:
            self.fn = fn

        with open(self.fn, "r") as fid:
            self.metadata = json.load(
                fid, object_hook=lambda d: SimpleNamespace(**d)
            )

    def to_mt_metadata(self):
        """
        translate to `mt_metadata.timeseries.Channel` object

        :return: mt_metadata object based on component and FAP filter if one
         exists.
        :rtype: TYPE

        """
        self._has_metadata()

        sensor_response_filter = self.get_sensor_response_filter()

        if self.component.startswith("e"):
            metadata_object = Electric(
                component=self.component,
                channel_number=self.channel_number,
                measurement_azimuth=self.metadata.angle,
                measurement_tilt=self.metadata.tilt,
                sample_rate=self.sample_rate,
                type="electric",
            )
        elif self.component.startswith("h"):
            metadata_object = Magnetic(
                component=self.component,
                channel_number=self.channel_number,
                measurement_azimuth=self.metadata.angle,
                measurement_tilt=self.metadata.tilt,
                sample_rate=self.sample_rate,
                type="magnetic",
            )
        else:
            msg = f"Do not understand channel component {self.component}"
            logger.error(msg)
            raise ValueError(msg)

        metadata_object.time_period.start = self.metadata.datetime
        metadata_object.location.latitude = self.metadata.latitude
        metadata_object.location.longitude = self.metadata.longitude
        metadata_object.location.elevation = self.metadata.elevation
        metadata_object.units = self.metadata.units

        if sensor_response_filter is not None:
            metadata_object.filter.name = self.metadata.filter.split(",") + [
                sensor_response_filter.name
            ]
        else:
            metadata_object.filter.name = self.metadata.filter.split(",")
        metadata_object.filter.applied = [True] * len(
            metadata_object.filter.name
        )

        return metadata_object, sensor_response_filter

    def get_sensor_response_filter(self):
        """
        get the sensor response FAP filter
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self._has_metadata()

        fap = FrequencyResponseTableFilter(
            calibration_date=self.metadata.sensor_calibration.datetime,
            name=self.metadata.sensor_calibration.sensor,
            frequencies=self.metadata.sensor_calibration.f,
            amplitudes=self.metadata.sensor_calibration.a,
            units_out=self.metadata.units,
            units_in=self.metadata.sensor_calibration.units_amplitude.split(
                "/"
            )[-1],
        )

        if self.metadata.sensor_calibration.units_phase in ["degrees", "deg"]:
            fap.phases = np.deg2rad(self.metadata.sensor_calibration.p)
        else:
            fap.phases = self.metadata.sensor_calibration.p

        if len(fap.frequencies) > 0:
            return fap
        return None
