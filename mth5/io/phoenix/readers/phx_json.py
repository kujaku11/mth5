# -*- coding: utf-8 -*-
"""

Created on Fri Jun 10 07:52:03 2022

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from types import SimpleNamespace

# Parse JSON into an object with attributes corresponding to dict keys.
import numpy as np

from mt_metadata.timeseries import Station, Run, Electric, Magnetic

# =============================================================================


def read_json_to_object(fn):
    """
    read a json file directly into an object

    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    with open(fn, "r") as fid:
        obj = json.load(fid, object_hook=lambda d: SimpleNamespace(**d))
    return obj


class ConfigJSON:
    """
    A container for the config.json file used to control the recording

    """

    def __init__(self, fn=None, **kwargs):

        self.fn = fn
        self.obj = None

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, fn):
        if fn is None:
            self._fn = None
        else:
            fn = Path(fn)
            if fn.exists():
                self._fn = Path(fn)
            else:
                raise ValueError(f"Could not find {fn}")

    def read(self, fn=None):
        """
        read a config.json file that is in the Phoenix format

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if fn is not None:
            self.fn = fn
        self.obj = read_json_to_object(self.fn)

    def has_obj(self):
        if self.obj is not None:
            return True
        return False

    @property
    def auto_power_enabled(self):
        if self.has_obj():
            return self.obj.auto_power_enabled

    @property
    def config(self):
        if self.has_obj():
            return self.obj.config[0]

    @property
    def empower_version(self):
        if self.has_obj():
            return self.obj.empower_version

    @property
    def mtc150_reset(self):
        if self.has_obj():
            return self.obj.mtc150_reset

    @property
    def network(self):
        if self.has_obj():
            return self.obj.network

    @property
    def receiver(self):
        if self.has_obj():
            return self.obj.receiver

    @property
    def schedule(self):
        if self.has_obj():
            return self.obj.schedule

    @property
    def surveyTechnique(self):
        if self.has_obj():
            return self.obj.surveyTechnique

    @property
    def timezone(self):
        if self.has_obj():
            return self.obj.timezone

    @property
    def timezone_offset(self):
        if self.has_obj():
            return self.obj.timezone_offset

    @property
    def version(self):
        if self.has_obj():
            return self.obj.version

    def station_metadata(self):
        s = Station()

        s.id = self.config.layout.Station_Name
        s.acquired_by.name = self.config.layout.Operator
        s.acquired_by.organization = self.config.layout.Company_Name
        s.comments = self.config.layout.Notes

        return s


class ReceiverMetadataJSON:
    """
    A container for the recmeta.json file used to control the recording

    """

    def __init__(self, fn=None, **kwargs):

        self.fn = fn
        self.obj = None

        self._e_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "pot_p": "contact_resistance.start",
            "pot_n": "contact_resistance.end",
        }

        self._h_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "type_name": "sensor.model",
            "type": "sensor.type",
            "serial": "sensor.id",
        }

        if self.fn is not None:
            self.read()

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, fn):
        if fn is None:
            self._fn = None
        else:
            fn = Path(fn)
            if fn.exists():
                self._fn = Path(fn)
            else:
                raise ValueError(f"Could not find {fn}")

    def read(self, fn=None):
        """
        read a config.json file that is in the Phoenix format

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if fn is not None:
            self.fn = fn
        self.obj = read_json_to_object(self.fn)

    def has_obj(self):
        if self.obj is not None:
            return True
        return False

    def _get_ch_index(self, tag):
        if self.has_obj():
            for item in self.obj.channel_map.mapping:
                if item.tag.lower() == tag.lower():
                    return item.idx

    def _to_electric_metadata(self, tag):
        c = Electric()

        if self.has_obj():
            ch = self.obj.chconfig.chans[self._get_ch_index(tag)]

            for p_key, m_value in self._e_map.items():
                if p_key == "ty":
                    m_value = "electric"
                c.set_attr_from_name(m_value, getattr(ch, p_key))
            c.channel_number = self._get_ch_index(tag)
            c.dipole_length = ch.length1 + ch.length2
            c.units = "millivolts"
        return c

    def _to_magnetic_metadata(self, tag):
        c = Magnetic()

        if self.has_obj():
            ch = self.obj.chconfig.chans[self._get_ch_index(tag)]

            for p_key, m_value in self._h_map.items():
                if p_key == "ty":
                    m_value = "magnetic"
                c.set_attr_from_name(m_value, getattr(ch, p_key))
            c.channel_number = self._get_ch_index(tag)
            c.sensor.manufacturer = "Phoenix Geophysics"
            c.units = "millivolts"
        return c

    @property
    def e1_metadata(self):
        return self._to_electric_metadata("e1")

    @property
    def e2_metadata(self):
        return self._to_electric_metadata("e2")

    @property
    def h1_metadata(self):
        return self._to_magnetic_metadata("h1")

    @property
    def h2_metadata(self):
        return self._to_magnetic_metadata("h2")

    @property
    def h3_metadata(self):
        return self._to_magnetic_metadata("h3")