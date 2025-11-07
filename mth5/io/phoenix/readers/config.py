# -*- coding: utf-8 -*-
"""

Created on Fri Jun 10 07:52:03 2022

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from loguru import logger
from mt_metadata.timeseries import Station

from .helpers import read_json_to_object


# =============================================================================


class PhoenixConfig:
    """
    A container for the config.json file used to control the recording

    """

    def __init__(self, fn=None, **kwargs):
        self.fn = fn
        self.obj = None
        self.logger = logger

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
