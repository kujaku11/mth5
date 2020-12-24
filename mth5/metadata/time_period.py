# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:14:57 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards
from mth5.utils.mttime import MTime

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Time Period
# =============================================================================
class TimePeriod(Base):
    __doc__ = write_lines(ATTR_DICT["time_period"])

    def __init__(self, **kwargs):

        self._start_dt = MTime()
        self._end_dt = MTime()
        super().__init__(attr_dict=ATTR_DICT["time_period"], **kwargs)

    @property
    def start(self):
        return self._start_dt.iso_str

    @start.setter
    def start(self, start_date):
        self._start_dt.from_str(start_date)

    @property
    def end(self):
        return self._end_dt.iso_str

    @end.setter
    def end(self, stop_date):
        self._end_dt.from_str(stop_date)

    @property
    def start_date(self):
        return self._start_dt.date

    @start_date.setter
    def start_date(self, start_date):
        self._start_dt.from_str(start_date)

    @property
    def end_date(self):
        return self._end_dt.date

    @end_date.setter
    def end_date(self, stop_date):
        self._end_dt.from_str(stop_date)