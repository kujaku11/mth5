# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:29:02 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base, DataQuality, Filtered, Location, TimePeriod, Instrument, Fdsn
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Base Channel
# =============================================================================
class Channel(Base):
    __doc__ = write_lines(ATTR_DICT["channel"])

    def __init__(self, **kwargs):
        self.type = "auxiliary"
        self.units = None
        self.channel_number = None
        self.comments = None
        self._component = None
        self.sample_rate = 0.0
        self.measurement_azimuth = 0.0
        self.measurement_tilt = 0.0
        self.data_quality = DataQuality()
        self.filter = Filtered()
        self.location = Location()
        self.time_period = TimePeriod()
        self.translated_azimuth = None
        self.translated_tilt = None
        self.sensor = Instrument()
        self.fdsn = Fdsn()

        super().__init__(attr_dict=ATTR_DICT["channel"], **kwargs)

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, value):
        if value is not None:
            self._component = value.lower()