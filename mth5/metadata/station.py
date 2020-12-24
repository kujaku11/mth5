# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:26:05 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import (
    Base,
    Fdsn,
    Orientation,
    Person,
    Provenance,
    Location,
    TimePeriod,
)
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Station Class
# =============================================================================
class Station(Base):
    __doc__ = write_lines(ATTR_DICT["station"])

    def __init__(self, **kwargs):
        self.id = None
        self.fdsn = Fdsn()
        self.geographic_name = None
        self.datum = None
        self.num_channels = None
        self.channels_recorded = []
        self.channel_layout = None
        self.comments = None
        self.data_type = None
        self.orientation = Orientation()
        self.acquired_by = Person()
        self.provenance = Provenance()
        self.location = Location()
        self.time_period = TimePeriod()

        super().__init__(attr_dict=ATTR_DICT["station"], **kwargs)
