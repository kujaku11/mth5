# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:33:41 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Channel, Instrument, Diagnostic
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Magnetic Channel
# =============================================================================
class Magnetic(Channel):
    __doc__ = write_lines(ATTR_DICT["magnetic"])

    def __init__(self, **kwargs):
        self.sensor = Instrument()
        self.h_field_min = Diagnostic()
        self.h_field_max = Diagnostic()

        Channel.__init__(self, **kwargs)
        self.type = "magnetic"

        self._attr_dict = ATTR_DICT["magnetic"]
