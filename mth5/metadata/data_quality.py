# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:01:17 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base, Rating
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# ==============================================================================
# Data Quality
# ==============================================================================
class DataQuality(Base):
    __doc__ = write_lines(ATTR_DICT["data_quality"])

    def __init__(self, **kwargs):

        self.rating = Rating()
        self.warnings = None

        super().__init__(attr_dict=ATTR_DICT["data_quality"], **kwargs)
        