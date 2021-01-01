# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:16:53 2020

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

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Orientation
# =============================================================================
class Orientation(Base):
    __doc__ = write_lines(ATTR_DICT["orientation"])

    def __init__(self, **kwargs):
        self.reference_frame = "geographic"
        self.method = None

        super().__init__(attr_dict=ATTR_DICT["orientation"], **kwargs)
