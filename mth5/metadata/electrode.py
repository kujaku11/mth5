# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:10:34 2020

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
# Electrode
# =============================================================================
class Electrode(Base):
    __doc__ = write_lines(ATTR_DICT["electrode"])

    def __init__(self, **kwargs):

        self.id = None
        self.manufacturer = None
        self.type = None
        self.model = None
        super().__init__(attr_dict=ATTR_DICT["electrode"], **kwargs)
