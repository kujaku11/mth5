# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:53:08 2020

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
# ============================================================================
# declination
# ============================================================================
class Declination(Base):
    __doc__ = write_lines(ATTR_DICT["declination"])

    def __init__(self, **kwargs):

        self.value = None
        self.epoch = None
        self.model = None
        self.comments = None
        super(Declination, self).__init__(attr_dict=ATTR_DICT["declination"], **kwargs)
