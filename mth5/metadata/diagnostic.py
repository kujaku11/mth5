# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:08:43 2020

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
# diagnostic
# =============================================================================
class Diagnostic(Base):
    __doc__ = write_lines(ATTR_DICT["diagnostic"])

    def __init__(self, **kwargs):
        self.units = None
        self.start = None
        self.end = None
        super().__init__(attr_dict=ATTR_DICT["diagnostic"], **kwargs)