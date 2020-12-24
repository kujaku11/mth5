# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:00:37 2020

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
# ==============================================================================
# Rating
# ==============================================================================
class Rating(Base):
    __doc__ = write_lines(ATTR_DICT["rating"])

    def __init__(self, **kwargs):
        self.author = None
        self.method = None
        self.value = 0.0

        super().__init__(attr_dict=ATTR_DICT["rating"], **kwargs)
