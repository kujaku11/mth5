# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:02:28 2020

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
# Citation
# ==============================================================================
class Citation(Base):
    __doc__ = write_lines(ATTR_DICT["citation"])

    def __init__(self, **kwargs):
        self.author = None
        self.title = None
        self.journal = None
        self.volume = None
        self.doi = None
        self.year = None
        super().__init__(attr_dict=ATTR_DICT["citation"], **kwargs)
