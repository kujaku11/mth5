# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:07:08 2020

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
# Person
# ==============================================================================
class Person(Base):
    __doc__ = write_lines(ATTR_DICT["person"])

    def __init__(self, **kwargs):

        self.email = None
        self.author = None
        self.organization = None
        self.comments = None

        super().__init__(attr_dict=ATTR_DICT["person"], **kwargs)
        
