# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:18:08 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base, Person
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# ==============================================================================
# Software
# ==============================================================================
class Software(Base):
    __doc__ = write_lines(ATTR_DICT["software"])

    def __init__(self, **kwargs):
        self.name = None
        self.version = None
        self._author = Person()

        super().__init__(attr_dict=ATTR_DICT["software"], **kwargs)

    @property
    def author(self):
        return self._author.author

    @author.setter
    def author(self, value):
        self._author.author = value
