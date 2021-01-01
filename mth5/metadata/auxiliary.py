# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:30:36 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Channel
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# auxiliary channel
# =============================================================================
class Auxiliary(Channel):
    __doc__ = write_lines(ATTR_DICT["channel"])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
