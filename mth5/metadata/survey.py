# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:24:01 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base, Person, Citation, Location, TimePeriod, Fdsn
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# ==============================================================================
# Site details
# ==============================================================================
class Survey(Base):
    __doc__ = write_lines(ATTR_DICT["survey"])

    def __init__(self, **kwargs):

        self.acquired_by = Person()
        self.fdsn = Fdsn()
        self.citation_dataset = Citation()
        self.citation_journal = Citation()
        self.comments = None
        self.country = None
        self.datum = None
        self.geographic_name = None
        self.name = None
        self.northwest_corner = Location()
        self.project = None
        self.project_lead = Person()
        self.release_license = "CC-0"
        self.southeast_corner = Location()
        self.summary = None
        self.survey_id = None
        self.time_period = TimePeriod()

        super().__init__(attr_dict=ATTR_DICT["survey"], **kwargs)
