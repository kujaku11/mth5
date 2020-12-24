# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:27:25 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.metadata import Base, Person, Provenance, TimePeriod, DataLogger, Fdsn
from mth5.metadata.helpers import write_lines
from mth5.metadata.standards.schema import Standards

ATTR_DICT = Standards().ATTR_DICT
# =============================================================================
# Run
# =============================================================================
class Run(Base):
    __doc__ = write_lines(ATTR_DICT["run"])

    def __init__(self, **kwargs):
        self.id = None
        self.sample_rate = None
        self.channels_recorded_auxiliary = None
        self.channels_recorded_electric = None
        self.channels_recorded_magnetic = None
        self.comments = None
        self._n_chan = None
        self.data_type = None
        self.acquired_by = Person()
        self.provenance = Provenance()
        self.time_period = TimePeriod()
        self.data_logger = DataLogger()
        self.metadata_by = Person()
        self.fdsn = Fdsn()
        super().__init__(attr_dict=ATTR_DICT["run"], **kwargs)

    @property
    def n_channels(self):
        number = 0
        for channel in ["auxiliary", "electric", "magnetic"]:
            channel_list = getattr(self, "channels_recorded_{0}".format(channel))
            if channel_list is not None:
                number += len(channel_list)
        return number

    @property
    def channels_recorded_all(self):
        """
        
        :return: a list of all channels recorded
        :rtype: TYPE

        """

        all_channels = []
        for recorded in ["electric", "magnetic", "auxiliary"]:
            rec_list = getattr(self, f"channels_recorded_{recorded}")
            if rec_list is None:
                continue
            else:
                all_channels += rec_list

        return all_channels
