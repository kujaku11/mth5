# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:08:40 2020

Need to make a group for FAP and FIR filters.


:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================

from mth5.groups.base import BaseGroup
from mth5.groups.filter_groups import (
    ZPKGroup,
    CoefficientGroup,
    TimeDelayGroup,
    FAPGroup,
    FIRGroup,
)

# =============================================================================
# Filters Group
# =============================================================================


class FiltersGroup(BaseGroup):
    """
    Not implemented yet
    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

        try:
            self.zpk_group = ZPKGroup(self.hdf5_group.create_group("zpk"))
        except ValueError:
            self.zpk_group = ZPKGroup(self.hdf5_group["zpk"])

        try:
            self.coefficient_group = CoefficientGroup(
                self.hdf5_group.create_group("coefficient")
            )
        except ValueError:
            self.coefficient_group = CoefficientGroup(self.hdf5_group["coefficient"])

        try:
            self.time_delay_group = TimeDelayGroup(
                self.hdf5_group.create_group("time_delay")
            )
        except ValueError:
            self.time_delay_group = TimeDelayGroup(self.hdf5_group["time_delay"])

        try:
            self.fap_group = FAPGroup(self.hdf5_group.create_group("fap"))
        except ValueError:
            self.fap_group = FAPGroup(self.hdf5_group["fap"])

        try:
            self.fir_group = FIRGroup(self.hdf5_group.create_group("fir"))
        except ValueError:
            self.fir_group = FIRGroup(self.hdf5_group["fir"])

    @property
    def filter_dict(self):
        filter_dict = {}
        filter_dict.update(self.zpk_group.filter_dict)
        filter_dict.update(self.coefficient_group.filter_dict)
        filter_dict.update(self.time_delay_group.filter_dict)
        filter_dict.update(self.fap_group.filter_dict)
        filter_dict.update(self.fir_group.filter_dict)

        return filter_dict

    def add_filter(self, filter_object):
        """
        Add a filter dataset based on type

        current types are:
            * zpk         -->  zeros, poles, gain
            * fap         -->  frequency look up table
            * time_delay  -->  time delay filter
            * coefficient -->  coefficient filter

        :param filter_object: An MT metadata filter object
        :type filter_object: :class:`mt_metadata.timeseries.filters`

        """
        filter_object.name = filter_object.name.replace("/", " per ")

        if filter_object.type in ["zpk", "poles_zeros"]:
            try:
                return self.zpk_group.from_object(filter_object)
            except ValueError:
                self.logger.debug("group %s already exists", filter_object.name)
                return self.zpk_group.get_filter(filter_object.name)

        elif filter_object.type in ["coefficient"]:
            try:
                return self.coefficient_group.from_object(filter_object)
            except ValueError:
                self.logger.debug("group %s already exists", filter_object.name)
                return self.coefficient_group.get_filter(filter_object.name)

        elif filter_object.type in ["time_delay", "time delay"]:
            try:
                return self.time_delay_group.from_object(filter_object)
            except ValueError:
                self.logger.debug("group %s already exists", filter_object.name)
                return self.time_delay_group.get_filter(filter_object.name)

        elif filter_object.type in ["fap", "frequency response table"]:
            try:
                return self.fap_group.from_object(filter_object)
            except ValueError:
                self.logger.debug("group %s already exists", filter_object.name)
                return self.fap_group.get_filter(filter_object.name)

        elif filter_object.type in ["fir"]:
            try:
                return self.fir_group.from_object(filter_object)
            except ValueError:
                self.logger.debug("group %s already exists", filter_object.name)
                return self.fir_group.get_filter(filter_object.name)

    def get_filter(self, name):
        """
        Get a filter by name
        """

        try:
            hdf5_ref = self.filter_dict[name]["hdf5_ref"]
        except KeyError:
            msg = "Could not find %s in the filter dictionary"
            self.logger.error(msg, name)
            raise KeyError(msg, name)

        return self.hdf5_group[hdf5_ref]

    def to_filter_object(self, name):
        """
        return the MT metadata representation of the filter
        """

        try:
            f_type = self.filter_dict[name]["type"]
        except KeyError:
            msg = "Could not find %s in the filter dictionary"
            self.logger.error(msg, name)
            raise KeyError(msg, name)

        if f_type in ["zpk"]:
            return self.zpk_group.to_object(name)
        elif f_type in ["coefficient"]:
            return self.coefficient_group.to_object(name)
        elif f_type in ["time_delay", "time delay"]:
            return self.time_delay_group.to_object(name)
        elif f_type in ["fap", "frequency response table"]:
            return self.fap_group.to_object(name)
        elif f_type in ["fir"]:
            return self.fir_group.to_object(name)
