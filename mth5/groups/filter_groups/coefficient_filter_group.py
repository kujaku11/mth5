# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:58:15 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.timeseries.filters import CoefficientFilter

from mth5.groups.base import BaseGroup

# =============================================================================
#  COEFFCIENT Group
# =============================================================================


class CoefficientGroup(BaseGroup):
    """
    Container for Coefficient type filters
    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @property
    def filter_dict(self):
        """

        Dictionary of available coefficient filters

        :return: DESCRIPTION
        :rtype: TYPE
        """
        f_dict = {}
        for key in self.hdf5_group.keys():
            coefficient_group = self.hdf5_group[key]
            f_dict[key] = {
                "type": coefficient_group.attrs["type"],
                "hdf5_ref": coefficient_group.ref,
            }

        return f_dict

    def add_filter(self, name, coefficient_metadata):
        """
        Add a coefficient Filter
        
        :param name: DESCRIPTION
        :type name: TYPE
        :param coefficient_metadata: DESCRIPTION
        :type coefficient_metadata: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        # create a group for the filter by the name
        coefficient_filter_group = self.hdf5_group.create_group(name)

        # fill in the metadata
        coefficient_filter_group.attrs.update(coefficient_metadata)

        return coefficient_filter_group

    def remove_filter(self):
        pass

    def get_filter(self, name):
        """
        Get a filter from the name

        :param name: name of the filter
        :type name: string

        :return: HDF5 group of the ZPK filter
        """
        return self.hdf5_group[name]

    def from_object(self, coefficient_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.CoefficientFilter`

        :param zpk_object: MT metadata Coefficient Filter
        :type zpk_object: :class:`mt_metadata.timeseries.filters.CoefficientFilter`

        """

        if not isinstance(coefficient_object, CoefficientFilter):
            msg = f"Filter must be a CoefficientFilter not {type(coefficient_object)}"
            self.logger.error(msg)
            raise TypeError(msg)

        coefficient_group = self.add_filter(
            coefficient_object.name, coefficient_object.to_dict(single=True))
        return coefficient_group

    def to_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.CoefficientFilter` object
        
        :return: DESCRIPTION
        :rtype: TYPE

        """

        coefficient_group = self.get_filter(name)

        coefficient_obj = CoefficientFilter(**coefficient_group.attrs)

        return coefficient_obj
