# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:01:55 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.timeseries.filters import TimeDelayFilter

from mth5.groups.base import BaseGroup

# =============================================================================
# TimeDelay Group
# =============================================================================


class TimeDelayGroup(BaseGroup):
    """
    Container for time_delay type filters

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @property
    def filter_dict(self):
        """

        Dictionary of available time_delay filters

        :return: DESCRIPTION
        :rtype: TYPE
        """
        f_dict = {}
        for key in self.hdf5_group.keys():
            time_delay_group = self.hdf5_group[key]
            f_dict[key] = {
                "type": time_delay_group.attrs["type"],
                "hdf5_ref": time_delay_group.ref,
            }

        return f_dict

    def add_filter(self, name, time_delay_metadata):
        """
        create an HDF5 group/dataset from information given.  

        :param name: Nane of the filter
        :type name: string
        :param poles: poles of the filter as complex numbers
        :type poles: np.ndarray(dtype=complex)
        :param zeros: zeros of the filter as complex numbers
        :type zeros: np.ndarray(dtype=comples)
        :param time_delay_metadata: metadata dictionary see 
        :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
        :type time_delay_metadata: dictionary

        """
        # create a group for the filter by the name
        time_delay_filter_group = self.hdf5_group.create_group(name)

        # fill in the metadata
        time_delay_filter_group.attrs.update(time_delay_metadata)

        return time_delay_filter_group

    def remove_filter(self):
        pass

    def get_filter(self, name):
        """
        Get a filter from the name

        :param name: name of the filter
        :type name: string

        :return: HDF5 group of the time_delay filter
        """
        return self.hdf5_group[name]

    def from_object(self, time_delay_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        :param time_delay_object: MT metadata PoleZeroFilter
        :type time_delay_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        """

        if not isinstance(time_delay_object, TimeDelayFilter):
            msg = f"Filter must be a TimeDelayFilter not {type(time_delay_object)}"
            self.logger.error(msg)
            raise TypeError(msg)

        time_delay_group = self.add_filter(
            time_delay_object.name,
            time_delay_object.poles,
            time_delay_object.zeros,
            {
                "name": time_delay_object.name,
                "delay": time_delay_object.delay,
                "type": time_delay_object.type,
                "units_in": time_delay_object.units_in,
                "units_out": time_delay_object.units_out,
                "comments": time_delay_object.comments,
            },
        )
        return time_delay_group

    def to_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object
        :return: DESCRIPTION
        :rtype: TYPE

        """

        time_delay_group = self.get_filter(name)

        time_delay_obj = TimeDelayFilter()
        time_delay_obj.name = time_delay_group.attrs["name"]
        time_delay_obj.delay = time_delay_group.attrs["delay"]
        time_delay_obj.units_in = time_delay_group.attrs["units_in"]
        time_delay_obj.units_out = time_delay_group.attrs["units_out"]

        return time_delay_obj
