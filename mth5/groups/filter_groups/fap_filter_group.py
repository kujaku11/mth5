# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:55:16 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

from mth5.groups.base import BaseGroup

# =============================================================================
# fap Group
# =============================================================================
class FAPGroup(BaseGroup):
    """
    Container for fap type filters

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @property
    def filter_dict(self):
        """

        Dictionary of available fap filters

        :return: DESCRIPTION
        :rtype: TYPE

        """
        f_dict = {}
        for key in self.hdf5_group.keys():
            fap_group = self.hdf5_group[key]
            f_dict[key] = {
                "type": fap_group.attrs["type"],
                "hdf5_ref": fap_group.ref,
            }

        return f_dict

    def add_filter(self, name, frequency, amplitude, phase, fap_metadata):
        """

        create an HDF5 group/dataset from information given.
        
        :param name: name of the filter
        :type name: string
        :param frequency: frequency array in samples per second
        :type frequency: list, np.ndarray
        :param amplitude: amplitude array in units of units out
        :type amplitude: list, np.ndarray
        :param phase: Phase in degrees
        :type phase: list, np.ndarray
        :param fap_metadata: other metadata for the filter see
        :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter` \
         for details on entries
        :type fap_metadata: dictionary
        :return: DESCRIPTION
        :rtype: TYPE

        """

        # create a group for the filter by the name
        fap_filter_group = self.hdf5_group.create_group(name)

        # create datasets for the poles and zeros
        fap_ds = fap_filter_group.create_dataset(
            "fap_table",
            frequency.shape,
            dtype=np.dtype(
                [("frequency", float), ("amplitude", float), ("phase", float)]
            ),
            **self.dataset_options,
        )

        fap_ds[:] = [(f, a, p) for f, a, p in zip(frequency, amplitude, phase)]

        # fill in the metadata
        fap_filter_group.attrs.update(fap_metadata)

        return fap_filter_group

    def remove_filter(self):
        pass

    def get_filter(self, name):
        """
        Get a filter from the name

        :param name: name of the filter
        :type name: string

        :return: HDF5 group of the fap filter
        """
        return self.hdf5_group[name]

    def from_object(self, fap_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        :param fap_object: MT metadata PoleZeroFilter
        :type fap_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        """

        if not isinstance(fap_object, FrequencyResponseTableFilter):
            msg = f"Filter must be a FrequencyResponseTableFilter not {type(fap_object)}"
            self.logger.error(msg)
            raise TypeError(msg)

        input_dict = fap_object.to_dict(single=True, required=False)
        input_dict.pop("frequencies")
        input_dict.pop("amplitudes")
        input_dict.pop("phases")
        for k, v in input_dict.items():
            if v is None:
                input_dict[k] = str(v)

        fap_group = self.add_filter(
            fap_object.name,
            fap_object.frequencies,
            fap_object.amplitudes,
            fap_object.phases,
            input_dict,
        )
        return fap_group

    def to_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        fap_group = self.get_filter(name)

        fap_obj = FrequencyResponseTableFilter(**fap_group.attrs)

        try:
            fap_obj.frequencies = fap_group["fap_table"]["frequency"][:]
        except TypeError:
            self.logger.debug(f"fap filter {name} has no frequency")
            fap_obj.frequencies = []

        try:
            fap_obj.amplitudes = fap_group["fap_table"]["amplitude"][:]
        except TypeError:
            self.logger.debug(f"fap filter {name} has no amplitudes")
            fap_obj.amplitudes = []

        try:
            fap_obj.phases = fap_group["fap_table"]["phase"][:]
        except TypeError:
            self.logger.debug(f"fap filter {name} has no phases")
            fap_obj.phases = []

        return fap_obj
