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

from mt_metadata.timeseries.filters import PoleZeroFilter

from mth5.groups.base import BaseGroup

# =============================================================================
# ZPK Group
# =============================================================================
class ZPKGroup(BaseGroup):
    """
    Container for ZPK type filters

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @property
    def filter_dict(self):
        """

        Dictionary of available ZPK filters

        :return: DESCRIPTION
        :rtype: TYPE
        """
        f_dict = {}
        for key in self.hdf5_group.keys():
            zpk_group = self.hdf5_group[key]
            f_dict[key] = {"type": zpk_group.attrs["type"],
                           "hdf5_ref": zpk_group.ref}

        return f_dict

    def add_filter(self, name, poles, zeros, zpk_metadata):
        """
        create an HDF5 group/dataset from information given.  

        :param name: Nane of the filter
        :type name: string
        :param poles: poles of the filter as complex numbers
        :type poles: np.ndarray(dtype=complex)
        :param zeros: zeros of the filter as complex numbers
        :type zeros: np.ndarray(dtype=comples)
        :param zpk_metadata: metadata dictionary see 
        :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
        :type zpk_metadata: dictionary

        """
        # create a group for the filter by the name
        zpk_filter_group = self.hdf5_group.create_group(name)

        # create datasets for the poles and zeros
        poles_ds = zpk_filter_group.create_dataset(
            "poles",
            poles.shape,
            dtype=np.dtype([("real", np.float), ("imag", np.float)]),
            **self.dataset_options,
        )

        # when filling data need to fill the full row for what ever reason.
        poles_ds[:] = [(pr, pi) for pr, pi in zip(poles.real, poles.imag)]
        
        zeros_ds = zpk_filter_group.create_dataset(
            "zeros",
            zeros.shape,
            dtype=np.dtype([("real", np.float), ("imag", np.float)]),
            **self.dataset_options,
        )
        zeros_ds[:] = [(pr, pi) for pr, pi in zip(zeros.real, zeros.imag)]

        # fill in the metadata
        zpk_filter_group.attrs.update(zpk_metadata)
        
        return zpk_filter_group

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

    def from_object(self, zpk_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        :param zpk_object: MT metadata PoleZeroFilter
        :type zpk_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        """

        if not isinstance(zpk_object, PoleZeroFilter):
            msg = f"Filter must be a PoleZeroFilter not {type(zpk_object)}"
            self.logger.error(msg)
            raise TypeError(msg)

        zpk_group = self.add_filter(zpk_object.name,
                        zpk_object.poles,
                        zpk_object.zeros,
                        {"name": zpk_object.name,
                         "gain": zpk_object.gain,
                         "normalization_factor": zpk_object.normalization_factor,
                         "type": zpk_object.type,
                         "units_in": zpk_object.units_in,
                         "units_out": zpk_object.units_out})
        return zpk_group

    def to_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object
        :return: DESCRIPTION
        :rtype: TYPE

        """

        zpk_group = self.get_filter(name)

        zpk_obj = PoleZeroFilter()
        zpk_obj.name = zpk_group.attrs["name"]
        zpk_obj.gain = zpk_group.attrs["gain"]
        zpk_obj.normalization_factor = zpk_group.attrs["normalization_factor"]
        zpk_obj.units_in = zpk_group.attrs["units_in"]
        zpk_obj.units_out = zpk_group.attrs["units_out"]
        try:
            zpk_obj.poles = zpk_group["poles"]["real"][:] + zpk_group["poles"]["imag"][:] * 1j
        except TypeError:
            self.logger.debug(f"ZPK filter {name} has no poles")
            zpk_obj.poles = []
        
        try:
            zpk_obj.zeros = zpk_group["zeros"]["real"][:] + zpk_group["zeros"]["imag"][:] * 1j
        except TypeError:
            self.logger.debug(f"ZPK filter {name} has no zeros")
            zpk_obj.zeros = []
            
        return zpk_obj