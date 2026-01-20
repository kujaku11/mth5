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


from mt_metadata.timeseries.filters import FIRFilter

from mth5.groups.base import BaseGroup


# =============================================================================
# fir Group
# =============================================================================
class FIRGroup(BaseGroup):
    """
    Container for fir type filters

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    @property
    def filter_dict(self):
        """

        Dictionary of available fir filters

        :return: DESCRIPTION
        :rtype: TYPE
        """
        f_dict = {}
        for key in self.hdf5_group.keys():
            fir_group = self.hdf5_group[key]
            f_dict[key] = {
                "type": fir_group.attrs["type"],
                "hdf5_ref": fir_group.ref,
            }

        return f_dict

    def add_filter(self, name, coefficients, fir_metadata):
        """
        create an HDF5 group/dataset from information given.

        :param name: Nane of the filter
        :type name: string
        :param poles: poles of the filter as complex numbers
        :type poles: np.ndarray(dtype=complex)
        :param zeros: zeros of the filter as complex numbers
        :type zeros: np.ndarray(dtype=comples)
        :param fir_metadata: metadata dictionary see
        :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
        :type fir_metadata: dictionary

        """
        # create a group for the filter by the name
        fir_filter_group = self.hdf5_group.create_group(name)

        # create datasets for the poles and zeros
        fir_ds = fir_filter_group.create_dataset(
            "coefficients",
            coefficients.shape,
            **self.dataset_options,
        )

        fir_ds[:] = coefficients

        # fill in the metadata
        fir_filter_group.attrs.update(fir_metadata)

        return fir_filter_group

    def remove_filter(self):
        pass

    def get_filter(self, name):
        """
        Get a filter from the name

        :param name: name of the filter
        :type name: string

        :return: HDF5 group of the fir filter
        """
        return self.hdf5_group[name]

    def from_object(self, fir_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        :param fir_object: MT metadata PoleZeroFilter
        :type fir_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        """

        if not isinstance(fir_object, FIRFilter):
            msg = "Filter must be a FrequencyResponseTableFilter not %s"
            self.logger.error(msg, type(fir_object))
            raise TypeError(msg)

        input_dict = fir_object.to_dict(single=True, required=False)
        input_dict.pop("coefficients")
        for k, v in input_dict.items():
            if v is None:
                input_dict[k] = str(v)

        fir_group = self.add_filter(
            fir_object.name,
            fir_object.coefficients,
            input_dict,
        )
        return fir_group

    def to_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        fir_group = self.get_filter(name)

        fir_obj = FIRFilter(**fir_group.attrs)

        try:
            fir_obj.coefficients = fir_group["coefficients"][:]
        except TypeError:
            self.logger.debug("fir filter %s has no coefficients", name)
            fir_obj.coefficients = []

        return fir_obj
