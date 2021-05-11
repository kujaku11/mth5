# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:08:40 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mt_metadata.timeseries import Filter
from mt_metadata.timeseries.filters import PoleZeroFilter

from mth5.groups.base import BaseGroup
from mth5.groups.filter_dataset import FilterDataset

# =============================================================================
# Standards Group
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
            f_dict[key] = {"type": zpk_group.attrs.type,
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
            (poles.size,),
            dtype=np.dtype([("real", np.float), ("imag", np.float)]),
            **self.dataset_options,
        )
        poles_ds["real"][:] = poles.real
        poles_ds["imag"][:] = poles.imag

        zeros_ds = zpk_filter_group.create_dataset(
            "zeros",
            (zeros.size),
            dtype=np.dtype([("real", np.float), ("imag", np.float)]),
            **self.dataset_options,
        )
        zeros_ds["real"][:] = zeros.real
        zeros_ds["imag"][:] = zeros.imag

        # fill in the metadata
        zpk_filter_group.attrs.update(zpk_metadata)

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

    def from_zpk_object(self, zpk_object):
        """
        make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        :param zpk_object: MT metadata PoleZeroFilter
        :type zpk_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

        """

        if not isinstance(zpk_object, PoleZeroFilter):
            msg = f"Filter must be a PoleZeroFilter not {type(zpk_object)}"
            self.logger.error(msg)
            raise TypeError(msg)

        self.add_filter(zpk_object.name,
                        zpk_object.poles,
                        zpk_object.zeros,
                        {"name": zpk_object.name,
                         "gain": zpk_object.gain,
                         "normalization_factor": zpk_object.normalization_factor,
                         "type": zpk_object.type,
                         "units_in": zpk_object.units_in,
                         "units_out": zpk_object.units_out})

    def to_zpk_object(self, name):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object
        :return: DESCRIPTION
        :rtype: TYPE

        """

        zpk_group = self.get_filter(name)

        zpk_obj = PoleZeroFilter()
        zpk_obj.name = zpk_group.attrs.name


class FiltersGroup(BaseGroup):
    """
    Not implemented yet
    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)
        self._dtype_dict = {
            "zpk": {
                "dtype": np.dtype(
                    [
                        ("poles_real", np.float),
                        ("poles_imag", np.float),
                        ("zeros_real", np.float),
                        ("zeros_imag", np.float),
                    ]
                ),
                "max_size": (100,),
            },
            "table": {
                "dtype": np.dtype(
                    [("frequency", np.float),
                     ("real", np.float), ("imag", np.float), ]
                ),
                "max_size": (500,),
            },
            "gain": {
                "dtype": np.dtype([("frequency", np.float), ("value", np.float)]),
                "max_size": (100,),
            },
            "conversion": {
                "dtype": np.dtype([("factor", np.float)]),
                "max_size": (1,),
            },
            "delay": {"dtype": np.dtype([("delay", np.float)]), "max_size": (10,), },
        }

    def add_filter(self, filter_name, filter_type, values=None, filter_metadata=None):
        """
        Add a filter dataset based on type

        current types are:
            * zpk --> zeros, poles, gain
            * table --> frequency look up table

        :param filter_name: DESCRIPTION
        :type filter_name: TYPE
        :param filter_type: DESCRIPTION
        :type filter_type: TYPE
        :param values: DESCRIPTION, defaults to None
        :type values: TYPE, optional
        :param metadata: DESCRIPTION, defaults to None
        :type metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if filter_type not in list(self._dtype_dict.keys()):
            msg = f"filter type {filter_type} not understood."
            self.logger.error(msg)
            raise ValueError(msg)

        if filter_metadata is not None:
            if not isinstance(filter_metadata, Filter):
                msg = (
                    "Input metadata must be of type mth5.metadata.Filter, "
                    + f"not {type(filter_metadata)}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            filter_metadata = Filter()
            filter_metadata.name = filter_name
            filter_metadata.type = filter_type

        if values is None:
            filter_table = self.hdf5_group.create_dataset(
                filter_name,
                (0,),
                maxshape=self._dtype_dict[filter_type]["max_size"],
                dtype=self._dtype_dict[filter_type]["dtype"],
                **self.dataset_options,
            )
        else:
            filter_table = self.hdf5_group.create_dataset(
                filter_name,
                data=values,
                dtype=self._dtype_dict[filter_type]["dtype"],
                **self.dataset_options,
            )

        filter_dataset = FilterDataset(
            filter_table, dataset_metadata=filter_metadata)
        filter_dataset.write_metadata()

        self.logger.debug(f"Created filter {filter_name}")

        return filter_dataset
