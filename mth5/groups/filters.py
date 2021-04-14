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
        
        pass
    
    
    def add_filter(self, name, poles, zeros, zpk_metadata):
        """
        create an HDF5 group/dataset from information given.  
        
        :param name: DESCRIPTION
        :type name: TYPE
        :param poles: DESCRIPTION
        :type poles: TYPE
        :param zeros: DESCRIPTION
        :type zeros: TYPE
        :param zpk_metadata: DESCRIPTION
        :type zpk_metadata: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

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
        
        pass
    
    def remove_filter(self): 
        pass
    
    def get_filter(self, name):
        # return self.filter_dict[name]
        pass
    
    def from_zpk_object(self, zpk_object):
        """
        make a filter from a zpk object
        
        :param zpk_object: DESCRIPTION
        :type zpk_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass
    
    def to_zpk_object(self):
        """
        make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object
        :return: DESCRIPTION
        :rtype: TYPE

        """
    
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
                    [("frequency", np.float), ("real", np.float), ("imag", np.float),]
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
            "delay": {"dtype": np.dtype([("delay", np.float)]), "max_size": (10,),},
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

        filter_dataset = FilterDataset(filter_table, dataset_metadata=filter_metadata)
        filter_dataset.write_metadata()

        self.logger.debug(f"Created filter {filter_name}")

        return filter_dataset
