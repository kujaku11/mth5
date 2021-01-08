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
                    [("frequency", np.float), ("real", np.float), ("imag", np.float)]
                ),
                "max_size": (500,),
            },
            "gain": {
                "dtype": np.dtype([("frequency", np.float), ("value", np.float)]),
                "max_size": (100,),
            },
            "conversion": {"dtype": np.dtype([("factor", np.float)]), "max_size": (1,)},
            "delay": {"dtype": np.dtype([("delay", np.float)]), "max_size": (10,)},
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
