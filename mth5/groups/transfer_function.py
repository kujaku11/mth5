# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:22:33 2022

@author: jpeacock
"""

# =============================================================================
# 
# =============================================================================

import numpy as np
import xarray as xr

from mth5.groups import BaseGroup, EstimateDataset
from mth5.helpers import to_numpy_type, inherit_doc_string, validate_name
from mth5.utils.exceptions import MTH5Error

from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.tf import StatisticalEstimate
# =============================================================================

class TransferFunctionsGroup(BaseGroup):
    """
    Object to hold transfer functions
    """
    
    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)
        
    def add_transfer_function(self, tf_object):
        """
        Add a transfer function to the group
        
        :param tf_object: DESCRIPTION
        :type tf_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        pass
    
    def get_transfer_function(self, tf_id):
        """
        Get transfer function from id
        
        :param tf_id: DESCRIPTION
        :type tf_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        pass
    
    def remove_transfer_function(self, tf_id):
        """
        Remove a transfer function from the group
        
        :param tf_id: DESCRIPTION
        :type tf_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        pass
    
class TransferFunction(BaseGroup):
    """
    Object to hold a single transfer function estimation
    """
    
    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)
        

    @property
    def period(self):
        """
        Get period from hdf5_group["period"]
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        try:
            return self.hdf5_group["period"][()]
        except KeyError:
            return None
        
    @period.setter
    def period(self, period):
        if period is not None:
            period = np.array(period, dtype=float)
            
            try:
                self.period = self.hdf5_group.create_dataset(
                    "period",
                    data=period,
                    dtype=float,
                    chunks=True,
                    maxshape=(None,),
                    **self.dataset_options,) 
                
            except (OSError, RuntimeError, ValueError) as error:
                self.logger.exception(error)
                self.logger.warning("period already exists, overwriting")
                self.hdf5_group["period"][...] = period
                

    def add_statistical_estimate(self,
                                 estimate_metadata,
                                 estimate_data,
                                 max_shape=(None, None, None),
                                 chunks=True,
                                 **kwargs,):
        """
        Add a StatisticalEstimate
        
        :param estimate: DESCRIPTION
        :type estimate: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
    
        estimate_metadata.name = validate_name(estimate_metadata.name)
        
        if estimate_data is not None: 
            estimate_metadata.data_type = estimate_data.dtype.name

        try:
            dataset = self.hdf5_group.create_dataset(
                estimate_metadata.name,
                data=estimate_data,
                dtype=estimate_data.dtype,
                chunks=chunks,
                maxshape=max_shape,
                **self.dataset_options,
            )
            
            estimate_dataset = EstimateDataset(dataset, 
                                               dataset_metadata=estimate_metadata)
            

        except (OSError, RuntimeError, ValueError) as error:
            self.logger.exception(error)
            msg = f"estimate {estimate_metadata.name} already exists, returning existing group."
            self.logger.debug(msg) 
            
            estimate_dataset = self.get_estimate(estimate_metadata.name)
            
        return estimate_dataset
    
    def get_estimate(self, estimate_name):
        """
        Get a statistical estimate dataset
        """
        estimate_name = validate_name(estimate_name)
        
        try:
            estimate_dataset = self.hdf5_group[estimate_name]
            estimate_metadata = StatisticalEstimate()
            estimate_metadata.from_dict(estimate_dataset.attrs)
            
        except KeyError:
            msg = (
                f"{estimate_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)   
            
    def remove_estimate(self, estimate_name):
        """
        remove a statistical estimate
        
        :param estimate_name: DESCRIPTION
        :type estimate_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        estimate_name = validate_name(estimate_name.lower())

        try:
            del self.hdf5_group[estimate_name]
            self.logger.info(
                "Deleting a estimate does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{estimate_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.exception(msg)
            raise MTH5Error(msg)
        
    