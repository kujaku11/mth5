# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:22:33 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr

from mth5.groups import BaseGroup, EstimateDataset
from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.tf import StatisticalEstimate
# =============================================================================

class TransferFunctionGroup(BaseGroup):
    """
    Object to hold a single transfer function estimation
    """
    
    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)
        
        self._accepted_estimates = [
            "transfer_function",
            "transfer_function_error",
            "inverse_signal_power",
            "residual_covariance",
            "impedance",
            "impedance_error",
            "tipper",
            "tipper_error",
            ]
        

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
                
            except (OSError, RuntimeError, ValueError):
                self.logger.warning("period already exists, overwriting")
                self.hdf5_group["period"][...] = period
                

    def add_statistical_estimate(self,
                                 estimate_name,
                                 estimate_data=None,
                                 estimate_metadata=None,
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
    
        estimate_name = validate_name(estimate_name)
        
        if estimate_metadata is None:
            estimate_metadata = StatisticalEstimate()
            estimate_metadata.name = estimate_name
        
        if estimate_data is not None:
            if not isinstance(estimate_data, (np.ndarray, xr.DataArray)):
                msg = f"Need to input a numpy or xarray.DataArray not {type(estimate_data)}" 
                self.logger.exception(msg)
                raise TypeError(msg)
                
            if isinstance(estimate_data, xr.DataArray):
                estimate_metadata.output_channels = estimate_data.coords["output"].values.tolist()
                estimate_metadata.input_channels = estimate_data.coords["input"].values.tolist()
                estimate_metadata.name = validate_name(estimate_data.name)
                estimate_metadata.data_type = estimate_data.dtype.name
                
                estimate_data = estimate_data.to_numpy()
                    
            dtype = estimate_data.dtype
        
        else:
            dtype = complex
            chunks = True
            estimate_data = np.zeros((1, 1, 1), dtype=dtype)

        try:
            dataset = self.hdf5_group.create_dataset(
                estimate_name,
                data=estimate_data,
                dtype=dtype,
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
            return EstimateDataset(estimate_dataset)
            
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
            
    def to_tf_object(self):
        """
        Create a mt_metadata.transfer_function.core.TF object from the 
        estimates in the group
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        tf_obj = TF()
        if self.period is not None:
            tf_obj.period = self.period
        else:
            msg = "Period must not be None to create a transfer function object"
            self.logger.error(msg)
            raise ValueError(msg)
            
        for estimate_name in self.groups_list:
            if estimate_name in ["period"]:
                continue
            estimate = self.get_estimate(estimate_name)
            
            try:
                setattr(tf_obj, estimate_name, estimate.to_numpy())
            
            except AttributeError as error:
                self.logger.exception(error)
            
        return tf_obj
    
    def from_tf_object(self, tf_obj):
        """
        Create data sets from a :class:`mt_metadata.transfer_function.core.TF`
        object.
        
        :param tf_obj: DESCRIPTION
        :type tf_obj: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        if not isinstance(tf_obj, TF):
            msg = "Input must be a TF object not %s"
            self.logger.error(msg, type(tf_obj))
            raise ValueError(msg % type(tf_obj))
            
        self.period = tf_obj.period
        self.metadata.update(tf_obj.station_metadata.transfer_function)
        self.write_metadata()
        
        for estimate_name in self._accepted_estimates:
            try:
                estimate = getattr(tf_obj, estimate_name)
                if estimate is not None:
                    _ = self.add_statistical_estimate(estimate_name, 
                                                      estimate)
                else:
                    self.logger.warning(f"Did not find {estimate_name} in TF. Skipping")
            except AttributeError:
                self.logger.warning(f"Did not find {estimate_name} in TF. Skipping")
                
                
                
            
            
        
        
    