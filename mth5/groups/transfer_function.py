# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:22:33 2022

@author: jpeacock
"""

# =============================================================================
# 
# =============================================================================
import xarray as xr

from mth5.groups import BaseGroup

from mt_metadata.transfer_functions.core import TF
# =============================================================================

class TransferFunctionGroup(BaseGroup):
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
    