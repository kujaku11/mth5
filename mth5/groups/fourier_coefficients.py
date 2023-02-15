# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:16:27 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import h5py

from mth5.groups import BaseGroup
from mth5.helpers import validate_name, from_numpy_type
from mth5.utils.exceptions import MTH5Error


# =============================================================================
class FourierCoefficientGroup(BaseGroup):
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

        self._period_metadata = StatisticalEstimate(
            **{
                "name": "period",
                "data_type": "float",
                "description": "Periods at which transfer function is estimated",
                "units": "samples per second",
            }
        )
