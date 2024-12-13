# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:40:34 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import pandas as pd
import h5py

from mth5.groups import BaseGroup, FCChannelDataset

# from mth5.groups import FCGroup

from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Channel,
)
from mt_metadata.transfer_functions.processing.fourier_coefficients.decimation import (
    Decimation,
)

# =============================================================================
"""fc -> FCMasterGroup -> FCGroup -> DecimationLevelGroup -> ChannelGroup -> FCChannelDataset"""


class MasterFeatureGroup(BaseGroup):
    """
    Master group to hold various Fourier coefficient estimations of time series
    data.
    No metadata needed as of yet.
    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    def add_feature_group(self, feature_name: str, feature_metadata=None):
        """
        Add a Fourier Coefficent group

        :param feature_name: DESCRIPTION
        :type feature_name: TYPE
        :param feature_metadata: DESCRIPTION, defaults to None
        :type feature_metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._add_group(
            feature_name,
            FeatureGroup,
            group_metadata=feature_metadata,
            match="id",
        )

    def get_feature_group(self, feature_name):
        """
        Get Fourier Coefficient group

        :param feature_name: DESCRIPTION
        :type feature_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return self._get_group(feature_name, featureGroup)

    def remove_feature_group(self, feature_name):
        """
        Remove an feature group

        :param feature_name: DESCRIPTION
        :type feature_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self._remove_group(feature_name)
