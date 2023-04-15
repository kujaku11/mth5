# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:49:32 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import h5py

from mth5.groups import BaseGroup
from mth5.utils.exceptions import MTH5Error
from mth5.helpers import validate_name

from mt_metadata.transfer_functions.fourier_coefficients import (
    Decimation,
    Channel,
    FC,
)

# =============================================================================
"""Station -> FCMasterGroup -> FCGroup -> DecimationLevelGroup -> ChannelGroup -> FCDataset"""


class MasterFCGroup(BaseGroup):
    """
    Master group to hold various Fourier coefficient estimations of time series
    data.
    No metadata needed as of yet.
    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

    @property
    def fc_summary(self):
        """
        Summar of fourier coefficients

        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def add_fc_group(self, fc_name, fc_metadata=None):
        """
        Add a Fourier Coefficent group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :param fc_metadata: DESCRIPTION, defaults to None
        :type fc_metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def get_fc_group(self, fc_name):
        """
        Get Fourier Coefficient group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def remove_fc_group(self, fc_name):
        """
        Remove an FC group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        pass


class FCGroup(BaseGroup):
    """
    Holds a set of Fourier Coefficients based on a single set of configuration
    parameters.

    .. note:: Must be calibrated FCs. Otherwise weird things will happen, can
     always rerun the FC estimation if the metadata changes.

    Metadata should include:

        - list of decimation levels
        - start time (earliest)
        - end time (latest)
        - method (fft, wavelet, ...)
        - list of channels (all inclusive)
        - list of acquistion runs (maybe)
        - starting sample rate

    """

    def __init__(self, group, fc_metadata=None, **kwargs):

        super().__init__(group, group_metadata=fc_metadata, **kwargs)

    def add_decimation_level(
        self, decimation_level_name, decimation_level_metadata=None
    ):
        """
        add a Decimation level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :param decimation_level_metadata: DESCRIPTION, defaults to None
        :type decimation_level_metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        pass

    def get_decimation_level(self, decimation_level_name):
        """
        Get a Decimation Level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def remove_decimation_level(self, decimation_level_name):
        """
        Remove decimation level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass


class FCDecimationLevelGroup(BaseGroup):
    """
    Holds a single decimation level

    Attributes

        - start time
        - end time
        - channels (list)
        - decimation factor
        - decimation level
        - decimation sample rate
        - method (FFT, wavelet, ...)
        - anti alias filter
        - prewhitening type
        - extra_pre_fft_detrend_type
        - recoloring (True | False)
        - harmonics_kept (index values of harmonics kept (list) | 'all')
        - window parameters
            - length
            - overlap
            - type
            - type parameters
            - window sample rate (method or property)
        - [optional] masking or weighting information

    """

    def __init__(self, group, decimation_level_metadata=None, **kwargs):

        super().__init__(
            group, group_metadata=decimation_level_metadata, **kwargs
        )

    def add_channel(self, channel_name, channel_metadata=None):
        """
        Add FC coefficients for a single channel

        :param channel_name: DESCRIPTION
        :type channel_name: TYPE
        :param channel_metadata: DESCRIPTION, defaults to None
        :type channel_metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def get_channel(self, channel_name):
        """
        Get Fourier Coefficients for given channel

        :param channel_name: DESCRIPTION
        :type channel_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def remove_channel(self, channel_name):
        """
        Remove channel

        :param channel_name: DESCRIPTION
        :type channel_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass


class FCChannel(BaseGroup):
    """
    Holds FC information for a single channel at a single decimation level.

    Attributes

        - name
        - start time
        - end time
        - acquistion_sample_rate
        - decimated_sample rate
        - window_sample_rate (delta_t within the window) [property?]
        - units
        - [optional] weights or masking

    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

    def add_fc_dataset(self, fc_datastet_name, fc_dataset_metadata=None):
        """
        Add a set of Fourier coefficients for a single channel at a single
        decimation level for a processing run.

        :param fc_datastet_name: DESCRIPTION
        :type fc_datastet_name: TYPE
        :param fc_dataset_metadata: DESCRIPTION, defaults to None
        :type fc_dataset_metadata: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def get_fc_dataset(self, fc_dataset_name):
        """
        get an fc dataset

        :param fc_dataset_name: DESCRIPTION
        :type fc_dataset_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass

    def remove_fc_dataset(self, fc_dataset_name):
        """
        remove an fc dataset

        :param fc_dataset_name: DESCRIPTION
        :type fc_dataset_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass
