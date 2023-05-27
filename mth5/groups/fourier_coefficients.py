# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:49:32 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

from mth5.groups import BaseGroup
from mth5.utils.exceptions import MTH5Error
from mth5.helpers import validate_name

from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Decimation,
    Channel,
    FC,
)

# =============================================================================
"""fc -> FCMasterGroup -> FCGroup -> DecimationLevelGroup -> ChannelGroup -> FCDataset"""


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

        return self._add_group(
            fc_name, FCGroup, group_metadata=fc_metadata, match="id"
        )

    def get_fc_group(self, fc_name):
        """
        Get Fourier Coefficient group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return self._get_group(fc_name, FCGroup)

    def remove_fc_group(self, fc_name):
        """
        Remove an FC group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        fc_name = validate_name(fc_name)
        try:
            del self.hdf5_group[fc_name]
            self.logger.info(
                "Deleting a fc does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{fc_name} does not exist, "
                + "check fc_list for existing names"
            )
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)


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

    def __init__(self, group, decimation_level_metadata=None, **kwargs):

        super().__init__(
            group, group_metadata=decimation_level_metadata, **kwargs
        )

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

        decimation_level_name = validate_name(decimation_level_name)

        try:
            decimation_group = self.hdf5_group.create_group(
                decimation_level_name
            )
            if decimation_level_metadata is None:
                decimation_level_metadata = Decimation(
                    decimation_level=decimation_level_name
                )
            else:
                if (
                    validate_name(decimation_level_metadata.decimation_level)
                    != decimation_level_name
                ):
                    msg = (
                        f"FC group name {decimation_level_name} must be same as "
                        f"fc_metadata.id {decimation_level_metadata.decimation_level}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            fc_obj = FCDecimationGroup(
                decimation_group,
                decimation_level_metadata=decimation_level_metadata,
                **self.dataset_options,
            )
            fc_obj.initialize_group()

        except ValueError:
            msg = "FC %s already exists, returning existing group."
            self.logger.info(msg, decimation_level_name)
            fc_obj = self.get_fc_group(decimation_level_name)
        return fc_obj

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


class FCDecimationGroup(BaseGroup):
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

    def add_fc_dataset(
        self,
        fc_name,
        fc_data=None,
        fc_metadata=None,
        max_shape=(None, None, None),
        chunks=True,
        **kwargs,
    ):
        """

        Add a set of Fourier coefficients for a single channel at a single
        decimation level for a processing run.

        - time
        - frequency [ integer as harmonic index or float ]
        - fc (complex)

        Weights should be a separate data set as a 1D array along with matching
        index as fcs.

        - weight_channel (maybe)
        - weight_band (maybe)
        - weight_time (maybe)

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

    def add_weights(
        self,
        weight_name,
        weight_data=None,
        weight_metadata=None,
        max_shape=(None, None, None),
        chunks=True,
        **kwargs,
    ):
        pass
