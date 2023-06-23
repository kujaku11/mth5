# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:49:32 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import pandas as pd

from mth5.groups import BaseGroup, FCDataset
from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Channel,
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

        self._remove_group(fc_name)


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

        return self._add_group(
            decimation_level_name,
            FCDecimationGroup,
            group_metadata=decimation_level_metadata,
            match="decimation_level",
        )

    def get_decimation_level(self, decimation_level_name):
        """
        Get a Decimation Level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return self._get_group(decimation_level_name, FCDecimationGroup)

    def remove_decimation_level(self, decimation_level_name):
        """
        Remove decimation level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self._remove_group(decimation_level_name)


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

    def from_dataframe(
        self, df, channel_key, time_key="time", frequency_key="frequency"
    ):
        """
        assumes channel_key is the coefficient values

        :param df: DESCRIPTION
        :type df: TYPE
        :param channel_key: DESCRIPTION
        :type channel_key: TYPE
        :param time_key: DESCRIPTION, defaults to "time"
        :type time_key: TYPE, optional
        :param frequency_key: DESCRIPTION, defaults to "frequency"
        :type frequency_key: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(df, pd.DataFrame):
            msg = "Must input a pandas dataframe not %s"
            self.logger.error(msg, type(df))
            raise TypeError(msg % type(df))

        for col in df.columns:
            df[col] = np.complex128(df[col])
            xrds = df[col].to_xarray()
            self.add_channel(col, fc_data=xrds.to_numpy())

    def from_xarray(self, data_array):
        """
        can input a dataarray or dataset

        :param data_array: DESCRIPTION
        :type data_array: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(data_array, (xr.Dataset, xr.DataArray)):
            msg = "Must input a xarray Dataset or DataArray not %s"
            self.logger.error(msg, type(data_array))
            raise TypeError(msg % type(data_array))

        ch_metadata = Channel()
        ch_metadata.time_period.start = data_array.time[0].values
        ch_metadata.time_period.end = data_array.time[-1].values
        ch_metadata.sample_rate_decimation_level = (
            data_array.coords["frequency"].values.max() * 2
        )
        ch_metadata.sample_rate_window_step = np.median(
            np.diff(data_array.coords["time"].values)
        ) / np.timedelta64(1, "s")
        try:
            ch_metadata.units = data_array.units
        except AttributeError:
            self.logger.debug("Could not find 'units' in xarray")

        if isinstance(data_array, xr.DataArray):
            self.add_channel(
                data_array.name,
                fc_data=data_array.to_numpy(),
                fc_metadata=ch_metadata,
            )

        else:
            for ch in data_array.data_vars.keys():
                ch_metadata.component = ch
                self.add_channel(
                    ch,
                    fc_data=data_array[ch].to_numpy(),
                    fc_metadata=ch_metadata,
                )

    def to_xarray(self, channels=None):
        """
        create an xarray dataset from the desired channels. If none grabs all
        channels in the decimation level.

        :param channels: DESCRIPTION, defaults to None
        :type channels: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if channels is None:
            channels = self.groups_list

        ch_dict = {}
        for ch in channels:
            ch_ds = self.get_channel(ch)
            ch_dict[ch] = ch_ds.to_xarray()

        return xr.Dataset(ch_dict)

    def from_numpy_array(self, nd_array, ch_name):
        """
        assumes shape of (n_frequencies, n_windows) or
        (n_channels, n_frequencies, n_windows)

        :param nd_array: DESCRIPTION
        :type nd_array: TYPE
        :param ch_name: name of channel(s)
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(nd_array, (np.nd_array)):
            msg = "Must input a numpy ndarray not %s"
            self.logger.error(msg, type(nd_array))
            raise TypeError(msg % type(nd_array))

        if len(nd_array.shape[0]) == 3:
            for index, ch in zip(nd_array.shape[0], ch_name):
                self.add_channel(ch, fc_data=nd_array[index])
        elif len(nd_array.shape) == 2:
            self.add_channel(ch_name, fc_data=nd_array)
        else:
            raise ValueError(
                "input array must be shaped (n_frequencies, n_windows)"
            )

    def add_channel(
        self,
        fc_name,
        fc_data=None,
        fc_metadata=None,
        max_shape=(None, None),
        chunks=True,
        **kwargs,
    ):
        """

        Add a set of Fourier coefficients for a single channel at a single
        decimation level for a processing run.

        - time
        - frequency [ integer as harmonic index or float ]
        - fc (complex)

        The input can be

        * a numpy array where the index values for the time, frequency, and
         coefficients are supplied in channel_key, frequency_key, channel_key
         as integer values.
        * a numpy structured array, dataframe, or xarray dataset or dataarray
         where the channel_key if not supplied is assumed to the same as the
         fc_name.

        haven't fully tested dataframe or xarray yet.

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

        fc_name = validate_name(fc_name)

        if fc_metadata is None:
            fc_metadata = Channel(name=fc_name)

        if fc_data is not None:
            if not isinstance(
                fc_data, (np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame)
            ):
                msg = (
                    "Need to input a numpy.array, xarray.DataArray, "
                    "xr.Dataset, pd.DataFrame not %s"
                )
                self.logger.exception(msg, type(fc_data))
                raise TypeError(msg % type(fc_data))

        else:
            chunks = True
            fc_data = np.zeros((1, 1), dtype=complex)
        try:
            dataset = self.hdf5_group.create_dataset(
                fc_name,
                data=fc_data,
                dtype=complex,
                chunks=chunks,
                maxshape=max_shape,
                **self.dataset_options,
            )

            fc_dataset = FCDataset(dataset, dataset_metadata=fc_metadata)
        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
            msg = f"estimate {fc_metadata.name} already exists, returning existing group."
            self.logger.debug(msg)

            fc_dataset = self.get_fc_dataset(fc_metadata.name)
        return fc_dataset

    def get_channel(self, fc_name):
        """
        get an fc dataset

        :param fc_dataset_name: DESCRIPTION
        :type fc_dataset_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        fc_name = validate_name(fc_name)

        try:
            fc_dataset = self.hdf5_group[fc_name]
            fc_metadata = Channel(**dict(fc_dataset.attrs))
            return FCDataset(fc_dataset, dataset_metadata=fc_metadata)

        except (KeyError):
            msg = (
                f"{fc_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)

        except (OSError) as error:
            self.logger.error(error)
            raise MTH5Error(error)

    def remove_channel(self, fc_name):
        """
        remove an fc dataset

        :param fc_dataset_name: DESCRIPTION
        :type fc_dataset_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        fc_name = validate_name(fc_name.lower())

        try:
            del self.hdf5_group[fc_name]
            self.logger.info(
                "Deleting a estimate does not reduce the HDF5"
                + "file size it simply remove the reference. If "
                + "file size reduction is your goal, simply copy"
                + " what you want into another file."
            )
        except KeyError:
            msg = (
                f"{fc_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)

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
