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

        self._dtype = np.dtype(
            [
                ("time", "S32"),
                ("frequency", float),
                ("coefficient", complex),
            ]
        )

        super().__init__(
            group, group_metadata=decimation_level_metadata, **kwargs
        )

    def from_dataframe(
        self, df, channel_key, time_key="time", frequency_key="frequency"
    ):
        """

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

        array = np.zeros(df.shape[0], dtype=self._dtype)
        array["time"] = df[time_key]
        array["frequency"] = df[frequency_key]
        array["coefficient"] = df[channel_key]

        return array

    def from_xarray(
        self,
        data_array,
        channel_key,
        time_key="time",
        frequency_key="frequency",
    ):
        """
         get information from an xarray

        :param data_array: DESCRIPTION
        :type data_array: TYPE
        :param coefficient_key: DESCRIPTION
        :type coefficient_key: TYPE
        :param time_key: DESCRIPTION, defaults to "time"
        :type time_key: TYPE, optional
        :param frequency_key: DESCRIPTION, defaults to "frequency"
        :type frequency_key: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(data_array, (xr.Dataset, xr.DataArray)):
            msg = "Must input a xarray Dataset or DataArray not %s"
            self.logger.error(msg, type(data_array))
            raise TypeError(msg % type(data_array))

        array = np.zeros(data_array.shape[0], dtype=self._dtype)
        array["time"] = data_array[time_key]
        array["frequency"] = data_array[frequency_key]
        array["coefficient"] = data_array[channel_key]

        return array

    def from_numpy_array(
        self, nd_array, channel_index=2, frequency_index=1, time_index=0
    ):
        """

        :param array: DESCRIPTION
        :type array: TYPE
        :param channel_index: DESCRIPTION, defaults to 2
        :type channel_index: TYPE, optional
        :param frequency_index: DESCRIPTION, defaults to 1
        :type frequency_index: TYPE, optional
        :param time_index: DESCRIPTION, defaults to 0
        :type time_index: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(nd_array, (np.nd_array)):
            msg = "Must input a numpy ndarray not %s"
            self.logger.error(msg, type(nd_array))
            raise TypeError(msg % type(nd_array))

        if nd_array.shape[0] == 3:
            nd_array = nd_array.T
        if len(nd_array.shape) > 2:
            raise ValueError("input array must be shaped (n, 3)")

        array = np.zeros(nd_array.shape[0], dtype=self._dtype)
        array["time"] = nd_array[:, time_index]
        array["frequency"] = nd_array[:, frequency_index]
        array["coefficient"] = nd_array[:, channel_index]

        return array

    def from_numpy_structured_array(
        self, st_array, channel_key, time_key="time", frequency_key="frequency"
    ):
        """

        :param st_array: DESCRIPTION
        :type st_array: TYPE
        :param channel_key: DESCRIPTION
        :type channel_key: TYPE
        :param time_key: DESCRIPTION, defaults to "time"
        :type time_key: TYPE, optional
        :param frequency_key: DESCRIPTION, defaults to "frequency"
        :type frequency_key: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(st_array, (np.ndarray)):
            msg = "Must input a numpy ndarray not %s"
            self.logger.error(msg, type(st_array))
            raise TypeError(msg % type(st_array))

        if st_array.shape[0] == 3:
            st_array = st_array.T
        if len(st_array.shape) > 2:
            raise ValueError("input array must be shaped (n, 3)")

        array = np.zeros(st_array.shape[0], dtype=self._dtype)
        array["time"] = st_array[time_key]
        array["frequency"] = st_array[frequency_key]
        array["coefficient"] = st_array[channel_key]

        return array

    def add_channel(
        self,
        fc_name,
        fc_data=None,
        fc_metadata=None,
        max_shape=(None),
        chunks=True,
        channel_key=None,
        frequency_key="frequency",
        time_key="time",
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
            if channel_key is None:
                channel_key = fc_name

            if isinstance(fc_data, np.ndarray):
                if fc_data.dtype.names is None:
                    fc_data = self.from_numpy_array(
                        fc_data,
                        channel_index=channel_key,
                        time_index=time_key,
                        frequency_index=frequency_key,
                    )
                else:
                    fc_data = self.from_numpy_structured_array(
                        fc_data,
                        channel_key,
                        time_key=time_key,
                        frequency_key=frequency_key,
                    )
            elif isinstance(fc_data, pd.DataFrame):
                fc_data = self.from_dataframe(
                    fc_data,
                    channel_key,
                    time_key=time_key,
                    frequency_key=frequency_key,
                )
            elif isinstance(fc_data, (xr.Dataset, xr.DataArray)):
                fc_data = self.from_xarray(
                    fc_data,
                    channel_key,
                    time_key=time_key,
                    frequency_key=frequency_key,
                )

            else:
                msg = (
                    "Need to input a numpy.array, xarray.DataArray, "
                    "xr.Dataset, pd.DataFrame not %s"
                )
                self.logger.exception(msg, type(fc_data))
                raise TypeError(msg % type(fc_data))

        else:

            chunks = True
            fc_data = np.zeros((1), dtype=self._dtype)
        try:
            dataset = self.hdf5_group.create_dataset(
                fc_name,
                data=fc_data,
                dtype=self._dtype,
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
