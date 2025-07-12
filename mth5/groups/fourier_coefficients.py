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
import h5py
from typing import Optional, Union, List, Dict

from mth5.groups import BaseGroup, FCChannelDataset

# from mth5.groups import FCGroup

from mth5.helpers import validate_name
from mth5.utils.exceptions import MTH5Error

import mt_metadata.transfer_functions.processing.fourier_coefficients as fc


# =============================================================================
"""fc -> FCMasterGroup -> FCGroup -> DecimationLevelGroup -> ChannelGroup -> FCChannelDataset"""


class MasterFCGroup(BaseGroup):
    """
    Master group to hold various Fourier coefficient estimations of time series
    data.
    No metadata needed as of yet.

    master -> processing run -> decimation levels -> channels
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

    def add_fc_group(self, fc_name: str, fc_metadata=None):  # -> FCGroup:
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

    def remove_fc_group(self, fc_name: str) -> None:
        """
        Remove an FC group

        :param fc_name: DESCRIPTION
        :type fc_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self._remove_group(fc_name)


class FCDecimationGroup(BaseGroup):
    """
    Holds a single decimation level

    FCDecimationGroup assumes two conditions on the data array (spectrogram):
        1. The data are uniformly sampled in frequency domain
        2. The data are uniformly sampled in time.
        (i.e. the FFT moving window has a uniform step size)

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
        - per_window_detrend_type
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

    @BaseGroup.metadata.getter
    def metadata(self):
        """Overwrite get metadata to include channel information in the runs"""

        self._metadata.channels = []
        for ch in self.groups_list:
            ch_group = self.get_channel(ch)
            self._metadata.channels.append(ch_group.metadata)
            self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def channel_summary(self):
        """

         summary of channels in run
        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FCChannel"]:
                    ch_list.append(
                        (
                            group.attrs["component"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.shape[0],
                            group.shape[1],
                            group.attrs["sample_rate_decimation_level"],
                            group.attrs["sample_rate_window_step"],
                            group.attrs["units"],
                            group.ref,
                        )
                    )
            except KeyError as error:
                self.logger.debug(f"Cannot find a key: {error}")
        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("n_frequency", np.int64),
                    ("n_windows", np.int64),
                    ("sample_rate_decimation_level", np.float64),
                    ("sample_rate_window_step", np.float64),
                    ("units", "U25"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def from_dataframe(
        self, df: pd.DataFrame, channel_key: str, time_key: str = "time", frequency_key: str = "frequency"
    ) -> None:
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
            msg = f"Must input a pandas dataframe not {type(df)}"
            self.logger.error(msg)
            raise TypeError(msg)
        for col in df.columns:
            df[col] = np.complex128(df[col])
            xrds = df[col].to_xarray()
            self.add_channel(col, fc_data=xrds.to_numpy())

    def from_xarray(self, data_array: Union[xr.Dataset, xr.DataArray], sample_rate_decimation_level: float) -> None:
        """
        can input a dataarray or dataset

        :param data_array: DESCRIPTION
        :type data_array: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(data_array, (xr.Dataset, xr.DataArray)):
            msg = f"Must input a xarray Dataset or DataArray not {type(data_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
        ch_metadata = fc.Channel()
        ch_metadata.time_period.start = data_array.time[0].values
        ch_metadata.time_period.end = data_array.time[-1].values
        ch_metadata.sample_rate_decimation_level = sample_rate_decimation_level
        ch_metadata.frequency_min = data_array.coords["frequency"].data.min()
        ch_metadata.frequency_max = data_array.coords["frequency"].data.max()
        step_size = (
            data_array.coords["time"].data[1]
            - data_array.coords["time"].data[0]
        )
        ch_metadata.sample_rate_window_step = step_size / np.timedelta64(1, "s")
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
                if ch in self.channel_summary.component.to_list():
                    self.remove_channel(ch)
                # time index should be the first index
                if data_array[ch].time.size == data_array[ch].shape[0]:
                    self.add_channel(
                        ch,
                        fc_data=data_array[ch].to_numpy(),
                        fc_metadata=ch_metadata,
                        dtype=data_array[ch].dtype,
                    )
                elif data_array[ch].time.size == data_array[ch].shape[1]:
                    self.add_channel(
                        ch,
                        fc_data=data_array[ch].to_numpy().T,
                        fc_metadata=ch_metadata,
                        dtype=data_array[ch].dtype,
                    )
        return

    def to_xarray(self, channels: Optional[List[str]] = None) -> xr.Dataset:
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

    def from_numpy_array(self, nd_array: np.ndarray, ch_name: str) -> None:
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
            msg = f"Must input a numpy ndarray not {type(nd_array)}"
            self.logger.error(msg)
            raise TypeError(msg)
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
        fc_name: str,
        fc_data: Optional[np.ndarray] = None,
        fc_metadata: Optional[fc.Channel] = None,
        max_shape: tuple = (None, None),
        chunks: bool = True,
        dtype: type = complex,
        **kwargs,
    ) -> FCChannelDataset:
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
            fc_metadata = fc.Channel(name=fc_name)
        if fc_data is not None:
            if not isinstance(
                fc_data, (np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame)
            ):
                msg = (
                    "Need to input a numpy.array, xarray.DataArray, "
                    f"xr.Dataset, pd.DataFrame not {type(fc_data)}"
                )
                self.logger.exception(msg)
                raise TypeError(msg)
        else:
            chunks = True
            fc_data = np.zeros((1, 1), dtype=dtype)
        try:
            dataset = self.hdf5_group.create_dataset(
                fc_name,
                data=fc_data,
                dtype=dtype,
                chunks=chunks,
                maxshape=max_shape,
                **self.dataset_options,
            )

            fc_dataset = FCChannelDataset(dataset, dataset_metadata=fc_metadata)
        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
            msg = f"estimate {fc_metadata.component} already exists, returning existing group."
            self.logger.debug(msg)

            fc_dataset = self.get_channel(fc_metadata.component)
        return fc_dataset

    def get_channel(self, fc_name: str) -> FCChannelDataset:
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
            fc_metadata = fc.Channel(**dict(fc_dataset.attrs))
            return FCChannelDataset(fc_dataset, dataset_metadata=fc_metadata)
        except KeyError:
            msg = f"{fc_name} does not exist, check groups_list for existing names"
            self.logger.error(msg)
            raise MTH5Error(msg)
        except OSError as error:
            self.logger.error(error)
            raise MTH5Error(error)

    def remove_channel(self, fc_name: str) -> None:
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
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = f"{fc_name} does not exist, check groups_list for existing names"
            self.logger.error(msg)
            raise MTH5Error(msg)

    def update_metadata(self) -> None:
        """
        update metadata from channels

        :return: DESCRIPTION
        :rtype: TYPE

        """
        channel_summary = self.channel_summary.copy()

        if not channel_summary.empty:
            self._metadata.time_period.start = (
                channel_summary.start.min().isoformat()
            )
            self._metadata.time_period.end = (
                channel_summary.end.max().isoformat()
            )
            self._metadata.sample_rate_decimation_level = (
                channel_summary.sample_rate_decimation_level.unique()[0]
            )
            self._metadata.sample_rate_window_step = (
                channel_summary.sample_rate_window_step.unique()[0]
            )
            self.write_metadata()

    def add_feature(
        self,
        feature_name: str,
        feature_data: Optional[np.ndarray] = None,
        feature_metadata: Optional[Dict] = None,
        max_shape: tuple = (None, None, None),
        chunks: bool = True,
        **kwargs,
    ) -> None:
        pass


class FCGroup(BaseGroup):
    """
    Holds a set of Fourier Coefficients based on a single set of configuration
    parameters. A processing run.

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

    @BaseGroup.metadata.getter
    def metadata(self) -> fc.Decimation:
        """Overwrite get metadata to include channel information in the runs"""

        self._metadata.channels = []
        for dl in self.groups_list:
            dl_group = self.get_decimation_level(dl)
            self._metadata.levels.append(dl_group.metadata)
        self._metadata.hdf5_reference = self.hdf5_group.ref
        return self._metadata

    @property
    def decimation_level_summary(self) -> pd.DataFrame:
        """

        summary of channels in run
        :return: DESCRIPTION
        :rtype: pd.DataFrame

        """

        ch_list = []
        for key, group in self.hdf5_group.items():
            try:
                ch_type = group.attrs["mth5_type"]
                if ch_type in ["FCDecimation"]:
                    ch_list.append(
                        (
                            group.attrs["decimation_level"],
                            group.attrs["time_period.start"].split("+")[0],
                            group.attrs["time_period.end"].split("+")[0],
                            group.ref,
                        )
                    )
            except KeyError as error:
                self.logger.debug(f"Could not find key: {error}")
    
        ch_summary = np.array(
            ch_list,
            dtype=np.dtype(
                [
                    ("component", "U20"),
                    ("start", "datetime64[ns]"),
                    ("end", "datetime64[ns]"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

        return pd.DataFrame(ch_summary)

    def add_decimation_level(
        self, decimation_level_name: str, decimation_level_metadata: Optional[Union[Dict, fc.Decimation]] = None
    ) -> FCDecimationGroup:
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

    def get_decimation_level(self, decimation_level_name: str) -> FCDecimationGroup:
        """
        Get a Decimation Level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return self._get_group(decimation_level_name, FCDecimationGroup)

    def remove_decimation_level(self, decimation_level_name: str) -> None:
        """
        Remove decimation level

        :param decimation_level_name: DESCRIPTION
        :type decimation_level_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self._remove_group(decimation_level_name)

    def update_metadata(self) -> None:
        """
        update metadata from channels

        :return: DESCRIPTION
        :rtype: TYPE

        """
        decimation_level_summary = self.decimation_level_summary.copy()
        if not decimation_level_summary.empty:
            self._metadata.time_period.start = (
                decimation_level_summary.start.min().isoformat()
            )
            self._metadata.time_period.end = (
                decimation_level_summary.end.max().isoformat()
            )
            self.write_metadata()

    def supports_aurora_processing_config(
        self, processing_config: 'aurora.config.metadata.processing.Processing', remote: bool
    ) -> bool:
        """

        An "all-or-nothing" check: Return True if every (valid) decimation needed to satisfy the processing_config
         is available in the FCGroup (self) otherwise return False (and we will build all FCs).

        Logic:
        1. Get a list of all fc groups in the FCGroup (self)
        2. Loop the processing_config decimations, checking if there is a corresponding, already built FCDecimation
         in the FCGroup.

        Parameters
        ----------
        processing_config: aurora.config.metadata.processing.Processing
        remote: bool

        Returns
        -------

        """
        pre_existing_fc_decimation_ids_to_check = self.groups_list
        levels_present = np.full(processing_config.num_decimation_levels, False)

        for i, aurora_decimation_level in enumerate(processing_config.decimations):

            # Quit checking if dec_level wasn't there
            if i > 0:
                if not levels_present[i - 1]:
                    return False

            # iterate over existing decimations
            for fc_decimation_id in pre_existing_fc_decimation_ids_to_check:
                fc_dec_group = self.get_decimation_level(fc_decimation_id)
                fc_decimation = fc_dec_group.metadata
                levels_present[i] = aurora_decimation_level.is_consistent_with_archived_fc_parameters(
                    fc_decimation=fc_decimation,
                    remote=remote
                )
                if levels_present[i]:
                    pre_existing_fc_decimation_ids_to_check.remove(
                        fc_decimation_id
                    )  # no need to check this one again
                    break  # break inner for-loop over decimations

        return levels_present.all()
