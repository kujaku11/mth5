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
import h5py

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

        self._period_metadata = StatisticalEstimate(
            **{
                "name": "period",
                "data_type": "float",
                "description": "Periods at which transfer function is estimated",
                "units": "samples per second",
            }
        )

    def has_estimate(self, estimate):
        """ 
        has estimate
        """

        if estimate in self.groups_list:
            est = self.get_estimate(estimate)
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            return True

        elif estimate in ["impedance"]:
            est = self.get_estimate("transfer_function")
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            elif (
                "ex" in est.metadata.output_channels
                and "ey" in est.metadata.output_channels
            ):
                return True
            return False

        elif estimate in ["tipper"]:
            est = self.get_estimate("transfer_function")
            if est.hdf5_dataset.shape == (1, 1, 1):
                return False
            elif "hz" in est.metadata.output_channels:
                return True
            return False

        elif estimate in ["covariance"]:
            try:
                res = self.get_estimate("residual_covariance")
                isp = self.get_estimate("inverse_signal_power")

                if res.hdf5_dataset.shape != (1, 1, 1) and isp.hdf5_dataset.shape != (
                    1,
                    1,
                    1,
                ):
                    return True
                return False
            except (KeyError, MTH5Error):
                return False

        return False

    @property
    def tf_entry(self):
        """
        Entry for the summary table
        
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return np.array(
            [
                (
                    "",
                    0,
                    0,
                    0,
                    self.metadata.id,
                    self.metadata.units,
                    self.has_estimate("impedance"),
                    self.has_estimate("tipper"),
                    self.has_estimate("covariance"),
                    self.period.min(),
                    self.period.max(),
                    self.hdf5_group.ref,
                    None,
                )
            ],
            dtype=np.dtype(
                [
                    ("station", "U10"),
                    ("latitude", float),
                    ("longitude", float),
                    ("elevation", float),
                    ("tf_id", "U20"),
                    ("units", "U25"),
                    ("has_impedance", bool),
                    ("has_tipper", bool),
                    ("has_covariance", bool),
                    ("period_min", float),
                    ("period_max", float),
                    ("hdf5_reference", h5py.ref_dtype),
                    ("station_hdf5_reference", h5py.ref_dtype),
                ]
            ),
        )

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
                _ = self.add_statistical_estimate(
                    "period",
                    estimate_data=period,
                    estimate_metadata=self._period_metadata,
                    chunks=True,
                    max_shape=(None,),
                )

            except (OSError, RuntimeError, ValueError):
                self.logger.debug("period already exists, overwriting")
                self.hdf5_group["period"][...] = period

    def add_statistical_estimate(
        self,
        estimate_name,
        estimate_data=None,
        estimate_metadata=None,
        max_shape=(None, None, None),
        chunks=True,
        **kwargs,
    ):
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
                estimate_metadata.output_channels = estimate_data.coords[
                    "output"
                ].values.tolist()
                estimate_metadata.input_channels = estimate_data.coords[
                    "input"
                ].values.tolist()
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

            estimate_dataset = EstimateDataset(
                dataset, dataset_metadata=estimate_metadata
            )

        except (OSError, RuntimeError, ValueError) as error:
            self.logger.error(error)
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
            estimate_metadata = StatisticalEstimate(**dict(estimate_dataset.attrs))
            return EstimateDataset(estimate_dataset, dataset_metadata=estimate_metadata)

        except KeyError:
            msg = (
                f"{estimate_name} does not exist, "
                + "check groups_list for existing names"
            )
            self.logger.error(msg)
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
            self.logger.error(msg)
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

        # if transfer function is available then impedance and tipper are
        # redundant.
        if tf_obj.has_transfer_function():
            accepted_estimates = self._accepted_estimates[0:4]
        else:
            accepted_estimates = self._accepted_estimates

        for estimate_name in accepted_estimates:
            try:
                estimate = getattr(tf_obj, estimate_name)
                if estimate is not None:
                    _ = self.add_statistical_estimate(estimate_name, estimate)
                else:
                    self.logger.warning(f"Did not find {estimate_name} in TF. Skipping")
            except AttributeError:
                self.logger.warning(f"Did not find {estimate_name} in TF. Skipping")
