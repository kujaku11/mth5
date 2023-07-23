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
import pandas as pd

from mth5.groups import BaseGroup, EstimateDataset
from mth5.helpers import validate_name, from_numpy_type
from mth5.utils.exceptions import MTH5Error

from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.tf import (
    StatisticalEstimate,
    Run,
    Electric,
    Magnetic,
)

# =============================================================================
# Transfer Functions Group
# =============================================================================
class TransferFunctionsGroup(BaseGroup):
    """
    Object to hold transfer functions

    The is the high level group, all transfer functions for the station are
    held here and each one will have its own TransferFunctionGroup.

    This has add, get, remove_transfer_function.
    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

    def tf_summary(self, as_dataframe=True):
        """
        Summary of all transfer functions in this group

        :return: DESCRIPTION
        :rtype: TYPE

        """

        tf_list = []
        for tf_id in self.groups_list:
            tf_group = self.get_transfer_function(tf_id)
            tf_entry = tf_group.tf_entry

            tf_entry["station_hdf5_reference"][:] = self.hdf5_group.parent.ref
            tf_entry["station"][:] = self.hdf5_group.parent.attrs["id"]
            tf_entry["latitude"][:] = self.hdf5_group.parent.attrs["location.latitude"]
            tf_entry["longitude"][:] = self.hdf5_group.parent.attrs[
                "location.longitude"
            ]
            tf_entry["elevation"][:] = self.hdf5_group.parent.attrs[
                "location.elevation"
            ]

            tf_list.append(tf_entry)
        tf_list = np.array(tf_list)

        if as_dataframe:
            return pd.DataFrame(tf_list.flatten())
        return tf_list

    def add_transfer_function(self, name, tf_object=None):
        """
        Add a transfer function to the group

        :param name: name of the transfer function
        :type name: string
        :param tf_object: Transfer Function object
        :type tf_object: :class:`mt_metadata.transfer_function.core.TF`
        :return: DESCRIPTION
        :rtype: TYPE

        >>> from mth5.mth5 import MTH5
        >>> m = MTH5()
        >>> m.open_mth5("example.h5", "a")
        >>> station_group = m.get_station("mt01", survey="test")
        >>> tf_group = station_group.transfer_functions_group
        >>> tf_group.add_transfer_function("mt01_4096", tf_object)


        """
        name = validate_name(name)

        tf_group = TransferFunctionGroup(
            self.hdf5_group.create_group(name), **self.dataset_options
        )

        if tf_object is not None:
            tf_group.from_tf_object(tf_object)
        return tf_group

    def get_transfer_function(self, tf_id):
        """
        Get transfer function from id

        :param tf_id: name of transfer function
        :type tf_id: string
        :return: Transfer function group
        :rtype: :class:`mth5.groups.TransferFunctionGroup`

        >>> from mth5.mth5 import MTH5
        >>> m = MTH5()
        >>> m.open_mth5("example.h5", "a")
        >>> station_group = m.get_station("mt01", survey="test")
        >>> tf_group = station_group.transfer_functions_group.get_transfer_function("mt01_4096")


        """

        tf_id = validate_name(tf_id)
        try:
            return TransferFunctionGroup(self.hdf5_group[tf_id], **self.dataset_options)
        except KeyError:
            msg = f"{tf_id} does not exist, " + "check station_list for existing names"
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)

    def remove_transfer_function(self, tf_id):
        """
        Remove a transfer function from the group

        :param tf_id: DESCRIPTION
        :type tf_id: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        >>> from mth5.mth5 import MTH5
        >>> m = MTH5()
        >>> m.open_mth5("example.h5", "a")
        >>> station_group = m.get_station("mt01", survey="test")
        >>> tf_group = station_group.transfer_functions_group
        >>> tf_group.remove_transfer_function("mt01_4096")

        """

        tf_id = validate_name(tf_id)
        try:
            del self.hdf5_group[tf_id]
            self.logger.info(
                "Deleting a station does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError:
            msg = f"{tf_id} does not exist, " "check station_list for existing names"
            self.logger.debug("Error" + msg)
            raise MTH5Error(msg)

    def get_tf_object(self, tf_id):
        """
        This is the function you want to use to get a proper
        :class:`mt_metadata.transfer_functions.core.TF` object with all the
        appropriate metadata.


        :param tf_id: name of the transfer function to get
        :type tf_id: string
        :return: Full transfer function with appropriate metadata
        :rtype: :class:`mt_metadata.transfer_functions.core.TF`

        >>> from mth5.mth5 import MTH5
        >>> m = MTH5()
        >>> m.open_mth5("example.h5", "a")
        >>> station_group = m.get_station("mt01", survey="test")
        >>> tf_group = station_group.transfer_functions_group
        >>> tf_object = tf_group.get_tf_object("mt01_4096")


        """

        tf_group = self.get_transfer_function(tf_id)

        return tf_group.to_tf_object()


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

                if res.hdf5_dataset.shape != (1, 1, 1,) and isp.hdf5_dataset.shape != (
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
        except (KeyError):
            msg = (
                f"{estimate_name} does not exist, "
                "check groups_list for existing names"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)
        except (OSError) as error:
            self.logger.error(error)
            raise MTH5Error(error)

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
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
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

        # get survey metadata
        survey_dict = dict(self.hdf5_group.parent.parent.parent.parent.attrs)
        for key, value in survey_dict.items():
            survey_dict[key] = from_numpy_type(value)
        tf_obj.survey_metadata.from_dict({"survey": survey_dict})

        # get station metadata
        station_dict = dict(self.hdf5_group.parent.parent.attrs)
        for key, value in station_dict.items():
            station_dict[key] = from_numpy_type(value)
        tf_obj.station_metadata.from_dict({"station": station_dict})

        # need to update transfer function metadata
        tf_dict = dict(self.hdf5_group.attrs)
        for key, value in tf_dict.items():
            tf_dict[key] = from_numpy_type(value)
        tf_obj.station_metadata.transfer_function.from_dict(
            {"transfer_function": tf_dict}
        )

        # add run and channel metadata
        tf_obj.station_metadata.runs = []
        for run_id in tf_obj.station_metadata.transfer_function.runs_processed:
            if run_id in ["", None, "None"]:
                continue
            try:
                run = self.hdf5_group.parent.parent[validate_name(run_id)]
                run_dict = dict(run.attrs)
                for key, value in run_dict.items():
                    run_dict[key] = from_numpy_type(value)
                run_obj = Run(**run_dict)

                for ch_id in run.keys():
                    ch = run[validate_name(ch_id)]
                    ch_dict = dict(ch.attrs)
                    for key, value in ch_dict.items():
                        ch_dict[key] = from_numpy_type(value)
                    if ch_dict["type"] == "electric":
                        ch_obj = Electric(**ch_dict)
                    elif ch_dict["type"] == "magnetic":
                        ch_obj = Magnetic(**ch_dict)
                    run_obj.add_channel(ch_obj)
                tf_obj.station_metadata.add_run(run_obj)
            except KeyError:
                self.logger.info(f"Could not get run {run_id} for transfer function")
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
            msg = f"Input must be a TF object not {type(tf_obj)}"
            self.logger.error(msg)
            raise ValueError(msg)
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
                    self.logger.debug(f"Did not find {estimate_name} in TF. Skipping")
            except AttributeError:
                self.logger.debug(f"Did not find {estimate_name} in TF. Skipping")
