# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:02:16 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import weakref

import h5py
import numpy as np
import xarray as xr
from loguru import logger

from mth5.utils.exceptions import MTH5Error
from mth5.helpers import to_numpy_type
from mth5.timeseries.ts_helpers import make_dt_coordinates

from mt_metadata.transfer_functions.processing.fourier_coefficients import (
    Channel,
)


# =============================================================================


class FCChannelDataset:
    """
    This will hold multi-dimensional set of Fourier Coefficients

    FCDataset assumes two conditions on the data array (spectrogram):
        1. The data are uniformly sampled in frequency domain
        2. The data are uniformly sampled in time.
        (i.e. the FFT moving window has a uniform step size)


    Columns

        - time
        - frequency [ integer as harmonic index or float ]
        - fc (complex)
        - weight_channel (maybe)
        - weight_band (maybe)
        - weight_time (maybe)

    Attributes:

        - name
        - start time
        - end time
        - acquistion_sample_rate
        - decimated_sample rate
        - window_sample_rate (delta_t within the window)
        - units
        - [optional] weights or masking
        - frequency method (integer * window length / delta_t of window)

    :param dataset: hdf5 dataset
    :type dataset: h5py.Dataset
    :param dataset_metadata: data set metadata see
    :class:`mt_metadata.transfer_functions.tf.StatisticalEstimate`,
     defaults to None
    :type dataset_metadata: :class:`mt_metadata.transfer_functions.tf.StatisticalEstimate`, optional
    :param write_metadata: True to write metadata, defaults to True
    :type write_metadata: Boolean, optional
    :param **kwargs: DESCRIPTION
    :type **kwargs: TYPE
    :raises MTH5Error: When an estimate is not present, or metadata name
     does not match the given name

    """

    def __init__(
        self, dataset, dataset_metadata=None, write_metadata=True, **kwargs
    ):

        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = Channel()

        if not hasattr(self.metadata, "mth5_type"):
            self._add_base_attributes()
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name
        # if the input data set already has filled attributes, namely if the
        # channel data already exists then read them in with our writing back
        if "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.metadata.from_dict(
                {self.hdf5_dataset.attrs["mth5_type"]: self.hdf5_dataset.attrs}
            )
        # if metadata is input, make sure that its the same class type amd write
        # to the hdf5 dataset
        if dataset_metadata is not None:
            if not isinstance(dataset_metadata, type(self.metadata)):
                msg = (
                    f"metadata must be type metadata.{self._class_name} not "
                    "{type(dataset_metadata)}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(dataset_metadata.to_dict())
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            try:
                self.write_metadata()
            except RuntimeError:
                # file is read only
                pass

        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

    def _add_base_attributes(self):
        # add 2 attributes that will help with querying
        # 1) the metadata class name
        self.metadata.add_base_attribute(
            "mth5_type",
            self._class_name,
            {
                "type": str,
                "required": True,
                "style": "free form",
                "description": "type of group",
                "units": None,
                "options": [],
                "alias": [],
                "example": "group_name",
                "default": None,
            },
        )

        # 2) the HDF5 reference that can be used instead of paths
        self.metadata.add_base_attribute(
            "hdf5_reference",
            self.hdf5_dataset.ref,
            {
                "type": "h5py_reference",
                "required": True,
                "style": "free form",
                "description": "hdf5 internal reference",
                "units": None,
                "options": [],
                "alias": [],
                "example": "<HDF5 Group Reference>",
                "default": None,
            },
        )

    def __str__(self):
        return self.metadata.to_json()

    def __repr__(self):
        return self.__str__()

    @property
    def _class_name(self):
        return self.__class__.__name__.split("Dataset")[0]

    def read_metadata(self):
        """
        Read metadata from the HDF5 file into the metadata container, that
        way it can be validated.

        """

        self.metadata.from_dict({self._class_name: self.hdf5_dataset.attrs})

    def write_metadata(self):
        """
        Write metadata from the metadata container to the HDF5 attrs
        dictionary.

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.hdf5_dataset.attrs.create(key, value)

    @property
    def n_windows(self):
        """number of time windows"""
        return self.hdf5_dataset.shape[0]

    @property
    def time(self):
        """
        Time array that includes the start of each time window

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return make_dt_coordinates(
            self.metadata.time_period.start,
            1.0 / self.metadata.sample_rate_window_step,
            self.n_windows,
        )

    @property
    def n_frequencies(self):
        """number of frequencies (window size)"""
        return self.hdf5_dataset.shape[1]

    @property
    def frequency(self):
        """
        frequency array dictated by window size and sample rate

        :return: DESCRIPTION
        :rtype: TYPE

        """
        return np.linspace(
            self.metadata.frequency_min,
            self.metadata.frequency_max,
            self.n_frequencies,
        )

    def replace_dataset(self, new_data_array):
        """
        replace the entire dataset with a new one, nothing left behind

        :param new_data_array: new data array
        :type new_data_array: :class:`numpy.ndarray`

        """
        if not isinstance(new_data_array, np.ndarray):
            try:
                new_data_array = np.array(new_data_array)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_data_array)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if new_data_array.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_data_array.shape)
        self.hdf5_dataset[...] = new_data_array

    def to_xarray(self):
        """
        :return: an xarray DataArray with appropriate metadata and the
         appropriate coordinates.
        :rtype: :class:`xarray.DataArray`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads from memory
        """

        return xr.DataArray(
            data=self.hdf5_dataset[()],
            dims=["time", "frequency"],
            name=self.metadata.component,
            coords=[
                ("time", self.time),
                ("frequency", self.frequency),
            ],
            attrs=self.metadata.to_dict(single=True),
        )

    def to_numpy(self):
        """
        :return: a numpy structured array with
        :rtype: :class:`numpy.ndarray`

        loads into RAM

        """

        return self.hdf5_dataset[()]

    def from_numpy(self, new_estimate):
        """
        :return: a numpy structured array
        :rtype: :class:`numpy.ndarray`

        .. note:: data is a builtin to numpy and cannot be used as a name

        loads into RAM

        """

        if not isinstance(new_estimate, np.ndarray):
            try:
                new_estimate = np.array(new_estimate)
            except (ValueError, TypeError) as error:
                msg = f"{error} Input must be a numpy array not {type(new_estimate)}"
                self.logger.exception(msg)
                raise TypeError(msg)
        if new_estimate.dtype != self.hdf5_dataset.dtype:
            msg = (
                f"Input array must be type {new_estimate.dtype} not "
                "{self.hdf5_dataset.dtype}"
            )
            self.logger.error(msg)
            raise TypeError(msg)
        if new_estimate.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_estimate.shape)
        self.hdf5_dataset[...] = new_estimate

    def from_xarray(self, data, sample_rate_decimation_level,):
        """


        :return: an xarray DataArray with appropriate metadata and the
         appropriate coordinates base on the metadata.
        :rtype: :class:`xarray.DataArray`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads from memory
        """
        self.metadata.time_period.start = data.time[0].values
        self.metadata.time_period.end = data.time[-1].values

        self.metadata.sample_rate_decimation_level = sample_rate_decimation_level
        self.metadata.frequency_min = data.coords["frequency"].data.min()
        self.metadata.frequency_max = data.coords["frequency"].data.max()
        step_size = data.coords["time"].data[1] - data.coords["time"].data[0]
        self.metadata.sample_rate_window_step = step_size / np.timedelta64(1, "s")
        self.metadata.component = data.name
        try:
            self.metadata.units = data.units
        except AttributeError:
            self.logger.debug("Could not find 'units' in xarray")
        self.write_metadata()

        self.from_numpy(data.to_numpy())
