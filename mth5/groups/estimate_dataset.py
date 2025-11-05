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

from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
from mt_metadata.utils.validators import validate_attribute
from mth5.utils.exceptions import MTH5Error
from mth5.helpers import to_numpy_type, add_attributes_to_metadata_class_pydantic

# =============================================================================


class EstimateDataset:
    """
    Holds a statistical estimate

    This will hold multi-dimensional statistical estimates for transfer
    functions.

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

    def __init__(self, dataset, dataset_metadata=None, write_metadata=True, **kwargs):

        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()
        self.logger = logger

        # set metadata to the appropriate class.  Standards is not a
        # Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = add_attributes_to_metadata_class_pydantic(StatisticalEstimate)
        self.metadata.hdf5_reference = self.hdf5_dataset.ref
        self.metadata.mth5_type = self._class_name

        # if the input data set already has filled attributes, namely if the
        # channel data already exists then read them in with our writing back
        if "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.metadata.from_dict(
                {self.hdf5_dataset.attrs["mth5_type"]: dict(self.hdf5_dataset.attrs)}
            )
        # if metadata is input, make sure that its the same class type amd write
        # to the hdf5 dataset
        if dataset_metadata is not None:
            if not isinstance(self.metadata, type(dataset_metadata)):
                msg = (
                    f"metadata must be type metadata.{self._class_name} not "
                    "{type(dataset_metadata)}"
                )
                self.logger.error(msg)
                raise MTH5Error(msg)
            # load from dict because of the extra attributes for MTH5
            self.metadata.update(dataset_metadata)
            # self.metadata.hdf5_reference = self.hdf5_dataset.ref
            # self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            if write_metadata:
                self.write_metadata()
        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

    def __str__(self):
        return self.metadata.to_json()

    def __repr__(self):
        return self.__str__()

    @property
    def _class_name(self):
        return validate_attribute(self.__class__.__name__.split("Dataset")[0])

    def read_metadata(self):
        """
        Read metadata from the HDF5 file into the metadata container, that
        way it can be validated.

        """

        self.metadata.from_dict({self._class_name: dict(self.hdf5_dataset.attrs)})

    def write_metadata(self):
        """
        Write metadata from the metadata container to the HDF5 attrs
        dictionary.

        """
        meta_dict = self.metadata.to_dict()[self.metadata._class_name.lower()]
        for key, value in meta_dict.items():
            value = to_numpy_type(value)
            self.hdf5_dataset.attrs.create(key, value)

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

    def to_xarray(self, period):
        """
        :return: an xarray DataArray with appropriate metadata and the
         appropriate coordinates.
        :rtype: :class:`xarray.DataArray`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads from memory
        """

        return xr.DataArray(
            data=self.hdf5_dataset[()],
            dims=["period", "output", "input"],
            name=self.metadata.name,
            coords=[
                ("period", period),
                ("output", self.metadata.output_channels),
                ("input", self.metadata.input_channels),
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
            msg = f"Input array must be type {new_estimate.dtype} not {self.hdf5_dataset.dtype}"
            self.logger.error(msg)
            raise TypeError(msg)
        if new_estimate.shape != self.hdf5_dataset.shape:
            self.hdf5_dataset.resize(new_estimate.shape)
        self.hdf5_dataset[...] = new_estimate

    def from_xarray(self, data):
        """
        :return: an xarray DataArray with appropriate metadata and the
         appropriate coordinates base on the metadata.
        :rtype: :class:`xarray.DataArray`

        .. note:: that metadta will not be validated if changed in an xarray.

        loads from memory
        """

        self.metadata.output_channels = data.coords["output"].values.tolist()
        self.metadata.input_channels = data.coords["input"].values.tolist()
        self.metadata.name = data.name
        self.metadata.data_type = data.dtype.name

        self.write_metadata()

        self.from_numpy(data.to_numpy())
