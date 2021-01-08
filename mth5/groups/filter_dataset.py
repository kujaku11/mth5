# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 22:28:28 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import weakref
import logging

import h5py
import numpy as np

from mth5.utils.exceptions import MTH5Error
from mt_metadata import timeseries as metadata
from mth5.helpers import to_numpy_type

# =============================================================================
# Filter Dataset
# =============================================================================
class FilterDataset:
    """
    Holds a filter dataset.  This is a simple container for the filter to make
    sure that the user has the flexibility to turn the channel into an object
    they want to deal with.

    For now all the numpy type slicing can be used on `hdf5_dataset`

    :param dataset: dataset object for the filter
    :type dataset: :class:`h5py.Dataset`
    :param dataset_metadata: metadata container, defaults to None
    :type dataset_metadata:  :class:`mth5.metadata.Filter`, optional
    
    :raises MTH5Error: If the dataset is not of the correct type

    Utilities will be written to create some common objects like:
        
        * xarray.DataArray
        * pandas.DataFrame
        * zarr
        * dask.Array

    The benefit of these other objects is that they can be indexed by time,
    and they have much more buit-in funcionality.

    >>> from mth5 import mth5
    >>> mth5_obj = mth5.MTH5()
    >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
    >>> f = mth5_obj.filter_group.add_filter("table")

    """

    def __init__(self, dataset, dataset_metadata=None, **kwargs):

        if dataset is not None and isinstance(dataset, (h5py.Dataset)):
            self.hdf5_dataset = weakref.ref(dataset)()

        self.logger = logging.getLogger(f"{__name__}.{self._class_name}")

        self.metadata = metadata.Filter()

        if not hasattr(self.metadata, "mth5_type"):
            self._add_base_attributes()

        # set summary attributes
        self.logger.debug(
            "Metadata class for {0} is {1}".format(
                self._class_name, type(self.metadata)
            )
        )

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
                msg = "metadata must be type metadata.{0} not {1}".format(
                    self._class_name, type(dataset_metadata)
                )
                self.logger.error(msg)
                raise MTH5Error(msg)

            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(dataset_metadata.to_dict())
            self.metadata.hdf5_reference = self.hdf5_dataset.ref
            self.metadata.mth5_type = self._class_name

            # write out metadata to make sure that its in the file.
            self.write_metadata()

        # if the attrs don't have the proper metadata keys yet write them
        if not "mth5_type" in list(self.hdf5_dataset.attrs.keys()):
            self.write_metadata()

        # if any other keywords
        for key, value in kwargs.items():
            setattr(self, key, value)

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
            },
        )

    def __str__(self):
        try:
            lines = ["Filter:"]
            lines.append("-" * (len(lines[0]) + 2))
            info_str = "\t{0:<18}{1}"
            lines.append(info_str.format("name", self.name))
            lines.append(info_str.format("type:", self.filter_type))
            lines.append(info_str.format("units in:", self.units_in))
            lines.append(info_str.format("units out:", self.units_out))
            return "\n".join(lines)
        except ValueError:
            return "MTH5 file is closed and cannot be accessed."

    def __repr__(self):
        return self.__str__()

    @property
    def _class_name(self):
        return self.__class__.__name__.split("Dataset")[0]

    @property
    def name(self):
        """ filter name """
        return self.metadata.name

    @name.setter
    def name(self, value):
        """ rename filter """
        self.metadata.name = value
        self.write_metadata()

    @property
    def filter_type(self):
        """ filter type """
        return self.metadata.type

    @filter_type.setter
    def filter_type(self, value):
        """ rename filter type """
        self.metadata.type = value
        self.write_metadata()

    @property
    def units_in(self):
        """ units in  """
        return self.metadata.units_in

    @units_in.setter
    def units_in(self, value):
        """ rename units in """
        self.metadata.units_in = value
        self.write_metadata()

    @property
    def units_out(self):
        """ units out  """
        return self.metadata.units_out

    @units_out.setter
    def units_out(self, value):
        """ rename units out """
        self.metadata.units_out = value
        self.write_metadata()

    @property
    def poles(self):
        """ convenience to poles if there are any """
        if "poles_real" in self.hdf5_dataset.dtype.names:
            pole_index = np.where(self.hdf5_dataset["poles_imag"] != 0)
            return (
                self.hdf5_dataset["poles_real"][pole_index]
                + 1j * self.hdf5_dataset["poles_imag"][pole_index]
            )

    @property
    def zeros(self):
        """ convenience to zeros if there are any """
        if "zeros_real" in self.hdf5_dataset.dtype.names:
            zero_index = np.where(self.hdf5_dataset["zeros_imag"] != 0)
            return (
                self.hdf5_dataset["zeros_real"][zero_index]
                + 1j * self.hdf5_dataset["zeros_imag"][zero_index]
            )

    @property
    def zpk_gain(self):
        return self.metadata.normalization_factor

    @property
    def frequency(self):
        if "frequency" in self.hdf5_dataset.dtype.names:
            return self.hdf5_dataset["frequency"]

    @property
    def real(self):
        if "real" in self.hdf5_dataset.dtype.names:
            return self.hdf5_dataset["real"]

    @property
    def imaginary(self):
        if "imag" in self.hdf5_dataset.dtype.names:
            return self.hdf5_dataset["imag"]

    def read_metadata(self):
        """
        read metadata from the HDF5 group into metadata object

        """

        self.metadata.from_dict({self._class_name: self.hdf5_group.attrs})

    def write_metadata(self):
        """
        Write HDF5 metadata from metadata object.

        """

        for key, value in self.metadata.to_dict(single=True):
            value = to_numpy_type(value)
            self.logger.debug(f"wrote metadata {key} = {value}".format(key, value))
            self.hdf5_dataset.attrs.create(key, value)

    def to_filter_object(self):
        """
        convert to a :class:`mth5.filters.Filter` object
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
