# -*- coding: utf-8 -*-
"""
Base Group Class
====================

Contains all the base functions that will be used by group classes.

Created on Fri May 29 15:09:48 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT
"""
# =============================================================================
# Imports
# =============================================================================
import inspect
import logging
import weakref

import h5py
import numpy as np

from mth5 import metadata
from mth5.helpers import get_tree
from mth5.utils.exceptions import MTH5Error
from mth5.utils.helpers import to_numpy_type

# make a dictionary of available metadata classes
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
# =============================================================================
#
# =============================================================================
class BaseGroup:
    """
    Generic object that will have functionality for reading/writing groups,
    including attributes. To access the hdf5 group directly use the
    `BaseGroup.hdf5_group` property.

    >>> base = BaseGroup(hdf5_group)
    >>> base.hdf5_group.ref
    <HDF5 Group Reference>

    .. note:: All attributes should be input into the metadata object, that
             way all input will be validated against the metadata standards.
             If you change attributes in metadata object, you should run the
             `BaseGroup.write_metadata` method.  This is a temporary solution
             working on an automatic updater if metadata is changed.

    >>> base.metadata.existing_attribute = 'update_existing_attribute'
    >>> base.write_metadata()

    If you want to add a new attribute this should be done using the
    `metadata.add_base_attribute` method.

    >>> base.metadata.add_base_attribute('new_attribute',
    ...                                  'new_attribute_value',
    ...                                  {'type':str,
    ...                                   'required':True,
    ...                                   'style':'free form',
    ...                                   'description': 'new attribute desc.',
    ...                                   'units':None,
    ...                                   'options':[],
    ...                                   'alias':[],
    ...                                   'example':'new attribute'})

    Includes intializing functions that makes a summary table and writes
    metadata.

    """

    def __init__(self, group, group_metadata=None, **kwargs):
        self.compression = "gzip"
        self.compression_opts = 3
        self.shuffle = True
        self.fletcher32 = True

        self.logger = logging.getLogger(f"{__name__}.{self._class_name}")

        # make sure the reference to the group is weak so there are no lingering
        # references to a closed HDF5 file.
        if group is not None and isinstance(group, (h5py.Group, h5py.Dataset)):
            self.hdf5_group = weakref.ref(group)()

        # set default columns of summary table.
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (10000,),
            "dtype": np.dtype([("default", np.float)]),
        }

        # set metadata to the appropriate class.  Standards is not a
        # metadata.Base object so should be skipped. If the class name is not
        # defined yet set to Base class.
        self.metadata = metadata.Base()
        if self._class_name not in ["Standards"]:
            try:
                self.metadata = meta_classes[self._class_name]()
            except KeyError:
                self.metadata = metadata.Base()

        # add 2 attributes that will help with querying
        # 1) the metadata class name
        self.metadata.add_base_attribute(
            "mth5_type",
            self._class_name.split("Group")[0],
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
            self.hdf5_group.ref,
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

        # add mth5 and hdf5 attributes
        self.metadata.mth5_type = self._class_name
        self.metadata.hdf5_reference = self.hdf5_group.ref

        # if metadata, make sure that its the same class type
        if group_metadata is not None:
            if not isinstance(group_metadata, (type(self.metadata), metadata.Base)):
                msg = "metadata must be type metadata.{0} not {1}".format(
                    self._class_name, type(group_metadata)
                )
                self.logger.error(msg)
                raise MTH5Error(msg)

            # load from dict because of the extra attributes for MTH5
            self.metadata.from_dict(group_metadata.to_dict())

            # add mth5 and hdf5 attributes because they are overwritten from
            # group metadata
            self.metadata.mth5_type = self._class_name
            self.metadata.hdf5_reference = self.hdf5_group.ref

            # write out metadata to make sure that its in the file.
            self.write_metadata()
        else:
            self.read_metadata()

        # if any other keywords
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        try:
            self.hdf5_group.ref

            return get_tree(self.hdf5_group)
        except ValueError:
            msg = "MTH5 file is closed and cannot be accessed."
            self.logger.warning(msg)
            return msg

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        raise MTH5Error("Cannot test equals yet")

    # Iterate over key, value pairs
    def __iter__(self):
        return self.hdf5_group.items().__iter__()

    @property
    def _class_name(self):
        return self.__class__.__name__.split("Group")[0]

    @property
    def summary_table(self):
        pass
        #return MTH5Table(self.hdf5_group["summary"])

    @property
    def groups_list(self):
        return list(self.hdf5_group.keys())

    @property
    def dataset_options(self):
        return {
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "shuffle": self.shuffle,
            "fletcher32": self.fletcher32,
        }

    def read_metadata(self):
        """
        read metadata from the HDF5 group into metadata object

        """

        self.metadata.from_dict({self._class_name: self.hdf5_group.attrs})

    def write_metadata(self):
        """
        Write HDF5 metadata from metadata object.

        """

        for key, value in self.metadata.to_dict(single=True).items():
            value = to_numpy_type(value)
            self.logger.debug("wrote metadata {0} = {1}".format(key, value))
            self.hdf5_group.attrs.create(key, value)

    def initialize_summary_table(self):
        """
        Initialize summary table as a dataset based on default values to

        ``/Group/summary``

        The initial size is 0, but is extentable to
        `self._defaults_summary_attrs[max_shape]`

        """

        summary_table = self.hdf5_group.create_dataset(
            self._defaults_summary_attrs["name"],
            (0,),
            maxshape=self._defaults_summary_attrs["max_shape"],
            dtype=self._defaults_summary_attrs["dtype"],
            **self.dataset_options,
        )

        summary_table.attrs.update(
            {
                "type": "summary table",
                "last_updated": "date_time",
                "reference": summary_table.ref,
            }
        )

        self.logger.debug(
            "Created {0} table with max_shape = {1}, dtype={2}".format(
                self._defaults_summary_attrs["name"],
                self._defaults_summary_attrs["max_shape"],
                self._defaults_summary_attrs["dtype"],
            )
        )
        self.logger.debug(
            "used options: "
            + "; ".join([f"{k} = {v}" for k, v in self.dataset_options.items()])
        )

    def initialize_group(self):
        """
        Initialize group by making a summary table and writing metadata

        """
        self.initialize_summary_table()
        self.write_metadata()



