# -*- coding: utf-8 -*-
"""
Helper functions for HDF5

Created on Tue Jun  2 12:37:50 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT

"""
# =============================================================================
# Imports
# =============================================================================
from collections.abc import Iterable
import inspect
import numpy as np
import h5py
import gc
from loguru import logger

from pydantic.fields import FieldInfo
from typing import Union

# =============================================================================
# Acceptable compressions
# =============================================================================
COMPRESSION = ["lzf", "gzip", "szip", None]
COMPRESSION_LEVELS = {
    "lzf": [None],
    "gzip": range(10),
    "szip": ["ec-8", "ee-10", "nn-8", "nn-10"],
    None: [None],
}


def validate_compression(compression, level):
    """
    validate that the input compression is supported.

    :param compression: type of lossless compression
    :type compression: string, [ 'lzf' | 'gzip' | 'szip' | None ]
    :param level: compression level if supported
    :type level: string for 'szip' or int for 'gzip'
    :return: compression type
    :rtype: string
    :return: compressiong level
    :rtype: string for 'szip' or int for 'gzip'
    :raises: ValueError if comporession or level are not supported
    :raises: TypeError if compression level is not a string

    """
    if compression is None:
        return None, None
    if not isinstance(compression, (str, type(None))):
        msg = f"compression type must be a string, not {type(compression)}"
        logger.error(msg)
        raise TypeError(msg)
    if not compression in COMPRESSION:
        msg = (
            f"Compression type {compression} not supported. "
            f"Supported options are {COMPRESSION}"
        )
        logger.error(msg)
        raise ValueError(msg)
    if compression == "lzf":
        level = COMPRESSION_LEVELS["lzf"][0]
    elif compression == " gzip":
        if not isinstance(level, (int)):
            msg = (
                f"Level type for gzip must be an int, not {type(level)}. "
                f"Options are {COMPRESSION_LEVELS['gzip']}"
            )
            logger.error(msg)
            raise TypeError(msg)
    elif compression == " szip":
        if not isinstance(level, (str)):
            msg = (
                f"Level type for szip must be an str, not {type(level)}. "
                f"Options are {COMPRESSION_LEVELS['szip']}"
            )
            logger.error(msg)
            raise TypeError(msg)
    if not level in COMPRESSION_LEVELS[compression]:
        msg = (
            f"compression level {level} not supported for {compression}. "
            f"Options are {COMPRESSION_LEVELS[compression]}"
        )

        logger.error(msg)
        raise ValueError(msg)
    return compression, level


def recursive_hdf5_tree(group, lines=[]):
    if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
        for key, value in group.items():
            lines.append(f"-{key}: {value}")
            recursive_hdf5_tree(value, lines)
    elif isinstance(group, h5py._hl.dataset.Dataset):
        for key, value in group.attrs.items():
            lines.append(f"\t-{key}: {value}")
    return "\n".join(lines)


def close_open_files():
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File):
                msg = "Found HDF5 File object "
                logger.debug(msg)
                try:
                    msg = f"{obj.filename}, "
                    obj.flush()
                    obj.close()
                    msg += "Closed File"
                    logger.info(msg)
                except:
                    msg += f"{obj.filename} file already closed."
                    logger.info(msg)
        except:
            logger.debug(f"Object {type(obj)} does not have __class__")


def get_tree(parent):
    """
    Simple function to recursively print the contents of an hdf5 group
    Parameters
    ----------
    parent : :class:`h5py.Group`
        HDF5 (sub-)tree to print

    """
    lines = ["{0}:".format(parent.name), "=" * 20]
    if not isinstance(parent, (h5py.File, h5py.Group)):
        raise TypeError("Provided object is not a h5py.File or h5py.Group " "object")

    def fancy_print(name, obj):
        # lines.append(name)
        spacing = " " * 4 * (name.count("/") + 1)
        group_name = name[name.rfind("/") + 1 :]

        if isinstance(obj, h5py.Group):
            lines.append(f"{spacing}|- Group: {group_name}")
            lines.append("{0}{1}".format(spacing, (len(group_name) + 10) * "-"))
        elif isinstance(obj, h5py.Dataset):
            lines.append(f"{spacing}--> Dataset: {group_name}")
            lines.append("{0}{1}".format(spacing, (len(group_name) + 15) * "."))

    # lines.append(parent.name)
    parent.visititems(fancy_print)
    return "\n".join(lines)


def to_numpy_type(value):
    """
    Need to make the attributes friendly with Numpy and HDF5.

    For numbers and bool this is straight forward they are automatically
    mapped in h5py to a numpy type.

    But for strings this can be a challenge, especially a list of strings.

    HDF5 should only deal with ASCII characters or Unicode.  No binary data
    is allowed.
    """

    if value is None:
        return "none"
    # For now turn references into a generic string
    if isinstance(value, h5py.h5r.Reference):
        value = str(value)
    if isinstance(
        value,
        (
            str,
            np.str_,
            int,
            float,
            bool,
            complex,
            np.int_,
            np.float64,
            np.bool_,
            np.complex128,
        ),
    ):
        return value
    if isinstance(value, Iterable):
        if np.any([type(x) in [str, bytes, np.str_] for x in value]):
            return np.array(value, dtype="S")
        else:
            try:
                return np.array(value)
            except (ValueError, TypeError):
                # If we can't convert to numpy array, convert to string representation
                return str(value)
    else:
        # For pydantic models and other complex objects, convert to string
        try:
            # First try to convert directly
            return np.array(value)
        except (ValueError, TypeError):
            # If that fails, convert to string representation
            return str(value)


def validate_name(name):
    """
    make sure the name has no spaces or slashes

    :param name: DESCRIPTION
    :type name: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    return name.replace(" ", "_").replace("/", "_")


def from_numpy_type(value):
    """
    Need to make the attributes friendly with Numpy and HDF5.

    For numbers and bool this is straight forward they are automatically
    mapped in h5py to a numpy type.

    But for strings this can be a challenge, especially a list of strings.

    HDF5 should only deal with ASCII characters or Unicode.  No binary data
    is allowed.
    """

    if value is None:
        return "none"

    # For now turn references into a generic string
    if isinstance(value, h5py.h5r.Reference):
        value = str(value)
    if isinstance(
        value,
        (
            str,
            np.str_,
            int,
            float,
            bool,
            complex,
            np.int32,
            np.float64,
            np.complex128,
            np.intp,
        ),
    ):
        return value
    # if isinstance(
    #     value,
    #     (
    #         np.int32,
    #     )
    # ):
    #     return np.int64(value)
    if isinstance(value, Iterable):
        if np.any([type(x) in [bytes, np.bytes_] for x in value]):
            return np.array(value, dtype="U").tolist()
        else:
            return np.array(value).tolist()
    else:
        raise TypeError("Type {0} not understood".format(type(value)))


# =============================================================================
#
# =============================================================================
def inherit_doc_string(cls):
    for base in inspect.getmro(cls):
        if base.__doc__ is not None:
            cls.__doc__ = base.__doc__
            break
    return cls


def validate_name(name, pattern=None):
    """
    Validate name

    :param name: DESCRIPTION
    :type name: TYPE
    :param pattern: DESCRIPTION, defaults to None
    :type pattern: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if name is None:
        return "unknown"
    return name.replace(" ", "_").replace(",", "")


def add_attributes_to_metadata_class_pydantic(obj):

    if not inspect.isclass(obj):
        raise TypeError("Input must be a class")

    # Create an instance of the class
    obj = obj()
    # Create FieldInfo for mth5_type
    mth5_type_field = FieldInfo(
        annotation=str,
        default=obj._class_name.split("Group")[0],
        description="type of group",
        json_schema_extra={
            "required": True,
            "units": None,
            "examples": ["group_name"],
        },
    )

    # Use add_new_field to add mth5_type - this returns a class, not an instance
    enhanced_class = obj.add_new_field("mth5_type", mth5_type_field)()

    # Create FieldInfo for hdf5_reference
    hdf5_ref_field = FieldInfo(
        annotation=Union[h5py.Reference, None, str],
        default=None,  # Will be set later
        description="hdf5 internal reference",
        json_schema_extra={
            "required": True,
            "units": None,
            "examples": ["<HDF5 Group Reference>"],
        },
    )

    # Create an instance of the enhanced class to add the second field
    return enhanced_class.add_new_field("hdf5_reference", hdf5_ref_field)()
