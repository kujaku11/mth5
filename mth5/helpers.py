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
import h5py
import gc
import logging

logger = logging.getLogger(__name__)

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
    if not isinstance(compression, (str, type(None))):
        msg = "compression type must be a string, not {0}".format(type(compression))
        logger.error(msg)
        raise TypeError(msg)

    if not compression in COMPRESSION:
        msg = (
            f"Compression type {compression} not supported. "
            + f"Supported options are {COMPRESSION}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if compression == "lzf":
        level = COMPRESSION_LEVELS["lzf"][0]
    elif compression == " gzip":
        if not isinstance(level, (int)):
            msg = "Level type for gzip must be an int, not {0}.".format(
                type(level) + f" Options are {0}".format(COMPRESSION_LEVELS["gzip"])
            )
            logger.error(msg)
            raise TypeError(msg)
    elif compression == " szip":
        if not isinstance(level, (str)):
            msg = "Level type for szip must be an str, not {0}.".format(
                type(level)
            ) + " Options are {0}".format(COMPRESSION_LEVELS["szip"])
            logger.error(msg)
            raise TypeError(msg)

    if not level in COMPRESSION_LEVELS[compression]:
        msg = (
            f"compression level {level} not supported for {compression}."
            + " Options are {0}".format(COMPRESSION_LEVELS[compression])
        )
        logger.error(msg)
        raise ValueError(msg)

    return compression, level


def recursive_hdf5_tree(group, lines=[]):
    if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
        for key, value in group.items():
            lines.append("-{0}: {1}".format(key, value))
            recursive_hdf5_tree(value, lines)
    elif isinstance(group, h5py._hl.dataset.Dataset):
        for key, value in group.attrs.items():
            lines.append("\t-{0}: {1}".format(key, value))
    return "\n".join(lines)


def close_open_files():
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            msg = "Found HDF5 File object "
            try:
                msg = "{0}, ".format(obj.filename)
                obj.flush()
                obj.close()
                msg += "Closed File"
                logger.info(msg)
            except:
                msg += "File already closed."
                logger.info(msg)


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
            lines.append("{0}|- Group: {1}".format(spacing, group_name))
            lines.append("{0}{1}".format(spacing, (len(group_name) + 10) * "-"))
        elif isinstance(obj, h5py.Dataset):
            lines.append("{0}--> Dataset: {1}".format(spacing, group_name))
            lines.append("{0}{1}".format(spacing, (len(group_name) + 15) * "."))

    # lines.append(parent.name)
    parent.visititems(fancy_print)
    return "\n".join(lines)
