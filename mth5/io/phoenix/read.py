# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:39:34 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from .readers import (
    NativeReader,
    DecimatedContinuousReader,
    DecimatedSegmentedReader,
)

# =============================================================================
READERS = {
    "bin": NativeReader,
    "td_24k": DecimatedSegmentedReader,
    "td_150": DecimatedContinuousReader,
    "td_30": DecimatedContinuousReader,
}


def open_phoenix(file_name, **kwargs):
    """
    Will put the file into the appropriate container


    :param file_name: full path to file to open
    :type file_name: string or :class:`pathlib.Path`
    :return: The appropriate container based on file extension
    :rtype:

    """
    file_name = Path(file_name)
    extension = file_name.suffix[1:]

    # need to put the data into a TS object

    return READERS[extension](file_name, **kwargs)


def read_phoenix(file_name, **kwargs):
    """
    Read a Phoenix file into a ChannelTS object
    :param file_name: DESCRIPTION
    :type file_name: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    phnx_obj = open_phoenix(file_name, **kwargs)

    return phnx_obj.to_channel_ts()
