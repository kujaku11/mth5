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


def open_file(file_name):
    """
    Will put the file into the appropriate container
    
    
    :param file_name: full path to file to open
    :type file_name: string or :class:`pathlib.Path`
    :return: The appropriate container based on file extension
    :rtype: 

    """
    file_name = Path(file_name)
    extension = file_name.suffix[1:]

    return READERS[extension](file_name)
