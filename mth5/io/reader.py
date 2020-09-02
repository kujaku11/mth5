# -*- coding: utf-8 -*-
"""
This is a utility function to get the appropriate reader for a given file type and 
return the appropriate object of :class:`mth5.timeseries`

This setup to be like plugins but a hack cause I did not find the time to set
this up properly as a true plugin.  

If you are writing your own reader you need the following structure:
    
    * Class object that will read the given file
    * a reader function that is read_{file_type}, for instance read_nims
    * the return value is a :class:`mth5.timeseries.MTTS` or 
    :class:`mth5.timeseries.RunTS` object and any extra metadata in the form
    of a dictionary with keys as {level.attribute}.
    
    :rubric:
        
        class NewFile
            def __init__(self, fn):
                self.fn = fn
            
            def read_header(self):
                return header_information
            
            def read_newfile(self):
                ex, ey, hx, hy, hz = read_in_channels_as_MTTS
                extra_metadata = {'station.location.elevation': self.elevation}
                return RunTS([ex, ey, hx, hy, hz]), extra_metadata
            
        def read_newfile(fn):
            new_file_obj = NewFile(fn)
            run_obj, extra_metadata = new_file_obj.read_newfile()
            
            return run_obj, extra_metadata
        
Then add your reader to the reader dictionary so that those files can be read.
        
.. seealso:: Existing readers for some guidance.

Created on Wed Aug 26 10:32:45 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import logging

from mth5.io import zen, nims, usgs_ascii

logger = logging.getLogger(__name__)
# =============================================================================
# generic reader for any file type
# =============================================================================
readers = {
    "zen": {"file_types": ["z3d"], "reader": zen.read_z3d},
    "nims": {"file_types": ["bin", "bnn"], "reader": nims.read_nims},
    "usgs_ascii": {"file_types": [".asc", ".zip"], "reader": usgs_ascii.read_ascii},
}


def get_reader(extension):
    """
    
    get the proper reader for file extension
    
    :param extension: file extension
    :type extension: string
    :return: the correct function to read the file
    :rtype: function

    """

    for key, vdict in readers.items():
        if extension.lower() in vdict["file_types"]:
            return key, vdict["reader"]

    msg = f"Could not find a reader for file type {extension}"
    logger.error(msg)
    raise ValueError(msg)


def read_file(fn, file_type=None):
    """
    
    :param fn: full path to file
    :type fn: string or :class:`pathlib.Path`
    :param string file_type: a specific file time if the extension is ambiguous.
    :return: MT time series object
    :rtype: :class:`mth5.timeseries.MTTS`

    """

    if not isinstance(fn, Path):
        fn = Path(fn)

    if not fn.exists():
        msg = f"Could not find file {fn}. Check path."
        logger.error(msg)
        raise IOError(msg)

    if file_type is not None:
        try:
            file_reader = readers[file_type]["reader"]
        except KeyError:
            msg = (
                f"Reader for the file type {file_type} does not currently exist. "
                f"Current readers {list(readers.keys())}"
            )
            logger.error(msg)
            raise KeyError(msg)
    else:
        file_type, file_reader = get_reader(fn.suffix.replace(".", ""))

    return file_reader(fn)
