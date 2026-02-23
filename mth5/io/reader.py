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

.. code-block:: python

    class NewFile
        def __init__(self, fn):
            self.fn = fn

        def read_header(self):
            return header_information

        def read_newfile(self):
            ex, ey, hx, hy, hz = read_in_channels_as_MTTS
            return RunTS([ex, ey, hx, hy, hz])

    def read_newfile(fn):
        new_file_obj = NewFile(fn)
        run_obj = new_file_obj.read_newfile()

        return run_obj, extra_metadata

Then add your reader to the reader dictionary so that those files can be read.

.. seealso:: Existing readers for some guidance found in `mth5.io`

Created on Wed Aug 26 10:32:45 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from loguru import logger

from mth5.io import zen, nims, usgs_ascii, miniseed, lemi, phoenix, metronix, uoa

# =============================================================================
# generic reader for any file type
# =============================================================================
readers = {
    "zen": {"file_types": ["z3d"], "reader": zen.read_z3d},
    "nims": {"file_types": ["bin", "bnn"], "reader": nims.read_nims},
    "usgs_ascii": {
        "file_types": ["asc", "zip"],
        "reader": usgs_ascii.read_ascii,
    },
    "miniseed": {
        "file_types": ["miniseed", "ms", "mseed"],
        "reader": miniseed.read_miniseed,
    },
    "lemi424": {
        "file_types": ["txt"],
        "reader": lemi.read_lemi424,
    },
    "lemi423": {
        "file_types": ["b423"],
        "reader": lemi.read_lemi423,
    },
    "phoenix": {
        "file_types": ["bin", "td_30", "td_150", "td_24k"],
        "reader": phoenix.read_phoenix,
    },
    "metronix": {
        "file_types": ["atss"],
        "reader": metronix.read_atss,
    },
    "uoa_pr624": {
        "file_types": ["bx", "by", "bz", "ex", "ey"],
        "reader": uoa.read_uoa,
    },
    "uoa_orange": {
        "file_types": ["bin"],
        "reader": uoa.read_orange,
    },
}


def detect_orange_box(fn):
    """
    Check if a .BIN file is an Orange Box file.

    Orange Box files start with ASCII hex sample rate (e.g., " 00008CA0 \n")
    NIMS files start with ">>>>>>>>>>>>>>>..."

    :param fn: Path to file
    :return: True if Orange Box, False otherwise
    """
    try:
        with open(fn, 'rb') as f:
            first_line = f.read(20)
            # Orange Box: starts with space + hex digits
            # Example: b' 00008CA0 \n' or b' 0000000A \n'
            if first_line.startswith(b' ') and b'\n' in first_line[:15]:
                header = first_line[:first_line.find(b'\n')].strip()
                try:
                    int(header, 16)  # Try to parse as hex
                    return True
                except ValueError:
                    return False
    except Exception:
        pass
    return False


def get_reader(extension, fn=None):
    """

    get the proper reader for file extension

    :param extension: file extension
    :type extension: string
    :param fn: file path (for .bin detection)
    :type fn: Path or str
    :return: the correct function to read the file
    :rtype: function

    """
    # For .bin files, try to auto-detect Orange Box vs NIMS/Phoenix
    if extension.lower() == "bin" and fn:
        if detect_orange_box(fn):
            return "uoa_orange", readers["uoa_orange"]["reader"]
        logger.warning("Suggest inputting file_type, bin could be nims or phoenix")

    for key, vdict in readers.items():
        if extension.lower() in vdict["file_types"]:
            return key, vdict["reader"]
    msg = f"Could not find a reader for file type {extension}"
    logger.error(msg)
    raise ValueError(msg)


def read_file(fn, file_type=None, **kwargs):
    """
    This is the universal reader for MT time series.  This will pick out the
    proper reader given the file type or extension.  Keyworkd arguments will
    depend on the reader and file type.

    :param fn: full path to file
    :type fn: string or :class:`pathlib.Path`
    :param string file_type: a specific file time if the extension is ambiguous.
    :return: channel or run time series object
    :rtype: :class:`mth5.timeseries.MTTS` or :class:`mth5.timeseries.RunTS`

    """

    if isinstance(fn, (list, tuple)):
        fn_list = [Path(ff) for ff in fn]
        if not fn_list[0].exists():
            msg = f"Could not find file {fn}. Check path."
            logger.error(msg)
            raise IOError(msg)
        file_ext = fn_list[0].suffix[1:]
        fn_for_detection = fn_list[0]
    else:
        fn = Path(fn)
        if not fn.exists():
            msg = f"Could not find file {fn}. Check path."
            logger.error(msg)
            raise IOError(msg)
        file_ext = fn.suffix[1:]
        fn_for_detection = fn

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
        file_type, file_reader = get_reader(file_ext, fn_for_detection)
    return file_reader(fn, **kwargs)
