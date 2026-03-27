# -*- coding: utf-8 -*-
"""
Universal reader for magnetotelluric time series data files.

This module provides a plugin-like system for reading various MT data formats
and returning appropriate :class:`mth5.timeseries` objects. The reader
automatically detects file types and dispatches to the correct parser.

Plugin Structure
----------------
If you are writing your own reader, implement the following structure:

    * Class object that reads the given file format
    * A reader function named read_{file_type} (e.g., read_nims)
    * Return value should be a :class:`mth5.timeseries.MTTS` or
      :class:`mth5.timeseries.RunTS` object plus extra metadata as a
      dictionary with keys formatted as {level.attribute}

Example Implementation
----------------------
.. code-block:: python

    class NewFile:
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

Then add your reader to the readers dictionary for automatic detection.

See Also
--------
Existing readers in `mth5.io` for implementation guidance.

Created on Wed Aug 26 10:32:45 2020

:author: Jared Peacock
:license: MIT
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from loguru import logger

from mth5.io import zen, nims, usgs_ascii, miniseed, lemi, phoenix, metronix, uoa

# =============================================================================
# Reader registry for MT data formats
# =============================================================================
readers: dict[str, dict[str, Any]] = {
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


def read_file(
    fn: str | Path | list[str | Path], file_type: str | None = None, **kwargs: Any
) -> Any:
    """
    Universal reader for magnetotelluric time series data files.

    Automatically detects the file type based on extension and dispatches
    to the appropriate reader function. Supports both single files and
    lists of files for multi-file formats.

    Parameters
    ----------
    fn : str, Path, or list of str/Path
        Full path(s) to data file(s) to be read. For multi-file formats,
        pass a list of file paths.
    file_type : str, optional
        Specific reader type to use if file extension is ambiguous.
        Must be one of the keys in the readers registry, by default None
    **kwargs : dict
        Additional keyword arguments passed to the specific reader function.
        Supported arguments depend on the file format and reader.

    Returns
    -------
    MTTS or RunTS
        Time series object containing the data:
        - :class:`mth5.timeseries.MTTS` for single channel data
        - :class:`mth5.timeseries.RunTS` for multi-channel run data

    Raises
    ------
    IOError
        If any specified file does not exist
    KeyError
        If the specified file_type is not supported
    ValueError
        If no reader can be found for the file extension

    Examples
    --------
    Read a single Z3D file (auto-detected)

    >>> data = read_file("/path/to/station_001.z3d")
    >>> print(type(data))  # <class 'mth5.timeseries.ChannelTS'>

    Read with explicit file type for ambiguous extensions

    >>> data = read_file("/path/to/data.bin", file_type="nims")
    >>> print(data.n_channels)

    Read multiple files for a multi-file format

    >>> files = ["/path/to/file1.asc", "/path/to/file2.asc"]
    >>> run_data = read_file(files, sample_rate=1.0)

    Notes
    -----
    Supported file types and extensions:
    - zen: .z3d (Zonge Z3D files)
    - nims: .bin, .bnn (USGS NIMS files)
    - usgs_ascii: .asc, .zip (USGS ASCII format)
    - miniseed: .miniseed, .ms, .mseed (miniSEED format)
    - lemi424: .txt (LEMI-424 format)
    - phoenix: .bin, .td_30, .td_150, .td_24k (Phoenix formats)
    - metronix: .atss (Metronix ADU format)

    For ambiguous extensions like .bin, specify file_type explicitly.
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
