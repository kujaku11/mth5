# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:39:34 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .readers import (
    DecimatedContinuousReader,
    DecimatedSegmentedReader,
    MTUTable,
    MTUTSN,
    NativeReader,
)


if TYPE_CHECKING:
    from mth5.timeseries import ChannelTS, RunTS


# =============================================================================
# Dictionary for reader types
# =============================================================================
READERS: dict[str, type] = {
    "bin": NativeReader,
    "td_24k": DecimatedSegmentedReader,
    "td_150": DecimatedContinuousReader,
    "td_30": DecimatedContinuousReader,
    "TS3": MTUTSN,
    "TS4": MTUTSN,
    "TS5": MTUTSN,
    "TBL": MTUTable,
    "TSL": MTUTSN,
    "TSH": MTUTSN,
}


def get_file_extenstion(file_name: str | Path) -> str:
    """
    Get the file extension from a file name.

    Parameters
    ----------
    file_name : str or pathlib.Path
        The file name to extract the extension from.

    Returns
    -------
    str
        The file extension without the leading dot.
    """
    file_name = Path(file_name)
    return file_name.suffix[1:]


def open_phoenix(
    file_name: str | Path, **kwargs: Any
) -> (
    DecimatedContinuousReader
    | DecimatedSegmentedReader
    | NativeReader
    | MTUTSN
    | MTUTable
):
    """
    Open a Phoenix Geophysics data file in the appropriate container.

    Parameters
    ----------
    file_name : str or pathlib.Path
        Full path to the Phoenix data file to open.
    **kwargs : Any
        Additional keyword arguments to pass to the reader constructor.

    Returns
    -------
    reader : DecimatedContinuousReader | DecimatedSegmentedReader | NativeReader
        The appropriate Phoenix reader container based on file extension:
        - .bin files: NativeReader
        - .td_24k files: DecimatedSegmentedReader
        - .td_150/.td_30 files: DecimatedContinuousReader

    Raises
    ------
    KeyError
        If the file extension is not supported by any Phoenix reader.
    """
    extension = get_file_extenstion(file_name)

    # need to put the data into a TS object

    return READERS[extension](file_name, **kwargs)


def read_phoenix(file_name: str | Path, **kwargs: Any) -> ChannelTS | RunTS | MTUTable:
    """
    Read a Phoenix Geophysics data file into a ChannelTS or RunTS object
    depending on the file type.  Newer files that end in .td_XX or .bin will be
    read into ChannelTS objects.  Older MTU files that end in .TS3, .TS4, .TS5,
    .TSL, or .TSH will be read into RunTS objects.

    Parameters
    ----------
    file_name : str or pathlib.Path
        Path to the Phoenix data file to read.
    **kwargs : Any
        Additional keyword arguments. May include:

        - rxcal_fn : str or pathlib.Path, optional
            Path to receiver calibration file.
        - scal_fn : str or pathlib.Path, optional
            Path to sensor calibration file.
        - table_filepath : str or pathlib.Path, optional
            Path to the MTU TBL file for use with MTUTSN files.
        - Other arguments passed to the Phoenix reader constructor.

    Returns
    -------
    channel_ts : ChannelTS
        Time series data object containing the Phoenix file data
        with calibration applied if calibration files were provided.

    run_ts : RunTS
        Time series data object containing the MTU data from the Phoenix MTU
        files with calibration applied if specified.

    mtu_table : MTUTable
        Metadata table object containing the MTU table data.

    Raises
    ------
    KeyError
        If the file extension is not supported by any Phoenix reader.
    ValueError
        If the file cannot be read or converted to ChannelTS or RunTS format.
    """
    extension = get_file_extenstion(file_name)
    if extension.startswith("td_") or extension == "bin":
        phnx_obj = open_phoenix(file_name, **kwargs)
        rxcal_fn = kwargs.pop("rxcal_fn", None)
        scal_fn = kwargs.pop("scal_fn", None)

        return phnx_obj.to_channel_ts(rxcal_fn=rxcal_fn, scal_fn=scal_fn)
    elif extension in ["TS3", "TS4", "TS5", "TSL", "TSH"]:
        tbl_file = kwargs.pop("table_filepath", None)
        mtu_obj = open_phoenix(file_name, **kwargs)
        run_ts = mtu_obj.to_runts(table_filepath=tbl_file, calibrate=True)
        return run_ts
    elif extension == "TBL":
        mtu_table = open_phoenix(file_name, **kwargs)
        return mtu_table
    else:
        raise KeyError(
            f"File extension '{extension}' is not supported by any Phoenix reader."
        )
