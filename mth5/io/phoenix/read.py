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

from .readers import DecimatedContinuousReader, DecimatedSegmentedReader, NativeReader


if TYPE_CHECKING:
    from mth5.timeseries import ChannelTS


# =============================================================================
# Dictionary for reader types
# =============================================================================
READERS: dict[str, type] = {
    "bin": NativeReader,
    "td_24k": DecimatedSegmentedReader,
    "td_150": DecimatedContinuousReader,
    "td_30": DecimatedContinuousReader,
}


def open_phoenix(
    file_name: str | Path, **kwargs: Any
) -> DecimatedContinuousReader | DecimatedSegmentedReader | NativeReader:
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
    file_name = Path(file_name)
    extension = file_name.suffix[1:]

    # need to put the data into a TS object

    return READERS[extension](file_name, **kwargs)


def read_phoenix(file_name: str | Path, **kwargs: Any) -> ChannelTS:
    """
    Read a Phoenix Geophysics data file into a ChannelTS object.

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
        - Other arguments passed to the Phoenix reader constructor.

    Returns
    -------
    channel_ts : ChannelTS
        Time series data object containing the Phoenix file data
        with calibration applied if calibration files were provided.

    Raises
    ------
    KeyError
        If the file extension is not supported by any Phoenix reader.
    ValueError
        If the file cannot be read or converted to ChannelTS format.
    """
    phnx_obj = open_phoenix(file_name, **kwargs)
    rxcal_fn = kwargs.pop("rxcal_fn", None)
    scal_fn = kwargs.pop("scal_fn", None)

    return phnx_obj.to_channel_ts(rxcal_fn=rxcal_fn, scal_fn=scal_fn)
